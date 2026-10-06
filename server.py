import os
import json
import time
import threading
from typing import Optional
import pyttsx3
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# 1. FastAPI App Initialization & CORS
app = FastAPI(title="OmniTrack Unified Telemetry Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Protocol DAG State Machine
class ProtocolDAG:
    def __init__(self, config_path="sop_assembly.json"):
        if not os.path.isabs(config_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, config_path)
            if os.path.exists(candidate):
                config_path = candidate
        with open(config_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.nodes = self.schema["nodes"]
        self.edges = set(self.schema["edges"])
        self.current_state = self.schema["initial_node"]
        self.visited = {self.current_state}
        self.ancestor_cache = self._precompute_ancestors()

    def _precompute_ancestors(self):
        ancestors = {n: set() for n in self.nodes}
        for node in self.nodes:
            q = list(self.nodes[node].get("prerequisites", []))
            while q:
                p = q.pop(0)
                if p not in ancestors[node]:
                    ancestors[node].add(p)
                    q.extend(self.nodes[p].get("prerequisites", []))
        return ancestors

    def evaluate(self, candidate_step: str):
        if candidate_step == self.current_state:
            return True, self.current_state, "VALID_TRANSITION", []
        edge = f"{self.current_state}->{candidate_step}"
        if edge in self.edges:
            self.current_state = candidate_step
            self.visited.add(candidate_step)
            return True, self.current_state, "VALID_TRANSITION", []
        elif candidate_step == self.schema.get("initial_node"):
            self.reset()
            return True, self.current_state, "VALID_TRANSITION", []
        else:
            missing = [
                m for m in self.nodes
                if m in self.ancestor_cache.get(candidate_step, set()) and m not in self.visited
            ]
            missing_names = [self.nodes[m]["title"] for m in missing if m in self.nodes]
            return False, self.current_state, "VIOLATION", missing_names

    def reset(self):
        self.current_state = self.schema["initial_node"]
        self.visited = {self.current_state}
        return self.current_state

dag = ProtocolDAG()

# 3. Global Telemetry State Dictionary & Video Source Tracking
telemetry = {
    "current_state": dag.current_state,
    "status": "INITIALIZED",
    "last_speech": "None detected yet",
    "vlm_tool": "None",
    "missing": []
}

CURRENT_SOURCE = 0

SOURCE_MAP = {
    "webcam": 0,
    "0": 0,
    "trial_1": "data/trial_1_normal.mp4",
    "data/trial_1_normal.mp4": "data/trial_1_normal.mp4",
    "trial_2": "data/trial_2_omission.mp4",
    "data/trial_2_omission.mp4": "data/trial_2_omission.mp4",
}

def resolve_source_path(src):
    if isinstance(src, str) and not os.path.isabs(src):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, src)
        if os.path.exists(candidate):
            return candidate
    return src

def open_capture_source(src):
    if src == 0 or src == "0" or src == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        return cap
    else:
        resolved = resolve_source_path(src)
        return cv2.VideoCapture(resolved)

# 4. Offline pyttsx3 Text-to-Speech Preemption Alert (Daemon Thread)
def speak_alert(msg: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.say(msg)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS Error]: {e}")

def trigger_transition(node_id: str):
    if node_id == "reset":
        dag.reset()
        telemetry["current_state"] = dag.current_state
        telemetry["status"] = "INITIALIZED"
        telemetry["missing"] = []
        return True
    valid, state, status, missing = dag.evaluate(node_id)
    telemetry["current_state"] = state
    telemetry["status"] = status
    telemetry["missing"] = missing
    if not valid:
        alert_msg = f"Warning: {missing[0]} was skipped!" if missing else "Invalid step sequence!"
        threading.Thread(target=speak_alert, args=(alert_msg,), daemon=True).start()
    return valid

# 5. Acoustic Vocabulary Intent Mapping & Whisper Ingestion (Threaded)
KEYWORD_MAP = {
    "strap": "v1_esd_grounding",
    "ground": "v1_esd_grounding",
    "seat": "v2_pcb_seating",
    "board": "v2_pcb_seating",
    "torque": "v3_cross_torque",
    "screw": "v3_cross_torque",
    "paste": "v4_thermal_paste",
    "compound": "v4_thermal_paste",
    "latch": "v5_heatsink_latch",
    "cooler": "v5_heatsink_latch",
    "heatsink": "v5_heatsink_latch"
}

SAMPLE_RATE = 16000
audio_buffer = []

def audio_callback(indata, frames, time_info, status):
    audio_buffer.extend(indata.flatten().tolist())

def whisper_listener():
    if not AUDIO_AVAILABLE:
        return
    try:
        print("[INIT] Loading Faster-Whisper base.en on CPU...")
        asr_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("[READY] Whisper Acoustic Pipeline Active.")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            while True:
                if len(audio_buffer) >= SAMPLE_RATE * 2:
                    chunk = np.array(audio_buffer[:SAMPLE_RATE * 2], dtype=np.float32)
                    del audio_buffer[:SAMPLE_RATE * 2]
                    if np.max(np.abs(chunk)) > 0.03:
                        segments, _ = asr_model.transcribe(
                            chunk,
                            beam_size=3,
                            language="en",
                            initial_prompt="OmniTrack assembly: ESD strap, PCB seating, cross torque screws, thermal paste compound, heatsink cooler latch."
                        )
                        text = " ".join([s.text for s in segments]).strip().lower()
                        if text:
                            telemetry["last_speech"] = text
                            for kw, node in KEYWORD_MAP.items():
                                if kw in text:
                                    trigger_transition(node)
                                    break
                time.sleep(0.1)
    except Exception as e:
        print(f"[Whisper Acoustic Notice]: {e}")

if AUDIO_AVAILABLE:
    threading.Thread(target=whisper_listener, daemon=True).start()

# 6. MJPEG Camera Video Generator with Video File Looping
camera_lock = threading.Lock()
camera = open_capture_source(CURRENT_SOURCE)

def generate_frames():
    global camera, CURRENT_SOURCE
    while True:
        frame = None
        success = False
        with camera_lock:
            if camera is not None and camera.isOpened():
                success, frame = camera.read()
                # If reading a video file and reaching EOF, loop automatically
                if not success and CURRENT_SOURCE != 0:
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = camera.read()

            if not success:
                if camera is None or not camera.isOpened():
                    camera = open_capture_source(CURRENT_SOURCE)
                    if camera.isOpened():
                        success, frame = camera.read()

        if not success or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "OmniTrack: Video Stream Offline",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )
        else:
            frame = frame.copy()

        h, w, _ = frame.shape
        # Inspection bounding box overlay (15% margin on each side)
        cv2.rectangle(
            frame,
            (int(w * 0.15), int(h * 0.15)),
            (int(w * 0.85), int(h * 0.85)),
            (0, 255, 0),
            2,
        )

        # Active step label overlay
        current_node = telemetry.get("current_state", "")
        step_title = dag.nodes.get(current_node, {}).get("title", current_node)
        cv2.putText(
            frame,
            f"Active Step: {step_title}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"State: {current_node} | Status: {telemetry.get('status', 'N/A')}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS

# 7. Endpoints
@app.get("/")
def get_root():
    return {
        "status": "online",
        "system": "OmniTrack Unified Multimodal Engine",
        "current_source": CURRENT_SOURCE,
        "telemetry": telemetry,
    }

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/telemetry")
def get_telemetry():
    return telemetry

@app.post("/trigger_step/{step_id}")
def trigger_step(step_id: str, tool: Optional[str] = None, speech: Optional[str] = None):
    if tool is not None:
        telemetry["vlm_tool"] = tool
    if speech is not None:
        telemetry["last_speech"] = speech
    trigger_transition(step_id)
    return telemetry

@app.post("/telemetry")
def update_telemetry(payload: dict):
    for key in ("last_speech", "vlm_tool", "status", "current_state", "missing"):
        if key in payload:
            telemetry[key] = payload[key]
    return telemetry

@app.post("/set_source")
def set_source(payload: dict):
    global camera, CURRENT_SOURCE
    src_key = payload.get("source", "webcam")
    if src_key not in SOURCE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source: '{src_key}'. Must be one of {list(SOURCE_MAP.keys())}"
        )

    target_source = SOURCE_MAP[src_key]

    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
        CURRENT_SOURCE = target_source
        camera = open_capture_source(CURRENT_SOURCE)

    # Reset the DAG state to initial when switching sources
    dag.reset()
    telemetry["current_state"] = dag.current_state
    telemetry["status"] = "INITIALIZED"
    telemetry["missing"] = []

    return {
        "status": "success",
        "source": src_key,
        "current_source": CURRENT_SOURCE,
        "telemetry": telemetry
    }

@app.get("/get_source")
def get_source():
    return {"current_source": CURRENT_SOURCE}

@app.post("/reset")
def reset_pipeline():
    trigger_transition("reset")
    return telemetry

@app.on_event("shutdown")
def shutdown_event():
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)