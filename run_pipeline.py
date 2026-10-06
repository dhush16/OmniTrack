import cv2
import time
import requests
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from vlm_worker import analyze_frame

SERVER_URL = "http://127.0.0.1:8080/trigger_step"

# 1. Faster-Whisper Setup
SAMPLE_RATE = 16000
asr_model = WhisperModel("base.en", device="cuda", compute_type="int8")
audio_buffer = []

def trigger_sop(node_id):
    if node_id and node_id != "unknown":
        try:
            res = requests.post(f"{SERVER_URL}/{node_id}")
            print(f"[DAG TRIGGER] Step: {node_id} | Status: {res.json().get('status')}")
        except Exception as e:
            print(f"Server error: {e}")

# 2. Audio Listener Thread
def audio_callback(indata, frames, time_info, status):
    audio_buffer.extend(indata.flatten().tolist())

def whisper_worker():
    KEYWORD_MAP = {
        "strap": "v1_esd_grounding",
        "ground": "v1_esd_grounding",
        "seat": "v2_pcb_seating",
        "board": "v2_pcb_seating",
        "torque": "v3_cross_torque",
        "screw": "v3_cross_torque",
        "paste": "v4_thermal_paste",
        "latch": "v5_heatsink_latch",
        "cooler": "v5_heatsink_latch",
    }
    while True:
        if len(audio_buffer) >= SAMPLE_RATE * 2:  # 2-second audio slice
            chunk = np.array(audio_buffer[:SAMPLE_RATE * 2], dtype=np.float32)
            del audio_buffer[:SAMPLE_RATE * 2]
            
            if np.max(np.abs(chunk)) > 0.03:
                segments, _ = asr_model.transcribe(chunk, beam_size=1)
                text = " ".join([s.text for s in segments]).lower()
                for kw, node in KEYWORD_MAP.items():
                    if kw in text:
                        print(f"\n[SPEECH CONFIRMATION DETECTED]: '{text}' -> {node}")
                        trigger_sop(node)
                        break
        time.sleep(0.2)

threading.Thread(target=whisper_worker, daemon=True).start()

# 3. Main Camera & VLM Loop
cap = cv2.VideoCapture(0)
last_vlm_check = 0

print("\n=======================================================")
print("OmniTrack Live Multimodal Pipeline Active")
print("Webcam: Active | Mic: Active | DAG Engine: Connected")
print("Press 'q' in the camera window to exit.")
print("=======================================================\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        # Sample keyframe every 1.5 seconds for VLM inference to avoid GPU contention
        if now - last_vlm_check > 1.5:
            last_vlm_check = now
            # Run VLM inference in background thread or sequentially
            pred = analyze_frame(frame)
            step = pred.get("target_step")
            tool = pred.get("detected_tool")
            if step and step != "unknown":
                print(f"[VISION INFERENCE]: Tool: {tool} | Predicted Step: {step}")
                trigger_sop(step)

        cv2.putText(frame, "OmniTrack: Multimodal Bench Ingestion", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Workstation Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()