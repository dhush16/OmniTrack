import cv2
import time
import requests
import numpy as np
import sounddevice as sd
import queue
import threading
from faster_whisper import WhisperModel

# 1. Setup Audio & Faster-Whisper on CPU (bypasses CUDA DLL conflicts)
SAMPLE_RATE = 16000
audio_queue = queue.Queue()
print("\n[INIT] Loading Faster-Whisper base.en on CPU...")
asr_model = WhisperModel("base.en", device="cpu", compute_type="int8")
print("[READY] Faster-Whisper ASR initialized successfully.")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Ingest Status]: {status}")
    audio_queue.put(indata.copy())

# 2. Vocabulary & Protocol DAG Mapping
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

def trigger_server(node_id):
    try:
        res = requests.post(f"http://127.0.0.1:8080/trigger_step/{node_id}", timeout=2.0)
        data = res.json()
        print(f"\n>>> [DAG RESPONSE] Step: {node_id} | Status: {data.get('status')} | Current: {data.get('title')}")
    except Exception as e:
        print(f"[Network Notice] Could not reach backend server at 8080: {e}")

# 3. Audio Listener Thread (2.5s Rolling Buffer)
def audio_listener():
    buffer = np.zeros((int(SAMPLE_RATE * 2.5), 1), dtype=np.float32)
    last_heard = 0
    while True:
        while not audio_queue.empty():
            chunk = audio_queue.get()
            buffer = np.roll(buffer, -len(chunk), axis=0)
            buffer[-len(chunk):] = chunk

        # Transcribe every 1.2s when energy threshold is met
        if time.time() - last_heard > 1.2:
            flat = buffer.flatten()
            if np.max(np.abs(flat)) > 0.03:
                try:
                    segments, _ = asr_model.transcribe(flat, beam_size=1, language="en")
                    text = " ".join([s.text for s in segments]).strip().lower()
                    if text:
                        print(f"[Acoustic Input]: '{text}'")
                        last_heard = time.time()
                        for word, node in KEYWORD_MAP.items():
                            if word in text:
                                trigger_server(node)
                                break
                except Exception as ex:
                    print(f"[ASR Worker Notice]: {ex}")
        time.sleep(0.1)

threading.Thread(target=audio_listener, daemon=True).start()

# 4. Optical Camera Feed (OpenCV)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("\n=======================================================")
print("OmniTrack Multimodal Ingestion Pipeline Active")
print("Camera Feed: Active | Whisper Acoustic ASR: Active")
print("Target Backend: http://127.0.0.1:8080")
print("Press 'q' inside the OpenCV window to exit.")
print("=======================================================\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        # Visual Workstation Overlay
        cv2.rectangle(frame, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
        cv2.putText(frame, "OmniTrack: Workstation Inspection Zone", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Speak: 'strap', 'seat', 'torque', 'paste', 'latch'", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Workstation Primary Vision (B_v)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()