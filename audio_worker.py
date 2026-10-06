import numpy as np
import sounddevice as sd
import requests
import queue
import threading
import time
from faster_whisper import WhisperModel

# 1. Initialize Faster-Whisper on RTX 4050 (INT8 compute)
print("Loading Faster-Whisper base model on CUDA...")
model = WhisperModel("base.en", device="cuda", compute_type="int8")
print("Faster-Whisper loaded successfully.")

# 2. Audio Ring Buffer Parameters (Chapter 6 formulation)
SAMPLE_RATE = 16000
BUFFER_DURATION = 2.5  # seconds
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION) # 40,000 samples

audio_queue = queue.Queue()

# Controlled assembly vocabulary mapping to SOP nodes
KEYWORD_MAPPING = {
    "ground": "v1_esd_grounding",
    "strap": "v1_esd_grounding",
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

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status]: {status}")
    audio_queue.put(indata.copy())

def trigger_server(node_id):
    try:
        res = requests.post(f"http://127.0.0.1:8080/trigger_step/{node_id}")
        data = res.json()
        print(f"\n[ACTION TRIGGERED]: {node_id} | Status: {data.get('status')}")
    except Exception as e:
        print(f"Error communicating with OmniTrack server: {e}")

def transcription_worker():
    audio_buffer = np.zeros((BUFFER_SIZE, 1), dtype=np.float32)
    last_spoken_time = 0
    
    print("\n>>> Acoustic Intent Grounding Active. Speak assembly actions into your microphone:")
    print("    Examples: 'Strap grounded', 'Board seated', 'Torque complete', 'Applying paste', 'Mounting cooler'\n")

    while True:
        # Pull newly arrived PCM samples
        while not audio_queue.empty():
            chunk = audio_queue.get()
            audio_buffer = np.roll(audio_buffer, -len(chunk), axis=0)
            audio_buffer[-len(chunk):] = chunk

        # Transcribe every 1.5 seconds if sufficient energy exists
        if time.time() - last_spoken_time > 1.5:
            # Flatten audio for Whisper
            flat_audio = audio_buffer.flatten()
            
            # Simple energy threshold gating (VAD proxy)
            if np.max(np.abs(flat_audio)) > 0.03:
                segments, _ = model.transcribe(flat_audio, beam_size=1, language="en")
                transcript = " ".join([s.text for s in segments]).strip().lower()
                
                if transcript:
                    print(f"[Heard]: \"{transcript}\"")
                    last_spoken_time = time.time()
                    
                    # Match transcript against controlled SOP keywords
                    matched_node = None
                    for word, node in KEYWORD_MAPPING.items():
                        if word in transcript:
                            matched_node = node
                            break
                    
                    if matched_node:
                        trigger_server(matched_node)

        time.sleep(0.1)

# Start background audio capture thread
threading.Thread(target=transcription_worker, daemon=True).start()

# Start live microphone stream
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    while True:
        time.sleep(0.5)