import os
import torch
import json
import re
import io
import time
import requests
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Set Hugging Face cache path to D: drive
os.environ["HF_HOME"] = "D:/huggingface_cache"
CACHE_PATH = "D:/huggingface_cache"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("==================================================")
print("Initializing Qwen2.5-VL-3B-Instruct (4-bit on CUDA)...")
print(f"Cache Location: {CACHE_PATH}")
print("==================================================")

# 1. Initialize Processor
processor = None
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
except Exception as e:
    print(f"[Notice] AutoProcessor initialization error: {e}")

# 2. Initialize Model in 4-bit precision on CUDA
model = None
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        load_in_4bit=True,
        cache_dir=CACHE_PATH,
    )
    print("Qwen2.5-VL-3B-Instruct successfully loaded in 4-bit on CUDA.")
except Exception as e:
    print(f"[Notice] CUDA 4-bit load error: {e}")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
            cache_dir=CACHE_PATH,
        )
        print("Model loaded in fallback mode.")
    except Exception as ex2:
        print(f"[Warning] Model could not be loaded: {ex2}")

VALID_TOOLS = {"screwdriver", "thermal paste syringe", "heatsink", "none"}

SYSTEM_PROMPT = """You are OmniTrack's workstation vision inspector.
Inspect the workstation camera image and identify which assembly tool or hardware component is present or being used by the operator.
Classify the object into EXACTLY ONE of the following valid categories:
- screwdriver
- thermal paste syringe
- heatsink
- none

Respond ONLY with a JSON object in this exact format:
{"detected_tool": "screwdriver" | "thermal paste syringe" | "heatsink" | "none"}"""

def normalize_tool(raw_text: str) -> str:
    """Extract and normalize tool name to one of the 4 target categories."""
    try:
        match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            candidate = str(parsed.get("detected_tool", "")).lower().strip()
            if candidate in VALID_TOOLS:
                return candidate
            if "screwdriver" in candidate or "screw" in candidate or "torque" in candidate:
                return "screwdriver"
            if "syringe" in candidate or "paste" in candidate or "compound" in candidate:
                return "thermal paste syringe"
            if "heatsink" in candidate or "cooler" in candidate:
                return "heatsink"
            if "none" in candidate:
                return "none"
    except Exception:
        pass

    lower_resp = raw_text.lower()
    if "thermal paste syringe" in lower_resp or "thermal paste" in lower_resp or "syringe" in lower_resp or "compound" in lower_resp:
        return "thermal paste syringe"
    if "screwdriver" in lower_resp or "screw" in lower_resp:
        return "screwdriver"
    if "heatsink" in lower_resp or "cooler" in lower_resp:
        return "heatsink"
    return "none"

def classify_tool(image_input):
    """Run Qwen2.5-VL inference on an image to classify the workstation tool."""
    if model is None or processor is None:
        return "none", "Model or processor uninitialized"

    if not isinstance(image_input, Image.Image):
        # Convert numpy array (OpenCV BGR) to PIL Image (RGB)
        image_input = Image.fromarray(image_input[:, :, ::-1])

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image_input], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=48)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    detected_tool = normalize_tool(response_text)
    return detected_tool, response_text

TOOL_TO_STEP = {
    "screwdriver": "v3_cross_torque",
    "thermal paste syringe": "v4_thermal_paste",
    "heatsink": "v5_heatsink_latch",
    "none": "unknown"
}

def analyze_frame(image_input):
    """Backwards-compatible interface for pipeline integration."""
    tool, raw = classify_tool(image_input)
    target_step = TOOL_TO_STEP.get(tool, "unknown")
    return {"detected_tool": tool, "target_step": target_step, "raw_response": raw}

def grab_frame(stream_url="http://127.0.0.1:8080/video_feed", timeout=3.0):
    """Grab a single JPEG frame from the MJPEG streaming endpoint."""
    try:
        with requests.get(stream_url, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None
            buffer = b""
            for chunk in resp.iter_content(chunk_size=4096):
                buffer += chunk
                start = buffer.find(b"\xff\xd8")  # JPEG start marker
                end = buffer.find(b"\xff\xd9")    # JPEG end marker
                if start != -1 and end != -1 and end > start:
                    jpg_bytes = buffer[start : end + 2]
                    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    except Exception as e:
        print(f"[Stream Warning] Unable to grab frame from {stream_url}: {e}")
        return None

def update_telemetry(tool_name: str, telemetry_url="http://127.0.0.1:8080/telemetry") -> bool:
    """Send detected tool classification to backend telemetry endpoint."""
    try:
        res = requests.post(telemetry_url, json={"vlm_tool": tool_name}, timeout=2.0)
        return res.status_code == 200
    except Exception as e:
        print(f"[Telemetry Warning] Failed to update telemetry at {telemetry_url}: {e}")
        return False

def run_worker(
    video_url="http://127.0.0.1:8080/video_feed",
    telemetry_url="http://127.0.0.1:8080/telemetry",
    interval=3.0,
):
    """Main worker loop: periodic frame grab -> classify -> POST telemetry."""
    print("==================================================")
    print("OmniTrack VLM Worker Active")
    print(f"Video Feed Source: {video_url}")
    print(f"Telemetry Target:  {telemetry_url}")
    print(f"Sampling Period:   Every {interval} seconds")
    print("Classifications:   screwdriver, thermal paste syringe, heatsink, none")
    print("==================================================\n")

    while True:
        try:
            frame = grab_frame(video_url)
            if frame is not None:
                tool, raw_resp = classify_tool(frame)
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] VLM Tool Classification: '{tool}'")
                success = update_telemetry(tool, telemetry_url)
                if success:
                    print(f"[{timestamp}] -> Successfully updated telemetry: vlm_tool='{tool}'")
            else:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Waiting for video stream at {video_url}...")
        except KeyboardInterrupt:
            print("\nVLM Worker terminated by user.")
            break
        except Exception as e:
            print(f"[Worker Error]: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    run_worker()