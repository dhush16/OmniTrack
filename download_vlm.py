import os
import torch

# Force Hugging Face cache path to Drive D:
os.environ["HF_HOME"] = "D:/huggingface_cache"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
cache_path = "D:/huggingface_cache"

print(f"Downloading {model_id} directly to {cache_path}...")

processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    load_in_4bit=True,
    cache_dir=cache_path
)

print("\nModel successfully downloaded to Drive D: and loaded onto RTX 4050!")