# Download model without requiring updated Transformers
from huggingface_hub import snapshot_download
import os

cache_dir = "/data/models"
os.makedirs(cache_dir, exist_ok=True)

# Download the model repository
print("Downloading model...")
model_path = snapshot_download(
    repo_id="Qwen/Qwen-VL",
    cache_dir=cache_dir,
    local_dir_use_symlinks=False  # 不使用软链接，直接复制
)

print(f"✓ Model downloaded to: {model_path}")