FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Basic system tools only.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-serverless.txt /app/requirements-serverless.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements-serverless.txt

# Application + authoritative prompts + advisor registry + LoRA checkpoints.
COPY handler_advisory_council.py /app/handler_advisory_council.py
COPY advisors/ /app/advisors/
COPY prompts/ /app/prompts/
COPY checkpoints/ /app/checkpoints/

# Fail during BUILD if Git LFS pointers were copied instead of the real adapters.
RUN python - <<'PY'
from pathlib import Path

required = [
    Path("/app/checkpoints/specialist/checkpoint-44/adapter_model.safetensors"),
    Path("/app/checkpoints/meta/checkpoint-18/adapter_model.safetensors"),
    Path("/app/checkpoints/specialist/checkpoint-44/adapter_config.json"),
    Path("/app/checkpoints/meta/checkpoint-18/adapter_config.json"),
]

for path in required:
    if not path.exists():
        raise SystemExit(f"MISSING REQUIRED FILE: {path}")

for path in required[:2]:
    size = path.stat().st_size
    if size < 1_000_000:
        head = path.read_bytes()[:200].decode("utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in head or size < 100_000:
            raise SystemExit(
                f"GIT LFS POINTER DETECTED instead of real checkpoint: {path} "
                f"(size={size} bytes)."
            )

print("LoRA checkpoint validation: OK")
PY

# Validate all 35 prompts and Meta prompt exist before building the expensive model layer.
RUN python - <<'PY'
from pathlib import Path
import json

root = Path("/app")

registry_path = root / "advisors" / "advisors_registry_35.json"
meta_path = root / "prompts" / "meta" / "AOS-META-00.md"

if not registry_path.exists():
    raise SystemExit(f"Missing registry: {registry_path}")
if not meta_path.exists():
    raise SystemExit(f"Missing Meta prompt: {meta_path}")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
advisors = registry.get("advisors", registry)

if len(advisors) != 35:
    raise SystemExit(f"Expected 35 advisors, found {len(advisors)}")

for advisor in advisors:
    p = root / "prompts" / "advisors" / advisor["prompt_file"]
    if not p.exists():
        raise SystemExit(f"Missing advisor prompt: {p}")
    if p.stat().st_size < 1000:
        raise SystemExit(f"Advisor prompt looks too small: {p}")

print("35 advisor prompts + Meta prompt validation: OK")
PY

# Bake Qwen3-14B into the image to avoid downloading it on every cold worker.
RUN mkdir -p /models/Qwen3-14B && \
    python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-14B",
    local_dir="/models/Qwen3-14B",
    local_dir_use_symlinks=False,
    ignore_patterns=[
        "*.msgpack",
        "*.h5",
        "*.ot",
        "*.onnx",
        "*.tflite",
    ],
)
print("Qwen3-14B baked into image: OK")
PY

ENV MODEL_ID=/models/Qwen3-14B \
    SPECIALIST_ADAPTER_PATH=/app/checkpoints/specialist/checkpoint-44 \
    META_ADAPTER_PATH=/app/checkpoints/meta/checkpoint-18 \
    ADVISOR_PROMPTS_DIR=/app/prompts/advisors \
    META_PROMPT_PATH=/app/prompts/meta/AOS-META-00.md \
    ADVISOR_REGISTRY_PATH=/app/advisors/advisors_registry_35.json \
    MAX_MODEL_INPUT_TOKENS=30000 \
    ADVISOR_MAX_NEW_TOKENS=1800 \
    META_MAX_NEW_TOKENS=2600 \
    GEN_TEMPERATURE=0.25 \
    GEN_TOP_P=0.90

# Import-only smoke test. Do NOT load the 14B model during Docker build.
RUN python - <<'PY'
import runpod
import torch
import transformers
import peft
import accelerate
import bitsandbytes
import handler_advisory_council

print("Python stack: OK")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("CUDA build:", torch.version.cuda)
PY

CMD ["python", "-u", "/app/handler_advisory_council.py"]
