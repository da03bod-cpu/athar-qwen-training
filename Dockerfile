FROM axolotlai/axolotl-cloud-uv:main-latest

ENV JUPYTER_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace/data/athar

# Keep the existing Axolotl environment and only add the RunPod SDK.
RUN uv pip install --python /workspace/axolotl-venv/bin/python runpod

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Fail during build if the runtime needed for both routes is not present.
RUN test -x /workspace/axolotl-venv/bin/python \
    && /workspace/axolotl-venv/bin/python -c "import runpod, torch, transformers, peft, bitsandbytes; print('ATHAR RUNTIME OK', torch.__version__, transformers.__version__, peft.__version__)"

# Existing training + advisor-match assets.
COPY configs/ ./configs/
COPY data/ ./data/

# New council assets. All prompts are normal UTF-8 Markdown, so they are baked
# into the image. Heavy LoRA weights remain in Git LFS and are fetched lazily
# on the first advisory_consultation request using GITHUB_TOKEN.
COPY advisors/ ./advisors/
COPY prompts/ ./prompts/
COPY handler_advisory_council.py ./handler_advisory_council.py

# Unified entrypoint: advisory_match + advisory_consultation + existing training.
COPY handler.py ./handler.py

LABEL athar.redeploy="2026-09-21-unified-council-v1"

ENTRYPOINT ["/workspace/axolotl-venv/bin/python", "-u", "/workspace/data/athar/handler.py"]
