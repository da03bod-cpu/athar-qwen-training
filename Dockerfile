
FROM axolotlai/axolotl-cloud-uv:main-latest

ENV JUPYTER_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace/data/athar

# Keep the existing Axolotl environment.
RUN uv pip install --python /workspace/axolotl-venv/bin/python runpod

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Validate existing runtime.
RUN test -x /workspace/axolotl-venv/bin/python \
    && /workspace/axolotl-venv/bin/python -c "import runpod, torch, transformers, peft, bitsandbytes; print('ATHAR RUNTIME OK', torch.__version__, transformers.__version__, peft.__version__)"

# Training and matching assets.
COPY configs/ ./configs/
COPY data/ ./data/

# Advisory council.
COPY advisors/ ./advisors/
COPY prompts/ ./prompts/
COPY handler_advisory_council.py ./handler_advisory_council.py

# Screen 5 Roadmap.
COPY handler_roadmap.py ./handler_roadmap.py

# Main unified handler.
COPY handler.py ./handler.py

# Verify Python syntax during the build.
RUN /workspace/axolotl-venv/bin/python -m py_compile \
    handler.py handler_advisory_council.py handler_roadmap.py

LABEL athar.redeploy="2026-10-08-da03-screen5-roadmap"

ENTRYPOINT ["/workspace/axolotl-venv/bin/python", "-u", "/workspace/data/athar/handler.py"]
