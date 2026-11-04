FROM axolotlai/axolotl-cloud-uv:main-latest

ENV JUPYTER_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATHAR_KNOWLEDGE_EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    ATHAR_KNOWLEDGE_OFFLINE=1

WORKDIR /workspace/data/athar

# Keep the existing Axolotl environment. Add only knowledge dependencies.
RUN uv pip install --python /workspace/axolotl-venv/bin/python \
    runpod 'numpy>=1.26,<3' 'faiss-cpu>=1.9,<2' \
    'pymupdf>=1.24,<2' 'python-docx>=1.1,<2'

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install

RUN test -x /workspace/axolotl-venv/bin/python \
    && /workspace/axolotl-venv/bin/python -c "import runpod, torch, transformers, peft, bitsandbytes, faiss, fitz, docx; print('ATHAR RUNTIME OK', torch.__version__, transformers.__version__, peft.__version__)"

# Embed the small CPU retrieval model at IMAGE BUILD TIME (NOT every cold start).
# Hugging Face connectivity is needed during build only; inference stays offline.
RUN /workspace/axolotl-venv/bin/python -c "from transformers import AutoModel, AutoTokenizer; n='intfloat/multilingual-e5-small'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n); print('ATHAR E5 MODEL BAKED')"

COPY configs/ ./configs/
COPY data/ ./data/
COPY advisors/ ./advisors/
COPY prompts/ ./prompts/
COPY handler_advisory_council.py ./handler_advisory_council.py
COPY handler_roadmap.py ./handler_roadmap.py

# New additive same-endpoint RAG and Stakeholder support.
COPY knowledge/ ./knowledge/
COPY handler_knowledge.py ./handler_knowledge.py
COPY handler_stakeholder.py ./handler_stakeholder.py
COPY handler.py ./handler.py

RUN /workspace/axolotl-venv/bin/python -m py_compile \
    handler.py handler_advisory_council.py handler_roadmap.py \
    handler_knowledge.py handler_stakeholder.py knowledge/*.py

LABEL athar.redeploy="2026-11-04-knowledge-v5-stakeholder-additive"
ENTRYPOINT ["/workspace/axolotl-venv/bin/python", "-u", "/workspace/data/athar/handler.py"]
