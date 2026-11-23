# syntax=docker/dockerfile:1
# OpenAI-compatible /v1/embeddings server (google/embeddinggemma-2 by default). Build context:
# the repository root. Weights are not baked in: the standard stack downloads them into a volume
# at /models on first start (HF_HUB_OFFLINE=0); the air-gapped one mounts a Hugging Face hub
# cache there, filled beforehand on a connected machine with
#   hf download google/embeddinggemma-2 --cache-dir ./models
# Any other OpenAI-compatible embeddings server can stand in; OpenRAG only needs the URL.
# CPU build of torch; set EMBED_DEVICE=cuda (with a CUDA torch build) and EMBED_DTYPE=bfloat16 on GPU.
# Pillow and torchvision: sentence-transformers loads the model's image and video processors even
# for text-only use.

FROM python:3.12.12-slim-bookworm
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 \
    && pip install --no-cache-dir sentence-transformers==6.3.0 transformers==5.24.0 numpy==2.5.5 pillow==13.0.0 \
    && useradd --system --uid 10001 --home-dir /app embed \
    && mkdir -p /models && chown embed /models
WORKDIR /app
COPY scripts/bench/st_embed_server.py /app/st_embed_server.py
USER embed
# The download cache (hf_xet) lives with the weights: the home directory is not writable.
ENV HF_HUB_OFFLINE=1 \
    HF_HUB_CACHE=/models \
    HF_XET_CACHE=/models/.xet \
    EMBED_MODEL=google/embeddinggemma-2 \
    EMBED_PORT=8082 \
    EMBED_DEVICE=cpu \
    EMBED_DTYPE=float32
EXPOSE 8082
CMD ["sh", "-c", "exec python /app/st_embed_server.py --model \"$EMBED_MODEL\" --host 0.0.0.0 --port \"$EMBED_PORT\" --device \"$EMBED_DEVICE\" --dtype \"$EMBED_DTYPE\""]
