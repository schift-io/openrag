# syntax=docker/dockerfile:1
# OpenRAG API image: openrag-server (HTTP API), openrag (CLI) and openrag-mcp.
# There is no separate worker process: the API drains the indexing outbox inline after each write,
# and `openrag index drain` drains it on demand.

FROM rust:1.94.0-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p openrag-server --bins

# Install kordoc (HWP/PDF parser and local PP-OCRv5 Korean OCR) in a dedicated stage.
FROM node:22.22.1-bookworm-slim AS kordoc
WORKDIR /opt/kordoc
# Copy pinned manifests first so npm ci stays cached across Rust source changes.
COPY docker/kordoc/package.json docker/kordoc/package-lock.json ./
# Install exact locked production dependencies without audit or funding noise.
RUN npm ci --omit=dev --no-audit --no-fund
# Remove onnxruntime-node binaries for non-Linux OSes and other CPU architectures to shrink the image.
RUN arch="$(node -p process.arch)" \
    && find /opt/kordoc/node_modules/onnxruntime-node/bin -mindepth 2 -maxdepth 2 -not -name linux -exec rm -rf {} + \
    && find /opt/kordoc/node_modules/onnxruntime-node/bin -mindepth 3 -maxdepth 3 -path '*/linux/*' -not -name "$arch" -exec rm -rf {} +
# Pre-warm the PP-OCRv5 model cache at build time so air-gapped deployments never download models.
ENV KORDOC_MODEL_CACHE=/opt/kordoc/models
RUN node -e 'const fs=require("node:fs"),zlib=require("node:zlib");const w=32,h=32;const t=new Uint32Array(256);for(let i=0;i<256;i++){let c=i;for(let k=0;k<8;k++)c=c&1?0xedb88320^(c>>>1):c>>>1;t[i]=c;}const crc=b=>{let c=~0;for(const x of b)c=t[(c^x)&255]^(c>>>8);return (~c)>>>0;};const ch=(ty,d)=>{const a=Buffer.from(ty),l=Buffer.alloc(4),r=Buffer.alloc(4);l.writeUInt32BE(d.length);r.writeUInt32BE(crc(Buffer.concat([a,d])));return Buffer.concat([l,a,d,r]);};const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(w,0);ihdr.writeUInt32BE(h,4);ihdr[8]=8;ihdr[9]=2;const raw=Buffer.alloc(h*(1+w*3),255);for(let y=0;y<h;y++)raw[y*(1+w*3)]=0;fs.writeFileSync("/tmp/warm.png",Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),ch("IHDR",ihdr),ch("IDAT",zlib.deflateSync(raw)),ch("IEND",Buffer.alloc(0))]));' \
    && /opt/kordoc/node_modules/.bin/kordoc /tmp/warm.png --format json --plain --no-images --ocr --silent > /dev/null \
    && rm -f /tmp/warm.png \
    && test -n "$(find /opt/kordoc/models -type f -size +0c)" \
    && chmod -R a+rX /opt/kordoc/models

# Use the Node bookworm-slim base so kordoc has node and onnxruntime-node has libstdc++.
FROM node:22.22.1-bookworm-slim AS api
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /app openrag
WORKDIR /app
COPY --from=build /src/target/release/openrag-server /src/target/release/openrag /src/target/release/openrag-mcp /usr/local/bin/
# Copy kordoc and its pre-warmed read-only OCR model cache into the API image.
COPY --from=kordoc /opt/kordoc /opt/kordoc
COPY configs /app/configs
RUN mkdir -p /app/blobs && chown openrag /app/blobs
USER openrag
# Wire the bundled kordoc CLI and its pre-warmed OCR model cache into the server environment.
ENV OPENRAG_BIND=0.0.0.0:8080 \
    OPENRAG_BLOB_DIR=/app/blobs \
    OPENRAG_KORDOC_BIN=/opt/kordoc/node_modules/.bin/kordoc \
    KORDOC_MODEL_CACHE=/opt/kordoc/models
EXPOSE 8080
CMD ["openrag-server"]
