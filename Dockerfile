# syntax=docker/dockerfile:1
# OpenRAG API image: openrag-server (HTTP API), openrag (CLI) and openrag-mcp.
# There is no separate worker process: the API drains the indexing outbox inline after each write,
# and `openrag index drain` drains it on demand.

FROM rust:1.94.0-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p openrag-server --bins

FROM debian:bookworm-slim AS api
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /app openrag
WORKDIR /app
COPY --from=build /src/target/release/openrag-server /src/target/release/openrag /src/target/release/openrag-mcp /usr/local/bin/
COPY configs /app/configs
RUN mkdir -p /app/blobs && chown openrag /app/blobs
USER openrag
ENV OPENRAG_BIND=0.0.0.0:8080 \
    OPENRAG_BLOB_DIR=/app/blobs
EXPOSE 8080
CMD ["openrag-server"]
