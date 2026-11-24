# syntax=docker/dockerfile:1
# Versions are pinned to the build verified on 2026-11-24 (clean clone, no cache): ParadeDB 0.27.0
# (pg_search 0.27.0, PostgreSQL 17.12), pg_pinecone main at d860f6f (v0.13.0 + 5 commits), Rust 1.100.0.
ARG PG_MAJOR=17
ARG PARADEDB_VERSION=0.27.0
FROM paradedb/paradedb:${PARADEDB_VERSION}-pg${PG_MAJOR} AS pg_pinecone_builder

ARG PG_MAJOR
ARG PG_PINECONE_REPO=https://github.com/pinecone-io/pg_pinecone.git
ARG PG_PINECONE_REF=d860f6fd3a21733d75ed2cde2b5b5e71e79bd3ac
ARG RUST_TOOLCHAIN=1.100.0
# amd64 CPU baseline for pg_pinecone. x86-64-v2 (SSE4.2 + POPCNT) also runs on CPUs and VMs without AVX2,
# such as Proxmox's default CPU type; pg_pinecone's distance and code-scan kernels still pick AVX-512 or AVX2
# at runtime (pg_pinecone docs/runbook.md, "Self-builds"). x86-64-v3 would crash those machines.
ARG PG_PINECONE_TARGET_CPU=x86-64-v2
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang libclang-dev pkg-config curl ca-certificates git jq \
    postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}" --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
# PG_PINECONE_REF may be a commit, a tag or a branch: pull just that one into an empty repository.
RUN git init -q pg_pinecone \
    && git -C pg_pinecone pull -q --depth 1 "${PG_PINECONE_REPO}" "${PG_PINECONE_REF}" \
    && git -C pg_pinecone log -1 --format='pg_pinecone %H %cs'
WORKDIR /build/pg_pinecone

RUN PGRX_VER=$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
        | jq -r '.packages[] | select(.name=="pg_pinecone") | .dependencies[] | select(.name=="pgrx") | .req' \
        | tr -d '^=~') \
    && PGRX_VER="${PGRX_VER:-0.19.2}" \
    && cargo install cargo-pgrx --version "${PGRX_VER}" --locked \
    && cargo pgrx init --pg${PG_MAJOR} "$(which pg_config)" \
    && if [ "${TARGETARCH}" = "amd64" ]; then export RUSTFLAGS="-C target-cpu=${PG_PINECONE_TARGET_CPU}"; fi \
    && cargo pgrx install --release --pg-config "$(which pg_config)"

FROM paradedb/paradedb:${PARADEDB_VERSION}-pg${PG_MAJOR}
ARG PG_MAJOR

COPY --from=pg_pinecone_builder \
    /usr/lib/postgresql/${PG_MAJOR}/lib/pg_pinecone.so \
    /usr/lib/postgresql/${PG_MAJOR}/lib/pg_pinecone.so
COPY --from=pg_pinecone_builder \
    /usr/share/postgresql/${PG_MAJOR}/extension/pg_pinecone* \
    /usr/share/postgresql/${PG_MAJOR}/extension/

# Preload pg_search (Tantivy + Lindera Korean mecab-ko-dic) and pg_pinecone
RUN echo "shared_preload_libraries = 'pg_search,pg_pinecone'" >> /usr/share/postgresql/${PG_MAJOR}/postgresql.conf.sample
