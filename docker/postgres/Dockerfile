# syntax=docker/dockerfile:1
ARG PG_MAJOR=17
FROM paradedb/paradedb:latest-pg${PG_MAJOR} AS pg_pinecone_builder

ARG PG_MAJOR
ARG PG_PINECONE_REPO=https://github.com/pinecone-io/pg_pinecone.git
ARG PG_PINECONE_REF=main
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang libclang-dev pkg-config curl ca-certificates git jq \
    postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
RUN git clone --depth 1 --branch "${PG_PINECONE_REF}" "${PG_PINECONE_REPO}" pg_pinecone
WORKDIR /build/pg_pinecone

RUN PGRX_VER=$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
        | jq -r '.packages[] | select(.name=="pg_pinecone") | .dependencies[] | select(.name=="pgrx") | .req' \
        | tr -d '^=~') \
    && PGRX_VER="${PGRX_VER:-0.19.2}" \
    && cargo install cargo-pgrx --version "${PGRX_VER}" --locked \
    && cargo pgrx init --pg${PG_MAJOR} "$(which pg_config)" \
    && if [ "${TARGETARCH}" = "amd64" ]; then export RUSTFLAGS="-C target-cpu=x86-64-v3"; fi \
    && cargo pgrx install --release --pg-config "$(which pg_config)"

FROM paradedb/paradedb:latest-pg${PG_MAJOR}
ARG PG_MAJOR

COPY --from=pg_pinecone_builder \
    /usr/lib/postgresql/${PG_MAJOR}/lib/pg_pinecone.so \
    /usr/lib/postgresql/${PG_MAJOR}/lib/pg_pinecone.so
COPY --from=pg_pinecone_builder \
    /usr/share/postgresql/${PG_MAJOR}/extension/pg_pinecone* \
    /usr/share/postgresql/${PG_MAJOR}/extension/

# Preload pg_search (Tantivy + Lindera Korean mecab-ko-dic) and pg_pinecone
RUN echo "shared_preload_libraries = 'pg_search,pg_pinecone'" >> /usr/share/postgresql/${PG_MAJOR}/postgresql.conf.sample
