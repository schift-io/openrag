# OpenRAG

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![PostgreSQL 17/18](https://img.shields.io/badge/PostgreSQL-17%20%7C%2018-336791.svg)](migrations/)
[![Rust Workspace](https://img.shields.io/badge/Rust-2021%20(1.85%2B)-orange.svg)](Cargo.toml)

**OpenRAG** is an open-source, institutional- and enterprise-grade RAG platform built in **Rust** and **PostgreSQL 17/18** (`pg_pinecone` + `pgvector` + `pg_search` Lindera Korean BM25), designed for zero-trust access control, Korean document fidelity, live memory/wiki evolution, and declarative AutoRAG optimization.

---

## Core Architecture & 5 Differentiators

```text
User / External AI (Claude Desktop/Code, Cursor, Codex, Docufinder, Kordoc)
   │
   ├─► App API (/api/app) ──► Chat, Upload, Wiki, Memory, Connect (backend for a user console)
   ├─► Studio API (/api/studio) ─► Pipeline YAML, Engine Switcher, IAM Simulator, Guard, Bench
   ├─► HTTP API (/v2/buckets, /v1/chat/completions, /v1/eval/ab-search, /mcp)
   └─► Local Stdio MCP (openrag mcp --stdio / openrag-mcp)
                                     │
   ┌─────────────────────────────────▼──────────────────────────────────┐
   │ Rust Core Platform                                                 │
   │  • 3-Axis IAM (ARN + Permission Group + Permission Level) & RLS    │
   │  • Configurable PII & Safe-Word Guard (allow | warn | mask | block)│
   │  • M1 Parse/Dewarp (kordoc, Schift OCR) ─► M2 Thumbnailer (256d)   │
   │  • M3 PII Capsule (AES-256-GCM) & Full-Layer Cascade Redaction     │
   │  • M4 CCLG Memory Patch Ledger & DB-Native Wiki (KoParaRel + .md)  │
   │  • M5 Incremental Outbox (SHA-256 No-Op, Watermarks, Quarantine)   │
   │  • M6 256d Multi-Space Hybrid Retriever (RRF/Convex) & Admission   │
   │  • M7 Policy LLM Router (no bundled LLM; L8+ -> on-prem only)      │
   │  • M8 4-Axis Golden Set, 3-Track Bench & AutoRAG Greedy Optimizer  │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     ▼
   PostgreSQL 17/18 (10 Schemas: iam, vectors, corpus, pii, memory,
                     wiki, indexing, gateway, plugins, eval)
   + Optional External Vectors (Amazon S3 Vectors STS ARN | OSS Vector)
```

1. **`pg_pinecone` 3-Engine + `256d` Multi-Space + S3 Vectors/OSS**: Table-level switching between `native` (RaBitQ IVF, zero external account), `pgvector` (HNSW), and `remote` (Pinecone Serverless) without changing SQL queries, plus optional Amazon S3 Vectors (per-tenant STS bucket-ARN isolation) and OSS vector adapters.
2. **Korean Document Ecosystem Native**: Table-preserving HWP/HWPX/HML parsing (`kordoc`), Schift OCR with automatic confidentiality mode guard (`share` / `private` / `zdr` / local vLLM) and `schift-dewarp`, plus ParadeDB `pg_search` Tantivy BM25 with Lindera Korean morphology (`pg_trgm` + `tsvector` fallback when compiled extensions are absent locally).
3. **Zero-Trust 3-Axis IAM, Fail-Closed RLS & PII/Safe-Word Guard**:
   - Unified authorization across **ARN** (`arn:openrag:<org>:<dept_path>:<scope>:<type>:L<1..10>/<id>`), **Permission Groups** (`iam.groups`, `iam.group_members`, `iam.resource_group_bindings`), and **Permission Levels** (`auth_level 1..10`), enforcing Explicit Deny > Allow and strict `dept_admin` vs `peer` isolation on `private` resources in PostgreSQL Fail-Closed RLS (`FORCE ROW LEVEL SECURITY`) and Rust `openrag-iam`.
   - 2-Stage AES-256-GCM PII Capsule (`schift-ko-pii-v7`, optional `ko-pii`, custom dictionaries), **Configurable PII & Safe-Word Guard** (`allow` | `warn` | `mask` | `block` across `inbound_query`, `llm_egress`, `mcp_egress`, `ingestion_gate`), and **Cascade Redaction** (text + thumbnail BBox black-box pixel masking + immutable audit trail).
4. **Living Knowledge & Incremental Outbox**: CCLG-style immutable `memory.episodes` + `memory.entries` + `memory.patches` (`amend`/`supersede`/`retract`/`redact`/`promote_scope`/`promote_wiki`), DB-Native LLM Wiki with KoParaRel 6-label edges and `.md`/Obsidian bidirectional sync, and `indexing.outbox` SHA-256 No-Op skip, watermarks, and 3-attempt quarantine.
5. **AutoRAG Greedy Optimizer & 2-Tier API**: Declarative `search_space.yaml` greedy node optimization with mandatory hard security gates (`0 PII leak`, `0 ARN/Group/Level violation`) and strict separation between the user-facing `/api/app` and the operator-only `/api/studio`.

---

## 3-Minute Quickstart

### Option A: Standard Docker Stack
```bash
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)   # keep it: it wraps the PII vault keys
docker compose -f docker/docker-compose.yml --profile app up -d postgres openrag-api
```

### Option B: Air-Gapped On-Premises Docker Stack
```bash
# on a connected machine, from docker/: fetch the embedding weights into ./models
hf download google/embeddinggemma-2 --cache-dir ./models
# on the air-gapped host, from docker/
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)
export AIRGAP_LLM_URL=http://<vllm-host>:8000/v1 AIRGAP_LLM_MODEL=<served model>   # optional
docker compose -f docker-compose.yml -f docker-compose.airgap.yml --profile app --profile airgap \
  up -d postgres local-embed openrag-api
```

### Then: one command from a running server to asking questions
```bash
# the stack's database, needed once to create the first organization and admin
export DATABASE_URL=postgres://openrag:openrag_dev_pass@127.0.0.1:5432/openrag OPENRAG_OPERATOR=1
openrag init --slug acme --name "ACME" --docs ./my-docs --connect claude
```
`openrag init` creates the organization, a department and an admin (permission level 10), mints
the admin's API key into `~/.config/openrag/config.toml` (mode 600), creates the bucket `docs`,
ingests every supported file under `./my-docs` and registers the server's `/mcp` in Claude Code
(`claude mcp add --scope user`). Then ask Claude Code about the documents: it searches them with
`rag_search` and answers with the sources. Running it again reuses the organization and bucket.
`--connect codex` registers Codex instead (it reads the key from `$OPENRAG_API_KEY`).

The API runs as one process (it drains the indexing outbox inline; there is no worker service).
Page thumbnails and figure crops are off by default (`thumbnail.enabled: false`): nothing shows them
yet. When enabled they go to an S3-compatible bucket (AWS S3, RustFS, …) set through
`OPENRAG_S3_ENDPOINT`, `OPENRAG_S3_BUCKET`, `OPENRAG_S3_REGION`, `OPENRAG_S3_ACCESS_KEY_ID`,
`OPENRAG_S3_SECRET_ACCESS_KEY` and `OPENRAG_S3_URL_STYLE` (`path` | `virtual`).

### Option C: Local Rust & PostgreSQL 17 Verification
Run all offline unit tests and strict lints:
```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Spin up an ephemeral local PostgreSQL 17 cluster, apply `migrations/000_init.sql`–`007_memory_redaction_indexing.sql`, verify non-superuser `openrag_app` Fail-Closed RLS and 3-axis parity tests, and tear down automatically:
```bash
./scripts/test-pg-local.sh
```

---

## Configuration (`configs/`)

- `configs/pipeline.default.yaml` — Standard profile (`256d`, `pg_pinecone: native`, `kordoc`, Schift OCR with `auth_level` mode guard, `schift_v7` PII Capsule, `pii.guard` PII & Safe-Word rules, RRF `k=60`, `L8+` local vLLM routing).
- `configs/pipeline.airgap.yaml` — Air-gapped profile (local OCR/vLLM/256d embedder, strict `default_action: block` guard, `local_only_min_auth_level: 1`).
- `configs/search_space.example.yaml` — AutoRAG greedy node search space with mandatory `require_zero_pii_leak: true` and `require_zero_arn_violation: true` security gates.

All YAML profiles map 1:1 to typed Rust structs in `openrag-core` with `#[serde(deny_unknown_fields)]`.

---

## CLI & API Overview

> Note: OpenRAG ships no graphical console. Users reach it through MCP clients and the HTTP API; `/api/app` and `/api/studio` are the backends a console would call.

### `openrag` CLI
- `openrag up --profile <standard|airgap>` — Start platform stack
- `openrag init [--docs <folder>] [--connect claude|codex]` — One-step setup: first organization and admin key, CLI config, a bucket, documents, MCP clients
- `openrag org init` — First organization, department and admin plus an API key (local operator, `DATABASE_URL`)
- `openrag bucket {create|list}` — Buckets
- `openrag connect {claude|codex}` — Register this server's `/mcp` in Claude Code or Codex
- `openrag ingest <file or folder> --bucket <slug> [--scope private|dept|group|org] [--auth-level 1..10]`
- `openrag index {engine|mirror|status|retry}` — Switch `pg_pinecone` engines, mirror S3 Vectors/OSS, inspect outbox/quarantine
- `openrag memory {patch|promote}` — Append CCLG patches or promote `private -> dept -> wiki`
- `openrag redact run` — Execute full-layer Cascade Redaction (including thumbnail black-box masking)
- `openrag wiki {import|export}` — Bidirectional `.md` / Obsidian vault sync
- `openrag bench {run|optimize}` & `openrag pipeline promote` — 3-Track Bench & AutoRAG optimizer
- `openrag mcp --stdio` (or `openrag-mcp`) — Start local stdio MCP server

### HTTP API
- **Core API**: `/v2/buckets` (CRUD, ingest, search), `/v1/chat/completions` (OpenAI & Docufinder Anything compatible), `/v1/eval/ab-search`, `/mcp` (Streamable HTTP JSON-RPC 2.0)
- **App API (`/api/app`)**: `chat`, `upload`, `pages/{id}/thumbnail` (404 while thumbnails are off), `wiki`, `memory`, `concepts`, `connect`, `feedback`
- **Studio API (`/api/studio`, org admins)**: `pipeline`, `index`, `iam`, `pii` (guard sandbox, cascade redaction), `gateway`, `eval`, `wiki`, `knowledge`, `concepts`, `plugins`

---

## Connect External AI Clients via MCP

OpenRAG exposes identical RAG, Wiki, Memory, and Guard capabilities over both **Local Stdio MCP** (`openrag mcp --stdio` / `openrag-mcp`) and **Streamable HTTP MCP** (`/mcp`). Every tool call enforces the caller's ARN, Permission Group, Permission Level, PostgreSQL RLS, and PII/Safe-Word Guard.

### LLM

OpenRAG bundles no LLM. By default the MCP client is the LLM: Claude Code, Codex or Cursor calls
`rag_search` and writes the answer itself. `/v1/chat/completions` and `rag_ask` answer only when a
provider is configured (otherwise HTTP 503):

- `OPENRAG_LOCAL_LLM_URL` + `OPENRAG_LOCAL_LLM_MODEL` — an on-prem OpenAI-compatible chat completions
  server (vLLM or any compatible one). Requests that must stay on-prem (all of them in the
  air-gapped profile; L8+ or unmasked-PII ones in the standard profile) can only use this one.
- `OPENRAG_LLM_CLI=claude` or `codex` (optional `OPENRAG_LLM_CLI_MODEL`, `OPENRAG_LLM_CLI_COMMAND`) —
  runs the Claude Code (`claude -p`, no tools, no MCP servers) or Codex (`codex exec`, read-only
  sandbox) CLI on the server host, logged in as the user the server runs as. The prompt leaves for
  Anthropic / OpenAI, so it is an external provider: masked PII only, up to L7. Run the server
  under a dedicated service account: the CLI also reads that account's own instructions
  (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`) and settings, which would otherwise shape every
  user's answers.

These seed an org's providers the first time it asks for an answer; change them later in
`gateway.providers`. In the air-gapped profile the server calls only the hosts it is configured
with (`OPENRAG_LOCAL_LLM_URL`, `OPENRAG_TEXT_EMBED_URL`, `OPENRAG_SCHIFT_OCR_URL`,
`OPENRAG_S3_ENDPOINT`) plus `egress.allow` in `configs/pipeline.airgap.yaml`; a provider pointed
anywhere else is refused with 403, and the CLI providers are off.

### PII

Stored text, search, embeddings and the HTTP API keep PII as written: whoever may read a document
sees it. PII is replaced with `[PII:<TYPE>:<hash>]` tokens only where text goes to an AI — the
prompt and context sent to an LLM provider, and every MCP tool result. `/v1/chat/completions`
answers come back with the original values; MCP `rag_ask` keeps the tokens. Names are detected
after a label (`성명:`, `담당자:`, …) or an attached title (`김민준과장`); a bare name in running
text is not.

### Exposed MCP Tools (9)
- `rag_search` — `256d` vector search over chunks, wiki and memory with citations (BM25 fusion and the page-thumbnail space are opt-in)
- `rag_ask` — Grounded RAG Q&A through the policy-enforced LLM router
- `rag_ingest` — Ingest document/text with scope, permission groups, and clearance level
- `rag_get_page` — Fetch redacted page text, thumbnail URI, and `bbox_norm_1000` coordinates
- `wiki_search` / `wiki_get` — Query and retrieve DB-native LLM Wiki pages
- `memory_search` / `memory_patch` — Search active memory view and append CCLG patches
- `guard_check` — Inspect text against active PII + Safe-Word Guard policies (`allow` | `warn` | `mask` | `block`)

### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "openrag": {
      "command": "openrag",
      "args": ["mcp", "--stdio"],
      "env": {
        "OPENRAG_API_KEY": "orag_your_api_key_here",
        "DATABASE_URL": "postgres://openrag_app@127.0.0.1:5432/openrag"
      }
    }
  }
}
```

`openrag connect claude` and `openrag connect codex` register the HTTP endpoint with the key from the
CLI config. The stdio server (`openrag mcp --stdio`, `openrag-mcp`) talks to the database directly,
so it only runs on the server host with `DATABASE_URL` set.

### Claude Code
```bash
openrag connect claude    # same as:
claude mcp add --transport http --scope user openrag http://localhost:8080/mcp \
  --header "Authorization: Bearer ${OPENRAG_API_KEY}"
```

### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "openrag-http": {
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${OPENRAG_API_KEY}"
      }
    }
  }
}
```

### Codex
```bash
openrag connect codex     # same as:
codex mcp add openrag --url http://localhost:8080/mcp --bearer-token-env-var OPENRAG_API_KEY
export OPENRAG_API_KEY=orag_…   # Codex reads the key from this variable
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
