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
   ├─► Tier 1 User Console (/app) ──► Zero-jargon workspace (Chat, Upload, Wiki, Memory, Connect)
   ├─► Tier 2 Ops Studio (/studio) ─► Pipeline YAML, Engine Switcher, IAM Simulator, Guard, Bench
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
   │  • M7 Policy LLM Router (L8+ -> Local vLLM; Masked -> OpenRouter)  │
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
5. **AutoRAG Greedy Optimizer & 2-Tier Console**: Declarative `search_space.yaml` greedy node optimization with mandatory hard security gates (`0 PII leak`, `0 ARN/Group/Level violation`) and strict separation between Tier 1 `/app` (zero technical jargon) and Tier 2 `/studio`.

---

## 3-Minute Quickstart

### Option A: Standard Docker Stack
```bash
docker compose -f docker/docker-compose.yml up -d
```

### Option B: Air-Gapped On-Premises Docker Stack
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.airgap.yml up -d
```

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

> Note: Core data models, database migrations/RLS, and 3-axis IAM engine (`openrag-core`, `openrag-db`, `openrag-iam`) are implemented in Milestone 1; the `openrag` CLI binary, HTTP API, MCP server, plugins, and Next.js 2-Tier Console are wired in subsequent milestones.

### `openrag` CLI
- `openrag up --profile <standard|airgap>` — Start platform stack
- `openrag ingest <path> --bucket <slug> [--scope private|dept|group|org] [--auth-level 1..10]`
- `openrag index {engine|mirror|status|retry}` — Switch `pg_pinecone` engines, mirror S3 Vectors/OSS, inspect outbox/quarantine
- `openrag memory {patch|promote}` — Append CCLG patches or promote `private -> dept -> wiki`
- `openrag redact run` — Execute full-layer Cascade Redaction (including thumbnail black-box masking)
- `openrag wiki {import|export}` — Bidirectional `.md` / Obsidian vault sync
- `openrag bench {run|optimize}` & `openrag pipeline promote` — 3-Track Bench & AutoRAG optimizer
- `openrag mcp --stdio` (or `openrag-mcp`) — Start local stdio MCP server

### HTTP API & 2-Tier Console
- **Core API**: `/v2/buckets` (CRUD, ingest, search), `/v1/chat/completions` (OpenAI & Docufinder Anything compatible), `/v1/eval/ab-search`, `/mcp` (Streamable HTTP JSON-RPC 2.0)
- **Tier 1 User Console (`/app`)**: `/app/chat`, `/app/upload`, `/app/wiki`, `/app/memory`, `/app/connect`
- **Tier 2 Admin/Ops Studio (`/studio`)**: Pipeline Studio, Index & Engine Ops, IAM/Groups/Levels/ARN Simulator, PII/Safe-Word Guard Sandbox & Cascade Redaction, Golden Set & AutoRAG Bench, Wiki/Plugins Ops

---

## Connect External AI Clients via MCP

OpenRAG exposes identical RAG, Wiki, Memory, and Guard capabilities over both **Local Stdio MCP** (`openrag mcp --stdio` / `openrag-mcp`) and **Streamable HTTP MCP** (`/mcp`). Every tool call enforces the caller's ARN, Permission Group, Permission Level, PostgreSQL RLS, and PII/Safe-Word Guard.

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
        "OPENRAG_URL": "http://localhost:8080",
        "DATABASE_URL": "postgres://openrag_app@127.0.0.1:5432/openrag"
      }
    }
  }
}
```

### Claude Code
```bash
# Local stdio transport
claude mcp add openrag -- openrag mcp --stdio

# Or Streamable HTTP transport
claude mcp add --transport http openrag http://localhost:8080/mcp \
  --header "Authorization: Bearer ${OPENRAG_API_KEY}"
```

### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "openrag-local": {
      "command": "openrag-mcp",
      "env": {
        "OPENRAG_API_KEY": "${OPENRAG_API_KEY}",
        "OPENRAG_URL": "http://localhost:8080"
      }
    },
    "openrag-http": {
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${OPENRAG_API_KEY}"
      }
    }
  }
}
```

### Codex (`~/.codex/config.toml`)
```toml
[mcp_servers.openrag]
command = "openrag"
args = ["mcp", "--stdio"]

[mcp_servers.openrag.env]
OPENRAG_API_KEY = "orag_your_api_key_here"
OPENRAG_URL = "http://localhost:8080"
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
