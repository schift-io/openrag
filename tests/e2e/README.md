# OpenRAG Opaque-Box E2E Test Suite (`tests/e2e`)

This directory contains the requirement-driven, opaque-box `pytest` end-to-end test harness and 4-tier verification suite covering OpenRAG features **F01–F34, F36–F37**.

## Quickstart

Run from the repository root:

```bash
# List all registered tier and feature markers
pytest -c tests/e2e/pytest.ini --markers

# Run harness self-tests (100% offline, no OpenRAG build required)
pytest -c tests/e2e/pytest.ini -m selftest -v

# Run Tier 1 (isolated happy-path tests)
pytest -c tests/e2e/pytest.ini --tier 1

# Run Tier 2 (boundary, negative, and corner-case tests)
pytest -c tests/e2e/pytest.ini --tier 2

# Run Tier 3 (cross-feature pairwise combinations)
pytest -c tests/e2e/pytest.ini --tier 3

# Run Tier 4 (institutional/government/enterprise scenarios)
pytest -c tests/e2e/pytest.ini --tier 4

# Run all E2E tiers (1..4, excluding selftest)
pytest -c tests/e2e/pytest.ini --tier all

# Filter by feature marker (e.g. F07 ARN & RLS)
pytest -c tests/e2e/pytest.ini --tier 1 -m f07
```

Or run directly from `tests/e2e/`:

```bash
cd tests/e2e
pytest -m selftest -v
pytest --tier 1
```

## CLI Options & Environment Variables

| Option | Env Fallback | Default | Description |
|---|---|---|---|
| `--tier` | — | `None` | Filter tests by `1`, `2`, `3`, `4`, or `all` (excludes `selftest` when set) |
| `--mode` | `OPENRAG_E2E_MODE` | `auto` | Execution mode: `auto`, `offline` (deterministic local adapters), or `live` |
| `--target-root` | `OPENRAG_ROOT` | the repository that contains `tests/e2e` | Root directory of the OpenRAG repository under test |
| `--api-url` | `OPENRAG_API_URL` | `None` | Base URL of an already-running OpenRAG HTTP server (if omitted, `EphemeralOpenRagServer` is spawned) |
| `--pg-port` | `OPENRAG_PG_PORT` | `0` | TCP port for ephemeral PostgreSQL 17 cluster (`0` picks a free port on `127.0.0.1`) |
| `--openrag-bin` | `OPENRAG_BIN` | `None` | Explicit path to the `openrag` binary |

## Offline vs. PostgreSQL 17 Ephemeral Mode

- **Harness Selftests (`-m selftest`)**: Run offline using temporary mock executables, in-process `ThreadingHTTPServer` test doubles, inline Python MCP stdio test doubles, and pure-Python fixture generators.
- **PostgreSQL 17 Ephemeral Cluster (`lib/pg_harness.py`)**:
  - Uses `LC_ALL=C` and `--locale=C` with `initdb`.
  - Starts PostgreSQL via `pg_ctl -o "-p <port> -h 127.0.0.1 -k ''" -w start` (`-k ''` disables Unix domain sockets so long paths never hit macOS's 103-byte `sun_path` limit).
  - Creates a dedicated `NOSUPERUSER` role (`openrag_app`) and runs principal queries with `SET LOCAL ROLE openrag_app` and all 7 `app.*` GUCs (`app.bypass_rls='false'`) so Fail-Closed RLS is genuinely enforced.
  - On macOS agent sandboxes where `shmget` / loopback TCP `127.0.0.1` is blocked, live PG17 and local TCP server tests require `BypassSandbox=true`.
  - Local Homebrew PG 17 provides `vector`, `pg_trgm`, and `pgcrypto` (fallback mode when `pg_pinecone` / `pg_search` are not installed).

## Non-Skipping Policy (Tiers 1–4)

Tests in **Tiers 1–4** MUST NEVER silently `pytest.skip()` when OpenRAG binaries, migrations, HTTP endpoints, plugins, or Next.js routes are missing or unbuilt. Missing targets fail immediately with an explicit actionable `pytest.fail` error so CI and release gates always receive genuine PASS/FAIL signal.

## Coverage Summary (F01–F34, F36–F37)

- **F01–F06**: Clean Apache-2.0 repo, Docker standard/airgap stacks, YAML configs, 10-schema PG DDL (`000`–`008`), capability-guarded fallback mode, `256d` multi-space vectors.
- **F07–F10**: ARN & `dept_admin` vs `peer` Fail-Closed RLS, Permission Groups, Permission Levels (`1..10`), Explicit Deny > Allow & 3-axis Policy Simulator.
- **F11–F13**: M1 `kordoc` HWP/HWPX/HML parser, Schift OCR + mode guard + dewarp + `pdf_native`, M2 WebP/PNG thumbnailer & `256d` visual embeddings.
- **F14–F17**: M3 multi-detector PII engine, 2-stage PII Capsule & AES-256-GCM vault, configurable PII & Safe-Word Guard (`allow|warn|mask|block`), full-layer Cascade Redaction (including BBox pixel black-box masking).
- **F18–F20**: M4 pluggable `MemoryBackend` (`openrag_pg` | `kumiho`, `mirror_to: kumiho`), `engage`/`reflect` cognitive memory operations, belief-revision typed edges (`derived_from`, `supports`, `contradicts`, `supersedes`), CCLG immutable memory patch ledger, promote ladder (`private -> dept -> group -> org -> wiki`), DB-native LLM Wiki + KoParaRel 6-label edges + `.md`/Obsidian sync.
- **F21–F23**: M5 `pg_pinecone` 3-engine table switching (`native|pgvector|remote`), S3 Vectors STS ARN & OSS/BM25 adapters, incremental Outbox + Watermarks + Quarantine.
- **F24–F27**: M6 `256d` multi-space hybrid retriever & admission, M7 policy LLM router (`L8+` -> local vLLM), M8 4-axis golden set, 3-Track RAG Bench, `/v1/eval/ab-search`, AutoRAG greedy YAML optimizer & security gates.
- **F28–F29**: `openrag` CLI & Axum HTTP API server, local stdio (`openrag mcp --stdio` / `openrag-mcp`) & Streamable HTTP `/mcp` server (13 tools: `rag_search`, `rag_ask`, `rag_ingest`, `rag_get_page`, `wiki_search`, `wiki_get`, `memory_search`, `memory_patch`, `memory_engage`, `memory_reflect`, `guard_check`, `concept_link`, `concept_resolve`) + client snippets.
- **F30–F34**: Plugins (`kordoc-mcp-bridge`, `docufinder-connector`, `ai-memory-hooks`), Tier 1 zero-jargon `/app` console, Tier 2 `/studio` 6 operational consoles.
- **F36–F37**: Canonical Concept / Alias Equivalence Graph (`wiki.concepts`, `wiki.concept_aliases`, `openrag concept link/unlink/list`, `/v1/concepts*`, `concept_link`/`concept_resolve`, NFKC/obfuscation-resistant Safe-Word guard) and First-Class Tag-Based Search & Faceted Filtering (`tags TEXT[]` + GIN, `tags_any`/`tags_all`/`tags_exclude`, inline `#tag`/`tag:...`, `expand_tag_aliases`, tag facets).
