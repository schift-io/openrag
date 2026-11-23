# E2E Test Suite Ready

## Test Runner
- Command: `pytest -c tests/e2e/pytest.ini --tier all` (or `/opt/homebrew/Caskroom/miniconda/base/bin/pytest -c tests/e2e/pytest.ini --tier all`)
- Selftests: `pytest -c tests/e2e/pytest.ini -m selftest -v` (`16 passed`)
- Expected: all `443` Tiers 1–4 tests pass with exit code `0` (zero `pytest.skip` / `skipif` / `xfail` in Tiers 1–4; `BypassSandbox=true` required on macOS agent sandboxes for ephemeral PG17 `shmget` and loopback HTTP)

## Coverage Summary
| Tier | Count | Description |
|------|------:|-------------|
| 1. Feature Coverage (`--tier 1`) | 194 | `>= 5` isolated happy-path tests per feature (`F01`–`F34`, `F36`, `F37`; `36` features) |
| 2. Boundary & Corner (`--tier 2`) | 193 | `>= 5` BVA, corner, and negative tests per feature (`F01`–`F34`, `F36`, `F37`) |
| 3. Cross-Feature (`--tier 3`) | 38 | Pairwise cross-feature combination tests covering all `36` features (`>= 2` features/test) |
| 4. Real-World Application (`--tier 4`) | 18 | Institutional/government/enterprise E2E scenarios (`S01`–`S18` incl. Follow-up 4 Concept/Alias + Tag Search) |
| **Total Tiers 1–4 (`--tier all`)** | **443** | Verified via `pytest -c tests/e2e/pytest.ini --tier all --collect-only -q` |
| Harness & Matrix Selftest (`-m selftest`) | 16 | `11` harness selftests + `5` coverage-matrix/anti-skip checks (`16 passed`) |
| **Total Collected (`tests/e2e/`)** | **459** | `E2E-M1`..`E2E-M5` all `GATE PASS` (`APPROVE` + `CONFIRMED` + `CLEAN`) |

## Feature Checklist
| Feature | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------|:------:|:------:|:------:|:------:|
| F01 Clean Apache-2.0 Repo | 5 | 5 | ✓ | ✓ |
| F02 Docker Standard & Airgap Stacks | 5 | 5 | ✓ | ✓ |
| F03 Declarative YAML Configs | 5 | 5 | ✓ | ✓ |
| F04 10-Schema PG 17/18 DDL (`000`–`008`) | 5 | 5 | ✓ | ✓ |
| F05 Capability-Guarded DB Fallback Mode | 5 | 5 | ✓ | ✓ |
| F06 `128d` Multi-Space Vector Columns | 5 | 5 | ✓ | ✓ |
| F07 ARN & `dept_admin` vs `peer` Fail-Closed RLS | 5 | 5 | ✓ | ✓ |
| F08 First-Class Permission Groups | 5 | 5 | ✓ | ✓ |
| F09 First-Class Permission Levels | 5 | 5 | ✓ | ✓ |
| F10 Explicit Deny > Allow & 3-Axis Policy Simulator | 5 | 5 | ✓ | ✓ |
| F11 M1 `kordoc` HWP/HWPX/HML Parser | 6 | 6 | ✓ | ✓ |
| F12 M1 Schift OCR + Mode Guard + Dewarp + `pdf_native` | 5 | 5 | ✓ | ✓ |
| F13 M2 Thumbnailer & `128d` Visual Embed | 5 | 5 | ✓ | ✓ |
| F14 M3 Multi-Detector PII Engine | 5 | 5 | ✓ | ✓ |
| F15 M3 2-Stage PII Capsule & AES-256-GCM Vault | 5 | 5 | ✓ | ✓ |
| F16 M3 Configurable PII & Safe-Word Guard | 6 | 6 | ✓ | ✓ |
| F17 M3 Full-Layer Cascade Redaction | 5 | 5 | ✓ | ✓ |
| F18 M4 CCLG Memory Ledger, `MemoryBackend` & Cognitive Loops | 6 | 6 | ✓ | ✓ |
| F19 M4 Memory Promote Ladder | 5 | 5 | ✓ | ✓ |
| F20 M4 DB-Native LLM Wiki + KoParaRel + `.md` Sync | 6 | 6 | ✓ | ✓ |
| F21 M5 `pg_pinecone` 3-Engine Table-Level Switching | 5 | 5 | ✓ | ✓ |
| F22 M5 External Vector Adapters (S3 Vectors & OSS) + BM25 | 5 | 6 | ✓ | ✓ |
| F23 M5 Incremental Outbox, Watermarks & Quarantine | 5 | 6 | ✓ | ✓ |
| F24 M6 `128d` Multi-Space Hybrid Retriever & Admission | 6 | 6 | ✓ | ✓ |
| F25 M7 Policy-Enforced LLM Router | 5 | 5 | ✓ | ✓ |
| F26 M8 4-Axis Golden Set & Feedback Promotion | 5 | 5 | ✓ | ✓ |
| F27 M8 3-Track RAG Bench, A/B Search & AutoRAG Optimizer | 6 | 6 | ✓ | ✓ |
| F28 `openrag` CLI & Axum HTTP API Server | 6 | 6 | ✓ | ✓ |
| F29 Local Stdio & Streamable HTTP MCP Server (`13` Tools) | 7 | 7 | ✓ | ✓ |
| F30 P1 `kordoc-mcp-bridge` Plugin | 5 | 5 | ✓ | ✓ |
| F31 P2 `docufinder-connector` Plugin | 5 | 5 | ✓ | ✓ |
| F32 P3 `ai-memory-hooks` Plugin | 5 | 5 | ✓ | ✓ |
| F33 Tier 1 User Console (`/app`) | 6 | 6 | ✓ | ✓ |
| F34 Tier 2 Admin/Ops Studio (`/studio`) | 8 | 6 | ✓ | ✓ |
| F36 Canonical Concept / Alias Equivalence Graph (Follow-up 4) | 6 | 5 | ✓ | ✓ |
| F37 First-Class Tag-Based Search & Faceted Filtering (Follow-up 4) | 5 | 5 | ✓ | ✓ |

## Verification Snapshot at Publication
- `pytest -c tests/e2e/pytest.ini --tier all --collect-only -q`: `443` collected (`T1=194, T2=193, T3=38, T4=18`)
- `pytest -c tests/e2e/pytest.ini -m selftest -v`: `16 passed, 443 deselected` (`11` harness selftests + `5` coverage-matrix selftests)
- Live PG17 Pure-DB Subset (`f04`–`f10` + `schema_008` + 3 pure-PG17 Tier 3 pairwise tests): `26 passed, 41 deselected` (`BypassSandbox=true`)
- Whole-Suite Forensic Audit (`teamwork_auditor_e2e_m5_r2`): `CLEAN`; Primary Reviewer: `APPROVE`; Adversarial Challenger: `CONFIRMED`.
- Note for `M4`/`M6` runtime alignment: Earlier Tier 1/2 files (`lib/fixtures.py:592,598`, `tier1/test_f30_f34_plugins_web.py:782`, and 8 payloads in `tier2/`) send `"kind"` in `guard-rules` payloads while `003_pii_capsule.sql` and Tier 3/4 use `"rule_kind"`; `openrag-server` should accept serde alias `rule_kind` / `kind`.
