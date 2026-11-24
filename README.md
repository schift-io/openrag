# OpenRAG

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![PostgreSQL 17/18](https://img.shields.io/badge/PostgreSQL-17%20%7C%2018-336791.svg)](migrations/)
[![Rust Workspace](https://img.shields.io/badge/Rust-2021%20(1.85%2B)-orange.svg)](Cargo.toml)

**OpenRAG**는 사내 문서를 Claude Code·Codex 같은 AI 도구가 근거로 쓸 수 있게 해 주는 RAG 서버예요.
Rust와 PostgreSQL 17/18로 만들었고, 누가 어떤 문서를 볼 수 있는지와 개인정보가 어디까지 나가는지를
서버가 지켜요.

**[사용 가이드](docs/guide.md)** · [English](README.en.md)

---

## 이런 걸 해요

- **AI 도구에 사내 문서 검색을 붙여요.** Claude Code나 Codex에 MCP로 연결하면, 질문을 받을 때
  OpenRAG에서 근거 문단을 찾아 출처와 함께 답해요.
- **사람마다 볼 수 있는 문서가 달라요.** 문서마다 최소 등급(1~10)과 공개 범위(조직·부서·그룹·개인)를
  정해요. PostgreSQL의 행 단위 보안(RLS)이 막기 때문에, 권한 없는 문서는 검색 결과에도 나오지 않아요.
- **개인정보는 AI로 나갈 때만 가려요.** 저장과 검색은 원문 그대로 하고, AI 모델로 가는 요청과 MCP 도구
  결과, 회사 밖 임베딩 서버로 보내는 문서에서만 주민등록번호·전화번호·이메일 같은 값을 토큰으로 바꿔요.
- **문서 폴더를 지켜봐요.** 폴더에 파일을 넣거나 고치거나 지우면 30초 안에 검색에 반영돼요. 옮기거나 이름을
  바꾼 파일은 내용 해시로 알아봐서 다시 읽지 않고, 지운 파일은 검색에서만 빼고 기록은 남겨요. 엑셀·CSV는
  안내 카드로 찾고 칸 값은 AI가 필요한 행만 읽거나, 찾는 값이 든 행만 골라 가요.
- **한국어 문서에 맞췄어요.** HWP·HWPX·HML을 표까지 살려 읽고, DOCX·PPTX·HTML도 읽어요. 스캔한 PDF와 이미지는 Schift OCR로 읽어요. 키워드
  검색은 한국어 형태소 분석(`pg_search` Lindera)을 써요.
- **AI 모델을 들고 있지 않아요.** 답은 연결한 Claude Code·Codex가 쓰거나, 사내 LLM 서버(vLLM 등)를
  연결하면 서버가 직접 써요.
- **폐쇄망에서도 돌아가요.** 임베딩 모델을 함께 띄우고, 바깥 연결은 설정한 주소로만 허용해요.

## 빠르게 시작하기

준비물은 Docker 하나예요. 호스트에 따로 설치할 프로그램은 없어요.

Claude Code나 Codex가 있다면 저장소 폴더에서 `/openrag-setup`(Codex는 `$openrag-setup`)을 입력하세요.
설치 마법사가 아래 과정을 대신 진행하고, 끝나면 연결과 검색까지 확인해 줘요.

```bash
# 1) 서버 띄우기
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)   # 다시 띄울 때도 같은 값을 쓰니 적어 두세요
export OPENRAG_DOCS_DIR=~/회사문서                    # 문서 폴더(생략하면 저장소의 documents/)
docker compose -f docker/docker-compose.yml --profile app up -d

# 2) 첫 설정 (조직·관리자·API 키·버킷을 한 번에). 문서 폴더는 서버가 30초마다 읽어요
docker compose -f docker/docker-compose.yml exec openrag-api \
  openrag init --slug acme --name "ACME"
```

`init`은 마지막에 PC에서 실행할 연결 명령(`connect.claude`, `connect.codex`)을 출력해요.
Claude Code를 쓰는 PC에서 그 줄을 실행하면 끝이에요.

```bash
claude mcp add --transport http --scope user openrag http://127.0.0.1:8080/mcp \
  --header 'Authorization: Bearer orag_…'
```

이제 Claude Code에 "사내 문서에서 찾아서 답해줘: 부산 출장 숙박비 한도가 얼마야?"처럼 물어보면 돼요.

Claude Code는 이 명령 대신 플러그인으로 붙여도 돼요(`/plugin marketplace add schift-io/openrag` →
`/plugin install openrag@openrag`). 플러그인에는 MCP 연결, 설치 마법사, kordoc으로 만든 문서를 바로 OpenRAG에
저장하는 기능이 들어 있어요. Codex 플러그인과 Docufinder 같이 쓰기는 [가이드 15장](docs/guide.md#15-다른-도구에-붙이기-플러그인-kordoc-docufinder)에 있어요.

폐쇄망 구성, 문서 추가, 팀원 추가, 권한, 개인정보 처리, OCR, 문제 해결은
**[사용 가이드](docs/guide.md)**에 순서대로 정리했어요.

> 처음 띄울 때 임베딩 모델(`google/embeddinggemma-2`, 약 1.5GB)을 Hugging Face에서 받아요. 다 받을
> 때까지 `openrag-api`는 기다렸다가 떠요. 진행 상황은
> `docker compose -f docker/docker-compose.yml logs -f local-embed`로 볼 수 있어요.
> 다른 임베딩 서버를 쓰려면 가이드 11장을 보세요.

---

## 구성

```text
Claude Code · Codex · Cursor ──▶ /mcp (MCP over HTTP)
HTTP 클라이언트 ───────────────▶ /v2/buckets · /v1/chat/completions · /api/app · /api/studio
서버 호스트 ───────────────────▶ openrag mcp --stdio
                                      │
   ┌──────────────────────────────────▼──────────────────────────────────┐
   │ OpenRAG API (Rust, 프로세스 하나)                                     │
   │  • 권한: ARN + 권한 그룹 + 등급(1~10), PostgreSQL RLS                │
   │  • 문서 읽기: kordoc(HWP 계열), PDF, DOCX·PPTX·HTML, Schift OCR      │
   │  • 문서 폴더: 30초마다 훑어 추가·수정·이동·삭제를 반영(SHA-256)      │
   │  • 개인정보 가드: allow · warn · mask · block                        │
   │  • 색인: 바뀐 내용만 다시 색인(SHA-256), 실패 3회면 격리             │
   │  • 검색: 256차원 벡터 검색(pg_pinecone), BM25는 선택                 │
   │  • LLM 라우터: 등급 8 이상은 사내 LLM만                              │
   │  • 메모리 원장 · DB 위키 · 평가(골든셋, AutoRAG)                     │
   └──────────────────────────────────┬──────────────────────────────────┘
                                      ▼
               PostgreSQL 17/18 (iam, corpus, vectors, pii, memory, wiki,
                                 indexing, gateway, plugins, eval)
```

- **벡터 엔진**은 표마다 바꿀 수 있어요: `native`(pg_pinecone RaBitQ IVF, 외부 계정 불필요),
  `pgvector`(HNSW), `remote`(Pinecone Serverless). Amazon S3 Vectors나 OSS 벡터 저장소로 복제할 수도 있어요.
  `native` 인덱스는 문서가 20% 늘 때마다 데이터베이스가 알아서 다시 만들어요(가이드 11장).
- **임베딩**은 `google/embeddinggemma-2`를 256차원으로 쓰도록 맞춰 뒀어요(기본 구성과 폐쇄망 구성
  모두 이 모델을 같이 띄워요). 한국어 검색 평가(Ko-StrategyQA)에서 nDCG@10 0.781이 나왔고, BM25를 섞으면 0.703으로 떨어져서
  기본으로는 섞지 않아요.
- **메모리와 위키**: 대화에서 얻은 사실을 고칠 수 없는 패치(`amend` · `supersede` · `retract` · `redact`)로
  쌓고, DB 안의 위키는 `.md`/Obsidian과 양방향으로 동기화해요.
- **페이지 썸네일**은 기본으로 꺼져 있어요(`thumbnail.enabled: false`). 켜면 S3 호환 저장소
  (`OPENRAG_S3_*`)에 저장해요.

## MCP 도구 (16개)

| 도구 | 하는 일 |
| --- | --- |
| `rag_search` | 내가 볼 수 있는 문서에서 찾기. 기본으로 지금 효력이 있는 문서만 찾아요 |
| `rag_ask` | 문서를 근거로 서버가 직접 답하기(서버에 LLM을 연결했을 때만) |
| `rag_ingest` | 텍스트를 버킷에 넣기. 파일 경로는 stdio 연결에서만 돼요 |
| `rag_get_page` | 문서 한 페이지 읽기(가린 본문, 그림 위치, 썸네일 링크) |
| `rag_list_documents` | 버킷에 들어 있는 문서 목록(원본 경로, 판, 효력) |
| `rag_read_sheet` | 엑셀·CSV의 행 읽기와 값 찾기. 검색은 안내 카드로 파일을 찾고, 칸 값은 이걸로 읽거나 `find`로 그 값이 든 행만 골라요 |
| `docs_status` | 문서 폴더 상태(실패한 파일과 이유, 진행 상황), 지금 바로 훑기(관리자) |
| `wiki_search` / `wiki_get` | 위키 페이지 찾기 / 읽기 |
| `memory_search` | 지금 유효한 메모리 찾기(철회·대체된 항목 제외) |
| `memory_patch` | 메모리에 패치 추가: 수정, 대체, 철회, 가림 |
| `memory_engage` | 개념을 넓혀 가며 메모리 떠올리기 |
| `memory_reflect` | 대화 한 토막과 거기서 얻은 메모리 기록하기 |
| `guard_check` | 글이 개인정보·금지어 규칙에 걸리는지 보기 |
| `concept_link` / `concept_resolve` | 같은 뜻의 용어(한글·영어 등) 묶기 / 찾기 |

모든 도구 호출은 호출한 사람의 권한으로 돌고, 결과의 개인정보는 가려져요.

Claude Code·Codex 외의 MCP 클라이언트도 HTTP 주소와 키만 넣으면 돼요. Cursor(`.cursor/mcp.json`)는 이렇게 써요.

```json
{
  "mcpServers": {
    "openrag": {
      "url": "http://localhost:8080/mcp",
      "headers": { "Authorization": "Bearer ${OPENRAG_API_KEY}" }
    }
  }
}
```

stdio 방식(`openrag mcp --stdio`, `openrag-mcp`)은 데이터베이스에 직접 붙기 때문에 서버 호스트에서
`DATABASE_URL`을 넣고 실행할 때만 써요.

## `openrag` 명령

서버 이미지 안에 들어 있어요. `docker compose -f docker/docker-compose.yml exec openrag-api openrag <명령>`으로
실행해요. 소스에서 직접 빌드(`cargo build --release -p openrag-server --bin openrag`)해 호스트에서 써도 돼요.

| 명령 | 하는 일 |
| --- | --- |
| `openrag init` | 첫 설정 한 번에: 조직·관리자·API 키·버킷, 연결 명령 출력(`--docs`로 폴더를 한 번 넣을 수도 있어요) |
| `openrag org init` | 조직·부서·관리자와 API 키만 만들기 |
| `openrag bucket {create\|list}` | 버킷 만들기 / 보기 |
| `openrag docs {list\|status\|scan}` | 문서 목록, 문서 폴더 상태(실패한 파일과 이유), 지금 바로 훑기 |
| `openrag member {add\|list}` | 팀원 추가(계정·키·연결 명령 한 번에) / 목록 |
| `openrag key {list\|revoke}` | API 키 목록 / 바로 막기 |
| `openrag ingest <파일 또는 폴더> --bucket <버킷>` | 한 번만 넣기. `--scope`, `--auth-level`로 볼 사람을 정해요. 같은 폴더를 다시 넣으면 바뀐 파일만 들어가요. 계속 맞춰 두려면 문서 폴더(`OPENRAG_DOCS_DIR`)를 쓰세요 |
| `openrag connect {claude\|codex}` | 이 서버를 Claude Code나 Codex에 등록(호스트에서 실행할 때) |
| `openrag up --profile <standard\|airgap>` | docker compose로 스택 띄우기 |
| `openrag index {engine\|mirror\|status\|retry}` | 벡터 엔진 바꾸기, 외부 복제, 색인 대기열 보기 |
| `openrag memory {patch\|promote}` | 메모리 패치, 공개 범위 올리기(개인 → 부서 → 위키) |
| `openrag redact run` | 개인정보를 모든 저장 위치에서 지우기 |
| `openrag wiki {import\|export}` | `.md` / Obsidian 동기화 |
| `openrag bench {run\|optimize}`, `openrag pipeline promote` | 평가와 AutoRAG 최적화, 설정 승격 |
| `openrag mcp --stdio` | 서버 호스트에서 stdio MCP 서버 실행(`DATABASE_URL` 필요) |

## HTTP API

- **기본 API**: `/v2/buckets`(버킷·문서·검색), `/v1/chat/completions`(OpenAI 호환), `/v1/eval/ab-search`,
  `/mcp`(Streamable HTTP, JSON-RPC 2.0)
- **앱 API (`/api/app`)**: 채팅, 업로드, 페이지 썸네일(썸네일이 꺼져 있으면 404), 위키, 메모리, 개념, 피드백
- **관리 API (`/api/studio`, 조직 관리자 전용)**: 파이프라인 설정, 색인, 계정·권한(IAM), 개인정보 가드,
  LLM 게이트웨이, 평가, 위키, 플러그인

OpenRAG에는 화면(GUI)이 없어요. 사람은 MCP 클라이언트로 쓰고, 관리는 HTTP API로 해요.

## LLM

OpenRAG는 LLM을 내장하지 않아요. 기본으로는 연결한 Claude Code·Codex·Cursor가 `rag_search`로 찾은
문단을 보고 답을 써요. 서버가 직접 답하는 `/v1/chat/completions`와 `rag_ask`는 아래 중 하나를 설정했을
때만 동작해요(설정이 없으면 503).

- `OPENRAG_LOCAL_LLM_URL` + `OPENRAG_LOCAL_LLM_MODEL`: 사내 OpenAI 호환 서버(vLLM 등). 회사 밖으로
  나가면 안 되는 요청(폐쇄망 구성의 모든 요청, 기본 구성에서는 등급 8 이상)은 이 서버만 써요.
- `OPENRAG_LLM_CLI=claude` 또는 `codex`: 서버 호스트의 Claude Code(`claude -p`)나 Codex(`codex exec`)
  CLI로 답해요. 내용이 Anthropic·OpenAI로 나가기 때문에 외부 LLM으로 취급해요(개인정보는 가리고,
  등급 7까지). 서버는 전용 계정으로 실행하세요. CLI가 그 계정의 지침 파일(`~/.claude/CLAUDE.md`,
  `~/.codex/AGENTS.md`)과 설정을 읽어서, 모든 사용자의 답에 영향을 줘요.

폐쇄망 구성에서는 설정한 주소(`OPENRAG_LOCAL_LLM_URL`, `OPENRAG_TEXT_EMBED_URL`, `OPENRAG_SCHIFT_OCR_URL`,
`OPENRAG_S3_ENDPOINT`)와 `configs/pipeline.airgap.yaml`의 `egress.allow`에만 연결해요. 다른 곳을 가리키면
403으로 막고, CLI 방식은 꺼져요.

## 설정 파일 (`configs/`)

- `pipeline.default.yaml`: 기본 구성. 256차원 임베딩, `pg_pinecone` native, kordoc, 등급별 Schift OCR 모드,
  개인정보 가드 규칙, 등급 8 이상 사내 LLM 전용
- `pipeline.airgap.yaml`: 폐쇄망 구성. 로컬 임베딩·사내 LLM, 바깥 연결 허용 목록, 엄격한 가드
- `search_space.example.yaml`: AutoRAG 탐색 공간. 개인정보 유출 0건, 권한 위반 0건을 통과해야만 채택해요

YAML은 `openrag-core`의 Rust 타입과 1:1로 맞고, 모르는 키가 있으면 설정을 읽을 때 오류가 나요.

## 개발자용: 빌드와 테스트

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

임시 PostgreSQL 17을 띄워 마이그레이션을 적용하고, `openrag_app` 계정의 RLS가 닫힌 상태로 동작하는지
확인한 뒤 정리해요.

```bash
./scripts/test-pg-local.sh
```

## 라이선스

[Apache License 2.0](LICENSE)
