# OpenRAG

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![PostgreSQL 17/18](https://img.shields.io/badge/PostgreSQL-17%20%7C%2018-336791.svg)](migrations/)
[![Rust Workspace](https://img.shields.io/badge/Rust-2021%20(1.85%2B)-orange.svg)](Cargo.toml)

**OpenRAG**는 Claude Code·Codex 같은 AI 도구가 사내 문서를 찾아 근거로 쓰게 해 주는 RAG(문서에서 근거를 먼저 찾아 답하게 하는 방식) 서버예요.
Rust와 PostgreSQL 17/18로 만들었어요.
누가 어떤 문서를 볼 수 있는지와 개인정보가 어디까지 나가는지를 서버가 지켜요.

**[사용 가이드](docs/guide.md)** · [English](README.en.md)

---

## 이런 걸 해요

- **AI 도구에 사내 문서 검색을 붙여요.** Claude Code나 Codex에 MCP(AI 도구에 외부 기능을 붙이는 표준 방식)로 연결하면 돼요. 질문을 받을 때 OpenRAG에서 근거 문단을 찾아 출처와 함께 답해요.
- **사람마다 볼 수 있는 문서가 달라요.** 문서마다 최소 등급(1~10)과 공개 범위(조직·부서·그룹·개인)를 정해요. PostgreSQL의 행 단위 보안(RLS, 데이터베이스가 저장된 행마다 권한을 검사해 막는 기능)이 지켜요. 그래서 권한 없는 문서는 검색 결과에도 나오지 않아요.
- **개인정보는 AI로 나갈 때만 가려요.** 저장과 검색은 원문 그대로 해요. 대신 AI 모델로 가는 요청과 MCP 도구 결과에서는 주민등록번호·전화번호·이메일 같은 값을 토큰으로 바꿔요. 회사 밖 임베딩(글의 뜻을 숫자로 바꿔 비슷한 내용을 찾게 만드는 처리) 서버로 문서를 보낼 때도 똑같이 가려요.
- **문서 폴더를 지켜봐요.** 폴더에 파일을 넣거나 고치거나 지우면 30초 안에 검색에 반영돼요. 옮기거나 이름을 바꾼 파일은 내용 해시로 알아봐서 다시 읽지 않아요. 지운 파일은 검색에서만 빼고 기록은 남겨요. 엑셀·CSV는 안내 카드로 찾고, 칸 값은 AI가 필요한 행만 읽거나 찾는 값이 든 행만 골라 가요.
- **한국어 문서에 맞췄어요.** HWP·HWPX·HML을 표까지 살려 읽고, DOCX·PPTX·HTML도 읽어요. PDF는 kordoc으로 읽어요. 스캔한 페이지는 서버 안에서 kordoc의 한국어 OCR(이미지나 스캔 문서 속 글자를 텍스트로 바꾸는 기술)로 읽어요. 사진 파일과 kordoc이 못 읽은 페이지만 Schift OCR로 보내요. 키워드 검색은 한국어 형태소 분석(`pg_search` Lindera)을 써요.
- **AI 모델을 들고 있지 않아요.** 답은 연결한 Claude Code·Codex가 써요. 사내 LLM 서버(vLLM 등)를 연결하면 서버가 직접 답을 쓰게 할 수도 있어요.
- **폐쇄망에서도 돌아가요.** 임베딩 모델을 함께 띄우고, 바깥 연결은 설정한 주소로만 허용해요.

## 빠르게 시작하기

Docker로 서버를 띄우고 Claude Code나 Codex에 연결해 첫 검색까지 해 보는 순서예요.

준비물은 Docker 하나예요. 호스트에 따로 설치할 프로그램은 없어요.

Claude Code나 Codex가 있다면 저장소 폴더에서 `/openrag-setup`(Codex는 `$openrag-setup`)을 입력하세요.
설치 마법사가 아래 과정을 대신 진행하고, 끝나면 연결과 검색까지 확인해 줘요.

직접 실행하려면 서버를 띄운 뒤 `openrag init`으로 조직·관리자·API 키와 버킷(문서를 묶어 담는 보관함)을 한 번에 만들어요.

```bash
# 1) 서버 띄우기
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)   # 다시 띄울 때도 같은 값을 쓰니 적어 두세요
export OPENRAG_DOCS_DIR=~/회사문서                    # 문서 폴더(생략하면 저장소의 documents/)
mkdir -p "${OPENRAG_DOCS_DIR:-documents}"            # 먼저 만들어 두세요. 없으면 Docker가 root 소유로 만들어요
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

Claude Code는 위 명령 대신 플러그인으로 붙여도 돼요(`/plugin marketplace add schift-io/openrag` → `/plugin install openrag@openrag`).
플러그인에는 MCP 연결, 설치 마법사, kordoc으로 만든 문서를 바로 OpenRAG에 저장하는 기능이 들어 있어요.
Codex 플러그인과 Docufinder 같이 쓰기는 [가이드 15장](docs/guide.md#15-다른-도구에-붙이기-플러그인-kordoc-docufinder)에 있어요.

폐쇄망 구성, 문서 추가, 팀원 추가, 권한, 개인정보 처리, OCR, 문제 해결은 **[사용 가이드](docs/guide.md)**에 순서대로 정리했어요.

> 처음 띄울 때 임베딩 모델(`google/embeddinggemma-2`, 약 1.5GB)을 Hugging Face에서 받아요.
> 다 받을 때까지 `openrag-api`는 기다렸다가 떠요.
> 진행 상황은 `docker compose -f docker/docker-compose.yml logs -f local-embed`로 볼 수 있어요.
> 다른 임베딩 서버를 쓰려면 가이드 11장을 보세요.

---

## AI 도구 연결과 MCP 도구 (16개)

Claude Code·Codex 외의 MCP 클라이언트도 HTTP 주소와 키만 넣으면 붙일 수 있어요.
Cursor(`.cursor/mcp.json`)는 이렇게 써요.

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

stdio(AI 도구가 같은 컴퓨터에서 프로그램을 직접 실행해 붙는 방식) 연결(`openrag mcp --stdio`, `openrag-mcp`)은 데이터베이스에 직접 붙어요.
그래서 서버 호스트에서 `DATABASE_URL`을 넣고 실행할 때만 써요.

연결한 AI 도구는 아래 16개 MCP 도구를 호출할 수 있어요.
모든 도구 호출은 호출한 사람의 권한으로 돌고, 결과의 개인정보는 가려져요.

| 도구 | 하는 일 |
| --- | --- |
| `rag_search` | 내가 볼 수 있는 문서에서 찾기. 기본으로 지금 효력이 있는 문서만 찾아요. 막연한 질문에는 AI가 예상 답(`hypothetical_answer`)을 같이 보내 비슷한 문단을 찾아요 |
| `rag_ask` | 문서를 근거로 서버가 직접 답하기(서버에 LLM을 연결했을 때만) |
| `rag_ingest` | 텍스트나 파일을 버킷에 넣기. 파일 경로는 stdio 연결에서만 받아요. HTTP 연결은 파일 올리기를 켰을 때 파일을 직접 받아요 |
| `rag_get_page` | 문서 한 페이지 읽기(가린 본문, 그림 위치, 썸네일 링크) |
| `rag_list_documents` | 버킷에 들어 있는 문서 목록(원본 경로, 판, 효력) |
| `rag_read_sheet` | 엑셀·CSV의 행 읽기와 값 찾기. 검색은 안내 카드로 파일을 찾아요. 칸 값은 이걸로 읽거나 `find`로 그 값이 든 행만 골라요 |
| `docs_status` | 문서 폴더 상태(실패한 파일과 이유, 진행 상황), 지금 바로 훑기(관리자) |
| `wiki_search` / `wiki_get` | 위키 페이지 찾기 / 읽기 |
| `memory_search` | 지금 유효한 메모리 찾기(철회·대체된 항목 제외) |
| `memory_patch` | 메모리에 패치 추가: 수정, 대체, 철회, 가림 |
| `memory_engage` | 개념을 넓혀 가며 메모리 떠올리기 |
| `memory_reflect` | 대화 한 토막과 거기서 얻은 메모리 기록하기 |
| `guard_check` | 글이 개인정보·금지어 규칙에 걸리는지 보기 |
| `concept_link` / `concept_resolve` | 같은 뜻의 용어(한글·영어 등) 묶기 / 찾기 |

MCP는 2026-07-28 규격과 이전 규격(2025-06-18, 2025-03-26, 2024-11-05)을 둘 다 받아요.
새 규격으로 붙는 클라이언트는 아래 기능도 써요.

- **스킬**: 검색 요령을 담은 `skill://openrag-search/SKILL.md`를 읽어 가요.
- **작업(Tasks)**: 폴더 훑기와 파일 적재처럼 오래 걸리는 호출은 작업으로 돌려요. 클라이언트가 끝날 때까지 상태를 물어봐요.
- **원격 OCR 전에 묻기**: 클라이언트가 elicitation(작업 중간에 사용자에게 직접 확인을 묻는 기능)을 지원하면, kordoc이 못 읽은 페이지를 회사 밖 Schift OCR로 보내기 전에 먼저 물어봐요. 거절하면 그 페이지는 글자 없이 넣고, 취소하면 넣지 않아요.
- **파일 올리기**: 서버에 `OPENRAG_MCP_FILE_UPLOADS=1`을 넣으면 `rag_ingest`가 파일을 받아요. 4MiB까지는 요청에 담아 보내고, 그보다 크면 64MiB까지 서버가 알려 주는 주소로 따로 올려요. 기본은 꺼져 있어요.

## `openrag` 명령

조직 설정, 팀원 관리, 문서 넣기 같은 운영 작업을 터미널에서 실행하는 명령 목록이에요.

`openrag` 명령은 서버 이미지 안에 들어 있어요.
`docker compose -f docker/docker-compose.yml exec openrag-api openrag <명령>`으로 실행해요.
소스에서 직접 빌드(`cargo build --release -p openrag-server --bin openrag`)해 호스트에서 써도 돼요.

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

OpenRAG에는 화면(GUI)이 없어요. 사람은 MCP 클라이언트로 쓰고, 관리는 아래 HTTP API로 해요.

- **기본 API**: `/v2/buckets`(버킷·문서·검색), `/v1/chat/completions`(OpenAI 호환), `/v1/eval/ab-search`, `/mcp`(Streamable HTTP, JSON-RPC 2.0)
- **앱 API (`/api/app`)**: 채팅, 업로드, 페이지 썸네일(썸네일이 꺼져 있으면 404), 위키, 메모리, 개념, 피드백
- **관리 API (`/api/studio`, 조직 관리자 전용)**: 파이프라인 설정, 색인, 계정·권한(IAM), 개인정보 가드, LLM 게이트웨이, 평가, 위키, 플러그인

## LLM

서버가 직접 답을 만들게 하고 싶을 때 사내 LLM이나 CLI를 연결하는 방법이에요.

기본으로는 연결한 Claude Code·Codex·Cursor가 `rag_search`로 찾은 문단을 보고 답을 써요.
서버가 직접 답하는 `/v1/chat/completions`와 `rag_ask`는 아래 중 하나를 설정했을 때만 동작해요(설정이 없으면 503).

- `OPENRAG_LOCAL_LLM_URL` + `OPENRAG_LOCAL_LLM_MODEL`: 사내 OpenAI 호환 서버(vLLM 등)를 연결해요. 회사 밖으로 나가면 안 되는 요청(폐쇄망 구성의 모든 요청, 기본 구성에서는 등급 8 이상)은 이 서버만 써요.
- `OPENRAG_LLM_CLI=claude` 또는 `codex`: 서버 호스트의 Claude Code(`claude -p`)나 Codex(`codex exec`) CLI로 답해요. 내용이 Anthropic·OpenAI로 나가기 때문에 외부 LLM으로 취급해요(개인정보는 가리고, 등급 7까지). 서버는 전용 계정으로 실행하세요. CLI가 그 계정의 지침 파일(`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`)과 설정을 읽어서, 모든 사용자의 답에 영향을 줘요.

폐쇄망 구성에서는 설정한 주소(`OPENRAG_LOCAL_LLM_URL`, `OPENRAG_TEXT_EMBED_URL`, `OPENRAG_SCHIFT_OCR_URL`, `OPENRAG_S3_ENDPOINT`)와 `configs/pipeline.airgap.yaml`의 `egress.allow`에만 연결해요.
다른 곳을 가리키면 403으로 막고, CLI 방식은 꺼져요.

## 설정 파일 (`configs/`)

검색 방식, 보안 규칙, 폐쇄망 동작을 바꾸는 YAML 설정 파일 목록이에요.

- `pipeline.default.yaml`: 기본 구성이에요. 256차원 임베딩, `pg_pinecone` native, kordoc(PDF 포함, 로컬 OCR), `structure` 청킹(긴 문서를 검색하기 좋은 조각으로 나누는 작업), 등급별 Schift OCR 모드, 개인정보 가드 규칙, 등급 8 이상 사내 LLM 전용 설정이 들어 있어요.
- `pipeline.airgap.yaml`: 폐쇄망 구성이에요. 로컬 임베딩·사내 LLM, 바깥 연결 허용 목록, 엄격한 가드를 묶어 뒀어요.
- `search_space.example.yaml`: AutoRAG 탐색 공간이에요. 개인정보 유출 0건, 권한 위반 0건을 통과해야만 채택해요.

YAML은 `openrag-core`의 Rust 타입과 1:1로 맞아요.
모르는 키가 있으면 설정을 읽을 때 오류가 나요.

## 내부 구성과 검색 품질

```text
Claude Code · Codex · Cursor ──▶ /mcp (MCP over HTTP)
HTTP 클라이언트 ───────────────▶ /v2/buckets · /v1/chat/completions · /api/app · /api/studio
서버 호스트 ───────────────────▶ openrag mcp --stdio
                                      │
   ┌──────────────────────────────────▼──────────────────────────────────┐
   │ OpenRAG API (Rust, 프로세스 하나)                                     │
   │  • 권한: ARN + 권한 그룹 + 등급(1~10), PostgreSQL RLS                │
   │  • 문서 읽기: kordoc(HWP 계열·PDF), DOCX·PPTX·HTML, Schift OCR       │
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

그림의 ARN은 문서·버킷 같은 자원을 조직·부서·공개 범위 순서로 가리키는 이름이에요.
BM25는 단어가 얼마나 자주 나오는지로 점수를 매기는 키워드 검색 방식이에요.

- **벡터 엔진**: 표마다 바꿀 수 있어요. 선택지는 `native`(pg_pinecone RaBitQ IVF, 외부 계정 불필요), `pgvector`(HNSW), `remote`(Pinecone Serverless)예요. Amazon S3 Vectors나 OSS 벡터 저장소로 복제할 수도 있어요. `native` 인덱스는 문서가 20% 늘 때마다 데이터베이스가 알아서 다시 만들어요(가이드 11장).
- **임베딩**: `google/embeddinggemma-2`를 256차원으로 쓰도록 맞춰 뒀어요(기본 구성과 폐쇄망 구성 모두 이 모델을 같이 띄워요). 한국어 검색 평가(Ko-StrategyQA)에서 nDCG@10 0.781이 나왔어요. BM25를 섞으면 0.703으로 떨어져서 기본으로는 섞지 않아요.
- **PDF와 청킹**: Allganize 한국어 RAG 평가(PDF 59개, 질문 257개)에서 정답 페이지가 상위 5위 안에 든 비율은 kordoc으로 읽을 때 0.848, pdftotext로 읽을 때 0.724였어요. 페이지는 제목·목록·표 경계에서 나눠요(`store.chunker: structure`). 고정 길이로 자를 때와 비교하면 정답 페이지를 1위에 올린 질문이 9개 늘고 2개 줄었어요.
- **메모리와 위키**: 대화에서 얻은 사실을 고칠 수 없는 패치(`amend` · `supersede` · `retract` · `redact`)로 쌓아요. DB 안의 위키는 `.md`/Obsidian과 양방향으로 동기화해요.
- **페이지 썸네일**: 기본으로 꺼져 있어요(`thumbnail.enabled: false`). 켜면 S3 호환 저장소(`OPENRAG_S3_*`)에 저장해요.

## 개발자용: 빌드와 테스트

소스를 직접 고친 뒤 코드 검사와 테스트를 돌리는 명령이에요.

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

아래 스크립트는 임시 PostgreSQL 17을 띄워 마이그레이션을 적용해요.
`openrag_app` 계정의 RLS가 닫힌 상태로 동작하는지 확인한 뒤 정리해요.

```bash
./scripts/test-pg-local.sh
```

## 라이선스

OpenRAG는 [Apache License 2.0](LICENSE)으로 배포해요.
