# OpenRAG 사용 가이드

OpenRAG를 처음 설치하는 분을 위한 안내예요. 위에서부터 순서대로 따라 하면 Claude Code(또는 Codex)에서
사내 문서를 묻고, 출처가 붙은 답을 받을 수 있어요.

- [1. OpenRAG가 하는 일](#1-openrag가-하는-일)
- [2. 준비물](#2-준비물)
- [3. 서버 띄우기](#3-서버-띄우기)
- [4. 첫 설정: 조직·관리자·문서를 한 번에](#4-첫-설정-조직관리자문서를-한-번에)
- [5. Claude Code / Codex 연결](#5-claude-code--codex-연결)
- [6. 질문하기](#6-질문하기)
- [7. 문서 더 넣기](#7-문서-더-넣기)
- [8. 팀원 추가와 권한](#8-팀원-추가와-권한)
- [9. 개인정보는 이렇게 다뤄요](#9-개인정보는-이렇게-다뤄요)
- [10. 스캔본 읽기(OCR)](#10-스캔본-읽기ocr)
- [11. 검색 품질: 임베딩 서버](#11-검색-품질-임베딩-서버)
- [12. 서버에서 직접 답하기(선택)](#12-서버에서-직접-답하기선택)
- [13. 자주 막히는 곳](#13-자주-막히는-곳)
- [14. 멈추기와 다시 켜기](#14-멈추기와-다시-켜기)

---

## 1. OpenRAG가 하는 일

문서를 OpenRAG 서버에 넣어 두면, Claude Code나 Codex가 질문을 받았을 때 그 문서에서 근거 문단을
찾아 답을 써요. OpenRAG는 AI 모델을 들고 있지 않아요. 답은 여러분이 쓰는 Claude Code나 Codex가
쓰고, OpenRAG는 "누가 어떤 문서를 볼 수 있는지"를 지키면서 찾아 주는 역할이에요.

```text
문서 폴더 ──(init · ingest)──▶ OpenRAG 서버 (PostgreSQL 안에 문서와 검색 색인)
                                     ▲
Claude Code / Codex ──(MCP: rag_search)──┘   찾은 문단을 근거로 답을 씀
```

- 문서와 검색 색인은 회사 서버(PostgreSQL)에 있어요.
- 사람마다 볼 수 있는 문서가 달라요. 권한이 없는 문서는 검색 결과에도 나오지 않아요.
- AI 쪽으로 나가는 문단에서는 전화번호·주민등록번호 같은 개인정보를 가려서 보내요([9장](#9-개인정보는-이렇게-다뤄요)).

## 2. 준비물

- **Docker**: Docker Desktop, 또는 Docker Engine과 Compose v2
- **질문할 PC의 Claude Code 또는 Codex**
- **넣을 문서 폴더**: HWP, HWPX, HML, PDF, MD, TXT, PNG, JPG, WEBP를 읽어요. 하위 폴더까지 다 읽고,
  이름이 `.`으로 시작하는 파일은 건너뛰어요.
- (선택) **Schift OCR 키**: 사진·스캔 이미지(PNG, JPG, WEBP)에서 글자를 읽을 때 필요해요([10장](#10-스캔본-읽기ocr)).

호스트에 따로 설치할 프로그램은 없어요. 설정에 쓰는 `openrag` 명령은 서버 이미지 안에 들어 있어요.

## 3. 서버 띄우기

저장소를 받은 폴더에서 실행해요.

```bash
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)
docker compose -f docker/docker-compose.yml --profile app up -d postgres openrag-api
```

`OPENRAG_PII_KEK_HEX`는 서버가 쓰는 암호화 키예요. 서버를 다시 띄울 때도 **같은 값**을 넣어야 하니,
처음 만든 값을 안전한 곳에 적어 두세요. 처음에는 이미지를 빌드하느라 몇 분 걸려요.

잘 떴는지 확인해요.

```bash
curl http://127.0.0.1:8080/health
# {"db_connected":true,"mode":"auto","status":"ok"}
```

포트 5432나 8080을 이미 다른 프로그램이 쓰고 있으면 바꿔서 띄우면 돼요.

```bash
PG_PORT=55432 OPENRAG_API_PORT=18080 docker compose -f docker/docker-compose.yml --profile app up -d postgres openrag-api
```

### 인터넷이 막힌 곳(폐쇄망)

임베딩 모델을 함께 띄우고, 바깥으로 나가는 연결을 설정한 주소로만 제한해요.

```bash
# 인터넷이 되는 PC에서, docker/ 폴더 안에: 임베딩 모델 파일을 ./models 에 받기
hf download google/embeddinggemma-2 --cache-dir ./models
# 폐쇄망 서버에서, docker/ 폴더 안에
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)
export AIRGAP_LLM_URL=http://<vllm-host>:8000/v1 AIRGAP_LLM_MODEL=<서빙 중인 모델>   # 선택
docker compose -f docker-compose.yml -f docker-compose.airgap.yml --profile app --profile airgap \
  up -d postgres local-embed openrag-api
```

이 구성으로 띄웠다면 아래 4·7장 명령의 `-f docker/docker-compose.yml` 자리에도 같은 `-f` 두 개를
넣어 주세요.

## 4. 첫 설정: 조직·관리자·문서를 한 번에

문서 폴더를 서버 컨테이너로 복사하고, 컨테이너 안에서 `openrag init`을 실행해요.

```bash
docker compose -f docker/docker-compose.yml cp ./my-docs openrag-api:/tmp/my-docs
docker compose -f docker/docker-compose.yml exec openrag-api \
  openrag init --slug acme --name "ACME" --docs /tmp/my-docs
```

`init`이 하는 일은 다섯 가지예요.

1. 조직(`acme`), 첫 부서, 관리자 계정(권한 등급 10)을 만들어요.
2. 관리자용 API 키(`orag_…`)를 발급해요.
3. 문서를 담을 버킷 `docs`를 만들어요.
4. 폴더 안의 문서를 전부 넣어요.
5. PC에서 실행할 연결 명령을 출력해요.

출력은 이렇게 생겼어요(일부 줄임).

```json
{
  "api_key": "orag_40b1…",
  "bucket": { "slug": "docs", "created": true },
  "documents": { "ingested": 3, "failed": 0 },
  "connect": {
    "claude": "claude mcp add --transport http --scope user openrag http://127.0.0.1:8080/mcp --header 'Authorization: Bearer orag_40b1…'",
    "codex": "export OPENRAG_API_KEY='orag_40b1…' && codex mcp add openrag --url http://127.0.0.1:8080/mcp --bearer-token-env-var OPENRAG_API_KEY"
  }
}
```

- **`api_key`는 관리자 키**예요. 문서를 더 넣거나 팀원을 추가할 때 다시 써요. 비밀번호처럼 보관하세요.
- 키를 잃어버리면 `init`을 한 번 더 실행하면 돼요. 조직과 버킷은 그대로 두고 새 키만 발급해요.
- `config.not_saved`라는 줄이 보이면 정상이에요. 컨테이너 안에는 설정 파일을 저장할 곳이 없어서 그래요.

자주 바꾸는 옵션은 이래요.

| 옵션 | 뜻 | 기본값 |
| --- | --- | --- |
| `--slug` | 조직 이름(영문 소문자, 숫자, `-`, `_`) | `default` |
| `--name` | 조직 표시 이름 | `Default` |
| `--bucket` | 문서를 담을 버킷 | `docs` |
| `--public-url` | PC에서 서버에 접속할 주소. 서버와 PC가 다른 기계면 꼭 넣으세요 | `http://127.0.0.1:8080` |

예를 들어 서버 주소가 `10.0.0.5`라면 `--public-url http://10.0.0.5:8080`을 붙여요. 그러면 출력되는
연결 명령에 그 주소가 들어가요.

## 5. Claude Code / Codex 연결

`init` 출력의 `connect.claude` 줄을 **Claude Code를 쓰는 PC에서** 그대로 실행해요.

```bash
claude mcp add --transport http --scope user openrag http://127.0.0.1:8080/mcp \
  --header 'Authorization: Bearer orag_…'
claude mcp list
# openrag: http://127.0.0.1:8080/mcp (HTTP) - ✔ Connected
```

`--scope user`는 이 PC의 모든 폴더에서 OpenRAG를 쓴다는 뜻이에요. 한 프로젝트 폴더에서만 쓰려면
`--scope project`로 바꾸세요. 그 폴더에 `.mcp.json`이 생기고, 그 폴더에서 `claude`를 처음 실행할 때
사용 승인을 한 번 물어봐요.

Codex는 `connect.codex` 줄을 실행해요. Codex는 키를 `OPENRAG_API_KEY` 환경변수에서 읽어요. 그래서
Codex를 실행하는 셸에 `export OPENRAG_API_KEY=orag_…`가 항상 있어야 해요. 셸 시작 파일에 넣어 두면 편해요.

## 6. 질문하기

Claude Code에 평소처럼 물어보면 돼요. "사내 문서에서"라고 붙이면 OpenRAG를 먼저 찾아봐요.

```text
> 사내 문서에서 찾아서 답해줘: 부산 출장 숙박비 한도가 얼마야? 출처도.

부산 출장 숙박비 한도는 1박 120,000원이에요.
출처: 출장비_규정.md「국내 출장비 규정 (2026년 개정)」
  원문: "숙박비 한도: 1박 100,000원. 서울·부산은 1박 120,000원"
```

위 예시는 실제로 받은 답을 줄인 거예요. 답이 나오기까지 20초쯤 걸렸어요. Claude Code는 `rag_search`로 문단을
찾은 뒤 그 문단을 근거로 답을 써요. 검색 결과에는 질문한 사람이 볼 수 있는 문서만 들어 있어요.

## 7. 문서 더 넣기

`init` 때 받은 관리자 키로 넣어요. 키는 `-e OPENRAG_API_KEY`로 컨테이너에 넘겨요.

```bash
export OPENRAG_API_KEY=orag_…            # init이 준 관리자 키
docker compose -f docker/docker-compose.yml cp ./new-docs openrag-api:/tmp/new-docs
docker compose -f docker/docker-compose.yml exec -e OPENRAG_API_KEY openrag-api \
  openrag ingest /tmp/new-docs --bucket docs
```

파일 하나만 넣어도 되고 폴더째 넣어도 돼요. 볼 수 있는 사람을 좁히려면 옵션을 붙여요([8장](#8-팀원-추가와-권한)).

```bash
# 등급 5 이상만 볼 수 있는 문서
... openrag ingest /tmp/secret --bucket docs --auth-level 5
# 올린 사람과 같은 부서 관리자만 볼 수 있는 문서
... openrag ingest /tmp/memo --bucket docs --scope private
```

> **같은 파일을 다시 넣으면 문서가 하나 더 생겨요.** 지금은 같은 파일인지 확인하지 않아요.
> 새로 생기거나 바뀐 파일만 넣으세요.

텍스트 문서라면 Claude Code에 "이 내용을 사내 문서 `docs`에 넣어줘"라고 해도 돼요(`rag_ingest` 도구).
이 방법은 글자만 넣을 수 있어서, PDF나 HWP 같은 파일은 위 명령으로 넣어야 해요.

## 8. 팀원 추가와 권한

### 권한은 두 가지로 정해요

- **등급(1~10)**: 문서마다 최소 등급이 있어요. 사람의 등급이 그보다 낮으면 그 문서는 보이지 않아요.
  `init`으로 만든 관리자는 10, 문서는 따로 정하지 않으면 1이에요.
- **공개 범위**: 문서를 누구에게 보여 줄지 정해요.

| 공개 범위 | 볼 수 있는 사람 |
| --- | --- |
| `org` | 조직 전체 (`init`이 넣은 문서의 기본값) |
| `dept` | 같은 부서 |
| `group` | 지정한 권한 그룹 |
| `private` | 올린 사람, 그리고 같은 부서의 부서 관리자 |

두 조건을 모두 통과해야 보여요. 예를 들어 실제로 등급 3인 팀원은 `--auth-level 5`로 넣은 문서를
검색해도 찾지 못했고, 관리자는 찾았어요.

### 팀원 추가

지금은 HTTP API로 추가해요. 관리자 키가 필요해요.

```bash
export OPENRAG_API_KEY=orag_…   # 관리자 키
# 1) 팀원 계정 만들기: 등급 3, init이 만든 부서(dept_id는 init 출력의 org.dept_id)
curl -s -X POST http://127.0.0.1:8080/api/studio/iam/principals \
  -H "Authorization: Bearer $OPENRAG_API_KEY" -H 'content-type: application/json' \
  -d '{"email":"kim@acme.example","display_name":"김사원","auth_level":3,"dept_ids":["<dept_id>"]}'
# 응답의 "id"가 팀원 ID예요

# 2) 팀원용 키 발급
curl -s -X POST http://127.0.0.1:8080/api/studio/iam/api-keys \
  -H "Authorization: Bearer $OPENRAG_API_KEY" -H 'content-type: application/json' \
  -d '{"principal_id":"<팀원 ID>","name":"kim-laptop"}'
# 응답의 "api_key"를 팀원에게 전달해요
```

팀원은 받은 키로 5장의 `claude mcp add …`를 실행하면 돼요. 그러면 팀원의 Claude Code는 팀원이 볼 수
있는 문서만 찾아요.

## 9. 개인정보는 이렇게 다뤄요

원칙은 하나예요. **개인정보는 AI로 나갈 때만 가려요.**

- 저장된 문서, 검색, HTTP API에는 원문이 그대로 있어요. 볼 권한이 있는 사람은 원문을 봐요.
- AI로 나가는 길에서는 `[PII:PHONE:70892581]` 같은 토큰으로 바꿔요. 서버가 AI 모델에 보내는 요청과
  MCP 도구 결과가 여기에 해당해요.
- 그래서 Claude Code 답에는 전화번호 같은 값이 가려져 나와요. 원문이 필요하면 문서를 직접 보거나
  HTTP API를 쓰세요.
- 질문에 직접 쓴 이름이나 번호는 가리지 않아요. 본인이 쓴 내용이니까요.

실제 문서로 확인한 결과는 이래요.

| 항목 | AI로 나갈 때 |
| --- | --- |
| 주민등록번호, 전화번호, 이메일, 카드번호, 여권번호 | 가려요 |
| 사람 이름 | `성명: 홍길동`처럼 라벨 뒤에 있거나, `김민준과장`·`박서연 대리`처럼 직함이 붙으면 가려요. 문장 속에 이름만 있으면 못 잡아요 |
| 계좌번호 | `계좌번호 123456-01-234567` 형태가 가려지지 않았어요. 계좌번호가 많은 문서는 주의하세요 |

`PROJECT-BLACKBOX`처럼 정해 둔 금지어가 들어간 질문·문서는 아예 막아요. 규칙은
`configs/pipeline.default.yaml`의 `pii.guard`에서 바꿀 수 있어요.

## 10. 스캔본 읽기(OCR)

HWP·HWPX·MD·TXT, 그리고 글자가 들어 있는 PDF는 서버 안에서 바로 읽어요. 사진이나 스캔 이미지
(PNG, JPG, WEBP)는 OCR이 있어야 글자를 읽을 수 있어요.

> **스캔한 PDF는 아직 OCR을 거치지 않아요.** 글자 층이 없는 PDF는 오류 없이 들어가지만, 읽은 글자가
> 없어서 검색에 걸리지 않아요. 스캔본은 페이지를 PNG나 JPG로 바꿔서 넣으세요.

OCR을 쓰려면 서버를 띄우기 전에 두 값을 넣으세요.

```bash
export OPENRAG_SCHIFT_OCR_URL=https://schift.io
export SCHIFT_OCR_API_KEY=<Schift OCR 키>
docker compose -f docker/docker-compose.yml --profile app up -d postgres openrag-api
```

OCR 모드는 문서 등급에 따라 자동으로 정해져요.

| 문서 등급 | 모드 | 뜻 |
| --- | --- | --- |
| 1~4 | `share` | 기본값. Schift가 페이지를 학습에 쓸 수 있어요 |
| 5~7 | `private` | 학습에 쓰지 않아요. 결과는 Schift에 남아요 |
| 8~10 | `zdr` | Schift에 아무것도 남기지 않아요 |

학습에 쓰이면 안 되는 문서는 `--auth-level 5` 이상으로 넣으세요. 기준은
`configs/pipeline.default.yaml`의 `ocr_mode_by_auth_level`에서 바꿀 수 있어요.

## 11. 검색 품질: 임베딩 서버

검색은 임베딩 모델로 뜻이 비슷한 문단을 찾아요. **기본 구성(3장)에는 임베딩 서버가 없어요.** 그대로
쓰면 키워드가 겹치는 문단만 찾아요(BM25). 서버 로그에 아래 줄이 보이면 이 상태예요.

```text
OPENRAG_TEXT_EMBED_URL is unset: … falls back to rrf with BM25
```

뜻으로 찾게 하려면 OpenAI 호환 임베딩 서버 주소와 모델을 넣고 서버를 다시 띄우세요.

```bash
export OPENRAG_TEXT_EMBED_URL=http://<embed-host>/v1/embeddings
export OPENRAG_TEXT_EMBED_MODEL=google/embeddinggemma-2
```

폐쇄망 구성(3장)은 `local-embed` 컨테이너가 이 모델을 같이 띄워요. 이 모델을 256차원으로 쓸 때
한국어 검색 평가(Ko-StrategyQA)에서 nDCG@10 0.781이 나왔어요.

## 12. 서버에서 직접 답하기(선택)

기본으로는 Claude Code나 Codex가 답을 써요. 서버가 직접 답하게 하려면(`/v1/chat/completions`,
MCP `rag_ask`) 사내 LLM 서버를 연결해요. 연결하지 않으면 이 두 경로는 503을 돌려줘요.

```bash
export OPENRAG_LOCAL_LLM_URL=http://<vllm-host>:8000/v1
export OPENRAG_LOCAL_LLM_MODEL=<서빙 중인 모델>
```

vLLM처럼 OpenAI 호환 API를 주는 서버면 돼요. 등급 8 이상 문서로 답할 때는 이 사내 서버만 써요.

## 13. 자주 막히는 곳

### `required variable OPENRAG_PII_KEK_HEX is missing a value`
3장의 `export OPENRAG_PII_KEK_HEX=…`를 빠뜨렸어요. 처음 만든 값을 그대로 다시 넣으세요.

### `port is already allocated` 또는 다른 DB에 붙음
5432·8080을 다른 프로그램이 쓰고 있어요. 3장처럼 `PG_PORT`, `OPENRAG_API_PORT`를 바꿔 띄우세요.

### `init`이 `no OpenRAG server answers`라고 함
서버가 아직 안 떴어요. `docker compose -f docker/docker-compose.yml logs openrag-api`로 로그를 보세요.

### `claude mcp list`에 `Pending approval`이 보임
`--scope project`로 등록했을 때 나와요. 그 폴더에서 `claude`를 실행해 승인하세요.

### Codex가 OpenRAG에 접속하지 못함
Codex를 실행한 셸에 `OPENRAG_API_KEY`가 없을 가능성이 커요. `echo $OPENRAG_API_KEY`로 확인하세요.

### 답에 `[PII:…]`가 보임
정상이에요. AI로 나가는 내용은 개인정보를 가려요([9장](#9-개인정보는-이렇게-다뤄요)).

### 검색이 엉뚱한 문서를 가져옴
임베딩 서버가 없는 상태일 수 있어요([11장](#11-검색-품질-임베딩-서버)).

### `/v1/chat/completions`가 503
서버 쪽 LLM을 연결하지 않았어요([12장](#12-서버에서-직접-답하기선택)). Claude Code로 물을 때는 필요 없어요.

## 14. 멈추기와 다시 켜기

```bash
docker compose -f docker/docker-compose.yml --profile app stop    # 멈추기
docker compose -f docker/docker-compose.yml --profile app start   # 다시 켜기
```

문서와 계정은 Docker 볼륨(`pg_data`)에 남아 있어서, 멈췄다 켜도 그대로예요.
