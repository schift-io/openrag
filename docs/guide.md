# OpenRAG 사용 가이드

OpenRAG를 처음 설치하는 분을 위한 안내예요. 위에서부터 순서대로 따라 하면 Claude Code(또는 Codex)에서
사내 문서를 묻고, 출처가 붙은 답을 받을 수 있어요.

- [1. OpenRAG가 하는 일](#1-openrag가-하는-일)
- [2. 준비물](#2-준비물)
- [3. 서버 띄우기](#3-서버-띄우기)
- [4. 첫 설정: 조직·관리자·키를 한 번에](#4-첫-설정-조직관리자키를-한-번에)
- [5. Claude Code / Codex 연결](#5-claude-code--codex-연결)
- [6. 질문하기](#6-질문하기)
- [7. 문서 넣기·고치기·지우기](#7-문서-넣기고치기지우기)
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
문서 폴더 ──(서버가 지켜보며 반영)──▶ OpenRAG 서버 (PostgreSQL 안에 문서와 검색 색인)
                                          ▲
Claude Code / Codex ──(MCP: rag_search)───────┘   찾은 문단을 근거로 답을 씀
```

- 문서와 검색 색인은 회사 서버(PostgreSQL)에 있어요.
- 사람마다 볼 수 있는 문서가 달라요. 권한이 없는 문서는 검색 결과에도 나오지 않아요.
- AI 쪽으로 나가는 문단에서는 전화번호·주민등록번호 같은 개인정보를 가려서 보내요([9장](#9-개인정보는-이렇게-다뤄요)).

## 2. 준비물

- **Docker**: Docker Desktop, 또는 Docker Engine과 Compose v2. 처음 띄울 때 이미지와 임베딩 모델(약 1.5GB)을
  받을 디스크 자리와, 임베딩 서버가 쓸 메모리 3GB쯤이 더 필요해요.
- **질문할 PC의 Claude Code 또는 Codex**
- **문서 폴더**: 서버가 이 폴더를 계속 지켜보면서 파일을 넣고, 고친 파일은 새 판으로 바꾸고, 지운 파일은
  검색에서 빼요([7장](#7-문서-넣기고치기지우기)). 기본으로 HWP, HWPX, MD, PDF, DOCX, PPTX, HTML을 읽고,
  하위 폴더까지 다 봐요.
- (선택) **Schift OCR 키**: 스캔한 PDF나 사진(PNG, JPG, WEBP)에서 글자를 읽을 때 필요해요([10장](#10-스캔본-읽기ocr)).

호스트에 따로 설치할 프로그램은 없어요. 설정에 쓰는 `openrag` 명령은 서버 이미지 안에 들어 있어요.

## 3. 서버 띄우기

저장소를 받은 폴더에서 실행해요.

```bash
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)
export OPENRAG_DOCS_DIR=~/회사문서      # 문서 폴더. 생략하면 저장소의 documents/ 폴더
docker compose -f docker/docker-compose.yml --profile app up -d
```

- `OPENRAG_PII_KEK_HEX`는 서버가 쓰는 암호화 키예요. 서버를 다시 띄울 때도 **같은 값**을 넣어야 하니,
  처음 만든 값을 안전한 곳에 적어 두세요.
- `OPENRAG_DOCS_DIR`은 서버가 지켜볼 문서 폴더예요. 서버는 이 폴더를 읽기만 하고, 파일을 바꾸거나
  지우지 않아요. 폴더를 여러 개 쓰는 방법은 [7장](#7-문서-넣기고치기지우기)에 있어요.

컨테이너는 세 개가 떠요: 데이터베이스(`postgres`), 임베딩 서버(`local-embed`), API(`openrag-api`).
처음 한 번은 오래 걸려요.

- 이미지 세 개를 빌드해요.
- `local-embed`가 임베딩 모델(약 1.5GB)을 Hugging Face에서 받아요. 다 받아서 준비될 때까지
  `openrag-api`는 기다렸다가 떠요. 진행 상황은 아래 명령으로 볼 수 있어요.
- 임베딩 서버는 메모리를 2.2GB쯤 써요.

시험해 본 PC(Docker Desktop, macOS)에서는 모델을 받아서 준비되기까지 2분 41초가 걸렸어요. 회선 속도에
따라 달라요.

```bash
docker compose -f docker/docker-compose.yml logs -f local-embed
```

잘 떴는지 확인해요.

```bash
curl http://127.0.0.1:8080/health
# {"db_connected":true,"mode":"auto","status":"ok"}
```

포트 5432나 8080을 이미 다른 프로그램이 쓰고 있으면 바꿔서 띄우면 돼요.

```bash
PG_PORT=55432 OPENRAG_API_PORT=18080 docker compose -f docker/docker-compose.yml --profile app up -d
```

### 인터넷이 막힌 곳(폐쇄망)

임베딩 모델을 함께 띄우고, 바깥으로 나가는 연결을 설정한 주소로만 제한해요.

```bash
# 인터넷이 되는 PC에서, docker/ 폴더 안에: 임베딩 모델 파일을 ./models 에 받기
hf download google/embeddinggemma-2 --cache-dir ./models
# 폐쇄망 서버에서, docker/ 폴더 안에
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)
export AIRGAP_LLM_URL=http://<vllm-host>:8000/v1 AIRGAP_LLM_MODEL=<서빙 중인 모델>   # 선택
docker compose -f docker-compose.yml -f docker-compose.airgap.yml --profile app \
  up -d postgres local-embed openrag-api
```

이 구성으로 띄웠다면 아래 4·7장 명령의 `-f docker/docker-compose.yml` 자리에도 같은 `-f` 두 개를
넣어 주세요.

## 4. 첫 설정: 조직·관리자·키를 한 번에

컨테이너 안에서 `openrag init`을 실행해요.

```bash
docker compose -f docker/docker-compose.yml exec openrag-api \
  openrag init --slug acme --name "ACME"
```

`init`이 하는 일은 네 가지예요.

1. 조직(`acme`), 첫 부서, 관리자 계정(권한 등급 10)을 만들어요.
2. 관리자용 API 키(`orag_…`)를 발급해요.
3. 문서를 담을 버킷 `docs`를 만들어요.
4. PC에서 실행할 연결 명령을 출력해요.

버킷이 생기면 서버가 문서 폴더를 읽기 시작해요. 폴더는 30초마다 훑으니 30초 안에 들어가기 시작해요.
들어간 결과는 로그에서 볼 수 있어요.

```bash
docker compose -f docker/docker-compose.yml logs openrag-api | grep "documents folder"
# … documents folder synced bucket=docs added=6 replaced=0 moved=0 removed=0 failed=0
```

출력은 이렇게 생겼어요(일부 줄임).

```json
{
  "api_key": "orag_40b1…",
  "bucket": { "slug": "docs", "created": true },
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
| `--public-url` | PC에서 서버에 접속할 주소. 서버와 PC가 다른 기계면 꼭 넣으세요 | `http://127.0.0.1:<API 포트>`(3장에서 `OPENRAG_API_PORT`를 바꿨다면 그 포트) |

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

## 7. 문서 넣기·고치기·지우기

문서 폴더(3장의 `OPENRAG_DOCS_DIR`)에서 파일을 다루기만 하면 돼요. 서버가 30초마다 폴더를 훑어서
바뀐 대로 따라가요. 명령을 따로 칠 필요가 없어요.

| 폴더에서 한 일 | OpenRAG에서 생기는 일 |
| --- | --- |
| 새 파일을 넣음 | 문서로 들어가요 |
| 파일을 고침 | 고친 내용이 새 판으로 들어가요. 예전 판은 그 시각에 효력이 끝나서 검색에 나오지 않아요 |
| 파일을 옮기거나 이름을 바꿈 | 같은 문서로 남아요. 다시 읽거나 임베딩하지 않고, 경로와 제목만 바뀌어요 |
| 파일을 지움 | 검색에서 빠져요. 문서 기록은 지우지 않고 남겨요 |
| 파일을 복사함 | 새 문서를 만들지 않아요. 내용이 같으니까요 |

같은 파일인지는 파일 내용의 해시(SHA-256)로 알아봐요. 그래서 폴더를 옮기거나 이름을 바꿔도 같은 문서로
남아요. 크기와 수정 시각이 그대로인 파일은 다시 읽지 않아서, 파일이 많아도 훑는 부담이 작아요. 서버를
다시 켜도 이미 들어간 파일은 다시 넣지 않아요.

폴더를 3초마다 훑게 해 놓고 재 봤더니, 바뀐 내용이 검색에 반영되기까지 이만큼 걸렸어요. 기본 간격(30초)에서는
최대 30초쯤 더 걸려요.

- 고친 파일: 2.9~5.6초
- 옮긴 파일: 2.8초
- 지운 파일: 2.8초
- 새 파일: 3.4초

### 읽는 파일 종류

기본은 `hwp, hwpx, md, pdf, docx, pptx, html`이에요. 다른 파일은 폴더에 있어도 건너뛰어요. 바꾸려면
`OPENRAG_DOCS_TYPES`에 적고 서버를 다시 띄워요.

```bash
export OPENRAG_DOCS_TYPES=hwp,hwpx,pdf,docx      # 이 네 가지만 읽기
```

적을 수 있는 종류는 hwp, hwpx, hml, pdf, md, markdown, txt, docx, pptx, html, htm, png, jpg, jpeg, webp예요.
`md`를 적으면 `.markdown`도, `html`을 적으면 `.htm`도 같이 읽어요.

이런 파일은 건너뛰어요.

- 이름이 `.`으로 시작하는 파일과 폴더, Office 잠금 파일(`~$보고서.docx`)
- 64MB보다 큰 파일
- 2초 안에 바뀐 파일(아직 쓰는 중일 수 있어서요). 다음에 훑을 때 넣어요.
- 읽다가 실패한 파일. 로그에 남기고, 파일이 바뀌면 다시 시도해요.

### 폴더 여러 개

`/docs` 아래에 붙인 폴더는 설정을 따로 바꾸지 않아도 같이 읽어요. `docker/docker-compose.yml`의
`openrag-api` → `volumes`에 한 줄 더 적고 서버를 다시 띄우면 돼요.

```yaml
      - ${OPENRAG_DOCS_DIR:-../documents}:/docs:ro
      - /Volumes/NAS/규정:/docs/nas-규정:ro          # 추가한 줄
```

폴더마다 버킷이나 파일 종류를 다르게 하려면, 그 폴더를 `/docs` 밖에 붙이고 `OPENRAG_DOCS_DIRS`에 적어요.
목록은 JSON으로 써요.

```yaml
      - /srv/인사:/docs-hr:ro                         # volumes에 추가
```

```bash
export OPENRAG_DOCS_DIRS='["/docs", {"path": "/docs-hr", "bucket": "hr", "types": ["pdf", "hwp"]}]'
```

볼 수 있는 사람은 버킷마다 정해요. 폴더에서 들어간 문서는 그 버킷의 권한을 따라요. 예를 들어 권한
등급 5 이상만 보는 `hr` 버킷은 `init` 때 받은 관리자 키로 이렇게 만들어요([8장](#8-팀원-추가와-권한)).

```bash
export OPENRAG_API_KEY=orag_…            # init이 준 관리자 키
docker compose -f docker/docker-compose.yml exec -e OPENRAG_API_KEY openrag-api \
  openrag bucket create hr --auth-level 5
```

### 폴더가 사라졌을 때

- 폴더를 읽을 수 없으면 그 폴더의 문서를 그대로 둬요. NAS 연결이 끊긴 경우가 여기에 해당해요.
- 파일이 두 개 이상 있던 폴더가 한꺼번에 비면 문서를 내리지 않고 로그에 남겨요. 마운트가 빠진 폴더도
  빈 폴더로 보여서, 실수로 문서가 전부 내려가는 것을 막으려는 거예요. 폴더를 정말 비우려면 파일을 하나만
  남기고 지운 뒤, 다음 훑기가 지나면 마지막 파일을 지우세요.

### 글만 넣을 때

텍스트라면 Claude Code에 "이 내용을 사내 문서 `docs`에 넣어줘"라고 해도 돼요(`rag_ingest` 도구).

## 8. 팀원 추가와 권한

### 권한은 두 가지로 정해요

- **등급(1~10)**: 문서마다 최소 등급이 있어요. 사람의 등급이 그보다 낮으면 그 문서는 보이지 않아요.
  `init`으로 만든 관리자는 10, 문서는 따로 정하지 않으면 1이에요.
- **공개 범위**: 문서를 누구에게 보여 줄지 정해요.

| 공개 범위 | 볼 수 있는 사람 |
| --- | --- |
| `org` | 조직 전체 (`init`이 만든 `docs` 버킷의 기본값) |
| `dept` | 같은 부서 |
| `group` | 지정한 권한 그룹 |
| `private` | 올린 사람, 그리고 같은 부서의 부서 관리자 |

두 조건을 모두 통과해야 보여요. 예를 들어 실제로 등급 3인 팀원은 등급 5로 넣은 문서를
검색해도 찾지 못했고, 관리자는 찾았어요. 문서 폴더에서 들어간 문서는 그 버킷의 등급과 공개 범위를 따라요
([7장](#7-문서-넣기고치기지우기)의 `hr` 버킷 예).

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
- 회사 밖 임베딩 서버로 보내는 문서도 가려서 보내요([11장](#11-검색-품질-임베딩-서버)). 기본 구성처럼
  임베딩 서버가 회사 안에 있으면 원문으로 만들어요.
- 그래서 Claude Code 답에는 전화번호 같은 값이 가려져 나와요. 원문이 필요하면 문서를 직접 보거나
  HTTP API를 쓰세요.
- 질문에 직접 쓴 이름이나 번호는 가리지 않아요. 본인이 쓴 내용이니까요.

실제 문서로 확인한 결과는 이래요.

| 항목 | AI로 나갈 때 |
| --- | --- |
| 주민등록번호, 전화번호, 이메일, 카드번호, 여권번호 | 가려요 |
| 계좌번호 | `계좌`·`입금`·`은행` 가까이에 있으면 가려요. 은행마다 다른 자릿수(국민 6-2-6, 신한 3-3-6, 농협 3-4-4-2 등)와 하이픈 없는 번호도 잡아요 |
| 사람 이름 | `성명: 홍길동`처럼 라벨 뒤에 있거나, `김민준과장`·`박서연 대리`처럼 직함이 붙으면 가려요. 문장 속에 이름만 있으면 못 잡아요 |

`PROJECT-BLACKBOX`처럼 정해 둔 금지어가 들어간 질문·문서는 아예 막아요. 규칙은
`configs/pipeline.default.yaml`의 `pii.guard`에서 바꿀 수 있어요.

## 10. 스캔본 읽기(OCR)

HWP·HWPX·MD·TXT, 그리고 글자가 들어 있는 PDF는 서버 안에서 바로 읽어요. 스캔한 PDF와 사진
(PNG, JPG, WEBP)은 OCR이 있어야 글자를 읽을 수 있어요.

- PDF에 글자 층이 없는 페이지(스캔한 페이지)가 있으면 PDF를 Schift OCR에 보내서 그 페이지를 채워요.
  글자 층이 있는 페이지는 원래 글자를 그대로 써요.
- 이때 PDF 전체가 한 번에 가요. Schift 사용량도 전체 페이지 수로 잡혀요.
- OCR을 설정하지 않았는데 글자가 하나도 없는 PDF를 넣으면, 빈 문서로 넣지 않고 오류로 알려 줘요
  (`this PDF has no text layer (a scan?) and OCR is not configured`).
- Schift가 바쁘다고(429) 하면 기다렸다가 다시 보내요. 최대 8번까지, 기다리는 시간은 모두 합쳐 길어야
  160초예요.

OCR을 쓰려면 서버를 띄우기 전에 두 값을 넣으세요.

```bash
export OPENRAG_SCHIFT_OCR_URL=https://schift.io
export SCHIFT_OCR_API_KEY=<Schift OCR 키>
docker compose -f docker/docker-compose.yml --profile app up -d
```

OCR 모드는 문서 등급에 따라 자동으로 정해져요.

| 문서 등급 | 모드 | 뜻 |
| --- | --- | --- |
| 1~4 | `share` | 기본값. Schift가 페이지를 학습에 쓸 수 있어요 |
| 5~7 | `private` | 학습에 쓰지 않아요. 결과는 Schift에 남아요 |
| 8~10 | `zdr` | Schift에 아무것도 남기지 않아요 |

학습에 쓰이면 안 되는 문서는 등급 5 이상인 버킷의 폴더에 넣으세요([7장](#7-문서-넣기고치기지우기)). 기준은
`configs/pipeline.default.yaml`의 `ocr_mode_by_auth_level`에서 바꿀 수 있어요.

## 11. 검색 품질: 임베딩 서버

검색은 임베딩 모델로 뜻이 비슷한 문단을 찾아요. 기본 구성에는 임베딩 서버 `local-embed`가 같이 떠요.
`google/embeddinggemma-2`를 CPU에서 돌리고, 256차원으로 잘라 써요. 이 설정으로 한국어 검색 평가
(Ko-StrategyQA)에서 nDCG@10 0.781이 나왔어요.

- 처음 띄울 때 모델 파일(약 1.5GB)을 Hugging Face에서 받아 `embed_models` 볼륨에 저장해요. 그다음부터는
  받지 않아요.
- 폐쇄망 구성은 미리 받아 둔 `./models`를 읽기 전용으로 써요(3장).

### 다른 임베딩 서버 쓰기

OpenAI 호환 `/v1/embeddings` 서버라면 어디든 연결할 수 있어요. 서버를 띄우기 전에 값을 넣으세요.

| 환경변수 | 뜻 |
| --- | --- |
| `OPENRAG_TEXT_EMBED_URL` | 임베딩 API 주소(전체 경로). 예: `https://api.openai.com/v1/embeddings` |
| `OPENRAG_TEXT_EMBED_MODEL` | 모델 이름 |
| `OPENRAG_TEXT_EMBED_API_KEY` | API 키(필요할 때) |
| `OPENRAG_TEXT_EMBED_QUERY_PREFIX` | 질문 앞에 붙일 말. embeddinggemma는 알아서 붙여요. e5 계열은 `query: ` |
| `OPENRAG_TEXT_EMBED_DOCUMENT_PREFIX` | 문서 앞에 붙일 말. e5 계열은 `passage: ` |

> **회사 밖 임베딩 서버로는 개인정보를 가려서 보내요.** 문서 안의 이름·전화번호·계좌번호 같은 값을
> `[PII:NAME]`처럼 바꿔서 보내고, 질문은 쓴 그대로 보내요. 서버가 뜰 때 로그에
> `embeddings server is outside this network`가 찍히면 이 상태예요.
>
> 회사 안으로 보는 주소는 `127.0.0.1`·`10.x`·`172.16~31.x`·`192.168.x` 같은 내부 주소, 점 없는 이름
> (`local-embed` 같은 compose 서비스), `.local`·`.internal`·`.lan`으로 끝나는 이름, 내부 주소로만
> 풀리는 이름이에요. 나머지는 모두 회사 밖으로 봐요.
>
> 가려서 만든 벡터는 이름·번호의 뜻을 담지 못해서, 이름으로 찾는 검색이 조금 약해져요. 시험 문서에서는
> "박서연 대리 연락처", "김민준 과장 남은 연차" 모두 맞는 문서가 1위로 나왔지만, 두 번째 질문은 2위와 차이가
> 작았어요(0.694 대 0.680).

외부 서버를 쓸 때는 `local-embed`를 띄우지 않아도 돼요. `--no-deps`로 띄우세요.

```bash
export OPENRAG_TEXT_EMBED_URL=https://api.openai.com/v1/embeddings
export OPENRAG_TEXT_EMBED_MODEL=text-embedding-3-small
export OPENRAG_TEXT_EMBED_API_KEY=sk-…
docker compose -f docker/docker-compose.yml --profile app up -d --no-deps postgres openrag-api
```

벡터는 어떤 모델이든 256차원으로 잘라서 저장해요. **모델을 바꾸면 이미 넣은 문서는 예전 모델의
벡터로 남아 있어서**, 새 모델로 찾으면 맞지 않아요. 모델은 문서를 넣기 전에 정하세요.

### 검색 인덱스

벡터는 PostgreSQL 안의 pg_pinecone 인덱스로 찾아요. 비슷한 벡터끼리 묶음(클러스터)으로 나눠 두고,
질문과 가까운 묶음만 뒤지는 방식이에요. 묶음은 인덱스를 만들 때 나누기 때문에, 문서가 늘면 다시
나눠야 해요.

- 기본 구성에서는 데이터베이스가 알아서 다시 나눠요. 마지막으로 나눈 뒤 넣은 문서가 전체의 20%를 넘으면
  인덱스를 다시 만들어요(`pinecone.auto_maintain=on`, 마이그레이션 014).
- 다시 만드는 동안은 문서 넣기와 검색이 잠깐 기다려요. 10만 개 기준으로 3초쯤 걸렸어요.
- PostgreSQL을 직접 운영한다면 `postgresql.conf`에 `pinecone.auto_maintain = on`을 넣으세요. 이게 꺼져
  있으면 인덱스가 다시 만들어지지 않아서, 검색할 때마다 전체를 훑어요.

10만 개로 재 보면 이래요(Ko-StrategyQA 질문 592개, 8명이 동시에 검색).

| 상태 | nDCG@10 | 응답 시간(중앙값) |
| --- | --- | --- |
| 인덱스를 다시 만들지 않을 때(전체를 훑음) | 0.7159 | 1,307ms |
| 다시 만든 뒤, 기본 설정 | 0.7172 | 252ms |

검색할 때 묶음을 얼마나 뒤질지는 `configs/pipeline.default.yaml`의 `store.pg_pinecone_search`에서 정해요.

| 항목 | 기본값 | 뜻 |
| --- | --- | --- |
| `scan_fraction` | `0.5` | 전체 묶음 중 뒤질 비율(0~1). 크면 정확하고 느려요 |
| `rescore_multiplier` | `4` | 정확한 거리로 다시 잴 후보 수(×100개) |

둘 다 `0`이면 pg_pinecone이 인덱스를 만들 때 정한 값을 써요. 위 10만 개에서는 nDCG@10이 0.7119로,
기본 설정보다 0.005 낮았고 속도는 같았어요.

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

### `dependency failed to start: container … local-embed … is unhealthy`
임베딩 모델을 아직 받는 중이거나, 받다가 실패했어요. `docker compose -f docker/docker-compose.yml logs local-embed`를
보세요. 받는 중이었다면 다 받은 뒤 3장 명령을 한 번 더 실행하면 `openrag-api`가 떠요.

### `no space left on device`
Docker가 쓰는 디스크가 꽉 찼어요. 모델(약 1.5GB)과 이미지를 받을 자리가 필요해요. `docker system df`로
무엇이 자리를 차지하는지 보고, 필요 없는 것을 정리하거나 Docker Desktop의 디스크 크기를 늘리세요.

### PDF를 넣었더니 `this PDF has no text layer (a scan?) and OCR is not configured`
스캔한 PDF예요. OCR을 설정하고 다시 넣으세요([10장](#10-스캔본-읽기ocr)).

### PDF가 안 들어가고 로그에 `invalid byte sequence for encoding "UTF8": 0x00`
글꼴을 파일 안에 넣어 저장한 PDF는 지금 글자를 제대로 읽지 못해요. LibreOffice에서 저장한 PDF로 확인했고,
Word나 한글에서 PDF로 저장한 파일도 같은 방식이라 같은 문제가 날 수 있어요(직접 시험하지는 못했어요).
그동안은 원본(HWP, HWPX, DOCX)을 폴더에 넣어 주세요.

### 임베딩 모델을 바꾼 뒤 검색이 엉뚱한 문서를 가져옴
이미 넣은 문서는 예전 모델로 만든 벡터로 남아 있어요([11장](#11-검색-품질-임베딩-서버)).

### `/v1/chat/completions`가 503
서버 쪽 LLM을 연결하지 않았어요([12장](#12-서버에서-직접-답하기선택)). Claude Code로 물을 때는 필요 없어요.

## 14. 멈추기와 다시 켜기

```bash
docker compose -f docker/docker-compose.yml --profile app stop    # 멈추기
docker compose -f docker/docker-compose.yml --profile app start   # 다시 켜기
```

문서와 계정은 Docker 볼륨(`pg_data`)에, 임베딩 모델은 `embed_models`에 남아 있어서, 멈췄다 켜도 그대로예요.
모델을 다시 받지도 않아요.
