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
- [14. 멈추기, 다시 켜기, 백업, 업데이트](#14-멈추기-다시-켜기-백업-업데이트)
- [15. 다른 도구에 붙이기: 플러그인, kordoc, Docufinder](#15-다른-도구에-붙이기-플러그인-kordoc-docufinder)

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

- **Docker**: Docker Desktop, 또는 Docker Engine과 Compose v2. 처음 띄울 때 Docker 디스크가 12GB쯤 비어 있어야
  해요. 이미지 3.8~4.5GB(`docker system df` 기준), 빌드 캐시 6GB쯤, 임베딩 모델 1.5GB가 들어가요. 빌드 캐시는 다 뜬 뒤 지워도 돼요. 임베딩
  서버가 쓸 메모리도 3GB쯤 더 필요해요.
- **질문할 PC의 Claude Code 또는 Codex**
- **문서 폴더**: 서버가 이 폴더를 계속 지켜보면서 파일을 넣고, 고친 파일은 새 판으로 바꾸고, 지운 파일은
  검색에서 빼요([7장](#7-문서-넣기고치기지우기)). 기본으로 HWP, HWPX, MD, PDF, DOCX, PPTX, HTML, XLSX, CSV를
  읽고, 하위 폴더까지 다 봐요.
- (선택) **Schift OCR 키**: 사진(PNG, JPG, WEBP)과, 서버 안 OCR이 읽지 못한 PDF 페이지를 읽을 때 필요해요.
  스캔한 PDF는 키 없이도 서버 안에서 읽어요([10장](#10-스캔본-읽기ocr)).

호스트에 따로 설치할 프로그램은 없어요. 설정에 쓰는 `openrag` 명령은 서버 이미지 안에 들어 있어요.

## 3. 서버 띄우기

### AI에게 맡기기(설치 마법사)

Claude Code나 Codex가 있으면 설치를 맡길 수 있어요. 저장소를 받은 폴더에서 Claude Code를 열고 `/openrag-setup`을
입력하세요. Codex에서는 `$openrag-setup`을 입력하면 돼요. "OpenRAG 설치해줘"라고만 해도 알아서 골라 써요.

마법사는 이 순서로 진행해요.

1. Docker, 디스크, 포트를 점검해요.
2. 문서 폴더, 조직 이름, 접속 주소, 연결할 도구를 한 번에 물어봐요.
3. 설정을 `docker/.env`에 적어요.
4. 서버를 띄우고 `init`을 실행해요.
5. 문서가 들어갔는지 확인해요.
6. Claude Code나 Codex에 등록해요.
7. 검색이 되는지까지 봐요.

MCP 등록처럼 PC 설정을 바꾸는 일은 먼저 물어봐요. `docker/.env`에는 암호화 키가 들어가니 따로 백업해 두세요.
이미 설치된 서버라면 새로 설치하지 않고, 문서 폴더 추가·키 다시 받기·다시 연결 중에서 고르게 해요.

직접 하려면 아래를 따라 하세요.

### 직접 띄우기

저장소를 받은 폴더에서 실행해요.

```bash
export OPENRAG_PII_KEK_HEX=$(openssl rand -hex 32)
export OPENRAG_DOCS_DIR=~/회사문서      # 문서 폴더. 생략하면 저장소의 documents/ 폴더
mkdir -p "${OPENRAG_DOCS_DIR:-documents}"   # 폴더를 먼저 만들어 두세요(아래 설명)
docker compose -f docker/docker-compose.yml --profile app up -d
```

- `OPENRAG_PII_KEK_HEX`는 서버가 쓰는 암호화 키예요. 서버를 다시 띄울 때도 **같은 값**을 넣어야 하니,
  처음 만든 값을 안전한 곳에 적어 두세요.
- `OPENRAG_DOCS_DIR`은 서버가 지켜볼 문서 폴더예요. 서버는 이 폴더를 읽기만 하고, 파일을 바꾸거나
  지우지 않아요. 폴더를 여러 개 쓰는 방법은 [7장](#7-문서-넣기고치기지우기)에 있어요.
- 문서 폴더는 `up` 전에 만들어 두세요. 없는 채로 띄우면 Linux에서는 Docker가 그 폴더를 root 소유로 만들어서,
  그 안에 파일을 넣을 수 없어요.

컨테이너는 세 개가 떠요: 데이터베이스(`postgres`), 임베딩 서버(`local-embed`), API(`openrag-api`).
처음 한 번은 오래 걸려요.

- 이미지 세 개를 빌드해요.
- `local-embed`가 임베딩 모델(약 1.5GB)을 Hugging Face에서 받아요. 다 받아서 준비될 때까지
  `openrag-api`는 기다렸다가 떠요. 진행 상황은 아래 명령으로 볼 수 있어요.
- 임베딩 서버는 메모리를 2.2GB쯤 써요.

아무것도 받아 두지 않은 PC(Docker Desktop, macOS, 2026-11-24)에서 빌드에 3분 24초, 모델을 받아 준비되기까지
1분 22초가 걸렸어요. Linux(Docker Engine 기본 빌더, 캐시 없이 빌드, 2026-11-25)에서는 빌드에 11분 24초,
띄운 뒤 준비되기까지 1분 17초가 걸렸어요. 회선 속도와 CPU에 따라 달라요.

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
- `config.not_saved`라는 줄이 보이면 정상이에요. 컨테이너 안에서는 설정 파일을 쓸 권한이 없어서(`Permission denied`)
  그래요. 키와 연결 명령은 위 출력에 다 있어요.

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

Claude Code는 명령 대신 OpenRAG 플러그인으로 붙여도 돼요. 주소와 키를 설치 화면에서 넣고, kordoc으로 만든 문서를
바로 저장하는 기능도 같이 들어와요(15장). 둘 중 하나만 하세요. 둘 다 하면 같은 서버가 두 번 붙어요.

### 다른 PC에서 접속할 때(HTTPS)

기본 연결은 HTTP라서 키가 네트워크에 그대로 지나가요. 다른 PC에서 쓰면 HTTPS 오버레이를 켜세요. 앞에 Caddy가
서서 HTTPS를 받고, 평문 HTTP 포트는 서버 PC 안(`127.0.0.1`)에만 열려요.

```bash
# docker/.env에 추가(사내 이름이나 IP)
OPENRAG_HTTPS_HOST=openrag.acme.local
OPENRAG_HTTPS_PORT=8443
OPENRAG_PUBLIC_URL=https://openrag.acme.local:8443

docker compose -f docker/docker-compose.yml -f docker/docker-compose.https.yml --profile app up -d
```

이후 모든 명령에 같은 `-f` 두 개를 붙이고, `init`을 다시 실행하면 연결 명령에 `https://` 주소가 나와요.

- **사내용(기본, `OPENRAG_HTTPS_TLS=internal`)**: Caddy가 자체 인증서를 만들어요. 접속하는 PC가 그 인증서를
  믿게 해야 해요. 루트 인증서는 이렇게 꺼내요.

  ```bash
  docker compose -f docker/docker-compose.yml -f docker/docker-compose.https.yml \
    cp https:/data/caddy/pki/authorities/local/root.crt ./openrag-ca.crt
  ```

  Claude Code는 실행할 때 `NODE_EXTRA_CA_CERTS=/경로/openrag-ca.crt`를 주면 믿어요. 이 방식으로 HTTPS 연결과
  검색까지 확인했어요. 다른 도구(Codex 등)는 PC의 신뢰 저장소에 인증서를 넣어 주세요(시험하지 못했어요).
- **인터넷 도메인이 있을 때**: `OPENRAG_HTTPS_TLS`에 이메일 주소를 넣으면 Let's Encrypt 인증서를 받아요. 그 이름이
  이 서버를 가리키고 80·443 포트가 밖에서 열려 있어야 해요(`OPENRAG_HTTPS_PORT=443`, `OPENRAG_HTTPS_HTTP_PORT=80`).

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
| 지운 파일을 되돌려 놓음(휴지통에서 꺼냄) | 예전 문서가 그대로 다시 검색돼요. 다시 읽거나 임베딩하지 않아요 |
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

기본은 `hwp, hwpx, md, pdf, docx, pptx, html, xlsx, csv`예요. 다른 파일은 폴더에 있어도 건너뛰어요. 바꾸려면
`OPENRAG_DOCS_TYPES`에 적고 서버를 다시 띄워요.

```bash
export OPENRAG_DOCS_TYPES=hwp,hwpx,pdf,docx      # 이 네 가지만 읽기
```

적을 수 있는 종류는 hwp, hwpx, hml, pdf, md, markdown, txt, docx, pptx, html, htm, xlsx, xlsm, csv, tsv, png,
jpg, jpeg, webp예요. 예전 형식(doc, ppt, xls)은 읽지 못해요.
`md`를 적으면 `.markdown`도, `html`을 적으면 `.htm`도 같이 읽어요.

이런 파일은 건너뛰어요.

- 이름이 `.`으로 시작하는 파일과 폴더, Office 잠금 파일(`~$보고서.docx`)
- 64MB보다 큰 파일
- 2초 안에 바뀐 파일(아직 쓰는 중일 수 있어서요). 다음에 훑을 때 넣어요.
- 읽다가 실패한 파일. 로그에 남기고, 파일이 바뀌면 다시 시도해요.

### 엑셀과 CSV

엑셀은 칸마다 쪼개서 넣지 않고, "이 파일이 무엇에 관한 것인지"를 적은 **안내 카드**로 넣어요. 카드에는
시트 이름, 제목 줄, 열 이름, 행 수, 날짜·숫자 범위, 자주 나오는 값, 앞쪽 10행이 들어가요. 검색은 이 카드로
파일을 찾고, 실제 칸 값은 AI가 `rag_read_sheet` 도구로 필요한 시트와 행만 서버에서 읽어 가요.

예를 들어 "2024년 판교 지점 노트북 판매 금액 합계"를 물었더니 Claude Code가 카드로 파일을 찾고, 행을 읽어
합계(13,300,000원)를 직접 계산했어요(22초).

특정 값이 든 행은 `rag_read_sheet`에 찾을 값(`find`)을 주면 서버가 시트를 끝까지 훑어 그 값이 든 행만 돌려줘요.
대소문자와 천 단위 쉼표(`1,350,000`)는 따지지 않고, 열 이름(`column`)을 주면 그 열에서만, `exact`를 켜면 칸 전체가
같은 것만 찾아요. 시트를 정하지 않으면 모든 시트를 찾아서 시트마다 몇 행이 걸렸는지도 알려 줘요. 돌려준 행마다
위치(`row_index`)가 붙어서, 앞뒤 행이 필요하면 그 위치부터 다시 읽으면 돼요.

- 검색은 카드에 적힌 내용으로만 걸려요. 칸 값은 색인하지 않아서(BM25를 켜도 마찬가지예요), 카드에 없는 값
  하나(주문번호 같은 것)만으로는 파일이 찾아지지 않아요. 무슨 파일인지("2025년 온라인 주문 내역에서")를 같이
  말해 주면 AI가 카드로 파일을 찾고, 그 안의 값은 `find`로 한 번에 찾아요.
- 행 단위 읽기는 문서 폴더에서 들어온 파일만 돼요. 서버가 원본을 그 폴더에서 바로 읽기 때문이에요.
- AI로 나가는 칸 값에서도 전화번호 같은 개인정보는 가려져요(9장).

### 지금 상태 보기

들어간 문서와 실패한 파일은 Claude Code에 "사내 문서 목록 보여줘", "문서 폴더 상태 알려줘"라고 물어보면
돼요(`rag_list_documents`, `docs_status` 도구). 명령으로도 볼 수 있어요.

```bash
export OPENRAG_API_KEY=orag_…            # init이 준 관리자 키
E="docker compose -f docker/docker-compose.yml exec -e OPENRAG_API_KEY -e OPENRAG_API_URL=http://127.0.0.1:8080 openrag-api"
$E openrag docs list                     # 문서 목록(원본 경로, 판, 효력, 문단 수). --query 일부, --all 예전 판까지
$E openrag docs status                   # 폴더별 파일 수, 문서 수, 실패한 파일과 이유, 진행 중인 반영
$E openrag docs scan                     # 30초 기다리지 않고 지금 훑기
```

- `OPENRAG_API_URL`은 명령이 컨테이너 안에서 돌고 있는 서버에 물어보게 해요. 빼면 `docs status`는 폴더를 지켜보는
  서버를 못 보고 `"watching": false`라고 잘못 답해요.
- 파일을 한꺼번에 많이 넣으면 로그와 `docs status`에 진행 상황이 나와요(`done / total`).
- 검색 결과와 목록의 `source.path`는 원본 파일을 이 PC의 경로로 보여 줘요(`OPENRAG_DOCS_DIR`에서 계산). 7장
  방식으로 붙인 다른 폴더는 `OPENRAG_DOCS_DIRS` 항목에 `"host": "/Volumes/NAS/규정"`처럼 적어 주면 그 경로로 나와요.

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

- 폴더를 읽을 수 없으면(경로가 없거나 권한이 없을 때) 그 폴더의 문서를 그대로 둬요.
- 폴더가 비면 들어 있던 문서는 검색에서 빠져요. NAS 연결이 끊겨 폴더가 비어 보인 경우도 마찬가지예요.
  다시 연결되면 같은 문서가 그대로 돌아오고, 다시 읽거나 임베딩하지 않아요.

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

관리자 키로 한 줄이면 돼요. 계정과 키를 만들고, 팀원 PC에서 실행할 연결 명령까지 출력해요. 설치 마법사에
"팀원 추가해줘"라고 해도 돼요.

```bash
export OPENRAG_API_KEY=orag_…   # 관리자 키
docker compose -f docker/docker-compose.yml exec -e OPENRAG_API_KEY openrag-api \
  openrag member add --email kim@acme.example --name 김사원 --level 3
```

- 부서는 따로 주지 않으면 관리자와 같은 부서예요(`--dept <부서 ID>`로 바꿔요).
- 출력의 `connect.claude`(또는 `connect.codex`) 한 줄을 팀원에게 보내요. 그 줄에 팀원 키가 들어 있어요.
- 팀원의 Claude Code는 팀원이 볼 수 있는 문서만 찾아요.
- 팀원 목록은 `openrag member list`예요.

### 키 목록과 폐기

키가 새었거나 팀원이 나가면 그 키를 바로 막아요.

```bash
$E openrag key list                 # 키 목록(앞 몇 글자만, 주인, 상태)
$E openrag key revoke <키 ID>        # 바로 막기. 그 키로 오는 요청은 401
```

(`$E`는 7장의 `docker compose … exec -e OPENRAG_API_KEY -e OPENRAG_API_URL=http://127.0.0.1:8080 openrag-api`예요.) 관리자가 아닌 사람은 자기 키만 보고
막을 수 있어요.

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

HWP·HWPX·MD·TXT는 서버 안에서 바로 읽어요. PDF는 서버 이미지에 들어 있는 kordoc이 읽어요. 글자 층이 있는
페이지는 그 글자를 쓰고, 스캔한 페이지는 kordoc의 한국어 OCR로 서버 안에서 읽어요. OCR 모델도 이미지에 들어
있어서, 스캔한 PDF는 Schift OCR 키 없이도 검색되고 회사 밖으로 나가는 것도 없어요.

- 서버 안 OCR은 CPU로 돌아서 느려요. 스캔한 4쪽짜리 PDF가 x86 서버에서 15초쯤 걸렸어요. 한 PDF가 10분을
  넘기면 멈추고 내장 파서로 읽어요.
- kordoc이 실패하면 서버의 내장 파서로 읽어요. 내장 파서는 스캔한 페이지를 읽지 못하고, 한글 문서에서 내보낸
  PDF는 글자를 많이 놓쳐요.
- 서버 안 OCR만 끄려면 `configs/pipeline.default.yaml`의 `parse.kordoc_ocr: false`, kordoc을 아예 쓰지
  않으려면 `parse.pdf_engine: native`로 바꾸세요.

Schift OCR은 이럴 때만 써요.

- 사진 파일(PNG, JPG, WEBP)
- 서버 안 OCR로도 글자가 나오지 않은 PDF 페이지

이때 PDF 전체가 한 번에 가요. Schift 사용량도 전체 페이지 수로 잡혀요.

- Schift OCR을 설정하지 않았으면 그런 페이지는 글자 없이 넣어요. PDF 전체에 글자가 하나도 없으면 빈 문서로
  넣지 않고 오류로 알려 줘요.
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

### 회사 밖으로 보내기 전에 묻기

- MCP 2026-07-28 규격으로 붙은 클라이언트가 사용자에게 물어볼 수 있으면(elicitation), Schift로 보내기 전에
  `scan.pdf: 2 of 6 pages have no readable text. Send those pages to Schift OCR, a cloud service, to read them?`
  처럼 물어요. 보내기를 고르면 보내고, 거절하면 그 페이지는 글자 없이 넣고, 취소하면 문서를 넣지 않아요.
- HTTP API로 파일을 넣을 때는 `remote_ocr` 값을 같이 보낼 수 있어요. `ask`는 보낼 페이지가 있으면 넣지 않고
  409(`remote_ocr_consent_required`)로 알려 줘요. `allow`는 보내고, `deny`는 보내지 않아요. 값이 없으면
  묻지 않고 보내요.
- 문서 폴더에서 읽는 파일은 묻지 않고 보내요. 회사 밖으로 보내면 안 되는 문서가 폴더에 있으면 Schift OCR을
  설정하지 마세요. 등급(위 표)은 Schift에 남는지만 정하고, 보내는 것 자체는 막지 않아요.

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

### `unknown command: docker compose` 또는 `unknown shorthand flag: 'f' in -f`
Docker에 Compose가 없어요. Ubuntu의 `docker.io` 패키지에는 Compose가 들어 있지 않아요(Ubuntu 24.04에서 확인).
`sudo apt install docker-compose-v2`로 넣고 `docker compose version`이 나오는지 보세요. Docker 공식 저장소로
설치했다면 패키지 이름이 `docker-compose-plugin`이에요. Ubuntu 패키지의 Compose 2.40.3이 이 저장소의 compose
파일을 읽는 것까지 확인했고, 그 패키지로 서버를 띄워 보지는 못했어요. 패키지를 못 넣는 곳에서는 Compose 단독
실행 파일(GitHub `docker/compose` 릴리스)을 받아 `docker compose` 대신 써도 돼요. v5.8.0으로 처음부터 빌드해 띄우고
검색까지 확인했어요(Ubuntu 24.04, 2026-11-25).

### 문서 폴더에 파일을 넣을 수 없음(`Permission denied`)
문서 폴더가 없는 채로 서버를 띄워서, Docker가 그 폴더를 root 소유로 만들었어요(Linux). `sudo chown "$USER" <문서 폴더>`로
주인을 바꾸면 돼요. 폴더를 먼저 만들어 두면(`mkdir -p`) 이렇게 되지 않아요.

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
Docker가 쓰는 디스크가 꽉 찼어요. 처음 띄울 때는 12GB쯤 필요해요(이미지 3.8~4.5GB, 빌드 캐시 6GB쯤, 모델 1.5GB).
`docker system df`로 무엇이 자리를 차지하는지 보세요. 빌드 캐시(`Build Cache`)가 크면 `docker builder prune -a`로
지울 수 있어요. 다른 프로젝트의 빌드 캐시도 같이 지워지니, 다음 빌드가 조금 느려지는 것 말고는 문제없어요. 그래도
모자라면 Docker Desktop 설정에서 디스크 크기를 늘리세요. 자리를 만든 뒤 같은 `up -d` 명령을 다시 실행해요. 처음
설치하는 중이라 아직 넣은 문서가 없다면, `down -v`로 볼륨까지 지우고 다시 띄우는 게 가장 확실해요. 볼륨을 지우면
데이터베이스도 지워지니 설치 중일 때만 그렇게 하세요.

### PDF를 넣었더니 `kordoc OCR found no text in this PDF and Schift OCR is not configured`
서버 안 OCR로도 글자가 하나도 나오지 않은 PDF예요. 빈 페이지만 있거나 그림뿐인 PDF일 수 있어요. 글자가
있어야 하는 문서라면 Schift OCR을 설정하고 다시 넣으세요([10장](#10-스캔본-읽기ocr)). 내장 파서로 읽을 때는
같은 경우에 `this PDF has no text layer (a scan?) and OCR is not configured`가 나와요.

### PDF의 글자가 빠지거나 `this PDF has no text layer`가 남
기본으로는 kordoc이 PDF를 읽어요. 아래는 kordoc을 끄거나(`parse.pdf_engine: native`) kordoc이 실패해서
서버의 내장 파서로 읽을 때 얘기예요. 서버 로그에 `falling back to native PDF parser`가 있는지 보세요.

PDF 안 글꼴에 "글자표"(ToUnicode)가 있으면 읽어요. LibreOffice와 Chrome에서 저장한 PDF로 확인했어요.
Word·한글이 쓰는 방식(압축된 객체 묶음, 2바이트 글꼴)은 같은 구조로 만든 시험 파일로 확인했고,
Word·한글에서 직접 저장한 파일로는 아직 시험하지 못했어요.

글꼴만 들어 있고 글자표가 없는 PDF도 있어요. 예를 들어 macOS의 `cupsfilter`로 만든 PDF가 그래요.
이런 PDF는 글자를 알아낼 방법이 없어서 스캔본처럼 다뤄요. OCR이 설정돼 있으면 OCR로 읽고, 없으면
위 오류가 나요([10장](#10-스캔본-읽기ocr)).

### 임베딩 모델을 바꾼 뒤 검색이 엉뚱한 문서를 가져옴
이미 넣은 문서는 예전 모델로 만든 벡터로 남아 있어요([11장](#11-검색-품질-임베딩-서버)).

### `/v1/chat/completions`가 503
서버 쪽 LLM을 연결하지 않았어요([12장](#12-서버에서-직접-답하기선택)). Claude Code로 물을 때는 필요 없어요.

## 14. 멈추기, 다시 켜기, 백업, 업데이트

```bash
docker compose -f docker/docker-compose.yml --profile app stop    # 멈추기
docker compose -f docker/docker-compose.yml --profile app start   # 다시 켜기
```

문서와 계정은 Docker 볼륨(`pg_data`)에, 임베딩 모델은 `embed_models`에 남아 있어서, 멈췄다 켜도 그대로예요.
모델을 다시 받지도 않아요.

### 백업

**두 가지를 함께** 백업해요. 데이터베이스와 `docker/.env`(암호화 키)예요. 둘 중 하나만 있으면 복원할 수 없어요.

```bash
docker compose -f docker/docker-compose.yml exec -T postgres \
  pg_dump -U openrag -d openrag -Fc --exclude-table-data=pinecone.index_maintenance \
  > openrag-$(date +%F).dump
cp docker/.env openrag-$(date +%F).env
```

서버를 켜 둔 채로 떠도 돼요. `--exclude-table-data`는 꼭 붙여요. 그 표는 인덱스를 내부 번호로 가리켜서
복원할 때 걸리고, 서버가 시작할 때 다시 채워요.

### 복원

**새로 설치한 빈 서버**에서만 해요. 같은 `docker/.env`(암호화 키)를 먼저 넣어 둬요.

```bash
docker compose -f docker/docker-compose.yml --profile app up -d postgres
docker compose -f docker/docker-compose.yml exec -T postgres dropdb -U openrag --force openrag   # 처음 생긴 빈 DB
docker compose -f docker/docker-compose.yml exec -T postgres createdb -U openrag openrag
docker compose -f docker/docker-compose.yml exec -T postgres \
  pg_restore -U openrag -d openrag --exit-on-error < openrag-2026-11-24.dump
docker compose -f docker/docker-compose.yml --profile app up -d
```

`dropdb`는 처음 시작할 때 생긴 빈 DB를 지우는 거예요. 문서가 있는 서버에서는 실행하지 마세요. 시험한
결과 문서·문단·키·개인정보 보관함이 그대로 돌아왔고, 서버가 문서 폴더를 다시 넣지 않았어요(`added=0`).

### 업데이트

```bash
git pull
docker compose -f docker/docker-compose.yml --profile app up -d --build
```

서버가 시작할 때 데이터베이스 구조를 알아서 새 판에 맞춰요. 문서와 키는 그대로예요. 올리기 전에 위 방법으로
백업해 두세요.

## 15. 다른 도구에 붙이기: 플러그인, kordoc, Docufinder

### Claude Code 플러그인

```text
/plugin marketplace add schift-io/openrag
/plugin install openrag@openrag
```

설치할 때 서버 주소(`http://127.0.0.1:8080`처럼 끝에 `/` 없이), API 키, 버킷을 물어요. 키는 OS 보안 저장소에
들어가요. 바꿀 때는 `/plugin configure openrag@openrag`를 열어요. 플러그인을 쓰면 5장의 `claude mcp add`는 하지
않아요.

플러그인에 들어 있는 것:

- 검색·문서 읽기 MCP(`rag_search`, `rag_read_document`, `rag_read_sheet` …)
- 설치·운영 마법사 `/openrag:openrag-setup`. 저장소 밖에서 부르면 저장소를 받을 곳부터 물어요.
- kordoc 자동 저장(아래)

### Codex 플러그인

```bash
codex plugin marketplace add schift-io/openrag
codex plugin add openrag@openrag
```

Codex 플러그인에는 키를 넣는 설정이 없어요. 그래서 MCP는 5장의 `connect.codex`로 연결해요. kordoc 자동 저장은
셸 환경변수 `OPENRAG_API_KEY`(5장과 같은 키)와 `OPENRAG_URL`(기본 `http://127.0.0.1:8080`)을 읽어요.
Codex는 플러그인 훅을 처음 한 번 `/hooks`에서 내용을 보고 승인해야 실행해요.

### kordoc으로 만든 문서 바로 저장하기

[kordoc](https://github.com/chrisryugj/kordoc)은 HWPX·HWP 공문서를 만들고 고치는 MCP예요. 플러그인이 있으면
kordoc이 쓴 파일이 그 자리에서 OpenRAG에 들어가요. 문서 폴더로 옮기지 않아도 바로 검색돼요.

```text
> kordoc으로 부산 출장 결과 보고서를 HWPX로 만들어 줘 …
  (kordoc 결과 바로 아래) OpenRAG: 부산출장_결과보고.hwpx 저장했어요. 버킷 docs
> 사내 문서에서 찾아서 답해줘: 부산 지사 항온항습기 교체 비용
  → 첫 결과: 부산출장_결과보고, 4,300,000원
```

- kordoc 연결은 따로 해요: `npx -y kordoc setup`, 또는 `claude mcp add kordoc -- npx -y kordoc mcp`
  (Codex는 `codex mcp add kordoc -- npx -y kordoc mcp`). MCP 이름에 `kordoc`이 들어 있으면 알아봐요.
- 저장하는 도구: `generate_document`, `fill_form`, `patch_document`, `convert_to_hwpx`, `place_seal`,
  `add_memo`, `redact_document`. 저장하는 종류는 문서 폴더와 같아요(hwpx, hwp, md, pdf, docx, pptx, html, xlsx, csv).
- 같은 경로에 다시 쓰면 이전 문서를 바꿔요. 검색에는 새 내용만 나와요. 내용이 같으면 그대로 둬요.
- 서버가 꺼져 있거나 키가 틀리면 경고 한 줄만 보이고 kordoc 결과는 그대로예요.
- 끄기: Claude Code는 플러그인 설정의 `kordoc 문서 자동 저장`, Codex는 `OPENRAG_KORDOC_AUTOSAVE=false`.
- kordoc이 서버의 문서 폴더 안에 쓴다면 그 폴더 경로를 설정 `docs_dir`(Codex는 `OPENRAG_DOCS_DIR`)에 넣으세요.
  그 안의 파일은 폴더 감시에 맡기고 두 번 올리지 않아요.

위 예시는 Claude Code에서 kordoc 4.25.11로 실제로 해 본 것을 줄인 거예요. Codex에서도 같은 방식으로 저장과 검색을
확인했어요. 저장에는 0.3~3.6초 걸렸어요.

### Docufinder(Anything)와 같이 쓰기

[Docufinder](https://github.com/chrisryugj/Docufinder)(앱 이름 Anything)는 PC에서 문서 본문을 키워드로 찾는
Windows·macOS 앱이에요. 밖에서 부를 수 있는 API나 플러그인이 없어서, OpenRAG와는 **같은 폴더를 함께 보는
방식**으로 붙여요.

- Docufinder의 "폴더 추가"에 OpenRAG 문서 폴더(`OPENRAG_DOCS_DIR`)를 넣어요. 사람은 Docufinder로 키워드를
  찾고, AI는 같은 파일을 OpenRAG로 찾아요. 파일은 그 폴더 한 곳에서만 관리하면 돼요.
- Docufinder 답변 아래 "MD 저장"이나 검색 결과의 "Markdown으로 저장"으로 만든 파일을 OpenRAG 문서 폴더에
  저장하면 30초 안에 OpenRAG에도 들어가요.
- 서버가 다른 PC에 있으면 문서 폴더를 공유 드라이브에 두고 양쪽이 같은 폴더를 보게 해요.

Docufinder와는 시험하지 못했어요. 이 환경에 앱을 설치하지 않았어요.

### 그 밖의 MCP 클라이언트

HTTP MCP에 헤더를 넣을 수 있는 도구면 주소 `http://<서버>:8080/mcp`와 `Authorization: Bearer orag_…` 헤더만
있으면 돼요. Cursor 예시는 [README](../README.md)에 있어요. VS Code(`.vscode/mcp.json`)는 이렇게 써요.

```json
{
  "servers": {
    "openrag": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp",
      "headers": { "Authorization": "Bearer orag_…" }
    }
  }
}
```

Cursor와 VS Code는 시험하지 못했어요.

OpenRAG는 MCP 2026-07-28 규격과 이전 규격(2025-06-18, 2025-03-26, 2024-11-05)을 둘 다 받아요. HTTP로
붙은 클라이언트가 파일을 직접 넣게 하려면 서버에 `OPENRAG_MCP_FILE_UPLOADS=1`을 넣으세요. 기본은 꺼져 있어요.
4MiB까지는 요청에 담아 보내고, 그보다 크면 64MiB까지 서버가 알려 주는 주소로 따로 올려요. 올린 파일은
넣은 뒤 서버에서 지워요. Claude Code 2.1.332는 새 규격으로 붙지만 파일 올리기는 아직 쓰지 않아요.
