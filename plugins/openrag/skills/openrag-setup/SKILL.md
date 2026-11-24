---
name: openrag-setup
description: OpenRAG 설치·운영 마법사. 이 저장소에서 OpenRAG 서버를 처음 띄우고(Docker), 조직·관리자 키를 만들고(init), 문서 폴더를 연결하고, Claude Code나 Codex에 MCP로 등록한 뒤 검색이 되는지 확인까지 한 번에 진행한다. 설치된 뒤에는 팀원 추가, 키 폐기, 문서 폴더 상태 점검, 백업·복원, 업데이트, HTTPS 켜기도 한다. "OpenRAG 설치", "openrag 셋업", "init 해줘", "사내 문서 검색 붙여줘", "설치 마법사", "팀원 추가해줘", "키 막아줘", "백업해줘", "업데이트해줘", "HTTPS로 바꿔줘" 같은 요청에 쓴다.
---

# OpenRAG 설치 마법사

사용자를 대신해 설치를 진행하는 마법사 역할을 한다. 명령은 직접 실행하고, 사용자에게는 결정할 것만 묻는다.
자세한 설명이 필요하면 `docs/guide.md`(3·4·5·7·13·15장)를 연다.

플러그인으로 불려서 지금 폴더에 `docker/docker-compose.yml`이 없으면, 저장소를 어디에 받을지 묻고
`git clone https://github.com/schift-io/openrag <폴더>`로 받은 뒤 그 폴더에서 진행한다. 이미 받아 둔 곳이 있으면
거기로 간다. 아래 명령은 모두 그 저장소 루트에서 실행한다.

## 원칙

- **묻는 것은 한 번에 모아서.** 1단계 점검이 끝나면 질문을 한 메시지에 묶고, 각 질문에 기본값을 적는다.
  사용자가 "기본값으로"라고 하면 그대로 진행한다.
- **사용자 PC 설정을 바꾸기 전에는 확인을 받는다.** MCP 등록(`claude mcp add`, `codex mcp add`)과
  셸 설정 파일(`~/.zshrc` 등) 수정이 여기에 해당한다. 서버와 저장소 안 파일(`docker/.env`)은 묻지 않고 만든다.
- **비밀값을 채팅에 그대로 옮겨 적지 않는다.** 암호화 키(`OPENRAG_PII_KEK_HEX`)는 `docker/.env`에만 둔다.
  관리자 API 키는 init 출력에서 한 번 보여 주고, 보관하라고 말한다.
- **`docker/.env`의 `OPENRAG_PII_KEK_HEX`는 절대 바꾸지 않는다.** 이미 값이 있으면 그대로 쓴다. 바꾸면 이미
  저장한 개인정보 보관함을 열 수 없게 된다.
- **지우지 않는다.** 컨테이너·볼륨·문서를 지우는 명령(`down -v`, `volume rm`, `rm`)은 쓰지 않는다. 막히면
  원인을 보여 주고 사용자에게 고르게 한다.
- 모든 compose 명령은 저장소 루트에서 `docker compose -f docker/docker-compose.yml --profile app …`으로 실행한다.
  compose는 `docker/.env`를 읽는다. 폐쇄망 구성이면 `-f docker/docker-compose.yml -f docker/docker-compose.airgap.yml`.

## 0단계: 이미 설치돼 있는지

```bash
docker compose -f docker/docker-compose.yml --profile app ps --format '{{.Service}} {{.Status}}' 2>/dev/null
test -f docker/.env && grep -E '^(OPENRAG_API_PORT|PG_PORT|OPENRAG_DOCS_DIR|OPENRAG_PUBLIC_URL)=' docker/.env
```

`openrag-api`가 떠 있고 `curl -sf http://127.0.0.1:<API 포트>/health`가 `"status":"ok"`면 새로 설치하지 않는다.
요청에 맞는 아래 "설치 뒤에 하는 일"로 가고, 요청이 없으면 무엇을 할지 묻는다: 상태 점검, 팀원 추가, 키 폐기,
문서 폴더 추가(7장), 관리자 키 다시 받기(init을 다시 실행하면 조직은 그대로 두고 새 키만 줌), Claude Code/Codex
다시 연결, 백업, 업데이트, HTTPS.

## 1단계: 점검 (묻지 않고 실행)

```bash
docker info --format '{{.ServerVersion}}'          # Docker가 떠 있는지
docker compose version                              # Compose v2
docker system df                                    # 디스크
lsof -nP -iTCP:8080 -sTCP:LISTEN; lsof -nP -iTCP:5432 -sTCP:LISTEN   # 포트
```

- Docker가 없거나 꺼져 있으면 설치·실행을 안내하고 멈춘다.
- 처음 설치에는 Docker 디스크가 12GB쯤 필요하다(이미지 3.8GB, 빌드 캐시 6GB쯤, 임베딩 모델 1.5GB). 여유가
  그보다 적어 보이면 먼저 알린다(13장 `no space left on device`).
- 8080이나 5432를 이미 쓰고 있으면 비어 있는 포트를 골라 기본값으로 제안한다(예: 18080, 55432).

## 2단계: 질문 (한 메시지로)

1. **문서 폴더**: 서버가 지켜볼 폴더의 절대 경로. 기본값은 저장소의 `documents/`. 여러 개면 첫 번째를 기본
   폴더로 쓰고 나머지는 7장 방식(볼륨 한 줄 추가)으로 붙인다.
2. **조직 이름**: 영문 소문자 slug와 표시 이름. 기본값 `default` / `Default`.
3. **접속 주소**: Claude Code·Codex를 이 PC에서만 쓰면 기본값(`http://127.0.0.1:<API 포트>`). 다른 PC에서도
   쓰면 이 서버의 주소(예: `http://10.0.0.5:8080`).
4. **연결할 도구**: Claude Code, Codex, 둘 다(기본값은 지금 이 마법사를 실행한 도구).
5. (선택) **스캔본 OCR**: Schift OCR 키가 있는지. 없으면 건너뛴다(10장).
6. (선택) **폐쇄망**인지. 그렇다면 3장의 폐쇄망 절차를 따른다.

## 3단계: `docker/.env` 만들기

`docker/.env`가 없으면 만든다. 있으면 아는 키만 고치고 나머지 줄은 그대로 둔다. `.env`는 git에 올라가지 않는다.

```bash
umask 077
test -f docker/.env || printf 'OPENRAG_PII_KEK_HEX=%s\n' "$(openssl rand -hex 32)" > docker/.env
```

그다음 필요한 줄을 넣는다(값이 기본값이면 생략해도 된다).

```text
OPENRAG_DOCS_DIR=/절대/경로/문서폴더
OPENRAG_API_PORT=8080
PG_PORT=5432
OPENRAG_PUBLIC_URL=http://<접속 주소>:<API 포트>
OPENRAG_SCHIFT_OCR_URL=https://schift.io      # OCR을 쓸 때만
SCHIFT_OCR_API_KEY=…                          # OCR을 쓸 때만. 사용자가 직접 넣게 한다
```

문서 폴더가 없으면 만든다(`mkdir -p`). 사용자에게 `docker/.env`에 암호화 키가 있고, 이 파일을 잃으면
개인정보 보관함을 열 수 없으니 따로 백업하라고 말한다. 키 값은 말하지 않는다.

## 4단계: 서버 띄우기

```bash
docker compose -f docker/docker-compose.yml --profile app up -d
```

처음에는 이미지 빌드와 모델 다운로드로 몇 분 걸린다(아무것도 없는 PC에서 빌드 3분 24초, 모델 준비 1분 22초). 30초마다 상태를
확인하며 기다리고, 진행 상황을 짧게 알린다.

```bash
docker compose -f docker/docker-compose.yml --profile app ps --format '{{.Service}} {{.Status}}'
curl -sf http://127.0.0.1:<API 포트>/health     # {"db_connected":true,…,"status":"ok"}
```

10초 간격으로 15분까지 기다린다. 15분이 지나도 안 뜨면 `docker compose -f docker/docker-compose.yml logs --tail 50 <서비스>`로 원인을 보고
13장에서 해당 항목을 찾아 안내한다. 자주 나오는 것: `port is already allocated`(다른 포트로), `local-embed …
unhealthy`(모델 다운로드 중이거나 메모리 부족), `no space left on device`.

## 5단계: init

```bash
umask 077
docker compose -f docker/docker-compose.yml --profile app exec -T openrag-api \
  openrag init --slug <slug> --name "<표시 이름>" > docker/.openrag-init.json
```

다른 PC에서 접속한다면 `--public-url <접속 주소>`를 붙인다(`docker/.env`의 `OPENRAG_PUBLIC_URL`이 있으면
생략 가능). 출력은 관리자 키가 든 JSON이라 화면에 그대로 찍지 않고 `docker/.openrag-init.json`에만 둔다
(git이 무시하는 파일이다). 내용을 볼 때는 키를 가리고 본다. 다른 곳에 저장하지 않는다.

- `api_key`: 관리자 키. 사용자에게 한 번 보여 주고 비밀번호 관리자 같은 곳에 보관하라고 한다.
- `connect.claude`, `connect.codex`: 6단계에서 쓸 등록 명령.
- `config.not_saved`는 정상이다(컨테이너 안에는 설정을 저장하지 않는다).

## 6단계: 문서가 들어갔는지

init이 버킷을 만들면 서버가 문서 폴더를 읽기 시작한다(30초마다 훑는다). 40초쯤 기다린 뒤 확인한다.

```bash
docker compose -f docker/docker-compose.yml logs openrag-api | grep "documents folder"
```

- `documents folder synced … added=N failed=M`: 들어간 수와 실패한 수.
- 실패한 파일은 `cannot ingest file` 줄에 이유가 있다. 스캔한 PDF에 OCR이 없으면 10장, 비밀값 차단(`guard
  blocked`)이면 9장을 안내한다.
- 폴더가 비어 있으면 아무 줄도 없다. 폴더에 파일을 넣으면 30초 안에 들어간다고 알려 준다. 기본으로 읽는 종류는
  hwp, hwpx, md, pdf, docx, pptx, html, xlsx, csv다(바꾸려면 `docker/.env`에 `OPENRAG_DOCS_TYPES`). 엑셀·CSV는
  안내 카드로 들어가고 칸 값은 AI가 `rag_read_sheet`로 읽거나 `find`로 찾는다고 알려 준다.

## 7단계: Claude Code / Codex 연결 (확인 받고)

등록 명령은 사용자 PC 설정을 바꾸므로, 실행해도 되는지 먼저 묻는다.

- **Claude Code**: OpenRAG 플러그인이 설치돼 있으면(`claude plugin list`에 `openrag@openrag`) `claude mcp add`를
  하지 않는다. 같은 서버가 두 번 붙기 때문이다. 대신 사용자에게 `/plugin configure openrag@openrag`에서 서버 주소와
  키를 넣으라고 안내한다(키는 그 화면에 직접 넣게 하고 채팅에 옮겨 적지 않는다). 플러그인이 없으면 init 출력의
  `connect.claude`를 그대로 실행한다. 이미 `openrag`가 등록돼 있다는 오류가 나면 `claude mcp remove openrag -s user`로
  지우고 다시 등록해도 되는지 묻는다. 어느 쪽이든 Claude Code를 새로 시작해야 도구가 보인다고 알린다.
- **Codex**: `connect.codex`는 `OPENRAG_API_KEY` 환경 변수를 쓴다. `codex mcp add …` 부분을 실행하고, 키를
  셸 설정 파일에 넣어도 되는지 물은 뒤 `export OPENRAG_API_KEY=…` 한 줄을 추가한다(거절하면 직접 넣는 방법만
  알려 준다). Codex용 OpenRAG 플러그인의 kordoc 자동 저장도 이 키를 쓴다. 서버 주소가
  `http://127.0.0.1:8080`이 아니면 같은 곳에 `export OPENRAG_URL=<접속 주소>`도 넣는다.
- kordoc을 쓰는 사용자면 kordoc으로 만든 문서를 바로 저장하는 플러그인 기능(가이드 15장)을 한 줄로 알려 준다.
- 마법사를 다른 PC에서 쓸 사람이 있으면, 그 PC에서 실행할 명령으로 `connect.*` 줄을 넘겨준다.

## 8단계: 검색 확인

들어간 문서의 제목 하나로 질문을 만들어 MCP로 직접 검색한다.

```bash
curl -s http://127.0.0.1:<API 포트>/mcp \
  -H "Authorization: Bearer <api_key>" -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rag_search","arguments":{"query":"<질문>","bucket":"docs","top_k":3}}}'
```

결과의 첫 문서 제목과 첫 문단 일부를 보여 준다. 문서가 아직 없으면 이 단계는 건너뛰고 그렇다고 적는다.

## 설치 뒤에 하는 일

관리자 키가 필요하다. `docker/.openrag-init.json`에 있으면 거기서 읽고, 없으면 사용자에게 받거나 init을 다시
실행해 새로 받는다. 아래 `$E`는 `docker compose -f docker/docker-compose.yml exec -T -e OPENRAG_API_KEY openrag-api`다
(`OPENRAG_API_KEY`에 관리자 키를 넣고 실행). HTTPS를 켰으면 모든 compose 명령에 `-f docker/docker-compose.https.yml`을 더한다.

- **상태 점검**: `$E openrag docs status` — 폴더별 파일 수, 문서 수, 실패한 파일과 이유, 진행 중인 반영.
  실패 이유별로 가이드 13장(스캔 PDF·OCR), 9장(비밀값 차단)을 안내한다. 지금 바로 훑으려면 `$E openrag docs scan`.
- **팀원 추가**: 이메일, 이름, 등급(기본 3)을 묻고 `$E openrag member add --email … --name … --level …`.
  출력의 `connect.claude`/`connect.codex` 한 줄(팀원 키가 들어 있다)을 팀원에게 전하라고 알려 준다. 키를
  채팅에 다시 옮겨 적지 않는다. 목록은 `$E openrag member list`.
- **키 폐기**: `$E openrag key list`로 보여 주고, 어느 키인지 확인을 받은 뒤 `$E openrag key revoke <id>`.
  바로 막히고 되돌릴 수 없다고 먼저 말한다(새 키는 member add나 init으로 다시 받는다).
- **백업**: 가이드 14장 명령을 그대로 쓴다. `pg_dump … -Fc --exclude-table-data=pinecone.index_maintenance`와
  `docker/.env` 복사를 **둘 다** 하고, 두 파일을 같은 곳에 보관하라고 말한다.
- **복원**: 새로 설치한 빈 서버에서만 한다. `dropdb` 전에 `corpus.documents`가 0건인지 확인하고, 0건이 아니면
  멈추고 알린다. 순서는 가이드 14장(같은 `docker/.env` → `up -d postgres` → `dropdb --force` → `createdb` →
  `pg_restore --exit-on-error` → `up -d`).
- **업데이트**: 먼저 백업을 권한 뒤 `git pull`과 `docker compose -f docker/docker-compose.yml --profile app up -d --build`.
  끝나면 상태 점검과 검색 확인(8단계)을 다시 한다.
- **HTTPS**(다른 PC에서 쓸 때): 접속 이름이나 IP를 묻고 `docker/.env`에 `OPENRAG_HTTPS_HOST`, `OPENRAG_HTTPS_PORT`
  (기본 8443), `OPENRAG_PUBLIC_URL=https://<이름>:<포트>`를 넣은 뒤 오버레이로 `up -d`. 사내용 인증서면 루트
  인증서를 꺼내(가이드 5장) 접속 PC가 믿게 하는 방법(Claude Code는 `NODE_EXTRA_CA_CERTS`)을 알려 주고, init을 다시
  실행해 `https://` 연결 명령을 준다.

## 마무리 보고

짧게, 이 순서로:

1. 된 것: 서버 주소, 조직, 들어간 문서 수(실패가 있으면 파일과 이유), 연결한 도구.
2. 사용자가 할 일: 관리자 키 보관(키를 옮겨 적은 뒤 `docker/.openrag-init.json`은 지워도 된다고 알린다.
   마법사가 지우지는 않는다), `docker/.env` 백업, Claude Code/Codex 새로 시작한 뒤 "사내 문서에서
   찾아서 답해줘: …"처럼 묻기.
3. 문서 다루는 법 한 줄: 문서 폴더에 넣고·고치고·옮기고·지우면 30초 안에 반영된다(7장).
4. 멈추기·다시 켜기: `docker compose -f docker/docker-compose.yml --profile app stop` / `start`(14장).
