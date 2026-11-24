# OpenRAG 플러그인

OpenRAG 서버를 Claude Code와 Codex에 붙이는 플러그인이에요. 들어 있는 것:

| | Claude Code | Codex |
| --- | --- | --- |
| 검색·문서 읽기 MCP (`rag_search`, `rag_read_document` …) | 플러그인이 연결 | `connect.codex`로 따로 연결 |
| 설치·운영 마법사 | `/openrag:openrag-setup` | 스킬 `openrag:openrag-setup` |
| kordoc 문서 자동 저장 훅 | 켜짐(설정에서 끔) | 켜짐(`/hooks`에서 한 번 승인) |

서버를 띄우는 방법은 [사용 가이드](../../docs/guide.md) 3·4장을 보세요. 마법사에게 맡겨도 돼요.

## 설치

Claude Code:

```text
/plugin marketplace add schift-io/openrag
/plugin install openrag@openrag
```

설치할 때 서버 주소, API 키, 버킷을 물어요. 키는 OS 보안 저장소에 들어가요. 나중에 바꾸려면
`/plugin configure openrag@openrag`. 플러그인이 MCP를 연결하니 가이드 5장의 `claude mcp add`는 하지 않아요.

Codex:

```bash
codex plugin marketplace add schift-io/openrag
codex plugin add openrag@openrag
```

Codex 플러그인에는 키를 받는 설정이 없어요. MCP는 가이드 5장의 `connect.codex`로 연결하고, 훅은 셸 환경변수를
읽어요.

## 설정

| Claude Code 설정 | Codex 환경변수 | 기본값 | 뜻 |
| --- | --- | --- | --- |
| `url` | `OPENRAG_URL` | `http://127.0.0.1:8080` | 서버 주소. 끝에 `/`를 붙이지 않아요 |
| `api_key` | `OPENRAG_API_KEY` | 없음 | `orag_…` 키. 없으면 훅이 아무것도 안 해요 |
| `bucket` | `OPENRAG_BUCKET` | `docs` | kordoc 문서를 넣을 버킷 |
| `kordoc_autosave` | `OPENRAG_KORDOC_AUTOSAVE` | 켜짐 | `false`면 자동 저장 끔 |
| `docs_dir` | `OPENRAG_DOCS_DIR` | 없음 | 서버 문서 폴더에 바로 쓰는 PC면 그 경로. 그 안의 파일은 폴더 감시에 맡겨요 |

Claude Code 설정이 비어 있으면 같은 줄의 환경변수를 읽어요.

## kordoc 자동 저장

[kordoc](https://github.com/chrisryugj/kordoc) MCP가 문서를 쓰면(`generate_document`, `fill_form`,
`patch_document`, `convert_to_hwpx`, `place_seal`, `add_memo`, `redact_document`) 그 파일을 바로 OpenRAG에
올려요. 문서 폴더로 옮기지 않아도 바로 `rag_search`로 찾을 수 있어요.

- kordoc은 따로 연결해요: `npx -y kordoc setup`, 또는 `claude mcp add kordoc -- npx -y kordoc mcp` /
  `codex mcp add kordoc -- npx -y kordoc mcp`. MCP 이름에 `kordoc`이 들어 있으면 훅이 알아봐요.
- 올리는 종류: hwpx, hwp, md, pdf, docx, pptx, html, xlsx, csv.
- 같은 경로에 다시 쓰면 이전 문서를 바꿔요(검색에는 새 내용만 나와요). 내용이 같으면 그대로 둬요.
- 서버가 꺼져 있거나 키가 틀리면 경고 한 줄만 보이고 kordoc 결과는 그대로예요.
- 훅은 저장이 끝날 때까지 기다려요(최대 2분). 시험한 HWPX 문서에서는 0.3~3.6초 걸렸어요.

시험한 것(2026-11-24): Claude Code 2.1.331과 Codex 0.171.0 모두 kordoc 4.25.11로 HWPX를 만들자, 플러그인 훅이 그 파일을
저장했고 같은 세션의 `rag_search` 첫 결과가 그 문서였어요. 두 도구 모두 GitHub에서 설치한 사본이 시험한 파일과
같은 것도 확인했어요.

Windows 11에서는 Codex 0.172.0으로 GitHub에서 설치하면 스킬이 보였어요. 훅은 Codex가 Windows에서 실행하는
방식(`cmd /C`) 그대로 kordoc 결과 이벤트를 넣어 돌렸고, HWPX를 서버에 저장한 뒤 `rag_search` 첫 결과가 그 문서였어요.
Windows에서 kordoc과 모델을 실제로 돌린 세션, Windows의 Claude Code는 시험하지 못했어요. `codex plugin` 명령이 없다고
나오면 Codex가 오래된 판이에요(0.42.0에는 없었어요). 업데이트한 뒤 다시 하세요.

Codex용 매니페스트는 `.codex-plugin/plugin.json`이에요. 플러그인 루트에 `plugin.json`(Agent Plugins 형식)을 두면
Codex가 플러그인 훅을 읽지 않아요(Codex 0.171.0 소스 `core-plugins/src/loader.rs`). `mcpServers: {}`는 Claude Code용
`.mcp.json`을 Codex가 읽지 않게 막아요.
