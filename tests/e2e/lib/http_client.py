"""HTTP REST, multipart, SSE client, and ephemeral OpenRAG server manager."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cli_client import CliBinaryNotFoundError, openrag_process_env, resolve_binary


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class HttpResponse:
    """Captured HTTP response without raising on 4xx/5xx."""

    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def assert_status(self, expected: int | Sequence[int]) -> HttpResponse:
        allowed = (expected,) if isinstance(expected, int) else tuple(expected)
        if self.status not in allowed:
            raise AssertionError(
                f"Expected HTTP status {allowed}, got {self.status}. "
                f"Body: {self.text[:600]!r}"
            )
        return self


@dataclass(frozen=True)
class SseEvent:
    """Single Server-Sent Event frame."""

    event: str | None
    data: str
    id: str | None = None

    def json(self) -> Any:
        stripped = self.data.strip()
        if not stripped or stripped == "[DONE]":
            return None
        return json.loads(stripped)


def extract_sse_text(events: Sequence[SseEvent]) -> str:
    """Aggregate assistant content deltas across OpenAI-compatible SSE chunks."""
    parts: list[str] = []
    for ev in events:
        payload = ev.json()
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                    continue
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    parts.append(message["content"])
                    continue
                if isinstance(first.get("text"), str):
                    parts.append(first["text"])
                    continue
        if isinstance(payload.get("text"), str):
            parts.append(payload["text"])
    return "".join(parts)


def extract_tag_facets(payload_or_resp: HttpResponse | Mapping[str, Any]) -> dict[str, int]:
    """Normalize `tag_facets` or `facets.tags` (dict or list of {tag|name, count}) to {tag: count}."""
    data: Any = (
        payload_or_resp.json()
        if isinstance(payload_or_resp, HttpResponse)
        else payload_or_resp
    )
    if not isinstance(data, Mapping):
        return {}
    raw = data.get("tag_facets")
    if raw is None and isinstance(data.get("facets"), Mapping):
        raw = data["facets"].get("tags")
    if isinstance(raw, Mapping):
        out: dict[str, int] = {}
        for k, v in raw.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[str(k)] = int(v)
        return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out = {}
        for item in raw:
            if isinstance(item, Mapping):
                key = item.get("tag") if item.get("tag") is not None else item.get("name")
                cnt = item.get("count")
                if key is not None and isinstance(cnt, (int, float)) and not isinstance(cnt, bool):
                    out[str(key)] = int(cnt)
        return out
    return {}


class EphemeralOpenRagServer:
    """Lifecycle manager for spawning an ephemeral OpenRAG HTTP server on 127.0.0.1."""

    def __init__(
        self,
        *,
        target_root: Path,
        database_url: str | None = None,
        explicit_bin: str | Path | None = None,
        mode: str = "offline",
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout: float = 20.0,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self.target_root = Path(target_root)
        self.database_url = database_url
        self.explicit_bin = Path(explicit_bin) if explicit_bin is not None else None
        self.mode = mode
        self.host = host
        self.port = port if port != 0 else _pick_free_port(host)
        self.startup_timeout = startup_timeout
        self.extra_env = dict(extra_env) if extra_env else {}

        self._proc: subprocess.Popen[bytes] | None = None
        self._log_dir: Path | None = None
        self._log_file: Path | None = None
        self._log_handle: Any = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def logs(self) -> str:
        if self._log_file and self._log_file.exists():
            return self._log_file.read_text(encoding="utf-8", errors="replace")
        return ""

    def _resolve_server_command(self) -> list[str]:
        try:
            server_bin = resolve_binary(
                "openrag-server",
                explicit_bin=self.explicit_bin,
                target_root=self.target_root,
            )
            return [str(server_bin)]
        except CliBinaryNotFoundError:
            openrag_bin = resolve_binary(
                "openrag",
                explicit_bin=self.explicit_bin,
                target_root=self.target_root,
            )
            return [str(openrag_bin), "serve"]

    def start(self) -> EphemeralOpenRagServer:
        if self._proc is not None and self._proc.poll() is None:
            return self

        cmd = self._resolve_server_command()
        self._log_dir = Path(tempfile.mkdtemp(prefix="orag_srv_"))
        self._log_file = self._log_dir / "server.log"
        self._log_handle = open(self._log_file, "wb")

        env = os.environ.copy()
        env.update(
            {
                "OPENRAG_BIND": f"{self.host}:{self.port}",
                "HOST": self.host,
                "PORT": str(self.port),
                "OPENRAG_MODE": self.mode,
            }
        )
        if self.database_url:
            env["DATABASE_URL"] = self.database_url
        env.update(self.extra_env)
        env = openrag_process_env(env)

        cwd = self.target_root if self.target_root.exists() else None
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )

        deadline = time.monotonic() + self.startup_timeout
        health_url = f"{self.base_url}/health"
        last_err: str = "no response yet"
        client = OpenRagHttpClient(self.base_url, timeout=1.5)

        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                self._close_log_handle()
                raise RuntimeError(
                    f"OpenRAG server {cmd!r} exited early with code {rc}.\nLogs:\n{self.logs()}"
                )
            try:
                resp = client.request("GET", "/health", timeout=1.0)
                if 200 <= resp.status < 300:
                    return self
                last_err = f"status={resp.status} body={resp.text[:200]}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(0.15)

        self.stop()
        raise RuntimeError(
            f"Timed out after {self.startup_timeout}s waiting for {health_url} ({last_err}).\n"
            f"Logs:\n{self.logs()}"
        )

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def stop(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            self._proc = None
        self._close_log_handle()
        if self._log_dir is not None and self._log_dir.exists():
            shutil.rmtree(self._log_dir, ignore_errors=True)
            self._log_dir = None

    def __enter__(self) -> EphemeralOpenRagServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def _parse_sse_lines(raw_lines: Sequence[bytes]) -> list[SseEvent]:
    events: list[SseEvent] = []
    ev_name: str | None = None
    ev_id: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal ev_name, ev_id, data_lines
        if data_lines or ev_name is not None or ev_id is not None:
            events.append(SseEvent(event=ev_name, data="\n".join(data_lines), id=ev_id))
        ev_name = None
        ev_id = None
        data_lines = []

    for raw in raw_lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""

        if field == "event":
            ev_name = value
        elif field == "data":
            data_lines.append(value)
        elif field == "id":
            ev_id = value

    flush()
    return events


def parse_sse_bytes(raw: bytes) -> list[SseEvent]:
    """Parse complete SSE payload bytes into SseEvent items."""
    return _parse_sse_lines(raw.splitlines(keepends=True))


class OpenRagHttpClient:
    """Stdlib HTTP/REST/multipart/SSE client for OpenRAG HTTP endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        _base_url_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._base_url_resolver = _base_url_resolver
        self.api_key = api_key
        self.default_headers: dict[str, str] = {
            k: v for k, v in (default_headers or {}).items()
        }
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        return self.ensure_ready()

    def ensure_ready(self) -> str:
        if self._base_url is None:
            if self._base_url_resolver is None:
                raise RuntimeError("OpenRagHttpClient has neither base_url nor resolver configured.")
            self._base_url = self._base_url_resolver().rstrip("/")
        return self._base_url

    def with_api_key(self, api_key: str) -> OpenRagHttpClient:
        return OpenRagHttpClient(
            self._base_url,
            api_key=api_key,
            default_headers=self.default_headers,
            timeout=self.timeout,
            _base_url_resolver=self._base_url_resolver,
        )

    def with_headers(self, headers: Mapping[str, str]) -> OpenRagHttpClient:
        merged = dict(self.default_headers)
        merged.update(headers)
        return OpenRagHttpClient(
            self._base_url,
            api_key=self.api_key,
            default_headers=merged,
            timeout=self.timeout,
            _base_url_resolver=self._base_url_resolver,
        )

    def _build_headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = dict(self.default_headers)
        if self.api_key and not any(k.lower() == "authorization" for k in headers):
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            for k, v in extra.items():
                headers[k] = v
        return headers

    def _open_connection(
        self,
        path: str,
        query: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[http.client.HTTPConnection, str]:
        root = self.ensure_ready()
        parsed = urllib.parse.urlsplit(root)
        base_path = parsed.path.rstrip("/")
        rel_path = path if path.startswith("/") else f"/{path}"
        full_path = f"{base_path}{rel_path}"

        if query:
            pairs: list[tuple[str, str]] = []
            for k, v in query.items():
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    for item in v:
                        pairs.append((str(k), str(item)))
                else:
                    pairs.append((str(k), str(v)))
            if pairs:
                sep = "&" if "?" in full_path else "?"
                full_path = f"{full_path}{sep}{urllib.parse.urlencode(pairs)}"

        eff_timeout = self.timeout if timeout is None else timeout
        if parsed.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                parsed.hostname or "127.0.0.1",
                parsed.port or 443,
                timeout=eff_timeout,
            )
        else:
            conn = http.client.HTTPConnection(
                parsed.hostname or "127.0.0.1",
                parsed.port or 80,
                timeout=eff_timeout,
            )
        return conn, full_path

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        req_headers = self._build_headers(headers)
        body: bytes | None = data
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            if not any(k.lower() == "content-type" for k in req_headers):
                req_headers["Content-Type"] = "application/json; charset=utf-8"

        conn, full_path = self._open_connection(path, query=query, timeout=timeout)
        try:
            conn.request(method.upper(), full_path, body=body, headers=req_headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            return HttpResponse(status=resp.status, headers=resp_headers, body=resp_body)
        finally:
            conn.close()

    def upload_multipart(
        self,
        path: str,
        *,
        files: Mapping[str, tuple[str, bytes, str]],
        fields: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        method: str = "POST",
    ) -> HttpResponse:
        boundary = f"----OpenRagE2E{uuid.uuid4().hex}"
        chunks: list[bytes] = []

        if fields:
            for field_name, field_value in fields.items():
                chunks.append(f"--{boundary}\r\n".encode("ascii"))
                chunks.append(
                    f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode("utf-8")
                )
                chunks.append(str(field_value).encode("utf-8"))
                chunks.append(b"\r\n")

        for field_name, (filename, file_bytes, content_type) in files.items():
            safe_filename = filename.replace('"', "_")
            chunks.append(f"--{boundary}\r\n".encode("ascii"))
            chunks.append(
                f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_filename}"\r\n'.encode(
                    "utf-8"
                )
            )
            chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            chunks.append(file_bytes)
            chunks.append(b"\r\n")

        chunks.append(f"--{boundary}--\r\n".encode("ascii"))
        payload = b"".join(chunks)

        req_headers = dict(headers or {})
        req_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        return self.request(method, path, data=payload, headers=req_headers)

    def stream_sse(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, str], list[SseEvent]]:
        req_headers = self._build_headers(headers)
        if not any(k.lower() == "accept" for k in req_headers):
            req_headers["Accept"] = "text/event-stream"

        body: bytes | None = None
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            if not any(k.lower() == "content-type" for k in req_headers):
                req_headers["Content-Type"] = "application/json; charset=utf-8"

        conn, full_path = self._open_connection(path, timeout=timeout)
        try:
            conn.request(method.upper(), full_path, body=body, headers=req_headers)
            resp = conn.getresponse()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            raw_lines: list[bytes] = []
            while True:
                line = resp.readline()
                if not line:
                    break
                raw_lines.append(line)
            return resp.status, resp_headers, _parse_sse_lines(raw_lines)
        finally:
            conn.close()

    # --- Typed Endpoint Methods ---

    def health(self) -> HttpResponse:
        return self.request("GET", "/health")

    def list_buckets(self) -> HttpResponse:
        return self.request("GET", "/v2/buckets")

    def create_bucket(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v2/buckets", json_body=dict(payload))

    def get_bucket(self, bucket_id: str) -> HttpResponse:
        return self.request("GET", f"/v2/buckets/{bucket_id}")

    def list_documents(self, bucket_id: str) -> HttpResponse:
        return self.request("GET", f"/v2/buckets/{bucket_id}/documents")

    def upload_document(
        self,
        bucket_id: str,
        *,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        fields: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        return self.upload_multipart(
            f"/v2/buckets/{bucket_id}/documents",
            files={"file": (filename, content, content_type)},
            fields=fields,
        )

    def create_document(self, bucket_id: str, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", f"/v2/buckets/{bucket_id}/documents", json_body=dict(payload))

    def delete_document(self, bucket_id: str, doc_id: str) -> HttpResponse:
        return self.request("DELETE", f"/v2/buckets/{bucket_id}/documents/{doc_id}")

    def search_bucket(
        self,
        bucket_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        query: str | None = None,
        top_k: int | None = None,
        tags_any: Sequence[str] | None = None,
        tags_all: Sequence[str] | None = None,
        tags_exclude: Sequence[str] | None = None,
        expand_tag_aliases: bool | None = None,
    ) -> HttpResponse:
        body: dict[str, Any] = dict(payload or {})
        if query is not None:
            body["query"] = query
        if top_k is not None:
            body["top_k"] = top_k
        if tags_any is not None:
            body["tags_any"] = list(tags_any)
        if tags_all is not None:
            body["tags_all"] = list(tags_all)
        if tags_exclude is not None:
            body["tags_exclude"] = list(tags_exclude)
        if expand_tag_aliases is not None:
            body["expand_tag_aliases"] = expand_tag_aliases
        return self.request("POST", f"/v2/buckets/{bucket_id}/search", json_body=body)

    def sync_documents(self, bucket: str, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("PUT", f"/v1/buckets/{bucket}/documents/sync", json_body=dict(payload))

    def chat_completions(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v1/chat/completions", json_body=dict(payload))

    def chat_completions_stream(
        self, payload: Mapping[str, Any]
    ) -> tuple[int, dict[str, str], list[SseEvent]]:
        body = dict(payload)
        body["stream"] = True
        return self.stream_sse("POST", "/v1/chat/completions", json_body=body)

    def ab_search(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v1/eval/ab-search", json_body=dict(payload))

    def cli_code_exchange(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v1/auth/cli/code-exchange", json_body=dict(payload))

    # --- Concepts & Tags CRUD ---

    def list_concepts(
        self, *, query: str | None = None, lang: str | None = None
    ) -> HttpResponse:
        q: dict[str, Any] = {}
        if query is not None:
            q["query"] = query
        if lang is not None:
            q["lang"] = lang
        return self.request("GET", "/v1/concepts", query=q or None)

    def create_concept(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v1/concepts", json_body=dict(payload))

    def get_concept(self, concept_id: str) -> HttpResponse:
        return self.request("GET", f"/v1/concepts/{concept_id}")

    def link_concepts(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v1/concepts/link", json_body=dict(payload))

    def resolve_concept(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/v1/concepts/resolve", json_body=dict(payload))

    def delete_concept(self, concept_id: str) -> HttpResponse:
        return self.request("DELETE", f"/v1/concepts/{concept_id}")

    def unlink_concept_alias(self, concept_id: str, alias_id: str) -> HttpResponse:
        return self.request("DELETE", f"/v1/concepts/{concept_id}/aliases/{alias_id}")

    def list_document_tags(self, bucket_id: str, doc_id: str) -> HttpResponse:
        return self.request("GET", f"/v2/buckets/{bucket_id}/documents/{doc_id}/tags")

    def add_document_tags(
        self, bucket_id: str, doc_id: str, tags: Sequence[str]
    ) -> HttpResponse:
        return self.request(
            "POST",
            f"/v2/buckets/{bucket_id}/documents/{doc_id}/tags",
            json_body={"tags": list(tags)},
        )

    def remove_document_tags(
        self, bucket_id: str, doc_id: str, tags: Sequence[str]
    ) -> HttpResponse:
        return self.request(
            "DELETE",
            f"/v2/buckets/{bucket_id}/documents/{doc_id}/tags",
            json_body={"tags": list(tags)},
        )

    def add_wiki_tags(self, page_id: str, tags: Sequence[str]) -> HttpResponse:
        return self.request(
            "POST", f"/v1/wiki/pages/{page_id}/tags", json_body={"tags": list(tags)}
        )

    def remove_wiki_tags(self, page_id: str, tags: Sequence[str]) -> HttpResponse:
        return self.request(
            "DELETE", f"/v1/wiki/pages/{page_id}/tags", json_body={"tags": list(tags)}
        )

    def add_memory_tags(self, memory_id: str, tags: Sequence[str]) -> HttpResponse:
        return self.request(
            "POST", f"/v1/memories/{memory_id}/tags", json_body={"tags": list(tags)}
        )

    def remove_memory_tags(self, memory_id: str, tags: Sequence[str]) -> HttpResponse:
        return self.request(
            "DELETE", f"/v1/memories/{memory_id}/tags", json_body={"tags": list(tags)}
        )

    def mcp_raw(
        self, rpc_payload: Mapping[str, Any], *, session_id: str | None = None
    ) -> HttpResponse:
        headers = {"Accept": "application/json, text/event-stream"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return self.request("POST", "/mcp", json_body=dict(rpc_payload), headers=headers)

    # --- Tier 1 User Console (/api/app/*) ---

    def app_chat(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/app/chat", json_body=dict(payload))

    def app_upload(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        fields: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        return self.upload_multipart(
            "/api/app/upload",
            files={"file": (filename, content, content_type)},
            fields=fields,
        )

    def app_page_thumbnail(self, page_id: str) -> HttpResponse:
        return self.request("GET", f"/api/app/pages/{page_id}/thumbnail")

    def app_feedback(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/app/feedback", json_body=dict(payload))

    def app_wiki(self, page_id: str | None = None) -> HttpResponse:
        path = "/api/app/wiki" if page_id is None else f"/api/app/wiki/{page_id}"
        return self.request("GET", path)

    def app_memory_list(self) -> HttpResponse:
        return self.request("GET", "/api/app/memory")

    def app_memory_patch(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/app/memory/patch", json_body=dict(payload))

    def app_memory_retract(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/app/memory/retract", json_body=dict(payload))

    def app_memory_promote(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/app/memory/promote", json_body=dict(payload))

    def app_connect(self) -> HttpResponse:
        return self.request("GET", "/api/app/connect")

    def app_concepts_request(
        self,
        subpath: str = "",
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        path = f"/api/app/concepts/{clean}" if clean else "/api/app/concepts"
        return self.request(
            method,
            path,
            json_body=dict(payload) if payload is not None else None,
        )

    # --- Tier 2 Ops Studio (/api/studio/*) ---

    def studio_pipeline_get(self) -> HttpResponse:
        return self.request("GET", "/api/studio/pipeline")

    def studio_pipeline_update(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("PUT", "/api/studio/pipeline", json_body=dict(payload))

    def studio_pipeline_test(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/studio/pipeline/test", json_body=dict(payload))

    def studio_index_engines(self) -> HttpResponse:
        return self.request("GET", "/api/studio/index/engines")

    def studio_index_set_engine(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("PUT", "/api/studio/index/engines", json_body=dict(payload))

    def studio_index_outbox(self) -> HttpResponse:
        return self.request("GET", "/api/studio/index/outbox")

    def studio_quarantine_retry(self, quarantine_id: str) -> HttpResponse:
        return self.request("POST", f"/api/studio/index/quarantine/{quarantine_id}/retry")

    def studio_iam_request(
        self,
        subpath: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        return self.request(
            method,
            f"/api/studio/iam/{clean}",
            json_body=dict(payload) if payload is not None else None,
        )

    def studio_iam_simulate(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/studio/iam/simulate", json_body=dict(payload))

    def studio_pii_request(
        self,
        subpath: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        return self.request(
            method,
            f"/api/studio/pii/{clean}",
            json_body=dict(payload) if payload is not None else None,
        )

    def studio_guard_test(self, payload: Mapping[str, Any]) -> HttpResponse:
        return self.request("POST", "/api/studio/pii/guard-test", json_body=dict(payload))

    def studio_gateway_request(
        self,
        subpath: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        return self.request(
            method,
            f"/api/studio/gateway/{clean}",
            json_body=dict(payload) if payload is not None else None,
        )

    def studio_eval_request(
        self,
        subpath: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        return self.request(
            method,
            f"/api/studio/eval/{clean}",
            json_body=dict(payload) if payload is not None else None,
        )

    def studio_wiki_conflicts(self) -> HttpResponse:
        return self.request("GET", "/api/studio/wiki/conflicts")

    def studio_plugins_connectors(self) -> HttpResponse:
        return self.request("GET", "/api/studio/plugins/connectors")

    def studio_concepts_request(
        self,
        subpath: str = "",
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        path = f"/api/studio/concepts/{clean}" if clean else "/api/studio/concepts"
        return self.request(
            method,
            path,
            json_body=dict(payload) if payload is not None else None,
        )

    def studio_knowledge_request(
        self,
        subpath: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> HttpResponse:
        clean = subpath.lstrip("/")
        return self.request(
            method,
            f"/api/studio/knowledge/{clean}",
            json_body=dict(payload) if payload is not None else None,
        )
