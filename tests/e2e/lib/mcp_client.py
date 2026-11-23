"""MCP JSON-RPC 2.0 Stdio and Streamable HTTP clients for OpenRAG."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .cli_client import openrag_process_env
from .http_client import OpenRagHttpClient, parse_sse_bytes

_CONTENT_LENGTH_RE = re.compile(rb"^content-length:\s*(\d+)\s*$", re.IGNORECASE)

OPENRAG_MCP_TOOLS: tuple[str, ...] = (
    "rag_search",
    "rag_ask",
    "rag_ingest",
    "rag_get_page",
    "wiki_search",
    "wiki_get",
    "memory_search",
    "memory_patch",
    "memory_engage",
    "memory_reflect",
    "guard_check",
    "concept_link",
    "concept_resolve",
)


class McpError(RuntimeError):
    """Base exception for MCP client errors."""


class McpProtocolError(McpError):
    """Raised on framing errors, unexpected EOF, timeouts, or malformed JSON-RPC."""


class McpJsonRpcError(McpError):
    """Raised when the MCP server returns a JSON-RPC `error` object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message} (data={data!r})")
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True)
class McpToolResult:
    """Parsed MCP `tools/call` result envelope."""

    raw: dict[str, Any]
    is_error: bool
    content: list[dict[str, Any]]
    structured_content: Any

    @classmethod
    def from_raw(cls, result: dict[str, Any]) -> McpToolResult:
        content = result.get("content")
        return cls(
            raw=result,
            is_error=bool(result.get("isError", False)),
            content=list(content) if isinstance(content, list) else [],
            structured_content=result.get("structuredContent"),
        )

    @property
    def text(self) -> str:
        parts: list[str] = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                val = item.get("text")
                if isinstance(val, str):
                    parts.append(val)
        return "\n".join(parts)

    def json(self) -> Any:
        if self.structured_content is not None:
            return self.structured_content
        txt = self.text.strip()
        if not txt:
            raise McpProtocolError(f"MCP tool result has neither structuredContent nor text JSON: {self.raw!r}")
        return json.loads(txt)


class McpSessionMixin:
    """Shared MCP JSON-RPC 2.0 session methods and 13 OpenRAG tool wrappers."""

    def _next_id(self) -> int:
        raise NotImplementedError

    def _send_request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        raise NotImplementedError

    def _send_notification(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        raise NotImplementedError

    def initialize(
        self,
        *,
        client_name: str = "openrag-e2e",
        client_version: str = "0.1.0",
        protocol_version: str = "2025-03-26",
    ) -> dict[str, Any]:
        res = self._send_request(
            "initialize",
            {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
        )
        if not isinstance(res, dict):
            raise McpProtocolError(f"Expected dict from initialize, got {type(res).__name__}")
        return res

    def send_initialized(self) -> None:
        self._send_notification("notifications/initialized", {})

    def handshake(
        self,
        *,
        client_name: str = "openrag-e2e",
        client_version: str = "0.1.0",
        protocol_version: str = "2025-03-26",
    ) -> dict[str, Any]:
        init_res = self.initialize(
            client_name=client_name,
            client_version=client_version,
            protocol_version=protocol_version,
        )
        self.send_initialized()
        return init_res

    def ping(self) -> dict[str, Any]:
        res = self._send_request("ping", {})
        return dict(res) if isinstance(res, dict) else {"result": res}

    def tools_list(self) -> list[dict[str, Any]]:
        res = self._send_request("tools/list", {})
        if not isinstance(res, dict) or not isinstance(res.get("tools"), list):
            raise McpProtocolError(f"Invalid tools/list response: {res!r}")
        return list(res["tools"])

    def tools_call(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> McpToolResult:
        res = self._send_request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
        )
        if not isinstance(res, dict):
            raise McpProtocolError(f"Invalid tools/call response for {name}: {res!r}")
        return McpToolResult.from_raw(res)

    def raw_call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        return self._send_request(method, params)

    # --- 13 OpenRAG First-Class MCP Tool Wrappers ---

    def rag_search(
        self,
        query: str,
        *,
        bucket: str | None = None,
        top_k: int | None = None,
        spaces: Sequence[str] | None = None,
        fusion: str | None = None,
        tags_any: Sequence[str] | None = None,
        tags_all: Sequence[str] | None = None,
        tags_exclude: Sequence[str] | None = None,
        expand_tag_aliases: bool | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"query": query}
        if bucket is not None:
            args["bucket"] = bucket
        if top_k is not None:
            args["top_k"] = top_k
        if spaces is not None:
            args["spaces"] = list(spaces)
        if fusion is not None:
            args["fusion"] = fusion
        if tags_any is not None:
            args["tags_any"] = list(tags_any)
        if tags_all is not None:
            args["tags_all"] = list(tags_all)
        if tags_exclude is not None:
            args["tags_exclude"] = list(tags_exclude)
        if expand_tag_aliases is not None:
            args["expand_tag_aliases"] = expand_tag_aliases
        return self.tools_call("rag_search", args)

    def rag_ask(
        self,
        query: str,
        *,
        bucket: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"query": query}
        if bucket is not None:
            args["bucket"] = bucket
        if top_k is not None:
            args["top_k"] = top_k
        if mode is not None:
            args["mode"] = mode
        return self.tools_call("rag_ask", args)

    def rag_ingest(
        self,
        *,
        bucket: str,
        path: str | None = None,
        text: str | None = None,
        access_scope: str | None = None,
        allowed_group_ids: Sequence[str] | None = None,
        denied_group_ids: Sequence[str] | None = None,
        min_auth_level: int | None = None,
        title: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"bucket": bucket}
        if path is not None:
            args["path"] = path
        if text is not None:
            args["text"] = text
        if access_scope is not None:
            args["access_scope"] = access_scope
        if allowed_group_ids is not None:
            args["allowed_group_ids"] = list(allowed_group_ids)
        if denied_group_ids is not None:
            args["denied_group_ids"] = list(denied_group_ids)
        if min_auth_level is not None:
            args["min_auth_level"] = min_auth_level
        if title is not None:
            args["title"] = title
        if tags is not None:
            args["tags"] = list(tags)
        return self.tools_call("rag_ingest", args)

    def rag_get_page(
        self,
        *,
        page_id: str | None = None,
        document_id: str | None = None,
        page_no: int | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {}
        if page_id is not None:
            args["page_id"] = page_id
        if document_id is not None:
            args["document_id"] = document_id
        if page_no is not None:
            args["page_no"] = page_no
        return self.tools_call("rag_get_page", args)

    def wiki_search(
        self,
        query: str,
        *,
        bucket: str | None = None,
        top_k: int | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"query": query}
        if bucket is not None:
            args["bucket"] = bucket
        if top_k is not None:
            args["top_k"] = top_k
        return self.tools_call("wiki_search", args)

    def wiki_get(
        self,
        *,
        page_id: str | None = None,
        slug: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {}
        if page_id is not None:
            args["page_id"] = page_id
        if slug is not None:
            args["slug"] = slug
        return self.tools_call("wiki_get", args)

    def memory_search(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"query": query}
        if scope is not None:
            args["scope"] = scope
        if top_k is not None:
            args["top_k"] = top_k
        return self.tools_call("memory_search", args)

    def memory_patch(
        self,
        memory_id: str,
        op: str,
        *,
        text: str | None = None,
        reason: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"memory_id": memory_id, "op": op}
        if text is not None:
            args["text"] = text
        if reason is not None:
            args["reason"] = reason
        return self.tools_call("memory_patch", args)

    def memory_engage(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int | None = None,
        backend: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"query": query}
        if scope is not None:
            args["scope"] = scope
        if top_k is not None:
            args["top_k"] = top_k
        if backend is not None:
            args["backend"] = backend
        return self.tools_call("memory_engage", args)

    def memory_reflect(
        self,
        content: str,
        *,
        episode_id: str | None = None,
        scope: str | None = None,
        edge_type: str | None = None,
        target_memory_id: str | None = None,
        backend: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"content": content}
        if episode_id is not None:
            args["episode_id"] = episode_id
        if scope is not None:
            args["scope"] = scope
        if edge_type is not None:
            args["edge_type"] = edge_type
        if target_memory_id is not None:
            args["target_memory_id"] = target_memory_id
        if backend is not None:
            args["backend"] = backend
        return self.tools_call("memory_reflect", args)

    def guard_check(
        self,
        text: str,
        *,
        surface: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"text": text}
        if surface is not None:
            args["surface"] = surface
        return self.tools_call("guard_check", args)

    def concept_link(
        self,
        source: str,
        target: str,
        *,
        canonical_name: str | None = None,
        lang: str | None = None,
        relation: str | None = None,
        concept_id: str | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"source": source, "target": target}
        if canonical_name is not None:
            args["canonical_name"] = canonical_name
        if lang is not None:
            args["lang"] = lang
        if relation is not None:
            args["relation"] = relation
        if concept_id is not None:
            args["concept_id"] = concept_id
        return self.tools_call("concept_link", args)

    def concept_resolve(
        self,
        term: str,
        *,
        include_aliases: bool | None = None,
        use_semantic: bool | None = None,
        top_k: int | None = None,
        embedding: Sequence[float] | None = None,
    ) -> McpToolResult:
        args: dict[str, Any] = {"term": term}
        if include_aliases is not None:
            args["include_aliases"] = include_aliases
        if use_semantic is not None:
            args["use_semantic"] = use_semantic
        if top_k is not None:
            args["top_k"] = top_k
        if embedding is not None:
            args["embedding"] = list(embedding)
        return self.tools_call("concept_resolve", args)


class McpStdioClient(McpSessionMixin):
    """Real subprocess stdio JSON-RPC 2.0 MCP client supporting NDJSON and Content-Length framing."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        framing: Literal["ndjson", "content-length"] = "ndjson",
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.command = [str(c) for c in command]
        self.framing = framing
        self.timeout = timeout
        self._req_counter = 0
        self.notifications: list[dict[str, Any]] = []
        self._stderr_chunks: list[bytes] = []
        self._msg_queue: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue()
        self._closed = False

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        merged_env = openrag_process_env(merged_env)

        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=merged_env,
            cwd=cwd,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def stderr_text(self) -> str:
        return b"".join(self._stderr_chunks).decode("utf-8", errors="replace")

    def _read_stderr_loop(self) -> None:
        assert self.proc.stderr is not None
        try:
            while True:
                chunk = self.proc.stderr.read(4096)
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
        except Exception:
            pass

    @staticmethod
    def _read_exact(stream: Any, length: int) -> bytes:
        buf = bytearray()
        while len(buf) < length:
            chunk = stream.read(length - len(buf))
            if not chunk:
                raise McpProtocolError(
                    f"Unexpected EOF while reading Content-Length body ({len(buf)}/{length} bytes read)"
                )
            buf.extend(chunk)
        return bytes(buf)

    def _read_stdout_loop(self) -> None:
        assert self.proc.stdout is not None
        stream = self.proc.stdout
        try:
            while True:
                line = stream.readline()
                if not line:
                    self._msg_queue.put(None)
                    return

                stripped = line.strip()
                if not stripped:
                    continue

                match = _CONTENT_LENGTH_RE.match(stripped)
                if match:
                    content_length = int(match.group(1))
                    while True:
                        hdr = stream.readline()
                        if not hdr:
                            raise McpProtocolError("Unexpected EOF inside Content-Length headers")
                        if hdr in (b"\r\n", b"\n"):
                            break
                        more = _CONTENT_LENGTH_RE.match(hdr.strip())
                        if more:
                            content_length = int(more.group(1))
                    body = self._read_exact(stream, content_length)
                    msg = json.loads(body.decode("utf-8"))
                    self._msg_queue.put(msg)
                    continue

                if stripped.startswith((b"{", b"[")):
                    msg = json.loads(stripped.decode("utf-8"))
                    self._msg_queue.put(msg)
                    continue

                raise McpProtocolError(
                    f"Invalid MCP stdio frame line: {stripped[:200]!r}"
                )
        except BaseException as exc:
            self._msg_queue.put(exc)

    def _write_message(self, payload: Mapping[str, Any]) -> None:
        if self.proc.stdin is None or self._closed:
            raise McpProtocolError("MCP stdio stdin is closed")
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if self.framing == "content-length":
            frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        elif self.framing == "ndjson":
            frame = body + b"\n"
        else:
            raise ValueError(f"Unsupported framing mode: {self.framing!r}")
        try:
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise McpProtocolError(
                f"Broken pipe writing to MCP server {self.command!r}. stderr={self.stderr_text()[:400]!r}"
            ) from exc

    def _send_notification(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = dict(params)
        self._write_message(msg)

    def _send_request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        req_id = self._next_id()
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = dict(params)
        self._write_message(msg)

        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpProtocolError(
                    f"Timed out after {self.timeout}s waiting for MCP response to {method!r} "
                    f"(id={req_id}). stderr={self.stderr_text()[:400]!r}"
                )
            try:
                item = self._msg_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise McpProtocolError(
                    f"Timed out after {self.timeout}s waiting for MCP response to {method!r} "
                    f"(id={req_id}). stderr={self.stderr_text()[:400]!r}"
                ) from exc

            if item is None:
                raise McpProtocolError(
                    f"MCP stdio server EOF before responding to {method!r} (id={req_id}). "
                    f"rc={self.proc.poll()}, stderr={self.stderr_text()[:400]!r}"
                )
            if isinstance(item, BaseException):
                raise McpProtocolError(
                    f"MCP stdio read failure on {method!r}: {item}. stderr={self.stderr_text()[:400]!r}"
                ) from item

            if not isinstance(item, dict):
                raise McpProtocolError(f"Expected JSON-RPC object, got {item!r}")

            if item.get("id") != req_id:
                if "id" not in item or item.get("id") is None:
                    self.notifications.append(item)
                    continue
                # Unexpected out-of-order response id; ignore or raise if mismatched
                continue

            if "error" in item and item["error"] is not None:
                err = item["error"]
                if isinstance(err, dict):
                    raise McpJsonRpcError(
                        int(err.get("code", -32000)),
                        str(err.get("message", "Unknown JSON-RPC error")),
                        err.get("data"),
                    )
                raise McpJsonRpcError(-32000, str(err))

            return item.get("result")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2.0)
        self._stdout_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)

    def __enter__(self) -> McpStdioClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class McpHttpClient(McpSessionMixin):
    """Streamable HTTP `/mcp` JSON-RPC 2.0 client supporting JSON, SSE, and `Mcp-Session-Id`."""

    def __init__(self, http_client: OpenRagHttpClient) -> None:
        self.http_client = http_client
        self.session_id: str | None = None
        self._req_counter = 0

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _extract_rpc_response(self, raw_body: bytes, content_type: str, req_id: int) -> dict[str, Any]:
        ctype = content_type.lower()
        if "text/event-stream" in ctype:
            events = parse_sse_bytes(raw_body)
            for ev in events:
                payload = ev.json()
                if isinstance(payload, dict) and payload.get("id") == req_id:
                    return payload
            for ev in events:
                payload = ev.json()
                if isinstance(payload, dict) and ("result" in payload or "error" in payload):
                    return payload
            raise McpProtocolError(f"No JSON-RPC payload found in MCP SSE response: {raw_body[:400]!r}")

        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            raise McpProtocolError(
                f"Invalid JSON from /mcp ({content_type}): {raw_body[:400]!r}"
            ) from exc

        if isinstance(decoded, list):
            for entry in decoded:
                if isinstance(entry, dict) and entry.get("id") == req_id:
                    return entry
            if decoded and isinstance(decoded[0], dict):
                return decoded[0]
        if isinstance(decoded, dict):
            return decoded
        raise McpProtocolError(f"Unexpected JSON-RPC response shape from /mcp: {decoded!r}")

    def _send_notification(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = dict(params)
        resp = self.http_client.mcp_raw(msg, session_id=self.session_id)
        if resp.headers.get("mcp-session-id"):
            self.session_id = resp.headers["mcp-session-id"]
        if resp.status not in (200, 202, 204):
            raise McpProtocolError(
                f"HTTP /mcp notification {method!r} returned status {resp.status}: {resp.text[:300]!r}"
            )

    def _send_request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        req_id = self._next_id()
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = dict(params)

        resp = self.http_client.mcp_raw(msg, session_id=self.session_id)
        if resp.headers.get("mcp-session-id"):
            self.session_id = resp.headers["mcp-session-id"]

        if resp.status >= 400:
            # Check if server returned JSON-RPC error body with HTTP 4xx
            try:
                maybe_rpc = json.loads(resp.text)
                if isinstance(maybe_rpc, dict) and maybe_rpc.get("error"):
                    err = maybe_rpc["error"]
                    raise McpJsonRpcError(
                        int(err.get("code", -32000)),
                        str(err.get("message", "JSON-RPC HTTP error")),
                        err.get("data"),
                    )
            except McpJsonRpcError:
                raise
            except Exception:
                pass
            raise McpProtocolError(
                f"HTTP /mcp {method!r} failed with HTTP {resp.status}: {resp.text[:400]!r}"
            )

        rpc = self._extract_rpc_response(
            resp.body, resp.headers.get("content-type", "application/json"), req_id
        )
        if "error" in rpc and rpc["error"] is not None:
            err = rpc["error"]
            if isinstance(err, dict):
                raise McpJsonRpcError(
                    int(err.get("code", -32000)),
                    str(err.get("message", "Unknown JSON-RPC error")),
                    err.get("data"),
                )
            raise McpJsonRpcError(-32000, str(err))

        return rpc.get("result")
