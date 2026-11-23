"""Next.js 2-Tier Console (`/app` + `/studio`) build, route, and zero-jargon verifier."""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path

from .cli_client import CliExecutionError, CliResult
from .http_client import HttpResponse, OpenRagHttpClient

APP_ROUTES: tuple[str, ...] = (
    "/app",
    "/app/chat",
    "/app/upload",
    "/app/wiki",
    "/app/memory",
    "/app/connect",
)

STUDIO_ROUTES: tuple[str, ...] = (
    "/studio",
    "/studio/index",
    "/studio/iam",
    "/studio/pii",
    "/studio/eval",
    "/studio/knowledge",
)

FORBIDDEN_APP_JARGON: tuple[str, ...] = (
    "vector",
    "BM25",
    "ARN",
    "pg_pinecone",
    "RaBitQ",
    "HNSW",
    "Outbox",
    "Quarantine",
)

_JARGON_RE = re.compile(
    r"(?<![A-Za-z0-9_])(vector|vectors|bm25|arn|arns|pg_pinecone|rabitq|hnsw|outbox|quarantine)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_USER_VISIBLE_ATTRS = frozenset({"placeholder", "title", "aria-label", "alt"})

_TSX_USER_FACING_PROPS: frozenset[str] = frozenset({
    "label",
    "placeholder",
    "title",
    "aria-label",
    "alt",
    "description",
    "badge",
    "text",
    "heading",
    "message",
    "tooltip",
    "emptyMessage",
    "emptyText",
    "helperText",
    "errorMessage",
    "caption",
    "subtitle",
    "confirmText",
    "cancelText",
})

_TSX_TAG_NAME_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$.]*")
_TSX_ATTR_NAME_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_.:-]*")
_TSX_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_TSX_NUMBER_RE = re.compile(r"(?:0[xXbBoO][0-9A-Fa-f_]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

_JSX_PRECEDE_KEYWORDS = frozenset({
    "return",
    "throw",
    "case",
    "default",
    "else",
    "yield",
    "await",
    "export",
    "do",
    "void",
    "typeof",
    "delete",
    "in",
    "of",
})

_JSX_PRECEDE_PUNCTS = frozenset({
    "(",
    "[",
    "{",
    ",",
    ";",
    ":",
    "?",
    "=",
    "=>",
    "!",
    "&&",
    "||",
    "??",
    "&",
    "|",
    "^",
    "+",
    "-",
    "*",
    "/",
    "%",
    "~",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "===",
    "!=",
    "!==",
})

_REGEX_PRECEDE_KEYWORDS = _JSX_PRECEDE_KEYWORDS | frozenset({"instanceof", "new"})
_COMPARISON_OPS = frozenset({"==", "===", "!=", "!=="})


class _TsxCopyExtractor:
    """State-machine TSX/JSX scanner extracting only user-visible JSX copy and UI props."""

    def __init__(self, source: str) -> None:
        self.src = source
        self.n = len(source)
        self.pos = 0
        self.chunks: list[str] = []
        self._tag_stack: list[str] = []

    def extract(self) -> list[str]:
        self._scan_code(stop_at_brace=False, collect_strings=False)
        return self.chunks

    def _skip_ws_and_comments(self) -> None:
        src = self.src
        n = self.n
        while self.pos < n:
            ch = src[self.pos]
            if ch.isspace():
                self.pos += 1
                continue
            if src.startswith("//", self.pos):
                nl = src.find("\n", self.pos + 2)
                self.pos = n if nl == -1 else nl + 1
                continue
            if src.startswith("/*", self.pos):
                end = src.find("*/", self.pos + 2)
                self.pos = n if end == -1 else end + 2
                continue
            break

    def _peek_next_op(self, pos: int) -> str:
        src = self.src
        n = self.n
        while pos < n:
            if src[pos].isspace():
                pos += 1
                continue
            if src.startswith("//", pos):
                nl = src.find("\n", pos + 2)
                pos = n if nl == -1 else nl + 1
                continue
            if src.startswith("/*", pos):
                end = src.find("*/", pos + 2)
                pos = n if end == -1 else end + 2
                continue
            break
        for op in ("===", "!==", "==", "!="):
            if src.startswith(op, pos):
                return op
        return ""

    def _scan_quoted_string(self) -> str:
        src = self.src
        n = self.n
        quote = src[self.pos]
        self.pos += 1
        out: list[str] = []
        while self.pos < n:
            ch = src[self.pos]
            if ch == "\\":
                if self.pos + 1 < n:
                    nxt = src[self.pos + 1]
                    if nxt == "n":
                        out.append("\n")
                    elif nxt == "t":
                        out.append("\t")
                    elif nxt == "r":
                        out.append("\r")
                    else:
                        out.append(nxt)
                    self.pos += 2
                else:
                    self.pos += 1
                continue
            if ch == quote:
                self.pos += 1
                break
            out.append(ch)
            self.pos += 1
        return "".join(out)

    def _scan_regex_literal(self) -> None:
        src = self.src
        n = self.n
        self.pos += 1  # consume opening '/'
        in_class = False
        while self.pos < n:
            ch = src[self.pos]
            if ch == "\\":
                self.pos = min(n, self.pos + 2)
                continue
            if ch == "\n":
                break
            if ch == "[" and not in_class:
                in_class = True
            elif ch == "]" and in_class:
                in_class = False
            elif ch == "/" and not in_class:
                self.pos += 1
                while self.pos < n and src[self.pos].isalpha():
                    self.pos += 1
                return
            self.pos += 1

    def _scan_template_literal(
        self,
        *,
        collect_strings: bool,
        in_fetch: bool,
        prev_op: str,
    ) -> None:
        src = self.src
        n = self.n
        self.pos += 1  # consume opening '`'
        parts: list[str] = []
        is_first_part = True
        skip_template_text = in_fetch or prev_op in _COMPARISON_OPS

        def _flush_part() -> None:
            nonlocal is_first_part, skip_template_text
            if not parts:
                is_first_part = False
                return
            text = "".join(parts)
            parts.clear()
            if is_first_part:
                is_first_part = False
                if text.lstrip().startswith("/api/"):
                    skip_template_text = True
                    return
            if collect_strings and not skip_template_text:
                # Look ahead only when template closes; checked at flush if needed
                stripped = text.strip()
                if stripped:
                    self.chunks.append(text)

        while self.pos < n:
            ch = src[self.pos]
            if ch == "\\":
                if self.pos + 1 < n:
                    nxt = src[self.pos + 1]
                    if nxt == "n":
                        parts.append("\n")
                    elif nxt == "t":
                        parts.append("\t")
                    elif nxt == "r":
                        parts.append("\r")
                    else:
                        parts.append(nxt)
                    self.pos += 2
                else:
                    self.pos += 1
                continue
            if src.startswith("${", self.pos):
                _flush_part()
                self.pos += 2
                self._scan_code(stop_at_brace=True, collect_strings=collect_strings)
                continue
            if ch == "`":
                self.pos += 1
                if self._peek_next_op(self.pos) in _COMPARISON_OPS:
                    skip_template_text = True
                _flush_part()
                return
            parts.append(ch)
            self.pos += 1
        _flush_part()

    def _has_closing_tag_ahead(self, tag_name: str, from_pos: int) -> bool:
        if tag_name == "":
            return "</>" in self.src[from_pos:]
        pattern = re.compile(rf"</{re.escape(tag_name)}\s*>")
        return pattern.search(self.src, from_pos) is not None

    def _skip_ts_type_args(self) -> bool:
        """Skip balanced `<...>` generic type arguments on a JSX component tag like `<Select<Row>>`."""
        src = self.src
        n = self.n
        if self.pos >= n or src[self.pos] != "<":
            return False
        depth = 0
        while self.pos < n:
            ch = src[self.pos]
            if ch in ('"', "'"):
                self._scan_quoted_string()
                continue
            if ch == "`":
                self._scan_template_literal(collect_strings=False, in_fetch=False, prev_op="")
                continue
            if src.startswith("//", self.pos):
                nl = src.find("\n", self.pos + 2)
                self.pos = n if nl == -1 else nl + 1
                continue
            if src.startswith("/*", self.pos):
                end = src.find("*/", self.pos + 2)
                self.pos = n if end == -1 else end + 2
                continue
            if ch == "<":
                depth += 1
                self.pos += 1
                continue
            if ch == ">":
                if self.pos > 0 and src[self.pos - 1] == "=":
                    # `=>` inside type arg
                    self.pos += 1
                    continue
                depth -= 1
                self.pos += 1
                if depth <= 0:
                    return True
                continue
            if ch in (";", "\n"):
                return False
            self.pos += 1
        return False

    def _try_parse_jsx_opening_tag(self) -> tuple[str, bool] | None:
        """Try to parse a JSX opening or self-closing tag at `self.pos`.

        Returns `(tag_name, is_self_closing)` (where `tag_name == ""` for fragments `<>`)
        and advances `self.pos` past `>` / `/>`, or restores `self.pos` and returns `None`
        if `<` is a TypeScript generic or comparison rather than JSX.
        """
        saved_pos = self.pos
        saved_chunks_len = len(self.chunks)
        src = self.src
        n = self.n

        if self.pos >= n or src[self.pos] != "<":
            return None

        if src.startswith("<>", self.pos):
            if self._has_closing_tag_ahead("", self.pos + 2):
                self.pos += 2
                return ("", False)
            return None

        if self.pos + 1 >= n:
            return None
        m_tag = _TSX_TAG_NAME_RE.match(src, self.pos + 1)
        if not m_tag:
            return None

        tag_name = m_tag.group(0)
        self.pos = m_tag.end()

        # Optional TypeScript generic type args on a JSX tag: <Table<Row> ...>
        if self.pos < n and src[self.pos] == "<" and not self._skip_ts_type_args():
            self.pos = saved_pos
            del self.chunks[saved_chunks_len:]
            return None

        while self.pos < n:
            self._skip_ws_and_comments()
            if self.pos >= n:
                break
            if src.startswith("/>", self.pos):
                self.pos += 2
                return (tag_name, True)
            if src[self.pos] == ">":
                if self._has_closing_tag_ahead(tag_name, self.pos + 1):
                    self.pos += 1
                    return (tag_name, False)
                self.pos = saved_pos
                del self.chunks[saved_chunks_len:]
                return None
            if src[self.pos] == "{":
                self.pos += 1
                self._scan_code(stop_at_brace=True, collect_strings=False)
                continue

            m_attr = _TSX_ATTR_NAME_RE.match(src, self.pos)
            if not m_attr:
                self.pos = saved_pos
                del self.chunks[saved_chunks_len:]
                return None

            attr_name = m_attr.group(0)
            if attr_name == "extends":
                self.pos = saved_pos
                del self.chunks[saved_chunks_len:]
                return None
            self.pos = m_attr.end()
            self._skip_ws_and_comments()

            if self.pos < n and src[self.pos] == "=":
                self.pos += 1
                self._skip_ws_and_comments()
                if self.pos >= n:
                    self.pos = saved_pos
                    del self.chunks[saved_chunks_len:]
                    return None
                is_ui_prop = attr_name in _TSX_USER_FACING_PROPS
                val_ch = src[self.pos]
                if val_ch in ('"', "'"):
                    val = self._scan_quoted_string()
                    if is_ui_prop and val.strip() and not val.lstrip().startswith("/api/"):
                        self.chunks.append(val)
                elif val_ch == "{":
                    self.pos += 1
                    self._scan_code(stop_at_brace=True, collect_strings=is_ui_prop)
                else:
                    self.pos = saved_pos
                    del self.chunks[saved_chunks_len:]
                    return None

        self.pos = saved_pos
        del self.chunks[saved_chunks_len:]
        return None

    def _scan_jsx_children(self, tag_name: str) -> None:
        src = self.src
        n = self.n
        self._tag_stack.append(tag_name)
        text_buf: list[str] = []

        def _flush_text() -> None:
            if not text_buf:
                return
            raw_text = "".join(text_buf)
            text_buf.clear()
            unescaped = html.unescape(raw_text).strip()
            if unescaped:
                self.chunks.append(unescaped)

        try:
            while self.pos < n:
                ch = src[self.pos]
                if ch == "{":
                    _flush_text()
                    self.pos += 1
                    self._scan_code(stop_at_brace=True, collect_strings=True)
                    continue
                if ch == "<":
                    if tag_name == "" and src.startswith("</>", self.pos):
                        _flush_text()
                        self.pos += 3
                        return
                    if src.startswith("</", self.pos):
                        m_close = re.match(r"</([A-Za-z_$][A-Za-z0-9_$.]*)\s*>", src[self.pos:])
                        if m_close:
                            _flush_text()
                            close_name = m_close.group(1)
                            if close_name == tag_name:
                                self.pos += m_close.end()
                                return
                            if close_name in self._tag_stack[:-1]:
                                # Unclosed inner tag recovery: let enclosing tag consume its closing tag
                                return
                            self.pos += m_close.end()
                            continue
                        if src.startswith("</>", self.pos):
                            _flush_text()
                            if "" in self._tag_stack[:-1]:
                                return
                            self.pos += 3
                            continue

                    opened = self._try_parse_jsx_opening_tag()
                    if opened is not None:
                        _flush_text()
                        child_tag, self_closing = opened
                        if not self_closing:
                            self._scan_jsx_children(child_tag)
                        continue

                    text_buf.append("<")
                    self.pos += 1
                    continue

                text_buf.append(ch)
                self.pos += 1
            _flush_text()
        finally:
            self._tag_stack.pop()

    def _scan_code(self, *, stop_at_brace: bool, collect_strings: bool) -> None:
        src = self.src
        n = self.n
        prev_kind = "SOF"
        prev_text = ""
        paren_is_fetch: list[bool] = []

        while self.pos < n:
            ch = src[self.pos]
            if ch.isspace():
                self.pos += 1
                continue

            if src.startswith("//", self.pos):
                nl = src.find("\n", self.pos + 2)
                self.pos = n if nl == -1 else nl + 1
                continue

            if src.startswith("/*", self.pos):
                end = src.find("*/", self.pos + 2)
                self.pos = n if end == -1 else end + 2
                continue

            if stop_at_brace and ch == "}":
                self.pos += 1
                return

            if ch == "{":
                self.pos += 1
                self._scan_code(stop_at_brace=True, collect_strings=collect_strings)
                prev_kind = "PUNCT"
                prev_text = "}"
                continue

            if ch == "(":
                is_fetch = prev_kind == "WORD" and prev_text == "fetch"
                paren_is_fetch.append(is_fetch)
                self.pos += 1
                prev_kind = "PUNCT"
                prev_text = "("
                continue

            if ch == ")":
                if paren_is_fetch:
                    paren_is_fetch.pop()
                self.pos += 1
                prev_kind = "PUNCT"
                prev_text = ")"
                continue

            if ch in ('"', "'"):
                val = self._scan_quoted_string()
                next_op = self._peek_next_op(self.pos)
                if (
                    collect_strings
                    and not any(paren_is_fetch)
                    and prev_text not in _COMPARISON_OPS
                    and next_op not in _COMPARISON_OPS
                    and val.strip()
                    and not val.lstrip().startswith("/api/")
                ):
                    self.chunks.append(val)
                prev_kind = "STRING"
                prev_text = val
                continue

            if ch == "`":
                self._scan_template_literal(
                    collect_strings=collect_strings,
                    in_fetch=any(paren_is_fetch),
                    prev_op=prev_text,
                )
                prev_kind = "STRING"
                prev_text = "`"
                continue

            if ch == "<":
                allows_jsx = (
                    prev_kind == "SOF"
                    or (prev_kind == "PUNCT" and prev_text in _JSX_PRECEDE_PUNCTS)
                    or (prev_kind == "WORD" and prev_text in _JSX_PRECEDE_KEYWORDS)
                )
                if allows_jsx:
                    opened = self._try_parse_jsx_opening_tag()
                    if opened is not None:
                        tag_name, self_closing = opened
                        if not self_closing:
                            self._scan_jsx_children(tag_name)
                        prev_kind = "JSX"
                        prev_text = ">"
                        continue

            if ch == "/":
                allows_regex = (
                    prev_kind == "SOF"
                    or (prev_kind == "PUNCT" and prev_text not in (")", "]", "++", "--"))
                    or (prev_kind == "WORD" and prev_text in _REGEX_PRECEDE_KEYWORDS)
                )
                if allows_regex:
                    self._scan_regex_literal()
                    prev_kind = "REGEX"
                    prev_text = "/"
                    continue

            m_word = _TSX_IDENT_RE.match(src, self.pos)
            if m_word:
                prev_kind = "WORD"
                prev_text = m_word.group(0)
                self.pos = m_word.end()
                continue

            m_num = _TSX_NUMBER_RE.match(src, self.pos)
            if m_num:
                prev_kind = "NUMBER"
                prev_text = m_num.group(0)
                self.pos = m_num.end()
                continue

            matched_punct = ""
            for op in ("===", "!==", "=>", "&&", "||", "??", "<=", ">=", "==", "!=", "++", "--"):
                if src.startswith(op, self.pos):
                    matched_punct = op
                    break
            if matched_punct:
                self.pos += len(matched_punct)
                prev_kind = "PUNCT"
                prev_text = matched_punct
                continue

            self.pos += 1
            prev_kind = "PUNCT"
            prev_text = ch


def extract_tsx_visible_copy(source: str) -> list[str]:
    """Extract user-visible JSX text, UI string props, and JSX string expressions from TSX source."""
    return _TsxCopyExtractor(source).extract()


class _VisibleHtmlTextExtractor(HTMLParser):
    """Extracts user-visible HTML text while ignoring <script>, <style>, <noscript>, and comments."""

    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._IGNORED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        for attr_name, attr_val in attrs:
            if attr_val and attr_name.lower() in _USER_VISIBLE_ATTRS:
                self._parts.append(attr_val)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


class WebClient:
    """Runner and verifier for `web/` Next.js TypeScript 2-tier console."""

    def __init__(self, target_root: Path, *, api_url: str | None = None) -> None:
        self.target_root = Path(target_root)
        self.api_url = api_url or "http://127.0.0.1:8080"

    def require_web_dir(self) -> Path:
        web_dir = self.target_root / "web"
        pkg_json = web_dir / "package.json"
        if not pkg_json.is_file():
            raise FileNotFoundError(
                f"Next.js package.json not found at {pkg_json}. "
                "Build Milestone M5 (`web/`) before running Web E2E checks."
            )
        return web_dir

    def _run_pnpm(self, args: list[str], *, timeout: float) -> CliResult:
        web_dir = self.require_web_dir()
        pnpm = shutil.which("pnpm")
        if pnpm is None:
            raise RuntimeError("`pnpm` binary not found on PATH.")
        cmd = [pnpm, "--dir", str(web_dir), *args]
        env = os.environ.copy()
        env.setdefault("NEXT_PUBLIC_OPENRAG_API_URL", self.api_url)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=web_dir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - start) * 1000.0
            res = CliResult(
                args=cmd,
                returncode=-1,
                stdout=exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                duration_ms=duration_ms,
            )
            raise CliExecutionError(res, f"pnpm command {cmd!r} timed out after {timeout}s") from exc

        duration_ms = (time.monotonic() - start) * 1000.0
        result = CliResult(
            args=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=duration_ms,
        )
        if result.returncode != 0:
            raise CliExecutionError(
                result,
                f"pnpm command {cmd!r} failed (rc={result.returncode}). "
                f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-500:]!r}",
            )
        return result

    def run_typecheck(self, timeout: float = 120.0) -> CliResult:
        return self._run_pnpm(["typecheck"], timeout=timeout)

    def run_lint(self, timeout: float = 120.0) -> CliResult:
        return self._run_pnpm(["lint"], timeout=timeout)

    def run_build(self, timeout: float = 240.0) -> CliResult:
        return self._run_pnpm(["build"], timeout=timeout)

    @staticmethod
    def extract_visible_html_text(html: str) -> str:
        parser = _VisibleHtmlTextExtractor()
        parser.feed(html)
        parser.close()
        return parser.get_text()

    @staticmethod
    def find_jargon_violations(text: str) -> list[str]:
        return [m.group(0) for m in _JARGON_RE.finditer(text)]

    def scan_app_tsx_copy(self) -> dict[str, list[str]]:
        web_dir = self.require_web_dir()
        candidates = [web_dir / "app" / "app", web_dir / "src" / "app" / "app"]
        app_dirs = [d for d in candidates if d.is_dir()]
        if not app_dirs:
            raise FileNotFoundError(
                f"Could not find Tier 1 `/app` Next.js route directory under {web_dir}"
            )

        violations: dict[str, list[str]] = {}
        for app_dir in app_dirs:
            for tsx_file in sorted(app_dir.rglob("*.tsx")):
                raw = tsx_file.read_text(encoding="utf-8", errors="replace")
                visible_chunks = extract_tsx_visible_copy(raw)
                found = self.find_jargon_violations(" ".join(visible_chunks))
                if found:
                    violations[str(tsx_file.relative_to(web_dir))] = found
        return violations

    def check_route_http(self, web_base_url: str, route: str) -> HttpResponse:
        client = OpenRagHttpClient(web_base_url, timeout=15.0)
        resp = client.request("GET", route)
        resp.assert_status(200)
        if route == "/app" or route.startswith("/app/"):
            visible = self.extract_visible_html_text(resp.text)
            bad = self.find_jargon_violations(visible)
            if bad:
                raise AssertionError(
                    f"Tier 1 route {route!r} leaked forbidden technical jargon in visible text: {bad}"
                )
        return resp

    @staticmethod
    def verify_connect_mcp_snippets(text_or_html: str) -> dict[str, bool]:
        lower = text_or_html.lower()
        return {
            "claude_desktop": "claude desktop" in lower or "claudedesktop" in lower or "claude_desktop_config" in lower,
            "claude_code": "claude code" in lower or "claude mcp add" in lower or ".mcp.json" in lower,
            "cursor": "cursor" in lower,
            "codex": "codex" in lower,
            "stdio": "mcp --stdio" in lower or "openrag-mcp" in lower,
            "http_mcp": "/mcp" in lower,
        }
