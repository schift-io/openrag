"""Plugin inspection and invocation client for OpenRAG P1-P3 plugins."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cli_client import CliExecutionError, CliResult
from .mcp_client import McpStdioClient


def _detect_command_from_dir(
    plugin_dir: Path,
    *,
    preferred_scripts: Sequence[str] = (),
) -> list[str] | None:
    """Resolve a runnable command inside a plugin directory."""
    for rel in preferred_scripts:
        candidate = plugin_dir / rel
        if candidate.is_file():
            if candidate.suffix == ".py":
                return [sys.executable, str(candidate)]
            if candidate.suffix in (".js", ".mjs", ".cjs"):
                node = shutil.which("node") or "node"
                return [node, str(candidate)]
            if os.access(candidate, os.X_OK):
                return [str(candidate)]
            if candidate.suffix == ".sh":
                return ["bash", str(candidate)]

    pkg_json = plugin_dir / "package.json"
    if pkg_json.is_file():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            bin_field = pkg.get("bin")
            if isinstance(bin_field, str):
                target = plugin_dir / bin_field
                if target.is_file():
                    node = shutil.which("node") or "node"
                    return [node, str(target)]
            elif isinstance(bin_field, dict) and bin_field:
                first_target = plugin_dir / str(next(iter(bin_field.values())))
                if first_target.is_file():
                    node = shutil.which("node") or "node"
                    return [node, str(first_target)]
            main_field = pkg.get("main")
            if isinstance(main_field, str) and (plugin_dir / main_field).is_file():
                node = shutil.which("node") or "node"
                return [node, str(plugin_dir / main_field)]
        except Exception:
            pass

    for py_name in ("main.py", "cli.py", "sync.py", "hook.py", "server.py"):
        py_path = plugin_dir / py_name
        if py_path.is_file():
            return [sys.executable, str(py_path)]

    return None


class PluginClient:
    """Inspector and invoker for P1 `kordoc-mcp-bridge`, P2 `docufinder-connector`, P3 `ai-memory-hooks`."""

    def __init__(self, plugins_dir: Path, *, timeout: float = 30.0) -> None:
        self.plugins_dir = Path(plugins_dir)
        self.timeout = timeout

    def _require_plugin_dir(self, name: str) -> Path:
        pdir = self.plugins_dir / name
        if not pdir.is_dir():
            raise FileNotFoundError(
                f"Expected plugin directory missing: {pdir} (plugins_dir={self.plugins_dir})"
            )
        return pdir

    def _run_subprocess(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
        expect_exit: int | Sequence[int] | None = None,
    ) -> CliResult:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [str(c) for c in cmd],
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=merged_env,
                cwd=cwd,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - start) * 1000.0
            res = CliResult(
                args=[str(c) for c in cmd],
                returncode=-1,
                stdout=exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                duration_ms=duration_ms,
            )
            raise CliExecutionError(res, f"Plugin command {cmd!r} timed out after {self.timeout}s") from exc

        duration_ms = (time.monotonic() - start) * 1000.0
        result = CliResult(
            args=[str(c) for c in cmd],
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=duration_ms,
        )
        if expect_exit is not None:
            allowed = (expect_exit,) if isinstance(expect_exit, int) else tuple(expect_exit)
            if result.returncode not in allowed:
                raise CliExecutionError(
                    result,
                    f"Plugin command {cmd!r} exited {result.returncode}, expected {allowed}. "
                    f"stderr={result.stderr[:400]!r}",
                )
        return result

    @staticmethod
    def _collect_manifests(pdir: Path) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": str(pdir),
            "files": sorted(str(p.relative_to(pdir)) for p in pdir.rglob("*") if p.is_file()),
        }
        for manifest_name in ("package.json", ".mcp.json", "hooks/hooks.json"):
            mpath = pdir / manifest_name
            if mpath.is_file():
                try:
                    info[manifest_name] = json.loads(mpath.read_text(encoding="utf-8"))
                except Exception as exc:
                    info[manifest_name] = {"_parse_error": str(exc)}
        for text_name in ("pyproject.toml", "Cargo.toml", "README.md"):
            tpath = pdir / text_name
            if tpath.is_file():
                info[text_name] = tpath.read_text(encoding="utf-8", errors="replace")
        return info

    # --- P1: kordoc-mcp-bridge ---

    def inspect_kordoc_bridge(self) -> dict[str, Any]:
        pdir = self._require_plugin_dir("kordoc-mcp-bridge")
        return self._collect_manifests(pdir)

    def invoke_kordoc_mcp(
        self,
        command_override: Sequence[str] | None = None,
        *,
        api_url: str,
        api_key: str,
    ) -> McpStdioClient:
        pdir = self._require_plugin_dir("kordoc-mcp-bridge")
        cmd: list[str] | None = list(command_override) if command_override else None

        if cmd is None:
            mcp_json = pdir / ".mcp.json"
            if mcp_json.is_file():
                data = json.loads(mcp_json.read_text(encoding="utf-8"))
                servers = data.get("mcpServers") if isinstance(data, dict) else None
                if isinstance(servers, dict) and servers:
                    first = next(iter(servers.values()))
                    if isinstance(first, dict) and first.get("command"):
                        cmd = [str(first["command"]), *[str(a) for a in first.get("args", [])]]

        if cmd is None:
            cmd = _detect_command_from_dir(
                pdir,
                preferred_scripts=(
                    "src/server.py",
                    "server.py",
                    "src/index.js",
                    "dist/index.js",
                    "bin/kordoc-mcp-bridge",
                ),
            )

        if cmd is None:
            raise FileNotFoundError(
                f"Cannot resolve MCP stdio entrypoint in {pdir}"
            )

        return McpStdioClient(
            cmd,
            env={"OPENRAG_API_URL": api_url, "OPENRAG_API_KEY": api_key},
            cwd=pdir,
            timeout=self.timeout,
        )

    def trigger_kordoc_hook(
        self,
        file_path: Path,
        *,
        bucket: str,
        api_url: str,
        api_key: str,
        extra_args: Sequence[str] = (),
    ) -> CliResult:
        pdir = self._require_plugin_dir("kordoc-mcp-bridge")
        cmd = _detect_command_from_dir(
            pdir,
            preferred_scripts=(
                "bin/kordoc-hook",
                "scripts/write_hook.py",
                "scripts/write-hook.sh",
                "src/hook.py",
                "hook.py",
            ),
        )
        if cmd is None:
            raise FileNotFoundError(f"Cannot resolve kordoc hook entrypoint in {pdir}")
        full_cmd = [*cmd, str(file_path), "--bucket", bucket, *[str(a) for a in extra_args]]
        return self._run_subprocess(
            full_cmd,
            cwd=pdir,
            env={
                "OPENRAG_API_URL": api_url,
                "OPENRAG_API_KEY": api_key,
                "OPENRAG_BUCKET": bucket,
            },
        )

    # --- P2: docufinder-connector ---

    def inspect_docufinder_connector(self) -> dict[str, Any]:
        pdir = self._require_plugin_dir("docufinder-connector")
        info = self._collect_manifests(pdir)
        anything_cfg = pdir / "anything.config.json"
        if anything_cfg.is_file():
            info["anything.config.json"] = json.loads(anything_cfg.read_text(encoding="utf-8"))
        return info

    def run_docufinder_sync(
        self,
        watch_dir: Path,
        *,
        bucket: str,
        api_url: str,
        api_key: str,
        state_file: Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> CliResult:
        pdir = self._require_plugin_dir("docufinder-connector")
        cmd = _detect_command_from_dir(
            pdir,
            preferred_scripts=(
                "bin/docufinder-sync",
                "scripts/sync.py",
                "src/sync.py",
                "sync.py",
                "src/sync.js",
                "dist/sync.js",
            ),
        )
        if cmd is None:
            raise FileNotFoundError(f"Cannot resolve docufinder sync entrypoint in {pdir}")

        full_cmd = [
            *cmd,
            "--watch-dir",
            str(watch_dir),
            "--bucket",
            bucket,
            "--once",
        ]
        if state_file is not None:
            full_cmd.extend(["--state-file", str(state_file)])
        full_cmd.extend(str(a) for a in extra_args)

        return self._run_subprocess(
            full_cmd,
            cwd=pdir,
            env={
                "OPENRAG_API_URL": api_url,
                "OPENRAG_API_KEY": api_key,
                "OPENRAG_BUCKET": bucket,
            },
        )

    def get_docufinder_anything_config(self) -> dict[str, Any]:
        pdir = self._require_plugin_dir("docufinder-connector")
        for candidate in (
            pdir / "anything.config.json",
            pdir / "docufinder-anything.json",
            pdir / "configs" / "anything.json",
            pdir / "examples" / "anything.json",
        ):
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        for json_file in sorted(pdir.rglob("*.json")):
            if json_file.name in ("package.json", "package-lock.json", "tsconfig.json"):
                continue
            text = json_file.read_text(encoding="utf-8", errors="replace")
            if "/v1/chat/completions" in text or "chat/completions" in text:
                return json.loads(text)
        raise FileNotFoundError(
            f"No Docufinder Anything OpenAI-compatible config JSON found in {pdir}"
        )

    # --- P3: ai-memory-hooks ---

    def inspect_ai_memory_hooks(self) -> dict[str, Any]:
        pdir = self._require_plugin_dir("ai-memory-hooks")
        return self._collect_manifests(pdir)

    def invoke_memory_hook(
        self,
        event_name: str,
        payload: Mapping[str, Any],
        *,
        api_url: str,
        api_key: str,
        dry_run: bool = False,
    ) -> CliResult:
        pdir = self._require_plugin_dir("ai-memory-hooks")
        cmd: list[str] | None = None

        hooks_json = pdir / "hooks" / "hooks.json"
        if hooks_json.is_file():
            data = json.loads(hooks_json.read_text(encoding="utf-8"))
            hooks_map = data.get("hooks") if isinstance(data, dict) else None
            if isinstance(hooks_map, dict) and event_name in hooks_map:
                entries = hooks_map[event_name]
                if isinstance(entries, list) and entries:
                    first = entries[0]
                    hook_list = first.get("hooks") if isinstance(first, dict) else None
                    if isinstance(hook_list, list) and hook_list and isinstance(hook_list[0], dict):
                        raw_cmd = str(hook_list[0].get("command", "")).strip()
                        if raw_cmd:
                            raw_cmd = raw_cmd.replace("${CLAUDE_PLUGIN_ROOT}", str(pdir))
                            cmd = ["bash", "-lc", raw_cmd]

        if cmd is None:
            cmd = _detect_command_from_dir(
                pdir,
                preferred_scripts=(
                    "scripts/memory_hook.py",
                    "scripts/hook.py",
                    "scripts/auto-save.sh",
                    "src/hook.py",
                    "hook.py",
                ),
            )

        if cmd is None:
            raise FileNotFoundError(
                f"Cannot resolve hook command for event {event_name!r} in {pdir}"
            )

        env = {
            "OPENRAG_API_URL": api_url,
            "OPENRAG_API_KEY": api_key,
            "OPENRAG_HOOK_EVENT": event_name,
            "CLAUDE_PLUGIN_ROOT": str(pdir),
        }
        if dry_run:
            env["OPENRAG_HOOK_DRY_RUN"] = "1"

        return self._run_subprocess(
            cmd,
            cwd=pdir,
            env=env,
            input_text=json.dumps(dict(payload), ensure_ascii=False),
        )
