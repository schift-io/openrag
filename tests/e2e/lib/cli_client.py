"""CLI subprocess runner and binary resolver for OpenRAG E2E tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CliError(RuntimeError):
    """Base error for OpenRAG CLI execution failures."""


class CliBinaryNotFoundError(CliError):
    """Raised when the requested OpenRAG executable cannot be located."""


class CliExecutionError(CliError):
    """Raised when CLI execution fails, times out, or returns invalid output."""

    def __init__(self, result: CliResult, message: str) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CliResult:
    """Captured result from a subprocess invocation."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float

    def json(self) -> Any:
        """Parse stdout as JSON, falling back to trailing JSON block if log lines precede it."""
        text = self.stdout.strip()
        if not text:
            raise CliExecutionError(
                self,
                f"Expected JSON stdout from {self.args!r}, got empty output (stderr={self.stderr[:300]!r})",
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: locate the last valid JSON object/array starting on a line boundary
        lines = text.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            stripped = lines[idx].lstrip()
            if stripped.startswith(("{", "[")):
                candidate = "\n".join(lines[idx:])
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        # Fallback: scan for last '{' or '[' in stdout
        decoder = json.JSONDecoder()
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in ("{", "["):
                try:
                    obj, end = decoder.raw_decode(text[idx:])
                    if text[idx + end :].strip() == "":
                        return obj
                except json.JSONDecodeError:
                    continue

        raise CliExecutionError(
            self,
            f"Failed to parse JSON from stdout of {self.args!r}: "
            f"stdout={self.stdout[:400]!r}, stderr={self.stderr[:300]!r}",
        )


# Superuser DSN -> app-role DSN for every ephemeral cluster started by the harness.
_APP_ROLE_DSNS: dict[str, str] = {}


def register_app_role_dsn(superuser_dsn: str, app_role_dsn: str) -> None:
    """Records that OpenRAG processes must reach `superuser_dsn`'s database as the app role."""
    _APP_ROLE_DSNS[superuser_dsn] = app_role_dsn


_HARNESS_CONFIG: Path | None = None


def harness_config_path() -> Path:
    """Harness-owned `config.toml` enabling local operator mode for spawned OpenRAG CLIs.

    Pointing `OPENRAG_CONFIG` here also keeps the developer's own
    `~/.config/openrag/config.toml` (and any api_key in it) out of the test run.
    """
    global _HARNESS_CONFIG
    if _HARNESS_CONFIG is None or not _HARNESS_CONFIG.is_file():
        cfg_dir = Path(tempfile.mkdtemp(prefix="openrag_e2e_cfg_"))
        cfg = cfg_dir / "config.toml"
        cfg.write_text("# OpenRAG E2E harness\noperator = true\n", encoding="utf-8")
        cfg.chmod(0o600)
        _HARNESS_CONFIG = cfg
    return _HARNESS_CONFIG


def openrag_process_env(env: dict[str, str]) -> dict[str, str]:
    """Prepares the environment of a spawned OpenRAG process.

    1. `OPENRAG_CONFIG` defaults to the harness config (operator mode on, no user config).
    2. RLS is actually enforced: tests keep handing the superuser DSN to OpenRAG
       (`DATABASE_URL=pg.dsn()`), but a superuser bypasses row-level security. OpenRAG's
       request path (`DATABASE_URL`) is switched to the NOSUPERUSER app role; worker/admin
       pools keep the superuser DSN for IAM lookups and migrations. Test-side
       seeding/inspection is unaffected.
    """
    if not env.get("OPENRAG_CONFIG"):
        env["OPENRAG_CONFIG"] = str(harness_config_path())
    raw = env.get("DATABASE_URL")
    if raw is None:
        return env
    app_dsn = _APP_ROLE_DSNS.get(raw)
    if app_dsn is None:
        return env
    env["DATABASE_URL"] = app_dsn
    env["WORKER_DATABASE_URL"] = raw
    env["ADMIN_DATABASE_URL"] = raw
    return env


_DEFAULT_TARGET_ROOT = Path(__file__).resolve().parents[3]


def issue_cli_code(org_id: str, principal_id: str, *, ttl_seconds: int | None = None) -> str:
    """Issues a one-time CLI authorization code through the real bootstrap path.

    Runs `openrag auth code` as the local database operator against the most recently
    started ephemeral cluster; the returned code is exchanged at
    `POST /v1/auth/cli/code-exchange` exactly like a user would.
    """
    if not _APP_ROLE_DSNS:
        raise RuntimeError("issue_cli_code needs a started PgEphemeralCluster")
    superuser_dsn = list(_APP_ROLE_DSNS)[-1]
    exe = resolve_binary(
        "openrag",
        explicit_bin=os.environ.get("OPENRAG_BIN") or None,
        target_root=_DEFAULT_TARGET_ROOT,
    )
    env = os.environ.copy()
    for var in ("OPENRAG_API_KEY", "OPENRAG_API_URL", "OPENRAG_PRINCIPAL_ID"):
        env.pop(var, None)
    env["DATABASE_URL"] = superuser_dsn
    env = openrag_process_env(env)
    cmd = [str(exe), "auth", "code", "--principal", str(principal_id), "--org", str(org_id)]
    if ttl_seconds is not None:
        cmd.extend(["--ttl", str(ttl_seconds)])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"`openrag auth code` failed ({proc.returncode}): {proc.stderr.strip()[:400]}")
    code = json.loads(proc.stdout).get("code")
    if not isinstance(code, str) or not code:
        raise RuntimeError(f"`openrag auth code` returned no code: {proc.stdout[:400]}")
    return code


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def resolve_binary(
    name: str = "openrag",
    *,
    explicit_bin: str | Path | None = None,
    target_root: Path | None = None,
) -> Path:
    """Resolve an executable in priority order: explicit -> target/debug -> target/release -> PATH."""
    checked: list[str] = []

    if explicit_bin is not None:
        explicit_path = Path(explicit_bin)
        candidate = explicit_path if name == "openrag" else explicit_path.parent / name
        checked.append(str(candidate))
        if _is_executable_file(candidate):
            return candidate.resolve()

    if target_root is not None:
        root = Path(target_root)
        for profile in ("debug", "release"):
            candidate = root / "target" / profile / name
            checked.append(str(candidate))
            if _is_executable_file(candidate):
                return candidate.resolve()

    which_hit = shutil.which(name)
    checked.append(f"PATH({name})")
    if which_hit is not None:
        which_path = Path(which_hit)
        if _is_executable_file(which_path):
            return which_path.resolve()

    raise CliBinaryNotFoundError(
        f"Binary '{name}' not found (checked {', '.join(checked)}). "
        "Build the Rust workspace (`cargo build`) or pass `--openrag-bin` / `OPENRAG_BIN`."
    )


def _decode_partial(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


class OpenRagCli:
    """Typed wrapper around the `openrag` and `openrag-mcp` CLI binaries."""

    def __init__(
        self,
        *,
        target_root: Path,
        explicit_bin: str | Path | None = None,
        default_timeout: float = 30.0,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self.target_root = Path(target_root)
        self.explicit_bin = Path(explicit_bin) if explicit_bin is not None else None
        self.default_timeout = default_timeout
        self.base_env: dict[str, str] = dict(base_env) if base_env else {}

    def resolve(self, name: str = "openrag") -> Path:
        return resolve_binary(
            name,
            explicit_bin=self.explicit_bin,
            target_root=self.target_root,
        )

    def require_binary(self, name: str = "openrag") -> Path:
        return self.resolve(name)

    def run(
        self,
        args: Sequence[str],
        *,
        binary_name: str = "openrag",
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        check: bool = False,
        expect_exit: int | Sequence[int] | None = None,
    ) -> CliResult:
        exe = self.require_binary(binary_name)
        cmd = [str(exe), *[str(a) for a in args]]
        merged_env = os.environ.copy()
        merged_env.update(self.base_env)
        if env:
            merged_env.update(env)
        merged_env = openrag_process_env(merged_env)

        effective_timeout = self.default_timeout if timeout is None else timeout
        effective_cwd = cwd if cwd is not None else (
            self.target_root if self.target_root.exists() else None
        )

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=merged_env,
                cwd=effective_cwd,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - start) * 1000.0
            partial = CliResult(
                args=cmd,
                returncode=-1,
                stdout=_decode_partial(exc.stdout),
                stderr=_decode_partial(exc.stderr),
                duration_ms=duration_ms,
            )
            raise CliExecutionError(
                partial,
                f"Command {cmd!r} timed out after {effective_timeout}s. "
                f"stdout={partial.stdout[:300]!r}, stderr={partial.stderr[:300]!r}",
            ) from exc

        duration_ms = (time.monotonic() - start) * 1000.0
        result = CliResult(
            args=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=duration_ms,
        )

        allowed_exits: tuple[int, ...] | None = None
        if expect_exit is not None:
            if isinstance(expect_exit, int):
                allowed_exits = (expect_exit,)
            else:
                allowed_exits = tuple(expect_exit)
        elif check:
            allowed_exits = (0,)

        if allowed_exits is not None and result.returncode not in allowed_exits:
            raise CliExecutionError(
                result,
                f"Command {cmd!r} exited with {result.returncode}, expected {allowed_exits}. "
                f"stdout={result.stdout[:400]!r}, stderr={result.stderr[:400]!r}",
            )

        return result

    def run_json(
        self,
        args: Sequence[str],
        *,
        binary_name: str = "openrag",
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
        expect_exit: int | Sequence[int] = (0,),
    ) -> tuple[CliResult, Any]:
        result = self.run(
            args,
            binary_name=binary_name,
            input_text=input_text,
            env=env,
            cwd=cwd,
            timeout=timeout,
            expect_exit=expect_exit,
        )
        return result, result.json()

    # --- Typed Subcommand Helpers ---

    def up(
        self,
        *,
        profile: str = "standard",
        extra_args: Sequence[str] = (),
        **kw: Any,
    ) -> CliResult:
        return self.run(["up", "--profile", profile, *extra_args], **kw)

    def ingest(
        self,
        path: Path | str,
        *,
        bucket: str,
        scope: str | None = None,
        auth_level: int | None = None,
        groups: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        mode: str | None = None,
        pipeline: Path | str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["ingest", str(path), "--bucket", bucket]
        if scope is not None:
            args.extend(["--scope", scope])
        if auth_level is not None:
            args.extend(["--auth-level", str(auth_level)])
        if groups:
            args.extend(["--groups", ",".join(groups)])
        if tags:
            args.extend(["--tags", ",".join(tags)])
        if mode is not None:
            args.extend(["--mode", mode])
        if pipeline is not None:
            args.extend(["--pipeline", str(pipeline)])
        return self.run(args, **kw)

    def index_engine(self, table: str, engine: str, **kw: Any) -> CliResult:
        return self.run(["index", "engine", table, "--set", engine], **kw)

    def index_mirror(self, table: str, to: str, **kw: Any) -> CliResult:
        return self.run(["index", "mirror", table, "--to", to], **kw)

    def index_status(self, **kw: Any) -> CliResult:
        return self.run(["index", "status"], **kw)

    def index_retry(
        self,
        *,
        quarantine_id: str | None = None,
        retry_all: bool = False,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["index", "retry"]
        if quarantine_id is not None:
            args.extend(["--id", quarantine_id])
        if retry_all:
            args.append("--all")
        return self.run(args, **kw)

    def memory_patch(
        self,
        memory_id: str,
        *,
        op: str,
        text: str | None = None,
        reason: str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["memory", "patch", memory_id, "--op", op]
        if text is not None:
            args.extend(["--text", text])
        if reason is not None:
            args.extend(["--reason", reason])
        return self.run(args, **kw)

    def memory_promote(
        self,
        memory_id: str,
        *,
        to: str,
        bucket: str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["memory", "promote", memory_id, "--to", to]
        if bucket is not None:
            args.extend(["--bucket", bucket])
        return self.run(args, **kw)

    def redact_run(
        self,
        *,
        doc_id: str | None = None,
        chunk_id: str | None = None,
        memory_id: str | None = None,
        wiki_id: str | None = None,
        pattern: str | None = None,
        vault_entry_id: str | None = None,
        scope: str = "org",
        redact_thumbnails: bool = True,
        hard_purge: bool = False,
        tombstone_delete: bool = False,
        reason: str = "e2e redaction",
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["redact", "run", "--scope", scope, "--reason", reason]
        if doc_id is not None:
            args.extend(["--doc", doc_id])
        if chunk_id is not None:
            args.extend(["--chunk", chunk_id])
        if memory_id is not None:
            args.extend(["--memory", memory_id])
        if wiki_id is not None:
            args.extend(["--wiki", wiki_id])
        if pattern is not None:
            args.extend(["--pattern", pattern])
        if vault_entry_id is not None:
            args.extend(["--vault-entry", vault_entry_id])
        if redact_thumbnails:
            args.append("--redact-thumbnails")
        if hard_purge:
            args.append("--hard-purge")
        if tombstone_delete:
            args.append("--tombstone-delete")
        return self.run(args, **kw)

    def wiki_import(
        self,
        directory: Path | str,
        *,
        bucket: str,
        scope: str | None = None,
        auth_level: int | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["wiki", "import", str(directory), "--bucket", bucket]
        if scope is not None:
            args.extend(["--scope", scope])
        if auth_level is not None:
            args.extend(["--auth-level", str(auth_level)])
        return self.run(args, **kw)

    def wiki_export(self, bucket: str, *, out_dir: Path | str, **kw: Any) -> CliResult:
        return self.run(["wiki", "export", bucket, "--out", str(out_dir)], **kw)

    def concept_link(
        self,
        source: str,
        target: str,
        *,
        canonical_name: str | None = None,
        lang: str | None = None,
        relation: str | None = None,
        concept_id: str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["concept", "link", source, target]
        if canonical_name is not None:
            args.extend(["--canonical", canonical_name])
        if lang is not None:
            args.extend(["--lang", lang])
        if relation is not None:
            args.extend(["--relation", relation])
        if concept_id is not None:
            args.extend(["--concept-id", concept_id])
        return self.run(args, **kw)

    def concept_unlink(
        self,
        source: str,
        target: str | None = None,
        *,
        concept_id: str | None = None,
        alias_id: str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["concept", "unlink", source]
        if target is not None:
            args.append(target)
        if concept_id is not None:
            args.extend(["--concept-id", concept_id])
        if alias_id is not None:
            args.extend(["--alias-id", alias_id])
        return self.run(args, **kw)

    def concept_list(
        self,
        *,
        query: str | None = None,
        concept_id: str | None = None,
        lang: str | None = None,
        json_output: bool = False,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["concept", "list"]
        if query is not None:
            args.extend(["--query", query])
        if concept_id is not None:
            args.extend(["--concept-id", concept_id])
        if lang is not None:
            args.extend(["--lang", lang])
        if json_output:
            args.append("--json")
        return self.run(args, **kw)

    def bench_run(
        self,
        *,
        golden_set: str,
        pipeline: Path | str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = ["bench", "run", "--golden-set", golden_set]
        if pipeline is not None:
            args.extend(["--pipeline", str(pipeline)])
        return self.run(args, **kw)

    def bench_optimize(
        self,
        *,
        golden_set: str,
        search_space: Path | str,
        require_zero_pii_leak: bool = True,
        require_zero_arn_violation: bool = True,
        export_best: Path | str | None = None,
        **kw: Any,
    ) -> CliResult:
        args: list[str] = [
            "bench",
            "optimize",
            "--golden-set",
            golden_set,
            "--search-space",
            str(search_space),
        ]
        if require_zero_pii_leak:
            args.append("--require-zero-pii-leak")
        if require_zero_arn_violation:
            args.append("--require-zero-arn-violation")
        if export_best is not None:
            args.extend(["--export-best", str(export_best)])
        return self.run(args, **kw)

    def pipeline_promote(self, yaml_path: Path | str, *, bucket: str, **kw: Any) -> CliResult:
        return self.run(["pipeline", "promote", str(yaml_path), "--bucket", bucket], **kw)

    def mcp_stdio_command(
        self,
        *,
        api_key: str | None = None,
        principal_id: str | None = None,
        prefer_standalone: bool = False,
    ) -> list[str]:
        cmd: list[str]
        if prefer_standalone:
            try:
                standalone = self.resolve("openrag-mcp")
                cmd = [str(standalone)]
            except CliBinaryNotFoundError:
                cmd = [str(self.require_binary("openrag")), "mcp", "--stdio"]
        else:
            cmd = [str(self.require_binary("openrag")), "mcp", "--stdio"]

        if api_key is not None:
            cmd.extend(["--api-key", api_key])
        if principal_id is not None:
            cmd.extend(["--principal", principal_id])
        return cmd
