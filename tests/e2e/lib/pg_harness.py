"""Ephemeral PostgreSQL 17 cluster harness and Fail-Closed RLS test client."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psycopg
import psycopg.rows

from .cli_client import register_app_role_dsn

_PG_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SANDBOX_MARKERS = (
    "operation not permitted",
    "shmget",
    "permission denied",
    "could not create shared memory segment",
)
_TEN_SCHEMAS = (
    "iam",
    "vectors",
    "corpus",
    "pii",
    "memory",
    "wiki",
    "indexing",
    "gateway",
    "plugins",
    "eval",
)
DEFAULT_REQUIRED_MIGRATION_PREFIXES: tuple[str, ...] = tuple(f"{i:03d}" for i in range(8))


def discover_migrations(
    migrations_dir: Path,
    *,
    required_prefixes: Sequence[str] | None = DEFAULT_REQUIRED_MIGRATION_PREFIXES,
) -> list[Path]:
    """Discover ordered NNN_*.sql migrations and verify required numeric prefixes exist."""
    mdir = Path(migrations_dir)
    if not mdir.is_dir():
        raise FileNotFoundError(f"Migrations directory does not exist: {mdir}")

    sql_files = sorted(mdir.glob("[0-9][0-9][0-9]_*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No ordered migration files (000_*.sql..) found in {mdir}")

    if required_prefixes is not None:
        prefixes = {p.name[:3] for p in sql_files}
        required = set(required_prefixes)
        missing = sorted(required - prefixes)
        if missing:
            raise FileNotFoundError(
                f"Missing required migration prefixes {missing} in {mdir} "
                f"(found {[p.name for p in sql_files]})"
            )
    return sql_files


class PgSandboxError(RuntimeError):
    """Raised when PG17 cluster startup is blocked by OS/agent sandbox restrictions."""


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _pg_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_ident(ident: str) -> str:
    if not _PG_IDENT_RE.match(ident):
        raise ValueError(f"Unsafe PostgreSQL identifier: {ident!r}")
    return ident


@dataclass(frozen=True)
class PgPrincipalContext:
    """Caller security context bound to PostgreSQL `app.*` GUCs per transaction."""

    org_id: str
    principal_id: str
    auth_level: int = 1
    dept_ids: tuple[str, ...] = ()
    dept_admin_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    bypass_rls: bool = False
    role: str = "openrag_app"
    uuid_list_style: Literal["csv", "pg_array"] = "csv"

    def validate(self) -> None:
        if not (1 <= self.auth_level <= 10):
            raise ValueError(f"auth_level must be between 1 and 10, got {self.auth_level!r}")
        if self.bypass_rls:
            raise ValueError("PgPrincipalContext.bypass_rls must be False for Fail-Closed RLS tests")
        if self.role:
            _validate_ident(self.role)
        if self.uuid_list_style not in ("csv", "pg_array"):
            raise ValueError(f"Unsupported uuid_list_style: {self.uuid_list_style!r}")

    def _format_uuid_list(self, items: tuple[str, ...]) -> str:
        if not items:
            return ""
        if self.uuid_list_style == "pg_array":
            return "{" + ",".join(items) + "}"
        return ",".join(items)

    def to_guc_map(self) -> dict[str, str]:
        self.validate()
        return {
            "app.org_id": self.org_id,
            "app.principal_id": self.principal_id,
            "app.auth_level": str(self.auth_level),
            "app.dept_ids": self._format_uuid_list(self.dept_ids),
            "app.dept_admin_ids": self._format_uuid_list(self.dept_admin_ids),
            "app.group_ids": self._format_uuid_list(self.group_ids),
            "app.bypass_rls": "false",
        }


def build_rls_preamble_sql(principal: PgPrincipalContext) -> str:
    """Build deterministic SQL preamble setting non-superuser ROLE and all 7 `app.*` GUCs."""
    gucs = principal.to_guc_map()
    lines: list[str] = []
    if principal.role:
        lines.append(f"SET LOCAL ROLE {_validate_ident(principal.role)};")
    set_calls = [
        f"set_config({_sql_literal(k)}, {_sql_literal(v)}, true)"
        for k, v in gucs.items()
    ]
    lines.append("SELECT " + ", ".join(set_calls) + ";")
    return "\n".join(lines)


class PgEphemeralCluster:
    """Manages an ephemeral PostgreSQL 17 cluster via `initdb` / `pg_ctl` on 127.0.0.1."""

    def __init__(
        self,
        *,
        port: int = 0,
        host: str = "127.0.0.1",
        dbname: str = "openrag_test",
        superuser: str = "postgres",
        app_role: str = "openrag_app",
        data_dir: Path | None = None,
        keep_data: bool = False,
    ) -> None:
        self.port = port
        self.host = host
        self.dbname = _validate_ident(dbname)
        self.superuser = _validate_ident(superuser)
        self.app_role = _validate_ident(app_role)
        self._explicit_data_dir = Path(data_dir) if data_dir is not None else None
        self.data_dir: Path | None = self._explicit_data_dir
        self.keep_data = keep_data
        self._started = False
        self._log_file: Path | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    def dsn(self, user: str | None = None) -> str:
        if self.port <= 0:
            raise RuntimeError("PgEphemeralCluster has not been started yet (port unset)")
        eff_user = user or self.superuser
        return f"postgresql://{eff_user}@{self.host}:{self.port}/{self.dbname}"

    def connection_info(self) -> dict[str, Any]:
        self.ensure_started()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "superuser": self.superuser,
            "app_role": self.app_role,
            "dsn": self.dsn(self.superuser),
        }

    def ensure_started(self) -> PgEphemeralCluster:
        if not self._started:
            return self.start()
        return self

    @staticmethod
    def _check_sandbox_error(output: str, step: str) -> None:
        lower = output.lower()
        if any(marker in lower for marker in _SANDBOX_MARKERS):
            raise PgSandboxError(
                f"PostgreSQL {step} failed due to OS/agent sandbox restriction: {output.strip()}\n"
                "Re-run with `BypassSandbox=true` (required on macOS agent sandboxes for SysV shmget "
                "and loopback TCP 127.0.0.1)."
            )

    def start(self) -> PgEphemeralCluster:
        if self._started:
            return self

        for binary in ("initdb", "pg_ctl", "psql", "pg_isready"):
            if shutil.which(binary) is None:
                raise RuntimeError(
                    f"Required PostgreSQL binary '{binary}' not found on PATH. "
                    "Install PostgreSQL 17 (Homebrew postgresql@17) and add it to PATH."
                )

        if self.port == 0:
            self.port = _pick_free_port(self.host)

        if self._explicit_data_dir is not None:
            self.data_dir = self._explicit_data_dir
            self.data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.data_dir = Path(tempfile.mkdtemp(prefix="orag_pg_"))

        self._log_file = self.data_dir / "postgresql.log"
        env = _pg_env()

        init_proc = subprocess.run(
            [
                "initdb",
                "-D",
                str(self.data_dir),
                "-U",
                self.superuser,
                "--auth=trust",
                "--encoding=UTF8",
                "--locale=C",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if init_proc.returncode != 0:
            combined = f"{init_proc.stdout}\n{init_proc.stderr}"
            self._check_sandbox_error(combined, "initdb")
            raise RuntimeError(f"initdb failed (rc={init_proc.returncode}): {combined}")

        pg_opts = f"-p {self.port} -h {self.host} -k ''"
        ctl_proc = subprocess.run(
            [
                "pg_ctl",
                "-D",
                str(self.data_dir),
                "-l",
                str(self._log_file),
                "-o",
                pg_opts,
                "-w",
                "start",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if ctl_proc.returncode != 0:
            log_text = (
                self._log_file.read_text(encoding="utf-8", errors="replace")
                if self._log_file.exists()
                else ""
            )
            combined = f"{ctl_proc.stdout}\n{ctl_proc.stderr}\n{log_text}"
            self._check_sandbox_error(combined, "pg_ctl start")
            raise RuntimeError(f"pg_ctl start failed (rc={ctl_proc.returncode}): {combined}")

        ready_proc = subprocess.run(
            ["pg_isready", "-h", self.host, "-p", str(self.port), "-U", self.superuser],
            capture_output=True,
            text=True,
            env=env,
        )
        if ready_proc.returncode != 0:
            self._check_sandbox_error(f"{ready_proc.stdout}\n{ready_proc.stderr}", "pg_isready")
            raise RuntimeError(
                f"pg_isready failed (rc={ready_proc.returncode}): "
                f"{ready_proc.stdout} {ready_proc.stderr}"
            )

        self._started = True
        try:
            self._init_db_and_role()
        except Exception:
            self.stop()
            raise
        register_app_role_dsn(self.dsn(self.superuser), self.dsn(self.app_role))
        return self

    def _init_db_and_role(self) -> None:
        env = _pg_env()
        # Create test database against default `postgres` db
        db_check = subprocess.run(
            [
                "psql",
                "-h",
                self.host,
                "-p",
                str(self.port),
                "-U",
                self.superuser,
                "-d",
                "postgres",
                "-X",
                "-v",
                "ON_ERROR_STOP=1",
                "-tAc",
                f"SELECT 1 FROM pg_database WHERE datname = {_sql_literal(self.dbname)}",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if db_check.returncode != 0:
            raise RuntimeError(f"Failed checking database: {db_check.stderr}")
        if db_check.stdout.strip() != "1":
            self._run_psql_file_or_cmd(
                database="postgres",
                sql=f"CREATE DATABASE {self.dbname};",
            )

        self._run_psql_file_or_cmd(
            database=self.dbname,
            sql=(
                f"DO $$ BEGIN "
                f"IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {_sql_literal(self.app_role)}) THEN "
                f"CREATE ROLE {self.app_role} NOSUPERUSER NOCREATEDB NOCREATEROLE LOGIN; "
                f"END IF; END $$;"
            ),
        )

    def _run_psql_file_or_cmd(
        self,
        *,
        database: str | None = None,
        sql: str | None = None,
        file_path: Path | None = None,
        as_role: str | None = None,
    ) -> str:
        db = database or self.dbname
        user = as_role or self.superuser
        cmd = [
            "psql",
            "-h",
            self.host,
            "-p",
            str(self.port),
            "-U",
            user,
            "-d",
            db,
            "-X",
            "-A",
            "-t",
            "-v",
            "ON_ERROR_STOP=1",
        ]
        if file_path is not None:
            cmd.extend(["-f", str(file_path)])
        elif sql is not None:
            cmd.extend(["-c", sql])
        else:
            raise ValueError("Either sql or file_path must be provided")

        proc = subprocess.run(cmd, capture_output=True, text=True, env=_pg_env())
        if proc.returncode != 0:
            raise RuntimeError(
                f"psql failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    def apply_migrations(
        self,
        migrations_dir: Path,
        *,
        required_prefixes: Sequence[str] | None = DEFAULT_REQUIRED_MIGRATION_PREFIXES,
    ) -> list[str]:
        self.ensure_started()
        sql_files = discover_migrations(migrations_dir, required_prefixes=required_prefixes)

        applied: list[str] = []
        for sql_file in sql_files:
            self._run_psql_file_or_cmd(file_path=sql_file)
            applied.append(sql_file.name)

        self.grant_app_role_privileges()
        return applied

    def grant_app_role_privileges(self) -> None:
        self.ensure_started()
        grant_stmts: list[str] = [
            f"GRANT CONNECT ON DATABASE {self.dbname} TO {self.app_role};"
        ]
        for schema in _TEN_SCHEMAS:
            grant_stmts.append(
                f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = {_sql_literal(schema)}) THEN "
                f"EXECUTE 'GRANT USAGE ON SCHEMA {schema} TO {self.app_role}'; "
                f"EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {self.app_role}'; "
                f"EXECUTE 'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {schema} TO {self.app_role}'; "
                f"EXECUTE 'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA {schema} TO {self.app_role}'; "
                f"END IF; END $$;"
            )
        self.execute_sql("\n".join(grant_stmts))

    def probe_capabilities(self) -> dict[str, bool]:
        self.ensure_started()
        targets = ("vector", "pg_trgm", "pgcrypto", "pg_pinecone", "pg_search")
        caps: dict[str, bool] = {name: False for name in targets}
        with psycopg.connect(self.dsn(self.superuser)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
                    (list(targets),),
                )
                for (extname,) in cur.fetchall():
                    caps[str(extname)] = True
        return caps

    def execute_sql(self, sql: str, *, as_role: str | None = None) -> str:
        self.ensure_started()
        return self._run_psql_file_or_cmd(sql=sql, as_role=as_role)

    def query_as_principal(
        self,
        sql: str,
        *,
        principal: PgPrincipalContext,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        fetch: bool = True,
    ) -> list[dict[str, Any]]:
        self.ensure_started()
        gucs = principal.to_guc_map()

        with psycopg.connect(
            self.dsn(self.superuser),
            row_factory=psycopg.rows.dict_row,
        ) as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    if principal.role:
                        cur.execute(f"SET LOCAL ROLE {_validate_ident(principal.role)}")
                    for key, val in gucs.items():
                        cur.execute("SELECT set_config(%s, %s, true)", (key, val))
                    cur.execute(sql, params)
                    if not fetch or cur.description is None:
                        return []
                    return [dict(row) for row in cur.fetchall()]

    def stop(self) -> None:
        if self.data_dir is not None and self._started:
            subprocess.run(
                [
                    "pg_ctl",
                    "-D",
                    str(self.data_dir),
                    "-m",
                    "immediate",
                    "-w",
                    "stop",
                ],
                capture_output=True,
                text=True,
                env=_pg_env(),
            )
        self._started = False
        if not self.keep_data and self.data_dir is not None and self.data_dir.exists():
            shutil.rmtree(self.data_dir, ignore_errors=True)
            if self._explicit_data_dir is None:
                self.data_dir = None

    def __enter__(self) -> PgEphemeralCluster:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
