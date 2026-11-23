"""Shared helpers and fixtures for Tier 3 pairwise cross-feature E2E tests."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import contextlib
import hashlib
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import psycopg

from conftest import fail_missing_target
from lib import OpenRagHttpClient, PgEphemeralCluster, WebClient


def insert_returning_one(pg: PgEphemeralCluster, sql: str) -> str:
    """Execute an INSERT/UPDATE/DELETE ... RETURNING <scalar> via psycopg without psql command tags."""
    pg.ensure_started()
    with psycopg.connect(pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        conn.commit()
    assert row is not None and row[0] is not None, f"Expected RETURNING row from SQL: {sql}"
    return str(row[0]).strip()


def ensure_migrated_all(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    """Start ephemeral PG17 if needed and ensure migrations 000..008 are applied."""
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations directory: {migrations_dir}")
    pg_ephemeral.ensure_started()
    has_orgs = pg_ephemeral.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;").strip()
    has_concepts = pg_ephemeral.execute_sql("SELECT to_regclass('wiki.concepts') IS NOT NULL;").strip()
    if has_orgs != "t":
        applied = pg_ephemeral.apply_migrations(
            migrations_dir,
            required_prefixes=[f"{i:03d}" for i in range(9)],
        )
        assert any(name.startswith("008_") for name in applied), applied
    elif has_concepts != "t":
        mig_008 = migrations_dir / "008_concepts_tags.sql"
        if not mig_008.is_file():
            fail_missing_target(f"Missing migration 008: {mig_008}")
        pg_ephemeral._run_psql_file_or_cmd(file_path=mig_008)
        pg_ephemeral.grant_app_role_privileges()
    return pg_ephemeral


@pytest.fixture(scope="module")
def migrated_pg(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    return ensure_migrated_all(pg_ephemeral, openrag_root)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unit_vec_str(hot_idx: int = 0, dim: int = 128) -> str:
    vec = [0.0] * dim
    vec[hot_idx % dim] = 1.0
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def seed_org_with_users(pg: PgEphemeralCluster, *, label: str = "T3") -> dict[str, str]:
    """Seed an org with 2 departments, owner(L6), peer(L6), dept_admin(L8), low(L3), other_dept(L6), admin(L9)."""
    org_id = str(uuid.uuid4())
    dept1_id = str(uuid.uuid4())
    dept2_id = str(uuid.uuid4())
    owner_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    dept_admin_id = str(uuid.uuid4())
    low_id = str(uuid.uuid4())
    other_dept_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    slug = f"{label.lower()}-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', '{label} Org {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
            ('{dept1_id}', '{org_id}', 'PolicyDept', 'policy'),
            ('{dept2_id}', '{org_id}', 'OpsDept', 'ops');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{owner_id}', '{org_id}', 'owner-{owner_id[:6]}@example.org', 'OwnerL6', 6, false, false),
            ('{peer_id}', '{org_id}', 'peer-{peer_id[:6]}@example.org', 'PeerL6', 6, false, false),
            ('{dept_admin_id}', '{org_id}', 'dadmin-{dept_admin_id[:6]}@example.org', 'DeptAdminL8', 8, true, false),
            ('{low_id}', '{org_id}', 'low-{low_id[:6]}@example.org', 'LowL3', 3, false, false),
            ('{other_dept_id}', '{org_id}', 'other-{other_dept_id[:6]}@example.org', 'OtherDeptL6', 6, false, false),
            ('{admin_id}', '{org_id}', 'admin-{admin_id[:6]}@example.org', 'AdminL9', 9, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept1_id}', '{owner_id}', 'member'),
            ('{org_id}', '{dept1_id}', '{peer_id}', 'member'),
            ('{org_id}', '{dept1_id}', '{dept_admin_id}', 'dept_admin'),
            ('{org_id}', '{dept1_id}', '{low_id}', 'member'),
            ('{org_id}', '{dept2_id}', '{other_dept_id}', 'member'),
            ('{org_id}', '{dept2_id}', '{admin_id}', 'dept_admin');
        """
    )
    return {
        "org_id": org_id,
        "slug": slug,
        "dept1_id": dept1_id,
        "dept2_id": dept2_id,
        "owner_id": owner_id,
        "peer_id": peer_id,
        "dept_admin_id": dept_admin_id,
        "low_id": low_id,
        "other_dept_id": other_dept_id,
        "admin_id": admin_id,
    }


def mint_api_key(
    api_client: OpenRagHttpClient,
    org_id: str,
    principal_id: str,
    *,
    tag: str = "t3",
    max_auth_level: int | None = None,
    scoped_arns: list[str] | None = None,
    scoped_group_ids: list[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "code": issue_cli_code(org_id, principal_id),
        "org_id": org_id,
        "principal_id": principal_id,
    }
    if max_auth_level is not None:
        payload["max_auth_level"] = max_auth_level
    if scoped_arns is not None:
        payload["scoped_arns"] = scoped_arns
    if scoped_group_ids is not None:
        payload["scoped_group_ids"] = scoped_group_ids
    resp = api_client.cli_code_exchange(payload)
    resp.assert_status((200, 201))
    data = resp.json()
    key = data.get("api_key") or data.get("token") or data.get("key")
    assert isinstance(key, str) and key, f"Expected minted api_key in {data!r}"
    return key


def extract_id(payload: Any, *keys: str) -> str:
    if isinstance(payload, dict):
        for key in keys:
            val = payload.get(key)
            if val:
                return str(val)
        for nested in ("document", "concept", "memory", "entry", "bucket", "page", "data", "job"):
            if isinstance(payload.get(nested), dict):
                try:
                    return extract_id(payload[nested], *keys)
                except AssertionError:
                    pass
    raise AssertionError(f"Could not extract any of {keys} from {payload!r}")


def extract_hits(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "hits", "matches", "documents", "memories"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def hit_doc_ids(hits: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for h in hits:
        for key in ("document_id", "doc_id", "id"):
            val = h.get(key)
            if val:
                out.add(str(val))
                break
    return out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def running_web_url(web_client: WebClient, api_url: str) -> Iterator[str]:
    env_url = os.environ.get("OPENRAG_WEB_URL", "").strip()
    if env_url:
        yield env_url.rstrip("/")
        return

    try:
        web_dir = web_client.require_web_dir()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    if not (web_dir / ".next").is_dir():
        web_client.run_build()

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        fail_missing_target("`pnpm` binary not found on PATH for running Next.js web server.")

    port = _free_port()
    env = os.environ.copy()
    env["NEXT_PUBLIC_OPENRAG_API_URL"] = api_url
    env["PORT"] = str(port)
    proc = subprocess.Popen(
        [pnpm, "--dir", str(web_dir), "start", "-p", str(port), "-H", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=web_dir,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 25.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2)
                pytest.fail(f"Next.js server exited early (rc={proc.returncode}): {out} {err}")
            try:
                with urllib.request.urlopen(f"{base_url}/app", timeout=1.5) as resp:
                    if 200 <= resp.status < 500:
                        break
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.25)
        else:
            pytest.fail(f"Next.js server did not become ready at {base_url}/app within 25s")
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
