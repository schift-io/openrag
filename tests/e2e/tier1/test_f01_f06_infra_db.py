"""Tier 1 isolated happy-path E2E tests for F01-F06 (Repo, Docker, YAML, DDL, Fallback, 128d)."""

from __future__ import annotations

import math
import re
import tomllib
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError
from lib.fixtures import make_pipeline_yaml
from lib.pg_harness import PgEphemeralCluster, PgPrincipalContext

pytestmark = pytest.mark.tier1

_EXCLUDED_SCAN_DIRS = {".git", "target", "node_modules", ".next", "dist", "__pycache__"}
_TEXT_EXTS = {
    ".rs",
    ".toml",
    ".sql",
    ".yml",
    ".yaml",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".py",
    ".env",
    ".example",
    ".lock",
    "",
}
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


def _ensure_migrated(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations directory: {migrations_dir}")
    pg_ephemeral.ensure_started()
    if getattr(pg_ephemeral, "_e2e_migrated_port", None) != pg_ephemeral.port:
        pg_ephemeral.apply_migrations(migrations_dir)
        setattr(pg_ephemeral, "_e2e_migrated_port", pg_ephemeral.port)
    return pg_ephemeral


@pytest.fixture(scope="module")
def migrated_pg(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    return _ensure_migrated(pg_ephemeral, openrag_root)


def _dynamic_insert(
    cur: psycopg.Cursor[Any],
    schema: str,
    table: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Insert a row into a spec-derived table while supplying defaults for extra NOT NULL columns."""
    cur.execute(
        """
        SELECT column_name, data_type, udt_name, is_nullable, column_default, is_generated
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    cols_meta = cur.fetchall()
    if not cols_meta:
        raise AssertionError(f"Table {schema}.{table} has no columns in information_schema")

    available = {row[0]: row for row in cols_meta}
    insert_data: dict[str, Any] = {}
    for key, val in values.items():
        if key in available:
            insert_data[key] = val

    for col_name, data_type, udt_name, is_nullable, col_default, is_generated in cols_meta:
        if col_name in insert_data:
            continue
        if is_generated == "ALWAYS" or is_nullable == "YES" or col_default is not None:
            continue
        if data_type == "uuid":
            insert_data[col_name] = str(uuid.uuid4())
        elif data_type in ("text", "character varying", "character"):
            insert_data[col_name] = f"e2e_{col_name}_{uuid.uuid4().hex[:8]}"
        elif data_type in ("integer", "smallint", "bigint"):
            insert_data[col_name] = 1
        elif data_type in ("double precision", "real", "numeric"):
            insert_data[col_name] = 0.0
        elif data_type == "boolean":
            insert_data[col_name] = False
        elif data_type in ("jsonb", "json"):
            insert_data[col_name] = "{}"
        elif data_type == "ARRAY":
            insert_data[col_name] = []
        elif data_type == "bytea":
            insert_data[col_name] = b"e2e"
        elif udt_name == "vector":
            insert_data[col_name] = _unit_vec_str(0)
        else:
            insert_data[col_name] = f"e2e_{uuid.uuid4().hex[:6]}"

    col_names = list(insert_data.keys())
    placeholders = []
    params: list[Any] = []
    for col in col_names:
        val = insert_data[col]
        udt_name = available[col][2]
        data_type = available[col][1]
        if udt_name == "vector":
            placeholders.append("%s::vector(128)")
            params.append(val)
        elif data_type in ("jsonb", "json") and isinstance(val, str):
            placeholders.append(f"%s::{data_type}")
            params.append(val)
        else:
            placeholders.append("%s")
            params.append(val)

    sql = (
        f'INSERT INTO "{schema}"."{table}" ({", ".join(col_names)}) '
        f'VALUES ({", ".join(placeholders)}) RETURNING *'
    )
    cur.execute(sql, params)
    row = cur.fetchone()
    assert cur.description is not None and row is not None
    return {desc[0]: row[idx] for idx, desc in enumerate(cur.description)}


def _unit_vec_str(hot_idx: int = 0, dim: int = 128) -> str:
    vec = [0.0] * dim
    vec[hot_idx % dim] = 1.0
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def _seed_tenant_context(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    principal_id = str(uuid.uuid4())
    bucket_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    fig_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    wiki_id = str(uuid.uuid4())
    ep_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())

    with psycopg.connect(pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "orgs",
                    {"id": org_id, "slug": f"org-{org_id[:8]}", "name": "E2E Org"},
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "departments",
                    {
                        "id": dept_id,
                        "org_id": org_id,
                        "slug": f"dept-{dept_id[:8]}",
                        "name": "E2E Dept",
                        "arn_path": f"org/{org_id[:8]}/dept/{dept_id[:8]}",
                    },
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": principal_id,
                        "org_id": org_id,
                        "email": f"u-{principal_id[:8]}@example.or.kr",
                        "display_name": "E2E User",
                        "auth_level": 5,
                        "can_unmask_pii": False,
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "buckets",
                    {
                        "id": bucket_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "owner_principal_id": principal_id,
                        "slug": f"bkt-{bucket_id[:8]}",
                        "name": "E2E Bucket",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:bucket:L1/{bucket_id}",
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "documents",
                    {
                        "id": doc_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "owner_principal_id": principal_id,
                        "title": "E2E Doc",
                        "source_uri": "doc.pdf",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:document:L1/{doc_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "pages",
                    {
                        "id": page_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "document_id": doc_id,
                        "owner_principal_id": principal_id,
                        "page_no": 1,
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:page:L1/{page_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                        "thumb_embedding": _unit_vec_str(0),
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "page_figures",
                    {
                        "id": fig_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "document_id": doc_id,
                        "page_id": page_id,
                        "owner_principal_id": principal_id,
                        "figure_idx": 0,
                        "figure_uri": "page-001-fig-1.png",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:figure:L1/{fig_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                        "figure_embedding": _unit_vec_str(1),
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": chunk_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "document_id": doc_id,
                        "page_id": page_id,
                        "owner_principal_id": principal_id,
                        "chunk_idx": 0,
                        "redacted_text": "Default seeded chunk",
                        "bm25_text": "Default seeded chunk",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:chunk:L1/{chunk_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                        "embedding": _unit_vec_str(2),
                    },
                )
                _dynamic_insert(
                    cur,
                    "wiki",
                    "pages",
                    {
                        "id": wiki_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "owner_principal_id": principal_id,
                        "slug": f"wiki-{wiki_id[:8]}",
                        "title": "E2E Wiki",
                        "body_md": "# Wiki\nContent",
                        "bm25_text": "Wiki Content",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:wiki:L1/{wiki_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                        "embedding": _unit_vec_str(3),
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "episodes",
                    {
                        "id": ep_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "owner_principal_id": principal_id,
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:episode:L1/{ep_id}",
                        "source_harness": "chat_ui",
                        "summary_md": "Seeded episode",
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "entries",
                    {
                        "id": mem_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "source_episode_id": ep_id,
                        "owner_principal_id": principal_id,
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:dept:dept:memory:L1/{mem_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                        "memory_kind": "fact",
                        "effective_text": "Seeded memory",
                        "bm25_text": "Seeded memory",
                        "embedding": _unit_vec_str(4),
                    },
                )

    return {
        "org_id": org_id,
        "dept_id": dept_id,
        "principal_id": principal_id,
        "bucket_id": bucket_id,
        "doc_id": doc_id,
        "page_id": page_id,
        "fig_id": fig_id,
        "chunk_id": chunk_id,
        "wiki_id": wiki_id,
        "ep_id": ep_id,
        "mem_id": mem_id,
    }


# ---------------------------------------------------------------------------
# F01 — Clean Apache-2.0 Repo & Zero Exclusions
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f01
def test_f01_apache2_license_and_readme_present(openrag_root: Path) -> None:
    license_path = openrag_root / "LICENSE"
    readme_path = openrag_root / "README.md"
    if not license_path.is_file():
        fail_missing_target(f"Missing LICENSE file at {license_path}")
    if not readme_path.is_file():
        fail_missing_target(f"Missing README.md file at {readme_path}")

    lic_text = license_path.read_text(encoding="utf-8")
    assert "Apache License" in lic_text
    assert "Version 2.0" in lic_text
    assert "http://www.apache.org/licenses/LICENSE-2.0" in lic_text

    readme_text = readme_path.read_text(encoding="utf-8")
    assert len(readme_text.strip()) > 100
    assert "openrag" in readme_text.lower()
    for client_name in ("Claude", "Cursor", "Codex"):
        assert client_name.lower() in readme_text.lower(), (
            f"README.md missing MCP client setup snippet/reference for {client_name}"
        )


@pytest.mark.tier1
@pytest.mark.f01
def test_f01_repo_directory_structure_complete(openrag_root: Path) -> None:
    required_files = [
        openrag_root / "LICENSE",
        openrag_root / "README.md",
        openrag_root / "Cargo.toml",
    ]
    required_dirs = [
        openrag_root / "configs",
        openrag_root / "docker",
        openrag_root / "migrations",
        openrag_root / "crates",
        openrag_root / "plugins",
        openrag_root / "web",
        openrag_root / "tests" / "e2e",
    ]
    for fpath in required_files:
        if not fpath.is_file():
            fail_missing_target(f"Missing mandated repository file: {fpath}")
    for dpath in required_dirs:
        if not dpath.is_dir():
            fail_missing_target(f"Missing mandated repository directory: {dpath}")


@pytest.mark.tier1
@pytest.mark.f01
def test_f01_rust_workspace_manifest_and_strict_lints(openrag_root: Path) -> None:
    cargo_toml = openrag_root / "Cargo.toml"
    if not cargo_toml.is_file():
        fail_missing_target(f"Missing workspace Cargo.toml at {cargo_toml}")

    data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    workspace = data.get("workspace")
    assert isinstance(workspace, dict), "Cargo.toml must define [workspace]"
    assert workspace.get("resolver") == "2"
    members = workspace.get("members")
    assert isinstance(members, list) and len(members) >= 1

    pkg_license = (workspace.get("package") or {}).get("license") or (data.get("package") or {}).get("license")
    assert pkg_license == "Apache-2.0", f"Expected Apache-2.0 license, got {pkg_license!r}"

    lints = workspace.get("lints") or {}
    rust_lints = lints.get("rust") or {}
    clippy_lints = lints.get("clippy") or {}

    def _lint_level(val: Any) -> str | None:
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            lvl = val.get("level")
            return str(lvl) if lvl is not None else None
        return None

    assert _lint_level(rust_lints.get("unsafe_code")) == "forbid"
    assert _lint_level(clippy_lints.get("unwrap_used")) in ("deny", "forbid")
    assert _lint_level(clippy_lints.get("expect_used")) in ("deny", "forbid")


@pytest.mark.tier1
@pytest.mark.f01
def test_f01_zero_excluded_schift_references_in_repo(openrag_root: Path) -> None:
    forbidden = ("schift-db", "schift_db", "schift-rag-ce", "schift_rag_ce")
    e2e_root = (openrag_root / "tests" / "e2e").resolve()
    hits: list[str] = []

    for path in openrag_root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == e2e_root or e2e_root in resolved.parents:
            continue
        if any(part in _EXCLUDED_SCAN_DIRS for part in path.relative_to(openrag_root).parts):
            continue
        if path.suffix.lower() not in _TEXT_EXTS and path.name not in ("Dockerfile", "Makefile"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lower = text.lower()
        for needle in forbidden:
            if needle in lower:
                hits.append(f"{path.relative_to(openrag_root)}:{needle}")

    assert hits == [], f"Forbidden excluded references found in repository: {hits}"


@pytest.mark.tier1
@pytest.mark.f01
def test_f01_openrag_cli_binary_builds_and_prints_help(openrag_cli: Any) -> None:
    try:
        res = openrag_cli.run(["--help"], expect_exit=0)
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    combined = (res.stdout + "\n" + res.stderr).lower()
    for subcmd in ("ingest", "index", "memory", "redact", "wiki", "bench", "pipeline", "mcp"):
        assert subcmd in combined, f"`openrag --help` missing core subcommand '{subcmd}'"


# ---------------------------------------------------------------------------
# F02 — Docker Standard & Airgap Stacks
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f02
def test_f02_postgres_dockerfile_extensions_and_lindera(openrag_root: Path) -> None:
    dockerfile = openrag_root / "docker" / "postgres" / "Dockerfile"
    if not dockerfile.is_file():
        fail_missing_target(f"Missing {dockerfile}")

    content = dockerfile.read_text(encoding="utf-8")
    lower = content.lower()
    arg_match = re.search(r"arg\s+pg_major\s*=\s*(17|18)\b", lower)
    resolved = lower
    if arg_match:
        ver = arg_match.group(1)
        resolved = resolved.replace("${pg_major}", ver).replace("$pg_major", ver)
    assert re.search(r"from\s+.*(17|18)", resolved), "Dockerfile must target PostgreSQL 17 or 18"
    assert "vector" in lower or "pgvector" in lower or "paradedb" in lower
    assert "pg_pinecone" in lower
    assert "pg_search" in lower
    assert "lindera" in lower or "mecab" in lower


@pytest.mark.tier1
@pytest.mark.f02
def test_f02_docker_compose_standard_services_and_healthchecks(openrag_root: Path) -> None:
    compose_path = openrag_root / "docker" / "docker-compose.yml"
    if not compose_path.is_file():
        fail_missing_target(f"Missing {compose_path}")

    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    services = data.get("services")
    assert isinstance(services, dict) and len(services) >= 3

    names_lower = {k.lower(): v for k, v in services.items()}
    pg_service = next((v for k, v in names_lower.items() if "postgres" in k or k == "db"), None)
    assert isinstance(pg_service, dict), f"Missing postgres service in {list(services.keys())}"
    assert "healthcheck" in pg_service, "PostgreSQL service must define a healthcheck"

    assert any("minio" in k or "s3" in k or "storage" in k for k in names_lower), (
        f"Expected object storage service in {list(services.keys())}"
    )
    assert any("web" in k or "console" in k for k in names_lower), (
        f"Expected web console service in {list(services.keys())}"
    )
    assert any("server" in k or "api" in k or "openrag" in k for k in names_lower), (
        f"Expected API server service in {list(services.keys())}"
    )


@pytest.mark.tier1
@pytest.mark.f02
def test_f02_docker_compose_airgap_local_services(openrag_root: Path) -> None:
    compose_path = openrag_root / "docker" / "docker-compose.airgap.yml"
    if not compose_path.is_file():
        fail_missing_target(f"Missing {compose_path}")

    raw_text = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    assert isinstance(data, dict)
    services = data.get("services")
    assert isinstance(services, dict) and len(services) >= 2

    lower = raw_text.lower()
    assert "airgap" in lower or "vllm" in lower or "local" in lower
    assert any("vllm" in k.lower() or "ocr" in k.lower() or "embed" in k.lower() for k in services) or (
        "vllm" in lower
    )


@pytest.mark.tier1
@pytest.mark.f02
def test_f02_docker_compose_ports_and_volumes_valid(openrag_root: Path) -> None:
    base_path = openrag_root / "docker" / "docker-compose.yml"
    airgap_path = openrag_root / "docker" / "docker-compose.airgap.yml"
    for fpath in (base_path, airgap_path):
        if not fpath.is_file():
            fail_missing_target(f"Missing {fpath}")

    base_data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert isinstance(base_data, dict)
    base_services = base_data.get("services")
    assert isinstance(base_services, dict) and base_services

    for fname, data, is_override in (
        ("docker-compose.yml", base_data, False),
        ("docker-compose.airgap.yml", yaml.safe_load(airgap_path.read_text(encoding="utf-8")), True),
    ):
        assert isinstance(data, dict)
        services = data.get("services")
        assert isinstance(services, dict) and services
        for svc_name, svc in services.items():
            assert isinstance(svc, dict), f"{fname}:{svc_name} must be a mapping"
            if not is_override or svc_name not in base_services:
                assert "image" in svc or "build" in svc, (
                    f"{fname}:{svc_name} must define 'image' or 'build'"
                )
            if "ports" in svc:
                assert isinstance(svc["ports"], list)
            if "volumes" in svc:
                assert isinstance(svc["volumes"], list)
            if "depends_on" in svc:
                assert isinstance(svc["depends_on"], (list, dict))


@pytest.mark.tier1
@pytest.mark.f02
def test_f02_cli_up_help_exposes_standard_and_airgap_profiles(openrag_cli: Any) -> None:
    try:
        res = openrag_cli.run(["up", "--help"], expect_exit=0)
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    combined = (res.stdout + "\n" + res.stderr).lower()
    assert "--profile" in combined or "profile" in combined
    assert "standard" in combined
    assert "airgap" in combined


# ---------------------------------------------------------------------------
# F03 — Declarative YAML Configs
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f03
def test_f03_pipeline_default_yaml_schema_and_defaults(openrag_root: Path) -> None:
    cfg_path = openrag_root / "configs" / "pipeline.default.yaml"
    if not cfg_path.is_file():
        fail_missing_target(f"Missing {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "version" in data
    assert isinstance(data.get("parse"), dict)
    thumb = data.get("thumbnail")
    assert isinstance(thumb, dict)
    assert thumb.get("format") == "webp"
    assert thumb.get("dim") == 128

    pii = data.get("pii")
    assert isinstance(pii, dict)
    assert pii.get("schift_v7") is True
    assert pii.get("capsule_enabled") is True
    assert isinstance(pii.get("guard"), dict)

    memory = data.get("memory")
    assert isinstance(memory, dict)
    assert memory.get("backend") == "openrag_pg"

    store = data.get("store")
    assert isinstance(store, dict)
    assert store.get("dim") == 128
    assert store.get("pg_pinecone_engine") in ("native", "pgvector", "remote")

    assert isinstance(data.get("retrieve"), dict)
    llm = data.get("llm")
    assert isinstance(llm, dict)
    assert int(llm.get("local_only_min_auth_level", 0)) >= 1


@pytest.mark.tier1
@pytest.mark.f03
def test_f03_pipeline_airgap_yaml_enforces_local_providers(openrag_root: Path) -> None:
    cfg_path = openrag_root / "configs" / "pipeline.airgap.yaml"
    if not cfg_path.is_file():
        fail_missing_target(f"Missing {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("profile") == "airgap"

    parse_cfg = data.get("parse") or {}
    ocr_engine = str(parse_cfg.get("ocr") or parse_cfg.get("ocr_engine") or "").lower()
    assert ocr_engine != "share"
    assert str(parse_cfg.get("default_ocr_mode", "")).lower() != "share"
    assert "local" in ocr_engine or "vllm" in ocr_engine

    llm_cfg = data.get("llm") or {}
    ext_provider = str(llm_cfg.get("external_provider", "")).lower()
    assert "openrouter" not in ext_provider
    assert "local" in ext_provider or "vllm" in ext_provider or ext_provider in ("none", "disabled")


@pytest.mark.tier1
@pytest.mark.f03
def test_f03_search_space_example_yaml_nodes_and_security_gates(openrag_root: Path) -> None:
    cfg_path = openrag_root / "configs" / "search_space.example.yaml"
    if not cfg_path.is_file():
        fail_missing_target(f"Missing {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("golden_set")
    assert data.get("strategy")
    gates = data.get("security_gates")
    assert isinstance(gates, dict)
    assert gates.get("require_zero_pii_leak") is True
    assert gates.get("require_zero_arn_violation") is True
    nodes = data.get("nodes")
    assert isinstance(nodes, dict) and len(nodes) >= 3


@pytest.mark.tier1
@pytest.mark.f03
def test_f03_studio_pipeline_get_and_valid_yaml_test_accepted(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    get_resp = api_client.studio_pipeline_get().assert_status(200).json()
    assert isinstance(get_resp, dict)

    yaml_text = sample_factory.pipeline_yaml("valid_std.yaml", profile="standard").read_text(
        encoding="utf-8"
    )
    test_resp = api_client.studio_pipeline_test({"pipeline_yaml": yaml_text}).assert_status(200).json()
    assert isinstance(test_resp, dict)
    assert "valid" in test_resp and test_resp["valid"] is True


@pytest.mark.tier1
@pytest.mark.f03
def test_f03_kumiho_memory_backend_config_accepted_by_pipeline_validator(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    yaml_text = make_pipeline_yaml(memory_backend="kumiho", memory_mirror_to="kumiho")
    resp = api_client.studio_pipeline_test({"pipeline_yaml": yaml_text}).assert_status(200).json()
    assert isinstance(resp, dict)
    assert "valid" in resp and resp["valid"] is True
    serialized = str(resp).lower()
    assert "kumiho" in serialized


# ---------------------------------------------------------------------------
# F04 — 10-Schema PG 17/18 DDL (000-007)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f04
def test_f04_migrations_000_to_007_exist_and_apply_cleanly(openrag_root: Path) -> None:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations directory: {migrations_dir}")

    with PgEphemeralCluster() as pg:
        applied = pg.apply_migrations(migrations_dir)
        prefixes = {name[:3] for name in applied}
        assert {f"{i:03d}" for i in range(8)}.issubset(prefixes), f"Applied={applied}"


@pytest.mark.tier1
@pytest.mark.f04
def test_f04_all_ten_schemas_created(migrated_pg: PgEphemeralCluster) -> None:
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = ANY(%s)",
                (list(_TEN_SCHEMAS),),
            )
            found = {row[0] for row in cur.fetchall()}
    assert found == set(_TEN_SCHEMAS), f"Missing schemas: {set(_TEN_SCHEMAS) - found}"


@pytest.mark.tier1
@pytest.mark.f04
def test_f04_all_core_tables_exist_across_ten_schemas(migrated_pg: PgEphemeralCluster) -> None:
    expected_tables: dict[str, set[str]] = {
        "iam": {
            "orgs",
            "departments",
            "principals",
            "dept_memberships",
            "groups",
            "group_members",
            "api_keys",
            "policies",
            "policy_attachments",
            "resource_group_bindings",
        },
        "vectors": {"index_bindings"},
        "corpus": {
            "buckets",
            "documents",
            "pages",
            "page_figures",
            "chunks",
            "chunk_edges",
        },
        "pii": {
            "detector_profiles",
            "vault_keys",
            "vault_entries",
            "chunk_spans",
            "audit_logs",
            "guard_policies",
            "guard_rules",
            "guard_events",
            "redaction_jobs",
            "redaction_events",
        },
        "memory": {"episodes", "entries", "patches"},
        "wiki": {"pages", "revisions", "edges", "page_sources", "sync_changes"},
        "indexing": {"outbox", "watermarks", "quarantine"},
        "gateway": {"providers", "routing_rules", "request_logs"},
        "plugins": {"connectors", "source_entries"},
        "eval": {
            "pipeline_configs",
            "golden_sets",
            "golden_items",
            "optimization_trials",
            "bench_runs",
            "bench_item_results",
            "search_feedback",
        },
    }

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE' AND table_schema = ANY(%s)
                """,
                (list(_TEN_SCHEMAS),),
            )
            present: dict[str, set[str]] = {s: set() for s in _TEN_SCHEMAS}
            for schema, table in cur.fetchall():
                present[schema].add(table)

    missing: dict[str, set[str]] = {}
    for schema, req in expected_tables.items():
        diff = req - present.get(schema, set())
        if diff:
            missing[schema] = diff
    assert not missing, f"Missing mandated tables: {missing}"


@pytest.mark.tier1
@pytest.mark.f04
def test_f04_rls_enabled_and_policies_attached_on_sensitive_tables(
    migrated_pg: PgEphemeralCluster,
) -> None:
    rls_tables = [
        ("corpus", "buckets"),
        ("corpus", "documents"),
        ("corpus", "pages"),
        ("corpus", "page_figures"),
        ("corpus", "chunks"),
        ("corpus", "chunk_edges"),
        ("wiki", "pages"),
        ("memory", "episodes"),
        ("memory", "entries"),
    ]
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            for schema, table in rls_tables:
                cur.execute(
                    """
                    SELECT c.relrowsecurity
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s
                    """,
                    (schema, table),
                )
                row = cur.fetchone()
                assert row is not None and row[0] is True, f"RLS not enabled on {schema}.{table}"

                cur.execute(
                    "SELECT COUNT(*) FROM pg_policies WHERE schemaname = %s AND tablename = %s",
                    (schema, table),
                )
                pol_count = cur.fetchone()
                assert pol_count is not None and pol_count[0] >= 1, (
                    f"No RLS policies attached to {schema}.{table}"
                )

            cur.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s",
                (migrated_pg.app_role,),
            )
            role_row = cur.fetchone()
            assert role_row == (False, False), f"Role {migrated_pg.app_role} flags={role_row}"


@pytest.mark.tier1
@pytest.mark.f04
def test_f04_idempotency_and_outbox_eval_ddl_constraints(migrated_pg: PgEphemeralCluster) -> None:
    seeded = _seed_tenant_context(migrated_pg)
    idem_key = f"idem-{uuid.uuid4().hex}"

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                p_cfg = _dynamic_insert(
                    cur,
                    "eval",
                    "pipeline_configs",
                    {"org_id": seeded["org_id"], "name": "cfg-1"},
                )
                g_set = _dynamic_insert(
                    cur,
                    "eval",
                    "golden_sets",
                    {"org_id": seeded["org_id"], "name": "gs-1", "axis": "retrieval"},
                )
                bench = _dynamic_insert(
                    cur,
                    "eval",
                    "bench_runs",
                    {
                        "org_id": seeded["org_id"],
                        "pipeline_config_id": p_cfg["id"],
                        "golden_set_id": g_set["id"],
                        "track": "track1_retrieval",
                        "pii_leak_count": 0,
                        "arn_violation_count": 0,
                    },
                )
                assert bench["security_gate_pass"] is True

                outbox_1 = _dynamic_insert(
                    cur,
                    "indexing",
                    "outbox",
                    {
                        "org_id": seeded["org_id"],
                        "target_table": "corpus.chunks",
                        "target_id": seeded["chunk_id"],
                        "op": "upsert_embed",
                        "status": "pending",
                        "idempotency_key": idem_key,
                    },
                )
                cur.execute(
                    "UPDATE indexing.outbox SET status = 'done' WHERE id = %s",
                    (outbox_1["id"],),
                )
                outbox_2 = _dynamic_insert(
                    cur,
                    "indexing",
                    "outbox",
                    {
                        "org_id": seeded["org_id"],
                        "target_table": "corpus.chunks",
                        "target_id": seeded["chunk_id"],
                        "op": "upsert_embed",
                        "status": "pending",
                        "idempotency_key": idem_key,
                    },
                )
                assert outbox_2["status"] == "pending"


# ---------------------------------------------------------------------------
# F05 — Capability-Guarded DB Fallback Mode
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f05
def test_f05_capability_probe_detects_installed_vs_fallback_extensions(
    migrated_pg: PgEphemeralCluster,
) -> None:
    caps = migrated_pg.probe_capabilities()
    assert caps["vector"] is True
    assert caps["pg_trgm"] is True
    assert caps["pgcrypto"] is True
    assert isinstance(caps["pg_pinecone"], bool)
    assert isinstance(caps["pg_search"], bool)


@pytest.mark.tier1
@pytest.mark.f05
def test_f05_fallback_vector_and_trgm_indexes_exist(migrated_pg: PgEphemeralCluster) -> None:
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            for schema, table in (
                ("corpus", "chunks"),
                ("corpus", "pages"),
                ("wiki", "pages"),
                ("memory", "entries"),
            ):
                cur.execute(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s",
                    (schema, table),
                )
                defs = " ".join(row[0].lower() for row in cur.fetchall())
                assert any(
                    tok in defs for tok in ("hnsw", "ivfflat", "pinecone", "vector_cosine_ops")
                ), f"Missing vector index on {schema}.{table}: {defs}"

            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'corpus' AND tablename = 'chunks'"
            )
            chunk_defs = " ".join(row[0].lower() for row in cur.fetchall())
            assert any(
                tok in chunk_defs for tok in ("gin_trgm_ops", "pg_search", "bm25", "tsvector", "gin")
            ), f"Missing lexical/trigram index on corpus.chunks: {chunk_defs}"


@pytest.mark.tier1
@pytest.mark.f05
def test_f05_fallback_cosine_vector_query_works_under_rls(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_tenant_context(migrated_pg)
    near_vec = _unit_vec_str(10)
    far_vec = _unit_vec_str(11)
    c1_id = str(uuid.uuid4())
    c2_id = str(uuid.uuid4())

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": c1_id,
                        "org_id": seeded["org_id"],
                        "dept_id": seeded["dept_id"],
                        "bucket_id": seeded["bucket_id"],
                        "document_id": seeded["doc_id"],
                        "page_id": seeded["page_id"],
                        "owner_principal_id": seeded["principal_id"],
                        "chunk_idx": 1,
                        "redacted_text": "Near vector chunk",
                        "bm25_text": "Near vector chunk",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{seeded['org_id']}:dept:dept:chunk:L1/{c1_id}",
                        "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:dept:L1",
                        "embedding": near_vec,
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": c2_id,
                        "org_id": seeded["org_id"],
                        "dept_id": seeded["dept_id"],
                        "bucket_id": seeded["bucket_id"],
                        "document_id": seeded["doc_id"],
                        "page_id": seeded["page_id"],
                        "owner_principal_id": seeded["principal_id"],
                        "chunk_idx": 2,
                        "redacted_text": "Far vector chunk",
                        "bm25_text": "Far vector chunk",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{seeded['org_id']}:dept:dept:chunk:L1/{c2_id}",
                        "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:dept:L1",
                        "embedding": far_vec,
                    },
                )

    principal = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=seeded["principal_id"],
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
    )
    rows = migrated_pg.query_as_principal(
        """
        SELECT id::text AS id, (embedding <=> %s::vector(128)) AS dist
        FROM corpus.chunks
        WHERE id = ANY(%s::uuid[])
        ORDER BY embedding <=> %s::vector(128) ASC
        LIMIT 2
        """,
        principal=principal,
        params=(near_vec, [c1_id, c2_id], near_vec),
    )
    assert len(rows) == 2
    assert rows[0]["id"] == c1_id
    assert abs(float(rows[0]["dist"])) < 1e-5
    assert float(rows[1]["dist"]) > 0.5


@pytest.mark.tier1
@pytest.mark.f05
def test_f05_fallback_korean_lexical_search_works_without_pg_search(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_tenant_context(migrated_pg)
    ko_id = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": ko_id,
                        "org_id": seeded["org_id"],
                        "dept_id": seeded["dept_id"],
                        "bucket_id": seeded["bucket_id"],
                        "document_id": seeded["doc_id"],
                        "page_id": seeded["page_id"],
                        "owner_principal_id": seeded["principal_id"],
                        "chunk_idx": 3,
                        "redacted_text": "제3조 국내 출장 여비 지급 기준",
                        "bm25_text": "제3조 국내 출장 여비 지급 기준",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{seeded['org_id']}:dept:dept:chunk:L1/{ko_id}",
                        "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:dept:L1",
                        "embedding": _unit_vec_str(15),
                    },
                )

    principal = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=seeded["principal_id"],
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
    )
    rows = migrated_pg.query_as_principal(
        """
        SELECT id::text AS id, similarity(bm25_text, %s) AS sim
        FROM corpus.chunks
        WHERE org_id = %s::uuid AND (bm25_text %% %s OR bm25_text ILIKE %s)
        ORDER BY sim DESC
        """,
        principal=principal,
        params=("출장 여비", seeded["org_id"], "출장 여비", "%출장 여비%"),
    )
    assert any(r["id"] == ko_id for r in rows)


@pytest.mark.tier1
@pytest.mark.f05
def test_f05_http_server_boots_and_searches_in_fallback_mode(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.health().assert_status(200)
    slug = f"f05-bkt-{uuid.uuid4().hex[:8]}"
    bkt = api_client.create_bucket(
        {"slug": slug, "name": "F05 Fallback Bucket", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bucket_id = str(bkt.get("id") or bkt.get("bucket_id") or slug)

    api_client.create_document(
        bucket_id,
        {
            "title": "Fallback Vector Search Doc",
            "text": "Deterministic fallback cosine and lexical search token alpha-f05",
            "access_scope": "org",
        },
    ).assert_status((200, 201))

    search = api_client.search_bucket(bucket_id, {"query": "alpha-f05", "top_k": 5}).assert_status(200).json()
    hits = search.get("results") or search.get("hits") or search.get("items") or []
    assert isinstance(hits, list) and len(hits) >= 1


# ---------------------------------------------------------------------------
# F06 — 128d Multi-Space Vector Columns & acl_scope_key
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f06
def test_f06_all_five_vector_columns_have_exact_128_dimensions(
    migrated_pg: PgEphemeralCluster,
) -> None:
    targets = [
        ("corpus", "pages", "thumb_embedding"),
        ("corpus", "page_figures", "figure_embedding"),
        ("corpus", "chunks", "embedding"),
        ("wiki", "pages", "embedding"),
        ("memory", "entries", "embedding"),
    ]
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            for schema, table, col in targets:
                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s
                      AND NOT a.attisdropped
                    """,
                    (schema, table, col),
                )
                row = cur.fetchone()
                assert row == ("vector(128)",), f"{schema}.{table}.{col} -> {row!r}"


@pytest.mark.tier1
@pytest.mark.f06
def test_f06_acl_scope_key_column_present_and_not_null_across_spaces(
    migrated_pg: PgEphemeralCluster,
) -> None:
    tables = [
        ("corpus", "documents"),
        ("corpus", "pages"),
        ("corpus", "page_figures"),
        ("corpus", "chunks"),
        ("wiki", "pages"),
        ("memory", "entries"),
    ]
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            for schema, table in tables:
                cur.execute(
                    """
                    SELECT is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = 'acl_scope_key'
                    """,
                    (schema, table),
                )
                row = cur.fetchone()
                assert row == ("NO",), f"{schema}.{table}.acl_scope_key -> {row!r}"


@pytest.mark.tier1
@pytest.mark.f06
def test_f06_vectors_index_bindings_defaults_128d_native_acl_scope_key(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_tenant_context(migrated_pg)
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'vectors' AND table_name = 'index_bindings'
                    """
                )
                defaults = {name: dflt for name, dflt in cur.fetchall()}
                assert "128" in str(defaults.get("dimension", ""))
                assert "native" in str(defaults.get("pg_pinecone_engine", ""))
                assert "acl_scope_key" in str(defaults.get("filter_column", ""))

                row = _dynamic_insert(
                    cur,
                    "vectors",
                    "index_bindings",
                    {
                        "org_id": seeded["org_id"],
                        "table_name": f"corpus.chunks_{uuid.uuid4().hex[:6]}",
                    },
                )
                assert row["dimension"] == 128
                assert row["pg_pinecone_engine"] == "native"
                assert row["filter_column"] == "acl_scope_key"


@pytest.mark.tier1
@pytest.mark.f06
def test_f06_128d_vectors_roundtrip_across_all_five_tables(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_tenant_context(migrated_pg)
    principal = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=seeded["principal_id"],
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
    )

    checks = [
        ("corpus.pages", "thumb_embedding", seeded["page_id"], _unit_vec_str(0)),
        ("corpus.page_figures", "figure_embedding", seeded["fig_id"], _unit_vec_str(1)),
        ("corpus.chunks", "embedding", seeded["chunk_id"], _unit_vec_str(2)),
        ("wiki.pages", "embedding", seeded["wiki_id"], _unit_vec_str(3)),
        ("memory.entries", "embedding", seeded["mem_id"], _unit_vec_str(4)),
    ]
    for tbl, col, row_id, vec in checks:
        rows = migrated_pg.query_as_principal(
            f"SELECT id::text AS id, ({col} <=> %s::vector(128)) AS dist FROM {tbl} WHERE id = %s::uuid",
            principal=principal,
            params=(vec, row_id),
        )
        assert len(rows) == 1, f"Expected 1 row from {tbl}"
        dist = float(rows[0]["dist"])
        assert math.isfinite(dist) and abs(dist) < 1e-5


@pytest.mark.tier1
@pytest.mark.f06
def test_f06_acl_scope_key_filter_separates_tenants_in_vector_queries(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_tenant_context(migrated_pg)
    vec = _unit_vec_str(25)
    c_a = str(uuid.uuid4())
    c_b = str(uuid.uuid4())
    scope_a = f"{seeded['org_id']}:{seeded['dept_id']}:dept:L3"
    scope_b = f"{seeded['org_id']}:other-dept:dept:L3"

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for cid, scope_key, idx in ((c_a, scope_a, 10), (c_b, scope_b, 11)):
                    _dynamic_insert(
                        cur,
                        "corpus",
                        "chunks",
                        {
                            "id": cid,
                            "org_id": seeded["org_id"],
                            "dept_id": seeded["dept_id"],
                            "bucket_id": seeded["bucket_id"],
                            "document_id": seeded["doc_id"],
                            "page_id": seeded["page_id"],
                            "owner_principal_id": seeded["principal_id"],
                            "chunk_idx": idx,
                            "redacted_text": f"Scope {scope_key}",
                            "bm25_text": f"Scope {scope_key}",
                            "access_scope": "dept",
                            "min_auth_level": 1,
                            "allowed_group_ids": [],
                            "arn": f"arn:openrag:{seeded['org_id']}:dept:dept:chunk:L3/{cid}",
                            "acl_scope_key": scope_key,
                            "embedding": vec,
                        },
                    )

    principal = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=seeded["principal_id"],
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
    )
    rows = migrated_pg.query_as_principal(
        """
        SELECT id::text AS id, acl_scope_key
        FROM corpus.chunks
        WHERE id = ANY(%s::uuid[]) AND acl_scope_key = %s
        ORDER BY embedding <=> %s::vector(128)
        """,
        principal=principal,
        params=([c_a, c_b], scope_a, vec),
    )
    assert [r["id"] for r in rows] == [c_a]
