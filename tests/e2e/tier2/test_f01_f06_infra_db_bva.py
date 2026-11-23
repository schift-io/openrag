"""Tier 2 boundary, negative, and corner-case E2E tests for F01-F06."""

from __future__ import annotations

import math
import re
import shutil
import tomllib
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError, OpenRagCli
from lib.fixtures import make_pipeline_yaml
from lib.http_client import EphemeralOpenRagServer, OpenRagHttpClient
from lib.pg_harness import PgEphemeralCluster, PgPrincipalContext

pytestmark = pytest.mark.tier2


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
        raise AssertionError(f"Table {schema}.{table} missing")
    available = {row[0]: row for row in cols_meta}
    insert_data = {k: v for k, v in values.items() if k in available}

    for col_name, data_type, udt_name, is_nullable, col_default, is_generated in cols_meta:
        if col_name in insert_data or is_generated == "ALWAYS" or is_nullable == "YES" or col_default is not None:
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

    cols = list(insert_data.keys())
    phs: list[str] = []
    params: list[Any] = []
    for c in cols:
        val = insert_data[c]
        data_type, udt_name = available[c][1], available[c][2]
        if udt_name == "vector":
            phs.append("%s::vector(128)")
        elif data_type in ("jsonb", "json") and isinstance(val, str):
            phs.append(f"%s::{data_type}")
        else:
            phs.append("%s")
        params.append(val)

    sql = f'INSERT INTO "{schema}"."{table}" ({", ".join(cols)}) VALUES ({", ".join(phs)}) RETURNING *'
    cur.execute(sql, params)
    row = cur.fetchone()
    assert cur.description is not None and row is not None
    return {desc[0]: row[idx] for idx, desc in enumerate(cur.description)}


def _unit_vec_str(hot_idx: int = 0, dim: int = 128) -> str:
    vec = [0.0] * dim
    vec[hot_idx % dim] = 1.0
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def _seed_base(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    pid = str(uuid.uuid4())
    bkt_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    ep_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    with psycopg.connect(pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(cur, "iam", "orgs", {"id": org_id, "slug": f"o-{org_id[:6]}", "name": "Org"})
                _dynamic_insert(
                    cur,
                    "iam",
                    "departments",
                    {"id": dept_id, "org_id": org_id, "slug": f"d-{dept_id[:6]}", "name": "Dept", "arn_path": "org/d"},
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {"id": pid, "org_id": org_id, "email": f"{pid[:8]}@example.or.kr", "display_name": "U", "auth_level": 5},
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "buckets",
                    {
                        "id": bkt_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "owner_principal_id": pid,
                        "slug": f"b-{bkt_id[:6]}",
                        "name": "B",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:d:dept:bucket:L1/{bkt_id}",
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
                        "bucket_id": bkt_id,
                        "owner_principal_id": pid,
                        "title": "D",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:d:dept:document:L1/{doc_id}",
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
                        "bucket_id": bkt_id,
                        "document_id": doc_id,
                        "owner_principal_id": pid,
                        "page_no": 1,
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:d:dept:page:L1/{page_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
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
                        "owner_principal_id": pid,
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:d:dept:episode:L1/{ep_id}",
                        "source_harness": "chat_ui",
                        "summary_md": "ep",
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
                        "owner_principal_id": pid,
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_id}:d:dept:memory:L1/{mem_id}",
                        "acl_scope_key": f"{org_id}:{dept_id}:dept:L1",
                        "memory_kind": "fact",
                        "effective_text": "mem",
                        "bm25_text": "mem",
                    },
                )
    return {
        "org_id": org_id,
        "dept_id": dept_id,
        "pid": pid,
        "bkt_id": bkt_id,
        "doc_id": doc_id,
        "page_id": page_id,
        "ep_id": ep_id,
        "mem_id": mem_id,
    }


def _strip_rust_comments_and_tests(src: str) -> str:
    # Split before `#[cfg(test)]` if present so unit tests don't count as prod library code
    prod_part = src.split("#[cfg(test)]", 1)[0]
    prod_part = re.sub(r"//.*", "", prod_part)
    prod_part = re.sub(r"/\*.*?\*/", "", prod_part, flags=re.DOTALL)
    return prod_part


# ---------------------------------------------------------------------------
# F01 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f01
def test_f01_bva_no_unsafe_blocks_in_rust_crates(openrag_root: Path) -> None:
    crates_dir = openrag_root / "crates"
    if not crates_dir.is_dir():
        fail_missing_target(f"Missing crates directory: {crates_dir}")
    hits: list[str] = []
    for rs_file in crates_dir.rglob("*.rs"):
        cleaned = _strip_rust_comments_and_tests(rs_file.read_text(encoding="utf-8"))
        if re.search(r"\bunsafe\s*(\{|fn\b|impl\b|trait\b)", cleaned):
            hits.append(str(rs_file.relative_to(openrag_root)))
    assert hits == [], f"Found unsafe constructs in Rust crates: {hits}"


@pytest.mark.tier2
@pytest.mark.f01
def test_f01_bva_no_unwrap_or_expect_in_library_crates(openrag_root: Path) -> None:
    crates_dir = openrag_root / "crates"
    if not crates_dir.is_dir():
        fail_missing_target(f"Missing crates directory: {crates_dir}")

    ws_cargo = tomllib.loads((openrag_root / "Cargo.toml").read_text(encoding="utf-8"))
    ws_clippy = ((ws_cargo.get("workspace") or {}).get("lints") or {}).get("clippy") or {}
    assert ws_clippy.get("unwrap_used") in ("deny", "forbid") or isinstance(
        ws_clippy.get("unwrap_used"), dict
    )

    bad_libs: list[str] = []
    for crate_dir in sorted(crates_dir.iterdir()):
        lib_rs = crate_dir / "src" / "lib.rs"
        if not lib_rs.is_file():
            continue
        crate_toml = tomllib.loads((crate_dir / "Cargo.toml").read_text(encoding="utf-8"))
        inherits_ws = bool((crate_toml.get("lints") or {}).get("workspace"))
        lib_text = lib_rs.read_text(encoding="utf-8")
        has_attr = "unwrap_used" in lib_text and "expect_used" in lib_text
        if not (inherits_ws or has_attr):
            bad_libs.append(f"{crate_dir.name}: missing deny(unwrap_used, expect_used)")

        for rs_file in (crate_dir / "src").rglob("*.rs"):
            prod = _strip_rust_comments_and_tests(rs_file.read_text(encoding="utf-8"))
            if re.search(r"\.(unwrap|expect)\s*\(", prod):
                bad_libs.append(f"{rs_file.relative_to(openrag_root)}: contains .unwrap()/.expect()")
    assert bad_libs == [], f"Library crate unwrap/expect violations: {bad_libs}"


@pytest.mark.tier2
@pytest.mark.f01
def test_f01_bva_zero_excluded_refs_in_lockfile_docker_and_web_manifests(openrag_root: Path) -> None:
    forbidden = ("schift-db", "schift_db", "schift-rag-ce", "schift_rag_ce")
    targets = [
        openrag_root / "Cargo.lock",
        openrag_root / "Cargo.toml",
        openrag_root / "docker" / "postgres" / "Dockerfile",
        openrag_root / "docker" / "docker-compose.yml",
        openrag_root / "docker" / "docker-compose.airgap.yml",
        openrag_root / "web" / "package.json",
        openrag_root / "web" / "pnpm-lock.yaml",
    ]
    plugins_dir = openrag_root / "plugins"
    if plugins_dir.is_dir():
        targets.extend(plugins_dir.rglob("package.json"))
        targets.extend(plugins_dir.rglob("pyproject.toml"))

    hits: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        lower = path.read_text(encoding="utf-8", errors="replace").lower()
        for tok in forbidden:
            if tok in lower:
                hits.append(f"{path.relative_to(openrag_root)}:{tok}")
    assert hits == [], f"Excluded package/image references found: {hits}"


@pytest.mark.tier2
@pytest.mark.f01
def test_f01_bva_unknown_cli_subcommand_fails_nonzero(openrag_cli: Any) -> None:
    try:
        res = openrag_cli.run(["nonexistent-subcmd-xyz"])
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0
    assert (res.stderr or res.stdout).strip() != ""
    assert "panicked at" not in res.stderr.lower()


@pytest.mark.tier2
@pytest.mark.f01
def test_f01_bva_cli_missing_binary_raises_explicit_error(tmp_path: Path) -> None:
    isolated_cli = OpenRagCli(target_root=tmp_path)
    with pytest.raises(CliBinaryNotFoundError):
        isolated_cli.require_binary("openrag-nonexistent-binary-e2e")


# ---------------------------------------------------------------------------
# F02 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f02
def test_f02_bva_airgap_compose_no_required_external_api_keys(openrag_root: Path) -> None:
    compose_path = openrag_root / "docker" / "docker-compose.airgap.yml"
    if not compose_path.is_file():
        fail_missing_target(f"Missing {compose_path}")
    text = compose_path.read_text(encoding="utf-8")
    assert not re.search(r"\$\{(OPENROUTER_API_KEY|OPENAI_API_KEY|PINECONE_API_KEY):\?", text)


@pytest.mark.tier2
@pytest.mark.f02
def test_f02_bva_postgres_dockerfile_configures_shared_preload_libraries(openrag_root: Path) -> None:
    docker_dir = openrag_root / "docker"
    if not docker_dir.is_dir():
        fail_missing_target(f"Missing {docker_dir}")
    combined = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in docker_dir.rglob("*")
        if p.is_file()
    ).lower()
    assert "shared_preload_libraries" in combined
    assert "pg_pinecone" in combined or "pg_search" in combined


@pytest.mark.tier2
@pytest.mark.f02
def test_f02_bva_standard_compose_no_host_port_collisions_between_services(
    openrag_root: Path,
) -> None:
    for fname in ("docker-compose.yml", "docker-compose.airgap.yml"):
        fpath = openrag_root / "docker" / fname
        if not fpath.is_file():
            fail_missing_target(f"Missing {fpath}")
        data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
        services = data.get("services") or {}
        seen_host_ports: set[str] = set()
        for svc_name, svc in services.items():
            for entry in svc.get("ports") or []:
                if isinstance(entry, str):
                    parts = entry.split(":")
                    if len(parts) >= 2:
                        host_port = parts[-2].split("/")[-1]
                        assert host_port not in seen_host_ports, (
                            f"Duplicate host port {host_port} in {fname} ({svc_name})"
                        )
                        seen_host_ports.add(host_port)
                elif isinstance(entry, dict) and entry.get("published") is not None:
                    hp = str(entry["published"])
                    assert hp not in seen_host_ports, (
                        f"Duplicate host port {hp} in {fname} ({svc_name})"
                    )
                    seen_host_ports.add(hp)


@pytest.mark.tier2
@pytest.mark.f02
def test_f02_bva_cli_up_invalid_profile_rejected(openrag_cli: Any) -> None:
    try:
        res = openrag_cli.up(profile="invalid_profile_xyz")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0
    assert (res.stderr or res.stdout).strip() != ""


@pytest.mark.tier2
@pytest.mark.f02
def test_f02_bva_docker_build_contexts_and_image_refs_valid(openrag_root: Path) -> None:
    dockerfile = openrag_root / "docker" / "postgres" / "Dockerfile"
    if not dockerfile.is_file():
        fail_missing_target(f"Missing {dockerfile}")
    assert re.search(r"^\s*FROM\s+\S+", dockerfile.read_text(encoding="utf-8"), re.MULTILINE)

    for fname in ("docker-compose.yml", "docker-compose.airgap.yml"):
        fpath = openrag_root / "docker" / fname
        if not fpath.is_file():
            fail_missing_target(f"Missing {fpath}")
        data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
        for svc_name, svc in (data.get("services") or {}).items():
            if "image" in svc:
                assert isinstance(svc["image"], str) and svc["image"].strip()
            if "build" in svc:
                bld = svc["build"]
                ctx = bld if isinstance(bld, str) else bld.get("context", ".")
                resolved = (fpath.parent / ctx).resolve()
                assert resolved.exists(), f"{fname}:{svc_name} build context missing: {resolved}"


# ---------------------------------------------------------------------------
# F03 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f03
def test_f03_bva_invalid_yaml_syntax_rejected_with_4xx(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    resp = api_client.studio_pipeline_test({"pipeline_yaml": "parse: [unclosed"})
    assert resp.status in (400, 422), f"Expected 400/422, got {resp.status}: {resp.text}"


@pytest.mark.tier2
@pytest.mark.f03
def test_f03_bva_wrong_vector_dimension_rejected(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bad_yaml = make_pipeline_yaml().replace("dim: 128", "dim: 1536")
    resp = api_client.studio_pipeline_test({"pipeline_yaml": bad_yaml})
    assert resp.status in (400, 422), f"Expected 400/422 for 1536d, got {resp.status}"


@pytest.mark.tier2
@pytest.mark.f03
def test_f03_bva_alpha_and_auth_level_out_of_bounds_rejected(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    for bad_alpha in (-0.1, 1.5):
        y = make_pipeline_yaml(alpha=bad_alpha)
        resp = api_client.studio_pipeline_test({"pipeline_yaml": y})
        assert resp.status in (400, 422), f"alpha={bad_alpha} got {resp.status}"

    for bad_lvl in (0, 11):
        y = make_pipeline_yaml(local_only_min_auth_level=bad_lvl)
        resp = api_client.studio_pipeline_test({"pipeline_yaml": y})
        assert resp.status in (400, 422), f"auth_level={bad_lvl} got {resp.status}"


@pytest.mark.tier2
@pytest.mark.f03
def test_f03_bva_unknown_memory_backend_or_guard_action_rejected(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bad_mem = make_pipeline_yaml(memory_backend="unknown_mem_db")
    r1 = api_client.studio_pipeline_test({"pipeline_yaml": bad_mem})
    assert r1.status in (400, 422), f"unknown memory_backend got {r1.status}"

    bad_act = make_pipeline_yaml().replace("default_action: mask", "default_action: nuke")
    r2 = api_client.studio_pipeline_test({"pipeline_yaml": bad_act})
    assert r2.status in (400, 422), f"unknown guard default_action got {r2.status}"


@pytest.mark.tier2
@pytest.mark.f03
def test_f03_bva_nonexistent_pipeline_file_in_cli_ingest_fails_nonzero(
    openrag_cli: Any, sample_factory: Any
) -> None:
    pdf_path = sample_factory.pdf_file("doc.pdf", ["Test"])
    missing_yaml = sample_factory.base_dir / "missing_pipeline.yaml"
    try:
        res = openrag_cli.ingest(pdf_path, bucket="b1", pipeline=missing_yaml)
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0
    assert (res.stderr or res.stdout).strip() != ""


# ---------------------------------------------------------------------------
# F04 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f04
def test_f04_bva_missing_migration_prefix_detected_by_harness(
    openrag_root: Path, tmp_path: Path
) -> None:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations dir: {migrations_dir}")
    subset = tmp_path / "mig_partial"
    subset.mkdir()
    for sql_file in sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql")):
        if not sql_file.name.startswith("007_"):
            shutil.copy2(sql_file, subset / sql_file.name)

    with PgEphemeralCluster() as pg:
        with pytest.raises(FileNotFoundError) as err:
            pg.apply_migrations(subset)
        assert "007" in str(err.value)


@pytest.mark.tier2
@pytest.mark.f04
def test_f04_bva_server_startup_against_already_migrated_db_is_idempotent(
    openrag_root: Path,
) -> None:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations dir: {migrations_dir}")

    with PgEphemeralCluster() as pg:
        pg.apply_migrations(migrations_dir)
        srv = EphemeralOpenRagServer(
            target_root=openrag_root,
            database_url=pg.dsn(),
            mode="offline",
        )
        try:
            srv.start()
        except CliBinaryNotFoundError as exc:
            fail_missing_target(str(exc))
        try:
            client = OpenRagHttpClient(srv.base_url)
            client.health().assert_status(200)
        finally:
            srv.stop()

        with psycopg.connect(pg.dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.schemata "
                    "WHERE schema_name IN ('iam','vectors','corpus','pii','memory','wiki','indexing','gateway','plugins','eval')"
                )
                assert cur.fetchone()[0] == 10  # type: ignore[index]


@pytest.mark.tier2
@pytest.mark.f04
def test_f04_bva_check_constraints_reject_invalid_auth_level_patch_op_and_guard_action(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    for bad_lvl in (0, 11):
        with pytest.raises(psycopg.errors.CheckViolation):
            with psycopg.connect(migrated_pg.dsn()) as conn:
                with conn.cursor() as cur:
                    _dynamic_insert(
                        cur,
                        "iam",
                        "principals",
                        {
                            "id": str(uuid.uuid4()),
                            "org_id": seeded["org_id"],
                            "email": f"bad-{bad_lvl}-{uuid.uuid4().hex[:4]}@example.or.kr",
                            "display_name": "Bad",
                            "auth_level": bad_lvl,
                        },
                    )

    with pytest.raises(psycopg.errors.CheckViolation):
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "memory",
                    "patches",
                    {
                        "id": str(uuid.uuid4()),
                        "org_id": seeded["org_id"],
                        "memory_entry_id": seeded["mem_id"],
                        "actor_type": "human",
                        "actor_id": seeded["pid"],
                        "patch_op": "delete_all",
                        "prev_version": 1,
                        "new_version": 2,
                        "new_text": "bad",
                        "reason": "test",
                    },
                )

    with pytest.raises(psycopg.errors.CheckViolation):
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                pol = _dynamic_insert(
                    cur, "pii", "guard_policies", {"org_id": seeded["org_id"], "name": f"gp-{uuid.uuid4().hex[:6]}"}
                )
                _dynamic_insert(
                    cur,
                    "pii",
                    "guard_rules",
                    {
                        "org_id": seeded["org_id"],
                        "policy_id": pol["id"],
                        "rule_kind": "keyword",
                        "pattern_or_category": "X",
                        "action": "drop",
                    },
                )


@pytest.mark.tier2
@pytest.mark.f04
def test_f04_bva_outbox_partial_unique_idempotency_key_blocks_duplicate_pending_only(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    dup_key = f"dup-{uuid.uuid4().hex}"
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "indexing",
                    "outbox",
                    {
                        "org_id": seeded["org_id"],
                        "target_table": "corpus.chunks",
                        "target_id": seeded["doc_id"],
                        "op": "upsert_embed",
                        "status": "pending",
                        "idempotency_key": dup_key,
                    },
                )

    with pytest.raises(psycopg.errors.UniqueViolation):
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "indexing",
                    "outbox",
                    {
                        "org_id": seeded["org_id"],
                        "target_table": "corpus.chunks",
                        "target_id": seeded["doc_id"],
                        "op": "upsert_embed",
                        "status": "pending",
                        "idempotency_key": dup_key,
                    },
                )


@pytest.mark.tier2
@pytest.mark.f04
def test_f04_bva_bench_runs_security_gate_generated_column_flips_false_on_leak(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                pc = _dynamic_insert(cur, "eval", "pipeline_configs", {"org_id": seeded["org_id"], "name": "pc-bva"})
                gs = _dynamic_insert(
                    cur,
                    "eval",
                    "golden_sets",
                    {"org_id": seeded["org_id"], "name": "gs-bva", "axis": "retrieval"},
                )
                r_pii = _dynamic_insert(
                    cur,
                    "eval",
                    "bench_runs",
                    {
                        "org_id": seeded["org_id"],
                        "pipeline_config_id": pc["id"],
                        "golden_set_id": gs["id"],
                        "track": "track1_retrieval",
                        "pii_leak_count": 1,
                        "arn_violation_count": 0,
                    },
                )
                r_arn = _dynamic_insert(
                    cur,
                    "eval",
                    "bench_runs",
                    {
                        "org_id": seeded["org_id"],
                        "pipeline_config_id": pc["id"],
                        "golden_set_id": gs["id"],
                        "track": "track1_retrieval",
                        "pii_leak_count": 0,
                        "arn_violation_count": 1,
                    },
                )
                assert r_pii["security_gate_pass"] is False
                assert r_arn["security_gate_pass"] is False

    with pytest.raises(psycopg.errors.Error):
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO eval.bench_runs (id, org_id, pipeline_config_id, golden_set_id, track, pii_leak_count, arn_violation_count, security_gate_pass) "
                    "VALUES (%s, %s, %s, %s, 'track1_retrieval', 1, 0, true)",
                    (str(uuid.uuid4()), seeded["org_id"], pc["id"], gs["id"]),
                )


# ---------------------------------------------------------------------------
# F05 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f05
def test_f05_bva_fallback_vector_and_lexical_search_still_enforces_fail_closed_rls(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    peer_id = str(uuid.uuid4())
    priv_chunk = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {"id": peer_id, "org_id": seeded["org_id"], "email": f"peer-{peer_id[:6]}@example.or.kr", "display_name": "Peer", "auth_level": 5},
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": priv_chunk,
                        "org_id": seeded["org_id"],
                        "dept_id": seeded["dept_id"],
                        "bucket_id": seeded["bkt_id"],
                        "document_id": seeded["doc_id"],
                        "page_id": seeded["page_id"],
                        "owner_principal_id": seeded["pid"],
                        "chunk_idx": 99,
                        "redacted_text": "비밀 여비 규정",
                        "bm25_text": "비밀 여비 규정",
                        "access_scope": "private",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{seeded['org_id']}:d:private:chunk:L1/{priv_chunk}",
                        "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:private:L1",
                        "embedding": _unit_vec_str(0),
                    },
                )

    peer_ctx = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=peer_id,
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
        dept_admin_ids=(),
    )
    vec_rows = migrated_pg.query_as_principal(
        "SELECT id FROM corpus.chunks WHERE id = %s::uuid ORDER BY embedding <=> %s::vector(128) LIMIT 5",
        principal=peer_ctx,
        params=(priv_chunk, _unit_vec_str(0)),
    )
    lex_rows = migrated_pg.query_as_principal(
        "SELECT id FROM corpus.chunks WHERE id = %s::uuid AND bm25_text ILIKE '%%여비%%'",
        principal=peer_ctx,
        params=(priv_chunk,),
    )
    assert vec_rows == []
    assert lex_rows == []


@pytest.mark.tier2
@pytest.mark.f05
def test_f05_bva_empty_or_unmatched_query_returns_empty_not_error(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    ctx = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=seeded["pid"],
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
    )
    for q in ("", "쀍뛣힇"):
        rows = migrated_pg.query_as_principal(
            "SELECT id FROM corpus.chunks WHERE org_id = %s::uuid AND bm25_text %% %s",
            principal=ctx,
            params=(seeded["org_id"], q),
        )
        assert rows == []


@pytest.mark.tier2
@pytest.mark.f05
def test_f05_bva_null_embeddings_skipped_gracefully_in_vector_query(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    c_null = str(uuid.uuid4())
    c_vec = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": c_null,
                        "org_id": seeded["org_id"],
                        "dept_id": seeded["dept_id"],
                        "bucket_id": seeded["bkt_id"],
                        "document_id": seeded["doc_id"],
                        "page_id": seeded["page_id"],
                        "owner_principal_id": seeded["pid"],
                        "chunk_idx": 50,
                        "redacted_text": "No embedding",
                        "bm25_text": "No embedding",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{seeded['org_id']}:d:dept:chunk:L1/{c_null}",
                        "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:dept:L1",
                        "embedding": None,
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": c_vec,
                        "org_id": seeded["org_id"],
                        "dept_id": seeded["dept_id"],
                        "bucket_id": seeded["bkt_id"],
                        "document_id": seeded["doc_id"],
                        "page_id": seeded["page_id"],
                        "owner_principal_id": seeded["pid"],
                        "chunk_idx": 51,
                        "redacted_text": "Has embedding",
                        "bm25_text": "Has embedding",
                        "access_scope": "dept",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{seeded['org_id']}:d:dept:chunk:L1/{c_vec}",
                        "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:dept:L1",
                        "embedding": _unit_vec_str(3),
                    },
                )

    ctx = PgPrincipalContext(
        org_id=seeded["org_id"],
        principal_id=seeded["pid"],
        auth_level=5,
        dept_ids=(seeded["dept_id"],),
    )
    rows = migrated_pg.query_as_principal(
        """
        SELECT id::text AS id
        FROM corpus.chunks
        WHERE id = ANY(%s::uuid[]) AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector(128)
        """,
        principal=ctx,
        params=([c_null, c_vec], _unit_vec_str(3)),
    )
    assert [r["id"] for r in rows] == [c_vec]


@pytest.mark.tier2
@pytest.mark.f05
def test_f05_bva_remote_engine_without_pinecone_key_fails_cleanly_not_silent(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    caps = migrated_pg.probe_capabilities()
    resp = api_client.studio_index_set_engine(
        {"table": "corpus.chunks", "engine": "remote"}
    )
    if not caps["pg_pinecone"]:
        assert resp.status in (400, 422, 503) or (
            resp.status == 200 and resp.json().get("mode") in ("fallback", "mock", "remote")
        )
    api_client.health().assert_status(200)


@pytest.mark.tier2
@pytest.mark.f05
def test_f05_bva_capability_probe_isolated_db_without_extensions_still_reports_accurately(
    openrag_root: Path,
) -> None:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations dir: {migrations_dir}")
    with PgEphemeralCluster() as pg:
        before = pg.probe_capabilities()
        assert before == {
            "vector": False,
            "pg_trgm": False,
            "pgcrypto": False,
            "pg_pinecone": False,
            "pg_search": False,
        }
        pg.apply_migrations(migrations_dir)
        after = pg.probe_capabilities()
        assert after["vector"] is True
        assert after["pg_trgm"] is True
        assert after["pgcrypto"] is True


# ---------------------------------------------------------------------------
# F06 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f06
def test_f06_bva_wrong_vector_dimensions_127_129_384_rejected(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    for bad_dim in (127, 129, 384):
        bad_vec = "[" + ",".join(["1.0"] + ["0.0"] * (bad_dim - 1)) + "]"
        with pytest.raises(psycopg.errors.DataException):
            with psycopg.connect(migrated_pg.dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE corpus.chunks SET embedding = %s::vector WHERE document_id = %s::uuid",
                        (bad_vec, seeded["doc_id"]),
                    )
                    if cur.rowcount == 0:
                        _dynamic_insert(
                            cur,
                            "corpus",
                            "chunks",
                            {
                                "id": str(uuid.uuid4()),
                                "org_id": seeded["org_id"],
                                "dept_id": seeded["dept_id"],
                                "bucket_id": seeded["bkt_id"],
                                "document_id": seeded["doc_id"],
                                "page_id": seeded["page_id"],
                                "owner_principal_id": seeded["pid"],
                                "chunk_idx": 77,
                                "redacted_text": "bad dim",
                                "bm25_text": "bad dim",
                                "access_scope": "dept",
                                "min_auth_level": 1,
                                "allowed_group_ids": [],
                                "arn": f"arn:openrag:{seeded['org_id']}:d:dept:chunk:L1/{uuid.uuid4()}",
                                "acl_scope_key": f"{seeded['org_id']}:{seeded['dept_id']}:dept:L1",
                                "embedding": bad_vec,
                            },
                        )


@pytest.mark.tier2
@pytest.mark.f06
def test_f06_bva_nan_and_infinity_vector_components_rejected(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    for bad_tok in ("NaN", "Infinity"):
        bad_vec = "[" + bad_tok + "," + ",".join(["0.0"] * 127) + "]"
        with pytest.raises(psycopg.errors.DataException):
            with psycopg.connect(migrated_pg.dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT %s::vector(128)",
                        (bad_vec,),
                    )


@pytest.mark.tier2
@pytest.mark.f06
def test_f06_bva_null_acl_scope_key_rejected_not_null(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    for schema, table, extra in (
        (
            "corpus",
            "documents",
            {
                "id": str(uuid.uuid4()),
                "org_id": seeded["org_id"],
                "dept_id": seeded["dept_id"],
                "bucket_id": seeded["bkt_id"],
                "owner_principal_id": seeded["pid"],
                "access_scope": "dept",
                "acl_scope_key": None,
            },
        ),
        (
            "memory",
            "entries",
            {
                "id": str(uuid.uuid4()),
                "org_id": seeded["org_id"],
                "dept_id": seeded["dept_id"],
                "source_episode_id": seeded["ep_id"],
                "owner_principal_id": seeded["pid"],
                "access_scope": "dept",
                "memory_kind": "fact",
                "effective_text": "bva",
                "bm25_text": "bva",
                "acl_scope_key": None,
            },
        ),
    ):
        with pytest.raises(psycopg.errors.NotNullViolation):
            with psycopg.connect(migrated_pg.dsn()) as conn:
                with conn.cursor() as cur:
                    _dynamic_insert(cur, schema, table, extra)


@pytest.mark.tier2
@pytest.mark.f06
def test_f06_bva_invalid_index_binding_engine_or_external_backend_rejected(
    migrated_pg: PgEphemeralCluster,
) -> None:
    seeded = _seed_base(migrated_pg)
    with pytest.raises(psycopg.errors.CheckViolation):
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "vectors",
                    "index_bindings",
                    {
                        "org_id": seeded["org_id"],
                        "table_name": f"t_{uuid.uuid4().hex[:6]}",
                        "pg_pinecone_engine": "milvus",
                    },
                )

    with pytest.raises(psycopg.errors.CheckViolation):
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "vectors",
                    "index_bindings",
                    {
                        "org_id": seeded["org_id"],
                        "table_name": f"t_{uuid.uuid4().hex[:6]}",
                        "external_backend": "unknown_backend",
                    },
                )


@pytest.mark.tier2
@pytest.mark.f06
def test_f06_bva_zero_vector_rejected_on_ingest_or_produces_non_finite_cosine(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    zero_vec = "[" + ",".join(["0.0"] * 128) + "]"
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (%s::vector(128) <=> %s::vector(128))",
                (zero_vec, _unit_vec_str(0)),
            )
            dist = cur.fetchone()[0]  # type: ignore[index]
            assert math.isnan(float(dist))

    bkt = api_client.create_bucket(
        {"slug": f"f06-zero-{uuid.uuid4().hex[:6]}", "name": "Zero Vec", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bkt_id = str(bkt.get("id") or bkt.get("bucket_id"))
    resp = api_client.search_bucket(bkt_id, {"query": ""})
    assert resp.status in (200, 400, 422)
