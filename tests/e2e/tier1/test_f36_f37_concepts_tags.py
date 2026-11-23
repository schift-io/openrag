"""Tier 1 isolated happy-path E2E tests for F36-F37 (Canonical Concept/Alias Graph & Tag Search)."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import math
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError, OpenRagCli
from lib.fixtures import FixtureFactory
from lib.http_client import OpenRagHttpClient, extract_tag_facets
from lib.mcp_client import McpHttpClient
from lib.pg_harness import PgEphemeralCluster, PgPrincipalContext

pytestmark = [pytest.mark.tier1]


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> PgEphemeralCluster:
    migrations_dir = root / "migrations"
    mig_008 = migrations_dir / "008_concepts_tags.sql"
    if not mig_008.is_file():
        fail_missing_target(
            f"Missing Follow-up 4 migration file: {mig_008} (required for F36/F37)."
        )
    pg.ensure_started()
    has_orgs = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;").strip()
    has_concepts = pg.execute_sql("SELECT to_regclass('wiki.concepts') IS NOT NULL;").strip()
    if has_orgs != "t":
        applied = pg.apply_migrations(
            migrations_dir,
            required_prefixes=[f"{i:03d}" for i in range(9)],
        )
        assert any(name.startswith("008_") for name in applied), (
            f"Expected 008 migration to be applied, got {applied}"
        )
    elif has_concepts != "t":
        pg._run_psql_file_or_cmd(file_path=mig_008)
        pg.grant_app_role_privileges()
    return pg


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit_vec_128(seed: int = 0) -> list[float]:
    vals = [0.0] * 128
    vals[seed % 128] = 1.0
    return vals


def _vec_literal(vals: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vals) + "]"


def _seed_two_orgs(pg: PgEphemeralCluster) -> dict[str, str]:
    org1_id = str(uuid.uuid4())
    org2_id = str(uuid.uuid4())
    dept1_id = str(uuid.uuid4())
    dept2_id = str(uuid.uuid4())
    owner_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    dept_admin_id = str(uuid.uuid4())
    outsider_id = str(uuid.uuid4())
    slug1 = f"org1-{org1_id[:8]}"
    slug2 = f"org2-{org2_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES
            ('{org1_id}', '{slug1}', 'Org One {slug1}'),
            ('{org2_id}', '{slug2}', 'Org Two {slug2}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
            ('{dept1_id}', '{org1_id}', 'Knowledge', 'knowledge'),
            ('{dept2_id}', '{org2_id}', 'External', 'external');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{owner_id}', '{org1_id}', 'owner-{owner_id[:6]}@example.org', 'OwnerL3', 3, false, false),
            ('{peer_id}', '{org1_id}', 'peer-{peer_id[:6]}@example.org', 'PeerL3', 3, false, false),
            ('{dept_admin_id}', '{org1_id}', 'dadmin-{dept_admin_id[:6]}@example.org', 'DeptAdminL8', 8, true, true),
            ('{outsider_id}', '{org2_id}', 'out-{outsider_id[:6]}@example.org', 'OutsiderL9', 9, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org1_id}', '{dept1_id}', '{owner_id}', 'member'),
            ('{org1_id}', '{dept1_id}', '{peer_id}', 'member'),
            ('{org1_id}', '{dept1_id}', '{dept_admin_id}', 'dept_admin'),
            ('{org2_id}', '{dept2_id}', '{outsider_id}', 'dept_admin');
        """
    )
    return {
        "org1_id": org1_id,
        "org2_id": org2_id,
        "slug1": slug1,
        "slug2": slug2,
        "dept1_id": dept1_id,
        "dept2_id": dept2_id,
        "owner_id": owner_id,
        "peer_id": peer_id,
        "dept_admin_id": dept_admin_id,
        "outsider_id": outsider_id,
    }


def _mint_api_key(
    api_client: OpenRagHttpClient,
    org_id: str,
    principal_id: str,
    tag: str,
) -> str:
    resp = api_client.cli_code_exchange(
        {
            "code": issue_cli_code(org_id, principal_id),
            "org_id": org_id,
            "principal_id": principal_id,
        }
    )
    resp.assert_status((200, 201))
    data = resp.json()
    key = data.get("api_key") or data.get("token") or data.get("key")
    assert isinstance(key, str) and key, f"Expected minted api_key in {data}"
    return key


def _extract_concept_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("concept_id", "canonical_id", "id"):
            val = payload.get(key)
            if val:
                return str(val)
        for nested in ("concept", "canonical", "data"):
            if isinstance(payload.get(nested), dict):
                return _extract_concept_id(payload[nested])
        matches = payload.get("matches") or payload.get("concepts") or payload.get("items")
        if isinstance(matches, list) and matches:
            return _extract_concept_id(matches[0])
    raise AssertionError(f"Could not extract concept_id from {payload!r}")


def _extract_concept_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("concepts", "items", "results", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _extract_search_hits(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "hits", "matches", "documents"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _hit_doc_ids(hits: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for h in hits:
        for key in ("document_id", "doc_id", "id"):
            val = h.get(key)
            if val:
                out.add(str(val))
                break
    return out


def _hit_blobs(hits: list[dict[str, Any]]) -> str:
    return "\n".join(str(h) for h in hits)


def _extract_doc_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("document_id", "id", "doc_id"):
            val = payload.get(key)
            if val:
                return str(val)
        if isinstance(payload.get("document"), dict):
            return _extract_doc_id(payload["document"])
    raise AssertionError(f"Could not extract document_id from {payload!r}")


# ============================================================================
# F36 — Canonical Concept / Alias Equivalence Graph (6 tests)
# ============================================================================


@pytest.mark.tier1
@pytest.mark.f36
def test_f36_schema_008_concepts_aliases_ddl_and_fail_closed_rls(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)

    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'wiki' AND c.relname IN ('concepts', 'concept_aliases')
                ORDER BY c.relname
                """
            )
            rls_rows = {row[0]: (bool(row[1]), bool(row[2])) for row in cur.fetchall()}
            assert set(rls_rows.keys()) == {"concepts", "concept_aliases"}, rls_rows
            for tbl, (rls_on, force_on) in rls_rows.items():
                assert rls_on is True, f"wiki.{tbl} must enable RLS"
                assert force_on is True, f"wiki.{tbl} must force RLS"

            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'wiki' AND table_name IN ('concepts', 'concept_aliases')
                """
            )
            cols_by_table: dict[str, set[str]] = {"concepts": set(), "concept_aliases": set()}
            for tbl_name, col_name in cur.fetchall():
                cols_by_table[tbl_name].add(col_name)

            expected_concept_cols = {
                "id",
                "org_id",
                "canonical_key",
                "display_name",
                "category",
                "description",
                "wiki_page_id",
                "embedding",
                "created_by",
                "created_at",
                "updated_at",
            }
            expected_alias_cols = {
                "id",
                "org_id",
                "concept_id",
                "alias_text",
                "normalized_alias",
                "compact_alias",
                "lang",
                "source",
                "confidence",
                "created_by",
                "created_at",
            }
            assert expected_concept_cols <= cols_by_table["concepts"], (
                f"wiki.concepts missing columns: {expected_concept_cols - cols_by_table['concepts']}"
            )
            assert expected_alias_cols <= cols_by_table["concept_aliases"], (
                f"wiki.concept_aliases missing columns: {expected_alias_cols - cols_by_table['concept_aliases']}"
            )

            c_owner_id = str(uuid.uuid4())
            a_owner_id = str(uuid.uuid4())
            c_owner_del_id = str(uuid.uuid4())
            a_owner_del_id = str(uuid.uuid4())
            c_admin_id = str(uuid.uuid4())
            a_admin_id = str(uuid.uuid4())

            cur.execute(
                """
                INSERT INTO wiki.concepts
                    (id, org_id, canonical_key, display_name, category, description, embedding, created_by)
                VALUES
                    (%s, %s, 'apple', 'Apple', 'domain', 'Cross-lingual fruit concept', %s::vector(128), %s),
                    (%s, %s, 'temp-owner-concept', 'Temp Owner Concept', 'general', 'Owner deletable concept', %s::vector(128), %s),
                    (%s, %s, 'llmwiki', 'LLM Wiki', 'domain', 'Knowledge architecture concept', %s::vector(128), %s)
                """,
                (
                    c_owner_id,
                    ctx["org1_id"],
                    _vec_literal(_unit_vec_128(1)),
                    ctx["owner_id"],
                    c_owner_del_id,
                    ctx["org1_id"],
                    _vec_literal(_unit_vec_128(2)),
                    ctx["owner_id"],
                    c_admin_id,
                    ctx["org1_id"],
                    _vec_literal(_unit_vec_128(3)),
                    ctx["dept_admin_id"],
                ),
            )
            cur.execute(
                """
                INSERT INTO wiki.concept_aliases
                    (id, org_id, concept_id, alias_text, normalized_alias, compact_alias, lang, source, confidence, created_by)
                VALUES
                    (%s, %s, %s, '사과', '사과', '사과', 'ko', 'manual', 1.0, %s),
                    (%s, %s, %s, '임시개념', '임시개념', '임시개념', 'ko', 'manual', 1.0, %s),
                    (%s, %s, %s, '세컨드 브레인', '세컨드 브레인', '세컨드브레인', 'ko', 'manual', 1.0, %s)
                """,
                (
                    a_owner_id,
                    ctx["org1_id"],
                    c_owner_id,
                    ctx["owner_id"],
                    a_owner_del_id,
                    ctx["org1_id"],
                    c_owner_del_id,
                    ctx["owner_id"],
                    a_admin_id,
                    ctx["org1_id"],
                    c_admin_id,
                    ctx["dept_admin_id"],
                ),
            )
        conn.commit()

    # (a) Unset GUCs (app.org_id / app.principal_id unset) must fail closed
    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL ROLE openrag_app")
                cur.execute("SELECT id FROM wiki.concepts")
                assert cur.fetchall() == []
                cur.execute("SELECT id FROM wiki.concept_aliases")
                assert cur.fetchall() == []
                cur.execute("DELETE FROM wiki.concepts WHERE id = %s RETURNING id", (c_owner_id,))
                assert cur.fetchall() == []
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL ROLE openrag_app")
                    cur.execute(
                        """
                        INSERT INTO wiki.concepts (org_id, canonical_key, display_name)
                        VALUES (%s, 'unset-fail', 'Unset Fail')
                        """,
                        (ctx["org1_id"],),
                    )

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org1_id"],
        principal_id=ctx["owner_id"],
        auth_level=3,
        dept_ids=(ctx["dept1_id"],),
    )
    peer_ctx = PgPrincipalContext(
        org_id=ctx["org1_id"],
        principal_id=ctx["peer_id"],
        auth_level=3,
        dept_ids=(ctx["dept1_id"],),
    )
    dadmin_ctx = PgPrincipalContext(
        org_id=ctx["org1_id"],
        principal_id=ctx["dept_admin_id"],
        auth_level=8,
        dept_ids=(ctx["dept1_id"],),
        dept_admin_ids=(ctx["dept1_id"],),
    )
    outsider_ctx = PgPrincipalContext(
        org_id=ctx["org2_id"],
        principal_id=ctx["outsider_id"],
        auth_level=9,
        dept_ids=(ctx["dept2_id"],),
        dept_admin_ids=(ctx["dept2_id"],),
    )

    # (b) Cross-org outsider returns [] on SELECT and is blocked on INSERT/UPDATE/DELETE
    assert (
        pg_ephemeral.query_as_principal(
            "SELECT id FROM wiki.concepts WHERE id IN (%s, %s)",
            principal=outsider_ctx,
            params=(c_owner_id, c_admin_id),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "SELECT id FROM wiki.concept_aliases WHERE concept_id IN (%s, %s)",
            principal=outsider_ctx,
            params=(c_owner_id, c_admin_id),
        )
        == []
    )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        pg_ephemeral.query_as_principal(
            """
            INSERT INTO wiki.concepts (org_id, canonical_key, display_name, created_by)
            VALUES (%s, 'cross-org-inject', 'Cross Org Inject', %s)
            """,
            principal=outsider_ctx,
            params=(ctx["org1_id"], ctx["outsider_id"]),
            fetch=False,
        )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        pg_ephemeral.query_as_principal(
            """
            INSERT INTO wiki.concept_aliases
                (org_id, concept_id, alias_text, normalized_alias, compact_alias, created_by)
            VALUES (%s, %s, 'cross-alias', 'cross-alias', 'crossalias', %s)
            """,
            principal=outsider_ctx,
            params=(ctx["org1_id"], c_owner_id, ctx["outsider_id"]),
            fetch=False,
        )
    assert (
        pg_ephemeral.query_as_principal(
            "UPDATE wiki.concepts SET display_name = 'Outsider Hijack' WHERE id = %s RETURNING id",
            principal=outsider_ctx,
            params=(c_owner_id,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "DELETE FROM wiki.concepts WHERE id = %s RETURNING id",
            principal=outsider_ctx,
            params=(c_owner_id,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "DELETE FROM wiki.concept_aliases WHERE id = %s RETURNING id",
            principal=outsider_ctx,
            params=(a_owner_id,),
        )
        == []
    )

    # (c) Same-org non-creator peer CAN SELECT org concepts/aliases,
    #     CANNOT UPDATE/DELETE owner's concept/alias, while owner and dept_admin CAN update/delete
    peer_concepts = pg_ephemeral.query_as_principal(
        "SELECT id FROM wiki.concepts WHERE id IN (%s, %s)",
        principal=peer_ctx,
        params=(c_owner_id, c_admin_id),
    )
    peer_aliases = pg_ephemeral.query_as_principal(
        "SELECT id FROM wiki.concept_aliases WHERE concept_id IN (%s, %s)",
        principal=peer_ctx,
        params=(c_owner_id, c_admin_id),
    )
    assert {str(r["id"]) for r in peer_concepts} == {c_owner_id, c_admin_id}
    assert {str(r["id"]) for r in peer_aliases} == {a_owner_id, a_admin_id}

    assert (
        pg_ephemeral.query_as_principal(
            "UPDATE wiki.concepts SET display_name = 'Peer Tamper' WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(c_owner_id,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "DELETE FROM wiki.concepts WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(c_owner_id,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "UPDATE wiki.concept_aliases SET alias_text = 'Peer Alias Tamper' WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(a_owner_id,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "DELETE FROM wiki.concept_aliases WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(a_owner_id,),
        )
        == []
    )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        pg_ephemeral.query_as_principal(
            """
            INSERT INTO wiki.concepts (org_id, canonical_key, display_name, created_by)
            VALUES (%s, 'peer-spoof', 'Peer Spoof', %s)
            """,
            principal=peer_ctx,
            params=(ctx["org1_id"], ctx["owner_id"]),
            fetch=False,
        )

    # Owner (creator) CAN update & delete own concept/alias
    owner_upd = pg_ephemeral.query_as_principal(
        "UPDATE wiki.concepts SET display_name = 'Apple Updated' WHERE id = %s RETURNING display_name",
        principal=owner_ctx,
        params=(c_owner_id,),
    )
    assert owner_upd == [{"display_name": "Apple Updated"}]
    owner_del_alias = pg_ephemeral.query_as_principal(
        "DELETE FROM wiki.concept_aliases WHERE id = %s RETURNING id",
        principal=owner_ctx,
        params=(a_owner_del_id,),
    )
    assert len(owner_del_alias) == 1
    owner_del_concept = pg_ephemeral.query_as_principal(
        "DELETE FROM wiki.concepts WHERE id = %s RETURNING id",
        principal=owner_ctx,
        params=(c_owner_del_id,),
    )
    assert len(owner_del_concept) == 1

    # Dept admin CAN update & delete owner's concept/alias
    admin_upd = pg_ephemeral.query_as_principal(
        "UPDATE wiki.concepts SET display_name = 'Apple Admin Edited' WHERE id = %s RETURNING display_name",
        principal=dadmin_ctx,
        params=(c_owner_id,),
    )
    assert admin_upd == [{"display_name": "Apple Admin Edited"}]
    admin_del_alias = pg_ephemeral.query_as_principal(
        "DELETE FROM wiki.concept_aliases WHERE id = %s RETURNING id",
        principal=dadmin_ctx,
        params=(a_owner_id,),
    )
    assert len(admin_del_alias) == 1
    admin_del_concept = pg_ephemeral.query_as_principal(
        "DELETE FROM wiki.concepts WHERE id = %s RETURNING id",
        principal=dadmin_ctx,
        params=(c_owner_id,),
    )
    assert len(admin_del_concept) == 1


@pytest.mark.tier1
@pytest.mark.f36
def test_f36_cross_lingual_and_domain_synonym_crud_via_http_cli_mcp_and_console(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    openrag_cli: OpenRagCli,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "crud")
    client = api_client.with_api_key(api_key)

    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    # 1. HTTP cross-lingual apple <-> 사과
    apple_link = client.link_concepts(
        {"source": "apple", "target": "사과", "canonical_name": "apple"}
    ).assert_status((200, 201)).json()
    apple_cid = _extract_concept_id(apple_link)

    listed_apple = client.list_concepts(query="apple").assert_status(200).json()
    apple_entries = _extract_concept_list(listed_apple)
    assert any(
        _extract_concept_id(e) == apple_cid or "사과" in str(e)
        for e in apple_entries
    ), f"Expected linked apple=사과 in {listed_apple!r}"

    # 2. CLI domain synonyms llmwiki <-> 세컨드 브레인 <-> LLM Wiki
    cli_env = {
        "OPENRAG_API_URL": client.base_url,
        "OPENRAG_API_KEY": api_key,
        "DATABASE_URL": pg_ephemeral.dsn(),
    }
    openrag_cli.concept_link("llmwiki", "세컨드 브레인", env=cli_env, check=True)
    openrag_cli.concept_link("llmwiki", "LLM Wiki", env=cli_env, check=True)
    _, cli_list_data = openrag_cli.run_json(
        ["concept", "list", "--query", "llmwiki", "--json"],
        env=cli_env,
    )
    assert "세컨드 브레인" in str(cli_list_data)
    assert "LLM Wiki" in str(cli_list_data)

    # 3. Stdio & HTTP MCP concept_link + concept_resolve
    mcp_http = McpHttpClient(client)
    mcp_http.handshake()
    mcp_http.concept_link("llmwiki", "Second Brain")
    mcp_http.concept_link("RFP", "제안요청서")

    with mcp_stdio(
        api_key=api_key,
        principal_id=ctx["dept_admin_id"],
        env={"DATABASE_URL": pg_ephemeral.dsn()},
    ) as mcp_s:
        mcp_s.handshake()
        res_sb_ko = mcp_s.concept_resolve("세컨드 브레인", include_aliases=True)
        res_sb_en = mcp_s.concept_resolve("Second Brain", include_aliases=True)
        res_llm = mcp_s.concept_resolve("LLM Wiki", include_aliases=True)
        res_apple = mcp_s.concept_resolve("apple", include_aliases=True)
        res_sagwa = mcp_s.concept_resolve("사과", include_aliases=True)
        res_rfp = mcp_s.concept_resolve("제안요청서", include_aliases=True)

    assert not res_sb_ko.is_error and not res_sb_en.is_error and not res_llm.is_error
    sb_ko_id = _extract_concept_id(res_sb_ko.json())
    sb_en_id = _extract_concept_id(res_sb_en.json())
    llm_id = _extract_concept_id(res_llm.json())
    assert sb_ko_id == sb_en_id == llm_id
    assert "세컨드 브레인" in str(res_sb_en.json())
    assert "Second Brain" in str(res_sb_ko.json())

    assert _extract_concept_id(res_apple.json()) == _extract_concept_id(res_sagwa.json())
    assert _extract_concept_id(res_apple.json()) != sb_ko_id
    assert "RFP" in str(res_rfp.json())

    # 4. Console /api/studio/concepts and /api/app/concepts visibility + unlink
    studio_list = client.studio_concepts_request("").assert_status(200)
    assert "세컨드 브레인" in studio_list.text
    app_list = client.app_concepts_request("").assert_status(200)
    assert "apple" in app_list.text

    openrag_cli.concept_unlink("llmwiki", "Second Brain", env=cli_env, check=True)
    after_unlink = mcp_http.concept_resolve("Second Brain", include_aliases=True)
    after_str = str(after_unlink.json()) if not after_unlink.is_error else ""
    assert sb_ko_id not in after_str and "Second Brain" not in str(
        client.get_concept(sb_ko_id).assert_status(200).json()
    )


@pytest.mark.tier1
@pytest.mark.f36
def test_f36_auto_discovery_parenthetical_pairs_on_ingest_and_semantic_match(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "autodisc")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f36-auto-{uuid.uuid4().hex[:6]}", "name": "AutoConcept", "access_scope": "dept", "min_auth_level": 3}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    pdoc = sample_factory.parenthetical_alias_doc(
        pairs=(("제안요청서", "RFP"), ("지식그래프", "Knowledge Graph")),
        extra_paragraphs=("기관 문서 표준화는 제안요청서(RFP) 작성과 지식그래프(Knowledge Graph) 구축을 포함한다.",),
    )
    hwpx_path = sample_factory.write_bytes("paren_concepts.hwpx", pdoc["hwpx_bytes"])
    client.upload_document(
        bid,
        filename=hwpx_path.name,
        content=hwpx_path.read_bytes(),
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202))

    mcp = McpHttpClient(client)
    mcp.handshake()
    rfp_res = mcp.concept_resolve("RFP", include_aliases=True)
    kg_res = mcp.concept_resolve("Knowledge Graph", include_aliases=True)
    assert not rfp_res.is_error
    assert not kg_res.is_error
    assert "제안요청서" in str(rfp_res.json())
    assert "지식그래프" in str(kg_res.json())

    # 128d semantic similarity match
    base_vec = _unit_vec_128(0)
    created = client.create_concept(
        {
            "canonical_name": "semantic-paraphrase-target",
            "aliases": ["의미유사개념"],
            "embedding": base_vec,
        }
    ).assert_status((200, 201)).json()
    sem_cid = _extract_concept_id(created)

    cos_095 = 0.95
    query_vec = [0.0] * 128
    query_vec[0] = cos_095
    query_vec[1] = math.sqrt(1.0 - cos_095 * cos_095)

    sem_res = mcp.concept_resolve(
        "uncatalogued-paraphrase-query",
        use_semantic=True,
        embedding=query_vec,
        top_k=3,
    )
    assert not sem_res.is_error
    sem_data = sem_res.json()
    assert sem_cid in str(sem_data) or "semantic-paraphrase-target" in str(sem_data)


@pytest.mark.tier1
@pytest.mark.f36
def test_f36_search_bm25_query_expansion_cross_lingual_and_domain_synonyms(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "search")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f36-srch-{uuid.uuid4().hex[:6]}", "name": "ConceptSearch", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    doc_a = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "PKM Note",
                "text": "llmwiki 기반 개인 지식 아키텍처 정리 문서입니다.",
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )
    doc_b = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "Orchard Supply",
                "text": "fresh apple orchard supply report for autumn harvest",
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )
    doc_c = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "Cafeteria Menu",
                "text": "구내식당 주간 식단표 안내",
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )

    # Non-vacuous baseline before linking
    pre_hits = _extract_search_hits(
        client.search_bucket(bid, query="세컨드 브레인", top_k=5).assert_status(200).json()
    )
    assert doc_a not in _hit_doc_ids(pre_hits)

    # Link concepts
    client.link_concepts({"source": "llmwiki", "target": "세컨드 브레인"}).assert_status((200, 201))
    client.link_concepts({"source": "apple", "target": "사과"}).assert_status((200, 201))

    # HTTP + MCP query expansion
    post_wiki = client.search_bucket(bid, query="세컨드 브레인", top_k=5).assert_status(200).json()
    wiki_hits = _extract_search_hits(post_wiki)
    assert doc_a in _hit_doc_ids(wiki_hits) or "llmwiki" in _hit_blobs(wiki_hits)
    assert doc_c not in _hit_doc_ids(wiki_hits)

    post_apple = client.search_bucket(bid, query="사과", top_k=5).assert_status(200).json()
    apple_hits = _extract_search_hits(post_apple)
    assert doc_b in _hit_doc_ids(apple_hits) or "apple" in _hit_blobs(apple_hits)
    assert doc_c not in _hit_doc_ids(apple_hits)

    mcp = McpHttpClient(client)
    mcp.handshake()
    mcp_res = mcp.rag_search("세컨드 브레인", bucket=bid, top_k=5)
    assert not mcp_res.is_error
    mcp_hits = _extract_search_hits(mcp_res.json())
    assert doc_a in _hit_doc_ids(mcp_hits) or "llmwiki" in _hit_blobs(mcp_hits)


@pytest.mark.tier1
@pytest.mark.f36
def test_f36_guard_safeword_alias_and_obfuscation_blocking(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "guard")
    client = api_client.with_api_key(api_key)

    client.link_concepts({"source": "사과", "target": "apple"}).assert_status((200, 201))
    rule_resp = client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "사과",
            "action": "block",
            "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"],
        },
    ).assert_status((200, 201)).json()
    rule_id = rule_resp.get("id") or rule_resp.get("rule_id")

    obf = sample_factory.obfuscated_variants("apple")
    mcp = McpHttpClient(client)
    mcp.handshake()

    try:
        for probe in (
            "사과 프로젝트 보고",
            "apple project report",
            f"{obf['fullwidth_nfkc']} report",
            f"{obf['punct_spaced']} report",
        ):
            gt = client.studio_guard_test({"text": probe, "surface": "inbound_query"}).assert_status(200).json()
            assert str(gt.get("decision") or gt.get("action") or "").lower() == "block", (
                f"Expected block for {probe!r}, got {gt!r}"
            )
            chat = client.chat_completions(
                {"model": "openrag-default", "messages": [{"role": "user", "content": probe}]}
            )
            assert chat.status in (400, 403, 422), f"Expected blocked HTTP status for {probe!r}, got {chat.status}"
            assert "guard_blocked" in chat.text.lower() or "block" in chat.text.lower()

            mres = mcp.guard_check(probe, surface="mcp_egress")
            assert mres.is_error is True or str(mres.json().get("decision", "")).lower() == "block"

        benign = "바나나 일반 회의록"
        benign_gt = client.studio_guard_test({"text": benign, "surface": "inbound_query"}).assert_status(200).json()
        assert str(benign_gt.get("decision") or benign_gt.get("action") or "").lower() == "allow"
        client.chat_completions(
            {"model": "openrag-default", "messages": [{"role": "user", "content": benign}]}
        ).assert_status(200)
    finally:
        if rule_id:
            client.studio_pii_request(f"guard-rules/{rule_id}", method="DELETE")


@pytest.mark.tier1
@pytest.mark.f36
def test_f36_wiki_and_memory_entity_resolution_deduplication(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "wikimem")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f36-wm-{uuid.uuid4().hex[:6]}", "name": "WikiMemConcept", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    for alias in ("LLM Wiki", "세컨드 브레인", "Second Brain"):
        client.link_concepts({"source": "llmwiki", "target": alias, "canonical_name": "llmwiki"}).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    r1 = mcp.memory_reflect("llmwiki 링크 구조 정리 MEM_TOKEN_LLMWIKI_ALPHA", scope="dept")
    r2 = mcp.memory_reflect("세컨드 브레인 회상 규칙 MEM_TOKEN_SECONDBRAIN_BETA", scope="dept")
    assert not r1.is_error and not r2.is_error

    md_a = sample_factory.tagged_markdown(
        title="LLM Wiki Architecture",
        tags=["llmwiki"],
        body="LLM Wiki 구조와 양방향 링크 설계 WIKI_TOKEN_ALPHA",
    )
    md_b = sample_factory.tagged_markdown(
        title="Second Brain Workflow",
        tags=["세컨드브레인"],
        body="Second Brain 개인 지식 워크플로우 WIKI_TOKEN_BETA",
    )
    client.sync_documents(
        bid,
        {
            "documents": [
                {"path": "llm-wiki.md", "content": md_a, "access_scope": "dept"},
                {"path": "second-brain.md", "content": md_b, "access_scope": "dept"},
            ]
        },
    ).assert_status((200, 201))

    concepts_resp = client.list_concepts(query="llmwiki").assert_status(200).json()
    matching = [
        c for c in _extract_concept_list(concepts_resp)
        if "llmwiki" in str(c).lower() or "세컨드 브레인" in str(c)
    ]
    unique_ids = {_extract_concept_id(c) for c in matching}
    assert len(unique_ids) == 1, f"Expected single deduplicated canonical concept, got {matching!r}"

    engaged = mcp.memory_engage("Second Brain", scope="dept", top_k=5)
    assert not engaged.is_error
    engaged_str = str(engaged.json())
    assert "MEM_TOKEN_LLMWIKI_ALPHA" in engaged_str
    assert "MEM_TOKEN_SECONDBRAIN_BETA" in engaged_str

    w_res = mcp.wiki_search("세컨드 브레인", bucket=bid, top_k=5)
    assert not w_res.is_error
    w_str = str(w_res.json())
    assert "WIKI_TOKEN_ALPHA" in w_str
    assert "WIKI_TOKEN_BETA" in w_str


# ============================================================================
# F37 — First-Class Tag-Based Search & Faceted Filtering (5 tests)
# ============================================================================


@pytest.mark.tier1
@pytest.mark.f37
def test_f37_schema_008_tags_array_and_gin_indexes_across_four_tables(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    targets = [
        ("corpus", "documents"),
        ("corpus", "chunks"),
        ("wiki", "pages"),
        ("memory", "entries"),
    ]

    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            for schema, table in targets:
                cur.execute(
                    """
                    SELECT data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = 'tags'
                    """,
                    (schema, table),
                )
                col = cur.fetchone()
                assert col is not None, f"Missing tags column on {schema}.{table}"
                assert col[0] == "ARRAY" and col[1] == "_text", f"{schema}.{table}.tags must be TEXT[], got {col}"

                cur.execute(
                    """
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = %s AND tablename = %s
                    """,
                    (schema, table),
                )
                defs = [row[0].lower() for row in cur.fetchall()]
                assert any("using gin" in d and "tags" in d for d in defs), (
                    f"Missing GIN index on {schema}.{table}.tags; indexes={defs}"
                )

            bucket_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
                VALUES (%s, %s, %s, %s, %s, 'F37 Bucket', 'dept', 2, %s)
                """,
                (
                    bucket_id,
                    ctx["org1_id"],
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"f37-bkt-{bucket_id[:8]}",
                    f"arn:openrag:{ctx['slug1']}:knowledge:dept:bucket:L2/{bucket_id}",
                ),
            )
            doc_id = str(uuid.uuid4())
            page_id = str(uuid.uuid4())
            chunk_id = str(uuid.uuid4())
            wiki_id = str(uuid.uuid4())
            ep_id = str(uuid.uuid4())
            mem_id = str(uuid.uuid4())
            vec_lit = _vec_literal(_unit_vec_128(3))

            cur.execute(
                """
                INSERT INTO corpus.documents
                    (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
                VALUES (%s, %s, %s, %s, %s, 'private', 3, %s, %s, 'Tagged Private Doc', %s, ARRAY['tag-a','tag-b']::text[])
                """,
                (
                    doc_id,
                    ctx["org1_id"],
                    bucket_id,
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"arn:openrag:{ctx['slug1']}:knowledge:private:doc:L3/{doc_id}",
                    f"{ctx['org1_id']}:private",
                    _sha256_hex(doc_id),
                ),
            )
            cur.execute(
                """
                INSERT INTO corpus.pages
                    (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, redacted_markdown)
                VALUES (%s, %s, %s, %s, %s, 'private', 3, %s, %s, 1, 'tagged page')
                """,
                (
                    page_id,
                    ctx["org1_id"],
                    doc_id,
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"arn:openrag:{ctx['slug1']}:knowledge:private:page:L3/{page_id}",
                    f"{ctx['org1_id']}:private",
                ),
            )
            cur.execute(
                """
                INSERT INTO corpus.chunks
                    (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, embedding, text_hash, tags)
                VALUES (%s, %s, %s, %s, %s, %s, 'private', 3, %s, %s, 0, 'chunk tag-a', 'chunk tag-a', %s::vector(128), %s, ARRAY['tag-a']::text[])
                """,
                (
                    chunk_id,
                    ctx["org1_id"],
                    doc_id,
                    page_id,
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"arn:openrag:{ctx['slug1']}:knowledge:private:chunk:L3/{chunk_id}",
                    f"{ctx['org1_id']}:private",
                    vec_lit,
                    _sha256_hex(chunk_id),
                ),
            )
            cur.execute(
                """
                INSERT INTO wiki.pages
                    (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding, tags)
                VALUES (%s, %s, %s, %s, %s, 'private', 3, %s, %s, %s, 'Tagged Wiki', 'wiki body', 'wiki body', %s, %s::vector(128), ARRAY['tag-a','tag-b']::text[])
                """,
                (
                    wiki_id,
                    ctx["org1_id"],
                    bucket_id,
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"arn:openrag:{ctx['slug1']}:knowledge:private:wiki:L3/{wiki_id}",
                    f"{ctx['org1_id']}:private",
                    f"tagged-wiki-{wiki_id[:6]}",
                    _sha256_hex(wiki_id),
                    vec_lit,
                ),
            )
            cur.execute(
                """
                INSERT INTO memory.episodes
                    (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
                VALUES (%s, %s, %s, %s, 'private', 3, %s, 'chat_ui', 'ep summary', %s)
                """,
                (
                    ep_id,
                    ctx["org1_id"],
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"arn:openrag:{ctx['slug1']}:knowledge:private:episode:L3/{ep_id}",
                    _sha256_hex(ep_id),
                ),
            )
            cur.execute(
                """
                INSERT INTO memory.entries
                    (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, title, effective_text, bm25_text, content_sha256, embedding, source_episode_id, tags)
                VALUES (%s, %s, %s, %s, %s, 'private', 3, %s, %s, 'fact', 'Tagged Mem', 'mem text', 'mem text', %s, %s::vector(128), %s, ARRAY['tag-a']::text[])
                """,
                (
                    mem_id,
                    ctx["org1_id"],
                    bucket_id,
                    ctx["dept1_id"],
                    ctx["owner_id"],
                    f"arn:openrag:{ctx['slug1']}:knowledge:private:memory:L3/{mem_id}",
                    f"{ctx['org1_id']}:private",
                    _sha256_hex(mem_id),
                    vec_lit,
                    ep_id,
                ),
            )
        conn.commit()

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org1_id"], principal_id=ctx["owner_id"], auth_level=3, dept_ids=(ctx["dept1_id"],)
    )
    peer_ctx = PgPrincipalContext(
        org_id=ctx["org1_id"], principal_id=ctx["peer_id"], auth_level=3, dept_ids=(ctx["dept1_id"],)
    )
    for schema, table in targets:
        owner_rows = pg_ephemeral.query_as_principal(
            f"SELECT id, tags FROM {schema}.{table} WHERE tags @> ARRAY['tag-a']::text[]",
            principal=owner_ctx,
        )
        peer_rows = pg_ephemeral.query_as_principal(
            f"SELECT id, tags FROM {schema}.{table} WHERE tags && ARRAY['tag-a','tag-b']::text[]",
            principal=peer_ctx,
        )
        assert len(owner_rows) >= 1, f"Owner should see tagged row in {schema}.{table}"
        assert peer_rows == [], f"Peer must NOT see private tagged row in {schema}.{table}"


@pytest.mark.tier1
@pytest.mark.f37
def test_f37_manual_tag_add_remove_endpoints_and_app_studio_controls(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "tagcrud")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-crud-{uuid.uuid4().hex[:6]}", "name": "TagCrud", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    doc_id = _extract_doc_id(
        client.create_document(
            bid, {"title": "Tag Target Doc", "text": "태그 추가 삭제 테스트 문서", "access_scope": "dept"}
        ).assert_status((200, 201)).json()
    )

    client.add_document_tags(bid, doc_id, ["policy-2026", "finance"]).assert_status((200, 201))
    listed = client.list_document_tags(bid, doc_id).assert_status(200).json()
    assert "policy-2026" in str(listed) and "finance" in str(listed)

    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tags FROM corpus.documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            assert row is not None and set(row[0]) >= {"policy-2026", "finance"}

    client.remove_document_tags(bid, doc_id, ["finance"]).assert_status(200)
    listed_after = client.list_document_tags(bid, doc_id).assert_status(200).json()
    assert "policy-2026" in str(listed_after)
    assert "finance" not in str(listed_after)

    # Memory tag add/remove
    mcp = McpHttpClient(client)
    mcp.handshake()
    mem_res = mcp.memory_reflect("태그 테스트 메모리 항목", scope="dept")
    assert not mem_res.is_error
    mem_data = mem_res.json()
    mem_id = str(mem_data.get("id") or mem_data.get("memory_id") or mem_data.get("entry_id"))

    client.add_memory_tags(mem_id, ["mem-tag-1", "mem-tag-2"]).assert_status((200, 201))
    client.remove_memory_tags(mem_id, ["mem-tag-2"]).assert_status(200)
    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tags FROM memory.entries WHERE id = %s", (mem_id,))
            mrow = cur.fetchone()
            assert mrow is not None
            assert "mem-tag-1" in mrow[0] and "mem-tag-2" not in mrow[0]

    # Wiki tag add/remove
    wiki_slug = f"tag-wiki-{uuid.uuid4().hex[:6]}"
    wiki_body = "# Tag Target Wiki\n위키 태그 추가 삭제 테스트 본문"
    client.sync_documents(
        bid,
        {
            "documents": [
                {
                    "path": f"{wiki_slug}.md",
                    "slug": wiki_slug,
                    "title": "Tag Target Wiki",
                    "body_md": wiki_body,
                    "body_sha256": _sha256_hex(wiki_body),
                    "access_scope": "dept",
                }
            ]
        },
    ).assert_status((200, 201))
    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM wiki.pages WHERE org_id = %s AND slug = %s",
                (ctx["org1_id"], wiki_slug),
            )
            prow = cur.fetchone()
            assert prow is not None, f"Synced wiki page {wiki_slug} not found in wiki.pages"
            page_id = str(prow[0])

    client.add_wiki_tags(page_id, ["wiki-tag-1", "wiki-tag-2"]).assert_status((200, 201))
    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tags FROM wiki.pages WHERE id = %s", (page_id,))
            wrow = cur.fetchone()
            assert wrow is not None and set(wrow[0]) >= {"wiki-tag-1", "wiki-tag-2"}

    client.remove_wiki_tags(page_id, ["wiki-tag-2"]).assert_status(200)
    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tags FROM wiki.pages WHERE id = %s", (page_id,))
            wrow_after = cur.fetchone()
            assert wrow_after is not None
            assert "wiki-tag-1" in wrow_after[0] and "wiki-tag-2" not in wrow_after[0]

    # Tier 1 /api/app/* & Tier 2 /api/studio/* tag/knowledge controls
    app_wiki_resp = client.app_wiki(page_id).assert_status(200)
    assert "wiki-tag-1" in app_wiki_resp.text and "wiki-tag-2" not in app_wiki_resp.text

    app_mem_resp = client.app_memory_list().assert_status(200)
    assert "mem-tag-1" in app_mem_resp.text and "mem-tag-2" not in app_mem_resp.text

    app_chat_resp = client.app_chat(
        {"message": "태그 추가 삭제 테스트 #policy-2026", "bucket": bid}
    ).assert_status(200)
    assert (
        doc_id in app_chat_resp.text
        or "Tag Target Doc" in app_chat_resp.text
        or "태그 추가 삭제" in app_chat_resp.text
    )

    studio_tags_resp = client.studio_knowledge_request("tags").assert_status(200)
    assert "policy-2026" in studio_tags_resp.text
    assert "wiki-tag-1" in studio_tags_resp.text
    assert "mem-tag-1" in studio_tags_resp.text


@pytest.mark.tier1
@pytest.mark.f37
def test_f37_structured_tag_filtering_any_all_exclude(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "tagflt")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-flt-{uuid.uuid4().hex[:6]}", "name": "TagFilter", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    specs = [
        ("d1", ["hr", "2026", "draft"]),
        ("d2", ["hr", "2026", "approved"]),
        ("d3", ["hr", "2025", "approved"]),
        ("d4", ["finance", "2026", "approved"]),
    ]
    ids: dict[str, str] = {}
    for key, tags in specs:
        resp = client.create_document(
            bid,
            {
                "title": f"Doc {key}",
                "text": f"공통규정키워드T9 문서 {key}",
                "tags": tags,
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
        ids[key] = _extract_doc_id(resp)

    mcp = McpHttpClient(client)
    mcp.handshake()

    any_hits = _hit_doc_ids(
        _extract_search_hits(
            client.search_bucket(
                bid, query="공통규정키워드T9", top_k=10, tags_any=["finance", "2025"]
            ).assert_status(200).json()
        )
    )
    assert any_hits == {ids["d3"], ids["d4"]}

    all_hits = _hit_doc_ids(
        _extract_search_hits(
            client.search_bucket(
                bid, query="공통규정키워드T9", top_k=10, tags_all=["hr", "2026"]
            ).assert_status(200).json()
        )
    )
    assert all_hits == {ids["d1"], ids["d2"]}

    excl_hits = _hit_doc_ids(
        _extract_search_hits(
            mcp.rag_search(
                "공통규정키워드T9",
                bucket=bid,
                top_k=10,
                tags_all=["hr", "approved"],
                tags_exclude=["draft"],
            ).json()
        )
    )
    assert excl_hits == {ids["d2"], ids["d3"]}


@pytest.mark.tier1
@pytest.mark.f37
def test_f37_inline_hash_and_tag_colon_syntax_in_search_chat_and_mcp(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "inline")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-inl-{uuid.uuid4().hex[:6]}", "name": "InlineTag", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    doc_sec = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "보안 공지",
                "text": "공지사항 비밀번호 변경 주기 안내입니다.",
                "tags": ["security", "2026"],
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )
    doc_cafe = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "식단 공지",
                "text": "공지사항 구내식당 운영 시간 안내입니다.",
                "tags": ["cafeteria"],
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )

    for q in ("공지사항 안내 #security", "공지사항 안내 tag:security"):
        hits = _extract_search_hits(
            client.search_bucket(bid, query=q, top_k=10).assert_status(200).json()
        )
        ids = _hit_doc_ids(hits)
        assert doc_sec in ids
        assert doc_cafe not in ids

    chat = client.chat_completions(
        {
            "model": "openrag-default",
            "bucket": bid,
            "include_citations": True,
            "messages": [{"role": "user", "content": "공지사항 안내 #security"}],
        }
    ).assert_status(200).json()
    assert doc_cafe not in str(chat) and "구내식당" not in str(chat)
    assert doc_sec in str(chat) or "비밀번호" in str(chat)

    with mcp_stdio(
        api_key=api_key,
        principal_id=ctx["dept_admin_id"],
        env={"DATABASE_URL": pg_ephemeral.dsn()},
    ) as mcp_s:
        mcp_s.handshake()
        s_res = mcp_s.rag_search("공지사항 안내 tag:security", bucket=bid, top_k=10)
        assert not s_res.is_error
        s_ids = _hit_doc_ids(_extract_search_hits(s_res.json()))
        assert doc_sec in s_ids and doc_cafe not in s_ids

    mcp_h = McpHttpClient(client)
    mcp_h.handshake()
    h_res = mcp_h.rag_search("공지사항 안내 #security", bucket=bid, top_k=10)
    assert not h_res.is_error
    h_ids = _hit_doc_ids(_extract_search_hits(h_res.json()))
    assert doc_sec in h_ids and doc_cafe not in h_ids


@pytest.mark.tier1
@pytest.mark.f37
def test_f37_expand_tag_aliases_and_tag_facet_counts(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["dept_admin_id"], "facets")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-fac-{uuid.uuid4().hex[:6]}", "name": "TagFacets", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    client.link_concepts({"source": "llmwiki", "target": "세컨드브레인"}).assert_status((200, 201))

    doc_wiki = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "Wiki Guide",
                "text": "지식관리 시스템 문서입니다.",
                "tags": ["llmwiki"],
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )
    doc_gen = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "General Guide",
                "text": "지식관리 일반 참고 문서입니다.",
                "tags": ["general"],
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )

    no_exp = _hit_doc_ids(
        _extract_search_hits(
            client.search_bucket(
                bid,
                query="지식관리",
                top_k=10,
                tags_any=["세컨드브레인"],
                expand_tag_aliases=False,
            ).assert_status(200).json()
        )
    )
    assert no_exp == set()

    with_exp = _hit_doc_ids(
        _extract_search_hits(
            client.search_bucket(
                bid,
                query="지식관리 #세컨드브레인",
                top_k=10,
                expand_tag_aliases=True,
            ).assert_status(200).json()
        )
    )
    assert with_exp == {doc_wiki}
    assert doc_gen not in with_exp

    unfiltered = client.search_bucket(bid, query="지식관리", top_k=10).assert_status(200)
    facets = extract_tag_facets(unfiltered)
    assert facets.get("llmwiki") == 1, f"Expected llmwiki facet count 1, got {facets!r}"
    assert facets.get("general") == 1, f"Expected general facet count 1, got {facets!r}"
