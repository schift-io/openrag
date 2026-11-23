"""Tier 2 boundary, negative, and corner-case E2E tests for F36-F37 (Concepts/Aliases & Tags)."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import math
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError, OpenRagCli
from lib.fixtures import FixtureFactory
from lib.http_client import OpenRagHttpClient, extract_tag_facets
from lib.mcp_client import McpHttpClient, McpJsonRpcError
from lib.pg_harness import PgEphemeralCluster, PgPrincipalContext

pytestmark = [pytest.mark.tier2]


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
        assert any(name.startswith("008_") for name in applied)
    elif has_concepts != "t":
        pg._run_psql_file_or_cmd(file_path=mig_008)
        pg.grant_app_role_privileges()
    return pg


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_two_orgs(pg: PgEphemeralCluster) -> dict[str, str]:
    org1_id = str(uuid.uuid4())
    org2_id = str(uuid.uuid4())
    dept1_id = str(uuid.uuid4())
    dept2_id = str(uuid.uuid4())
    owner_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    outsider_id = str(uuid.uuid4())
    slug1 = f"bva1-{org1_id[:8]}"
    slug2 = f"bva2-{org2_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES
            ('{org1_id}', '{slug1}', 'BVA Org1 {slug1}'),
            ('{org2_id}', '{slug2}', 'BVA Org2 {slug2}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
            ('{dept1_id}', '{org1_id}', 'Policy', 'policy'),
            ('{dept2_id}', '{org2_id}', 'Ext', 'ext');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{owner_id}', '{org1_id}', 'owner-{owner_id[:6]}@example.org', 'OwnerL3', 3, false, false),
            ('{peer_id}', '{org1_id}', 'peer-{peer_id[:6]}@example.org', 'PeerL3', 3, false, false),
            ('{admin_id}', '{org1_id}', 'admin-{admin_id[:6]}@example.org', 'AdminL9', 9, true, true),
            ('{outsider_id}', '{org2_id}', 'out-{outsider_id[:6]}@example.org', 'OutL9', 9, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org1_id}', '{dept1_id}', '{owner_id}', 'member'),
            ('{org1_id}', '{dept1_id}', '{peer_id}', 'member'),
            ('{org1_id}', '{dept1_id}', '{admin_id}', 'dept_admin'),
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
        "admin_id": admin_id,
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
    assert isinstance(key, str) and key
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


def _extract_doc_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("document_id", "id", "doc_id"):
            val = payload.get(key)
            if val:
                return str(val)
        if isinstance(payload.get("document"), dict):
            return _extract_doc_id(payload["document"])
    raise AssertionError(f"Could not extract document_id from {payload!r}")


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


# ============================================================================
# F36 BVA / Negative / Adversarial (5 tests)
# ============================================================================


@pytest.mark.tier2
@pytest.mark.f36
def test_f36_bva_reject_empty_whitespace_duplicate_and_self_link_aliases(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "self")
    client = api_client.with_api_key(api_key)

    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    cli_env = {
        "OPENRAG_API_URL": client.base_url,
        "OPENRAG_API_KEY": api_key,
        "DATABASE_URL": pg_ephemeral.dsn(),
    }
    mcp = McpHttpClient(client)
    mcp.handshake()

    for bad_src, bad_tgt in (
        ("", "apple"),
        ("   \t\n ", "apple"),
        ("apple", "   "),
        ("apple", "apple"),
        ("Apple", " apple "),
    ):
        resp = client.link_concepts({"source": bad_src, "target": bad_tgt})
        assert resp.status in (400, 422), (
            f"Expected 400/422 for invalid link ({bad_src!r}, {bad_tgt!r}), got {resp.status}: {resp.text}"
        )

        cli_res = openrag_cli.concept_link(bad_src, bad_tgt, env=cli_env)
        assert cli_res.returncode != 0, (
            f"Expected non-zero CLI exit for invalid link ({bad_src!r}, {bad_tgt!r})"
        )

        try:
            mres = mcp.concept_link(bad_src, bad_tgt)
            assert mres.is_error is True, (
                f"Expected MCP is_error=True for invalid link ({bad_src!r}, {bad_tgt!r})"
            )
        except McpJsonRpcError:
            pass

    # Duplicate link must be idempotent or 409 without duplicate alias rows
    uniq_src = f"dup-apple-{uuid.uuid4().hex[:6]}"
    uniq_tgt = f"중복사과-{uuid.uuid4().hex[:6]}"
    first = client.link_concepts({"source": uniq_src, "target": uniq_tgt}).assert_status((200, 201)).json()
    cid = _extract_concept_id(first)
    second = client.link_concepts({"source": uniq_src, "target": uniq_tgt})
    assert second.status in (200, 201, 409)

    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM wiki.concept_aliases
                WHERE concept_id = %s
                """,
                (cid,),
            )
            cnt = cur.fetchone()
            assert cnt is not None and cnt[0] <= 2, f"Unexpected duplicate alias rows for {cid}: {cnt}"


@pytest.mark.tier2
@pytest.mark.f36
def test_f36_bva_safeword_guard_blocks_nfkc_zero_width_and_punct_obfuscation(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "obfguard")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f36-obf-{uuid.uuid4().hex[:6]}", "name": "ObfGuard", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

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

    obf_en = sample_factory.obfuscated_variants("apple")
    obf_ko = sample_factory.obfuscated_variants("사과")
    probes = [
        obf_en["fullwidth_nfkc"],
        "ＡＰＰＬＥ",
        obf_en["zero_width"],
        obf_ko["zero_width"],
        "a\ufeffpple",
        obf_en["punct_spaced"],
        "a / p / p / l / e",
        obf_en["mixed"],
    ]

    mcp = McpHttpClient(client)
    mcp.handshake()

    try:
        for p in probes:
            gt = client.studio_guard_test({"text": f"report {p}", "surface": "inbound_query"}).assert_status(200).json()
            assert str(gt.get("decision") or gt.get("action") or "").lower() == "block", (
                f"Expected block for obfuscated probe {p!r}, got {gt!r}"
            )
            chat = client.chat_completions(
                {"model": "openrag-default", "messages": [{"role": "user", "content": p}]}
            )
            assert chat.status in (400, 403, 422)
            assert "guard_blocked" in chat.text.lower() or "block" in chat.text.lower()

            mres = mcp.guard_check(p, surface="mcp_egress")
            assert mres.is_error is True or str(mres.json().get("decision", "")).lower() == "block"

            ing = client.create_document(
                bid, {"title": "Blocked Obf", "text": f"secret {p}", "access_scope": "dept"}
            )
            assert ing.status in (400, 403, 422), f"Expected ingestion_gate block for {p!r}, got {ing.status}"

        # Boundary: substring 'appl' in 'application' must NOT be falsely blocked
        benign = "application architecture review"
        benign_gt = client.studio_guard_test({"text": benign, "surface": "inbound_query"}).assert_status(200).json()
        assert str(benign_gt.get("decision") or benign_gt.get("action") or "").lower() == "allow"

        # Audit metadata must never leak raw blocked obfuscated tokens
        with psycopg.connect(pg_ephemeral.dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT metadata::text FROM pii.guard_events WHERE org_id = %s AND decision = 'block'",
                    (ctx["org1_id"],),
                )
                metas = " ".join(row[0] for row in cur.fetchall())
                assert "ａｐｐｌｅ" not in metas
                assert "a\u200bp" not in metas
    finally:
        if rule_id:
            client.studio_pii_request(f"guard-rules/{rule_id}", method="DELETE")


@pytest.mark.tier2
@pytest.mark.f36
def test_f36_bva_unknown_concept_id_and_multi_hop_or_cyclic_alias_resolution(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "cycle")
    client = api_client.with_api_key(api_key)

    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    cli_env = {
        "OPENRAG_API_URL": client.base_url,
        "OPENRAG_API_KEY": api_key,
        "DATABASE_URL": pg_ephemeral.dsn(),
    }

    missing_uuid = "00000000-0000-0000-0000-00000000dead"
    del_resp = client.delete_concept(missing_uuid)
    assert del_resp.status in (400, 404), f"Expected 400/404, got {del_resp.status}"

    cli_un = openrag_cli.concept_unlink("nonexistent-src-999", concept_id=missing_uuid, env=cli_env)
    assert cli_un.returncode != 0

    mcp = McpHttpClient(client)
    mcp.handshake()
    unk_res = mcp.concept_resolve("nonexistent-concept-999", include_aliases=True)
    if not unk_res.is_error:
        unk_data = unk_res.json()
        assert unk_data.get("found") is False or unk_data.get("concept_id") is None or unk_data.get("canonical_id") is None

    # 3-hop chain + cycle A -> B -> C -> A
    tag = uuid.uuid4().hex[:6]
    a, b, c = f"hopA-{tag}", f"hopB-{tag}", f"hopC-{tag}"
    client.link_concepts({"source": a, "target": b}).assert_status((200, 201))
    client.link_concepts({"source": b, "target": c}).assert_status((200, 201))
    cyc_resp = client.link_concepts({"source": c, "target": a})
    assert cyc_resp.status in (200, 201, 409)

    t0 = time.monotonic()
    ra = mcp.concept_resolve(a, include_aliases=True)
    rb = mcp.concept_resolve(b, include_aliases=True)
    rc = mcp.concept_resolve(c, include_aliases=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"Cyclic/multi-hop concept resolution took too long: {elapsed:.2f}s"
    assert not ra.is_error and not rb.is_error and not rc.is_error
    assert _extract_concept_id(ra.json()) == _extract_concept_id(rb.json()) == _extract_concept_id(rc.json())


@pytest.mark.tier2
@pytest.mark.f36
def test_f36_bva_128d_semantic_similarity_threshold_boundary(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "sembva")
    client = api_client.with_api_key(api_key)

    base_vec = [0.0] * 128
    base_vec[0] = 1.0
    created = client.create_concept(
        {
            "canonical_name": f"sem-bva-{uuid.uuid4().hex[:6]}",
            "aliases": ["시맨틱경계개념"],
            "embedding": base_vec,
        }
    ).assert_status((200, 201)).json()
    cid = _extract_concept_id(created)

    def _cos_vec(cos_val: float) -> list[float]:
        v = [0.0] * 128
        v[0] = cos_val
        v[1] = math.sqrt(max(0.0, 1.0 - cos_val * cos_val))
        return v

    above = client.resolve_concept(
        {"term": "unseen-above", "use_semantic": True, "embedding": _cos_vec(0.88), "threshold": 0.85}
    ).assert_status(200).json()
    assert cid in str(above) and above.get("matched", True) is True

    below = client.resolve_concept(
        {"term": "unseen-below", "use_semantic": True, "embedding": _cos_vec(0.82), "threshold": 0.85}
    ).assert_status(200).json()
    assert below.get("matched") is False or cid not in str(below)

    bad_dim = client.resolve_concept(
        {"term": "bad-dim", "use_semantic": True, "embedding": [1.0] + [0.0] * 126}
    )
    assert bad_dim.status in (400, 422), f"Expected 400/422 for 127d vector, got {bad_dim.status}"


@pytest.mark.tier2
@pytest.mark.f36
def test_f36_bva_rls_and_auth_boundary_on_concept_mutations_and_reads(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    owner_key = _mint_api_key(api_client, ctx["org1_id"], ctx["owner_id"], "own")
    peer_key = _mint_api_key(api_client, ctx["org1_id"], ctx["peer_id"], "peer")
    outsider_key = _mint_api_key(api_client, ctx["org2_id"], ctx["outsider_id"], "out")

    owner_cli = api_client.with_api_key(owner_key)
    peer_cli = api_client.with_api_key(peer_key)
    outsider_cli = api_client.with_api_key(outsider_key)

    created = owner_cli.create_concept(
        {
            "canonical_name": "ORG-CODENAME",
            "aliases": ["사내코드명"],
        }
    ).assert_status((200, 201)).json()
    cid = _extract_concept_id(created)

    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM wiki.concept_aliases WHERE concept_id = %s LIMIT 1",
                (cid,),
            )
            alias_row = cur.fetchone()
            assert alias_row is not None, f"Expected persisted alias row for concept {cid}"
            alias_id = str(alias_row[0])

    # Same-org non-creator peer (L3): denied HTTP DELETE/unlink and denied PG DELETE/UPDATE
    assert peer_cli.delete_concept(cid).status in (403, 404)
    assert peer_cli.unlink_concept_alias(cid, alias_id).status in (403, 404)

    peer_ctx = PgPrincipalContext(
        org_id=ctx["org1_id"], principal_id=ctx["peer_id"], auth_level=3, dept_ids=(ctx["dept1_id"],)
    )
    assert (
        pg_ephemeral.query_as_principal(
            "UPDATE wiki.concepts SET display_name = 'Peer Hijack' WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(cid,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "DELETE FROM wiki.concepts WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(cid,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "UPDATE wiki.concept_aliases SET alias_text = 'Peer Hijack' WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(alias_id,),
        )
        == []
    )
    assert (
        pg_ephemeral.query_as_principal(
            "DELETE FROM wiki.concept_aliases WHERE id = %s RETURNING id",
            principal=peer_ctx,
            params=(alias_id,),
        )
        == []
    )

    # Cross-org outsider (L9): denied GET/list/resolve/link/delete/unlink and 0 rows via PG RLS SELECT
    assert outsider_cli.get_concept(cid).status in (403, 404)
    listed = outsider_cli.list_concepts(query="ORG-CODENAME").assert_status(200).text
    assert cid not in listed and "사내코드명" not in listed

    mcp_out = McpHttpClient(outsider_cli)
    mcp_out.handshake()
    r = mcp_out.concept_resolve("사내코드명", include_aliases=True)
    if not r.is_error:
        assert cid not in str(r.json())

    assert outsider_cli.link_concepts({"source": "사내코드명", "target": "hijack", "concept_id": cid}).status in (403, 404)
    assert outsider_cli.delete_concept(cid).status in (403, 404)
    assert outsider_cli.unlink_concept_alias(cid, alias_id).status in (403, 404)

    out_ctx = PgPrincipalContext(
        org_id=ctx["org2_id"],
        principal_id=ctx["outsider_id"],
        auth_level=9,
        dept_ids=(ctx["dept2_id"],),
        dept_admin_ids=(ctx["dept2_id"],),
    )
    assert pg_ephemeral.query_as_principal("SELECT id FROM wiki.concepts WHERE id = %s", principal=out_ctx, params=(cid,)) == []
    assert pg_ephemeral.query_as_principal("SELECT id FROM wiki.concept_aliases WHERE concept_id = %s", principal=out_ctx, params=(cid,)) == []

    # Verify owner's concept remains intact
    owner_got = owner_cli.get_concept(cid).assert_status(200).json()
    assert cid in str(owner_got) and "사내코드명" in str(owner_got)


# ============================================================================
# F37 BVA / Negative / Adversarial (5 tests)
# ============================================================================


@pytest.mark.tier2
@pytest.mark.f37
def test_f37_bva_tag_normalization_dedup_empty_whitespace_and_special_chars(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "tagnorm")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-nrm-{uuid.uuid4().hex[:6]}", "name": "TagNorm", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    doc_id = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "Norm Doc",
                "text": "태그 정규화 테스트",
                "tags": ["  #Policy-2026 ", "policy-2026", "POLICY-2026", "한글_태그"],
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )

    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tags FROM corpus.documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            assert row is not None
            stored = list(row[0])
            assert all(t.strip() != "" for t in stored)
            assert "한글_태그" in stored
            policy_matches = [t for t in stored if t.lower().lstrip("#") == "policy-2026"]
            assert len(policy_matches) == 1, f"Expected deduplicated policy-2026 tag, got {stored!r}"

    for bad_tags in ([""], ["   \t  "]):
        bad_resp = client.add_document_tags(bid, doc_id, bad_tags)
        assert bad_resp.status in (200, 400, 422)
        with psycopg.connect(pg_ephemeral.dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT tags FROM corpus.documents WHERE id = %s", (doc_id,))
                stored_after = list(cur.fetchone()[0])  # type: ignore[index]
                assert all(t.strip() != "" for t in stored_after)


@pytest.mark.tier2
@pytest.mark.f37
def test_f37_bva_conflicting_tags_all_and_tags_exclude_returns_empty_deterministically(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "conflict")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-cnf-{uuid.uuid4().hex[:6]}", "name": "TagConflict", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    doc_id = _extract_doc_id(
        client.create_document(
            bid,
            {"title": "HR 2026", "text": "인사 규정 충돌 테스트", "tags": ["hr", "2026"], "access_scope": "dept"},
        ).assert_status((200, 201)).json()
    )

    r1 = client.search_bucket(
        bid, query="인사 규정", top_k=10, tags_all=["hr"], tags_exclude=["hr"]
    )
    assert r1.status in (200, 400, 422)
    if r1.status == 200:
        assert doc_id not in _hit_doc_ids(_extract_search_hits(r1.json()))
        assert _extract_search_hits(r1.json()) == []

    mcp = McpHttpClient(client)
    mcp.handshake()
    m1 = mcp.rag_search("인사 규정", bucket=bid, top_k=10, tags_any=["hr"], tags_exclude=["hr"])
    if not m1.is_error:
        assert _extract_search_hits(m1.json()) == []


@pytest.mark.tier2
@pytest.mark.f37
def test_f37_bva_nonexistent_tag_returns_empty_results_with_valid_facet_structure(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "nonex")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-nex-{uuid.uuid4().hex[:6]}", "name": "TagNonEx", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    client.create_document(
        bid,
        {"title": "Populated", "text": "실제 존재하는 문서 질의 본문", "tags": ["real-tag"], "access_scope": "dept"},
    ).assert_status((200, 201))

    for resp in (
        client.search_bucket(bid, query="질의", top_k=10, tags_any=["no_such_tag_999"]),
        client.search_bucket(bid, query="질의 #no_such_tag_999", top_k=10),
    ):
        resp.assert_status(200)
        body = resp.json()
        assert _extract_search_hits(body) == []
        assert ("tag_facets" in body) or (
            isinstance(body.get("facets"), dict) and "tags" in body["facets"]
        )
        assert extract_tag_facets(body) == {}


@pytest.mark.tier2
@pytest.mark.f37
def test_f37_bva_inline_tag_edge_cases_punctuation_multiple_and_url_fragments(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    api_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "inledge")
    client = api_client.with_api_key(api_key)

    bkt = client.create_bucket(
        {"slug": f"f37-ied-{uuid.uuid4().hex[:6]}", "name": "InlineEdge", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    d_alpha = _extract_doc_id(
        client.create_document(
            bid, {"title": "Alpha", "text": "규정 알파 확인 문서", "tags": ["alpha"], "access_scope": "dept"}
        ).assert_status((200, 201)).json()
    )
    d_beta = _extract_doc_id(
        client.create_document(
            bid, {"title": "Beta", "text": "규정 베타 확인 문서", "tags": ["beta"], "access_scope": "dept"}
        ).assert_status((200, 201)).json()
    )
    d_both = _extract_doc_id(
        client.create_document(
            bid, {"title": "Both", "text": "규정 알파 베타 동시 문서", "tags": ["alpha", "beta"], "access_scope": "dept"}
        ).assert_status((200, 201)).json()
    )
    d_url = _extract_doc_id(
        client.create_document(
            bid,
            {
                "title": "URL Ref",
                "text": "참고 URL: https://example.org/docs#section1 가이드",
                "tags": ["guide"],
                "access_scope": "dept",
            },
        ).assert_status((200, 201)).json()
    )

    h_comma = _hit_doc_ids(_extract_search_hits(client.search_bucket(bid, query="규정 #alpha, 확인.", top_k=10).assert_status(200).json()))
    assert d_alpha in h_comma and d_both in h_comma and d_beta not in h_comma

    h_semi = _hit_doc_ids(_extract_search_hits(client.search_bucket(bid, query="규정 tag:alpha;", top_k=10).assert_status(200).json()))
    assert d_alpha in h_semi and d_both in h_semi and d_beta not in h_semi

    h_multi = _hit_doc_ids(_extract_search_hits(client.search_bucket(bid, query="#alpha #beta 규정", top_k=10).assert_status(200).json()))
    assert h_multi == {d_both}

    h_url = _hit_doc_ids(
        _extract_search_hits(
            client.search_bucket(bid, query="https://example.org/docs#section1 참고", top_k=10).assert_status(200).json()
        )
    )
    assert d_url in h_url

    malformed = client.search_bucket(bid, query="# tag: 공백", top_k=10)
    assert malformed.status in (200, 400, 422)


@pytest.mark.tier2
@pytest.mark.f37
def test_f37_bva_rls_isolation_on_tag_facet_counts(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_two_orgs(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx["org1_id"], ctx["admin_id"], "facadm")
    peer_key = _mint_api_key(api_client, ctx["org1_id"], ctx["peer_id"], "facpeer")
    admin_cli = api_client.with_api_key(admin_key)
    peer_cli = api_client.with_api_key(peer_key)

    bkt = admin_cli.create_bucket(
        {"slug": f"f37-rlsf-{uuid.uuid4().hex[:6]}", "name": "FacetRls", "access_scope": "dept", "min_auth_level": 1}
    ).assert_status((200, 201)).json()
    bid = str(bkt.get("id") or bkt.get("bucket_id"))

    admin_cli.create_document(
        bid,
        {
            "title": "Public 1",
            "text": "보안공통문서 일반 안내 1",
            "tags": ["public-tag"],
            "access_scope": "dept",
            "min_auth_level": 2,
        },
    ).assert_status((200, 201))
    admin_cli.create_document(
        bid,
        {
            "title": "Public 2",
            "text": "보안공통문서 일반 안내 2",
            "tags": ["shared-tag"],
            "access_scope": "dept",
            "min_auth_level": 2,
        },
    ).assert_status((200, 201))
    admin_cli.create_document(
        bid,
        {
            "title": "Secret L9",
            "text": "보안공통문서 기밀 전용",
            "tags": ["secret-tag-leak", "shared-tag"],
            "access_scope": "private",
            "min_auth_level": 9,
        },
    ).assert_status((200, 201))

    peer_search = peer_cli.search_bucket(bid, query="보안공통문서", top_k=10).assert_status(200)
    peer_facets = extract_tag_facets(peer_search)
    assert "secret-tag-leak" not in peer_facets, f"Peer saw secret tag in facets: {peer_facets!r}"
    assert peer_facets.get("shared-tag") == 1, f"Peer shared-tag count leaked secret doc: {peer_facets!r}"
    assert peer_facets.get("public-tag") == 1

    admin_search = admin_cli.search_bucket(bid, query="보안공통문서", top_k=10).assert_status(200)
    admin_facets = extract_tag_facets(admin_search)
    assert admin_facets.get("secret-tag-leak") == 1
    assert admin_facets.get("shared-tag") == 2

    peer_pg = PgPrincipalContext(
        org_id=ctx["org1_id"], principal_id=ctx["peer_id"], auth_level=3, dept_ids=(ctx["dept1_id"],)
    )
    for schema, table in (
        ("corpus", "documents"),
        ("corpus", "chunks"),
        ("wiki", "pages"),
        ("memory", "entries"),
    ):
        leaked = pg_ephemeral.query_as_principal(
            f"SELECT id FROM {schema}.{table} WHERE tags && ARRAY['secret-tag-leak']::text[]",
            principal=peer_pg,
        )
        assert leaked == [], f"RLS leaked secret-tag-leak row in {schema}.{table}: {leaked!r}"
