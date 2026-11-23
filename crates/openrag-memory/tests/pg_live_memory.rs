use std::fs;
use std::sync::Arc;

use openrag_core::{
    AccessScope, AuthLevel, GuardAction, GuardConfig, GuardRuleConfig, GuardRuleKind,
    GuardSurface, MemoryBackendKind, MemoryConfig, OpenRagError, PrincipalContext,
};
use openrag_db::DbPool;
use openrag_memory::{
    parse_frontmatter, render_frontmatter_markdown, ActorType, ApplyPatchInput, ConceptCategory,
    ConceptGraph, ConceptSource, DefaultMemoryGuard, DeterministicTextEmbedder, EngageRequest,
    InMemoryConceptStore, KumihoMemoryBackend, MarkdownSync, MemoryBackend, MemoryEdgeType,
    MemoryKind, MemoryRouter, PatchOp, PostgresMemoryBackend, PromoteMemoryInput, PromoteTarget,
    RecordEpisodeInput, ReflectRequest, SourceHarness, SyncDocInput, SyncDocOutcome,
    UpsertMemoryInput, WikiCompiler, WikiEditorType, WikiRelationType,
};
use openrag_store::TagFilter;
use sqlx::Row;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

struct SeedMemOrg {
    org_id: Uuid,
    dept_fin: Uuid,
    alice: Uuid,
    bob: Uuid,
    bucket_open: Uuid,
    bucket_restricted: Uuid,
    alice_ctx: PrincipalContext,
    bob_ctx: PrincipalContext,
    admin_ctx: PrincipalContext,
}

async fn seed_mem_org(admin: &DbPool) -> Result<SeedMemOrg, Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_fin = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let bucket_open = Uuid::new_v4();
    let bucket_restricted = Uuid::new_v4();
    let g_denied = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'MemOrg')")
        .bind(org_id)
        .bind(format!("mem-{}", org_id.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Finance', 'fin')",
    )
    .bind(dept_fin)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level) VALUES
         ($1, $4, $5, 'Alice', 7),
         ($2, $4, $6, 'Bob', 9),
         ($3, $4, $7, 'Carol', 8)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(org_id)
    .bind(format!("alice-{}@mem.test", org_id.simple()))
    .bind(format!("bob-{}@mem.test", org_id.simple()))
    .bind(format!("carol-{}@mem.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'member'),
         ($1, $2, $4, 'member'),
         ($1, $2, $5, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_fin)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, group_type) VALUES ($1, $2, 'denied-g', 'permission_group')",
    )
    .bind(g_denied)
    .bind(org_id)
    .execute(admin.inner())
    .await?;
    sqlx::query("INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES ($1, $2, $3)")
        .bind(org_id)
        .bind(g_denied)
        .bind(alice)
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, denied_group_ids, min_auth_level, arn) VALUES
         ($1, $3, $4, $5, 'open-wiki', 'Open Wiki', 'dept', ARRAY[]::UUID[], 3, $7),
         ($2, $3, $4, $6, 'l9-wiki', 'L9 Restricted', 'dept', ARRAY[$9], 9, $8)",
    )
    .bind(bucket_open)
    .bind(bucket_restricted)
    .bind(org_id)
    .bind(dept_fin)
    .bind(alice)
    .bind(carol)
    .bind(format!("arn:openrag:mem:fin:dept:bucket:L3/{bucket_open}"))
    .bind(format!("arn:openrag:mem:fin:dept:bucket:L9/{bucket_restricted}"))
    .bind(g_denied)
    .execute(admin.inner())
    .await?;

    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(7)?);
    alice_ctx.dept_ids = vec![dept_fin];
    alice_ctx.group_ids = vec![g_denied];

    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![dept_fin];

    let mut admin_ctx = PrincipalContext::new(org_id, carol, AuthLevel::new(8)?);
    admin_ctx.dept_ids = vec![dept_fin];
    admin_ctx.dept_admin_ids = vec![dept_fin];

    Ok(SeedMemOrg {
        org_id,
        dept_fin,
        alice,
        bob,
        bucket_open,
        bucket_restricted,
        alice_ctx,
        bob_ctx,
        admin_ctx,
    })
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_pg_memory_cclg_rls_edges_promote_and_mirror() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_mem_org(&admin).await?;

    let concepts = InMemoryConceptStore::new();
    let guard = Arc::new(DefaultMemoryGuard::new(
        GuardConfig {
            enabled: true,
            default_action: GuardAction::Mask,
            allowlist: vec![],
            rules: vec![GuardRuleConfig {
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "PROJECT-ORION".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::IngestionGate, GuardSurface::McpEgress],
                replacement_text: None,
                enabled: true,
                priority: 1,
            }],
        },
        concepts.clone(),
    ));
    let embedder = Arc::new(DeterministicTextEmbedder::default());
    let pg_backend = Arc::new(PostgresMemoryBackend::with_components(
        app.clone(),
        embedder.clone(),
        guard.clone(),
    ));
    let kumiho_backend = Arc::new(KumihoMemoryBackend::with_config(
        None,
        None,
        true,
        None,
        embedder,
        guard,
        concepts,
    ));

    let router = MemoryRouter::from_backends(
        &MemoryConfig {
            backend: MemoryBackendKind::OpenragPg,
            mirror_to: Some(MemoryBackendKind::Kumiho),
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        },
        Some(pg_backend.clone()),
        Some(kumiho_backend.clone()),
    )?;

    // 1. Safe-word guard blocks reflect before any DB write
    let blocked = router
        .reflect(
            &s.alice_ctx,
            ReflectRequest {
                dept_id: s.dept_fin,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                content: "Secret note mentioning P R O J E C T - O R I O N".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec![],
                edge_type: None,
                target_memory_id: None,
                extract_concepts: false,
            },
            None,
        )
        .await;
    assert!(matches!(blocked, Err(OpenRagError::GuardBlocked { .. })));

    // 2. Episode + reflect + mirror_to kumiho
    let ep = router
        .record_episode(
            &s.alice_ctx,
            RecordEpisodeInput {
                dept_id: s.dept_fin,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                source_harness: SourceHarness::ClaudeCode,
                session_key: Some("s1".into()),
                summary_md: "Immutable episode summary".into(),
                raw_redacted_json: None,
            },
            None,
        )
        .await?;

    let r1 = router
        .reflect(
            &s.alice_ctx,
            ReflectRequest {
                dept_id: s.dept_fin,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                content: "사과(apple) 공급망 정책은 세컨드 브레인(llmwiki) 문서에 기록됩니다.".into(),
                title: Some("Supply Policy".into()),
                memory_kind: Some(MemoryKind::Fact),
                episode_id: Some(ep.id),
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec!["llmwiki".into(), "supply".into()],
                edge_type: None,
                target_memory_id: None,
                extract_concepts: true,
            },
            None,
        )
        .await?;
    assert!(r1.extracted_concepts.len() >= 2);
    // Mirrored to Kumiho with exact propagated entry.id and episode_id
    assert!(kumiho_backend.get_entry(&s.alice_ctx, r1.entry.id).await?.is_some());
    assert!(kumiho_backend.get_episode(r1.episode_id).await.is_some());

    // 3. Private RLS isolation: Bob (L9 peer) blocked; Carol (dept_admin L8) allowed
    assert!(pg_backend.get_entry(&s.bob_ctx, r1.entry.id).await?.is_none());
    assert!(pg_backend.get_entry(&s.admin_ctx, r1.entry.id).await?.is_some());

    // 4. Amend updates effective_text without mutating memory.episodes
    let _ = pg_backend
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r1.entry.id,
                patch_op: PatchOp::Amend,
                new_text: Some("수정된 apple 공급망 정책: 분기별 감사 필수.".into()),
                reason: Some("Q2 audit".into()),
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;
    let ep_summary: String =
        sqlx::query_scalar("SELECT summary_md FROM memory.episodes WHERE id = $1")
            .bind(ep.id)
            .fetch_one(admin.inner())
            .await?;
    assert_eq!(ep_summary, "Immutable episode summary");

    // 5. Reflect r2 with DerivedFrom edge & engage with ConceptGraph expansion + #tag
    let r2 = pg_backend
        .reflect(
            &s.alice_ctx,
            ReflectRequest {
                dept_id: s.dept_fin,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                content: "후속 지침: 공급망 감사 보고서 제출".into(),
                title: Some("Followup".into()),
                memory_kind: Some(MemoryKind::Procedure),
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec!["llmwiki".into()],
                edge_type: Some(MemoryEdgeType::DerivedFrom),
                target_memory_id: Some(r1.entry.id),
                extract_concepts: false,
            },
        )
        .await?;
    assert_eq!(
        r2.edge.as_ref().map(|e| e.relation_type),
        Some(MemoryEdgeType::DerivedFrom)
    );

    // Engage for Alice using cross-lingual "사과" + alias-expanded tag "#세컨드브레인"
    let engage_res = pg_backend
        .engage(
            &s.alice_ctx,
            EngageRequest {
                query: "사과 #세컨드브레인".into(),
                scope: None,
                top_k: 5,
                tag_filter: Some(TagFilter {
                    tags_any: vec![],
                    tags_all: vec![],
                    tags_exclude: vec![],
                    expand_tag_aliases: true,
                }),
            },
        )
        .await?;
    assert!(engage_res.memories.iter().any(|m| m.entry.id == r1.entry.id));

    // Bob's engage sees r2 (dept) but NOT r1 (private), and memory.edges RLS hides DerivedFrom edge pointing to private r1!
    let bob_engage = pg_backend
        .engage(
            &s.bob_ctx,
            EngageRequest {
                query: "공급망".into(),
                scope: None,
                top_k: 5,
                tag_filter: None,
            },
        )
        .await?;
    assert!(bob_engage.memories.iter().all(|m| m.entry.id != r1.entry.id));
    if let Some(hit_r2) = bob_engage.memories.iter().find(|m| m.entry.id == r2.entry.id) {
        assert!(hit_r2.edges.is_empty());
    }

    // 6. Promote ladder: restricted bucket rejected, open bucket succeeds + creates wiki.page + outbox
    let forbidden_prom = pg_backend
        .promote(
            &s.alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: r2.entry.id,
                target: PromoteTarget::Wiki,
                target_bucket_id: Some(s.bucket_restricted),
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: Some("denied-page".into()),
                wiki_title: None,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                reason: None,
            },
        )
        .await;
    assert!(matches!(forbidden_prom, Err(OpenRagError::Forbidden(_))));

    let ok_prom = pg_backend
        .promote(
            &s.alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: r2.entry.id,
                target: PromoteTarget::Wiki,
                target_bucket_id: Some(s.bucket_open),
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: Some("supply-audit-guide".into()),
                wiki_title: Some("Supply Audit Guide".into()),
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                reason: Some("team wiki".into()),
            },
        )
        .await?;
    let wiki_id = ok_prom.promoted_wiki_id.ok_or("missing promoted_wiki_id")?;
    let src_cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM wiki.page_sources WHERE page_id = $1 AND source_type = 'memory' AND source_id = $2",
    )
    .bind(wiki_id)
    .bind(r2.entry.id)
    .fetch_one(admin.inner())
    .await?;
    assert_eq!(src_cnt, 1);

    // 7. Retract hides from active view & fails closed on subsequent non-admin patch
    let _ = pg_backend
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r1.entry.id,
                patch_op: PatchOp::Retract,
                new_text: None,
                reason: Some("obsolete".into()),
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;
    assert!(pg_backend.get_entry(&s.alice_ctx, r1.entry.id).await?.is_none());
    let patch_after_retract = pg_backend
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r1.entry.id,
                patch_op: PatchOp::Amend,
                new_text: Some("revive".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await;
    assert!(matches!(patch_after_retract, Err(OpenRagError::Conflict(_))));

    // 8. Hard purge redact clears patch prev_text/new_text
    let _ = pg_backend
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r2.entry.id,
                patch_op: PatchOp::Redact,
                new_text: None,
                reason: Some("pii purge".into()),
                actor_type: ActorType::Redactor,
                hard_purge: true,
            },
        )
        .await?;
    let patch_rows = sqlx::query(
        "SELECT prev_text, new_text FROM memory.patches WHERE memory_entry_id = $1",
    )
    .bind(r2.entry.id)
    .fetch_all(admin.inner())
    .await?;
    for pr in patch_rows {
        assert!(pr.try_get::<Option<String>, _>("prev_text")?.is_none());
        assert!(pr.try_get::<Option<String>, _>("new_text")?.is_none());
    }

    let _unused = (s.alice, s.bob);
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_pg_wiki_kopararel_concepts_and_markdown_sync() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_mem_org(&admin).await?;
    let embedder = DeterministicTextEmbedder::default();

    // 1. ConceptGraph CRUD & merge under RLS
    let mut tx_c = app.begin_rls_tx(&s.alice_ctx).await?;
    let _ = ConceptGraph::link_terms_rls(
        &mut tx_c,
        s.org_id,
        Some(s.alice),
        "llmwiki",
        "LLM Wiki",
        ConceptCategory::Domain,
        ConceptSource::Manual,
    )
    .await?;
    let _ = ConceptGraph::link_terms_rls(
        &mut tx_c,
        s.org_id,
        Some(s.alice),
        "세컨드 브레인",
        "Second Brain",
        ConceptCategory::Domain,
        ConceptSource::Manual,
    )
    .await?;
    let merged = ConceptGraph::link_terms_rls(
        &mut tx_c,
        s.org_id,
        Some(s.alice),
        "llmwiki",
        "세컨드 브레인",
        ConceptCategory::Domain,
        ConceptSource::Manual,
    )
    .await?;
    assert_eq!(merged.aliases.len(), 4);
    tx_c.commit().await?;

    // 2. Markdown import -> No-Op skip -> Export roundtrip -> Conflict detection
    let tmp = tempfile::tempdir()?;
    let vault_in = tmp.path().join("vault_in");
    let vault_out = tmp.path().join("vault_out");
    fs::create_dir_all(&vault_in)?;

    fs::write(
        vault_in.join("expense-guide.md"),
        "---\ntitle: Expense Guide\nslug: expense-guide\nscope: dept\nmin_auth_level: 3\ntags:\n  - finance\n  - llmwiki\n---\n\nSubmit receipts within 7 days.\n",
    )?;
    fs::write(
        vault_in.join("travel-policy.md"),
        "---\ntitle: Travel Policy\nslug: travel-policy\nscope: dept\nmin_auth_level: 3\ntags:\n  - travel\n---\n\nSee [[expense-guide]].\n",
    )?;

    let mut tx1 = app.begin_rls_tx(&s.alice_ctx).await?;
    let rep1 = MarkdownSync::import_markdown_dir_rls(
        &mut tx1,
        &s.alice_ctx,
        &vault_in,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        &embedder,
    )
    .await?;
    assert_eq!(rep1.imported, 2);
    assert_eq!(rep1.edges_created, 1);

    // Re-importing identical directory produces 2 No-Op skips
    let rep2 = MarkdownSync::import_markdown_dir_rls(
        &mut tx1,
        &s.alice_ctx,
        &vault_in,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        &embedder,
    )
    .await?;
    assert_eq!(rep2.skipped_noop, 2);

    // KoParaRel contradicts edge + conflict resolution
    let p_exp = WikiCompiler::get_page_by_slug_rls(&mut tx1, s.org_id, Some(s.bucket_open), "expense-guide")
        .await?
        .ok_or("missing expense-guide")?;
    let p_trv = WikiCompiler::get_page_by_slug_rls(&mut tx1, s.org_id, Some(s.bucket_open), "travel-policy")
        .await?
        .ok_or("missing travel-policy")?;

    let c_edge = WikiCompiler::link_edge_rls(
        &mut tx1,
        s.org_id,
        p_trv.id,
        p_exp.id,
        WikiRelationType::Contradicts,
        0.92,
    )
    .await?;
    let conflicts = WikiCompiler::list_conflicts_rls(&mut tx1, s.org_id).await?;
    assert_eq!(conflicts.len(), 1);
    let resolved = WikiCompiler::resolve_conflict_rls(&mut tx1, s.org_id, c_edge.id).await?;
    assert!(resolved.resolved_at.is_some());

    // Search wiki with alias-expanded tag #세컨드브레인
    let wiki_hits = WikiCompiler::search_pages_rls(
        &mut tx1,
        s.org_id,
        "receipts",
        Some(&TagFilter {
            tags_any: vec!["#세컨드브레인".into()],
            tags_all: vec![],
            tags_exclude: vec![],
            expand_tag_aliases: true,
        }),
        5,
        &embedder,
    )
    .await?;
    assert_eq!(wiki_hits.len(), 1);
    assert_eq!(wiki_hits[0].page.slug, "expense-guide");

    // Export preserves YAML frontmatter + tags
    let exported =
        MarkdownSync::export_markdown_dir_rls(&mut tx1, s.org_id, Some(s.bucket_open), &vault_out)
            .await?;
    assert_eq!(exported.len(), 2);
    let exp_md = fs::read_to_string(vault_out.join("expense-guide.md"))?;
    assert!(exp_md.starts_with("---\n"));
    assert!(exp_md.contains("llmwiki"));

    // Stale base_sha256 conflict detection via sync_documents_rls
    let sync_rep = MarkdownSync::sync_documents_rls(
        &mut tx1,
        &s.alice_ctx,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        &[SyncDocInput {
            file_path: "expense-guide.md".into(),
            slug: "expense-guide".into(),
            title: "Expense Guide".into(),
            body_md: "Conflicting external edit".into(),
            tags: vec!["finance".into()],
            base_sha256: Some("stale-sha-mismatch".into()),
            base_version: None,
        }],
        &embedder,
    )
    .await?;
    assert_eq!(sync_rep.conflicts, 1);
    assert_eq!(sync_rep.results[0].outcome, SyncDocOutcome::Conflict);

    tx1.commit().await?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_pg_transition_oracle_self_supersede_roundtrip_and_mirror_ops() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_mem_org(&admin).await?;

    let pg_backend = Arc::new(PostgresMemoryBackend::new(app.clone()));
    let kumiho_backend = Arc::new(KumihoMemoryBackend::new_offline());
    let router = MemoryRouter::from_backends(
        &MemoryConfig {
            backend: MemoryBackendKind::OpenragPg,
            mirror_to: Some(MemoryBackendKind::Kumiho),
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        },
        Some(pg_backend.clone()),
        Some(kumiho_backend.clone()),
    )?;

    // 1. Router upsert_entry + reflect propagate exact IDs so patch/promote/add_edge work on mirror
    let u1 = router
        .upsert_entry(
            &s.alice_ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id: s.dept_fin,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: Some("Mirror U1".into()),
                text: "Initial mirrored memory".into(),
                tags: vec!["mirror".into()],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
            None,
        )
        .await?;
    assert!(kumiho_backend.get_entry(&s.alice_ctx, u1.id).await?.is_some());

    let _ = router
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: u1.id,
                patch_op: PatchOp::Amend,
                new_text: Some("Amended mirrored memory".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await?;
    assert_eq!(
        kumiho_backend
            .get_entry(&s.alice_ctx, u1.id)
            .await?
            .ok_or("missing u1 in mirror")?
            .effective_text,
        "Amended mirrored memory"
    );

    let _ = router
        .promote(
            &s.alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: u1.id,
                target: PromoteTarget::Dept,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                reason: None,
            },
            None,
        )
        .await?;
    assert_eq!(
        kumiho_backend
            .get_entry(&s.alice_ctx, u1.id)
            .await?
            .ok_or("missing promoted u1 in mirror")?
            .access_scope,
        AccessScope::Dept
    );

    // 1b. Router apply_patch(Supersede) propagates new_entry.id from primary PG to mirror Kumiho
    let _sup_p = router
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: u1.id,
                patch_op: PatchOp::Supersede,
                new_text: Some("Superseded mirrored memory".into()),
                reason: Some("v2".into()),
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await?;
    let pg_sup_by: Uuid = sqlx::query_scalar("SELECT superseded_by FROM memory.entries WHERE id = $1")
        .bind(u1.id)
        .fetch_one(admin.inner())
        .await?;
    assert!(kumiho_backend.get_entry(&s.alice_ctx, pg_sup_by).await?.is_some());
    let _ = router
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: pg_sup_by,
                patch_op: PatchOp::Amend,
                new_text: Some("Superseded + amended on both".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await?;
    assert_eq!(
        kumiho_backend
            .get_entry(&s.alice_ctx, pg_sup_by)
            .await?
            .ok_or("missing superseded+amended entry on mirror")?
            .effective_text,
        "Superseded + amended on both"
    );

    // 2. iam.memory_transition_status uniform 42501 error (zero existence oracle) + self-supersede/version checks
    let priv_entry = pg_backend
        .upsert_entry(
            &s.alice_ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id: s.dept_fin,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "Alice private oracle probe target".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
        )
        .await?;

    let ret_entry = pg_backend
        .upsert_entry(
            &s.alice_ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id: s.dept_fin,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                org_slug: "mem".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "Retracted target".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
        )
        .await?;
    let _ = pg_backend
        .apply_patch(
            &s.alice_ctx,
            ApplyPatchInput {
                memory_entry_id: ret_entry.id,
                patch_op: PatchOp::Retract,
                new_text: None,
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;

    let missing_id = Uuid::new_v4();
    let mut tx_bob = app.begin_rls_tx(&s.bob_ctx).await?;
    let err_missing = sqlx::query("SELECT iam.memory_transition_status($1, $2, 'retracted', NULL, 2)")
        .bind(s.org_id)
        .bind(missing_id)
        .execute(&mut **tx_bob.tx_mut())
        .await
        .err()
        .ok_or("expected missing_id error")?;
    tx_bob.rollback().await?;

    let mut tx_bob2 = app.begin_rls_tx(&s.bob_ctx).await?;
    let err_priv = sqlx::query("SELECT iam.memory_transition_status($1, $2, 'retracted', NULL, 2)")
        .bind(s.org_id)
        .bind(priv_entry.id)
        .execute(&mut **tx_bob2.tx_mut())
        .await
        .err()
        .ok_or("expected priv_entry error")?;
    tx_bob2.rollback().await?;

    let mut tx_bob3 = app.begin_rls_tx(&s.bob_ctx).await?;
    let err_ret = sqlx::query("SELECT iam.memory_transition_status($1, $2, 'retracted', NULL, 3)")
        .bind(s.org_id)
        .bind(ret_entry.id)
        .execute(&mut **tx_bob3.tx_mut())
        .await
        .err()
        .ok_or("expected ret_entry error")?;
    tx_bob3.rollback().await?;

    let msg_missing = err_missing.as_database_error().ok_or("db err")?.message().replace(&missing_id.to_string(), "<ID>");
    let msg_priv = err_priv.as_database_error().ok_or("db err")?.message().replace(&priv_entry.id.to_string(), "<ID>");
    let msg_ret = err_ret.as_database_error().ok_or("db err")?.message().replace(&ret_entry.id.to_string(), "<ID>");
    assert_eq!(msg_missing, msg_priv);
    assert_eq!(msg_missing, msg_ret);

    // Self-supersession & non-monotonic version rejected
    let mut tx_alice = app.begin_rls_tx(&s.alice_ctx).await?;
    let self_sup_err = sqlx::query("SELECT iam.memory_transition_status($1, $2, 'superseded', $2, 2)")
        .bind(s.org_id)
        .bind(priv_entry.id)
        .execute(&mut **tx_alice.tx_mut())
        .await;
    assert!(self_sup_err.is_err());
    tx_alice.rollback().await?;

    let mut tx_alice2 = app.begin_rls_tx(&s.alice_ctx).await?;
    let non_mono_err = sqlx::query("SELECT iam.memory_transition_status($1, $2, 'retracted', NULL, 1)")
        .bind(s.org_id)
        .bind(priv_entry.id)
        .execute(&mut **tx_alice2.tx_mut())
        .await;
    assert!(non_mono_err.is_err());
    tx_alice2.rollback().await?;

    // 3. MarkdownSync export -> import roundtrip without trailing newline yields SkippedNoop
    let embedder = DeterministicTextEmbedder::default();
    let mut tx_wiki = app.begin_rls_tx(&s.alice_ctx).await?;
    let (created_page, _) = WikiCompiler::upsert_page_rls(
        &mut tx_wiki,
        &s.alice_ctx,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        "no-trailing-nl",
        "No Trailing NL",
        "See [[expense-policy|Policy]] and [[hotel-cap]].",
        &["travel".into()],
        WikiEditorType::Human,
        Some("initial"),
        &embedder,
    )
    .await?;
    let exported_md = render_frontmatter_markdown(&created_page)?;
    let parsed_back = parse_frontmatter(&exported_md)?;
    let rt_rep = MarkdownSync::sync_documents_rls(
        &mut tx_wiki,
        &s.alice_ctx,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        &[SyncDocInput {
            file_path: "no-trailing-nl.md".into(),
            slug: created_page.slug.clone(),
            title: created_page.title.clone(),
            body_md: parsed_back.body_md,
            tags: parsed_back.tags,
            base_sha256: None,
            base_version: None,
        }],
        &embedder,
    )
    .await?;
    assert_eq!(rt_rep.skipped_noop, 1);
    assert_eq!(rt_rep.results[0].outcome, SyncDocOutcome::SkippedNoop);

    // 4. MarkdownSync export -> import roundtrip WITH trailing newline yields SkippedNoop
    let (created_nl, _) = WikiCompiler::upsert_page_rls(
        &mut tx_wiki,
        &s.alice_ctx,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        "with-trailing-nl",
        "With Trailing NL",
        "Line 1\n",
        &["travel".into()],
        WikiEditorType::Human,
        Some("initial with nl"),
        &embedder,
    )
    .await?;
    let exported_nl = render_frontmatter_markdown(&created_nl)?;
    let parsed_nl = parse_frontmatter(&exported_nl)?;
    assert_eq!(parsed_nl.body_md, created_nl.body_md);
    let rt_nl = MarkdownSync::sync_documents_rls(
        &mut tx_wiki,
        &s.alice_ctx,
        Some(s.bucket_open),
        s.dept_fin,
        "mem",
        "fin",
        AccessScope::Dept,
        AuthLevel::new(3)?,
        &[SyncDocInput {
            file_path: "with-trailing-nl.md".into(),
            slug: created_nl.slug.clone(),
            title: created_nl.title.clone(),
            body_md: parsed_nl.body_md,
            tags: parsed_nl.tags,
            base_sha256: None,
            base_version: None,
        }],
        &embedder,
    )
    .await?;
    assert_eq!(rt_nl.skipped_noop, 1);
    assert_eq!(rt_nl.results[0].outcome, SyncDocOutcome::SkippedNoop);
    tx_wiki.commit().await?;

    Ok(())
}
