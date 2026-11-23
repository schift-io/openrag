use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use openrag_core::{
    AccessScope, AuthLevel, GuardAction, GuardConfig, GuardRuleConfig, GuardRuleKind,
    GuardSurface, MemoryBackendKind, MemoryConfig, OpenRagError, PrincipalContext,
};
use openrag_memory::{
    compact_term, extract_parenthetical_pairs, extract_wikilinks, normalize_term,
    parse_frontmatter, render_frontmatter_markdown, ActorType, ApplyPatchInput, ConceptCategory,
    ConceptSource, DefaultMemoryGuard, DeterministicTextEmbedder, EngageRequest, EngageResponse,
    InMemoryConceptStore, KumihoMemoryBackend, MemoryBackend, MemoryEdge, MemoryEdgeType,
    MemoryEntry, MemoryEpisode, MemoryGuardHook, MemoryKind, MemoryPatch, MemoryRouter, PatchOp,
    PromoteMemoryInput, PromoteResult, PromoteTarget, RecordEpisodeInput, ReflectRequest,
    ReflectResponse, SourceHarness, UpsertMemoryInput, WikiPage, WikiStatus,
};
use openrag_store::{sha256_hex, TagFilter};
use uuid::Uuid;

#[tokio::test]
async fn concept_graph_cross_lingual_domain_synonyms_and_merge() -> Result<(), Box<dyn std::error::Error>> {
    let store = InMemoryConceptStore::new();
    let org_id = Uuid::new_v4();
    let user_id = Uuid::new_v4();

    // Cross-lingual & domain synonyms
    let c_apple = store
        .link_terms(
            org_id,
            Some(user_id),
            "apple",
            "사과",
            ConceptCategory::General,
            ConceptSource::Manual,
        )
        .await?;
    assert_eq!(c_apple.aliases.len(), 2);

    // Separate concept clusters merged when linked
    let _ = store
        .link_terms(
            org_id,
            Some(user_id),
            "llmwiki",
            "LLM Wiki",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    let _ = store
        .link_terms(
            org_id,
            Some(user_id),
            "세컨드 브레인",
            "Second Brain",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    let merged = store
        .link_terms(
            org_id,
            Some(user_id),
            "llmwiki",
            "세컨드 브레인",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    assert_eq!(merged.aliases.len(), 4);

    let expanded = store
        .expand_terms(org_id, &["세컨드브레인".into()])
        .await;
    assert!(expanded.contains(&"llmwiki".to_string()));
    assert!(expanded.contains(&"second brain".to_string()));

    // Parenthetical extraction: 한글(English) & English(한글)
    let pairs = extract_parenthetical_pairs(
        "제안요청서(RFP) 작성 시 사과(apple)와 LLM Wiki(세컨드 브레인)를 참고하세요.",
    );
    assert_eq!(pairs.len(), 3);
    assert!(pairs.contains(&("제안요청서".into(), "RFP".into())));
    assert!(pairs.contains(&("사과".into(), "apple".into())));
    assert!(pairs.contains(&("LLM Wiki".into(), "세컨드 브레인".into())));

    // NFKC & compact anti-obfuscation
    assert_eq!(normalize_term("  ＡＰＰＬＥ  "), "apple");
    assert_eq!(compact_term("P R O J E C T - O R I O N"), "projectorion");

    // 256d semantic concept similarity
    let emb = DeterministicTextEmbedder::embed_sync("apple 사과")?;
    let matches = store.find_similar_by_vector(org_id, &emb, 0.5, 5).await?;
    assert!(!matches.is_empty());

    // Unlink alias
    assert!(store.unlink_alias(org_id, "Second Brain").await?);
    let after = store.resolve(org_id, "llmwiki").await.ok_or("missing concept")?;
    assert_eq!(after.aliases.len(), 3);
    Ok(())
}

#[tokio::test]
async fn kumiho_memory_backend_cclg_iam_guard_and_edges() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_fin = Uuid::new_v4();
    let dept_hr = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let fin_admin = Uuid::new_v4();

    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(6)?);
    alice_ctx.dept_ids = vec![dept_fin];

    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![dept_fin];

    let mut admin_ctx = PrincipalContext::new(org_id, fin_admin, AuthLevel::new(7)?);
    admin_ctx.dept_ids = vec![dept_fin];
    admin_ctx.dept_admin_ids = vec![dept_fin];

    let mut hr_admin_ctx = PrincipalContext::new(org_id, Uuid::new_v4(), AuthLevel::new(9)?);
    hr_admin_ctx.dept_ids = vec![dept_hr];
    hr_admin_ctx.dept_admin_ids = vec![dept_hr];

    let concepts = InMemoryConceptStore::new();
    concepts
        .link_terms(
            org_id,
            Some(alice),
            "PROJECT-ORION",
            "오리온프로젝트",
            ConceptCategory::SafewordAlias,
            ConceptSource::Manual,
        )
        .await?;

    let guard_cfg = GuardConfig {
        enabled: true,
        default_action: GuardAction::Mask,
        allowlist: vec![],
        rules: vec![GuardRuleConfig {
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "PROJECT-ORION".into(),
            action: GuardAction::Block,
            surfaces: vec![
                GuardSurface::InboundQuery,
                GuardSurface::IngestionGate,
                GuardSurface::McpEgress,
            ],
            replacement_text: None,
            enabled: true,
            priority: 1,
        }],
    };
    let guard = Arc::new(DefaultMemoryGuard::new(guard_cfg, concepts.clone()));
    let embedder = Arc::new(DeterministicTextEmbedder::default());
    let kumiho = Arc::new(KumihoMemoryBackend::with_config(
        None,
        None,
        true,
        None,
        embedder,
        guard,
        concepts.clone(),
    ));

    // Guard blocks safe-word AND its linked alias + obfuscation
    let blocked_alias = kumiho
        .reflect(
            &alice_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "오리온 프로젝트 비밀 메모".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                tags: vec![],
                edge_type: None,
                target_memory_id: None,
                extract_concepts: false,
            },
        )
        .await;
    assert!(matches!(blocked_alias, Err(OpenRagError::GuardBlocked { .. })));

    // Record episode + reflect initial memory
    let ep = kumiho
        .record_episode(
            &alice_ctx,
            RecordEpisodeInput {
                dept_id: dept_fin,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                source_harness: SourceHarness::ClaudeCode,
                session_key: Some("sess-1".into()),
                summary_md: "Initial budget episode".into(),
                raw_redacted_json: None,
            },
        )
        .await?;

    let r1 = kumiho
        .reflect(
            &alice_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "출장 숙박비 상한은 15만원입니다. 제안요청서(RFP) 참고.".into(),
                title: Some("Travel Cap".into()),
                memory_kind: Some(MemoryKind::Decision),
                episode_id: Some(ep.id),
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec!["travel".into(), "llmwiki".into()],
                edge_type: None,
                target_memory_id: None,
                extract_concepts: true,
            },
        )
        .await?;

    // Private isolation: Bob (peer, L9) and HR admin CANNOT read Alice's private memory; FinAdmin CAN
    assert!(kumiho.get_entry(&bob_ctx, r1.entry.id).await?.is_none());
    assert!(kumiho.get_entry(&hr_admin_ctx, r1.entry.id).await?.is_none());
    assert!(kumiho.get_entry(&admin_ctx, r1.entry.id).await?.is_some());

    // Amend preserves episode immutability
    let _ = kumiho
        .apply_patch(
            &alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r1.entry.id,
                patch_op: PatchOp::Amend,
                new_text: Some("출장 숙박비 상한은 18만원으로 상향되었습니다.".into()),
                reason: Some("2026 revision".into()),
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;
    let ep_after = kumiho.get_episode(ep.id).await.ok_or("episode missing")?;
    assert_eq!(ep_after.summary_md, "Initial budget episode");

    // Reflect with Supersedes edge marks r1 superseded & creates edge
    let r2 = kumiho
        .reflect(
            &alice_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "출장 숙박비 상한은 20만원(서울 기준)입니다.".into(),
                title: Some("Travel Cap v2".into()),
                memory_kind: Some(MemoryKind::Correction),
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec!["travel".into()],
                edge_type: Some(MemoryEdgeType::Supersedes),
                target_memory_id: Some(r1.entry.id),
                extract_concepts: false,
            },
        )
        .await?;
    assert_eq!(
        r2.edge.as_ref().map(|e| e.relation_type),
        Some(MemoryEdgeType::Supersedes)
    );
    // r1 is now superseded -> hidden from active view
    assert!(kumiho.get_entry(&alice_ctx, r1.entry.id).await?.is_none());
    // Patching superseded entry fails closed
    let patch_sup = kumiho
        .apply_patch(
            &alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r1.entry.id,
                patch_op: PatchOp::Amend,
                new_text: Some("x".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await;
    assert!(matches!(patch_sup, Err(OpenRagError::Conflict(_))));

    // Invalid patch_op & edge_type rejected fail-closed
    assert!(PatchOp::from_str("drop_table").is_err());
    assert!(MemoryEdgeType::from_str("random_edge").is_err());

    // Engage with inline tag + router works
    let router = MemoryRouter::from_backends(
        &MemoryConfig {
            backend: MemoryBackendKind::Kumiho,
            mirror_to: None,
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        },
        None,
        Some(kumiho.clone()),
    )?;
    let engage = router
        .engage(
            &bob_ctx,
            EngageRequest {
                query: "숙박비 #travel".into(),
                scope: None,
                top_k: 5,
                tag_filter: Some(TagFilter::default()),
            },
            None,
        )
        .await?;
    assert_eq!(engage.memories.len(), 1);
    assert_eq!(engage.memories[0].entry.id, r2.entry.id);
    // r1 is Alice's Private memory -> Bob MUST NOT receive the Supersedes edge leaking r1's UUID!
    assert!(engage.memories[0].edges.is_empty());

    // Alice (owner) DOES see the Supersedes edge on r2
    let alice_engage = router
        .engage(
            &alice_ctx,
            EngageRequest {
                query: "숙박비 #travel".into(),
                scope: None,
                top_k: 5,
                tag_filter: Some(TagFilter::default()),
            },
            None,
        )
        .await?;
    assert_eq!(alice_engage.memories[0].edges.len(), 1);

    // Promote dept -> org works; demote rejected
    let prom = router
        .promote(
            &alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: r2.entry.id,
                target: PromoteTarget::Org,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                reason: Some("org-wide policy".into()),
            },
            None,
        )
        .await?;
    assert_eq!(prom.entry.access_scope, AccessScope::Org);

    let demote = router
        .promote(
            &alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: r2.entry.id,
                target: PromoteTarget::Dept,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                reason: None,
            },
            None,
        )
        .await;
    assert!(matches!(demote, Err(OpenRagError::Validation(_))));

    // Hard purge clears patch history
    let _ = kumiho
        .apply_patch(
            &alice_ctx,
            ApplyPatchInput {
                memory_entry_id: r2.entry.id,
                patch_op: PatchOp::Redact,
                new_text: None,
                reason: Some("secret leak cleanup".into()),
                actor_type: ActorType::Redactor,
                hard_purge: true,
            },
        )
        .await?;
    let patches = kumiho.list_patches_for_entry(r2.entry.id).await;
    assert!(patches
        .iter()
        .all(|p| p.prev_text.is_none() && p.new_text.is_none()));
    Ok(())
}

#[test]
fn markdown_frontmatter_wikilinks_and_path_traversal_guard() -> Result<(), Box<dyn std::error::Error>> {
    let md = "---\ntitle: Travel Policy\nslug: travel-policy\nscope: dept\nmin_auth_level: 4\ntags:\n  - \"#Travel\"\n  - finance\naliases:\n  - 출장규정\n---\n\nSee [[expense-guide|Expense Guide]] and [[approval-flow]].\n";
    let fm = parse_frontmatter(md)?;
    assert_eq!(fm.title.as_deref(), Some("Travel Policy"));
    assert_eq!(fm.scope, Some(AccessScope::Dept));
    assert_eq!(fm.min_auth_level, Some(4));
    assert_eq!(fm.tags, vec!["finance", "travel"]);
    assert_eq!(fm.aliases, vec!["출장규정"]);

    let links = extract_wikilinks(&fm.body_md);
    assert_eq!(links, vec!["expense-guide", "approval-flow"]);

    // Path traversal rejected
    assert!(openrag_memory::MarkdownSync::validate_safe_relative_path("../etc/passwd").is_err());
    assert!(openrag_memory::MarkdownSync::validate_safe_relative_path("/etc/passwd").is_err());
    assert!(openrag_memory::MarkdownSync::validate_safe_relative_path("notes/guide.md").is_ok());

    // Symlink check via temp dir
    let dir = tempfile::tempdir()?;
    let target = dir.path().join("outside.md");
    fs::write(&target, "# Outside")?;
    let vault = dir.path().join("vault");
    fs::create_dir_all(&vault)?;
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&target, vault.join("evil.md"))?;
        let meta = fs::symlink_metadata(vault.join("evil.md"))?;
        assert!(meta.file_type().is_symlink());
    }
    let _unused: PathBuf = vault;
    Ok(())
}

#[tokio::test]
async fn default_memory_guard_mask_anti_obfuscation_case_and_aliases() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let concepts = InMemoryConceptStore::new();
    concepts
        .link_terms(
            org_id,
            None,
            "PROJECT-ORION",
            "오리온프로젝트",
            ConceptCategory::SafewordAlias,
            ConceptSource::Manual,
        )
        .await?;

    let guard_cfg = GuardConfig {
        enabled: true,
        default_action: GuardAction::Mask,
        allowlist: vec!["safe-project-orion".into()],
        rules: vec![
            GuardRuleConfig {
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "PROJECT-ORION".into(),
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::IngestionGate],
                replacement_text: None,
                enabled: true,
                priority: 1,
            },
            GuardRuleConfig {
                kind: GuardRuleKind::Keyword,
                pattern_or_category: "SECRET-WORD".into(),
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::IngestionGate],
                replacement_text: Some("[REDACTED_KW]".into()),
                enabled: true,
                priority: 2,
            },
        ],
    };
    let guard = DefaultMemoryGuard::new(guard_cfg, concepts);

    // Case-insensitive match
    let d1 = guard
        .inspect(org_id, "contains secret-word here", GuardSurface::IngestionGate)
        .await?;
    assert_eq!(d1.action, GuardAction::Mask);
    assert_eq!(d1.sanitized_text.as_deref(), Some("contains [REDACTED_KW] here"));

    // Fullwidth NFKC
    let d2 = guard
        .inspect(org_id, "ＰＲＯＪＥＣＴ－ＯＲＩＯＮ secret", GuardSurface::IngestionGate)
        .await?;
    assert_eq!(d2.action, GuardAction::Mask);
    assert_eq!(d2.sanitized_text.as_deref(), Some("[MASKED] secret"));

    // Zero-width characters
    let d3 = guard
        .inspect(org_id, "P\u{200b}R\u{200b}O\u{200b}JECT-ORION", GuardSurface::IngestionGate)
        .await?;
    assert_eq!(d3.action, GuardAction::Mask);
    assert_eq!(d3.sanitized_text.as_deref(), Some("[MASKED]"));

    // Spaced obfuscation
    let d4 = guard
        .inspect(org_id, "leak P R O J E C T - O R I O N end", GuardSurface::IngestionGate)
        .await?;
    assert_eq!(d4.action, GuardAction::Mask);
    assert_eq!(d4.sanitized_text.as_deref(), Some("leak [MASKED] end"));

    // Linked alias with space
    let d5 = guard
        .inspect(org_id, "오리온 프로젝트 문서", GuardSurface::IngestionGate)
        .await?;
    assert_eq!(d5.action, GuardAction::Mask);
    assert_eq!(d5.sanitized_text.as_deref(), Some("[MASKED] 문서"));

    // Allowlist preserves allowlisted occurrence while masking non-allowlisted occurrence
    let d6 = guard
        .inspect(
            org_id,
            "safe-project-orion ok, but PROJECT-ORION masked",
            GuardSurface::IngestionGate,
        )
        .await?;
    assert_eq!(d6.action, GuardAction::Mask);
    assert_eq!(
        d6.sanitized_text.as_deref(),
        Some("safe-project-orion ok, but [MASKED] masked")
    );
    Ok(())
}

#[tokio::test]
async fn kumiho_reflect_auth_atomicity_apply_patch_mask_and_parity_guards() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_fin = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();

    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(7)?);
    alice_ctx.dept_ids = vec![dept_fin];

    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(7)?);
    bob_ctx.dept_ids = vec![dept_fin];

    let concepts = InMemoryConceptStore::new();
    let guard_cfg = GuardConfig {
        enabled: true,
        default_action: GuardAction::Mask,
        allowlist: vec![],
        rules: vec![GuardRuleConfig {
            kind: GuardRuleKind::PiiCategory,
            pattern_or_category: "rrn_kr".into(),
            action: GuardAction::Mask,
            surfaces: vec![GuardSurface::IngestionGate],
            replacement_text: Some("[REDACTED_RRN]".into()),
            enabled: true,
            priority: 1,
        }],
    };
    let guard = Arc::new(DefaultMemoryGuard::new(guard_cfg, concepts.clone()));
    let embedder = Arc::new(DeterministicTextEmbedder::default());
    let kumiho = KumihoMemoryBackend::with_config(
        None,
        None,
        true,
        None,
        embedder,
        guard,
        concepts.clone(),
    );

    // Alice creates Dept memory m1 and Private memory m_priv
    let m1 = kumiho
        .reflect(
            &alice_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "Authoritative dept policy".into(),
                title: Some("Policy".into()),
                memory_kind: Some(MemoryKind::Decision),
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec!["policy".into()],
                edge_type: None,
                target_memory_id: None,
                extract_concepts: false,
            },
        )
        .await?;

    let m_priv = kumiho
        .reflect(
            &alice_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "Alice private secret".into(),
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
        )
        .await?;

    // 1. Critical: Peer Bob without Write cannot supersede Alice's Dept memory via reflect(Supersedes)
    //    AND atomicity: zero orphaned entries/episodes/concepts created when Forbidden!
    let bob_before = kumiho.list_entries(&bob_ctx, None, 100).await?.len();
    let bob_sup = kumiho
        .reflect(
            &bob_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "Hijacked policy 사과(apple)".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec![],
                edge_type: Some(MemoryEdgeType::Supersedes),
                target_memory_id: Some(m1.entry.id),
                extract_concepts: true,
            },
        )
        .await;
    assert!(matches!(bob_sup, Err(OpenRagError::Forbidden(_))));
    assert_eq!(kumiho.list_entries(&bob_ctx, None, 100).await?.len(), bob_before);
    assert!(kumiho.get_entry(&alice_ctx, m1.entry.id).await?.is_some());
    assert!(concepts.resolve(org_id, "apple").await.is_none());

    // Bob also cannot link an edge to Alice's private memory via reflect(DerivedFrom) without leaking/committing
    let bob_priv_link = kumiho
        .reflect(
            &bob_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "Bob note referencing private".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(4)?,
                tags: vec![],
                edge_type: Some(MemoryEdgeType::DerivedFrom),
                target_memory_id: Some(m_priv.entry.id),
                extract_concepts: false,
            },
        )
        .await;
    assert!(matches!(bob_priv_link, Err(OpenRagError::Forbidden(_))));
    assert_eq!(kumiho.list_entries(&bob_ctx, None, 100).await?.len(), bob_before);

    // 2. High: apply_patch(Amend & Supersede) apply GuardAction::Mask sanitized_text
    let amend_p = kumiho
        .apply_patch(
            &alice_ctx,
            ApplyPatchInput {
                memory_entry_id: m1.entry.id,
                patch_op: PatchOp::Amend,
                new_text: Some("Updated with RRN 900101-1234567".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;
    assert_eq!(amend_p.new_text.as_deref(), Some("Updated with RRN [REDACTED_RRN]"));
    let m1_after = kumiho
        .get_entry(&alice_ctx, m1.entry.id)
        .await?
        .ok_or("missing m1")?;
    assert_eq!(m1_after.effective_text, "Updated with RRN [REDACTED_RRN]");
    assert_eq!(m1_after.bm25_text, "Updated with RRN [REDACTED_RRN]");

    let sup_p = kumiho
        .apply_patch(
            &alice_ctx,
            ApplyPatchInput {
                memory_entry_id: m1.entry.id,
                patch_op: PatchOp::Supersede,
                new_text: Some("Superseded RRN 880202-2345678".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;
    assert_eq!(sup_p.new_text.as_deref(), Some("Superseded RRN [REDACTED_RRN]"));

    // 3. Medium: promote(Group) rejects empty target_group_ids & raising min_auth_level above ctx.auth_level
    let m_dept = kumiho
        .reflect(
            &alice_ctx,
            ReflectRequest {
                dept_id: dept_fin,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "Promote parity test".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                tags: vec![],
                edge_type: Some(MemoryEdgeType::DerivedFrom),
                target_memory_id: Some(m_priv.entry.id),
                extract_concepts: false,
            },
        )
        .await?;

    let empty_grp = kumiho
        .promote(
            &alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: m_dept.entry.id,
                target: PromoteTarget::Group,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                reason: None,
            },
        )
        .await;
    assert!(matches!(empty_grp, Err(OpenRagError::Validation(_))));

    let raise_lvl = kumiho
        .promote(
            &alice_ctx,
            PromoteMemoryInput {
                memory_entry_id: m_dept.entry.id,
                target: PromoteTarget::Org,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: Some(AuthLevel::new(9)?),
                wiki_slug: None,
                wiki_title: None,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                reason: None,
            },
        )
        .await;
    assert!(matches!(raise_lvl, Err(OpenRagError::Forbidden(_))));

    // 4. High & Medium: engage and list_edges hide private/retracted endpoints & retracted entry returns []
    let bob_eng = kumiho
        .engage(
            &bob_ctx,
            EngageRequest {
                query: "Promote parity test".into(),
                scope: None,
                top_k: 5,
                tag_filter: None,
            },
        )
        .await?;
    let hit = bob_eng
        .memories
        .iter()
        .find(|h| h.entry.id == m_dept.entry.id)
        .ok_or("missing m_dept in engage")?;
    assert!(hit.edges.is_empty());

    assert_eq!(kumiho.list_edges(&alice_ctx, m_dept.entry.id).await?.len(), 1);
    let _ = kumiho
        .apply_patch(
            &alice_ctx,
            ApplyPatchInput {
                memory_entry_id: m_dept.entry.id,
                patch_op: PatchOp::Retract,
                new_text: None,
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await?;
    assert!(kumiho.list_edges(&alice_ctx, m_dept.entry.id).await?.is_empty());
    Ok(())
}

#[test]
fn markdown_export_import_roundtrip_sha256_noop() -> Result<(), Box<dyn std::error::Error>> {
    let now = chrono::Utc::now();
    let org_id = Uuid::new_v4();
    let page = WikiPage {
        id: Uuid::new_v4(),
        org_id,
        bucket_id: None,
        dept_id: Uuid::new_v4(),
        owner_principal_id: Uuid::new_v4(),
        access_scope: AccessScope::Dept,
        allowed_group_ids: vec![],
        denied_group_ids: vec![],
        min_auth_level: AuthLevel::new(3)?,
        arn: "arn:openrag:acme:fin:dept:wiki:L3/1".into(),
        acl_scope_key: "fin:dept:L3".into(),
        slug: "travel-cap".into(),
        title: "Travel Cap".into(),
        body_md: "See [[expense-policy|Policy]] and [[hotel-cap]].".into(),
        bm25_text: "Travel Cap\nSee [[expense-policy|Policy]] and [[hotel-cap]].".into(),
        body_sha256: sha256_hex("See [[expense-policy|Policy]] and [[hotel-cap]].".as_bytes()),
        tags: vec!["travel".into()],
        status: WikiStatus::Active,
        version: 1,
        created_at: now,
        updated_at: now,
    };
    let exported = render_frontmatter_markdown(&page)?;
    let reimported = parse_frontmatter(&exported)?;
    assert_eq!(reimported.body_md, page.body_md);
    assert_eq!(sha256_hex(reimported.body_md.as_bytes()), page.body_sha256);

    // Trailing newline input normalized consistently across parse -> render -> parse
    let parsed_nl = parse_frontmatter("---\ntitle: Trailing\n---\n\nLine 1\n")?;
    assert_eq!(parsed_nl.body_md, "Line 1");
    let page_nl = WikiPage {
        body_md: parsed_nl.body_md.clone(),
        body_sha256: sha256_hex(parsed_nl.body_md.as_bytes()),
        ..page
    };
    let exported_nl = render_frontmatter_markdown(&page_nl)?;
    let reimported_nl = parse_frontmatter(&exported_nl)?;
    assert_eq!(reimported_nl.body_md, page_nl.body_md);
    assert_eq!(sha256_hex(reimported_nl.body_md.as_bytes()), page_nl.body_sha256);
    Ok(())
}

struct MirrorWrapperBackend {
    kind: MemoryBackendKind,
    inner: Arc<KumihoMemoryBackend>,
}

#[async_trait::async_trait]
impl MemoryBackend for MirrorWrapperBackend {
    fn kind(&self) -> MemoryBackendKind {
        self.kind
    }
    async fn record_episode(
        &self,
        ctx: &PrincipalContext,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError> {
        self.inner.record_episode(ctx, input).await
    }
    async fn upsert_entry(
        &self,
        ctx: &PrincipalContext,
        input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        self.inner.upsert_entry(ctx, input).await
    }
    async fn get_entry(
        &self,
        ctx: &PrincipalContext,
        entry_id: Uuid,
    ) -> Result<Option<MemoryEntry>, OpenRagError> {
        self.inner.get_entry(ctx, entry_id).await
    }
    async fn list_entries(
        &self,
        ctx: &PrincipalContext,
        tag_filter: Option<&TagFilter>,
        limit: i64,
    ) -> Result<Vec<MemoryEntry>, OpenRagError> {
        self.inner.list_entries(ctx, tag_filter, limit).await
    }
    async fn apply_patch(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
    ) -> Result<MemoryPatch, OpenRagError> {
        self.inner.apply_patch(ctx, input).await
    }
    async fn apply_patch_with_new_id(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
        new_entry_id: Option<Uuid>,
    ) -> Result<MemoryPatch, OpenRagError> {
        self.inner
            .apply_patch_with_new_id(ctx, input, new_entry_id)
            .await
    }
    async fn promote(
        &self,
        ctx: &PrincipalContext,
        input: PromoteMemoryInput,
    ) -> Result<PromoteResult, OpenRagError> {
        self.inner.promote(ctx, input).await
    }
    async fn search(
        &self,
        ctx: &PrincipalContext,
        query: &str,
        tag_filter: Option<&TagFilter>,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>, OpenRagError> {
        self.inner.search(ctx, query, tag_filter, top_k).await
    }
    async fn engage(
        &self,
        ctx: &PrincipalContext,
        req: EngageRequest,
    ) -> Result<EngageResponse, OpenRagError> {
        self.inner.engage(ctx, req).await
    }
    async fn reflect(
        &self,
        ctx: &PrincipalContext,
        req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError> {
        self.inner.reflect(ctx, req).await
    }
    async fn add_edge(
        &self,
        ctx: &PrincipalContext,
        src_memory_id: Uuid,
        dst_memory_id: Uuid,
        relation_type: MemoryEdgeType,
        confidence: f32,
    ) -> Result<MemoryEdge, OpenRagError> {
        self.inner
            .add_edge(ctx, src_memory_id, dst_memory_id, relation_type, confidence)
            .await
    }
    async fn list_edges(
        &self,
        ctx: &PrincipalContext,
        memory_id: Uuid,
    ) -> Result<Vec<MemoryEdge>, OpenRagError> {
        self.inner.list_edges(ctx, memory_id).await
    }
    async fn record_episode_with_id(
        &self,
        ctx: &PrincipalContext,
        episode_id: Option<Uuid>,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError> {
        self.inner
            .record_episode_with_id(ctx, episode_id, input)
            .await
    }
    async fn upsert_entry_with_id(
        &self,
        ctx: &PrincipalContext,
        entry_id: Option<Uuid>,
        input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        self.inner.upsert_entry_with_id(ctx, entry_id, input).await
    }
    async fn reflect_with_ids(
        &self,
        ctx: &PrincipalContext,
        entry_id: Option<Uuid>,
        episode_id: Option<Uuid>,
        req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError> {
        self.inner
            .reflect_with_ids(ctx, entry_id, episode_id, req)
            .await
    }
}

#[tokio::test]
async fn memory_router_mirror_propagates_ids_for_patch_promote_and_edge() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let mut ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(7)?);
    ctx.dept_ids = vec![dept_id];

    let primary_inner = Arc::new(KumihoMemoryBackend::new_offline());
    let mirror = Arc::new(KumihoMemoryBackend::new_offline());
    let primary_wrapped: Arc<dyn MemoryBackend> = Arc::new(MirrorWrapperBackend {
        kind: MemoryBackendKind::OpenragPg,
        inner: primary_inner,
    });

    let router = MemoryRouter::from_backends(
        &MemoryConfig {
            backend: MemoryBackendKind::OpenragPg,
            mirror_to: Some(MemoryBackendKind::Kumiho),
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        },
        Some(primary_wrapped),
        Some(mirror.clone()),
    )?;

    let e1 = router
        .upsert_entry(
            &ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "First mirrored entry".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
            None,
        )
        .await?;
    assert!(mirror.get_entry(&ctx, e1.id).await?.is_some());

    let r2 = router
        .reflect(
            &ctx,
            ReflectRequest {
                dept_id,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                content: "Reflected mirrored entry".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                tags: vec![],
                edge_type: Some(MemoryEdgeType::DerivedFrom),
                target_memory_id: Some(e1.id),
                extract_concepts: false,
            },
            None,
        )
        .await?;
    assert!(mirror.get_entry(&ctx, r2.entry.id).await?.is_some());
    assert!(mirror.get_episode(r2.episode_id).await.is_some());

    // Subsequent amend + promote + add_edge succeed on mirror with matching IDs
    let _ = router
        .apply_patch(
            &ctx,
            ApplyPatchInput {
                memory_entry_id: e1.id,
                patch_op: PatchOp::Amend,
                new_text: Some("Amended mirrored entry".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await?;
    assert_eq!(
        mirror
            .get_entry(&ctx, e1.id)
            .await?
            .ok_or("missing e1 on mirror")?
            .effective_text,
        "Amended mirrored entry"
    );

    let _ = router
        .promote(
            &ctx,
            PromoteMemoryInput {
                memory_entry_id: e1.id,
                target: PromoteTarget::Dept,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                reason: None,
            },
            None,
        )
        .await?;
    assert_eq!(
        mirror
            .get_entry(&ctx, e1.id)
            .await?
            .ok_or("missing promoted e1 on mirror")?
            .access_scope,
        AccessScope::Dept
    );

    let _ = router
        .add_edge(&ctx, r2.entry.id, e1.id, MemoryEdgeType::Supports, 0.9, None)
        .await?;
    assert_eq!(mirror.list_edges(&ctx, r2.entry.id).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn kumiho_list_entries_and_engage_expand_normalized_tag_aliases() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let mut ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(6)?);
    ctx.dept_ids = vec![dept_id];

    let concepts = InMemoryConceptStore::new();
    concepts
        .link_terms(
            org_id,
            Some(alice),
            "llmwiki",
            "세컨드 브레인",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    concepts
        .link_terms(
            org_id,
            Some(alice),
            "llmwiki",
            "LLM Wiki",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;

    let kumiho = KumihoMemoryBackend::with_config(
        None,
        None,
        true,
        None,
        Arc::new(DeterministicTextEmbedder::default()),
        Arc::new(DefaultMemoryGuard::default()),
        concepts,
    );

    let entry = kumiho
        .upsert_entry(
            &ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: Some("Wiki Notes".into()),
                text: "Personal knowledge base notes".into(),
                tags: vec!["llmwiki".into()],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
        )
        .await?;

    let tf_hash = TagFilter {
        tags_any: vec!["#세컨드브레인".into()],
        tags_all: vec![],
        tags_exclude: vec![],
        expand_tag_aliases: true,
    };
    let listed = kumiho.list_entries(&ctx, Some(&tf_hash), 20).await?;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, entry.id);

    let engaged = kumiho
        .engage(
            &ctx,
            EngageRequest {
                query: "knowledge".into(),
                scope: None,
                top_k: 5,
                tag_filter: Some(tf_hash),
            },
        )
        .await?;
    assert_eq!(engaged.memories.len(), 1);
    assert_eq!(engaged.memories[0].entry.id, entry.id);

    let tf_prefix = TagFilter {
        tags_any: vec![],
        tags_all: vec!["tag:LLM Wiki".into()],
        tags_exclude: vec![],
        expand_tag_aliases: true,
    };
    assert_eq!(kumiho.list_entries(&ctx, Some(&tf_prefix), 20).await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn memory_router_supersede_propagates_id_and_mirror_errors() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let mut ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(7)?);
    ctx.dept_ids = vec![dept_id];

    let primary_inner = Arc::new(KumihoMemoryBackend::new_offline());
    let mirror = Arc::new(KumihoMemoryBackend::new_offline());
    let primary_wrapped: Arc<dyn MemoryBackend> = Arc::new(MirrorWrapperBackend {
        kind: MemoryBackendKind::OpenragPg,
        inner: primary_inner.clone(),
    });

    let router = MemoryRouter::from_backends(
        &MemoryConfig {
            backend: MemoryBackendKind::OpenragPg,
            mirror_to: Some(MemoryBackendKind::Kumiho),
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        },
        Some(primary_wrapped),
        Some(mirror.clone()),
    )?;

    let e1 = router
        .upsert_entry(
            &ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "v1".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
            None,
        )
        .await?;

    let _sup_patch = router
        .apply_patch(
            &ctx,
            ApplyPatchInput {
                memory_entry_id: e1.id,
                patch_op: PatchOp::Supersede,
                new_text: Some("v2".into()),
                reason: Some("updated".into()),
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await?;

    let primary_active = primary_inner.list_entries(&ctx, None, 10).await?;
    let mirror_active = mirror.list_entries(&ctx, None, 10).await?;
    assert_eq!(primary_active.len(), 1);
    assert_eq!(mirror_active.len(), 1);
    assert_eq!(primary_active[0].id, mirror_active[0].id);

    let new_id = primary_active[0].id;
    let _ = router
        .apply_patch(
            &ctx,
            ApplyPatchInput {
                memory_entry_id: new_id,
                patch_op: PatchOp::Amend,
                new_text: Some("v2-amended".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await?;
    assert_eq!(
        mirror
            .get_entry(&ctx, new_id)
            .await?
            .ok_or("missing amended entry on mirror")?
            .effective_text,
        "v2-amended"
    );

    let promoted = router
        .promote(
            &ctx,
            PromoteMemoryInput {
                memory_entry_id: new_id,
                target: PromoteTarget::Dept,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                reason: None,
            },
            None,
        )
        .await?;
    assert_eq!(promoted.entry.access_scope, AccessScope::Dept);
    assert_eq!(
        mirror
            .get_entry(&ctx, new_id)
            .await?
            .ok_or("missing promoted entry on mirror")?
            .access_scope,
        AccessScope::Dept
    );

    // Mirror error propagation: create entry only on primary, then apply_patch via router -> propagates mirror NotFound
    let orphan_primary = primary_inner
        .upsert_entry(
            &ctx,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(3)?,
                org_slug: "acme".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "primary only".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
        )
        .await?;
    let mirror_err = router
        .apply_patch(
            &ctx,
            ApplyPatchInput {
                memory_entry_id: orphan_primary.id,
                patch_op: PatchOp::Amend,
                new_text: Some("fail on mirror".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            None,
        )
        .await;
    assert!(matches!(mirror_err, Err(OpenRagError::NotFound(_))));
    Ok(())
}

#[tokio::test]
async fn default_memory_guard_allowlist_exempts_pii_and_regex_in_mask_and_block_modes() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let concepts = InMemoryConceptStore::new();

    // Mask mode with allowlisted email & regex token
    let mask_guard = DefaultMemoryGuard::new(
        GuardConfig {
            enabled: true,
            default_action: GuardAction::Mask,
            allowlist: vec!["support@acme.com".into(), "SAFE-TOKEN-123".into()],
            rules: vec![
                GuardRuleConfig {
                    kind: GuardRuleKind::PiiCategory,
                    pattern_or_category: "email".into(),
                    action: GuardAction::Mask,
                    surfaces: vec![GuardSurface::IngestionGate],
                    replacement_text: Some("[EMAIL]".into()),
                    enabled: true,
                    priority: 1,
                },
                GuardRuleConfig {
                    kind: GuardRuleKind::Regex,
                    pattern_or_category: r"TOKEN-\d+".into(),
                    action: GuardAction::Mask,
                    surfaces: vec![GuardSurface::IngestionGate],
                    replacement_text: Some("[TOK]".into()),
                    enabled: true,
                    priority: 2,
                },
            ],
        },
        concepts.clone(),
    );

    let dec_allow = mask_guard
        .inspect(
            org_id,
            "Contact support@acme.com with SAFE-TOKEN-123",
            GuardSurface::IngestionGate,
        )
        .await?;
    assert_eq!(dec_allow.action, GuardAction::Allow);
    assert!(dec_allow.matched_rules.is_empty());
    assert_eq!(
        dec_allow.sanitized_text.as_deref(),
        Some("Contact support@acme.com with SAFE-TOKEN-123")
    );

    let dec_mixed = mask_guard
        .inspect(
            org_id,
            "Contact support@acme.com or leak@evil.com with SAFE-TOKEN-123 and TOKEN-999",
            GuardSurface::IngestionGate,
        )
        .await?;
    assert_eq!(dec_mixed.action, GuardAction::Mask);
    assert_eq!(
        dec_mixed.sanitized_text.as_deref(),
        Some("Contact support@acme.com or [EMAIL] with SAFE-TOKEN-123 and [TOK]")
    );

    // Block mode with allowlisted email & regex token
    let block_guard = DefaultMemoryGuard::new(
        GuardConfig {
            enabled: true,
            default_action: GuardAction::Block,
            allowlist: vec!["support@acme.com".into(), "SAFE-TOKEN-123".into()],
            rules: vec![
                GuardRuleConfig {
                    kind: GuardRuleKind::PiiCategory,
                    pattern_or_category: "email".into(),
                    action: GuardAction::Block,
                    surfaces: vec![GuardSurface::IngestionGate],
                    replacement_text: None,
                    enabled: true,
                    priority: 1,
                },
                GuardRuleConfig {
                    kind: GuardRuleKind::Regex,
                    pattern_or_category: r"TOKEN-\d+".into(),
                    action: GuardAction::Block,
                    surfaces: vec![GuardSurface::IngestionGate],
                    replacement_text: None,
                    enabled: true,
                    priority: 2,
                },
            ],
        },
        concepts,
    );

    let block_allow = block_guard
        .inspect(
            org_id,
            "Contact support@acme.com with SAFE-TOKEN-123",
            GuardSurface::IngestionGate,
        )
        .await?;
    assert_eq!(block_allow.action, GuardAction::Allow);

    let block_err_email = block_guard
        .inspect(
            org_id,
            "Contact leak@evil.com with SAFE-TOKEN-123",
            GuardSurface::IngestionGate,
        )
        .await;
    assert!(matches!(block_err_email, Err(OpenRagError::GuardBlocked { .. })));

    let block_err_regex = block_guard
        .inspect(
            org_id,
            "Contact support@acme.com with TOKEN-999",
            GuardSurface::IngestionGate,
        )
        .await;
    assert!(matches!(block_err_regex, Err(OpenRagError::GuardBlocked { .. })));
    Ok(())
}

#[tokio::test]
async fn kumiho_cross_org_oracle_and_add_edge_confidence_bounds() -> Result<(), Box<dyn std::error::Error>> {
    let org_a = Uuid::new_v4();
    let org_b = Uuid::new_v4();
    let dept_a = Uuid::new_v4();
    let dept_b = Uuid::new_v4();
    let user_a = Uuid::new_v4();
    let user_b = Uuid::new_v4();

    let mut ctx_a = PrincipalContext::new(org_a, user_a, AuthLevel::new(7)?);
    ctx_a.dept_ids = vec![dept_a];
    let mut ctx_b = PrincipalContext::new(org_b, user_b, AuthLevel::new(10)?);
    ctx_b.dept_ids = vec![dept_b];
    ctx_b.dept_admin_ids = vec![dept_b];

    let kumiho = KumihoMemoryBackend::new_offline();
    let ep_a = kumiho
        .record_episode(
            &ctx_a,
            RecordEpisodeInput {
                dept_id: dept_a,
                access_scope: AccessScope::Org,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(1)?,
                org_slug: "orga".into(),
                dept_path: "fin".into(),
                source_harness: SourceHarness::ChatUi,
                session_key: None,
                summary_md: "Org A episode".into(),
                raw_redacted_json: None,
            },
        )
        .await?;
    let entry_a1 = kumiho
        .upsert_entry(
            &ctx_a,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id: dept_a,
                access_scope: AccessScope::Org,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(1)?,
                org_slug: "orga".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "Org A entry 1".into(),
                tags: vec![],
                source_episode_id: Some(ep_a.id),
                source_document_id: None,
                confidence: None,
            },
        )
        .await?;
    let entry_a2 = kumiho
        .upsert_entry(
            &ctx_a,
            UpsertMemoryInput {
                bucket_id: None,
                dept_id: dept_a,
                access_scope: AccessScope::Org,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(1)?,
                org_slug: "orga".into(),
                dept_path: "fin".into(),
                memory_kind: MemoryKind::Fact,
                title: None,
                text: "Org A entry 2".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: None,
            },
        )
        .await?;

    // Cross-org apply_patch, promote, reflect(episode_id), reflect(target_memory_id) MUST return NotFound (not Forbidden)
    let err_patch = kumiho
        .apply_patch(
            &ctx_b,
            ApplyPatchInput {
                memory_entry_id: entry_a1.id,
                patch_op: PatchOp::Amend,
                new_text: Some("x".into()),
                reason: None,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
        )
        .await;
    assert!(matches!(err_patch, Err(OpenRagError::NotFound(_))));

    let err_promote = kumiho
        .promote(
            &ctx_b,
            PromoteMemoryInput {
                memory_entry_id: entry_a1.id,
                target: PromoteTarget::Wiki,
                target_bucket_id: None,
                target_group_ids: vec![],
                min_auth_level: None,
                wiki_slug: None,
                wiki_title: None,
                org_slug: "orgb".into(),
                dept_path: "fin".into(),
                reason: None,
            },
        )
        .await;
    assert!(matches!(err_promote, Err(OpenRagError::NotFound(_))));

    let err_reflect_ep = kumiho
        .reflect(
            &ctx_b,
            ReflectRequest {
                dept_id: dept_b,
                org_slug: "orgb".into(),
                dept_path: "fin".into(),
                content: "cross-org".into(),
                title: None,
                memory_kind: None,
                episode_id: Some(ep_a.id),
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(1)?,
                tags: vec![],
                edge_type: None,
                target_memory_id: None,
                extract_concepts: false,
            },
        )
        .await;
    assert!(matches!(err_reflect_ep, Err(OpenRagError::NotFound(_))));

    let err_reflect_tgt = kumiho
        .reflect(
            &ctx_b,
            ReflectRequest {
                dept_id: dept_b,
                org_slug: "orgb".into(),
                dept_path: "fin".into(),
                content: "cross-org".into(),
                title: None,
                memory_kind: None,
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(1)?,
                tags: vec![],
                edge_type: Some(MemoryEdgeType::DerivedFrom),
                target_memory_id: Some(entry_a1.id),
                extract_concepts: false,
            },
        )
        .await;
    assert!(matches!(err_reflect_tgt, Err(OpenRagError::NotFound(_))));

    // add_edge confidence bounds [0.0, 1.0]
    for bad in [1.5_f32, -0.1_f32, f32::NAN, f32::INFINITY] {
        let res = kumiho
            .add_edge(&ctx_a, entry_a1.id, entry_a2.id, MemoryEdgeType::Supports, bad)
            .await;
        assert!(matches!(res, Err(OpenRagError::Validation(_))));
    }
    assert!(kumiho
        .add_edge(&ctx_a, entry_a1.id, entry_a2.id, MemoryEdgeType::Supports, 0.0)
        .await
        .is_ok());
    assert!(kumiho
        .add_edge(&ctx_a, entry_a1.id, entry_a2.id, MemoryEdgeType::Supports, 1.0)
        .await
        .is_ok());
    Ok(())
}
