use openrag_core::{
    arn_pattern_matches, AccessScope, Arn, ArnResourceType, AuthLevel, BBoxNorm1000, GuardAction,
    GuardSurface, IamAction, MemoryBackendKind, OpenRagError, PipelineConfig, PipelineProfile,
    PrincipalContext, SearchSpaceConfig,
};
use uuid::Uuid;

#[test]
fn pipeline_default_yaml_roundtrips_1_to_1() -> Result<(), Box<dyn std::error::Error>> {
    let raw = include_str!("../../../configs/pipeline.default.yaml");
    let cfg = PipelineConfig::from_yaml_str(raw)?;
    assert_eq!(cfg.version, 1);
    assert_eq!(cfg.profile, PipelineProfile::Standard);
    assert_eq!(cfg.store.dim, 256);
    assert!(cfg.pii.guard.enabled);
    assert_eq!(cfg.pii.guard.default_action, GuardAction::Mask);
    assert_eq!(cfg.pii.guard.rules.len(), 6);
    assert_eq!(cfg.memory.backend, MemoryBackendKind::OpenragPg);
    assert!(cfg.memory.offline_fallback);
    assert_eq!(cfg.llm.local_only_min_auth_level.as_u8(), 8);

    let reserialized = serde_yaml_ng::to_string(&cfg)?;
    let reparsed = PipelineConfig::from_yaml_str(&reserialized)?;
    assert_eq!(cfg, reparsed);
    Ok(())
}

#[test]
fn pipeline_airgap_yaml_roundtrips_1_to_1() -> Result<(), Box<dyn std::error::Error>> {
    let raw = include_str!("../../../configs/pipeline.airgap.yaml");
    let cfg = PipelineConfig::from_yaml_str(raw)?;
    assert_eq!(cfg.profile, PipelineProfile::Airgap);
    assert_eq!(cfg.parse.ocr_engine, "local_vllm");
    assert_eq!(cfg.llm.external_provider, "vllm_local");
    assert_eq!(cfg.llm.local_only_min_auth_level.as_u8(), 1);
    assert_eq!(cfg.pii.guard.default_action, GuardAction::Block);

    let reserialized = serde_yaml_ng::to_string(&cfg)?;
    let reparsed = PipelineConfig::from_yaml_str(&reserialized)?;
    assert_eq!(cfg, reparsed);
    Ok(())
}

#[test]
fn search_space_example_yaml_roundtrips_1_to_1() -> Result<(), Box<dyn std::error::Error>> {
    let raw = include_str!("../../../configs/search_space.example.yaml");
    let cfg = SearchSpaceConfig::from_yaml_str(raw)?;
    assert_eq!(cfg.golden_set, "gov_v1");
    assert!(cfg.security_gates.require_zero_pii_leak);
    assert!(cfg.security_gates.require_zero_arn_violation);
    assert!(cfg.nodes.contains_key("retrieve_fusion"));

    let reserialized = serde_yaml_ng::to_string(&cfg)?;
    let reparsed = SearchSpaceConfig::from_yaml_str(&reserialized)?;
    assert_eq!(cfg, reparsed);
    Ok(())
}

#[test]
fn pg_pinecone_search_defaults_and_bounds() -> Result<(), Box<dyn std::error::Error>> {
    let base = include_str!("../../../configs/pipeline.default.yaml");
    let parsed = PipelineConfig::from_yaml_str(base)?;
    assert!((parsed.store.pg_pinecone_search.scan_fraction - 0.5).abs() < 1e-6);
    assert_eq!(parsed.store.pg_pinecone_search.rescore_multiplier, 4);
    let omitted = base.replace("  pg_pinecone_search:\n    scan_fraction: 0.5\n    rescore_multiplier: 4\n", "");
    assert_ne!(omitted, base);
    let parsed = PipelineConfig::from_yaml_str(&omitted)?;
    assert_eq!(parsed.store.pg_pinecone_search, openrag_core::PgPineconeSearchConfig::default());
    for (from, to) in [
        ("scan_fraction: 0.5", "scan_fraction: 1.5"),
        ("scan_fraction: 0.5", "scan_fraction: .nan"),
        ("rescore_multiplier: 4", "rescore_multiplier: 1001"),
    ] {
        let yaml = base.replace(from, to);
        assert!(
            matches!(PipelineConfig::from_yaml_str(&yaml), Err(OpenRagError::Config(_))),
            "expected Config error for {to}"
        );
    }
    Ok(())
}

#[test]
fn embed_document_title_defaults_and_parsing() -> Result<(), Box<dyn std::error::Error>> {
    let base = include_str!("../../../configs/pipeline.default.yaml");
    let parsed = PipelineConfig::from_yaml_str(base)?;
    assert!(parsed.store.embed_document_title);

    let airgap = include_str!("../../../configs/pipeline.airgap.yaml");
    assert!(PipelineConfig::from_yaml_str(airgap)?.store.embed_document_title);

    let disabled = base.replace("embed_document_title: true", "embed_document_title: false");
    assert_ne!(disabled, base);
    assert!(!PipelineConfig::from_yaml_str(&disabled)?.store.embed_document_title);

    let omitted = base.replace("  embed_document_title: true\n", "");
    assert_ne!(omitted, base);
    assert!(PipelineConfig::from_yaml_str(&omitted)?.store.embed_document_title);
    Ok(())
}

#[test]
fn store_chunker_defaults_and_parsing() -> Result<(), Box<dyn std::error::Error>> {
    use openrag_core::ChunkerKind;

    let base = include_str!("../../../configs/pipeline.default.yaml");
    assert_eq!(PipelineConfig::from_yaml_str(base)?.store.chunker, ChunkerKind::Fixed);

    let airgap = include_str!("../../../configs/pipeline.airgap.yaml");
    assert_eq!(PipelineConfig::from_yaml_str(airgap)?.store.chunker, ChunkerKind::Fixed);

    let structure = base.replace("chunker: fixed", "chunker: structure");
    assert_ne!(structure, base);
    assert_eq!(PipelineConfig::from_yaml_str(&structure)?.store.chunker, ChunkerKind::Structure);

    let semantic = base.replace("chunker: fixed", "chunker: semantic");
    assert_ne!(semantic, base);
    assert_eq!(PipelineConfig::from_yaml_str(&semantic)?.store.chunker, ChunkerKind::Semantic);

    let omitted = base.replace("  chunker: fixed\n", "");
    assert_ne!(omitted, base);
    assert_eq!(PipelineConfig::from_yaml_str(&omitted)?.store.chunker, ChunkerKind::Fixed);
    Ok(())
}

#[test]
fn invalid_configs_are_rejected() {
    let base = include_str!("../../../configs/pipeline.default.yaml");

    // Unknown field rejected via deny_unknown_fields
    let unknown = format!("{base}\nunexpected_field: true\n");
    assert!(matches!(
        PipelineConfig::from_yaml_str(&unknown),
        Err(OpenRagError::Config(_))
    ));

    // Non-256 dim rejected
    let bad_dim = base.replacen("dim: 256", "dim: 128", 1);
    assert!(matches!(
        PipelineConfig::from_yaml_str(&bad_dim),
        Err(OpenRagError::Config(_))
    ));

    // Invalid regex in guard rules rejected
    assert!(base.contains(SK_KEY_PATTERN), "default pipeline must carry the sk- key rule");
    let bad_regex = base.replace(SK_KEY_PATTERN, "[unclosed(");
    assert!(matches!(
        PipelineConfig::from_yaml_str(&bad_regex),
        Err(OpenRagError::Config(_))
    ));

    // Invalid alpha rejected
    let bad_alpha = base.replace("alpha: 0.0", "alpha: 1.5");
    assert!(matches!(
        PipelineConfig::from_yaml_str(&bad_alpha),
        Err(OpenRagError::Config(_))
    ));

    // NaN / Inf / negative latest_revision_boost rejected
    for bad_boost in [".nan", ".inf", "-.inf", "-5.0"] {
        let yaml = base.replace(
            "latest_revision_boost: 0.15",
            &format!("latest_revision_boost: {bad_boost}"),
        );
        assert!(
            matches!(PipelineConfig::from_yaml_str(&yaml), Err(OpenRagError::Config(_))),
            "expected Config error for latest_revision_boost={bad_boost}"
        );
    }

    // NaN / Inf alpha & admission_threshold rejected
    for bad_val in [".nan", ".inf"] {
        let yaml_a = base.replace("alpha: 0.0", &format!("alpha: {bad_val}"));
        assert_ne!(yaml_a, base);
        assert!(matches!(
            PipelineConfig::from_yaml_str(&yaml_a),
            Err(OpenRagError::Config(_))
        ));
        let yaml_t = base.replace(
            "admission_threshold: 0.35",
            &format!("admission_threshold: {bad_val}"),
        );
        assert_ne!(yaml_t, base);
        assert!(matches!(
            PipelineConfig::from_yaml_str(&yaml_t),
            Err(OpenRagError::Config(_))
        ));
    }

    // Zero thumbnail / outbox bounds rejected
    for (needle, repl) in [
        ("max_edge_px: 1600", "max_edge_px: 0"),
        ("figure_max_edge_px: 1280", "figure_max_edge_px: 0"),
        ("figure_max_bytes: 307200", "figure_max_bytes: 0"),
        ("outbox_batch_size: 64", "outbox_batch_size: 0"),
        ("outbox_max_attempts: 3", "outbox_max_attempts: 0"),
    ] {
        let yaml = base.replace(needle, repl);
        assert_ne!(yaml, base, "expected needle '{needle}' in default yaml");
        assert!(
            matches!(PipelineConfig::from_yaml_str(&yaml), Err(OpenRagError::Config(_))),
            "expected Config error for {repl}"
        );
    }
}

#[test]
fn arn_parsing_canonical_and_glob_matching() -> Result<(), Box<dyn std::error::Error>> {
    let raw = "arn:openrag:acme:finance/payroll:private:document:L8/doc-42";
    let arn = Arn::parse(raw)?;
    assert_eq!(arn.org, "acme");
    assert_eq!(arn.dept_path, "finance/payroll");
    assert_eq!(arn.scope, AccessScope::Private);
    assert_eq!(arn.resource_type, ArnResourceType::Document);
    assert_eq!(arn.auth_level.as_u8(), 8);
    assert_eq!(arn.resource_id, "doc-42");
    assert_eq!(arn.to_canonical(), raw);

    assert!(arn.matches_pattern("*"));
    assert!(arn.matches_pattern("arn:openrag:acme:finance/*:document:L*/doc-*"));
    assert!(arn.matches_pattern("arn:openrag:acme:*:*:document:*"));
    assert!(!arn.matches_pattern("arn:openrag:acme:hr/*"));
    assert!(!arn_pattern_matches("", raw));
    assert!(!arn_pattern_matches("", ""));
    assert!(!arn_pattern_matches("*", ""));

    assert!(Arn::parse("arn:other:acme:finance:private:document:L3/1").is_err());
    assert!(Arn::parse("arn:openrag:acme:finance:private:document:L11/1").is_err());
    assert!(Arn::parse("arn:openrag:acme:finance:private:document:L0/1").is_err());
    Ok(())
}

#[test]
fn auth_level_and_bbox_validation() -> Result<(), Box<dyn std::error::Error>> {
    assert!(AuthLevel::new(0).is_err());
    assert!(AuthLevel::new(11).is_err());
    assert_eq!(AuthLevel::new(1)?.as_u8(), 1);
    assert_eq!(AuthLevel::new(10)?.as_i16(), 10);

    let bbox = BBoxNorm1000::from_array([100, 200, 500, 800])?;
    assert_eq!(bbox.to_array(), [100, 200, 500, 800]);
    let (x, y, w, h) = bbox.to_pixel_rect(1000, 2000);
    assert_eq!((x, y, w, h), (100, 400, 401, 1202));

    assert!(BBoxNorm1000::new(-1, 0, 10, 10).is_err());
    assert!(BBoxNorm1000::new(0, 0, 1000, 10).is_err());
    assert!(BBoxNorm1000::new(500, 10, 100, 20).is_err());
    Ok(())
}

#[test]
fn guard_and_action_enums() {
    assert!(GuardAction::Block > GuardAction::Mask);
    assert!(GuardAction::Mask > GuardAction::Warn);
    assert!(GuardAction::Warn > GuardAction::Allow);
    assert_eq!(GuardSurface::LlmEgress.as_str(), "llm_egress");
    assert_eq!(IamAction::PiiUnmask.as_str(), "pii:unmask");
    assert!(IamAction::Wildcard.matches(IamAction::Write));

    let ctx = PrincipalContext::new(Uuid::nil(), Uuid::nil(), AuthLevel::MIN);
    assert!(!ctx.bypass_rls);
    assert!(ctx.scoped_arns.is_empty());
    assert!(ctx.scoped_group_ids.is_empty());
}

#[test]
fn memory_config_variants_and_validation() -> Result<(), Box<dyn std::error::Error>> {
    let base = include_str!("../../../configs/pipeline.default.yaml");

    // Kumiho backend + mirror_to kumiho accepted
    let kumiho_yaml = base.replace(
        "memory:\n  backend: openrag_pg\n  offline_fallback: true",
        "memory:\n  backend: kumiho\n  mirror_to: kumiho\n  kumiho_url: \"http://127.0.0.1:8088\"\n  kumiho_api_key_env: KUMIHO_API_KEY\n  offline_fallback: true",
    );
    let cfg = PipelineConfig::from_yaml_str(&kumiho_yaml)?;
    assert_eq!(cfg.memory.backend, MemoryBackendKind::Kumiho);
    assert_eq!(cfg.memory.mirror_to, Some(MemoryBackendKind::Kumiho));
    assert_eq!(
        cfg.memory.kumiho_url.as_deref(),
        Some("http://127.0.0.1:8088")
    );

    // Unknown memory backend rejected
    let unknown_backend = base.replace("backend: openrag_pg", "backend: unknown_mem");
    assert!(matches!(
        PipelineConfig::from_yaml_str(&unknown_backend),
        Err(OpenRagError::Config(_))
    ));

    // Unknown field inside memory rejected via deny_unknown_fields
    let unknown_field = base.replace(
        "offline_fallback: true\n\nstore:",
        "offline_fallback: true\n  bogus_flag: 1\n\nstore:",
    );
    assert!(matches!(
        PipelineConfig::from_yaml_str(&unknown_field),
        Err(OpenRagError::Config(_))
    ));

    // Empty kumiho_url rejected
    let empty_url = base.replace(
        "offline_fallback: true\n\nstore:",
        "offline_fallback: true\n  kumiho_url: \"   \"\n\nstore:",
    );
    assert!(matches!(
        PipelineConfig::from_yaml_str(&empty_url),
        Err(OpenRagError::Config(_))
    ));

    // Self-mirroring openrag_pg -> openrag_pg rejected
    let self_mirror = base.replace(
        "backend: openrag_pg",
        "backend: openrag_pg\n  mirror_to: openrag_pg",
    );
    assert!(matches!(
        PipelineConfig::from_yaml_str(&self_mirror),
        Err(OpenRagError::Config(_))
    ));
    Ok(())
}

const SK_KEY_PATTERN: &str = r"(?i)\bsk-[a-z0-9][a-z0-9_-]{15,}";

#[test]
fn sk_key_guard_pattern_catches_hyphenated_keys_without_word_false_positives() {
    let re = regex::Regex::new(SK_KEY_PATTERN).ok();
    assert!(re.is_some(), "pattern must compile");
    let Some(re) = re else { return };
    for hit in [
        "sk-test-openrag-1234567890abcdef secret credential",
        "key=sk-proj-AbCdEf0123456789_ghIJ",
        "sk-ant-api03-abcdefghijklmnop",
        "SK-0123456789abcdefghij",
        "(sk-abcdefghijklmnopqrstuvwxyz0123)",
    ] {
        assert!(re.is_match(hit), "should match: {hit}");
    }
    for miss in [
        "risk-assessment-framework-2026-q1",
        "task-runner-configuration-guide",
        "desk-booking-system-overview",
        "sk-short",
        "sk--------------------",
    ] {
        assert!(!re.is_match(miss), "should not match: {miss}");
    }
}
