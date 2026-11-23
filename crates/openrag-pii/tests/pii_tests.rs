use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, BBoxNorm1000, GuardAction, GuardRuleKind,
    GuardSurface, IamAction, OpenRagError, PrincipalContext,
};
use openrag_iam::{IamPolicy, PolicyEffect, PolicyStatement, ResourceDescriptor};
use openrag_pii::{
    luhn_valid, mask_image_bboxes, rrn_mod11_checksum_valid, Capsule, CapsuleIamCtx,
    DeterministicSemanticEmbedder128, GuardEngine, GuardPolicyRuntime, GuardRepo,
    GuardRuleRuntime, InMemoryConceptAliasLookup, InMemoryVaultStore, MasterKek, PiiCategory,
    PiiEngine, VaultCrypto,
};
use uuid::Uuid;

#[test]
fn f14_rrn_calendar_gender_and_mod11_trace() -> Result<(), Box<dyn std::error::Error>> {
    let engine = PiiEngine {
        use_marker_ko_pii: true,
        ..Default::default()
    };

    // 900101-1234567 has valid date+gender (mod-11 check digit is 8, not 7) -> detected with checksum_valid=false
    // 900101-1234568 has valid date+gender AND valid mod-11 -> checksum_valid=true
    // 991332-9999999 has invalid month 13, day 32, gender 9 -> rejected
    let text = "A: 900101-1234567, B: 900101-1234568, C: 991332-9999999";
    let spans = engine.detect(text);
    let rrn_spans: Vec<_> = spans
        .iter()
        .filter(|s| s.category == PiiCategory::RrnKr)
        .collect();
    assert_eq!(rrn_spans.len(), 2);
    assert_eq!(rrn_spans[0].matched_text, "900101-1234567");
    assert_eq!(rrn_spans[0].detector_trace["checksum_valid"], false);
    assert_eq!(rrn_spans[1].matched_text, "900101-1234568");
    assert_eq!(rrn_spans[1].detector_trace["checksum_valid"], true);

    assert!(rrn_mod11_checksum_valid(&[9, 0, 0, 1, 0, 1, 1, 2, 3, 4, 5, 6, 8]));
    assert!(!rrn_mod11_checksum_valid(&[9, 0, 0, 1, 0, 1, 1, 2, 3, 4, 5, 6, 7]));
    Ok(())
}

#[test]
fn f14_multibyte_boundaries_overlap_allowlist_and_all_off() -> Result<(), Box<dyn std::error::Error>> {
    // Multibyte Korean without spaces around PII must preserve "성명:" and trailing "끝"
    let engine = PiiEngine::default();
    let sample = "성명:홍길동(900101-1234567)연락처:010-1234-5678이메일:hong.gildong@example.or.kr끝";
    let spans = engine.detect(sample);
    let kek = MasterKek::from_bytes([7u8; 32]);
    let store = InMemoryVaultStore::new();
    let sealed = Capsule::mask_and_seal_in_memory(sample, &spans, Uuid::new_v4(), &kek, &store)?;
    assert!(sealed.masked_text.starts_with("성명:"));
    assert!(sealed.masked_text.ends_with("끝"));
    assert!(!sealed.masked_text.contains("홍길동"));
    assert!(!sealed.masked_text.contains("900101-1234567"));
    assert!(!sealed.masked_text.contains("010-1234-5678"));
    assert!(!sealed.masked_text.contains("hong.gildong@example.or.kr"));

    // Overlapping custom dict longer span wins over inner regex phone span
    let custom_engine = PiiEngine {
        custom_dict_terms: vec!["010-1234-5678-내선99".into()],
        allowlist_terms: vec!["02-120".into(), "홍길동".into()],
        ..Default::default()
    };
    let overlap_text = "내선: 010-1234-5678-내선99, 민원실: 02-120, 담당자:홍길동";
    let o_spans = custom_engine.detect(overlap_text);
    assert_eq!(o_spans.len(), 1);
    assert_eq!(o_spans[0].category, PiiCategory::CustomDict);
    assert_eq!(o_spans[0].matched_text, "010-1234-5678-내선99");

    // All detectors disabled -> zero spans
    let off_engine = PiiEngine {
        use_schift_v7: false,
        use_marker_ko_pii: false,
        use_schift_v6_org: false,
        custom_dict_terms: Vec::new(),
        allowlist_terms: Vec::new(),
    };
    assert!(off_engine.detect(sample).is_empty());

    // Luhn credit card check
    assert!(luhn_valid(&[4, 5, 3, 2, 0, 1, 5, 1, 1, 2, 8, 3, 0, 3, 6, 6]));
    assert!(!luhn_valid(&[4, 5, 3, 2, 0, 1, 5, 1, 1, 2, 8, 3, 0, 3, 6, 7]));
    Ok(())
}

#[test]
fn f15_aes_gcm_capsule_tamper_purge_and_iam_unmask() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let other_org = Uuid::new_v4();
    let kek = MasterKek::from_bytes([0x42u8; 32]);
    let store = InMemoryVaultStore::new();
    let engine = PiiEngine::default();

    let raw = "주민번호 900101-1234567, 연락처 010-9876-5432";
    let spans = engine.detect(raw);
    let sealed = Capsule::mask_and_seal_in_memory(raw, &spans, org_id, &kek, &store)?;
    assert!(!sealed.masked_text.contains("900101-1234567"));
    assert!(!sealed.masked_text.contains("010-9876-5432"));
    assert_eq!(sealed.tokens.len(), 2);

    // Wrong org cannot unmask
    let wrong_org_user = PrincipalContext {
        can_unmask_pii: true,
        ..PrincipalContext::new(other_org, Uuid::new_v4(), AuthLevel::new(9)?)
    };
    let res_other = Capsule::unmask_in_memory(&sealed.masked_text, &wrong_org_user, None, &kek, &store)?;
    assert!(!res_other.allowed);
    assert_eq!(res_other.unmasked_count, 0);
    assert_eq!(res_other.text, sealed.masked_text);

    // Principal without can_unmask_pii denied
    let no_cap_user = PrincipalContext::new(org_id, Uuid::new_v4(), AuthLevel::new(9)?);
    let res_nocap = Capsule::unmask_in_memory(&sealed.masked_text, &no_cap_user, None, &kek, &store)?;
    assert!(!res_nocap.allowed);
    assert_eq!(res_nocap.reason, "principal_cannot_unmask_pii");
    assert_eq!(res_nocap.text, sealed.masked_text);

    // Explicit IAM Deny on PiiUnmask overrides can_unmask_pii=true
    let owner_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let doc_id = Uuid::new_v4();
    let mut auth_user = PrincipalContext::new(org_id, owner_id, AuthLevel::new(8)?);
    auth_user.can_unmask_pii = true;
    auth_user.dept_ids = vec![dept_id];

    let resource = ResourceDescriptor {
        org_id,
        dept_id,
        owner_principal_id: owner_id,
        access_scope: AccessScope::Dept,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        min_auth_level: AuthLevel::new(3)?,
        arn: Arn::new("acme", "fin", AccessScope::Dept, ArnResourceType::Document, AuthLevel::new(3)?, doc_id.to_string())?,
        resource_type: ArnResourceType::Document,
        resource_id: doc_id,
    };
    let deny_policy = IamPolicy {
        id: Uuid::new_v4(),
        name: "deny-unmask".into(),
        min_auth_level_override: None,
        statements: vec![PolicyStatement {
            sid: Some("deny-pii".into()),
            effect: PolicyEffect::Deny,
            actions: vec![IamAction::PiiUnmask],
            resources: vec!["*".into()],
            condition: None,
        }],
    };
    let iam_ctx = CapsuleIamCtx {
        resource: &resource,
        groups: &[],
        memberships: &[],
        bindings: &[],
        policies: std::slice::from_ref(&deny_policy),
    };
    let res_iam_deny =
        Capsule::unmask_in_memory(&sealed.masked_text, &auth_user, Some(&iam_ctx), &kek, &store)?;
    assert!(!res_iam_deny.allowed);
    assert_eq!(res_iam_deny.reason, "explicit_deny");

    // Authorized unmask without deny succeeds
    let iam_allow_ctx = CapsuleIamCtx {
        resource: &resource,
        groups: &[],
        memberships: &[],
        bindings: &[],
        policies: &[],
    };
    let res_ok = Capsule::unmask_in_memory(
        &sealed.masked_text,
        &auth_user,
        Some(&iam_allow_ctx),
        &kek,
        &store,
    )?;
    assert!(res_ok.allowed);
    assert_eq!(res_ok.text, raw);

    // 1-bit ciphertext tamper fails closed
    store.tamper_ciphertext_for_test(org_id, &sealed.tokens[0].token)?;
    let res_tamper =
        Capsule::unmask_in_memory(&sealed.masked_text, &auth_user, None, &kek, &store)?;
    assert!(!res_tamper.allowed);
    assert_eq!(res_tamper.denied_count, 1);
    assert!(res_tamper.text.contains(&sealed.tokens[0].token));

    // Direct VaultCrypto roundtrip & AAD mismatch check
    let dek = VaultCrypto::generate_dek()?;
    let (ct, nonce) = VaultCrypto::encrypt_value(&dek, org_id, "[PII:RRN:12345678]", "900101-1234567")?;
    assert_eq!(
        VaultCrypto::decrypt_value(&dek, org_id, "[PII:RRN:12345678]", &ct, &nonce)?,
        "900101-1234567"
    );
    assert!(VaultCrypto::decrypt_value(&dek, other_org, "[PII:RRN:12345678]", &ct, &nonce).is_err());
    Ok(())
}

#[test]
fn f16_f36_guard_precedence_surfaces_aliases_nfkc_obfuscation_and_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let pii_engine = PiiEngine::default();
    let mut aliases = InMemoryConceptAliasLookup::new();
    aliases.link_terms(&["사과", "apple"]);
    aliases.link_terms(&["llmwiki", "LLM Wiki", "세컨드 브레인", "Second Brain"]);

    let policy = GuardPolicyRuntime {
        id: Some(Uuid::new_v4()),
        org_id,
        master_enabled: true,
        default_action: GuardAction::Mask,
        allowlist_terms: vec!["PROJECT-ORION-DEMO".into()],
        semantic_threshold: None,
        rules: vec![
            GuardRuleRuntime {
                id: Some(Uuid::new_v4()),
                kind: GuardRuleKind::PiiCategory,
                pattern_or_category: "RRN".into(), // uppercase alias for rrn_kr
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::LlmEgress, GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 10,
            },
            GuardRuleRuntime {
                id: Some(Uuid::new_v4()),
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "PROJECT-ORION".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery, GuardSurface::McpEgress],
                replacement_text: None,
                enabled: true,
                priority: 20,
            },
            GuardRuleRuntime {
                id: Some(Uuid::new_v4()),
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "사과".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 30,
            },
            GuardRuleRuntime {
                id: Some(Uuid::new_v4()),
                kind: GuardRuleKind::Keyword,
                pattern_or_category: "llmwiki".into(),
                action: GuardAction::Warn,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 40,
            },
        ],
    };

    // 1. Block > Mask precedence & metadata never leaks raw secret
    let block_and_mask = GuardEngine::evaluate(
        "900101-1234567 PROJECT-ORION",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(block_and_mask.action, GuardAction::Block);
    assert!(block_and_mask.sanitized_text.is_none());
    let meta = GuardRepo::build_safe_event_metadata(
        GuardSurface::InboundQuery,
        "900101-1234567 PROJECT-ORION",
        &block_and_mask,
    );
    let meta_str = meta.to_string();
    assert!(!meta_str.contains("PROJECT-ORION"));
    assert!(!meta_str.contains("900101-1234567"));

    // 2. Surface isolation: mcp_egress allows "사과" (which is only inbound_query), while blocking PROJECT-ORION
    let mcp_apple = GuardEngine::evaluate(
        "사과와 apple 이야기",
        GuardSurface::McpEgress,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(mcp_apple.action, GuardAction::Allow);

    // 3. Cross-lingual alias blocking: "apple" blocked because "사과" is blocked
    let inbound_apple = GuardEngine::enforce(
        "Tell me about apple project",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    );
    assert!(matches!(inbound_apple, Err(OpenRagError::GuardBlocked { .. })));

    // 4. NFKC fullwidth, zero-width, and punctuation/whitespace obfuscation resistance
    for obfuscated in [
        "ＰＲＯＪＥＣＴ－ＯＲＩＯＮ",
        "PRO\u{200B}JECT-ORION",
        "P.R.O.J.E.C.T - O R I O N",
        "ａｐｐｌｅ",
    ] {
        let dec = GuardEngine::evaluate(
            obfuscated,
            GuardSurface::InboundQuery,
            &policy,
            &pii_engine,
            Some(&aliases),
            None,
        )?;
        assert_eq!(dec.action, GuardAction::Block, "failed to catch obfuscated '{obfuscated}'");
    }

    // 5. Allowlist exemption: "PROJECT-ORION-DEMO 공개 발표" allowed
    let allow_demo = GuardEngine::evaluate(
        "PROJECT-ORION-DEMO 공개 발표",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(allow_demo.action, GuardAction::Allow);

    // 6. Keyword concept alias Warn: "세컨드 브레인" triggers Warn for "llmwiki"
    let warn_res = GuardEngine::evaluate(
        "세컨드 브레인 정리",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(warn_res.action, GuardAction::Warn);
    assert_eq!(warn_res.warnings.len(), 1);

    // 7. RRN category alias "RRN" masks on LlmEgress
    let masked = GuardEngine::evaluate(
        "주민번호 900101-1234567 입니다",
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(masked.action, GuardAction::Mask);
    assert_eq!(
        masked.sanitized_text.as_deref(),
        Some("주민번호 [REDACTED:RRN] 입니다")
    );

    // 8. Master toggle OFF returns Allow
    let mut off_policy = policy.clone();
    off_policy.master_enabled = false;
    let off_dec = GuardEngine::evaluate(
        "PROJECT-ORION 900101-1234567",
        GuardSurface::InboundQuery,
        &off_policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(off_dec.action, GuardAction::Allow);

    // 9. Optional 128d semantic similarity match
    let sem_policy = GuardPolicyRuntime {
        id: None,
        org_id,
        master_enabled: true,
        default_action: GuardAction::Allow,
        allowlist_terms: Vec::new(),
        semantic_threshold: Some(0.80),
        rules: vec![GuardRuleRuntime {
            id: None,
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "Project Phoenix Secret".into(),
            action: GuardAction::Block,
            surfaces: vec![GuardSurface::InboundQuery],
            replacement_text: None,
            enabled: true,
            priority: 1,
        }],
    };
    let sem_emb = DeterministicSemanticEmbedder128::new();
    let sem_dec = GuardEngine::evaluate(
        "Project Phoenix Secrets",
        GuardSurface::InboundQuery,
        &sem_policy,
        &pii_engine,
        None,
        Some(&sem_emb),
    )?;
    assert_eq!(sem_dec.action, GuardAction::Block);
    Ok(())
}

#[test]
fn f17_thumbnail_black_box_pixel_masking() -> Result<(), Box<dyn std::error::Error>> {
    let img = RgbaImage::from_pixel(400, 400, Rgba([255, 255, 255, 255]));
    let mut png_cur = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img.clone()).write_to(&mut png_cur, ImageFormat::Png)?;
    let png_bytes = png_cur.into_inner();

    let mut webp_cur = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img).write_to(&mut webp_cur, ImageFormat::WebP)?;
    let webp_bytes = webp_cur.into_inner();

    let bbox = BBoxNorm1000::new(100, 100, 700, 400)?;

    for (label, orig_bytes) in [("png", &png_bytes), ("webp", &webp_bytes)] {
        let masked_bytes = mask_image_bboxes(orig_bytes, &[bbox])?;
        assert_ne!(orig_bytes.as_slice(), masked_bytes.as_slice(), "{label} bytes must change");

        let decoded = image::load_from_memory(&masked_bytes)?.to_rgba8();
        let (rx, ry, rw, rh) = bbox.to_pixel_rect(400, 400);
        let mut inside_total = 0u64;
        let mut inside_black = 0u64;
        for py in ry..ry + rh {
            for px in rx..rx + rw {
                inside_total += 1;
                let p = decoded.get_pixel(px, py);
                if p[0] <= 16 && p[1] <= 16 && p[2] <= 16 {
                    inside_black += 1;
                }
            }
        }
        assert!(inside_total > 0);
        assert!(
            (inside_black as f64) / (inside_total as f64) >= 0.99,
            "{label} inside box must be >=99% black"
        );

        let outside = decoded.get_pixel(10, 10);
        assert!(outside[0] >= 240 && outside[1] >= 240 && outside[2] >= 240);
    }
    Ok(())
}

#[test]
fn f16_f36_nfd_hangul_jamo_and_combining_marks_cannot_bypass_guard() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let pii_engine = PiiEngine::default();
    let mut aliases = InMemoryConceptAliasLookup::new();
    aliases.link_terms(&["사과", "apple"]);

    let policy = GuardPolicyRuntime {
        id: None,
        org_id,
        master_enabled: true,
        default_action: GuardAction::Allow,
        allowlist_terms: Vec::new(),
        semantic_threshold: None,
        rules: vec![
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "사과".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 1,
            },
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "사과".into(),
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::LlmEgress],
                replacement_text: Some("[REDACTED]".into()),
                enabled: true,
                priority: 1,
            },
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "café".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 2,
            },
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "café".into(),
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::LlmEgress],
                replacement_text: Some("[CAFE]".into()),
                enabled: true,
                priority: 2,
            },
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "각".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 3,
            },
        ],
    };

    // NFD Hangul Jamo ("사과")
    let nfd_apple = "\u{1109}\u{1161}\u{1100}\u{116A}";
    let dec_nfd_apple = GuardEngine::evaluate(
        nfd_apple,
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(dec_nfd_apple.action, GuardAction::Block);

    // NFD Hangul masking inside larger NFD word ("맛있는 사과나무" -> masks only "사과")
    let nfd_sentence = "맛있는 \u{1109}\u{1161}\u{1100}\u{116A}\u{1102}\u{1161}\u{1106}\u{116E}";
    let dec_nfd_mask = GuardEngine::evaluate(
        nfd_sentence,
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(dec_nfd_mask.action, GuardAction::Mask);
    assert_eq!(
        dec_nfd_mask.sanitized_text.as_deref(),
        Some("맛있는 [REDACTED]\u{1102}\u{1161}\u{1106}\u{116E}")
    );

    // Precomposed LV ("가") + trailing Jamo ("\u{11A8}") = "각"
    let lv_t = "\u{AC00}\u{11A8}";
    let dec_lvt = GuardEngine::evaluate(
        lv_t,
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(dec_lvt.action, GuardAction::Block);

    // NFD Latin combining mark ("cafe\u{0301}" vs "café")
    let dec_cafe_block = GuardEngine::evaluate(
        "cafe\u{0301}",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(dec_cafe_block.action, GuardAction::Block);

    let dec_cafe_mask = GuardEngine::evaluate(
        "Visit cafe\u{0301} today",
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(dec_cafe_mask.action, GuardAction::Mask);
    assert_eq!(
        dec_cafe_mask.sanitized_text.as_deref(),
        Some("Visit [CAFE] today")
    );

    // Mixed NFD + whitespace/punctuation/zero-width obfuscation inside syllable & combining marks
    let mixed_nfd = "\u{1109} \u{1161}.\u{1100}\u{116A}";
    let dec_mixed_block = GuardEngine::evaluate(
        mixed_nfd,
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(dec_mixed_block.action, GuardAction::Block);

    let dec_mixed_mask = GuardEngine::evaluate(
        mixed_nfd,
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(dec_mixed_mask.action, GuardAction::Mask);
    assert_eq!(dec_mixed_mask.sanitized_text.as_deref(), Some("[REDACTED]"));

    let dec_mixed_wrapped = GuardEngine::evaluate(
        "앞\u{1109} \u{1161}.\u{1100}\u{116A}뒤",
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(dec_mixed_wrapped.action, GuardAction::Mask);
    assert_eq!(
        dec_mixed_wrapped.sanitized_text.as_deref(),
        Some("앞[REDACTED]뒤")
    );

    let dec_zw_jamo = GuardEngine::evaluate(
        "\u{1109}\u{200B}\u{1161}\u{1100}\u{116A}",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&aliases),
        None,
    )?;
    assert_eq!(dec_zw_jamo.action, GuardAction::Block);

    let dec_cafe_space = GuardEngine::evaluate(
        "cafe \u{0301}",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(dec_cafe_space.action, GuardAction::Block);

    let dec_cafe_zw = GuardEngine::evaluate(
        "cafe\u{200B}\u{0301}",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(dec_cafe_zw.action, GuardAction::Block);

    let dec_cafe_zw_mask = GuardEngine::evaluate(
        "Visit cafe\u{200B}\u{0301} today",
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(dec_cafe_zw_mask.action, GuardAction::Mask);
    assert_eq!(
        dec_cafe_zw_mask.sanitized_text.as_deref(),
        Some("Visit [CAFE] today")
    );
    Ok(())
}

#[test]
fn f16_f36_semantic_threshold_respects_allowlist_with_whitespace_and_surrounding_words() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let pii_engine = PiiEngine::default();
    let sem_emb = DeterministicSemanticEmbedder128::new();

    let policy = GuardPolicyRuntime {
        id: None,
        org_id,
        master_enabled: true,
        default_action: GuardAction::Allow,
        allowlist_terms: vec!["PROJECT-ORION-DEMO".into()],
        semantic_threshold: Some(0.60),
        rules: vec![
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "PROJECT-ORION".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 1,
            },
            GuardRuleRuntime {
                id: None,
                kind: GuardRuleKind::SafeWord,
                pattern_or_category: "Project Phoenix Secret".into(),
                action: GuardAction::Block,
                surfaces: vec![GuardSurface::InboundQuery],
                replacement_text: None,
                enabled: true,
                priority: 2,
            },
        ],
    };

    for allowed_input in [
        "  PROJECT-ORION-DEMO",
        "  PROJECT-ORION-DEMO  ",
        "Tell me about PROJECT-ORION-DEMO",
    ] {
        let dec = GuardEngine::evaluate(
            allowed_input,
            GuardSurface::InboundQuery,
            &policy,
            &pii_engine,
            None,
            Some(&sem_emb),
        )?;
        assert_eq!(
            dec.action,
            GuardAction::Allow,
            "expected Allow for '{allowed_input}'"
        );
    }

    // Non-allowlisted secret in a sentence containing an allowlisted span still blocks
    let blocked = GuardEngine::evaluate(
        "Tell me about PROJECT-ORION-DEMO and Project Phoenix Secrets",
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        None,
        Some(&sem_emb),
    )?;
    assert_eq!(blocked.action, GuardAction::Block);
    Ok(())
}

#[test]
fn korean_bank_accounts_in_each_banks_shape() {
    let engine = PiiEngine::default();
    let accounts = |text: &str| -> Vec<String> {
        engine
            .detect(text)
            .into_iter()
            .filter(|s| s.category == PiiCategory::BankAccount)
            .map(|s| s.matched_text)
            .collect()
    };
    for (text, want) in [
        ("결제 계좌 국민은행 계좌번호 123456-01-234567", "123456-01-234567"),
        ("결제 계좌: 신한은행 110-234-567890", "110-234-567890"),
        ("농협 계좌 301-0123-4567-11로 보내 주세요", "301-0123-4567-11"),
        ("기업은행 계좌 123-456789-01-012", "123-456789-01-012"),
        ("입금 계좌 1002345678901", "1002345678901"),
    ] {
        assert_eq!(accounts(text), [want], "{text}");
    }
    for text in [
        "입금 문의 010-1234-5678",
        "은행 방문일 2026-11-23",
        "법인카드 계좌 연결 4111-1111-1111-1111",
        "계좌 없음",
    ] {
        assert!(accounts(text).is_empty(), "{text}: {:?}", accounts(text));
    }
}

#[test]
fn korean_names_followed_by_a_spaced_job_title() {
    let engine = PiiEngine::default();
    let names = |text: &str| -> Vec<String> {
        engine
            .detect(text)
            .into_iter()
            .filter(|s| s.category == PiiCategory::PersonName)
            .map(|s| s.matched_text)
            .collect()
    };
    for (text, want) in [
        ("김민준 과장은 마케팅팀 소속이다.", "김민준"),
        ("담당: 이서연 대리", "이서연"),
        ("박지우 사원에게 전달", "박지우"),
        ("(정하준 부장)", "정하준"),
    ] {
        assert_eq!(names(text), vec![want.to_string()], "{text}");
    }
    for text in [
        "신입 사원 교육 일정",
        "전체 사원 공지",
        "전산실 과장 회의",
        "구매팀 차장 보고",
    ] {
        assert!(names(text).is_empty(), "{text}: {:?}", names(text));
    }
    // 님 after the title still ends it.
    assert_eq!(names("김민준 과장님께서"), vec!["김민준".to_string()]);
    // A title that only starts a longer word is not a title.
    assert!(names("홍길동 대리점 위치").is_empty());
}
