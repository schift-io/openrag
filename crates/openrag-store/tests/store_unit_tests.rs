use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openrag_core::OpenRagError;
use openrag_store::{
    build_sts_session_policy, decode_s3_metadata, encode_s3_metadata, expand_tag_filter_in_memory,
    l2_normalize, parse_inline_tags, s3_bucket_arn, s3_bucket_name, s3_tenant_hash,
    sign_aws_sigv4, tokenize_bm25, validate_vector, Bm25DocInput, Bm25Index,
    ExpandedTagFilter, HttpRequest, HttpResponse, HttpTransport, InMemoryOssVectorStore,
    InMemoryS3VectorsStore, OssBackendKind, OssVectorBackend, OssVectorPoint, OutboxWorker,
    QdrantHttpBackend, S3Filter, S3MetadataSchema, S3VectorPutItem, S3VectorsBackend, TagFilter,
    VectorTable, MAX_FILTERABLE_BYTES, MAX_TEXT_BYTES, VECTOR_DIM,
};
use serde_json::{json, Map, Value};
use uuid::Uuid;

fn unit_vec(axis: usize, value: f32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut v = vec![0.01_f32; VECTOR_DIM];
    v[axis % VECTOR_DIM] = value;
    Ok(l2_normalize(&v)?)
}

#[test]
fn vector_table_and_256d_validation() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(VectorTable::from_str("corpus.chunks")?, VectorTable::CorpusChunks);
    assert_eq!(VectorTable::from_str("corpus.chunks")?.vector_column(), Some("embedding"));
    // Page thumbnails and figures are not embedded (migration 012).
    assert_eq!(VectorTable::from_str("corpus.pages")?.vector_column(), None);
    assert_eq!(VectorTable::from_str("corpus.page_figures")?.vector_column(), None);
    assert!(matches!(
        VectorTable::CorpusPages.require_vector_column(),
        Err(OpenRagError::Validation(_))
    ));
    assert_eq!(VectorTable::from_str("wiki.pages")?.vector_column(), Some("embedding"));
    assert_eq!(VectorTable::from_str("memory.entries")?.vector_column(), Some("embedding"));

    assert!(matches!(
        VectorTable::from_str("iam.principals"),
        Err(OpenRagError::Validation(_))
    ));
    assert!(matches!(
        VectorTable::from_str("nonexistent.table"),
        Err(OpenRagError::Validation(_))
    ));

    let valid = unit_vec(0, 1.0)?;
    validate_vector(&valid)?;

    assert!(validate_vector(&[0.1_f32; 127]).is_err());
    assert!(validate_vector(&[0.1_f32; 129]).is_err());
    assert!(validate_vector(&[0.0_f32; 256]).is_err());

    let mut nan_vec = valid.clone();
    nan_vec[5] = f32::NAN;
    assert!(validate_vector(&nan_vec).is_err());

    let mut inf_vec = valid;
    inf_vec[5] = f32::INFINITY;
    assert!(validate_vector(&inf_vec).is_err());
    Ok(())
}

#[tokio::test]
async fn s3_vectors_tenant_isolation_caps_and_empty_sentinel() -> Result<(), Box<dyn std::error::Error>> {
    let org_a = Uuid::parse_str("11111111-1111-1111-1111-111111111111")?;
    let org_b = Uuid::parse_str("22222222-2222-2222-2222-222222222222")?;

    let bname_a = s3_bucket_name("sv-prod", org_a);
    let bname_b = s3_bucket_name("sv-prod", org_b);
    assert_ne!(bname_a, bname_b);
    assert!(bname_a.starts_with("sv-prod-"));
    assert_eq!(bname_a.len(), "sv-prod-".len() + 24);

    let hash_a = s3_tenant_hash(org_a);
    let policy_a = build_sts_session_policy("ap-northeast-2", "123456789012", "sv-prod", "meta", org_a);
    let policy_str = policy_a.to_string();
    assert!(policy_str.contains(&s3_bucket_arn("ap-northeast-2", "123456789012", "sv-prod", org_a)));
    assert!(policy_str.contains(&format!("t/{hash_a}/*")));

    let sig_headers = sign_aws_sigv4(
        "POST",
        "/QueryVectors",
        "",
        "s3vectors.ap-northeast-2.amazonaws.com",
        "ap-northeast-2",
        "s3vectors",
        "AKIAIOSFODNN7EXAMPLE",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        Some("session-tok"),
        "20261122T100000Z",
        b"{}",
    );
    assert!(sig_headers
        .iter()
        .any(|(k, v)| k == "Authorization" && v.contains("AWS4-HMAC-SHA256")));

    // _empty sentinel & raw [] rejection
    let store = InMemoryS3VectorsStore::default();
    let scoped_a = store.scoped(org_a).await?;
    let scoped_b = store.scoped(org_b).await?;
    scoped_a.ensure_index("corpus.chunks", VECTOR_DIM).await?;
    scoped_b.ensure_index("corpus.chunks", VECTOR_DIM).await?;

    let mut raw_empty_map = Map::new();
    raw_empty_map.insert("allowed_group_ids".into(), Value::Array(Vec::new()));
    let raw_err = scoped_a
        .put(
            "corpus.chunks",
            &[S3VectorPutItem {
                key: "k-raw".into(),
                vector: unit_vec(1, 0.9)?,
                metadata: raw_empty_map.clone(),
            }],
        )
        .await;
    assert!(matches!(raw_err, Err(OpenRagError::Validation(_))));

    let mut schema = S3MetadataSchema::default();
    schema.string_list_keys.insert("allowed_group_ids".into());
    schema.declared_filterable_keys.insert("allowed_group_ids".into());
    schema.declared_filterable_keys.insert("acl_scope_key".into());
    schema.declared_filterable_keys.insert("pad".into());

    raw_empty_map.insert("acl_scope_key".into(), json!("dept:fin"));
    let encoded = encode_s3_metadata("doc-1", 1_700_000_000, Some("hello"), &raw_empty_map, &schema)?;
    assert!(!encoded.contains_key("allowed_group_ids"));
    assert_eq!(
        encoded.get("_empty").and_then(|v| v.as_str()),
        Some("[\"allowed_group_ids\"]")
    );

    scoped_a
        .put(
            "corpus.chunks",
            &[S3VectorPutItem {
                key: "doc-1".into(),
                vector: unit_vec(1, 0.9)?,
                metadata: encoded,
            }],
        )
        .await?;

    let fetched = scoped_a
        .get("corpus.chunks", &["doc-1".into()], true)
        .await?;
    assert_eq!(fetched.len(), 1);
    let decoded = decode_s3_metadata(&fetched[0].metadata, &schema)?;
    assert_eq!(
        decoded.metadata.get("allowed_group_ids"),
        Some(&Value::Array(Vec::new()))
    );
    assert_eq!(decoded.text.as_deref(), Some("hello"));

    // Tenant B cannot query tenant A vectors
    let hits_b = scoped_b
        .query("corpus.chunks", &unit_vec(1, 0.9)?, 5, None)
        .await?;
    assert!(hits_b.is_empty());

    let hits_a = scoped_a
        .query(
            "corpus.chunks",
            &unit_vec(1, 0.9)?,
            5,
            Some(&S3Filter::Eq {
                field: "acl_scope_key".into(),
                value: json!("dept:fin"),
            }),
        )
        .await?;
    assert_eq!(hits_a.len(), 1);

    // 2 KB filterable cap boundary (<= 2048 vs 2049+)
    let mut fit_meta = Map::new();
    fit_meta.insert("pad".into(), json!("x".repeat(1950)));
    assert!(encode_s3_metadata("doc-fit", 1, None, &fit_meta, &schema).is_ok());

    let mut over_meta = Map::new();
    over_meta.insert("pad".into(), json!("x".repeat(MAX_FILTERABLE_BYTES + 1)));
    let err_over = encode_s3_metadata("doc-over", 1, None, &over_meta, &schema);
    assert!(matches!(err_over, Err(OpenRagError::Validation(msg)) if msg.contains("filterable_metadata_too_large")));

    // 64 keys vs 65 keys
    let mut keys_64 = Map::new();
    let mut open_schema = S3MetadataSchema::default();
    for i in 0..64 {
        let k = format!("k{i}");
        open_schema.declared_filterable_keys.insert(k.clone());
        keys_64.insert(k, json!(1));
    }
    assert!(encode_s3_metadata("k64", 1, None, &keys_64, &open_schema).is_ok());
    keys_64.insert("k64".into(), json!(1));
    assert!(matches!(
        encode_s3_metadata("k65", 1, None, &keys_64, &open_schema),
        Err(OpenRagError::Validation(msg)) if msg.contains("too_many_keys")
    ));

    // 30720 text bytes vs 30721
    let empty_meta = Map::new();
    assert!(encode_s3_metadata("t-ok", 1, Some(&"a".repeat(MAX_TEXT_BYTES)), &empty_meta, &open_schema).is_ok());
    assert!(matches!(
        encode_s3_metadata("t-over", 1, Some(&"a".repeat(MAX_TEXT_BYTES + 1)), &empty_meta, &open_schema),
        Err(OpenRagError::Validation(msg)) if msg.contains("text_too_large")
    ));

    Ok(())
}

#[tokio::test]
async fn oss_vector_store_org_isolation_and_tags() -> Result<(), Box<dyn std::error::Error>> {
    let store = InMemoryOssVectorStore::new(OssBackendKind::Qdrant);
    let org_a = Uuid::new_v4();
    let org_b = Uuid::new_v4();
    let p1 = Uuid::new_v4();
    let p2 = Uuid::new_v4();

    store
        .ensure_collection(org_a, VectorTable::CorpusChunks, VECTOR_DIM)
        .await?;
    store
        .upsert_points(
            org_a,
            VectorTable::CorpusChunks,
            &[
                OssVectorPoint {
                    id: p1,
                    org_id: org_a,
                    document_id: None,
                    acl_scope_key: "dept:fin".into(),
                    vector: unit_vec(2, 1.0)?,
                    tags: vec!["finance".into(), "2026".into()],
                    payload: Map::new(),
                },
                OssVectorPoint {
                    id: p2,
                    org_id: org_a,
                    document_id: None,
                    acl_scope_key: "dept:hr".into(),
                    vector: unit_vec(8, 1.0)?,
                    tags: vec!["hr".into()],
                    payload: Map::new(),
                },
            ],
        )
        .await?;

    // Cross-org upsert rejected
    let cross = store
        .upsert_points(
            org_b,
            VectorTable::CorpusChunks,
            &[OssVectorPoint {
                id: Uuid::new_v4(),
                org_id: org_a,
                document_id: None,
                acl_scope_key: "dept:fin".into(),
                vector: unit_vec(2, 1.0)?,
                tags: vec![],
                payload: Map::new(),
            }],
        )
        .await;
    assert!(matches!(cross, Err(OpenRagError::Forbidden(_))));

    let hits_b = store
        .search_points(org_b, VectorTable::CorpusChunks, &unit_vec(2, 1.0)?, 5, None, None)
        .await?;
    assert!(hits_b.is_empty());

    let tf = TagFilter {
        tags_any: vec!["finance".into()],
        tags_all: vec![],
        tags_exclude: vec!["hr".into()],
        expand_tag_aliases: false,
    };
    let hits_a = store
        .search_points(
            org_a,
            VectorTable::CorpusChunks,
            &unit_vec(2, 1.0)?,
            5,
            Some("dept:fin"),
            Some(&tf),
        )
        .await?;
    assert_eq!(hits_a.len(), 1);
    assert_eq!(hits_a[0].id, p1);
    Ok(())
}

#[test]
fn bm25_k1_12_b_0_and_korean_clause_normalization() -> Result<(), Box<dyn std::error::Error>> {
    let tokens = tokenize_bm25("제3조(출장 여비 지급 기준)");
    assert!(tokens.contains(&"제3조".to_string()));
    assert!(tokens.contains(&"3조".to_string()));
    assert!(tokens.contains(&"출장".to_string()));

    let d1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001")?;
    let d2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002")?;
    let d3 = Uuid::parse_str("00000000-0000-0000-0000-000000000003")?;
    let d_clause = Uuid::parse_str("00000000-0000-0000-0000-000000000004")?;

    let pad = "일반행정문서 ".repeat(40);
    let idx = Bm25Index::from_docs(&[
        Bm25DocInput {
            id: d1,
            document_id: None,
            text: "감사원특수활동비 보고서".into(),
            tags: vec![],
        },
        Bm25DocInput {
            id: d2,
            document_id: None,
            text: format!("감사원특수활동비 {pad} 감사원특수활동비"),
            tags: vec![],
        },
        Bm25DocInput {
            id: d3,
            document_id: None,
            text: "일반 복지 예산 안내".into(),
            tags: vec![],
        },
        Bm25DocInput {
            id: d_clause,
            document_id: None,
            text: "제3조(출장 여비 지급 기준) 공무 출장 시 일비를 지급한다.".into(),
            tags: vec![],
        },
    ]);

    // b = 0 -> length-invariant: D2 (2 occurrences + 40 padding tokens) > D1 (1 occurrence)
    let hits = idx.search(VectorTable::CorpusChunks, "감사원특수활동비", 10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, d2);
    assert_eq!(hits[1].id, d1);

    // Korean clause query
    let clause_hits = idx.search(VectorTable::CorpusChunks, "3조 출장 여비", 10);
    assert!(!clause_hits.is_empty());
    assert_eq!(clause_hits[0].id, d_clause);

    // Empty / punctuation-only queries return empty
    assert!(idx.search(VectorTable::CorpusChunks, "", 10).is_empty());
    assert!(idx
        .search(VectorTable::CorpusChunks, "!@#$%^&*()", 10)
        .is_empty());
    Ok(())
}

#[test]
fn inline_tag_parsing_and_alias_expansion() {
    let parsed = parse_inline_tags("예산 #재무 tag:2026_개정 -#폐기");
    assert_eq!(parsed.clean_query, "예산");
    assert_eq!(parsed.inline_tags, vec!["재무", "2026_개정"]);
    assert_eq!(parsed.excluded_tags, vec!["폐기"]);

    let tf = TagFilter {
        tags_any: vec!["#세컨드브레인".into()],
        tags_all: vec![],
        tags_exclude: vec![],
        expand_tag_aliases: true,
    };
    let expanded = expand_tag_filter_in_memory(&tf, |t| {
        if t.contains("세컨드") {
            vec!["llmwiki".into(), "second brain".into()]
        } else {
            vec![]
        }
    });
    assert!(expanded.matches_tags(&["llmwiki".into()]));
    assert!(!expanded.matches_tags(&["other".into()]));
}

#[test]
fn outbox_document_delta_diff() -> Result<(), Box<dyn std::error::Error>> {
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();
    let id4 = Uuid::new_v4();

    let prev = HashMap::from([
        (id1, "h1".into()),
        (id2, "h2_old".into()),
        (id3, "h3".into()),
    ]);
    let next = HashMap::from([
        (id1, "h1".into()),
        (id2, "h2_new".into()),
        (id4, "h4".into()),
    ]);
    let plan = OutboxWorker::check_document_delta(&prev, &next);
    let mut expected_upserts = vec![id2, id4];
    expected_upserts.sort();
    assert_eq!(plan.skipped_ids, vec![id1]);
    assert_eq!(plan.upsert_ids, expected_upserts);
    assert_eq!(plan.delete_ids, vec![id3]);
    Ok(())
}

#[derive(Default)]
struct CaptureTransport {
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait]
impl HttpTransport for CaptureTransport {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, OpenRagError> {
        self.requests
            .lock()
            .map_err(|_| OpenRagError::Internal("lock poisoned".into()))?
            .push(req);
        Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: br#"{"result":[]}"#.to_vec(),
        })
    }
}

#[tokio::test]
async fn qdrant_http_backend_builds_tag_filter_payload_clauses() -> Result<(), Box<dyn std::error::Error>> {
    let transport = Arc::new(CaptureTransport::default());
    let qdrant = QdrantHttpBackend::new("http://127.0.0.1:6333", Some("key".into()), transport.clone());
    let org_id = Uuid::new_v4();

    let tf = TagFilter {
        tags_any: vec!["finance".into()],
        tags_all: vec!["2026".into()],
        tags_exclude: vec!["draft".into()],
        expand_tag_aliases: false,
    };
    let _ = qdrant
        .search_points(
            org_id,
            VectorTable::CorpusChunks,
            &unit_vec(0, 1.0)?,
            5,
            Some("dept:fin"),
            Some(&tf),
        )
        .await?;

    let expanded = ExpandedTagFilter {
        tags_any: vec!["llmwiki".into(), "세컨드브레인".into()],
        tags_all_groups: vec![vec!["2026".into(), "올해".into()]],
        tags_exclude: vec!["draft".into()],
    };
    let _ = qdrant
        .search_points_expanded(
            org_id,
            VectorTable::CorpusChunks,
            &unit_vec(0, 1.0)?,
            5,
            None,
            Some(&expanded),
        )
        .await?;

    let reqs = transport
        .requests
        .lock()
        .map_err(|_| "lock poisoned")?;
    assert_eq!(reqs.len(), 2);

    let body1: Value = serde_json::from_slice(&reqs[0].body)?;
    let must1 = body1["filter"]["must"]
        .as_array()
        .ok_or("missing filter.must")?;
    assert!(must1.iter().any(|c| c["key"] == "tags" && c["match"]["any"] == json!(["finance"])));
    assert!(must1.iter().any(|c| c["key"] == "tags" && c["match"]["any"] == json!(["2026"])));
    let must_not1 = body1["filter"]["must_not"]
        .as_array()
        .ok_or("missing filter.must_not")?;
    assert!(must_not1.iter().any(|c| c["key"] == "tags" && c["match"]["any"] == json!(["draft"])));

    let body2: Value = serde_json::from_slice(&reqs[1].body)?;
    let must2 = body2["filter"]["must"]
        .as_array()
        .ok_or("missing filter.must")?;
    assert!(must2
        .iter()
        .any(|c| c["key"] == "tags" && c["match"]["any"] == json!(["llmwiki", "세컨드브레인"])));
    assert!(must2
        .iter()
        .any(|c| c["key"] == "tags" && c["match"]["any"] == json!(["2026", "올해"])));
    Ok(())
}
