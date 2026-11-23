use std::collections::BTreeMap;

use chrono::{Duration, Utc};
use openrag_core::{AuthLevel, FusionMode, PgPineconeEngine, PrincipalContext};
use openrag_db::DbPool;
use openrag_memory::{ConceptCategory, ConceptGraph, ConceptSource, DeterministicTextEmbedder};
use openrag_retrieve::{
    EvidenceAdmission, PgHybridRetriever, PgRetrieveBackend, RetrieveSpace, SearchRequest,
};
use openrag_store::{
    l2_normalize, PgPineconeManager, StoreMode, TagFilter, VectorTable, VECTOR_DIM,
};
use pgvector::Vector;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

fn norm_vec(axis: usize, val: f32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut raw = vec![0.02_f32; VECTOR_DIM];
    raw[axis % VECTOR_DIM] = val;
    Ok(l2_normalize(&raw)?)
}

struct SeededOrg {
    org_id: Uuid,
    dept_id: Uuid,
    slug: String,
    alice: Uuid,
    bob: Uuid,
    admin: Uuid,
    tf_group: Uuid,
    denied_group: Uuid,
    bucket_a: Uuid,
    bucket_b: Uuid,
    alice_ctx: PrincipalContext,
    bob_ctx: PrincipalContext,
    admin_ctx: PrincipalContext,
}

async fn seed_org(admin_pool: &DbPool) -> Result<SeededOrg, Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let admin = Uuid::new_v4();
    let tf_group = Uuid::new_v4();
    let denied_group = Uuid::new_v4();
    let bucket_a = Uuid::new_v4();
    let bucket_b = Uuid::new_v4();
    let slug = format!("ret-{}", &org_id.simple().to_string()[..8]);

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Retrieve Test Org')")
        .bind(org_id)
        .bind(&slug)
        .execute(admin_pool.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Policy', 'policy')",
    )
    .bind(dept_id)
    .bind(org_id)
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $4, $5, 'Alice', 3, false, false),
         ($2, $4, $6, 'Bob', 3, false, false),
         ($3, $4, $7, 'Admin', 10, true, true)",
    )
    .bind(alice)
    .bind(bob)
    .bind(admin)
    .bind(org_id)
    .bind(format!("alice-{}@ret.test", org_id.simple()))
    .bind(format!("bob-{}@ret.test", org_id.simple()))
    .bind(format!("admin-{}@ret.test", org_id.simple()))
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'member'),
         ($1, $2, $4, 'member'),
         ($1, $2, $5, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(alice)
    .bind(bob)
    .bind(admin)
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, group_type) VALUES
         ($1, $3, 'TF Alpha', 'project_tf'),
         ($2, $3, 'Denied Group', 'permission_group')",
    )
    .bind(tf_group)
    .bind(denied_group)
    .bind(org_id)
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
         ($1, $2, $4),
         ($1, $3, $5)",
    )
    .bind(org_id)
    .bind(tf_group)
    .bind(denied_group)
    .bind(alice)
    .bind(bob)
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $3, $4, $5, 'bucket-a', 'Bucket A', 'dept', 1, $6),
         ($2, $3, $4, $5, 'bucket-b', 'Bucket B', 'dept', 1, $7)",
    )
    .bind(bucket_a)
    .bind(bucket_b)
    .bind(org_id)
    .bind(dept_id)
    .bind(admin)
    .bind(format!("arn:openrag:{slug}:policy:dept:bucket:L1/{bucket_a}"))
    .bind(format!("arn:openrag:{slug}:policy:dept:bucket:L1/{bucket_b}"))
    .execute(admin_pool.inner())
    .await?;

    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(3)?);
    alice_ctx.dept_ids = vec![dept_id];
    alice_ctx.group_ids = vec![tf_group];

    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(3)?);
    bob_ctx.dept_ids = vec![dept_id];
    bob_ctx.group_ids = vec![denied_group];

    let mut admin_ctx = PrincipalContext::new(org_id, admin, AuthLevel::new(10)?);
    admin_ctx.dept_ids = vec![dept_id];
    admin_ctx.dept_admin_ids = vec![dept_id];
    admin_ctx.is_org_admin = true;
    admin_ctx.can_unmask_pii = true;

    Ok(SeededOrg {
        org_id,
        dept_id,
        slug,
        alice,
        bob,
        admin,
        tf_group,
        denied_group,
        bucket_a,
        bucket_b,
        alice_ctx,
        bob_ctx,
        admin_ctx,
    })
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_multispace_retrieval_rls_and_bucket_scoping() -> Result<(), Box<dyn std::error::Error>>
{
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let concept = "예산집행투명성지침";
    let emb = Vector::from(DeterministicTextEmbedder::embed_sync(concept)?);

    let doc_dept = Uuid::new_v4();
    let doc_priv = Uuid::new_v4();
    let doc_b = Uuid::new_v4();
    let doc_group = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
         VALUES
         ($1, $5, $6, $8, $9, 'dept', ARRAY[]::uuid[], ARRAY[]::uuid[], 3, $11, 'dept:3', 'Dept Budget', 'sha1', ARRAY['budget','public']::text[]),
         ($2, $5, $6, $8, $9, 'private', ARRAY[]::uuid[], ARRAY[]::uuid[], 9, $12, 'priv:9', 'Secret L9', 'sha2', ARRAY['secret-tag']::text[]),
         ($3, $5, $7, $8, $9, 'dept', ARRAY[]::uuid[], ARRAY[]::uuid[], 3, $13, 'dept:b', 'Bucket B Doc', 'sha3', ARRAY['bucket-b']::text[]),
         ($4, $5, $6, $8, $10, 'group', ARRAY[$14]::uuid[], ARRAY[$15]::uuid[], 3, $16, 'grp:tf', 'TF Group Doc', 'sha4', ARRAY['tf-only']::text[])",
    )
    .bind(doc_dept)
    .bind(doc_priv)
    .bind(doc_b)
    .bind(doc_group)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.bucket_b)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_dept}", s.slug))
    .bind(format!("arn:openrag:{}:policy:private:doc:L9/{doc_priv}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_b}", s.slug))
    .bind(s.tf_group)
    .bind(s.denied_group)
    .bind(format!("arn:openrag:{}:policy:group:doc:L3/{doc_group}", s.slug))
    .execute(admin.inner())
    .await?;

    let c_dept = Uuid::new_v4();
    let c_priv = Uuid::new_v4();
    let c_b = Uuid::new_v4();
    let c_group = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
         VALUES
         ($1, $5, $9, $13, $14, 'dept', ARRAY[]::uuid[], ARRAY[]::uuid[], 3, $17, 'dept:3', 0, $21, $21, 'h1', $25, ARRAY['budget','public']::text[]),
         ($2, $5, $10, $13, $14, 'private', ARRAY[]::uuid[], ARRAY[]::uuid[], 9, $18, 'priv:9', 0, $22, $22, 'h2', $25, ARRAY['secret-tag']::text[]),
         ($3, $5, $11, $13, $14, 'dept', ARRAY[]::uuid[], ARRAY[]::uuid[], 3, $19, 'dept:b', 0, $23, $23, 'h3', $25, ARRAY['bucket-b']::text[]),
         ($4, $5, $12, $13, $15, 'group', ARRAY[$6]::uuid[], ARRAY[$7]::uuid[], 3, $20, 'grp:tf', 0, $24, $24, 'h4', $25, ARRAY['tf-only']::text[])",
    )
    .bind(c_dept)
    .bind(c_priv)
    .bind(c_b)
    .bind(c_group)
    .bind(s.org_id)
    .bind(s.tf_group)
    .bind(s.denied_group)
    .bind(s.bucket_a) // $8 unused
    .bind(doc_dept)
    .bind(doc_priv)
    .bind(doc_b)
    .bind(doc_group)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(s.admin)
    .bind(s.bob) // $16 unused
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_dept}", s.slug))
    .bind(format!("arn:openrag:{}:policy:private:chunk:L9/{c_priv}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_b}", s.slug))
    .bind(format!("arn:openrag:{}:policy:group:chunk:L3/{c_group}", s.slug))
    .bind(format!("{concept} 부서 공개 본문"))
    .bind(format!("{concept} SECRET_PRIVATE_TOKEN"))
    .bind(format!("{concept} 다른 버킷 본문"))
    .bind(format!("{concept} TF 그룹 전용"))
    .bind(emb.clone())
    .execute(admin.inner())
    .await?;

    // Thumb (corpus.pages)
    let page_dept = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.pages
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, redacted_markdown, thumb_uri)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'dept:3', 1, $7, 's3://t/1.webp')",
    )
    .bind(page_dept)
    .bind(s.org_id)
    .bind(doc_dept)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:{}:policy:dept:page:L3/{page_dept}", s.slug))
    .bind(format!("{concept} 썸네일 OCR"))
    .execute(admin.inner())
    .await?;

    // Wiki page with embedding IS NULL (must still be retrieved via BM25!)
    let wiki_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'dept:3', 'wiki-b', '예산 위키', $7, $7, 'wsha', NULL, ARRAY['budget']::text[])",
    )
    .bind(wiki_id)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:{}:policy:dept:wiki:L3/{wiki_id}", s.slug))
    .bind(format!("{concept} 위키 요약 본문"))
    .execute(admin.inner())
    .await?;

    // Memory entry
    let mem_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.entries
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, title, effective_text, bm25_text, content_sha256, embedding, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'dept:3', 'fact', '예산 메모리', $7, $7, 'msha', $8, ARRAY['budget']::text[])",
    )
    .bind(mem_id)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:{}:policy:dept:mem:L3/{mem_id}", s.slug))
    .bind(format!("{concept} 기억 메모"))
    .bind(emb)
    .execute(admin.inner())
    .await?;

    let retriever = PgHybridRetriever::new_pg(app);

    // 1. Bob (L3, in denied_group, NOT owner of private L9): sees Chunks, Thumb, Wiki (NULL vec!), Memory in bucket_a,
    //    and ZERO private L9 / TF-group / bucket_b hits or tags!
    let mut req_bob = SearchRequest::new(concept);
    req_bob.bucket_id = Some(s.bucket_a);
    req_bob.top_k = Some(20);
    // The defaults are vector-only over text spaces; this case covers every space, including
    // thumbnails and the NULL-embedding wiki row, which only BM25 reaches.
    req_bob.fusion = Some(FusionMode::Rrf);
    req_bob.spaces = Some(vec![
        RetrieveSpace::Chunks,
        RetrieveSpace::Thumb,
        RetrieveSpace::Wiki,
        RetrieveSpace::Memory,
    ]);
    let res_bob = retriever.search(&s.bob_ctx, &req_bob).await?;

    let spaces_found: std::collections::BTreeSet<RetrieveSpace> =
        res_bob.hits.iter().map(|h| h.space).collect();
    assert!(spaces_found.contains(&RetrieveSpace::Chunks));
    assert!(spaces_found.contains(&RetrieveSpace::Thumb));
    assert!(
        spaces_found.contains(&RetrieveSpace::Wiki),
        "wiki with NULL embedding must be retrieved via BM25"
    );
    assert!(spaces_found.contains(&RetrieveSpace::Memory));

    let ids_bob: Vec<Uuid> = res_bob.hits.iter().map(|h| h.id).collect();
    assert!(
        !ids_bob.contains(&c_priv),
        "Bob must not see private L9 chunk"
    );
    assert!(
        !ids_bob.contains(&c_group),
        "Bob in denied_group must not see group chunk"
    );
    assert!(
        !ids_bob.contains(&c_b),
        "bucket_a scope must exclude bucket_b"
    );
    assert!(
        !res_bob.tag_facets.contains_key("secret-tag"),
        "secret-tag leaked in facets"
    );
    assert!(
        !res_bob.tag_facets.contains_key("tf-only"),
        "tf-only leaked in facets"
    );
    assert_eq!(res_bob.top_auth_level, Some(3));
    assert_eq!(res_bob.admission, EvidenceAdmission::Admit);

    // 2. Alice (L3, in tf_group): sees c_group too, still NOT c_priv (min_auth_level=9 > Alice L3)!
    let res_alice = retriever.search(&s.alice_ctx, &req_bob).await?;
    let ids_alice: Vec<Uuid> = res_alice.hits.iter().map(|h| h.id).collect();
    assert!(ids_alice.contains(&c_group));
    assert!(!ids_alice.contains(&c_priv));

    // 3. Admin (L10, dept_admin): sees c_priv too, top_auth_level = 9, secret-tag in facets
    let res_admin = retriever.search(&s.admin_ctx, &req_bob).await?;
    let ids_admin: Vec<Uuid> = res_admin.hits.iter().map(|h| h.id).collect();
    assert!(ids_admin.contains(&c_priv));
    assert_eq!(res_admin.top_auth_level, Some(9));
    assert_eq!(res_admin.tag_facets.get("secret-tag"), Some(&1));

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_concept_expansion_corpus_pages_tag_filter_and_facets(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    // Seed concepts via RlsTransaction as admin
    {
        let mut tx = app.begin_rls_tx(&s.admin_ctx).await?;
        ConceptGraph::link_terms_rls(
            &mut tx,
            s.org_id,
            Some(s.admin),
            "apple",
            "사과",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
        ConceptGraph::link_terms_rls(
            &mut tx,
            s.org_id,
            Some(s.admin),
            "llmwiki",
            "세컨드 브레인",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
        ConceptGraph::link_terms_rls(
            &mut tx,
            s.org_id,
            Some(s.admin),
            "llmwiki",
            "세컨드브레인",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
        tx.commit().await?;
    }

    let doc_wiki = Uuid::new_v4();
    let doc_gen = Uuid::new_v4();
    let doc_apple = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
         VALUES
         ($1, $4, $5, $6, $7, 'dept', 3, $8, 'd:3', 'Wiki Guide', 's1', ARRAY['llmwiki']::text[]),
         ($2, $4, $5, $6, $7, 'dept', 3, $9, 'd:3', 'General Guide', 's2', ARRAY['general']::text[]),
         ($3, $4, $5, $6, $7, 'dept', 3, $10, 'd:3', 'Apple Doc', 's3', ARRAY['fruit']::text[])",
    )
    .bind(doc_wiki)
    .bind(doc_gen)
    .bind(doc_apple)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_wiki}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_gen}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_apple}", s.slug))
    .execute(admin.inner())
    .await?;

    let p_wiki = Uuid::new_v4();
    let p_gen = Uuid::new_v4();
    let c_wiki = Uuid::new_v4();
    let c_apple = Uuid::new_v4();
    let emb_wiki = Vector::from(DeterministicTextEmbedder::embed_sync("지식관리 llmwiki")?);
    let emb_apple = Vector::from(DeterministicTextEmbedder::embed_sync("apple project")?);

    sqlx::query(
        "INSERT INTO corpus.pages
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, redacted_markdown)
         VALUES
         ($1, $3, $5, $7, $8, 'dept', 3, $9, 'd:3', 1, '지식관리 llmwiki 다이어그램'),
         ($2, $3, $6, $7, $8, 'dept', 3, $10, 'd:3', 1, '지식관리 일반 다이어그램')",
    )
    .bind(p_wiki)
    .bind(p_gen)
    .bind(s.org_id)
    .bind(s.bucket_a) // $4 unused
    .bind(doc_wiki)
    .bind(doc_gen)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:page:L3/{p_wiki}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:page:L3/{p_gen}", s.slug))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
         VALUES
         ($1, $3, $5, $7, $8, 'dept', 3, $9, 'd:3', 0, '지식관리 llmwiki 청크', '지식관리 llmwiki 청크', 'cw', $11, ARRAY['llmwiki']::text[]),
         ($2, $3, $6, $7, $8, 'dept', 3, $10, 'd:3', 0, 'apple project specification', 'apple project specification', 'ca', $12, ARRAY['fruit']::text[])",
    )
    .bind(c_wiki)
    .bind(c_apple)
    .bind(s.org_id)
    .bind(s.bucket_a) // $4 unused
    .bind(doc_wiki)
    .bind(doc_apple)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_wiki}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_apple}", s.slug))
    .bind(emb_wiki)
    .bind(emb_apple)
    .execute(admin.inner())
    .await?;

    let retriever = PgHybridRetriever::new_pg(app);

    // 1. Multi-word concept expansion: "세컨드 브레인" -> finds "llmwiki"
    let mut r_multi = SearchRequest::new("세컨드 브레인");
    r_multi.bucket_id = Some(s.bucket_a);
    let resp_multi = retriever.search(&s.bob_ctx, &r_multi).await?;
    assert!(resp_multi.expanded_terms.iter().any(|t| t == "llmwiki"));
    assert!(resp_multi
        .hits
        .iter()
        .any(|h| h.document_id == Some(doc_wiki)));

    // 2. Single-word cross-lingual concept expansion: "사과" -> finds "apple"
    let mut r_single = SearchRequest::new("사과 프로젝트");
    r_single.bucket_id = Some(s.bucket_a);
    let resp_single = retriever.search(&s.bob_ctx, &r_single).await?;
    assert!(resp_single.expanded_terms.iter().any(|t| t == "apple"));
    assert!(resp_single
        .hits
        .iter()
        .any(|h| h.document_id == Some(doc_apple)));

    // 3. CorpusPages (thumb) tag filtering + expand_tag_aliases=false vs true + deduplicated facets
    let mut r_no_exp = SearchRequest::new("지식관리");
    r_no_exp.bucket_id = Some(s.bucket_a);
    r_no_exp.tag_filter = Some(TagFilter {
        tags_any: vec!["세컨드브레인".into()],
        expand_tag_aliases: false,
        ..TagFilter::default()
    });
    let resp_no_exp = retriever.search(&s.bob_ctx, &r_no_exp).await?;
    assert!(resp_no_exp.hits.is_empty());

    let mut r_tag = SearchRequest::new("지식관리 #세컨드브레인,");
    r_tag.bucket_id = Some(s.bucket_a);
    r_tag.spaces = Some(vec![RetrieveSpace::Chunks, RetrieveSpace::Thumb]);
    r_tag.tag_filter = Some(TagFilter {
        expand_tag_aliases: true,
        ..TagFilter::default()
    });
    let resp_tag = retriever.search(&s.bob_ctx, &r_tag).await?;
    assert!(resp_tag
        .hits
        .iter()
        .any(|h| h.space == RetrieveSpace::Thumb && h.id == p_wiki));
    assert!(resp_tag.hits.iter().all(|h| h.id != p_gen));
    assert_eq!(resp_tag.tag_facets.get("llmwiki"), Some(&1));

    // 4. Tag-only query ("#llmwiki") retrieves both Chunks and Thumb without literal query text
    let mut r_tag_only = SearchRequest::new("#llmwiki");
    r_tag_only.bucket_id = Some(s.bucket_a);
    r_tag_only.spaces = Some(vec![RetrieveSpace::Chunks, RetrieveSpace::Thumb]);
    let resp_tag_only = retriever.search(&s.bob_ctx, &r_tag_only).await?;
    assert!(resp_tag_only
        .hits
        .iter()
        .any(|h| h.space == RetrieveSpace::Chunks && h.id == c_wiki));
    assert!(resp_tag_only
        .hits
        .iter()
        .any(|h| h.space == RetrieveSpace::Thumb && h.id == p_wiki));

    // 5. Nonexistent tag returns empty hits and {} facets
    let mut r_none = SearchRequest::new("지식관리 #no_such_tag_999");
    r_none.bucket_id = Some(s.bucket_a);
    let resp_none = retriever.search(&s.bob_ctx, &r_none).await?;
    assert!(resp_none.hits.is_empty());
    assert_eq!(resp_none.tag_facets, BTreeMap::new());

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_latest_revision_valid_to_and_engine_switch_invariance(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let doc_v1 = Uuid::new_v4();
    let doc_v2 = Uuid::new_v4();
    let past = Utc::now() - Duration::days(30);
    let recent = Utc::now() - Duration::days(1);

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, revision, valid_from, valid_to)
         VALUES
         ($1, $3, $4, $5, $6, 'dept', 3, $7, 'd:3', '2025년 출장 여비 규정', 'sv1', 1, $9, $9),
         ($2, $3, $4, $5, $6, 'dept', 3, $8, 'd:3', '2026년 출장 여비 개정 규정', 'sv2', 2, $10, NULL)",
    )
    .bind(doc_v1)
    .bind(doc_v2)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_v1}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_v2}", s.slug))
    .bind(past)
    .bind(recent)
    .execute(admin.inner())
    .await?;

    let c1 = Uuid::new_v4();
    let c2 = Uuid::new_v4();
    let vec_same = Vector::from(norm_vec(1, 0.99)?);
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
         VALUES
         ($1, $3, $5, $7, $8, 'dept', 3, $9, 'd:3', 0, '출장 여비 5만원 기준', '출장 여비 5만원 기준', 'ch1', $11),
         ($2, $3, $6, $7, $8, 'dept', 3, $10, 'd:3', 0, '출장 여비 7만원 기준', '출장 여비 7만원 기준', 'ch2', $11)",
    )
    .bind(c1)
    .bind(c2)
    .bind(s.org_id)
    .bind(s.bucket_a) // $4 unused
    .bind(doc_v1)
    .bind(doc_v2)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c1}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c2}", s.slug))
    .bind(vec_same)
    .execute(admin.inner())
    .await?;

    // Also test wiki.edges supersedes
    let w1 = Uuid::new_v4();
    let w2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, version)
         VALUES
         ($1, $3, $4, $5, $6, 'dept', 3, $7, 'd:3', 'w-old', '출장 여비 위키 구버전', '출장 여비 구버전', '출장 여비 구버전', 'ws1', 1),
         ($2, $3, $4, $5, $6, 'dept', 3, $8, 'd:3', 'w-new', '출장 여비 위키 신버전', '출장 여비 신버전', '출장 여비 신버전', 'ws2', 2)",
    )
    .bind(w1)
    .bind(w2)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:wiki:L3/{w1}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:wiki:L3/{w2}", s.slug))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type) VALUES ($1, $2, $3, 'supersedes')",
    )
    .bind(s.org_id)
    .bind(w2)
    .bind(w1)
    .execute(admin.inner())
    .await?;

    let backend = PgRetrieveBackend::new(app.clone());
    let retriever = PgHybridRetriever::new(backend);

    // 1. Default latest_revision_boost ranks v2 > v1 and w2 > w1
    let mut req = SearchRequest::new("출장 여비");
    req.bucket_id = Some(s.bucket_a);
    // The wiki pages here have no embedding, so they are reachable through BM25 (rrf) only.
    req.fusion = Some(FusionMode::Rrf);
    let resp = retriever.search(&s.bob_ctx, &req).await?;
    let ids: Vec<Uuid> = resp.hits.iter().map(|h| h.id).collect();
    let p_c2 = ids.iter().position(|x| *x == c2).ok_or("missing c2")?;
    let p_c1 = ids.iter().position(|x| *x == c1).ok_or("missing c1")?;
    assert!(p_c2 < p_c1);
    let p_w2 = ids.iter().position(|x| *x == w2).ok_or("missing w2")?;
    let p_w1 = ids.iter().position(|x| *x == w1).ok_or("missing w1")?;
    assert!(p_w2 < p_w1);

    // 2. latest_revision_only excludes expired v1 & superseded w1
    let mut req_only = req.clone();
    req_only.latest_revision_only = true;
    let resp_only = retriever.search(&s.bob_ctx, &req_only).await?;
    let ids_only: Vec<Uuid> = resp_only.hits.iter().map(|h| h.id).collect();
    assert!(ids_only.contains(&c2));
    assert!(ids_only.contains(&w2));
    assert!(!ids_only.contains(&c1));
    assert!(!ids_only.contains(&w1));

    // 3. Switch pg_pinecone engine between Native and Pgvector and verify identical search works
    PgPineconeManager::set_engine(
        &worker,
        s.org_id,
        VectorTable::CorpusChunks,
        PgPineconeEngine::Pgvector,
        StoreMode::Offline,
        None,
    )
    .await?;
    let backend_loaded = PgRetrieveBackend::new(app)
        .load_bindings_from_worker(&worker, s.org_id)
        .await?;
    let retriever_loaded = PgHybridRetriever::new(backend_loaded);
    let mut convex_req = req.clone();
    convex_req.fusion = Some(FusionMode::Convex);
    convex_req.alpha = Some(0.3);
    let resp_pgvec = retriever_loaded.search(&s.bob_ctx, &convex_req).await?;
    assert!(!resp_pgvec.hits.is_empty());

    // 4. EvidenceAdmission Abstain at high threshold 0.95 for off-topic query
    let mut off_req = SearchRequest::new("화성 탐사선 로켓 연료 배합 비율은?");
    off_req.bucket_id = Some(s.bucket_a);
    off_req.admission_threshold = Some(0.95);
    let resp_off = retriever_loaded.search(&s.bob_ctx, &off_req).await?;
    assert_eq!(resp_off.admission, EvidenceAdmission::Abstain);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_bug1_bug4_archived_starvation_and_bucket_tag_facets(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let topic = "아카이브고갈테스트규정";
    let emb = Vector::from(DeterministicTextEmbedder::embed_sync(topic)?);

    // Seed 45 archived documents + chunks (exceeds candidate_limit_per_space = 40 for top_k=5)
    for i in 0..45 {
        let doc_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO corpus.documents
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, status, tags)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', $7, $8, 'archived', ARRAY['archived-only-tag']::text[])",
        )
        .bind(doc_id)
        .bind(s.org_id)
        .bind(s.bucket_a)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_id}", s.slug))
        .bind(format!("Archived Doc {i}"))
        .bind(format!("sha-arch-{i}"))
        .execute(admin.inner())
        .await?;

        sqlx::query(
            "INSERT INTO corpus.chunks
             (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 0, $7, $7, $8, $9, ARRAY['archived-only-tag']::text[])",
        )
        .bind(chunk_id)
        .bind(s.org_id)
        .bind(doc_id)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{chunk_id}", s.slug))
        .bind(format!("{topic} 아카이브 문서 {topic} {topic} {i}"))
        .bind(format!("h-arch-{i}"))
        .bind(emb.clone())
        .execute(admin.inner())
        .await?;
    }

    // Seed 1 active document + chunk
    let active_doc = Uuid::new_v4();
    let active_chunk = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, status, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 'Active Spec', 'sha-act', 'indexed', ARRAY['active-only-tag']::text[])",
    )
    .bind(active_doc)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{active_doc}", s.slug))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 0, $7, $7, 'h-act', $8, ARRAY['active-only-tag']::text[])",
    )
    .bind(active_chunk)
    .bind(s.org_id)
    .bind(active_doc)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{active_chunk}", s.slug))
    .bind(format!("{topic} 현행 활성 문서"))
    .bind(emb)
    .execute(admin.inner())
    .await?;

    let retriever = PgHybridRetriever::new_pg(app);

    let mut req = SearchRequest::new(topic);
    req.bucket_id = Some(s.bucket_a);
    req.spaces = Some(vec![RetrieveSpace::Chunks]);
    req.top_k = Some(5);
    req.fusion = Some(FusionMode::Convex);
    req.alpha = Some(1.0);
    req.rerank = Some(false);
    let resp = retriever.search(&s.bob_ctx, &req).await?;
    assert_eq!(resp.hits.len(), 1);
    assert_eq!(resp.hits[0].id, active_chunk);
    // BUG-1: archived hits must not pollute max_bm25 or RRF rank-1
    assert!((resp.hits[0].score_breakdown.bm25_score - 1.0).abs() < 1e-5);
    assert!((resp.hits[0].score_breakdown.rrf_score - 1.0).abs() < 1e-5);
    assert_eq!(resp.admission, EvidenceAdmission::Admit);

    // BUG-4: bucket_tag_facets must exclude archived documents
    let facets = retriever
        .bucket_tag_facets(&s.bob_ctx, None, Some(s.bucket_a), 20)
        .await?;
    assert!(
        !facets.contains_key("archived-only-tag"),
        "archived-only-tag leaked into bucket_tag_facets: {facets:?}"
    );
    assert_eq!(facets.get("active-only-tag"), Some(&1));

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_bug2_bug3_bug5_offtopic_abstain_future_valid_from_and_tag_only_convex(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let now = Utc::now();
    let doc_curr = Uuid::new_v4();
    let doc_future = Uuid::new_v4();
    let doc_payroll = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, revision, valid_from, tags)
         VALUES
         ($1, $4, $5, $6, $7, 'dept', 3, $8, 'd:3', '연차 휴가 현행 v1', 'sc1', 1, $11, ARRAY['hr']::text[]),
         ($2, $4, $5, $6, $7, 'dept', 3, $9, 'd:3', '연차 휴가 예정 v2', 'sc2', 2, $12, ARRAY['hr']::text[]),
         ($3, $4, $5, $6, $7, 'dept', 3, $10, 'd:3', '급여 이체 지침', 'sc3', 1, $11, ARRAY['payroll']::text[])",
    )
    .bind(doc_curr)
    .bind(doc_future)
    .bind(doc_payroll)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_curr}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_future}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_payroll}", s.slug))
    .bind(now - Duration::days(5))
    .bind(now + Duration::days(30))
    .execute(admin.inner())
    .await?;

    let c_curr = Uuid::new_v4();
    let c_future = Uuid::new_v4();
    let c_payroll = Uuid::new_v4();
    let emb_hr = Vector::from(DeterministicTextEmbedder::embed_sync(
        "연차 휴가 규정 안내",
    )?);
    let emb_pay = Vector::from(DeterministicTextEmbedder::embed_sync(
        "매월 25일 임직원 급여 계좌 이체 기준",
    )?);

    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
         VALUES
         ($1, $4, $7, $10, $11, 'dept', 3, $12, 'd:3', 0, '신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.', '신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.', 'hc1', $15, ARRAY['hr']::text[]),
         ($2, $4, $8, $10, $11, 'dept', 3, $13, 'd:3', 0, '신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.', '신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.', 'hc2', $15, ARRAY['hr']::text[]),
         ($3, $4, $9, $10, $11, 'dept', 3, $14, 'd:3', 0, '매월 25일 임직원 급여 계좌 이체 기준', '매월 25일 임직원 급여 계좌 이체 기준', 'hc3', $16, ARRAY['payroll']::text[])",
    )
    .bind(c_curr)
    .bind(c_future)
    .bind(c_payroll)
    .bind(s.org_id)
    .bind(s.bucket_a) // $5 unused
    .bind(s.dept_id)  // $6 unused
    .bind(doc_curr)
    .bind(doc_future)
    .bind(doc_payroll)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_curr}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_future}", s.slug))
    .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{c_payroll}", s.slug))
    .bind(emb_hr)
    .bind(emb_pay)
    .execute(admin.inner())
    .await?;

    let retriever = PgHybridRetriever::new_pg(app);

    // BUG-2: Off-topic query MUST Abstain at default 0.35 (both rerank=true and rerank=false)
    for (fusion, rerank) in [
        (FusionMode::Rrf, true),
        (FusionMode::Rrf, false),
        (FusionMode::Convex, true),
    ] {
        let mut off_req = SearchRequest::new("화성 탐사선 로켓 연료 배합 비율은?");
        off_req.bucket_id = Some(s.bucket_a);
        off_req.fusion = Some(fusion);
        off_req.rerank = Some(rerank);
        let resp = retriever.search(&s.bob_ctx, &off_req).await?;
        assert_eq!(
            resp.admission,
            EvidenceAdmission::Abstain,
            "off-topic query admitted on PG17 at default 0.35 (fusion={fusion:?}, rerank={rerank}, top_score={})",
            resp.top_score
        );
    }

    // Grounded query at default 0.35 MUST Admit
    let mut grounded = SearchRequest::new("신입 직원 연차 휴가는 며칠인가요?");
    grounded.bucket_id = Some(s.bucket_a);
    let resp_g = retriever.search(&s.bob_ctx, &grounded).await?;
    assert_eq!(resp_g.admission, EvidenceAdmission::Admit);

    // BUG-3: Future valid_from > now must rank below currently active and be excluded by latest_revision_only
    let ids_g: Vec<Uuid> = resp_g.hits.iter().map(|h| h.id).collect();
    let pos_curr = ids_g
        .iter()
        .position(|x| *x == c_curr)
        .ok_or("missing curr")?;
    let pos_future = ids_g
        .iter()
        .position(|x| *x == c_future)
        .ok_or("missing future")?;
    assert!(pos_curr < pos_future);
    assert_eq!(resp_g.hits[pos_future].score_breakdown.revision_boost, 0.0);

    let mut only_req = grounded;
    only_req.latest_revision_only = true;
    let resp_only = retriever.search(&s.bob_ctx, &only_req).await?;
    let ids_only: Vec<Uuid> = resp_only.hits.iter().map(|h| h.id).collect();
    assert!(ids_only.contains(&c_curr));
    assert!(!ids_only.contains(&c_future));

    // BUG-5: Tag-only "#payroll" under Convex (alpha = 0.3) MUST Admit
    let mut pay_req = SearchRequest::new("#payroll");
    pay_req.bucket_id = Some(s.bucket_a);
    pay_req.fusion = Some(FusionMode::Convex);
    pay_req.alpha = Some(0.3);
    let resp_pay = retriever.search(&s.bob_ctx, &pay_req).await?;
    assert!(resp_pay.hits.iter().any(|h| h.id == c_payroll));
    assert_eq!(resp_pay.admission, EvidenceAdmission::Admit);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_bug6_thumb_tag_starvation_and_punct_tag_null_embedding(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    // Seed 95 noise-tagged Thumb documents + pages (exceeds overfetch 80)
    for i in 0..95 {
        let doc_id = Uuid::new_v4();
        let page_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO corpus.documents
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', $7, $8, ARRAY['noise']::text[])",
        )
        .bind(doc_id)
        .bind(s.org_id)
        .bind(s.bucket_a)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_id}", s.slug))
        .bind(format!("Noise Thumb {i}"))
        .bind(format!("sha-noise-{i}"))
        .execute(admin.inner())
        .await?;

        sqlx::query(
            "INSERT INTO corpus.pages
             (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, redacted_markdown)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 1, '아키텍처 다이어그램 상세 설명 노이즈')",
        )
        .bind(page_id)
        .bind(s.org_id)
        .bind(doc_id)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:page:L3/{page_id}", s.slug))
        .execute(admin.inner())
        .await?;
    }

    // Seed 1 target-tag Thumb document + page
    let target_doc = Uuid::new_v4();
    let target_page = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 'Target Thumb', 'sha-target', ARRAY['target-tag']::text[])",
    )
    .bind(target_doc)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{target_doc}", s.slug))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.pages
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, redacted_markdown)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 1, '아키텍처 다이어그램 상세 설명')",
    )
    .bind(target_page)
    .bind(s.org_id)
    .bind(target_doc)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:page:L3/{target_page}", s.slug))
    .execute(admin.inner())
    .await?;

    // Reviewer minor note: wiki page with embedding IS NULL and tag 'security'
    let sec_wiki = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 'sec-null', '보안 위키', '망분리 점검', '망분리 점검', 'wsec', NULL, ARRAY['security']::text[])",
    )
    .bind(sec_wiki)
    .bind(s.org_id)
    .bind(s.bucket_a)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:policy:dept:wiki:L3/{sec_wiki}", s.slug))
    .execute(admin.inner())
    .await?;

    let retriever = PgHybridRetriever::new_pg(app);

    let mut req_thumb = SearchRequest::new("아키텍처 다이어그램 상세 설명 #target-tag");
    req_thumb.bucket_id = Some(s.bucket_a);
    req_thumb.spaces = Some(vec![RetrieveSpace::Thumb]);
    req_thumb.top_k = Some(10);
    let resp_thumb = retriever.search(&s.bob_ctx, &req_thumb).await?;
    assert_eq!(resp_thumb.hits.len(), 1);
    assert_eq!(resp_thumb.hits[0].id, target_page);
    assert_eq!(resp_thumb.admission, EvidenceAdmission::Admit);

    let mut req_punct = SearchRequest::new("#security !!!");
    req_punct.bucket_id = Some(s.bucket_a);
    req_punct.spaces = Some(vec![RetrieveSpace::Wiki]);
    let resp_punct = retriever.search(&s.bob_ctx, &req_punct).await?;
    assert!(resp_punct.hits.iter().any(|h| h.id == sec_wiki));
    assert_eq!(resp_punct.admission, EvidenceAdmission::Admit);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_bm25_partial_archived_saturation_consistent_idf_scale(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let term = "idfinversionprobe";
    let emb = Vector::from(DeterministicTextEmbedder::embed_sync(term)?);

    // Seed 37 archived documents + chunks with 4x TF (partial saturation of space_limit = 40 for top_k = 10).
    for i in 0..37 {
        let doc_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO corpus.documents
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, status)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', $7, $8, 'archived')",
        )
        .bind(doc_id)
        .bind(s.org_id)
        .bind(s.bucket_a)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_id}", s.slug))
        .bind(format!("Archived Saturation {i}"))
        .bind(format!("sha-sat-arch-{i}"))
        .execute(admin.inner())
        .await?;

        let body = format!("{term} {term} {term} {term} archived {i}");
        sqlx::query(
            "INSERT INTO corpus.chunks
             (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 0, $7, $7, $8, $9)",
        )
        .bind(chunk_id)
        .bind(s.org_id)
        .bind(doc_id)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{chunk_id}", s.slug))
        .bind(&body)
        .bind(format!("h-sat-arch-{i}"))
        .bind(emb.clone())
        .execute(admin.inner())
        .await?;
    }

    // Seed 5 active chunks: 3x TF, 2x TF, and three 1x TF chunks.
    let active_top = Uuid::new_v4();
    let active_mid = Uuid::new_v4();
    let active_low_a = Uuid::new_v4();
    let active_low_b = Uuid::new_v4();
    let active_low_c = Uuid::new_v4();

    let active_specs = [
        (
            active_top,
            "Active Top 3x",
            "sha-act-top",
            format!("{term} {term} {term} top active"),
        ),
        (
            active_mid,
            "Active Mid 2x",
            "sha-act-mid",
            format!("{term} {term} mid active"),
        ),
        (
            active_low_a,
            "Active Low A 1x",
            "sha-act-lowa",
            format!("{term} low a"),
        ),
        (
            active_low_b,
            "Active Low B 1x",
            "sha-act-lowb",
            format!("{term} low b"),
        ),
        (
            active_low_c,
            "Active Low C 1x",
            "sha-act-lowc",
            format!("{term} low c"),
        ),
    ];

    for (chunk_id, title, sha, body) in &active_specs {
        let doc_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO corpus.documents
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, status)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', $7, $8, 'indexed')",
        )
        .bind(doc_id)
        .bind(s.org_id)
        .bind(s.bucket_a)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:doc:L3/{doc_id}", s.slug))
        .bind(title)
        .bind(sha)
        .execute(admin.inner())
        .await?;

        sqlx::query(
            "INSERT INTO corpus.chunks
             (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 0, $7, $7, $8, $9)",
        )
        .bind(chunk_id)
        .bind(s.org_id)
        .bind(doc_id)
        .bind(s.dept_id)
        .bind(s.admin)
        .bind(format!("arn:openrag:{}:policy:dept:chunk:L3/{chunk_id}", s.slug))
        .bind(body)
        .bind(format!("h-{sha}"))
        .bind(emb.clone())
        .execute(admin.inner())
        .await?;
    }

    let retriever = PgHybridRetriever::new_pg(app);
    let mut req = SearchRequest::new(term);
    req.bucket_id = Some(s.bucket_a);
    req.spaces = Some(vec![RetrieveSpace::Chunks]);
    req.top_k = Some(10);
    req.fusion = Some(FusionMode::Convex);
    req.alpha = Some(1.0);
    req.rerank = Some(false);

    let resp = retriever.search(&s.bob_ctx, &req).await?;
    assert_eq!(
        resp.hits.len(),
        5,
        "expected all 5 active chunks and 0 archived"
    );
    assert_eq!(
        resp.hits[0].id, active_top,
        "3x TF active chunk must rank #1"
    );
    assert_eq!(
        resp.hits[1].id, active_mid,
        "2x TF active chunk must rank #2"
    );

    let low_ids = [active_low_a, active_low_b, active_low_c];
    for hit in &resp.hits[2..] {
        assert!(
            low_ids.contains(&hit.id),
            "expected 1x TF active chunk after top and mid, got {}",
            hit.id
        );
    }

    assert!(
        (resp.hits[0].score_breakdown.bm25_score - 1.0).abs() < 1e-5,
        "3x TF chunk should have normalized bm25_score = 1.0, got {}",
        resp.hits[0].score_breakdown.bm25_score
    );
    assert!(
        resp.hits[0].score_breakdown.bm25_score > resp.hits[1].score_breakdown.bm25_score
            && resp.hits[1].score_breakdown.bm25_score > resp.hits[2].score_breakdown.bm25_score,
        "expected strict bm25_score ordering 3x > 2x > 1x: {:?}",
        resp.hits
            .iter()
            .map(|h| (h.id, h.score_breakdown.bm25_score))
            .collect::<Vec<_>>()
    );
    assert_eq!(resp.admission, EvidenceAdmission::Admit);

    Ok(())
}
