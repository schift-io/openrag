use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, IamAction, OpenRagError, PrincipalContext,
};
use openrag_db::DbPool;
use openrag_iam::{
    GroupDefinition, GroupMembership, IamEvaluator, IamPolicy, PolicyEffect, PolicyStatement,
    ResourceDescriptor, ResourceGroupBinding,
};
use pgvector::Vector;
use serde_json::json;
use sqlx::Row;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

fn normalized_256d(seed: f32) -> Vector {
    let mut vals = vec![0.01_f32; 256];
    vals[0] = seed;
    vals[1] = seed * 0.5;
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vals {
            *v /= norm;
        }
    }
    Vector::from(vals)
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_run_migrations_on_fresh_database() -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let admin_pool = DbPool::connect(&admin_url, 4).await?;

    let fresh_db = format!(
        "openrag_mig_{}",
        Uuid::new_v4().to_string().replace('-', "")
    );
    sqlx::query(&format!("CREATE DATABASE {fresh_db}"))
        .execute(admin_pool.inner())
        .await?;

    let base = admin_url
        .rsplit_once('/')
        .map(|(prefix, _)| prefix)
        .ok_or("invalid ADMIN_DATABASE_URL")?;
    let fresh_url = format!("{base}/{fresh_db}");

    let fresh_pool = DbPool::connect(&fresh_url, 4).await?;
    let first = fresh_pool.run_migrations().await?;
    assert_eq!(first.len(), 15);

    // Idempotent second run
    let second = fresh_pool.run_migrations().await?;
    assert_eq!(second.len(), 15);
    assert_eq!(first, second);

    // Verify all 10 schemas exist
    let schema_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM information_schema.schemata
         WHERE schema_name IN ('iam','vectors','corpus','pii','memory','wiki','indexing','gateway','plugins','eval')",
    )
    .fetch_one(fresh_pool.inner())
    .await?;
    assert_eq!(schema_count, 10);

    // Close fresh pool before dropping DB
    fresh_pool.inner().close().await;
    sqlx::query(&format!("DROP DATABASE {fresh_db}"))
        .execute(admin_pool.inner())
        .await?;

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_detect_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    let app_url = env_url("DATABASE_URL")?;
    let pool = DbPool::connect(&app_url, 4).await?;
    pool.health_check().await?;
    let caps = pool.capabilities().await?;
    assert!(caps.has_pgvector);
    assert!(caps.has_pg_trgm);
    assert!(!caps.has_pg_pinecone);
    assert!(!caps.has_pg_search);
    assert!(caps.pg_version_num >= 170_000);
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_rls_non_superuser_3axis_and_iam_parity() -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let app_url = env_url("DATABASE_URL")?;

    let admin = DbPool::connect(&admin_url, 4).await?;
    let app = DbPool::connect(&app_url, 4).await?;

    // 1. Confirm non-superuser openrag_app role
    let role_row = sqlx::query(
        "SELECT current_user::text AS uname, usesuper, usebypassrls FROM pg_user WHERE usename = current_user",
    )
    .fetch_one(app.inner())
    .await?;
    let uname: String = role_row.try_get("uname")?;
    let usesuper: bool = role_row.try_get("usesuper")?;
    let usebypassrls: bool = role_row.try_get("usebypassrls")?;
    assert_eq!(uname, "openrag_app");
    assert!(!usesuper);
    assert!(!usebypassrls);

    // 2. Seed rich fixtures via admin pool
    let org_id = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let hr_dept = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let dave = Uuid::new_v4();

    let g_audit = Uuid::new_v4();
    let g_blocked = Uuid::new_v4();
    let g_capped = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Acme Gov')")
        .bind(org_id)
        .bind(format!("acme-{}", org_id.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $3, 'Finance', 'finance'),
         ($2, $3, 'HR', 'hr')",
    )
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $5, $6, 'Alice Owner', 8, true, false),
         ($2, $5, $7, 'Bob Peer', 9, false, false),
         ($3, $5, $8, 'Carol FinAdmin', 7, true, false),
         ($4, $5, $9, 'Dave HRAdmin', 9, false, false)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(org_id)
    .bind(format!("alice-{}@acme.test", org_id.simple()))
    .bind(format!("bob-{}@acme.test", org_id.simple()))
    .bind(format!("carol-{}@acme.test", org_id.simple()))
    .bind(format!("dave-{}@acme.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $4, 'member'),
         ($1, $2, $5, 'member'),
         ($1, $2, $6, 'dept_admin'),
         ($1, $3, $7, 'dept_admin')",
    )
    .bind(org_id)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, group_type, min_auth_level, max_auth_level, allowed_actions) VALUES
         ($1, $4, 'audit-tf', 'project_tf', 1, 10, ARRAY['read','search']),
         ($2, $4, 'blocked-group', 'permission_group', 1, 10, ARRAY['read']),
         ($3, $4, 'capped-group', 'clearance', 1, 8, ARRAY['read','search'])",
    )
    .bind(g_audit)
    .bind(g_blocked)
    .bind(g_capped)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    // Bob in audit-tf & blocked-group; Carol in audit-tf (level_override=5) & capped-group
    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id, level_override) VALUES
         ($1, $2, $5, NULL),
         ($1, $3, $5, NULL),
         ($1, $2, $6, 5),
         ($1, $4, $6, NULL)",
    )
    .bind(org_id)
    .bind(g_audit)
    .bind(g_blocked)
    .bind(g_capped)
    .bind(bob)
    .bind(carol)
    .execute(admin.inner())
    .await?;

    // Org-wide Allow * policy attached to Bob (proves private isolation & deny override even Allow *)
    let pol_allow_id = Uuid::new_v4();
    sqlx::query("INSERT INTO iam.policies (id, org_id, name, document) VALUES ($1, $2, 'allow-all', $3)")
        .bind(pol_allow_id)
        .bind(org_id)
        .bind(json!({
            "statements": [{
                "sid": "allow-star",
                "effect": "Allow",
                "actions": ["*"],
                "resources": ["*"]
            }]
        }))
        .execute(admin.inner())
        .await?;
    sqlx::query(
        "INSERT INTO iam.policy_attachments (org_id, policy_id, target_type, target_id) VALUES ($1, $2, 'principal', $3)",
    )
    .bind(org_id)
    .bind(pol_allow_id)
    .bind(bob)
    .execute(admin.inner())
    .await?;

    // Policy with min_auth_level_override = 9 attached to Carol (L7) vs Bob (L9) on an L4 doc
    let pol_l9_id = Uuid::new_v4();
    let doc_arn_l9_target = Uuid::new_v4();
    let doc_l9_arn = format!("arn:openrag:acme:hr:group:document:L4/{doc_arn_l9_target}");
    sqlx::query("INSERT INTO iam.policies (id, org_id, name, document) VALUES ($1, $2, 'l9-arn-allow', $3)")
        .bind(pol_l9_id)
        .bind(org_id)
        .bind(json!({
            "statements": [{
                "sid": "allow-hr-doc",
                "effect": "Allow",
                "actions": ["read"],
                "resources": [doc_l9_arn]
            }]
        }))
        .execute(admin.inner())
        .await?;
    sqlx::query(
        "INSERT INTO iam.policy_attachments (org_id, policy_id, target_type, target_id, min_auth_level_override) VALUES
         ($1, $2, 'principal', $3, 9),
         ($1, $2, 'principal', $4, 9)",
    )
    .bind(org_id)
    .bind(pol_l9_id)
    .bind(carol)
    .bind(dave)
    .execute(admin.inner())
    .await?;

    // API key for Alice capped at max_auth_level = 5 and scoped to finance/*
    let alice_api_key = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.api_keys (id, org_id, principal_id, name, key_prefix, key_hash, scoped_arns, max_auth_level)
         VALUES ($1, $2, $3, 'alice-scoped', 'orag_test', $4, ARRAY['arn:openrag:acme:finance:*'], 5)",
    )
    .bind(alice_api_key)
    .bind(org_id)
    .bind(alice)
    .bind(format!("hash_{}", alice_api_key.simple()))
    .execute(admin.inner())
    .await?;

    // Bucket & Documents
    let bucket_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
         VALUES ($1, $2, $3, $4, 'fin-bucket', 'Finance Bucket', 'dept', 1, $5)",
    )
    .bind(bucket_id)
    .bind(org_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:dept:bucket:L1/{bucket_id}"))
    .execute(admin.inner())
    .await?;

    let doc_private = Uuid::new_v4();
    let doc_group_deny = Uuid::new_v4();
    let doc_l9_capped = Uuid::new_v4();
    let doc_binding_l6 = Uuid::new_v4();
    let doc_org_l3 = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $6, $7, $8, $9, 'private', ARRAY[]::UUID[], ARRAY[]::UUID[], 5, $10, 'fin:priv:L5', 'Alice Private Budget', 'sha1'),
         ($2, $6, $7, $8, $9, 'group', ARRAY[$11], ARRAY[$12], 3, $13, 'fin:grp:L3', 'Audit TF Blocked', 'sha2'),
         ($3, $6, $7, $8, $9, 'group', ARRAY[$14], ARRAY[]::UUID[], 9, $15, 'fin:grp:L9', 'Capped L9 Secret', 'sha3'),
         ($4, $6, $7, $8, $9, 'group', ARRAY[]::UUID[], ARRAY[]::UUID[], 4, $16, 'fin:grp:L4', 'Binding L6 Report', 'sha4'),
         ($5, $6, $7, $8, $9, 'org', ARRAY[]::UUID[], ARRAY[]::UUID[], 3, $17, 'org:L3', 'Org Handbook', 'sha5'),
         ($18, $6, $7, $19, $9, 'group', ARRAY[]::UUID[], ARRAY[]::UUID[], 4, $20, 'hr:grp:L4', 'Policy L9 Override Doc', 'sha6')",
    )
    .bind(doc_private)
    .bind(doc_group_deny)
    .bind(doc_l9_capped)
    .bind(doc_binding_l6)
    .bind(doc_org_l3)
    .bind(org_id)
    .bind(bucket_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:document:L5/{doc_private}"))
    .bind(g_audit)
    .bind(g_blocked)
    .bind(format!("arn:openrag:acme:finance:group:document:L3/{doc_group_deny}"))
    .bind(g_capped)
    .bind(format!("arn:openrag:acme:finance:group:document:L9/{doc_l9_capped}"))
    .bind(format!("arn:openrag:acme:finance:group:document:L4/{doc_binding_l6}"))
    .bind(format!("arn:openrag:acme:finance:org:document:L3/{doc_org_l3}"))
    .bind(doc_arn_l9_target)
    .bind(hr_dept)
    .bind(doc_l9_arn.clone())
    .execute(admin.inner())
    .await?;

    // ResourceGroupBinding allow on doc_binding_l6 requiring min_auth_level = 6 for audit-tf
    sqlx::query(
        "INSERT INTO iam.resource_group_bindings (org_id, resource_type, resource_id, group_id, effect, actions, min_auth_level)
         VALUES ($1, 'document', $2, $3, 'allow', ARRAY['read','search'], 6)",
    )
    .bind(org_id)
    .bind(doc_binding_l6)
    .bind(g_audit)
    .execute(admin.inner())
    .await?;

    // Seed 256d chunk embeddings + Korean bm25_text on doc_private and doc_org_l3
    let chunk_priv = Uuid::new_v4();
    let chunk_org = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
         VALUES
         ($1, $3, $5, $7, $8, 'private', 5, $9, 'fin:priv:L5', 0, '재무팀 내부 감사 기밀 보고서', '재무팀 내부 감사 기밀 보고서', 'th1', $11),
         ($2, $4, $6, $7, $8, 'org', 3, $10, 'org:L3', 0, '전사 출장비 정산 규정 안내', '전사 출장비 정산 규정 안내', 'th2', $12)",
    )
    .bind(chunk_priv)
    .bind(chunk_org)
    .bind(org_id)
    .bind(org_id)
    .bind(doc_private)
    .bind(doc_org_l3)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:chunk:L5/{chunk_priv}"))
    .bind(format!("arn:openrag:acme:finance:org:chunk:L3/{chunk_org}"))
    .bind(normalized_256d(0.95))
    .bind(normalized_256d(0.55))
    .execute(admin.inner())
    .await?;

    // Seed private memory entry
    let mem_priv = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.entries
         (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, embedding)
         VALUES ($1, $2, $3, $4, 'private', 5, $5, 'fin:mem:L5', 'fact', 'Alice private note', 'Alice private note', 'msha1', $6)",
    )
    .bind(mem_priv)
    .bind(org_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:memory:L5/{mem_priv}"))
    .bind(normalized_256d(0.9))
    .execute(admin.inner())
    .await?;

    // 3. Verify Fail-Closed when GUCs unset or malformed on openrag_app
    let unset_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM corpus.documents")
        .fetch_one(app.inner())
        .await?;
    assert_eq!(unset_cnt, 0);

    {
        let mut tx = app.inner().begin().await?;
        sqlx::query(
            "SELECT set_config('app.org_id', 'not-a-uuid', true),
                    set_config('app.principal_id', '', true),
                    set_config('app.dept_ids', '{bad-uuid}', true)",
        )
        .execute(&mut *tx)
        .await?;
        let bad_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM corpus.documents")
            .fetch_one(&mut *tx)
            .await?;
        assert_eq!(bad_cnt, 0);
        tx.rollback().await?;
    }

    // 4. Helper contexts
    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(8)?);
    alice_ctx.dept_ids = vec![fin_dept];
    alice_ctx.can_unmask_pii = true;

    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![fin_dept];
    bob_ctx.group_ids = vec![g_audit, g_blocked];

    let mut carol_ctx = PrincipalContext::new(org_id, carol, AuthLevel::new(7)?);
    carol_ctx.dept_ids = vec![fin_dept];
    carol_ctx.dept_admin_ids = vec![fin_dept];
    carol_ctx.group_ids = vec![g_audit, g_capped];
    carol_ctx.can_unmask_pii = true;

    let mut dave_ctx = PrincipalContext::new(org_id, dave, AuthLevel::new(9)?);
    dave_ctx.dept_ids = vec![hr_dept];
    dave_ctx.dept_admin_ids = vec![hr_dept];

    // 5. Verify Private Isolation + Parity with openrag-iam
    let mut tx_bob = app.begin_rls_tx(&bob_ctx).await?;
    let bob_docs: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM corpus.documents")
        .fetch_all(&mut **tx_bob.as_mut())
        .await?;
    let bob_mems: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM memory.entries")
        .fetch_all(&mut **tx_bob.as_mut())
        .await?;
    tx_bob.rollback().await?;

    // Bob CANNOT see Alice's private doc/memory, CANNOT see doc_group_deny (blocked-group), CANNOT see doc_l9_capped
    assert!(!bob_docs.contains(&doc_private));
    assert!(bob_mems.is_empty());
    assert!(!bob_docs.contains(&doc_group_deny));
    assert!(bob_docs.contains(&doc_binding_l6)); // Bob L9 >= 6 via audit-tf
    assert!(bob_docs.contains(&doc_org_l3));

    // Carol (Finance dept_admin, L7, level_override=5 in audit-tf) sees private doc + memory,
    // does NOT see doc_l9_capped (L9), does NOT see doc_binding_l6 (5 < 6), does NOT see doc_arn_l9_target (7 < 9)
    let mut tx_carol = app.begin_rls_tx(&carol_ctx).await?;
    let carol_docs: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM corpus.documents")
        .fetch_all(&mut **tx_carol.as_mut())
        .await?;
    let carol_mems: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM memory.entries")
        .fetch_all(&mut **tx_carol.as_mut())
        .await?;
    tx_carol.rollback().await?;
    assert!(carol_docs.contains(&doc_private));
    assert_eq!(carol_mems, vec![mem_priv]);
    assert!(!carol_docs.contains(&doc_l9_capped));
    assert!(!carol_docs.contains(&doc_binding_l6));
    assert!(!carol_docs.contains(&doc_arn_l9_target));

    // Dave (HR dept_admin, L9) does NOT see Finance private doc/memory, DOES see doc_arn_l9_target (L9 >= override 9)
    let mut tx_dave = app.begin_rls_tx(&dave_ctx).await?;
    let dave_docs: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM corpus.documents")
        .fetch_all(&mut **tx_dave.as_mut())
        .await?;
    let dave_mems: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM memory.entries")
        .fetch_all(&mut **tx_dave.as_mut())
        .await?;
    tx_dave.rollback().await?;
    assert!(!dave_docs.contains(&doc_private));
    assert!(dave_mems.is_empty());
    assert!(dave_docs.contains(&doc_arn_l9_target));

    // Alice with API key ceiling (max_auth_level=5, scoped_arns=finance/*) sees doc_private (L5) and not doc_l9_capped or hr doc
    let mut alice_ak_ctx = alice_ctx.clone();
    alice_ak_ctx.api_key_id = Some(alice_api_key);
    alice_ak_ctx.scoped_arns = vec!["arn:openrag:acme:finance:*".into()];
    let mut tx_alice_ak = app.begin_rls_tx(&alice_ak_ctx).await?;
    let alice_ak_docs: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM corpus.documents")
        .fetch_all(&mut **tx_alice_ak.as_mut())
        .await?;
    tx_alice_ak.rollback().await?;
    assert!(alice_ak_docs.contains(&doc_private));
    assert!(alice_ak_docs.contains(&doc_org_l3));
    assert!(!alice_ak_docs.contains(&doc_arn_l9_target));

    // 6. Verify 100% SQL RLS <-> Rust IamEvaluator Parity on private & group_deny cases
    let priv_desc = ResourceDescriptor {
        org_id,
        resource_type: ArnResourceType::Document,
        resource_id: doc_private,
        dept_id: fin_dept,
        owner_principal_id: alice,
        access_scope: AccessScope::Private,
        allowed_group_ids: vec![],
        denied_group_ids: vec![],
        min_auth_level: AuthLevel::new(5)?,
        arn: Arn::parse(&format!("arn:openrag:acme:finance:private:document:L5/{doc_private}"))?,
    };
    let allow_star = IamPolicy {
        id: pol_allow_id,
        name: "allow-all".into(),
        statements: vec![PolicyStatement {
            sid: Some("allow-star".into()),
            effect: PolicyEffect::Allow,
            actions: vec![IamAction::Wildcard],
            resources: vec!["*".into()],
            condition: None,
        }],
        min_auth_level_override: None,
    };
    assert_eq!(
        bob_docs.contains(&doc_private),
        IamEvaluator::evaluate(&bob_ctx, &priv_desc, IamAction::Read, &[], &[], &[], &[allow_star]).allowed
    );
    assert_eq!(
        carol_docs.contains(&doc_private),
        IamEvaluator::evaluate(&carol_ctx, &priv_desc, IamAction::Read, &[], &[], &[], &[]).allowed
    );

    let deny_desc = ResourceDescriptor {
        org_id,
        resource_type: ArnResourceType::Document,
        resource_id: doc_group_deny,
        dept_id: fin_dept,
        owner_principal_id: alice,
        access_scope: AccessScope::Group,
        allowed_group_ids: vec![g_audit],
        denied_group_ids: vec![g_blocked],
        min_auth_level: AuthLevel::new(3)?,
        arn: Arn::parse(&format!("arn:openrag:acme:finance:group:document:L3/{doc_group_deny}"))?,
    };
    let rust_groups = vec![GroupDefinition {
        id: g_audit,
        org_id,
        name: "audit-tf".into(),
        min_auth_level: AuthLevel::new(1)?,
        max_auth_level: AuthLevel::new(10)?,
        allowed_actions: vec![IamAction::Read, IamAction::Search],
    }];
    let rust_members = vec![GroupMembership {
        group_id: g_audit,
        principal_id: bob,
        action_overrides: None,
        level_override: None,
    }];
    assert_eq!(
        bob_docs.contains(&doc_group_deny),
        IamEvaluator::evaluate(&bob_ctx, &deny_desc, IamAction::Read, &rust_groups, &rust_members, &[], &[]).allowed
    );

    // 7. Per-action SQL check: pii:unmask denied for Bob (can_unmask_pii=false), allowed for Alice
    let mut tx_bob_act = app.begin_rls_tx(&bob_ctx).await?;
    let bob_can_unmask: bool = sqlx::query_scalar(
        "SELECT iam.rls_can_access($1, $2, $3, 'org', ARRAY[]::UUID[], 'arn:openrag:acme:finance:org:document:L3/x', 3::smallint, ARRAY[]::UUID[], 'pii:unmask', NULL, NULL)",
    )
    .bind(org_id)
    .bind(fin_dept)
    .bind(alice)
    .fetch_one(&mut **tx_bob_act.as_mut())
    .await?;
    tx_bob_act.rollback().await?;
    assert!(!bob_can_unmask);

    // 8. INSERT & UPDATE RLS checks on non-superuser openrag_app
    let mut tx_insert = app.begin_rls_tx(&alice_ctx).await?;
    let new_doc = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES ($1, $2, $3, $4, $5, 'private', 5, $6, 'fin:priv:L5', 'Alice New Doc', 'sha_new')",
    )
    .bind(new_doc)
    .bind(org_id)
    .bind(bucket_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:document:L5/{new_doc}"))
    .execute(&mut **tx_insert.as_mut())
    .await
    .map_err(|e| format!("alice owner insert failed: {e}"))?;
    tx_insert.commit().await?;

    // Peer Bob trying to spoof Alice as owner_principal_id MUST fail with 42501
    let mut tx_spoof = app.begin_rls_tx(&bob_ctx).await?;
    let spoof_doc = Uuid::new_v4();
    let spoof_res = sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES ($1, $2, $3, $4, $5, 'private', 5, $6, 'fin:priv:L5', 'Spoofed', 'sha_spoof')",
    )
    .bind(spoof_doc)
    .bind(org_id)
    .bind(bucket_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:document:L5/{spoof_doc}"))
    .execute(&mut **tx_spoof.as_mut())
    .await;
    assert!(spoof_res.is_err());
    tx_spoof.rollback().await?;

    // Carol (Finance dept_admin) can UPDATE Alice's doc scope from private -> dept
    let mut tx_upd = app.begin_rls_tx(&carol_ctx).await?;
    let upd_res = sqlx::query("UPDATE corpus.documents SET access_scope = 'dept' WHERE id = $1")
        .bind(new_doc)
        .execute(&mut **tx_upd.as_mut())
        .await
        .map_err(|e| format!("carol dept_admin update failed: {e}"))?;
    assert_eq!(upd_res.rows_affected(), 1);
    tx_upd.rollback().await?;

    // 9. 256d pgvector cosine query + pg_trgm fallback query under RLS
    let mut tx_vec = app.begin_rls_tx(&bob_ctx).await?;
    let query_vec = normalized_256d(0.95);
    let vec_ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT id FROM corpus.chunks ORDER BY embedding <=> $1 LIMIT 5",
    )
    .bind(query_vec)
    .fetch_all(&mut **tx_vec.as_mut())
    .await?;
    // Bob only gets chunk_org, NOT chunk_priv!
    assert_eq!(vec_ids, vec![chunk_org]);

    let trgm_ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT id FROM corpus.chunks WHERE bm25_text ILIKE '%출장비%' OR bm25_text % '출장비'",
    )
    .fetch_all(&mut **tx_vec.as_mut())
    .await?;
    assert_eq!(trgm_ids, vec![chunk_org]);
    tx_vec.rollback().await?;

    // Clean up unused variable warning check
    let _ = OpenRagError::Internal("ok".into());
    let _ = ResourceGroupBinding {
        resource_type: ArnResourceType::Document,
        resource_id: doc_binding_l6,
        group_id: g_audit,
        effect: PolicyEffect::Allow,
        actions: vec![IamAction::Read],
        min_auth_level: Some(AuthLevel::new(6)?),
    };

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_rls_iteration2_security_regressions() -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let app_url = env_url("DATABASE_URL")?;

    let admin = DbPool::connect(&admin_url, 4).await?;
    let app = DbPool::connect(&app_url, 4).await?;

    // Seed isolated org & principals via ADMIN_DATABASE_URL
    let org_id = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let hr_dept = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let dave = Uuid::new_v4();
    let g_audit = Uuid::new_v4();
    let g_blocked = Uuid::new_v4();
    let g_other = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'RegTest Org')")
        .bind(org_id)
        .bind(format!("reg-{}", org_id.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $3, 'Finance', 'finance'),
         ($2, $3, 'HR', 'hr')",
    )
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $5, $6, 'Alice', 8, true, false),
         ($2, $5, $7, 'Bob', 9, false, false),
         ($3, $5, $8, 'Carol', 7, true, false),
         ($4, $5, $9, 'Dave', 9, false, true)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(org_id)
    .bind(format!("alice-r2-{}@acme.test", org_id.simple()))
    .bind(format!("bob-r2-{}@acme.test", org_id.simple()))
    .bind(format!("carol-r2-{}@acme.test", org_id.simple()))
    .bind(format!("dave-r2-{}@acme.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $4, 'member'),
         ($1, $2, $5, 'member'),
         ($1, $2, $6, 'dept_admin'),
         ($1, $3, $7, 'dept_admin')",
    )
    .bind(org_id)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, group_type, min_auth_level, max_auth_level, allowed_actions) VALUES
         ($1, $4, 'audit', 'project_tf', 1, 10, ARRAY['read','search']),
         ($2, $4, 'blocked', 'permission_group', 1, 10, ARRAY['read']),
         ($3, $4, 'other', 'permission_group', 1, 10, ARRAY['read'])",
    )
    .bind(g_audit)
    .bind(g_blocked)
    .bind(g_other)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
         ($1, $2, $5),
         ($1, $3, $5),
         ($1, $4, $5)",
    )
    .bind(org_id)
    .bind(g_audit)
    .bind(g_blocked)
    .bind(g_other)
    .bind(bob)
    .execute(admin.inner())
    .await?;

    // -------------------------------------------------------------------------
    // CRITICAL-1: API key scoped_group_ids & GUC omission CANNOT bypass Explicit Deny
    // -------------------------------------------------------------------------
    let alice_priv_bucket = Uuid::new_v4();
    let bob_bucket = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $3, $4, $5, 'alice-priv', 'Alice Priv Bucket', 'private', 1, $7),
         ($2, $3, $4, $6, 'bob-own', 'Bob Bucket', 'dept', 1, $8)",
    )
    .bind(alice_priv_bucket)
    .bind(bob_bucket)
    .bind(org_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(bob)
    .bind(format!("arn:openrag:acme:finance:private:bucket:L1/{alice_priv_bucket}"))
    .bind(format!("arn:openrag:acme:finance:dept:bucket:L1/{bob_bucket}"))
    .execute(admin.inner())
    .await?;

    let doc_deny_col = Uuid::new_v4();
    let doc_deny_rgb = Uuid::new_v4();
    let doc_deny_pol = Uuid::new_v4();
    let doc_allow_only = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $5, $6, $7, $8, 'group', ARRAY[$9], ARRAY[$10], 3, $11, 'k1', 'Deny Col', 's1'),
         ($2, $5, $6, $7, $8, 'group', ARRAY[$9], ARRAY[]::UUID[], 3, $12, 'k2', 'Deny RGB', 's2'),
         ($3, $5, $6, $7, $8, 'group', ARRAY[$9], ARRAY[]::UUID[], 3, $13, 'k3', 'Deny Pol', 's3'),
         ($4, $5, $6, $7, $8, 'group', ARRAY[$9], ARRAY[]::UUID[], 3, $14, 'k4', 'Allow Only', 's4')",
    )
    .bind(doc_deny_col)
    .bind(doc_deny_rgb)
    .bind(doc_deny_pol)
    .bind(doc_allow_only)
    .bind(org_id)
    .bind(alice_priv_bucket)
    .bind(fin_dept)
    .bind(alice)
    .bind(g_audit)
    .bind(g_blocked)
    .bind(format!("arn:openrag:acme:finance:group:document:L3/{doc_deny_col}"))
    .bind(format!("arn:openrag:acme:finance:group:document:L3/{doc_deny_rgb}"))
    .bind(format!("arn:openrag:acme:finance:group:document:L3/{doc_deny_pol}"))
    .bind(format!("arn:openrag:acme:finance:group:document:L3/{doc_allow_only}"))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.resource_group_bindings (org_id, resource_type, resource_id, group_id, effect, actions)
         VALUES ($1, 'document', $2, $3, 'deny', ARRAY['read'])",
    )
    .bind(org_id)
    .bind(doc_deny_rgb)
    .bind(g_blocked)
    .execute(admin.inner())
    .await?;

    let deny_pol_id = Uuid::new_v4();
    let deny_pol_arn = format!("arn:openrag:acme:finance:group:document:L3/{doc_deny_pol}");
    sqlx::query("INSERT INTO iam.policies (id, org_id, name, document) VALUES ($1, $2, 'deny-pol', $3)")
        .bind(deny_pol_id)
        .bind(org_id)
        .bind(json!({
            "statements": [{
                "sid": "deny-blocked-group",
                "effect": "Deny",
                "actions": ["read"],
                "resources": [deny_pol_arn]
            }]
        }))
        .execute(admin.inner())
        .await?;
    sqlx::query(
        "INSERT INTO iam.policy_attachments (org_id, policy_id, target_type, target_id)
         VALUES ($1, $2, 'group', $3)",
    )
    .bind(org_id)
    .bind(deny_pol_id)
    .bind(g_blocked)
    .execute(admin.inner())
    .await?;

    let bob_ak_audit = Uuid::new_v4();
    let bob_ak_other = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.api_keys (id, org_id, principal_id, name, key_prefix, key_hash, scoped_group_ids) VALUES
         ($1, $3, $4, 'bob-audit', 'orag', $5, ARRAY[$7]),
         ($2, $3, $4, 'bob-other', 'orag', $6, ARRAY[$8])",
    )
    .bind(bob_ak_audit)
    .bind(bob_ak_other)
    .bind(org_id)
    .bind(bob)
    .bind(format!("hash_a_{}", bob_ak_audit.simple()))
    .bind(format!("hash_o_{}", bob_ak_other.simple()))
    .bind(g_audit)
    .bind(g_other)
    .execute(admin.inner())
    .await?;

    // Even when Bob uses API key scoped to [g_audit] AND omits g_blocked from app.group_ids GUC,
    // all 3 Explicit Deny mechanisms block the denied docs, while doc_allow_only IS visible!
    let mut bob_ak_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(9)?);
    bob_ak_ctx.dept_ids = vec![fin_dept];
    bob_ak_ctx.group_ids = vec![g_audit]; // g_blocked intentionally omitted from GUC
    bob_ak_ctx.scoped_group_ids = vec![g_audit];
    bob_ak_ctx.api_key_id = Some(bob_ak_audit);

    let mut tx_c1 = app.begin_rls_tx(&bob_ak_ctx).await?;
    let visible_c1: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM corpus.documents")
        .fetch_all(&mut **tx_c1.as_mut())
        .await?;
    tx_c1.rollback().await?;
    assert!(!visible_c1.contains(&doc_deny_col));
    assert!(!visible_c1.contains(&doc_deny_rgb));
    assert!(!visible_c1.contains(&doc_deny_pol));
    assert!(visible_c1.contains(&doc_allow_only));

    // When Bob uses API key scoped to [g_other], Allow is narrowed and doc_allow_only is NOT visible!
    let mut bob_other_ctx = bob_ak_ctx.clone();
    bob_other_ctx.group_ids = vec![g_audit, g_blocked, g_other];
    bob_other_ctx.scoped_group_ids = vec![g_other];
    bob_other_ctx.api_key_id = Some(bob_ak_other);
    let mut tx_c1_narrow = app.begin_rls_tx(&bob_other_ctx).await?;
    let visible_narrow: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM corpus.documents")
        .fetch_all(&mut **tx_c1_narrow.as_mut())
        .await?;
    tx_c1_narrow.rollback().await?;
    assert!(visible_narrow.is_empty());

    // -------------------------------------------------------------------------
    // CRITICAL-2: Auxiliary tables enforce parent RLS (no plaintext leak)
    // -------------------------------------------------------------------------
    let mem_priv = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.entries
         (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
         VALUES ($1, $2, $3, $4, 'private', 5, $5, 'mem:priv', 'fact', 'SECRET NEW MEMORY', 'SECRET NEW MEMORY', 'msha')",
    )
    .bind(mem_priv)
    .bind(org_id)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:memory:L5/{mem_priv}"))
    .execute(admin.inner())
    .await?;
    sqlx::query(
        "INSERT INTO memory.patches (org_id, memory_entry_id, patch_op, prev_version, new_version, prev_text, new_text, actor_type, actor_id)
         VALUES ($1, $2, 'amend', 1, 2, 'SECRET PREV MEMORY TEXT', 'SECRET NEW MEMORY', 'human', $3)",
    )
    .bind(org_id)
    .bind(mem_priv)
    .bind(alice.to_string())
    .execute(admin.inner())
    .await?;

    let wiki_priv = Uuid::new_v4();
    let wiki_org = Uuid::new_v4();
    let wiki_redacted = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, status)
         VALUES
         ($1, $4, $5, $6, $7, 'private', 5, $8, 'w:priv', 'priv-page', 'Private Wiki', 'SECRET WIKI', 'SECRET WIKI', 'ws1', 'active'),
         ($2, $4, $11, $6, $7, 'org', 1, $9, 'w:org', 'org-page', 'Org Wiki', 'PUBLIC WIKI', 'PUBLIC WIKI', 'ws2', 'active'),
         ($3, $4, $11, $6, $7, 'org', 1, $10, 'w:red', 'red-page', 'Redacted Wiki', 'REDACTED', 'REDACTED', 'ws3', 'redacted')",
    )
    .bind(wiki_priv)
    .bind(wiki_org)
    .bind(wiki_redacted)
    .bind(org_id)
    .bind(alice_priv_bucket)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:wiki:L5/{wiki_priv}"))
    .bind(format!("arn:openrag:acme:finance:org:wiki:L1/{wiki_org}"))
    .bind(format!("arn:openrag:acme:finance:org:wiki:L1/{wiki_redacted}"))
    .bind(bob_bucket)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.revisions (org_id, page_id, version, body_md, body_sha256, editor_type, editor_id) VALUES
         ($1, $2, 1, 'SECRET WIKI REVISION BODY', 'rs1', 'human', $5),
         ($1, $3, 1, 'PUBLIC WIKI REVISION BODY', 'rs2', 'human', $5),
         ($1, $4, 1, 'REDACTED WIKI REVISION BODY', 'rs3', 'redactor', $5)",
    )
    .bind(org_id)
    .bind(wiki_priv)
    .bind(wiki_org)
    .bind(wiki_redacted)
    .bind(alice.to_string())
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type) VALUES ($1, $2, $3, 'related')",
    )
    .bind(org_id)
    .bind(wiki_priv)
    .bind(wiki_org)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.page_sources (org_id, page_id, source_type, source_id) VALUES ($1, $2, 'memory', $3)",
    )
    .bind(org_id)
    .bind(wiki_priv)
    .bind(mem_priv)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.sync_changes (org_id, bucket_id, page_id, file_path, direction, body_sha256) VALUES ($1, $2, $3, 'secret.md', 'export', 'ss1')",
    )
    .bind(org_id)
    .bind(alice_priv_bucket)
    .bind(wiki_priv)
    .execute(admin.inner())
    .await?;

    let chunk_priv = Uuid::new_v4();
    let chunk_org = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES
         ($1, $3, $5, $7, $8, 'private', 5, $9, 'c:priv', 0, 'SECRET CHUNK', 'SECRET CHUNK', 'ch1'),
         ($2, $3, $6, $7, $8, 'org', 1, $10, 'c:org', 0, 'ORG CHUNK', 'ORG CHUNK', 'ch2')",
    )
    .bind(chunk_priv)
    .bind(chunk_org)
    .bind(org_id)
    .bind(org_id)
    .bind(doc_deny_col)
    .bind(doc_allow_only)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:acme:finance:private:chunk:L5/{chunk_priv}"))
    .bind(format!("arn:openrag:acme:finance:org:chunk:L1/{chunk_org}"))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunk_edges (org_id, src_chunk_id, dst_chunk_id, relation_type) VALUES ($1, $2, $3, 'related')",
    )
    .bind(org_id)
    .bind(chunk_priv)
    .bind(chunk_org)
    .execute(admin.inner())
    .await?;

    // Bob (same-dept peer) and Dave (is_org_admin=true, HR dept) CANNOT read any private/redacted auxiliary rows
    let mut bob_plain_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(9)?);
    bob_plain_ctx.dept_ids = vec![fin_dept];
    let mut dave_ctx = PrincipalContext::new(org_id, dave, AuthLevel::new(9)?);
    dave_ctx.dept_ids = vec![hr_dept];
    dave_ctx.dept_admin_ids = vec![hr_dept];
    dave_ctx.is_org_admin = true;

    for ctx in [&bob_plain_ctx, &dave_ctx] {
        let mut tx = app.begin_rls_tx(ctx).await?;
        let patch_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM memory.patches")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        let rev_bodies: Vec<String> = sqlx::query_scalar("SELECT body_md FROM wiki.revisions")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        let edge_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM wiki.edges")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        let src_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM wiki.page_sources")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        let sync_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM wiki.sync_changes")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        let cedge_cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM corpus.chunk_edges")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        tx.rollback().await?;

        assert_eq!(patch_cnt, 0);
        assert_eq!(rev_bodies, vec!["PUBLIC WIKI REVISION BODY".to_string()]);
        assert_eq!(edge_cnt, 0);
        assert_eq!(src_cnt, 0);
        assert_eq!(sync_cnt, 0);
        assert_eq!(cedge_cnt, 0);
    }

    // Carol (Finance dept_admin, L7) CAN read Alice's active Finance private auxiliary rows (except redacted)
    let mut carol_ctx = PrincipalContext::new(org_id, carol, AuthLevel::new(7)?);
    carol_ctx.dept_ids = vec![fin_dept];
    carol_ctx.dept_admin_ids = vec![fin_dept];
    let mut tx_carol = app.begin_rls_tx(&carol_ctx).await?;
    let carol_patches: i64 = sqlx::query_scalar("SELECT count(*) FROM memory.patches")
        .fetch_one(&mut **tx_carol.as_mut())
        .await?;
    let carol_revs: i64 = sqlx::query_scalar("SELECT count(*) FROM wiki.revisions")
        .fetch_one(&mut **tx_carol.as_mut())
        .await?;
    let carol_edges: i64 = sqlx::query_scalar("SELECT count(*) FROM wiki.edges")
        .fetch_one(&mut **tx_carol.as_mut())
        .await?;
    tx_carol.rollback().await?;
    assert_eq!(carol_patches, 1);
    assert_eq!(carol_revs, 2); // active priv + active org, NOT redacted!
    assert_eq!(carol_edges, 1);

    // -------------------------------------------------------------------------
    // HIGH-1, MEDIUM-1, MEDIUM-2: Malformed safe_text_array, {NULL} uuid array, arn_matches('')
    // -------------------------------------------------------------------------
    {
        let mut tx = app.begin_rls_tx(&bob_ak_ctx).await?;
        sqlx::query("SELECT set_config('app.scoped_arns', '{unclosed', true)")
            .execute(&mut **tx.as_mut())
            .await?;
        let cnt_bad_arns: i64 = sqlx::query_scalar("SELECT count(*) FROM corpus.documents")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        assert_eq!(cnt_bad_arns, 0);

        sqlx::query("SELECT set_config('app.scoped_arns', '{}', true), set_config('app.dept_ids', '{NULL}', true)")
            .execute(&mut **tx.as_mut())
            .await?;
        let rls_null_elem: Option<bool> = sqlx::query_scalar(
            "SELECT iam.rls_can_access($1, $2, $3, 'dept', ARRAY[]::UUID[], 'arn:openrag:acme:finance:dept:document:L1/x', 1::smallint, ARRAY[]::UUID[], 'read', NULL, NULL)",
        )
        .bind(org_id)
        .bind(fin_dept)
        .bind(alice)
        .fetch_one(&mut **tx.as_mut())
        .await?;
        assert_eq!(rls_null_elem, Some(false));

        let arn_empty: (bool, bool, bool) = sqlx::query_as(
            "SELECT iam.arn_matches('', ''), iam.arn_matches('*', ''), iam.arn_matches('', 'arn:openrag:acme:fin:org:document:L1/1')",
        )
        .fetch_one(&mut **tx.as_mut())
        .await?;
        assert_eq!(arn_empty, (false, false, false));
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // HIGH-2: Tightened INSERT/UPDATE RLS, Ownership Reassignment Trigger, Least-Privilege Grants
    // -------------------------------------------------------------------------
    // 1. Bob (with owner_principal_id = bob) CANNOT insert doc into Alice's private bucket
    let mut tx_ins_priv = app.begin_rls_tx(&bob_plain_ctx).await?;
    let bad_doc = Uuid::new_v4();
    let ins_priv_res = sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES ($1, $2, $3, $4, $5, 'private', 3, $6, 'k', 'Bob In Alice Bucket', 'shab')",
    )
    .bind(bad_doc)
    .bind(org_id)
    .bind(alice_priv_bucket)
    .bind(fin_dept)
    .bind(bob)
    .bind(format!("arn:openrag:acme:finance:private:document:L3/{bad_doc}"))
    .execute(&mut **tx_ins_priv.as_mut())
    .await;
    assert!(ins_priv_res.is_err());
    tx_ins_priv.rollback().await?;

    // 2. Bob CAN insert doc into Bob's own bucket, but CANNOT reassign dept_id or owner_principal_id or raise min_auth_level > 9
    let bob_doc = Uuid::new_v4();
    let mut tx_bob_own = app.begin_rls_tx(&bob_plain_ctx).await?;
    sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES ($1, $2, $3, $4, $5, 'org', 3, $6, 'k', 'Bob Own Doc', 'shabob')",
    )
    .bind(bob_doc)
    .bind(org_id)
    .bind(bob_bucket)
    .bind(fin_dept)
    .bind(bob)
    .bind(format!("arn:openrag:acme:finance:org:document:L3/{bob_doc}"))
    .execute(&mut **tx_bob_own.as_mut())
    .await?;
    tx_bob_own.commit().await?;

    let mut tx_re_dept = app.begin_rls_tx(&bob_plain_ctx).await?;
    let re_dept_res = sqlx::query("UPDATE corpus.documents SET dept_id = $1 WHERE id = $2")
        .bind(hr_dept)
        .bind(bob_doc)
        .execute(&mut **tx_re_dept.as_mut())
        .await;
    assert!(re_dept_res.is_err());
    tx_re_dept.rollback().await?;

    let mut tx_re_owner = app.begin_rls_tx(&bob_plain_ctx).await?;
    let re_owner_res = sqlx::query("UPDATE corpus.documents SET owner_principal_id = $1 WHERE id = $2")
        .bind(alice)
        .bind(bob_doc)
        .execute(&mut **tx_re_owner.as_mut())
        .await;
    assert!(re_owner_res.is_err());
    tx_re_owner.rollback().await?;

    let mut tx_raise_lvl = app.begin_rls_tx(&bob_plain_ctx).await?;
    let raise_lvl_res = sqlx::query("UPDATE corpus.documents SET min_auth_level = 10 WHERE id = $1")
        .bind(bob_doc)
        .execute(&mut **tx_raise_lvl.as_mut())
        .await;
    assert!(raise_lvl_res.is_err());
    tx_raise_lvl.rollback().await?;

    // 3. Least-privilege openrag_app grant checks (denied pii.vault_keys, iam.api_keys, iam.orgs DML)
    assert!(sqlx::query("SELECT 1 FROM pii.vault_keys LIMIT 1").fetch_optional(app.inner()).await.is_err());
    assert!(sqlx::query("SELECT 1 FROM iam.api_keys LIMIT 1").fetch_optional(app.inner()).await.is_err());
    assert!(
        sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, 'rogue', 'Rogue')")
            .bind(Uuid::new_v4())
            .execute(app.inner())
            .await
            .is_err()
    );
    assert!(
        sqlx::query("INSERT INTO pii.guard_policies (id, org_id, name) VALUES ($1, $2, 'rogue')")
            .bind(Uuid::new_v4())
            .bind(org_id)
            .execute(app.inner())
            .await
            .is_err()
    );

    Ok(())
}

fn assert_pg_err_code<T>(
    res: Result<T, sqlx::Error>,
    expected_codes: &[&str],
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match res {
        Ok(_) => Err(format!(
            "{label}: expected PostgreSQL error {expected_codes:?}, but operation succeeded"
        )
        .into()),
        Err(sqlx::Error::Database(db_err)) => {
            let code = db_err.code().unwrap_or_default();
            if expected_codes.iter().any(|c| *c == code.as_ref()) {
                Ok(())
            } else {
                Err(format!(
                    "{label}: expected SQLSTATE {expected_codes:?}, got {code} ({db_err})"
                )
                .into())
            }
        }
        Err(e) => Err(format!(
            "{label}: expected Database error {expected_codes:?}, got {e}"
        )
        .into()),
    }
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_rls_iteration3_parent_rebinding_regressions() -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let app_url = env_url("DATABASE_URL")?;
    let worker_url = env_url("WORKER_DATABASE_URL")?;

    let admin = DbPool::connect(&admin_url, 4).await?;
    let app = DbPool::connect(&app_url, 4).await?;
    let worker = DbPool::connect(&worker_url, 4).await?;

    // Confirm worker has BYPASSRLS (structural FK protection must hold even for worker/superuser)
    let worker_bypass: bool = sqlx::query_scalar(
        "SELECT usebypassrls FROM pg_user WHERE usename = current_user",
    )
    .fetch_one(worker.inner())
    .await?;
    assert!(worker_bypass);

    // Seed Org 1 and Org 2
    let org1 = Uuid::new_v4();
    let org2 = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let org2_dept = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let eve = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Org1'), ($3, $4, 'Org2')")
        .bind(org1)
        .bind(format!("i3-org1-{}", org1.simple()))
        .bind(org2)
        .bind(format!("i3-org2-{}", org2.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $2, 'Finance', 'finance'),
         ($3, $4, 'Org2Dept', 'org2dept')",
    )
    .bind(fin_dept)
    .bind(org1)
    .bind(org2_dept)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $5, $7, 'Alice', 8, true, false),
         ($2, $5, $8, 'Bob', 9, false, false),
         ($3, $5, $9, 'Carol', 7, true, false),
         ($4, $6, $10, 'Eve', 9, true, true)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(eve)
    .bind(org1)
    .bind(org2)
    .bind(format!("alice-i3-{}@org1.test", org1.simple()))
    .bind(format!("bob-i3-{}@org1.test", org1.simple()))
    .bind(format!("carol-i3-{}@org1.test", org1.simple()))
    .bind(format!("eve-i3-{}@org2.test", org2.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $3, $5, 'member'),
         ($1, $3, $6, 'member'),
         ($1, $3, $7, 'dept_admin'),
         ($2, $4, $8, 'dept_admin')",
    )
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    // Buckets
    let alice_priv_bucket = Uuid::new_v4();
    let bob_bucket_1 = Uuid::new_v4();
    let bob_bucket_2 = Uuid::new_v4();
    let fin_dept_bucket = Uuid::new_v4();
    let org2_bucket = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $6, $8, $10, 'alice-priv', 'Alice Priv', 'private', 1, $13),
         ($2, $6, $8, $11, 'bob-1', 'Bob 1', 'dept', 1, $14),
         ($3, $6, $8, $11, 'bob-2', 'Bob 2', 'dept', 1, $15),
         ($4, $6, $8, $10, 'fin-dept', 'Fin Dept', 'dept', 1, $16),
         ($5, $7, $9, $12, 'org2-priv', 'Org2 Priv', 'private', 1, $17)",
    )
    .bind(alice_priv_bucket)
    .bind(bob_bucket_1)
    .bind(bob_bucket_2)
    .bind(fin_dept_bucket)
    .bind(org2_bucket)
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:bucket:L1/{alice_priv_bucket}"))
    .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bob_bucket_1}"))
    .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bob_bucket_2}"))
    .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{fin_dept_bucket}"))
    .bind(format!("arn:openrag:org2:org2dept:private:bucket:L1/{org2_bucket}"))
    .execute(admin.inner())
    .await?;

    // Documents
    let alice_priv_doc = Uuid::new_v4();
    let bob_doc = Uuid::new_v4();
    let bob_doc_2 = Uuid::new_v4();
    let carol_doc = Uuid::new_v4();
    let org2_doc = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $6, $8,  $13, $15, 'private', 3, $18, 'k1', 'Alice Priv Doc', 's1'),
         ($2, $6, $9,  $13, $16, 'org',     1, $19, 'k2', 'Bob Doc 1', 's2'),
         ($3, $6, $10, $13, $16, 'org',     1, $20, 'k3', 'Bob Doc 2', 's3'),
         ($4, $6, $11, $13, $15, 'private', 3, $21, 'k4', 'Carol Dept Doc', 's4'),
         ($5, $7, $12, $14, $17, 'private', 3, $22, 'k5', 'Org2 Priv Doc', 's5')",
    )
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(bob_doc_2)
    .bind(carol_doc)
    .bind(org2_doc)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_bucket)
    .bind(bob_bucket_1)
    .bind(bob_bucket_2)
    .bind(fin_dept_bucket)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:document:L3/{alice_priv_doc}"))
    .bind(format!("arn:openrag:org1:finance:org:document:L1/{bob_doc}"))
    .bind(format!("arn:openrag:org1:finance:org:document:L1/{bob_doc_2}"))
    .bind(format!("arn:openrag:org1:finance:private:document:L3/{carol_doc}"))
    .bind(format!("arn:openrag:org2:org2dept:private:document:L3/{org2_doc}"))
    .execute(admin.inner())
    .await?;

    // Pages
    let alice_priv_page = Uuid::new_v4();
    let bob_page = Uuid::new_v4();
    let bob_page_2 = Uuid::new_v4();
    let org2_page = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.pages
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no)
         VALUES
         ($1, $5, $7,  $11, $13, 'private', 3, $16, 'p1', 1),
         ($2, $5, $8,  $11, $14, 'org',     1, $17, 'p2', 1),
         ($3, $5, $9,  $11, $14, 'org',     1, $18, 'p3', 1),
         ($4, $6, $10, $12, $15, 'private', 3, $19, 'p4', 1)",
    )
    .bind(alice_priv_page)
    .bind(bob_page)
    .bind(bob_page_2)
    .bind(org2_page)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(bob_doc_2)
    .bind(org2_doc)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:page:L3/{alice_priv_page}"))
    .bind(format!("arn:openrag:org1:finance:org:page:L1/{bob_page}"))
    .bind(format!("arn:openrag:org1:finance:org:page:L1/{bob_page_2}"))
    .bind(format!("arn:openrag:org2:org2dept:private:page:L3/{org2_page}"))
    .execute(admin.inner())
    .await?;

    // Bob's page_figure and chunk
    let bob_figure = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.page_figures
         (id, org_id, page_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, figure_idx)
         VALUES ($1, $2, $3, $4, $5, $6, 'org', 1, $7, 'f1', 0)",
    )
    .bind(bob_figure)
    .bind(org1)
    .bind(bob_page)
    .bind(bob_doc)
    .bind(fin_dept)
    .bind(bob)
    .bind(format!("arn:openrag:org1:finance:org:figure:L1/{bob_figure}"))
    .execute(admin.inner())
    .await?;

    let bob_chunk = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES ($1, $2, $3, $4, $5, $6, 'org', 1, $7, 'c1', 0, 'bob chunk', 'bob chunk', 'cth1')",
    )
    .bind(bob_chunk)
    .bind(org1)
    .bind(bob_doc)
    .bind(bob_page)
    .bind(fin_dept)
    .bind(bob)
    .bind(format!("arn:openrag:org1:finance:org:chunk:L1/{bob_chunk}"))
    .execute(admin.inner())
    .await?;

    // Wiki pages (Alice priv, Bob, Org2)
    let alice_priv_wiki = Uuid::new_v4();
    let bob_wiki = Uuid::new_v4();
    let org2_wiki = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
         VALUES
         ($1, $4, $6, $9,  $11, 'private', 3, $14, 'w1', 'alice-w', 'Alice W', 'sec', 'sec', 'ws1'),
         ($2, $4, $7, $9,  $12, 'org',     1, $15, 'w2', 'bob-w',   'Bob W',   'pub', 'pub', 'ws2'),
         ($3, $5, $8, $10, $13, 'private', 3, $16, 'w3', 'org2-w',  'Org2 W',  'o2',  'o2',  'ws3')",
    )
    .bind(alice_priv_wiki)
    .bind(bob_wiki)
    .bind(org2_wiki)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_bucket)
    .bind(bob_bucket_1)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:wiki:L3/{alice_priv_wiki}"))
    .bind(format!("arn:openrag:org1:finance:org:wiki:L1/{bob_wiki}"))
    .bind(format!("arn:openrag:org2:org2dept:private:wiki:L3/{org2_wiki}"))
    .execute(admin.inner())
    .await?;

    // Episodes (Alice priv, Org2)
    let alice_priv_episode = Uuid::new_v4();
    let org2_episode = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.episodes
         (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
         VALUES
         ($1, $3, $5, $7, 'private', 3, $9,  'claude_code', 'alice ep', 'es1'),
         ($2, $4, $6, $8, 'private', 3, $10, 'claude_code', 'org2 ep',  'es2')",
    )
    .bind(alice_priv_episode)
    .bind(org2_episode)
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:episode:L3/{alice_priv_episode}"))
    .bind(format!("arn:openrag:org2:org2dept:private:episode:L3/{org2_episode}"))
    .execute(admin.inner())
    .await?;

    // Memory entries (Alice priv, Bob, Org2)
    let alice_priv_mem = Uuid::new_v4();
    let bob_mem = Uuid::new_v4();
    let org2_mem = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.entries
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
         VALUES
         ($1, $4, $6, $9,  $11, 'private', 3, $14, 'm1', 'fact', 'alice mem', 'alice mem', 'ms1'),
         ($2, $4, $7, $9,  $12, 'org',     1, $15, 'm2', 'fact', 'bob mem',   'bob mem',   'ms2'),
         ($3, $5, $8, $10, $13, 'private', 3, $16, 'm3', 'fact', 'org2 mem',  'org2 mem',  'ms3')",
    )
    .bind(alice_priv_mem)
    .bind(bob_mem)
    .bind(org2_mem)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_bucket)
    .bind(bob_bucket_1)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:memory:L3/{alice_priv_mem}"))
    .bind(format!("arn:openrag:org1:finance:org:memory:L1/{bob_mem}"))
    .bind(format!("arn:openrag:org2:org2dept:private:memory:L3/{org2_mem}"))
    .execute(admin.inner())
    .await?;

    let mut bob_ctx = PrincipalContext::new(org1, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![fin_dept];

    let mut carol_ctx = PrincipalContext::new(org1, carol, AuthLevel::new(7)?);
    carol_ctx.dept_ids = vec![fin_dept];
    carol_ctx.dept_admin_ids = vec![fin_dept];

    // -------------------------------------------------------------------------
    // A. All 15 Challenger 2 parent-FK rebinding & omitted INSERT bypasses -> 42501
    // -------------------------------------------------------------------------
    let rebind_updates: Vec<(&str, &str, Uuid, Uuid)> = vec![
        ("corpus.documents bucket_id -> alice_priv_bucket", "UPDATE corpus.documents SET bucket_id = $1 WHERE id = $2", alice_priv_bucket, bob_doc),
        ("corpus.documents bucket_id -> org2_bucket", "UPDATE corpus.documents SET bucket_id = $1 WHERE id = $2", org2_bucket, bob_doc),
        ("corpus.pages document_id -> alice_priv_doc", "UPDATE corpus.pages SET document_id = $1, page_no = 99 WHERE id = $2", alice_priv_doc, bob_page),
        ("corpus.pages document_id -> org2_doc", "UPDATE corpus.pages SET document_id = $1, page_no = 99 WHERE id = $2", org2_doc, bob_page),
        ("corpus.page_figures document_id -> alice_priv_doc", "UPDATE corpus.page_figures SET document_id = $1 WHERE id = $2", alice_priv_doc, bob_figure),
        ("corpus.page_figures page_id -> alice_priv_page", "UPDATE corpus.page_figures SET page_id = $1 WHERE id = $2", alice_priv_page, bob_figure),
        ("corpus.page_figures doc+page -> org2", "UPDATE corpus.page_figures SET document_id = $1, page_id = $2 WHERE id = $3", org2_doc, bob_figure),
        ("corpus.chunks document_id -> alice_priv_doc", "UPDATE corpus.chunks SET document_id = $1, chunk_idx = 99 WHERE id = $2", alice_priv_doc, bob_chunk),
        ("corpus.chunks document_id -> org2_doc", "UPDATE corpus.chunks SET document_id = $1, chunk_idx = 99 WHERE id = $2", org2_doc, bob_chunk),
        ("corpus.chunks page_id -> alice_priv_page", "UPDATE corpus.chunks SET page_id = $1 WHERE id = $2", alice_priv_page, bob_chunk),
        ("corpus.chunks page_id -> org2_page", "UPDATE corpus.chunks SET page_id = $1 WHERE id = $2", org2_page, bob_chunk),
        ("wiki.pages bucket_id -> alice_priv_bucket", "UPDATE wiki.pages SET bucket_id = $1 WHERE id = $2", alice_priv_bucket, bob_wiki),
        ("wiki.pages bucket_id -> org2_bucket", "UPDATE wiki.pages SET bucket_id = $1 WHERE id = $2", org2_bucket, bob_wiki),
        ("memory.entries bucket_id -> alice_priv_bucket", "UPDATE memory.entries SET bucket_id = $1 WHERE id = $2", alice_priv_bucket, bob_mem),
        ("memory.entries bucket_id -> org2_bucket", "UPDATE memory.entries SET bucket_id = $1 WHERE id = $2", org2_bucket, bob_mem),
    ];

    for (label, sql_str, target_parent, row_id) in rebind_updates {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let res = if label == "corpus.page_figures doc+page -> org2" {
            sqlx::query(sql_str)
                .bind(target_parent)
                .bind(org2_page)
                .bind(row_id)
                .execute(&mut **tx.as_mut())
                .await
        } else {
            sqlx::query(sql_str)
                .bind(target_parent)
                .bind(row_id)
                .execute(&mut **tx.as_mut())
                .await
        };
        assert_pg_err_code(res, &["42501"], label)?;
        tx.rollback().await?;
    }

    // Omitted INSERT cases: memory.entries(bucket_id = alice_priv_bucket | org2_bucket)
    for (label, bad_bucket) in [
        ("memory.entries INSERT bucket_id = alice_priv_bucket", alice_priv_bucket),
        ("memory.entries INSERT bucket_id = org2_bucket", org2_bucket),
    ] {
        let mem_id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let res = sqlx::query(
            "INSERT INTO memory.entries
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
             VALUES ($1, $2, $3, $4, $5, 'org', 1, $6, 'mk', 'fact', 'x', 'x', 'msx')",
        )
        .bind(mem_id)
        .bind(org1)
        .bind(bad_bucket)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:org:memory:L1/{mem_id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(res, &["42501"], label)?;
        tx.rollback().await?;
    }

    // Omitted INSERT case: corpus.chunks(document_id = bob_doc, page_id = alice_priv_page | org2_page)
    for (label, bad_page) in [
        ("corpus.chunks INSERT page_id = alice_priv_page", alice_priv_page),
        ("corpus.chunks INSERT page_id = org2_page", org2_page),
    ] {
        let cid = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let res = sqlx::query(
            "INSERT INTO corpus.chunks
             (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
             VALUES ($1, $2, $3, $4, $5, $6, 'org', 1, $7, 'ck', 55, 'x', 'x', 'thx')",
        )
        .bind(cid)
        .bind(org1)
        .bind(bob_doc)
        .bind(bad_page)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:org:chunk:L1/{cid}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(res, &["42501"], label)?;
        tx.rollback().await?;
    }

    // Also corpus.page_figures INSERT with page_id = alice_priv_page (while document_id = bob_doc)
    {
        let fid = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let res = sqlx::query(
            "INSERT INTO corpus.page_figures
             (id, org_id, page_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, figure_idx)
             VALUES ($1, $2, $3, $4, $5, $6, 'org', 1, $7, 'fk', 99)",
        )
        .bind(fid)
        .bind(org1)
        .bind(alice_priv_page)
        .bind(bob_doc)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:org:figure:L1/{fid}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(res, &["42501"], "corpus.page_figures INSERT page_id = alice_priv_page")?;
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // B. memory.entries nullable reference columns (INSERT & UPDATE) -> 42501
    // -------------------------------------------------------------------------
    let mem_ref_cases: Vec<(&str, &str, Uuid)> = vec![
        ("source_episode_id -> alice_priv_episode", "source_episode_id", alice_priv_episode),
        ("source_episode_id -> org2_episode", "source_episode_id", org2_episode),
        ("source_document_id -> alice_priv_doc", "source_document_id", alice_priv_doc),
        ("source_document_id -> org2_doc", "source_document_id", org2_doc),
        ("promoted_wiki_id -> alice_priv_wiki", "promoted_wiki_id", alice_priv_wiki),
        ("promoted_wiki_id -> org2_wiki", "promoted_wiki_id", org2_wiki),
        ("superseded_by -> alice_priv_mem", "superseded_by", alice_priv_mem),
        ("superseded_by -> org2_mem", "superseded_by", org2_mem),
    ];

    for (label, col, target_id) in mem_ref_cases {
        // UPDATE
        let mut tx_u = app.begin_rls_tx(&bob_ctx).await?;
        let res_u = sqlx::query(&format!("UPDATE memory.entries SET {col} = $1 WHERE id = $2"))
            .bind(target_id)
            .bind(bob_mem)
            .execute(&mut **tx_u.as_mut())
            .await;
        assert_pg_err_code(res_u, &["42501"], &format!("UPDATE {label}"))?;
        tx_u.rollback().await?;

        // INSERT
        let mid = Uuid::new_v4();
        let mut tx_i = app.begin_rls_tx(&bob_ctx).await?;
        let res_i = sqlx::query(&format!(
            "INSERT INTO memory.entries
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, {col})
             VALUES ($1, $2, $3, $4, $5, 'org', 1, $6, 'mk', 'fact', 'x', 'x', 'msx', $7)"
        ))
        .bind(mid)
        .bind(org1)
        .bind(bob_bucket_1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:org:memory:L1/{mid}"))
        .bind(target_id)
        .execute(&mut **tx_i.as_mut())
        .await;
        assert_pg_err_code(res_i, &["42501"], &format!("INSERT {label}"))?;
        tx_i.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // C. Positive legitimate cases MUST succeed
    // -------------------------------------------------------------------------
    {
        // Bob moves bob_doc from bob_bucket_1 -> bob_bucket_2
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r1 = sqlx::query("UPDATE corpus.documents SET bucket_id = $1 WHERE id = $2")
            .bind(bob_bucket_2)
            .bind(bob_doc)
            .execute(&mut **tx.as_mut())
            .await?;
        assert_eq!(r1.rows_affected(), 1);

        // Bob moves bob_wiki from bob_bucket_1 -> bob_bucket_2
        let r2 = sqlx::query("UPDATE wiki.pages SET bucket_id = $1 WHERE id = $2")
            .bind(bob_bucket_2)
            .bind(bob_wiki)
            .execute(&mut **tx.as_mut())
            .await?;
        assert_eq!(r2.rows_affected(), 1);

        // Bob moves bob_mem from bob_bucket_1 -> bob_bucket_2 and links bob_doc
        let r3 = sqlx::query("UPDATE memory.entries SET bucket_id = $1, source_document_id = $2 WHERE id = $3")
            .bind(bob_bucket_2)
            .bind(bob_doc)
            .bind(bob_mem)
            .execute(&mut **tx.as_mut())
            .await?;
        assert_eq!(r3.rows_affected(), 1);

        // Bob inserts a chunk on bob_doc / bob_page
        let ok_chunk = Uuid::new_v4();
        let r4 = sqlx::query(
            "INSERT INTO corpus.chunks
             (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
             VALUES ($1, $2, $3, $4, $5, $6, 'org', 1, $7, 'ck', 1, 'ok', 'ok', 'thok')",
        )
        .bind(ok_chunk)
        .bind(org1)
        .bind(bob_doc)
        .bind(bob_page)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:org:chunk:L1/{ok_chunk}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r4.rows_affected(), 1);
        tx.rollback().await?;
    }

    {
        // Carol (fin_dept admin) updates carol_doc inside fin_dept_bucket
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let r = sqlx::query("UPDATE corpus.documents SET access_scope = 'dept' WHERE id = $1")
            .bind(carol_doc)
            .execute(&mut **tx.as_mut())
            .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // D. Structural cross-org FK rejection even as BYPASSRLS openrag_worker -> 23503
    // -------------------------------------------------------------------------
    {
        let mut tx = worker.inner().begin().await?;
        let r_doc = sqlx::query("UPDATE corpus.documents SET bucket_id = $1 WHERE id = $2")
            .bind(org2_bucket)
            .bind(bob_doc)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_doc, &["23503"], "worker cross-org documents.bucket_id")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r_page = sqlx::query("UPDATE corpus.pages SET document_id = $1, page_no = 77 WHERE id = $2")
            .bind(org2_doc)
            .bind(bob_page)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_page, &["23503"], "worker cross-org pages.document_id")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r_fig = sqlx::query("UPDATE corpus.page_figures SET document_id = $1, page_id = $2 WHERE id = $3")
            .bind(org2_doc)
            .bind(org2_page)
            .bind(bob_figure)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_fig, &["23503"], "worker cross-org page_figures")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r_chunk_org = sqlx::query("UPDATE corpus.chunks SET document_id = $1, chunk_idx = 77 WHERE id = $2")
            .bind(org2_doc)
            .bind(bob_chunk)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_chunk_org, &["23503"], "worker cross-org chunks.document_id")?;
        tx.rollback().await?;
    }
    {
        // Even within same org, chunk cannot link page_id from a different document
        let mut tx = worker.inner().begin().await?;
        let r_chunk_doc = sqlx::query("UPDATE corpus.chunks SET page_id = $1 WHERE id = $2")
            .bind(bob_page_2)
            .bind(bob_chunk)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_chunk_doc, &["23503"], "worker cross-doc chunks.page_id")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r_wiki = sqlx::query("UPDATE wiki.pages SET bucket_id = $1 WHERE id = $2")
            .bind(org2_bucket)
            .bind(bob_wiki)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_wiki, &["23503"], "worker cross-org wiki.pages.bucket_id")?;
        tx.rollback().await?;
    }
    for (label, col, target_id) in [
        ("worker cross-org memory.entries.bucket_id", "bucket_id", org2_bucket),
        ("worker cross-org memory.entries.source_episode_id", "source_episode_id", org2_episode),
        ("worker cross-org memory.entries.source_document_id", "source_document_id", org2_doc),
        ("worker cross-org memory.entries.promoted_wiki_id", "promoted_wiki_id", org2_wiki),
        ("worker cross-org memory.entries.superseded_by", "superseded_by", org2_mem),
    ] {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(&format!("UPDATE memory.entries SET {col} = $1 WHERE id = $2"))
            .bind(target_id)
            .bind(bob_mem)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r, &["23503"], label)?;
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // E. Verify ON DELETE SET NULL (col) and ON DELETE CASCADE still work cleanly
    // -------------------------------------------------------------------------
    {
        let mut tx = admin.inner().begin().await?;
        // Deleting bob_page sets bob_chunk.page_id = NULL without deleting bob_chunk
        sqlx::query("DELETE FROM corpus.pages WHERE id = $1")
            .bind(bob_page)
            .execute(&mut *tx)
            .await?;
        let chunk_page_id: Option<Option<Uuid>> =
            sqlx::query_scalar("SELECT page_id FROM corpus.chunks WHERE id = $1")
                .bind(bob_chunk)
                .fetch_optional(&mut *tx)
                .await?;
        assert_eq!(chunk_page_id, Some(None));

        // Deleting bob_bucket_1 sets bob_wiki.bucket_id = NULL and bob_mem.bucket_id = NULL while cascading bob_doc
        sqlx::query("DELETE FROM corpus.buckets WHERE id = $1")
            .bind(bob_bucket_1)
            .execute(&mut *tx)
            .await?;
        let wiki_bucket: Option<Option<Uuid>> =
            sqlx::query_scalar("SELECT bucket_id FROM wiki.pages WHERE id = $1")
                .bind(bob_wiki)
                .fetch_optional(&mut *tx)
                .await?;
        assert_eq!(wiki_bucket, Some(None));
        let doc_exists: bool =
            sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM corpus.documents WHERE id = $1)")
                .bind(bob_doc)
                .fetch_one(&mut *tx)
                .await?;
        assert!(!doc_exists);

        // Deleting org2 cascades cleanly
        sqlx::query("DELETE FROM iam.orgs WHERE id = $1")
            .bind(org2)
            .execute(&mut *tx)
            .await?;
        tx.rollback().await?;
    }

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_rls_iteration4_cross_org_fk_and_oracle_regressions(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let app_url = env_url("DATABASE_URL")?;
    let worker_url = env_url("WORKER_DATABASE_URL")?;

    let admin = DbPool::connect(&admin_url, 4).await?;
    let app = DbPool::connect(&app_url, 4).await?;
    let worker = DbPool::connect(&worker_url, 4).await?;

    // Seed Org 1 and Org 2
    let org1 = Uuid::new_v4();
    let org2 = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let hr_dept = Uuid::new_v4();
    let org2_dept = Uuid::new_v4();

    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let dave = Uuid::new_v4();
    let eve = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'I4 Org1'), ($3, $4, 'I4 Org2')")
        .bind(org1)
        .bind(format!("i4-org1-{}", org1.simple()))
        .bind(org2)
        .bind(format!("i4-org2-{}", org2.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $4, 'Finance', 'finance'),
         ($2, $4, 'HR', 'hr'),
         ($3, $5, 'Org2Dept', 'org2dept')",
    )
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $6, $8,  'Alice', 8, true,  false),
         ($2, $6, $9,  'Bob',   9, false, false),
         ($3, $6, $10, 'Carol', 7, true,  false),
         ($4, $6, $11, 'Dave',  9, true,  true),
         ($5, $7, $12, 'Eve',   9, true,  true)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(eve)
    .bind(org1)
    .bind(org2)
    .bind(format!("alice-i4-{}@org1.test", org1.simple()))
    .bind(format!("bob-i4-{}@org1.test", org1.simple()))
    .bind(format!("carol-i4-{}@org1.test", org1.simple()))
    .bind(format!("dave-i4-{}@org1.test", org1.simple()))
    .bind(format!("eve-i4-{}@org2.test", org2.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $3, $6, 'member'),
         ($1, $3, $7, 'member'),
         ($1, $3, $8, 'dept_admin'),
         ($1, $4, $9, 'dept_admin'),
         ($2, $5, $10, 'dept_admin')",
    )
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    // Pipeline configs in org1 & org2
    let org1_pc = Uuid::new_v4();
    let org2_pc = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.pipeline_configs (id, org_id, name, profile, pipeline_yaml, config_json, created_by) VALUES
         ($1, $3, 'org1-cfg', 'standard', 'v: 1', '{}'::jsonb, $5),
         ($2, $4, 'org2-cfg', 'standard', 'v: 1', '{}'::jsonb, $6)",
    )
    .bind(org1_pc)
    .bind(org2_pc)
    .bind(org1)
    .bind(org2)
    .bind(dave)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    // Buckets
    let alice_priv_bucket = Uuid::new_v4();
    let bob_bucket = Uuid::new_v4();
    let org2_bucket = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $4, $6, $8,  'alice-priv', 'Alice Priv', 'private', 1, $11),
         ($2, $4, $6, $9,  'bob-bkt',    'Bob Bkt',    'dept',    1, $12),
         ($3, $5, $7, $10, 'org2-bkt',   'Org2 Bkt',   'private', 1, $13)",
    )
    .bind(alice_priv_bucket)
    .bind(bob_bucket)
    .bind(org2_bucket)
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:bucket:L1/{alice_priv_bucket}"))
    .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bob_bucket}"))
    .bind(format!("arn:openrag:org2:org2dept:private:bucket:L1/{org2_bucket}"))
    .execute(admin.inner())
    .await?;

    // Documents & Pages & Chunks
    let alice_priv_doc = Uuid::new_v4();
    let bob_doc = Uuid::new_v4();
    let carol_doc = Uuid::new_v4();
    let org2_doc = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $5, $7, $10, $12, 'private', 3, $15, 'k1', 'Alice Priv', 's1'),
         ($2, $5, $8, $10, $13, 'org',     1, $16, 'k2', 'Bob Doc',    's2'),
         ($3, $5, $8, $10, $12, 'private', 3, $17, 'k3', 'Carol Doc',  's3'),
         ($4, $6, $9, $11, $14, 'private', 3, $18, 'k4', 'Org2 Doc',   's4')",
    )
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(carol_doc)
    .bind(org2_doc)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_bucket)
    .bind(bob_bucket)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:document:L3/{alice_priv_doc}"))
    .bind(format!("arn:openrag:org1:finance:org:document:L1/{bob_doc}"))
    .bind(format!("arn:openrag:org1:finance:private:document:L3/{carol_doc}"))
    .bind(format!("arn:openrag:org2:org2dept:private:document:L3/{org2_doc}"))
    .execute(admin.inner())
    .await?;

    let bob_page = Uuid::new_v4();
    let org2_page = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.pages
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no)
         VALUES
         ($1, $3, $5, $7, $9,  'org',     1, $11, 'p1', 1),
         ($2, $4, $6, $8, $10, 'private', 3, $12, 'p2', 1)",
    )
    .bind(bob_page)
    .bind(org2_page)
    .bind(org1)
    .bind(org2)
    .bind(bob_doc)
    .bind(org2_doc)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:org:page:L1/{bob_page}"))
    .bind(format!("arn:openrag:org2:org2dept:private:page:L3/{org2_page}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_chunk = Uuid::new_v4();
    let bob_chunk = Uuid::new_v4();
    let org2_chunk = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES
         ($1, $4, $6, NULL, $9,  $11, 'private', 3, $14, 'c1', 0, 'alice chunk', 'alice chunk', 'ch1'),
         ($2, $4, $7, $17,  $9,  $12, 'org',     1, $15, 'c2', 0, 'bob chunk',   'bob chunk',   'ch2'),
         ($3, $5, $8, $18,  $10, $13, 'private', 3, $16, 'c3', 0, 'org2 chunk',  'org2 chunk',  'ch3')",
    )
    .bind(alice_priv_chunk)
    .bind(bob_chunk)
    .bind(org2_chunk)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(org2_doc)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:chunk:L3/{alice_priv_chunk}"))
    .bind(format!("arn:openrag:org1:finance:org:chunk:L1/{bob_chunk}"))
    .bind(format!("arn:openrag:org2:org2dept:private:chunk:L3/{org2_chunk}"))
    .bind(bob_page)
    .bind(org2_page)
    .execute(admin.inner())
    .await?;

    // Wiki pages & Memory entries
    let bob_wiki = Uuid::new_v4();
    let org2_wiki = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
         VALUES
         ($1, $3, $5, $7, $9,  'org',     1, $11, 'w1', 'bob-w',  'Bob W',  'pub', 'pub', 'ws1'),
         ($2, $4, $6, $8, $10, 'private', 3, $12, 'w2', 'org2-w', 'Org2 W', 'o2',  'o2',  'ws2')",
    )
    .bind(bob_wiki)
    .bind(org2_wiki)
    .bind(org1)
    .bind(org2)
    .bind(bob_bucket)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:org:wiki:L1/{bob_wiki}"))
    .bind(format!("arn:openrag:org2:org2dept:private:wiki:L3/{org2_wiki}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_mem = Uuid::new_v4();
    let alice_redacted_mem = Uuid::new_v4();
    let bob_mem = Uuid::new_v4();
    let org2_mem = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.entries
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, status)
         VALUES
         ($1, $5, $7, $10, $12, 'private', 3, $15, 'm1', 'fact', 'alice mem', 'alice mem', 'ms1', 'active'),
         ($2, $5, $8, $10, $13, 'org',     1, $16, 'm2', 'fact', 'redacted',  'redacted',  'ms2', 'redacted'),
         ($3, $5, $8, $10, $13, 'org',     1, $17, 'm3', 'fact', 'bob mem',   'bob mem',   'ms3', 'active'),
         ($4, $6, $9, $11, $14, 'private', 3, $18, 'm4', 'fact', 'org2 mem',  'org2 mem',  'ms4', 'active')",
    )
    .bind(alice_priv_mem)
    .bind(alice_redacted_mem)
    .bind(bob_mem)
    .bind(org2_mem)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_bucket)
    .bind(bob_bucket)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:memory:L3/{alice_priv_mem}"))
    .bind(format!("arn:openrag:org1:finance:org:memory:L1/{alice_redacted_mem}"))
    .bind(format!("arn:openrag:org1:finance:org:memory:L1/{bob_mem}"))
    .bind(format!("arn:openrag:org2:org2dept:private:memory:L3/{org2_mem}"))
    .execute(admin.inner())
    .await?;

    // Operational tables for org1 & org2 (plugins, pii, gateway, eval)
    let org1_conn = Uuid::new_v4();
    let org2_conn = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO plugins.connectors (id, org_id, connector_type, name) VALUES
         ($1, $3, 'docufinder_sync', 'org1-conn'),
         ($2, $4, 'docufinder_sync', 'org2-conn')",
    )
    .bind(org1_conn)
    .bind(org2_conn)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let org1_src_entry = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO plugins.source_entries (id, org_id, connector_id, external_key, content_sha256, document_id)
         VALUES ($1, $2, $3, 'ext-1', 'sha-ext-1', $4)",
    )
    .bind(org1_src_entry)
    .bind(org1)
    .bind(org1_conn)
    .bind(bob_doc)
    .execute(admin.inner())
    .await?;

    let org1_vk = Uuid::new_v4();
    let org2_vk = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.vault_keys (id, org_id, key_version, wrapped_dek) VALUES
         ($1, $3, 1, '\\x01'),
         ($2, $4, 1, '\\x02')",
    )
    .bind(org1_vk)
    .bind(org2_vk)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let org2_ve = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.vault_entries (id, org_id, vault_key_id, token, entity_type, ciphertext, nonce, detector_source)
         VALUES ($1, $2, $3, 'tok-org2', 'PHONE', '\\x01', '\\x02', 'schift_v7')",
    )
    .bind(org2_ve)
    .bind(org2)
    .bind(org2_vk)
    .execute(admin.inner())
    .await?;

    let org1_prov = Uuid::new_v4();
    let org2_prov = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO gateway.providers (id, org_id, name, provider_type, base_url, model_id) VALUES
         ($1, $3, 'org1-p', 'mock', 'http://localhost:1', 'm1'),
         ($2, $4, 'org2-p', 'mock', 'http://localhost:2', 'm2')",
    )
    .bind(org1_prov)
    .bind(org2_prov)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let org1_gs = Uuid::new_v4();
    let org2_gs = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.golden_sets (id, org_id, name, axis) VALUES
         ($1, $3, 'org1-gs', 'retrieval'),
         ($2, $4, 'org2-gs', 'retrieval')",
    )
    .bind(org1_gs)
    .bind(org2_gs)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let org2_gi = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.golden_items (id, org_id, golden_set_id, axis, query_text)
         VALUES ($1, $2, $3, 'retrieval', 'org2 query')",
    )
    .bind(org2_gi)
    .bind(org2)
    .bind(org2_gs)
    .execute(admin.inner())
    .await?;

    let org2_trial = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.optimization_trials (id, org_id, golden_set_id, search_space_yaml, strategy)
         VALUES ($1, $2, $3, 'k: 10', 'greedy_node')",
    )
    .bind(org2_trial)
    .bind(org2)
    .bind(org2_gs)
    .execute(admin.inner())
    .await?;

    let mut bob_ctx = PrincipalContext::new(org1, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![fin_dept];

    let mut carol_ctx = PrincipalContext::new(org1, carol, AuthLevel::new(7)?);
    carol_ctx.dept_ids = vec![fin_dept];
    carol_ctx.dept_admin_ids = vec![fin_dept];

    let mut dave_ctx = PrincipalContext::new(org1, dave, AuthLevel::new(9)?);
    dave_ctx.dept_ids = vec![hr_dept];
    dave_ctx.dept_admin_ids = vec![hr_dept];
    dave_ctx.is_org_admin = true;

    let nonexistent_uuid = Uuid::new_v4();

    // -------------------------------------------------------------------------
    // 1. NEW-HIGH-2: corpus.buckets.pipeline_config_id Cross-Org FK & Oracle
    // -------------------------------------------------------------------------
    for (label, target_pc) in [
        ("Bob INSERT bucket pipeline_config_id -> org2_pc", org2_pc),
        ("Bob INSERT bucket pipeline_config_id -> nonexistent", nonexistent_uuid),
    ] {
        let bid = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn, pipeline_config_id)
             VALUES ($1, $2, $3, $4, $5, 'Bad PC', 'dept', 1, $6, $7)",
        )
        .bind(bid)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("bad-pc-{}", bid.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bid}"))
        .bind(target_pc)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }

    for (label, target_pc) in [
        ("Bob UPDATE bucket pipeline_config_id -> org2_pc", org2_pc),
        ("Bob UPDATE bucket pipeline_config_id -> nonexistent", nonexistent_uuid),
    ] {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query("UPDATE corpus.buckets SET pipeline_config_id = $1 WHERE id = $2")
            .bind(target_pc)
            .bind(bob_bucket)
            .execute(&mut **tx.as_mut())
            .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }

    {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query("UPDATE corpus.buckets SET pipeline_config_id = $1 WHERE id = $2")
            .bind(org2_pc)
            .bind(bob_bucket)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r, &["23503"], "worker cross-org corpus.buckets.pipeline_config_id")?;
        tx.rollback().await?;
    }

    // Positive: Bob links org1_pc; deleting org1_pc sets pipeline_config_id = NULL cleanly
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query("UPDATE corpus.buckets SET pipeline_config_id = $1 WHERE id = $2")
            .bind(org1_pc)
            .bind(bob_bucket)
            .execute(&mut **tx.as_mut())
            .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.commit().await?;

        let mut tx_del = admin.inner().begin().await?;
        sqlx::query("DELETE FROM eval.pipeline_configs WHERE id = $1")
            .bind(org1_pc)
            .execute(&mut *tx_del)
            .await?;
        let pc_after: Option<Option<Uuid>> =
            sqlx::query_scalar("SELECT pipeline_config_id FROM corpus.buckets WHERE id = $1")
                .bind(bob_bucket)
                .fetch_optional(&mut *tx_del)
                .await?;
        assert_eq!(pc_after, Some(None));
        tx_del.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 2. NEW-HIGH-3: Cross-Org owner_principal_id & dept_id Isolation
    // -------------------------------------------------------------------------
    // Carol (dept_admin) INSERT with eve (org2) or nonexistent owner -> 42501
    for (label, bad_owner) in [
        ("Carol INSERT bucket owner = eve", eve),
        ("Carol INSERT bucket owner = nonexistent", nonexistent_uuid),
    ] {
        let bid = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'Bad Owner', 'dept', 1, $6)",
        )
        .bind(bid)
        .bind(org1)
        .bind(fin_dept)
        .bind(bad_owner)
        .bind(format!("bad-own-{}", bid.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bid}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }

    {
        let did = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
             VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'k', 'Bad Owner Doc', 'sh')",
        )
        .bind(did)
        .bind(org1)
        .bind(bob_bucket)
        .bind(fin_dept)
        .bind(eve)
        .bind(format!("arn:openrag:org1:finance:dept:document:L1/{did}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "Carol INSERT doc owner = eve")?;
        tx.rollback().await?;
    }

    // Carol UPDATE owner_principal_id -> eve / nonexistent, or dept_id -> org2_dept -> 42501
    for (label, bad_owner) in [
        ("Carol UPDATE doc owner = eve", eve),
        ("Carol UPDATE doc owner = nonexistent", nonexistent_uuid),
    ] {
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let r = sqlx::query("UPDATE corpus.documents SET owner_principal_id = $1 WHERE id = $2")
            .bind(bad_owner)
            .bind(carol_doc)
            .execute(&mut **tx.as_mut())
            .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }
    {
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let r = sqlx::query("UPDATE corpus.documents SET dept_id = $1 WHERE id = $2")
            .bind(org2_dept)
            .bind(carol_doc)
            .execute(&mut **tx.as_mut())
            .await;
        assert_pg_err_code(r, &["42501"], "Carol UPDATE doc dept = org2_dept")?;
        tx.rollback().await?;
    }

    // Dave (is_org_admin) INSERT with org2 dept_id / eve / nonexistent -> 42501
    for (label, bad_dept, bad_owner) in [
        ("Dave INSERT bucket dept=org2_dept, owner=eve", org2_dept, eve),
        ("Dave INSERT bucket dept=fin_dept, owner=eve", fin_dept, eve),
        ("Dave INSERT bucket dept=nonexistent, owner=dave", nonexistent_uuid, dave),
    ] {
        let bid = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&dave_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'Bad OrgAdmin', 'dept', 1, $6)",
        )
        .bind(bid)
        .bind(org1)
        .bind(bad_dept)
        .bind(bad_owner)
        .bind(format!("bad-dave-{}", bid.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bid}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }

    // Positive: Carol (dept_admin) reassigns carol_doc owner -> bob (active in org1)
    {
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let r = sqlx::query("UPDATE corpus.documents SET owner_principal_id = $1 WHERE id = $2")
            .bind(bob)
            .bind(carol_doc)
            .execute(&mut **tx.as_mut())
            .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.rollback().await?;
    }

    // BYPASSRLS openrag_worker blocked with 23503 on cross-org dept_id / owner_principal_id
    for (label, sql_str, target_uuid, row_id) in [
        ("worker buckets.dept_id -> org2_dept", "UPDATE corpus.buckets SET dept_id = $1 WHERE id = $2", org2_dept, bob_bucket),
        ("worker buckets.owner -> eve", "UPDATE corpus.buckets SET owner_principal_id = $1 WHERE id = $2", eve, bob_bucket),
        ("worker documents.owner -> eve", "UPDATE corpus.documents SET owner_principal_id = $1 WHERE id = $2", eve, bob_doc),
        ("worker wiki.pages.owner -> eve", "UPDATE wiki.pages SET owner_principal_id = $1 WHERE id = $2", eve, bob_wiki),
        ("worker memory.entries.owner -> eve", "UPDATE memory.entries SET owner_principal_id = $1 WHERE id = $2", eve, bob_mem),
    ] {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(sql_str)
            .bind(target_uuid)
            .bind(row_id)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r, &["23503"], label)?;
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 3. NEW-MEDIUM-1: plugins.source_entries.document_id Composite FK
    // -------------------------------------------------------------------------
    {
        let mut tx = worker.inner().begin().await?;
        let r_upd = sqlx::query("UPDATE plugins.source_entries SET document_id = $1 WHERE id = $2")
            .bind(org2_doc)
            .bind(org1_src_entry)
            .execute(&mut *tx)
            .await;
        assert_pg_err_code(r_upd, &["23503"], "worker UPDATE plugins.source_entries.document_id -> org2_doc")?;
        tx.rollback().await?;

        let mut tx_ins = worker.inner().begin().await?;
        let r_ins = sqlx::query(
            "INSERT INTO plugins.source_entries (org_id, connector_id, external_key, content_sha256, document_id)
             VALUES ($1, $2, 'ext-bad', 'sha-bad', $3)",
        )
        .bind(org1)
        .bind(org1_conn)
        .bind(org2_doc)
        .execute(&mut *tx_ins)
        .await;
        assert_pg_err_code(r_ins, &["23503"], "worker INSERT plugins.source_entries.document_id -> org2_doc")?;
        tx_ins.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 4. NEW-LOW-1: wiki.page_sources.source_id Cross-Org / Private / Oracle RLS
    // -------------------------------------------------------------------------
    for (label, src_type, bad_src_id) in [
        ("Bob INSERT page_sources doc -> org2_doc", "document", org2_doc),
        ("Bob INSERT page_sources doc -> alice_priv_doc", "document", alice_priv_doc),
        ("Bob INSERT page_sources doc -> nonexistent", "document", nonexistent_uuid),
        ("Bob INSERT page_sources chunk -> org2_chunk", "chunk", org2_chunk),
        ("Bob INSERT page_sources chunk -> alice_priv_chunk", "chunk", alice_priv_chunk),
        ("Bob INSERT page_sources memory -> org2_mem", "memory", org2_mem),
        ("Bob INSERT page_sources memory -> alice_priv_mem", "memory", alice_priv_mem),
        ("Bob INSERT page_sources memory -> alice_redacted_mem", "memory", alice_redacted_mem),
    ] {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO wiki.page_sources (org_id, page_id, source_type, source_id) VALUES ($1, $2, $3, $4)",
        )
        .bind(org1)
        .bind(bob_wiki)
        .bind(src_type)
        .bind(bad_src_id)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }

    // Positive + UPDATE rejection for wiki.page_sources
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        for (src_type, ok_src_id) in [
            ("document", bob_doc),
            ("chunk", bob_chunk),
            ("memory", bob_mem),
        ] {
            let r = sqlx::query(
                "INSERT INTO wiki.page_sources (org_id, page_id, source_type, source_id) VALUES ($1, $2, $3, $4)",
            )
            .bind(org1)
            .bind(bob_wiki)
            .bind(src_type)
            .bind(ok_src_id)
            .execute(&mut **tx.as_mut())
            .await?;
            assert_eq!(r.rows_affected(), 1);
        }
        tx.commit().await?;

        let mut tx_u = app.begin_rls_tx(&bob_ctx).await?;
        let r_u = sqlx::query(
            "UPDATE wiki.page_sources SET source_id = $1 WHERE page_id = $2 AND source_type = 'document' AND source_id = $3",
        )
        .bind(alice_priv_doc)
        .bind(bob_wiki)
        .bind(bob_doc)
        .execute(&mut **tx_u.as_mut())
        .await;
        assert_pg_err_code(r_u, &["42501"], "Bob UPDATE page_sources -> alice_priv_doc")?;
        tx_u.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 5. Operational Tables Composite FKs (003, 005, 006) as BYPASSRLS worker
    // -------------------------------------------------------------------------
    {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(
            "INSERT INTO pii.chunk_spans (org_id, chunk_id, page_id, vault_entry_id, entity_type, start_offset, end_offset)
             VALUES ($1, $2, $3, $4, 'PHONE', 0, 4)",
        )
        .bind(org1)
        .bind(org2_chunk)
        .bind(bob_page)
        .bind(org2_ve)
        .execute(&mut *tx)
        .await;
        assert_pg_err_code(r, &["23503"], "worker cross-org pii.chunk_spans")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(
            "INSERT INTO pii.guard_policies (org_id, bucket_id, name) VALUES ($1, $2, 'bad-gp')",
        )
        .bind(org1)
        .bind(org2_bucket)
        .execute(&mut *tx)
        .await;
        assert_pg_err_code(r, &["23503"], "worker cross-org pii.guard_policies.bucket_id")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(
            "INSERT INTO gateway.routing_rules (org_id, target_provider_id, fallback_provider_id) VALUES ($1, $2, $3)",
        )
        .bind(org1)
        .bind(org1_prov)
        .bind(org2_prov)
        .execute(&mut *tx)
        .await;
        assert_pg_err_code(r, &["23503"], "worker cross-org gateway.routing_rules.fallback_provider_id")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(
            "INSERT INTO eval.bench_runs (org_id, golden_set_id, pipeline_config_id, trial_id, track)
             VALUES ($1, $2, $3, $4, 'track1_retrieval')",
        )
        .bind(org1)
        .bind(org1_gs)
        .bind(org2_pc)
        .bind(org2_trial)
        .execute(&mut *tx)
        .await;
        assert_pg_err_code(r, &["23503"], "worker cross-org eval.bench_runs")?;
        tx.rollback().await?;
    }
    {
        let mut tx = worker.inner().begin().await?;
        let r = sqlx::query(
            "INSERT INTO eval.search_feedback (org_id, principal_id, query_text, verdict, promoted_golden_item_id)
             VALUES ($1, $2, 'q', 'up', $3)",
        )
        .bind(org1)
        .bind(bob)
        .bind(org2_gi)
        .execute(&mut *tx)
        .await;
        assert_pg_err_code(r, &["23503"], "worker cross-org eval.search_feedback")?;
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 6. Verify DELETE FROM iam.orgs WHERE id = org2 cascades cleanly!
    // -------------------------------------------------------------------------
    {
        let mut tx = admin.inner().begin().await?;
        let del = sqlx::query("DELETE FROM iam.orgs WHERE id = $1")
            .bind(org2)
            .execute(&mut *tx)
            .await?;
        assert_eq!(del.rows_affected(), 1);
        tx.commit().await?;
    }

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_rls_iteration5_operational_rls_and_insert_api_key_regressions(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let app_url = env_url("DATABASE_URL")?;

    let admin = DbPool::connect(&admin_url, 4).await?;
    let app = DbPool::connect(&app_url, 4).await?;

    let org1 = Uuid::new_v4();
    let org2 = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let hr_dept = Uuid::new_v4();
    let org2_dept = Uuid::new_v4();

    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let dave = Uuid::new_v4();
    let eve = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'I5 Org1'), ($3, $4, 'I5 Org2')")
        .bind(org1)
        .bind(format!("i5-org1-{}", org1.simple()))
        .bind(org2)
        .bind(format!("i5-org2-{}", org2.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $4, 'Finance', 'finance'),
         ($2, $4, 'HR', 'hr'),
         ($3, $5, 'Org2Dept', 'org2dept')",
    )
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $6, $8,  'Alice', 8, true,  false),
         ($2, $6, $9,  'Bob',   9, false, false),
         ($3, $6, $10, 'Carol', 7, true,  false),
         ($4, $6, $11, 'Dave',  9, true,  true),
         ($5, $7, $12, 'Eve',   9, true,  true)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(eve)
    .bind(org1)
    .bind(org2)
    .bind(format!("alice-i5-{}@org1.test", org1.simple()))
    .bind(format!("bob-i5-{}@org1.test", org1.simple()))
    .bind(format!("carol-i5-{}@org1.test", org1.simple()))
    .bind(format!("dave-i5-{}@org1.test", org1.simple()))
    .bind(format!("eve-i5-{}@org2.test", org2.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $3, $6, 'member'),
         ($1, $3, $7, 'member'),
         ($1, $3, $8, 'dept_admin'),
         ($1, $4, $9, 'dept_admin'),
         ($2, $5, $10, 'dept_admin')",
    )
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    let g1 = Uuid::new_v4();
    let g2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, group_type) VALUES
         ($1, $3, 'org1-grp', 'permission_group'),
         ($2, $4, 'org2-grp', 'permission_group')",
    )
    .bind(g1)
    .bind(g2)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
         ($1, $3, $5),
         ($1, $3, $6),
         ($2, $4, $7)",
    )
    .bind(org1)
    .bind(org2)
    .bind(g1)
    .bind(g2)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    // Corpus, Wiki, Memory fixtures for outbox & insert tests
    let alice_priv_bucket = Uuid::new_v4();
    let bob_bucket = Uuid::new_v4();
    let org2_bucket = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $4, $6, $8,  'i5-alice-priv', 'Alice Priv', 'private', 1, $11),
         ($2, $4, $6, $9,  'i5-bob-bkt',    'Bob Bkt',    'dept',    1, $12),
         ($3, $5, $7, $10, 'i5-org2-bkt',   'Org2 Bkt',   'private', 1, $13)",
    )
    .bind(alice_priv_bucket)
    .bind(bob_bucket)
    .bind(org2_bucket)
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:bucket:L1/{alice_priv_bucket}"))
    .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{bob_bucket}"))
    .bind(format!("arn:openrag:org2:org2dept:private:bucket:L1/{org2_bucket}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_doc = Uuid::new_v4();
    let bob_doc = Uuid::new_v4();
    let org2_doc = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $4, $6, $9,  $11, 'private', 3, $14, 'k1', 'Alice Priv', 's1'),
         ($2, $4, $7, $9,  $12, 'org',     1, $15, 'k2', 'Bob Doc',    's2'),
         ($3, $5, $8, $10, $13, 'private', 3, $16, 'k3', 'Org2 Doc',   's3')",
    )
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(org2_doc)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_bucket)
    .bind(bob_bucket)
    .bind(org2_bucket)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:private:document:L3/{alice_priv_doc}"))
    .bind(format!("arn:openrag:org1:finance:org:document:L1/{bob_doc}"))
    .bind(format!("arn:openrag:org2:org2dept:private:document:L3/{org2_doc}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_page = Uuid::new_v4();
    let bob_page = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.pages
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no)
         VALUES
         ($1, $3, $4, $6, $7, 'private', 3, $9,  'p1', 1),
         ($2, $3, $5, $6, $8, 'org',     1, $10, 'p2', 1)",
    )
    .bind(alice_priv_page)
    .bind(bob_page)
    .bind(org1)
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(fin_dept)
    .bind(alice)
    .bind(bob)
    .bind(format!("arn:openrag:org1:finance:private:page:L3/{alice_priv_page}"))
    .bind(format!("arn:openrag:org1:finance:org:page:L1/{bob_page}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_fig = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.page_figures
         (id, org_id, page_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, figure_idx)
         VALUES ($1, $2, $3, $4, $5, $6, 'private', 3, $7, 'f1', 0)",
    )
    .bind(alice_priv_fig)
    .bind(org1)
    .bind(alice_priv_page)
    .bind(alice_priv_doc)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:org1:finance:private:figure:L3/{alice_priv_fig}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_chunk = Uuid::new_v4();
    let bob_chunk = Uuid::new_v4();
    let org2_chunk = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES
         ($1, $4, $6, $16,  $9,  $11, 'private', 3, $13, 'c1', 0, 'alice priv', 'alice priv', 'th1'),
         ($2, $4, $7, $17,  $9,  $12, 'org',     1, $14, 'c2', 0, 'bob org',    'bob org',    'th2'),
         ($3, $5, $8, NULL, $10, $18, 'private', 3, $15, 'c3', 0, 'org2 priv',  'org2 priv',  'th3')",
    )
    .bind(alice_priv_chunk)
    .bind(bob_chunk)
    .bind(org2_chunk)
    .bind(org1)
    .bind(org2)
    .bind(alice_priv_doc)
    .bind(bob_doc)
    .bind(org2_doc)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(format!("arn:openrag:org1:finance:private:chunk:L3/{alice_priv_chunk}"))
    .bind(format!("arn:openrag:org1:finance:org:chunk:L1/{bob_chunk}"))
    .bind(format!("arn:openrag:org2:org2dept:private:chunk:L3/{org2_chunk}"))
    .bind(alice_priv_page)
    .bind(bob_page)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    let alice_priv_wiki = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.pages
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
         VALUES ($1, $2, $3, $4, $5, 'private', 3, $6, 'w1', 'i5-alice-w', 'Alice W', 'sec', 'sec', 'ws1')",
    )
    .bind(alice_priv_wiki)
    .bind(org1)
    .bind(alice_priv_bucket)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:org1:finance:private:wiki:L3/{alice_priv_wiki}"))
    .execute(admin.inner())
    .await?;

    let alice_priv_mem = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO memory.entries
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
         VALUES ($1, $2, $3, $4, $5, 'private', 3, $6, 'm1', 'fact', 'alice priv mem', 'alice priv mem', 'ms1')",
    )
    .bind(alice_priv_mem)
    .bind(org1)
    .bind(alice_priv_bucket)
    .bind(fin_dept)
    .bind(alice)
    .bind(format!("arn:openrag:org1:finance:private:memory:L3/{alice_priv_mem}"))
    .execute(admin.inner())
    .await?;

    // PII Vault, Guard Policies, Golden Items
    let org1_vk = Uuid::new_v4();
    let org2_vk = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.vault_keys (id, org_id, key_version, wrapped_dek) VALUES
         ($1, $3, 1, '\\x01'),
         ($2, $4, 1, '\\x02')",
    )
    .bind(org1_vk)
    .bind(org2_vk)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let ve_org1 = Uuid::new_v4();
    let ve_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.vault_entries (id, org_id, vault_key_id, token, entity_type, ciphertext, nonce, detector_source) VALUES
         ($1, $3, $5, 'tok-org1', 'PHONE', '\\x01', '\\x02', 'schift_v7'),
         ($2, $4, $6, 'tok-org2', 'PHONE', '\\x01', '\\x02', 'schift_v7')",
    )
    .bind(ve_org1)
    .bind(ve_org2)
    .bind(org1)
    .bind(org2)
    .bind(org1_vk)
    .bind(org2_vk)
    .execute(admin.inner())
    .await?;

    let gp_org1 = Uuid::new_v4();
    let gp_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.guard_policies (id, org_id, name) VALUES
         ($1, $3, 'i5-gp-org1'),
         ($2, $4, 'i5-gp-org2')",
    )
    .bind(gp_org1)
    .bind(gp_org2)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let gs_org1 = Uuid::new_v4();
    let gs_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.golden_sets (id, org_id, name, axis) VALUES
         ($1, $3, 'i5-gs-org1', 'retrieval'),
         ($2, $4, 'i5-gs-org2', 'retrieval')",
    )
    .bind(gs_org1)
    .bind(gs_org2)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    let gi_org1 = Uuid::new_v4();
    let gi_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.golden_items (id, org_id, golden_set_id, axis, query_text) VALUES
         ($1, $3, $5, 'retrieval', 'org1 q'),
         ($2, $4, $6, 'retrieval', 'org2 q')",
    )
    .bind(gi_org1)
    .bind(gi_org2)
    .bind(org1)
    .bind(org2)
    .bind(gs_org1)
    .bind(gs_org2)
    .execute(admin.inner())
    .await?;

    // Seed rows in pii.audit_logs, pii.guard_events, eval.search_feedback, indexing.outbox across org1 & org2
    let audit_alice = Uuid::new_v4();
    let audit_bob = Uuid::new_v4();
    let audit_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.audit_logs (id, org_id, principal_id, action, allowed, reason) VALUES
         ($1, $4, $6, 'unmask', true, 'alice-secret-audit'),
         ($2, $4, $7, 'unmask', false, 'bob-audit'),
         ($3, $5, $8, 'unmask', true, 'org2-secret-audit')",
    )
    .bind(audit_alice)
    .bind(audit_bob)
    .bind(audit_org2)
    .bind(org1)
    .bind(org2)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    let ge_alice = Uuid::new_v4();
    let ge_bob = Uuid::new_v4();
    let ge_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO pii.guard_events (id, org_id, policy_id, principal_id, surface, decision) VALUES
         ($1, $4, $6, $8,  'inbound_query', 'mask'),
         ($2, $4, $6, $9,  'inbound_query', 'allow'),
         ($3, $5, $7, $10, 'inbound_query', 'block')",
    )
    .bind(ge_alice)
    .bind(ge_bob)
    .bind(ge_org2)
    .bind(org1)
    .bind(org2)
    .bind(gp_org1)
    .bind(gp_org2)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    let fb_alice = Uuid::new_v4();
    let fb_bob = Uuid::new_v4();
    let fb_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO eval.search_feedback (id, org_id, principal_id, query_text, verdict) VALUES
         ($1, $4, $6, 'alice secret q', 'up'),
         ($2, $4, $7, 'bob q', 'up'),
         ($3, $5, $8, 'org2 confidential q', 'down')",
    )
    .bind(fb_alice)
    .bind(fb_bob)
    .bind(fb_org2)
    .bind(org1)
    .bind(org2)
    .bind(alice)
    .bind(bob)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    let outbox_alice_id: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"secret\":\"alice\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(alice_priv_chunk)
    .fetch_one(admin.inner())
    .await?;
    let outbox_bob_id: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"pub\":\"bob\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(bob_chunk)
    .fetch_one(admin.inner())
    .await?;
    let outbox_org2_id: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"secret\":\"org2\"}'::jsonb) RETURNING id",
    )
    .bind(org2)
    .bind(org2_chunk)
    .fetch_one(admin.inner())
    .await?;

    let mut bob_ctx = PrincipalContext::new(org1, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![fin_dept];

    let mut dave_ctx = PrincipalContext::new(org1, dave, AuthLevel::new(9)?);
    dave_ctx.dept_ids = vec![hr_dept];
    dave_ctx.dept_admin_ids = vec![hr_dept];
    dave_ctx.is_org_admin = true;

    // -------------------------------------------------------------------------
    // 1. NEW-HIGH-4: Fail-Closed SELECT when GUCs unset & Cross-Tenant Isolation
    // -------------------------------------------------------------------------
    for table in [
        "iam.orgs",
        "iam.departments",
        "iam.groups",
        "iam.group_members",
        "pii.audit_logs",
        "pii.guard_events",
        "eval.search_feedback",
        "indexing.outbox",
    ] {
        let cnt: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
            .fetch_one(app.inner())
            .await?;
        assert_eq!(cnt, 0, "unset GUCs must yield 0 rows for {table}");
    }

    // Bob (non-admin in org1) sees ONLY org1 IAM directory (and ONLY his own group_members),
    // ONLY his own audit_logs/guard_events/search_feedback, and ONLY accessible indexing.outbox!
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let seen_orgs: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM iam.orgs")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(seen_orgs.contains(&org1));
        assert!(!seen_orgs.contains(&org2));

        let seen_depts: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM iam.departments")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(seen_depts.contains(&fin_dept));
        assert!(!seen_depts.contains(&org2_dept));

        let seen_groups: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM iam.groups")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(seen_groups.contains(&g1));
        assert!(!seen_groups.contains(&g2));

        let seen_gm: Vec<Uuid> = sqlx::query_scalar("SELECT principal_id FROM iam.group_members")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert_eq!(seen_gm, vec![bob]);

        let seen_audits: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM pii.audit_logs")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert_eq!(seen_audits, vec![audit_bob]);

        let seen_ge: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM pii.guard_events")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert_eq!(seen_ge, vec![ge_bob]);

        let seen_fb: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM eval.search_feedback")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert_eq!(seen_fb, vec![fb_bob]);

        let seen_outbox: Vec<i64> = sqlx::query_scalar("SELECT id FROM indexing.outbox")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(seen_outbox.contains(&outbox_bob_id));
        assert!(!seen_outbox.contains(&outbox_alice_id));
        assert!(!seen_outbox.contains(&outbox_org2_id));
        tx.rollback().await?;
    }

    // Dave (org1 is_org_admin) sees all org1 rows on operational & group_members tables, ZERO org2 rows
    {
        let mut tx = app.begin_rls_tx(&dave_ctx).await?;
        let gm: Vec<Uuid> = sqlx::query_scalar("SELECT principal_id FROM iam.group_members")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(gm.contains(&alice) && gm.contains(&bob) && !gm.contains(&eve));

        let audits: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM pii.audit_logs")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(audits.contains(&audit_alice) && audits.contains(&audit_bob) && !audits.contains(&audit_org2));

        let ge: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM pii.guard_events")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(ge.contains(&ge_alice) && ge.contains(&ge_bob) && !ge.contains(&ge_org2));

        let fb: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM eval.search_feedback")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(fb.contains(&fb_alice) && fb.contains(&fb_bob) && !fb.contains(&fb_org2));

        let ob: Vec<i64> = sqlx::query_scalar("SELECT id FROM indexing.outbox")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        // NEW-LOW-2: HR org_admin Dave sees org-scoped Bob outbox row, NOT Finance private Alice outbox row or org2 row
        assert!(
            !ob.contains(&outbox_alice_id)
                && ob.contains(&outbox_bob_id)
                && !ob.contains(&outbox_org2_id)
        );
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 2. NEW-HIGH-4: Cross-Tenant / Spoofed / 23503-Oracle INSERT -> uniform 42501
    // -------------------------------------------------------------------------
    let nonexistent = Uuid::new_v4();

    // pii.audit_logs
    for (label, target_org, target_princ, target_ve) in [
        ("audit_logs org2 + ve_org2", org2, Some(eve), Some(ve_org2)),
        ("audit_logs org2 + nonexistent ve", org2, Some(eve), Some(nonexistent)),
        ("audit_logs org1 + ve_org2", org1, Some(bob), Some(ve_org2)),
        ("audit_logs org1 + nonexistent ve", org1, Some(bob), Some(nonexistent)),
        ("audit_logs spoof principal alice", org1, Some(alice), None),
        ("audit_logs null principal", org1, None, None),
    ] {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO pii.audit_logs (org_id, principal_id, action, vault_entry_id, allowed)
             VALUES ($1, $2, 'spoof', $3, true)",
        )
        .bind(target_org)
        .bind(target_princ)
        .bind(target_ve)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }
    // Positive Bob pii.audit_logs insert
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO pii.audit_logs (org_id, principal_id, action, vault_entry_id, allowed)
             VALUES ($1, $2, 'read', $3, true)",
        )
        .bind(org1)
        .bind(bob)
        .bind(ve_org1)
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.rollback().await?;
    }

    // pii.guard_events
    for (label, target_org, target_princ, target_gp) in [
        ("guard_events org2 + gp_org2", org2, Some(eve), Some(gp_org2)),
        ("guard_events org2 + nonexistent gp", org2, Some(eve), Some(nonexistent)),
        ("guard_events org1 + gp_org2", org1, Some(bob), Some(gp_org2)),
        ("guard_events org1 + nonexistent gp", org1, Some(bob), Some(nonexistent)),
        ("guard_events spoof principal alice", org1, Some(alice), None),
        ("guard_events null principal", org1, None, None),
    ] {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO pii.guard_events (org_id, policy_id, principal_id, surface, decision)
             VALUES ($1, $2, $3, 'inbound_query', 'block')",
        )
        .bind(target_org)
        .bind(target_gp)
        .bind(target_princ)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }
    // Positive Bob pii.guard_events insert
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO pii.guard_events (org_id, policy_id, principal_id, surface, decision)
             VALUES ($1, $2, $3, 'inbound_query', 'allow')",
        )
        .bind(org1)
        .bind(gp_org1)
        .bind(bob)
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.rollback().await?;
    }

    // eval.search_feedback
    for (label, target_org, target_princ, target_gi) in [
        ("search_feedback org2 + gi_org2", org2, eve, Some(gi_org2)),
        ("search_feedback org2 + nonexistent gi", org2, eve, Some(nonexistent)),
        ("search_feedback non-admin promote gi_org1", org1, bob, Some(gi_org1)),
        ("search_feedback spoof principal alice", org1, alice, None),
    ] {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO eval.search_feedback (org_id, principal_id, query_text, verdict, promoted_golden_item_id)
             VALUES ($1, $2, 'q', 'up', $3)",
        )
        .bind(target_org)
        .bind(target_princ)
        .bind(target_gi)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }
    // Positive Bob feedback (no promotion) & Dave (is_org_admin) promotion in org1, blocked on gi_org2/nonexistent
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO eval.search_feedback (org_id, principal_id, query_text, verdict)
             VALUES ($1, $2, 'bob valid q', 'up')",
        )
        .bind(org1)
        .bind(bob)
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.rollback().await?;

        let mut tx_d = app.begin_rls_tx(&dave_ctx).await?;
        let r_d = sqlx::query(
            "INSERT INTO eval.search_feedback (org_id, principal_id, query_text, verdict, promoted_golden_item_id)
             VALUES ($1, $2, 'dave valid promote', 'up', $3)",
        )
        .bind(org1)
        .bind(dave)
        .bind(gi_org1)
        .execute(&mut **tx_d.as_mut())
        .await?;
        assert_eq!(r_d.rows_affected(), 1);
        tx_d.rollback().await?;

        for (label, bad_gi) in [
            ("Dave search_feedback gi_org2", gi_org2),
            ("Dave search_feedback nonexistent gi", nonexistent),
        ] {
            let mut tx_bad = app.begin_rls_tx(&dave_ctx).await?;
            let r_bad = sqlx::query(
                "INSERT INTO eval.search_feedback (org_id, principal_id, query_text, verdict, promoted_golden_item_id)
                 VALUES ($1, $2, 'dave bad', 'up', $3)",
            )
            .bind(org1)
            .bind(dave)
            .bind(bad_gi)
            .execute(&mut **tx_bad.as_mut())
            .await;
            assert_pg_err_code(r_bad, &["42501"], label)?;
            tx_bad.rollback().await?;
        }
    }

    // indexing.outbox
    for (label, target_org, target_table, target_id) in [
        ("outbox org2 chunk", org2, "corpus.chunks", org2_chunk),
        ("outbox org1 with org2_chunk", org1, "corpus.chunks", org2_chunk),
        ("outbox org1 with alice_priv_chunk", org1, "corpus.chunks", alice_priv_chunk),
        ("outbox org1 with nonexistent", org1, "corpus.chunks", nonexistent),
        ("outbox org1 with alice_priv_page", org1, "corpus.pages", alice_priv_page),
        ("outbox org1 with alice_priv_fig", org1, "corpus.page_figures", alice_priv_fig),
        ("outbox org1 with alice_priv_wiki", org1, "wiki.pages", alice_priv_wiki),
        ("outbox org1 with alice_priv_mem", org1, "memory.entries", alice_priv_mem),
    ] {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op)
             VALUES ($1, $2, $3, 'delete_vector')",
        )
        .bind(target_org)
        .bind(target_table)
        .bind(target_id)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }
    // Positive Bob indexing.outbox insert on Bob's writable chunk
    {
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op)
             VALUES ($1, 'corpus.chunks', $2, 'upsert_embed')",
        )
        .bind(org1)
        .bind(bob_chunk)
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r.rows_affected(), 1);
        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // 3. NEW-MEDIUM-2: iam.rls_can_insert API Key & scoped_arns Enforcement
    // -------------------------------------------------------------------------
    let revoked_ak = Uuid::new_v4();
    let expired_ak = Uuid::new_v4();
    let scoped_ak = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.api_keys (id, org_id, principal_id, name, key_prefix, key_hash, scoped_arns, max_auth_level, expires_at, revoked_at) VALUES
         ($1, $4, $5, 'bob-revoked', 'orag', $6, ARRAY[]::TEXT[], 9, NULL, now() - interval '1 minute'),
         ($2, $4, $5, 'bob-expired', 'orag', $7, ARRAY[]::TEXT[], 9, now() - interval '1 minute', NULL),
         ($3, $4, $5, 'bob-scoped',  'orag', $8, ARRAY['arn:openrag:org1:finance:*'], 5, now() + interval '1 hour', NULL)",
    )
    .bind(revoked_ak)
    .bind(expired_ak)
    .bind(scoped_ak)
    .bind(org1)
    .bind(bob)
    .bind(format!("hash_rev_{}", revoked_ak.simple()))
    .bind(format!("hash_exp_{}", expired_ak.simple()))
    .bind(format!("hash_scp_{}", scoped_ak.simple()))
    .execute(admin.inner())
    .await?;

    let mut bob_revoked_ctx = bob_ctx.clone();
    bob_revoked_ctx.api_key_id = Some(revoked_ak);

    let mut bob_expired_ctx = bob_ctx.clone();
    bob_expired_ctx.api_key_id = Some(expired_ak);

    let mut bob_scoped_ctx = bob_ctx.clone();
    bob_scoped_ctx.api_key_id = Some(scoped_ak);
    bob_scoped_ctx.scoped_arns = vec!["arn:openrag:org1:finance:*".to_string()];

    // Helper closure-style checks for all 4 INSERT tables: buckets, wiki.pages (bucket NULL), memory.episodes, memory.entries (bucket NULL)
    for (ctx_label, ctx, dept_arn_seg, level_seg) in [
        ("revoked_ak", &bob_revoked_ctx, "finance", "L3"),
        ("expired_ak", &bob_expired_ctx, "finance", "L3"),
        ("scoped_ak out-of-scope hr", &bob_scoped_ctx, "hr", "L3"),
        ("scoped_ak exceeding max_auth_level L8", &bob_scoped_ctx, "finance", "L8"),
    ] {
        // 1) corpus.buckets
        {
            let id = Uuid::new_v4();
            let mut tx = app.begin_rls_tx(ctx).await?;
            let r = sqlx::query(
                "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
                 VALUES ($1, $2, $3, $4, $5, 'AK Test', 'dept', 1, $6)",
            )
            .bind(id)
            .bind(org1)
            .bind(fin_dept)
            .bind(bob)
            .bind(format!("ak-bkt-{}", id.simple()))
            .bind(format!("arn:openrag:org1:{dept_arn_seg}:dept:bucket:{level_seg}/{id}"))
            .execute(&mut **tx.as_mut())
            .await;
            assert_pg_err_code(r, &["42501"], &format!("{ctx_label} INSERT corpus.buckets"))?;
            tx.rollback().await?;
        }
        // 2) wiki.pages (bucket_id IS NULL)
        {
            let id = Uuid::new_v4();
            let mut tx = app.begin_rls_tx(ctx).await?;
            let r = sqlx::query(
                "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
                 VALUES ($1, $2, NULL, $3, $4, 'dept', 1, $5, 'k', $6, 'W', 'b', 'b', 'sha')",
            )
            .bind(id)
            .bind(org1)
            .bind(fin_dept)
            .bind(bob)
            .bind(format!("arn:openrag:org1:{dept_arn_seg}:dept:wiki:{level_seg}/{id}"))
            .bind(format!("ak-w-{}", id.simple()))
            .execute(&mut **tx.as_mut())
            .await;
            assert_pg_err_code(r, &["42501"], &format!("{ctx_label} INSERT wiki.pages"))?;
            tx.rollback().await?;
        }
        // 3) memory.episodes
        {
            let id = Uuid::new_v4();
            let mut tx = app.begin_rls_tx(ctx).await?;
            let r = sqlx::query(
                "INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
                 VALUES ($1, $2, $3, $4, 'private', 3, $5, 'claude_code', 'ep', 'sha')",
            )
            .bind(id)
            .bind(org1)
            .bind(fin_dept)
            .bind(bob)
            .bind(format!("arn:openrag:org1:{dept_arn_seg}:private:episode:{level_seg}/{id}"))
            .execute(&mut **tx.as_mut())
            .await;
            assert_pg_err_code(r, &["42501"], &format!("{ctx_label} INSERT memory.episodes"))?;
            tx.rollback().await?;
        }
        // 4) memory.entries (bucket_id IS NULL)
        {
            let id = Uuid::new_v4();
            let mut tx = app.begin_rls_tx(ctx).await?;
            let r = sqlx::query(
                "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
                 VALUES ($1, $2, NULL, $3, $4, 'private', 3, $5, 'k', 'fact', 'm', 'm', 'sha')",
            )
            .bind(id)
            .bind(org1)
            .bind(fin_dept)
            .bind(bob)
            .bind(format!("arn:openrag:org1:{dept_arn_seg}:private:memory:{level_seg}/{id}"))
            .execute(&mut **tx.as_mut())
            .await;
            assert_pg_err_code(r, &["42501"], &format!("{ctx_label} INSERT memory.entries"))?;
            tx.rollback().await?;
        }
    }

    // Malformed app.api_key_id & malformed app.scoped_arns fail closed with 42501
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        sqlx::query("SELECT set_config('app.api_key_id', 'not-a-uuid', true)")
            .execute(&mut **tx.as_mut())
            .await?;
        let r_bad_ak = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'Bad AK', 'dept', 1, $6)",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("bad-ak-{}", id.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r_bad_ak, &["42501"], "malformed app.api_key_id INSERT")?;
        tx.rollback().await?;

        let mut tx2 = app.begin_rls_tx(&bob_ctx).await?;
        sqlx::query("SELECT set_config('app.scoped_arns', '{unclosed', true)")
            .execute(&mut **tx2.as_mut())
            .await?;
        let r_bad_arns = sqlx::query(
            "INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
             VALUES ($1, $2, $3, $4, 'private', 3, $5, 'claude_code', 'ep', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:private:episode:L3/{id}"))
        .execute(&mut **tx2.as_mut())
        .await;
        assert_pg_err_code(r_bad_arns, &["42501"], "malformed app.scoped_arns INSERT")?;
        tx2.rollback().await?;
    }

    // Positive: active scoped_ak in-scope finance L3 succeeds across all 4 tables
    {
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let b_id = Uuid::new_v4();
        let r1 = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'OK Scoped Bkt', 'dept', 1, $6)",
        )
        .bind(b_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("ok-ak-bkt-{}", b_id.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L3/{b_id}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r1.rows_affected(), 1);

        let w_id = Uuid::new_v4();
        let r2 = sqlx::query(
            "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'dept', 1, $5, 'k', $6, 'OK W', 'b', 'b', 'sha')",
        )
        .bind(w_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:dept:wiki:L3/{w_id}"))
        .bind(format!("ok-ak-w-{}", w_id.simple()))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r2.rows_affected(), 1);

        let e_id = Uuid::new_v4();
        let r3 = sqlx::query(
            "INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
             VALUES ($1, $2, $3, $4, 'private', 3, $5, 'claude_code', 'ep', 'sha')",
        )
        .bind(e_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:private:episode:L3/{e_id}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r3.rows_affected(), 1);

        let m_id = Uuid::new_v4();
        let r4 = sqlx::query(
            "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'private', 3, $5, 'k', 'fact', 'm', 'm', 'sha')",
        )
        .bind(m_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:private:memory:L3/{m_id}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r4.rows_affected(), 1);

        tx.rollback().await?;
    }

    // Verify DELETE FROM iam.orgs WHERE id = org2 cascades cleanly
    {
        let mut tx = admin.inner().begin().await?;
        let del = sqlx::query("DELETE FROM iam.orgs WHERE id = $1")
            .bind(org2)
            .execute(&mut *tx)
            .await?;
        assert_eq!(del.rows_affected(), 1);
        tx.commit().await?;
    }

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_rls_iteration6_hardening_regressions() -> Result<(), Box<dyn std::error::Error>> {
    let admin_url = env_url("ADMIN_DATABASE_URL")?;
    let app_url = env_url("DATABASE_URL")?;

    let admin = DbPool::connect(&admin_url, 4).await?;
    let app = DbPool::connect(&app_url, 4).await?;

    // Verify pg_proc has exactly 1 overload for iam.rls_can_insert (5-arg only, no 4-arg legacy overload)
    let rls_insert_overloads: Vec<i16> = sqlx::query_scalar(
        "SELECT pronargs FROM pg_proc
         WHERE pronamespace = 'iam'::regnamespace
           AND proname = 'rls_can_insert'",
    )
    .fetch_all(admin.inner())
    .await?;
    assert_eq!(
        rls_insert_overloads,
        vec![5],
        "expected exactly one 5-arg iam.rls_can_insert overload"
    );

    // Seed isolated tenants and principals
    let org1 = Uuid::new_v4();
    let org2 = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let hr_dept = Uuid::new_v4();
    let org2_dept = Uuid::new_v4();

    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let dave = Uuid::new_v4();
    let frank = Uuid::new_v4();
    let eve = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO iam.orgs (id, slug, name) VALUES
         ($1, $3, 'Iter6 Org1'),
         ($2, $4, 'Iter6 Org2')",
    )
    .bind(org1)
    .bind(org2)
    .bind(format!("i6-org1-{}", org1.simple()))
    .bind(format!("i6-org2-{}", org2.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $4, 'Finance', 'finance'),
         ($2, $4, 'HR', 'hr'),
         ($3, $5, 'Org2Dept', 'org2dept')",
    )
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(org1)
    .bind(org2)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $7, $9,  'Alice', 8, true,  false),
         ($2, $7, $10, 'Bob',   9, false, false),
         ($3, $7, $11, 'Carol', 7, true,  false),
         ($4, $7, $12, 'Dave',  9, true,  true),
         ($5, $7, $13, 'Frank', 6, false, false),
         ($6, $8, $14, 'Eve',   9, true,  true)",
    )
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(frank)
    .bind(eve)
    .bind(org1)
    .bind(org2)
    .bind(format!("alice-i6-{}@org1.test", org1.simple()))
    .bind(format!("bob-i6-{}@org1.test", org1.simple()))
    .bind(format!("carol-i6-{}@org1.test", org1.simple()))
    .bind(format!("dave-i6-{}@org1.test", org1.simple()))
    .bind(format!("frank-i6-{}@org1.test", org1.simple()))
    .bind(format!("eve-i6-{}@org2.test", org2.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $3, $6,  'member'),
         ($1, $3, $7,  'member'),
         ($1, $3, $8,  'dept_admin'),
         ($1, $4, $9,  'dept_admin'),
         ($1, $4, $10, 'member'),
         ($2, $5, $11, 'dept_admin')",
    )
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(frank)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    // Groups for NEW-MEDIUM-5 (dept-scoped vs NULL-dept org-wide vs cross-org)
    let g_fin = Uuid::new_v4();
    let g_hr = Uuid::new_v4();
    let g_org_wide = Uuid::new_v4();
    let g_org2 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, dept_id, name, group_type) VALUES
         ($1, $5, $7,   'i6-fin-grp',  'permission_group'),
         ($2, $5, $8,   'i6-hr-grp',   'permission_group'),
         ($3, $5, NULL, 'i6-tf-grp',   'project_tf'),
         ($4, $6, $9,   'i6-org2-grp', 'permission_group')",
    )
    .bind(g_fin)
    .bind(g_hr)
    .bind(g_org_wide)
    .bind(g_org2)
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
         ($1, $3, $7),
         ($1, $3, $8),
         ($1, $4, $10),
         ($1, $4, $11),
         ($1, $5, $7),
         ($1, $5, $9),
         ($1, $5, $11),
         ($2, $6, $12)",
    )
    .bind(org1)
    .bind(org2)
    .bind(g_fin)
    .bind(g_hr)
    .bind(g_org_wide)
    .bind(g_org2)
    .bind(alice)
    .bind(bob)
    .bind(carol)
    .bind(dave)
    .bind(frank)
    .bind(eve)
    .execute(admin.inner())
    .await?;

    // API keys
    let bob_scoped_ak = Uuid::new_v4();
    let dave_scoped_ak = Uuid::new_v4();
    let dave_revoked_ak = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.api_keys (id, org_id, principal_id, name, key_prefix, key_hash, scoped_arns, max_auth_level, expires_at, revoked_at) VALUES
         ($1, $4, $5, 'i6-bob-scoped',   'orag', $7, ARRAY['arn:openrag:org1:finance:*'], 5, now() + interval '1 hour', NULL),
         ($2, $4, $6, 'i6-dave-scoped',  'orag', $8, ARRAY['arn:openrag:org1:hr:*'],      2, now() + interval '1 hour', NULL),
         ($3, $4, $6, 'i6-dave-revoked', 'orag', $9, ARRAY[]::TEXT[],                     9, NULL, now() - interval '1 minute')",
    )
    .bind(bob_scoped_ak)
    .bind(dave_scoped_ak)
    .bind(dave_revoked_ak)
    .bind(org1)
    .bind(bob)
    .bind(dave)
    .bind(format!("i6_hash_bob_{}", bob_scoped_ak.simple()))
    .bind(format!("i6_hash_dave_s_{}", dave_scoped_ak.simple()))
    .bind(format!("i6_hash_dave_r_{}", dave_revoked_ak.simple()))
    .execute(admin.inner())
    .await?;

    // Seed parent corpus/wiki/memory/pii/eval fixtures via admin
    let bob_fin_bkt = Uuid::new_v4();
    let bob_fin_doc = Uuid::new_v4();
    let bob_fin_page = Uuid::new_v4();
    let fin_priv_chunk = Uuid::new_v4();
    let fin_dept_chunk = Uuid::new_v4();
    let fin_org_chunk = Uuid::new_v4();
    let hr_priv_chunk = Uuid::new_v4();
    let hr_dept_chunk = Uuid::new_v4();
    let hr_org_chunk = Uuid::new_v4();
    let org2_bkt = Uuid::new_v4();
    let org2_doc = Uuid::new_v4();
    let org2_chunk = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $3, $5, $7, 'i6-bob-fin-bkt', 'Bob Fin Bkt', 'org', 1, $9),
         ($2, $4, $6, $8, 'i6-org2-bkt',    'Org2 Bkt',    'org', 1, $10)",
    )
    .bind(bob_fin_bkt)
    .bind(org2_bkt)
    .bind(org1)
    .bind(org2)
    .bind(fin_dept)
    .bind(org2_dept)
    .bind(bob)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:org:bucket:L1/{bob_fin_bkt}"))
    .bind(format!("arn:openrag:org2:org2dept:org:bucket:L1/{org2_bkt}"))
    .execute(admin.inner())
    .await?;

    let hr_bkt = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $2, $3, $4, 'i6-hr-bkt', 'HR Bkt', 'org', 1, $5)",
    )
    .bind(hr_bkt)
    .bind(org1)
    .bind(hr_dept)
    .bind(frank)
    .bind(format!("arn:openrag:org1:hr:org:bucket:L1/{hr_bkt}"))
    .execute(admin.inner())
    .await?;

    let hr_doc = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256) VALUES
         ($1, $4, $6, $9,  $12, 'org', 1, $15, 'k1', 'Bob Fin Doc', 's1'),
         ($2, $4, $7, $10, $13, 'org', 1, $16, 'k2', 'HR Doc',      's2'),
         ($3, $5, $8, $11, $14, 'org', 1, $17, 'k3', 'Org2 Doc',    's3')",
    )
    .bind(bob_fin_doc)
    .bind(hr_doc)
    .bind(org2_doc)
    .bind(org1)
    .bind(org2)
    .bind(bob_fin_bkt)
    .bind(hr_bkt)
    .bind(org2_bkt)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(bob)
    .bind(frank)
    .bind(eve)
    .bind(format!("arn:openrag:org1:finance:org:document:L1/{bob_fin_doc}"))
    .bind(format!("arn:openrag:org1:hr:org:document:L1/{hr_doc}"))
    .bind(format!("arn:openrag:org2:org2dept:org:document:L1/{org2_doc}"))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.pages (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no) VALUES
         ($1, $2, $3, $4, $5, 'org', 1, $6, 'p1', 1)",
    )
    .bind(bob_fin_page)
    .bind(org1)
    .bind(bob_fin_doc)
    .bind(fin_dept)
    .bind(bob)
    .bind(format!("arn:openrag:org1:finance:org:page:L1/{bob_fin_page}"))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES
         ($1,  $8,  $10, $13, $14, $17, 'private', 3, $20, 'c1', 0, 'fin priv',  'fin priv',  'th1'),
         ($2,  $8,  $10, $13, $14, $17, 'dept',    2, $21, 'c2', 1, 'fin dept',  'fin dept',  'th2'),
         ($3,  $8,  $10, $13, $14, $18, 'org',     1, $22, 'c3', 2, 'fin org',   'fin org',   'th3'),
         ($4,  $8,  $11, NULL, $15, $19, 'private', 2, $23, 'c4', 0, 'hr priv',  'hr priv',   'th4'),
         ($5,  $8,  $11, NULL, $15, $19, 'dept',    3, $24, 'c5', 1, 'hr dept',  'hr dept',   'th5'),
         ($6,  $8,  $11, NULL, $15, $19, 'org',     1, $25, 'c6', 2, 'hr org',   'hr org',    'th6'),
         ($7,  $9,  $12, NULL, $16, $27, 'private', 3, $26, 'c7', 0, 'org2 priv', 'org2 priv', 'th7')",
    )
    .bind(fin_priv_chunk)
    .bind(fin_dept_chunk)
    .bind(fin_org_chunk)
    .bind(hr_priv_chunk)
    .bind(hr_dept_chunk)
    .bind(hr_org_chunk)
    .bind(org2_chunk)
    .bind(org1)
    .bind(org2)
    .bind(bob_fin_doc)
    .bind(hr_doc)
    .bind(org2_doc)
    .bind(bob_fin_page)
    .bind(fin_dept)
    .bind(hr_dept)
    .bind(org2_dept)
    .bind(alice)
    .bind(bob)
    .bind(frank)
    .bind(format!("arn:openrag:org1:finance:private:chunk:L3/{fin_priv_chunk}"))
    .bind(format!("arn:openrag:org1:finance:dept:chunk:L2/{fin_dept_chunk}"))
    .bind(format!("arn:openrag:org1:finance:org:chunk:L1/{fin_org_chunk}"))
    .bind(format!("arn:openrag:org1:hr:private:chunk:L2/{hr_priv_chunk}"))
    .bind(format!("arn:openrag:org1:hr:dept:chunk:L3/{hr_dept_chunk}"))
    .bind(format!("arn:openrag:org1:hr:org:chunk:L1/{hr_org_chunk}"))
    .bind(format!("arn:openrag:org2:org2dept:private:chunk:L3/{org2_chunk}"))
    .bind(eve)
    .execute(admin.inner())
    .await?;

    let gp_org1 = Uuid::new_v4();
    sqlx::query("INSERT INTO pii.guard_policies (id, org_id, name) VALUES ($1, $2, 'i6-gp-org1')")
        .bind(gp_org1)
        .bind(org1)
        .execute(admin.inner())
        .await?;

    // Outbox rows for NEW-LOW-2
    let ob_fin_priv_l3: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"fin_priv\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(fin_priv_chunk)
    .fetch_one(admin.inner())
    .await?;
    let ob_fin_dept_l2: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"fin_dept\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(fin_dept_chunk)
    .fetch_one(admin.inner())
    .await?;
    let ob_fin_org_l1: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"fin_org\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(fin_org_chunk)
    .fetch_one(admin.inner())
    .await?;
    let ob_hr_priv_l2: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"hr_priv\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(hr_priv_chunk)
    .fetch_one(admin.inner())
    .await?;
    let ob_hr_dept_l3: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"hr_dept\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(hr_dept_chunk)
    .fetch_one(admin.inner())
    .await?;
    let ob_hr_org_l1: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"hr_org\"}'::jsonb) RETURNING id",
    )
    .bind(org1)
    .bind(hr_org_chunk)
    .fetch_one(admin.inner())
    .await?;
    let ob_org2: i64 = sqlx::query_scalar(
        "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, payload)
         VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', '{\"k\":\"org2\"}'::jsonb) RETURNING id",
    )
    .bind(org2)
    .bind(org2_chunk)
    .fetch_one(admin.inner())
    .await?;

    // Principal contexts
    let mut alice_ctx = PrincipalContext::new(org1, alice, AuthLevel::new(8)?);
    alice_ctx.dept_ids = vec![fin_dept];

    let mut bob_ctx = PrincipalContext::new(org1, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![fin_dept];

    let mut bob_low_session_ctx = PrincipalContext::new(org1, bob, AuthLevel::new(3)?);
    bob_low_session_ctx.dept_ids = vec![fin_dept];

    let mut bob_scoped_ctx = bob_ctx.clone();
    bob_scoped_ctx.api_key_id = Some(bob_scoped_ak);
    bob_scoped_ctx.scoped_arns = vec!["arn:openrag:org1:finance:*".to_string()];

    let mut carol_ctx = PrincipalContext::new(org1, carol, AuthLevel::new(7)?);
    carol_ctx.dept_ids = vec![fin_dept];
    carol_ctx.dept_admin_ids = vec![fin_dept];

    let mut dave_ctx = PrincipalContext::new(org1, dave, AuthLevel::new(9)?);
    dave_ctx.dept_ids = vec![hr_dept];
    dave_ctx.dept_admin_ids = vec![hr_dept];
    dave_ctx.is_org_admin = true;

    let mut dave_scoped_ctx = dave_ctx.clone();
    dave_scoped_ctx.api_key_id = Some(dave_scoped_ak);
    dave_scoped_ctx.scoped_arns = vec!["arn:openrag:org1:hr:*".to_string()];

    let mut dave_revoked_ctx = dave_ctx.clone();
    dave_revoked_ctx.api_key_id = Some(dave_revoked_ak);

    // -------------------------------------------------------------------------
    // Block 1: NEW-MEDIUM-3 (rls_can_insert min_auth_level + ARN level + app.auth_level)
    // -------------------------------------------------------------------------
    // 1a. Spoofed :L1/ with min_auth_level = 10 under bob_scoped_ctx (max_auth_level = 5) across all 8 tables
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'Spoof Bkt', 'dept', 10, $6)",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("i6-spoof-bkt-{}", id.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 corpus.buckets")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
             VALUES ($1, $2, $3, $4, $5, 'dept', 10, $6, 'k', 'Spoof Doc', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(bob_fin_bkt)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:dept:document:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 corpus.documents")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.pages (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no)
             VALUES ($1, $2, $3, $4, $5, 'dept', 10, $6, 'k', 2)",
        )
        .bind(id)
        .bind(org1)
        .bind(bob_fin_doc)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:dept:page:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 corpus.pages")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.page_figures (id, org_id, page_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, figure_idx)
             VALUES ($1, $2, $3, $4, $5, $6, 'dept', 10, $7, 'k', 1)",
        )
        .bind(id)
        .bind(org1)
        .bind(bob_fin_page)
        .bind(bob_fin_doc)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:dept:figure:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 corpus.page_figures")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.chunks (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
             VALUES ($1, $2, $3, $4, $5, $6, 'dept', 10, $7, 'k', 9, 't', 't', 'h')",
        )
        .bind(id)
        .bind(org1)
        .bind(bob_fin_doc)
        .bind(bob_fin_page)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:dept:chunk:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 corpus.chunks")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'dept', 10, $5, 'k', $6, 'W', 'b', 'b', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:dept:wiki:L1/{id}"))
        .bind(format!("i6-spoof-w-{}", id.simple()))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 wiki.pages")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
             VALUES ($1, $2, $3, $4, 'private', 10, $5, 'claude_code', 'ep', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:private:episode:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 memory.episodes")?;
        tx.rollback().await?;
    }
    {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r = sqlx::query(
            "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'private', 10, $5, 'k', 'fact', 'm', 'm', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:finance:private:memory:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], "spoofed L1 + min_auth_level=10 memory.entries")?;
        tx.rollback().await?;
    }

    // 1b. Low-session app.auth_level=3 with spoofed :L2/ and min_auth_level=10, plus ARN spoofs/malformed levels
    for (label, ctx, min_lvl, arn_fn) in [
        (
            "low session auth_level=3 spoofed :L2/ min=10",
            &bob_low_session_ctx,
            10_i16,
            format!("arn:openrag:org1:finance:dept:bucket:L2/{}", Uuid::new_v4()),
        ),
        (
            "omitted :L<n>/ with min=10",
            &bob_scoped_ctx,
            10_i16,
            format!("arn:openrag:org1:finance:dept:bucket:no-level/{}", Uuid::new_v4()),
        ),
        (
            "empty arn",
            &bob_ctx,
            1_i16,
            String::new(),
        ),
        (
            "out-of-range :L0/",
            &bob_ctx,
            1_i16,
            format!("arn:openrag:org1:finance:dept:bucket:L0/{}", Uuid::new_v4()),
        ),
        (
            "out-of-range :L11/",
            &bob_ctx,
            1_i16,
            format!("arn:openrag:org1:finance:dept:bucket:L11/{}", Uuid::new_v4()),
        ),
        (
            "earlier-segment :L1/ spoof before :L10/",
            &bob_scoped_ctx,
            1_i16,
            format!("arn:openrag:org1:finance:L1/spoof:bucket:L10/{}", Uuid::new_v4()),
        ),
        (
            "v_arn_level < min_auth_level (L2 < min 5)",
            &bob_ctx,
            5_i16,
            format!("arn:openrag:org1:finance:dept:bucket:L2/{}", Uuid::new_v4()),
        ),
    ] {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(ctx).await?;
        let r = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'Bypass Bkt', 'dept', $6, $7)",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("i6-byp-{}", id.simple()))
        .bind(min_lvl)
        .bind(arn_fn)
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r, &["42501"], label)?;
        tx.rollback().await?;
    }

    // 1c. Unset / empty / '0' / '11' / 'abc' app.auth_level fails closed in BOTH rls_can_insert AND session_active_principal()
    for bad_auth in ["", "0", "11", "abc"] {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_ctx).await?;
        sqlx::query("SELECT set_config('app.auth_level', $1, true)")
            .bind(bad_auth)
            .execute(&mut **tx.as_mut())
            .await?;

        let r_ins = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'Bad Auth', 'dept', 1, $6)",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("i6-bad-auth-{}", id.simple()))
        .bind(format!("arn:openrag:org1:finance:dept:bucket:L1/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(
            r_ins,
            &["42501"],
            &format!("bad app.auth_level='{bad_auth}' rls_can_insert"),
        )?;
        tx.rollback().await?;

        // Check session_active_principal() SELECTs return 0 rows and INSERTs fail with 42501
        let mut tx_sel = app.begin_rls_tx(&dave_ctx).await?;
        sqlx::query("SELECT set_config('app.auth_level', $1, true)")
            .bind(bad_auth)
            .execute(&mut **tx_sel.as_mut())
            .await?;
        for table in [
            "iam.orgs",
            "iam.departments",
            "iam.groups",
            "iam.group_members",
            "pii.audit_logs",
            "pii.guard_events",
            "eval.search_feedback",
            "indexing.outbox",
        ] {
            let cnt: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
                .fetch_one(&mut **tx_sel.as_mut())
                .await?;
            assert_eq!(
                cnt, 0,
                "bad app.auth_level='{bad_auth}' must yield 0 rows on {table}"
            );
        }
        tx_sel.rollback().await?;

        let mut tx_al = app.begin_rls_tx(&bob_ctx).await?;
        sqlx::query("SELECT set_config('app.auth_level', $1, true)")
            .bind(bad_auth)
            .execute(&mut **tx_al.as_mut())
            .await?;
        let r_al = sqlx::query(
            "INSERT INTO pii.audit_logs (org_id, principal_id, action, allowed)
             VALUES ($1, $2, 'read', true)",
        )
        .bind(org1)
        .bind(bob)
        .execute(&mut **tx_al.as_mut())
        .await;
        assert_pg_err_code(
            r_al,
            &["42501"],
            &format!("bad app.auth_level='{bad_auth}' pii.audit_logs INSERT"),
        )?;
        tx_al.rollback().await?;

        let mut tx_ge = app.begin_rls_tx(&bob_ctx).await?;
        sqlx::query("SELECT set_config('app.auth_level', $1, true)")
            .bind(bad_auth)
            .execute(&mut **tx_ge.as_mut())
            .await?;
        let r_ge = sqlx::query(
            "INSERT INTO pii.guard_events (org_id, policy_id, principal_id, surface, decision)
             VALUES ($1, $2, $3, 'inbound_query', 'allow')",
        )
        .bind(org1)
        .bind(gp_org1)
        .bind(bob)
        .execute(&mut **tx_ge.as_mut())
        .await;
        assert_pg_err_code(
            r_ge,
            &["42501"],
            &format!("bad app.auth_level='{bad_auth}' pii.guard_events INSERT"),
        )?;
        tx_ge.rollback().await?;

        let mut tx_fb = app.begin_rls_tx(&bob_ctx).await?;
        sqlx::query("SELECT set_config('app.auth_level', $1, true)")
            .bind(bad_auth)
            .execute(&mut **tx_fb.as_mut())
            .await?;
        let r_fb = sqlx::query(
            "INSERT INTO eval.search_feedback (org_id, principal_id, query_text, verdict)
             VALUES ($1, $2, 'q', 'up')",
        )
        .bind(org1)
        .bind(bob)
        .execute(&mut **tx_fb.as_mut())
        .await;
        assert_pg_err_code(
            r_fb,
            &["42501"],
            &format!("bad app.auth_level='{bad_auth}' eval.search_feedback INSERT"),
        )?;
        tx_fb.rollback().await?;
    }

    // Malformed array GUCs in session_active_principal() also fail closed
    for guc_name in [
        "app.dept_ids",
        "app.group_ids",
        "app.scoped_group_ids",
        "app.scoped_arns",
    ] {
        let mut tx = app.begin_rls_tx(&dave_ctx).await?;
        sqlx::query("SELECT set_config($1, '{unclosed', true)")
            .bind(guc_name)
            .execute(&mut **tx.as_mut())
            .await?;
        let cnt: i64 = sqlx::query_scalar("SELECT count(*) FROM iam.orgs")
            .fetch_one(&mut **tx.as_mut())
            .await?;
        assert_eq!(cnt, 0, "malformed {guc_name} must fail closed in session_active_principal()");
        tx.rollback().await?;
    }

    // 1d. Positive controls under bob_scoped_ctx (max_auth_level=5): min=5 + :L5/ AND min=1 + :L3/
    for (min_lvl, arn_lvl) in [(5_i16, "L5"), (1_i16, "L3")] {
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let b_id = Uuid::new_v4();
        let r_b = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'OK Bkt', 'dept', $6, $7)",
        )
        .bind(b_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("i6-ok-bkt-{}", b_id.simple()))
        .bind(min_lvl)
        .bind(format!("arn:openrag:org1:finance:dept:bucket:{arn_lvl}/{b_id}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r_b.rows_affected(), 1);

        let w_id = Uuid::new_v4();
        let r_w = sqlx::query(
            "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'dept', $5, $6, 'k', $7, 'OK W', 'b', 'b', 'sha')",
        )
        .bind(w_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(min_lvl)
        .bind(format!("arn:openrag:org1:finance:dept:wiki:{arn_lvl}/{w_id}"))
        .bind(format!("i6-ok-w-{}", w_id.simple()))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r_w.rows_affected(), 1);

        let e_id = Uuid::new_v4();
        let r_e = sqlx::query(
            "INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
             VALUES ($1, $2, $3, $4, 'private', $5, $6, 'claude_code', 'ep', 'sha')",
        )
        .bind(e_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(min_lvl)
        .bind(format!("arn:openrag:org1:finance:private:episode:{arn_lvl}/{e_id}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r_e.rows_affected(), 1);

        let m_id = Uuid::new_v4();
        let r_m = sqlx::query(
            "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'private', $5, $6, 'k', 'fact', 'm', 'm', 'sha')",
        )
        .bind(m_id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(min_lvl)
        .bind(format!("arn:openrag:org1:finance:private:memory:{arn_lvl}/{m_id}"))
        .execute(&mut **tx.as_mut())
        .await?;
        assert_eq!(r_m.rows_affected(), 1);

        tx.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // Block 2: NEW-MEDIUM-4 (idx_indexing_outbox_idem scoped to org_id)
    // -------------------------------------------------------------------------
    {
        // Seed pending and processing rows in org2
        sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, status, idempotency_key) VALUES
             ($1, 'corpus.chunks', $2, 'upsert_embed', 'pending',    'i6-shared-idem-key'),
             ($1, 'corpus.chunks', $2, 'upsert_embed', 'processing', 'i6-shared-proc-key')",
        )
        .bind(org2)
        .bind(org2_chunk)
        .execute(admin.inner())
        .await?;

        // Same idempotency_key allowed in org1 for both pending and processing
        let mut tx1 = app.begin_rls_tx(&bob_ctx).await?;
        let r1 = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, status, idempotency_key)
             VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', 'pending', 'i6-shared-idem-key')",
        )
        .bind(org1)
        .bind(fin_org_chunk)
        .execute(&mut **tx1.as_mut())
        .await?;
        assert_eq!(r1.rows_affected(), 1);

        let r2 = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, status, idempotency_key)
             VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', 'processing', 'i6-shared-proc-key')",
        )
        .bind(org1)
        .bind(fin_org_chunk)
        .execute(&mut **tx1.as_mut())
        .await?;
        assert_eq!(r2.rows_affected(), 1);
        tx1.commit().await?;

        // Same-tenant duplicate pending blocked with 23505
        let mut tx_dup = app.begin_rls_tx(&bob_ctx).await?;
        let r_dup = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, status, idempotency_key)
             VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', 'pending', 'i6-shared-idem-key')",
        )
        .bind(org1)
        .bind(fin_org_chunk)
        .execute(&mut **tx_dup.as_mut())
        .await;
        assert_pg_err_code(r_dup, &["23505"], "same-tenant duplicate pending idempotency_key")?;
        tx_dup.rollback().await?;

        // Same-tenant duplicate processing blocked with 23505
        let mut tx_dup_proc = app.begin_rls_tx(&bob_ctx).await?;
        let r_dup_proc = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, status, idempotency_key)
             VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', 'pending', 'i6-shared-proc-key')",
        )
        .bind(org1)
        .bind(fin_org_chunk)
        .execute(&mut **tx_dup_proc.as_mut())
        .await;
        assert_pg_err_code(
            r_dup_proc,
            &["23505"],
            "same-tenant duplicate vs processing idempotency_key",
        )?;
        tx_dup_proc.rollback().await?;

        // Mark org1's 'i6-shared-idem-key' row done -> reuse allowed
        sqlx::query(
            "UPDATE indexing.outbox SET status = 'done', processed_at = now()
             WHERE org_id = $1 AND idempotency_key = 'i6-shared-idem-key'",
        )
        .bind(org1)
        .execute(admin.inner())
        .await?;

        let mut tx_reuse = app.begin_rls_tx(&bob_ctx).await?;
        let r_reuse = sqlx::query(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, status, idempotency_key)
             VALUES ($1, 'corpus.chunks', $2, 'upsert_embed', 'pending', 'i6-shared-idem-key')",
        )
        .bind(org1)
        .bind(fin_org_chunk)
        .execute(&mut **tx_reuse.as_mut())
        .await?;
        assert_eq!(r_reuse.rows_affected(), 1);
        tx_reuse.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // Block 3: NEW-MEDIUM-5 (dept_admin group_members scoping)
    // -------------------------------------------------------------------------
    {
        // Carol (fin_dept dept_admin, non-org-admin): sees g_fin (alice, bob), 0 g_hr, ONLY self in g_org_wide (dept_id IS NULL), 0 org2
        let mut tx = app.begin_rls_tx(&carol_ctx).await?;
        let rows: Vec<(Uuid, Uuid)> =
            sqlx::query_as("SELECT group_id, principal_id FROM iam.group_members")
                .fetch_all(&mut **tx.as_mut())
                .await?;
        assert!(rows.contains(&(g_fin, alice)));
        assert!(rows.contains(&(g_fin, bob)));
        assert!(rows.contains(&(g_org_wide, carol)));
        assert!(!rows.iter().any(|(gid, _)| *gid == g_hr));
        assert!(!rows.iter().any(|(gid, _)| *gid == g_org2));
        assert!(!rows.contains(&(g_org_wide, alice)));
        assert!(!rows.contains(&(g_org_wide, frank)));
        assert_eq!(rows.len(), 3);
        tx.rollback().await?;

        // Dave (org_admin): sees all org1 memberships (g_fin, g_hr, g_org_wide), 0 org2
        let mut tx_d = app.begin_rls_tx(&dave_ctx).await?;
        let rows_d: Vec<(Uuid, Uuid)> =
            sqlx::query_as("SELECT group_id, principal_id FROM iam.group_members")
                .fetch_all(&mut **tx_d.as_mut())
                .await?;
        assert_eq!(rows_d.len(), 7);
        assert!(!rows_d.iter().any(|(gid, _)| *gid == g_org2));
        tx_d.rollback().await?;

        // Alice (non-admin): sees ONLY self
        let mut tx_a = app.begin_rls_tx(&alice_ctx).await?;
        let rows_a: Vec<(Uuid, Uuid)> =
            sqlx::query_as("SELECT group_id, principal_id FROM iam.group_members")
                .fetch_all(&mut **tx_a.as_mut())
                .await?;
        assert_eq!(rows_a.len(), 2);
        assert!(rows_a.iter().all(|(_, pid)| *pid == alice));
        tx_a.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // Block 4: NEW-LOW-2 (rls_indexing_outbox_read private/dept/scoped_ak isolation)
    // -------------------------------------------------------------------------
    {
        // Dave unscoped is_org_admin (hr_dept dept_admin, L9):
        // Sees ob_fin_org_l1, ob_hr_priv_l2, ob_hr_dept_l3, ob_hr_org_l1; NOT ob_fin_priv_l3, ob_fin_dept_l2, ob_org2
        let mut tx = app.begin_rls_tx(&dave_ctx).await?;
        let ob: Vec<i64> = sqlx::query_scalar("SELECT id FROM indexing.outbox")
            .fetch_all(&mut **tx.as_mut())
            .await?;
        assert!(ob.contains(&ob_fin_org_l1));
        assert!(ob.contains(&ob_hr_priv_l2));
        assert!(ob.contains(&ob_hr_dept_l3));
        assert!(ob.contains(&ob_hr_org_l1));
        assert!(!ob.contains(&ob_fin_priv_l3));
        assert!(!ob.contains(&ob_fin_dept_l2));
        assert!(!ob.contains(&ob_org2));
        tx.rollback().await?;

        // Dave with scoped_ak (scoped_arns = ['arn:openrag:org1:hr:*'], max_auth_level = 2):
        // Sees ONLY ob_hr_priv_l2 and ob_hr_org_l1
        let mut tx_s = app.begin_rls_tx(&dave_scoped_ctx).await?;
        let ob_s: Vec<i64> = sqlx::query_scalar("SELECT id FROM indexing.outbox")
            .fetch_all(&mut **tx_s.as_mut())
            .await?;
        assert!(ob_s.contains(&ob_hr_priv_l2));
        assert!(ob_s.contains(&ob_hr_org_l1));
        assert!(!ob_s.contains(&ob_fin_org_l1));
        assert!(!ob_s.contains(&ob_hr_dept_l3));
        assert!(!ob_s.contains(&ob_fin_priv_l3));
        assert!(!ob_s.contains(&ob_fin_dept_l2));
        assert!(!ob_s.contains(&ob_org2));
        tx_s.rollback().await?;

        // Dave with revoked_ak: sees 0 rows
        let mut tx_r = app.begin_rls_tx(&dave_revoked_ctx).await?;
        let ob_r: Vec<i64> = sqlx::query_scalar("SELECT id FROM indexing.outbox")
            .fetch_all(&mut **tx_r.as_mut())
            .await?;
        assert!(ob_r.is_empty());
        tx_r.rollback().await?;

        // Carol (fin_dept dept_admin): sees ob_fin_priv_l3, ob_fin_dept_l2, ob_fin_org_l1, ob_hr_org_l1; NOT hr_priv/hr_dept/org2
        let mut tx_c = app.begin_rls_tx(&carol_ctx).await?;
        let ob_c: Vec<i64> = sqlx::query_scalar("SELECT id FROM indexing.outbox")
            .fetch_all(&mut **tx_c.as_mut())
            .await?;
        assert!(ob_c.contains(&ob_fin_priv_l3));
        assert!(ob_c.contains(&ob_fin_dept_l2));
        assert!(ob_c.contains(&ob_fin_org_l1));
        assert!(ob_c.contains(&ob_hr_org_l1));
        assert!(!ob_c.contains(&ob_hr_priv_l2));
        assert!(!ob_c.contains(&ob_hr_dept_l3));
        assert!(!ob_c.contains(&ob_org2));
        tx_c.rollback().await?;
    }

    // -------------------------------------------------------------------------
    // Block 5: Non-regression of NEW-HIGH-4 & NEW-MEDIUM-2
    // -------------------------------------------------------------------------
    for table in [
        "iam.orgs",
        "iam.departments",
        "iam.groups",
        "iam.group_members",
        "pii.audit_logs",
        "pii.guard_events",
        "eval.search_feedback",
        "indexing.outbox",
    ] {
        let cnt: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
            .fetch_one(app.inner())
            .await?;
        assert_eq!(cnt, 0, "unset GUCs must yield 0 rows for {table}");
    }

    for (label, dept_arn_seg, level_seg) in [
        ("scoped_ak out-of-scope hr", "hr", "L3"),
        ("scoped_ak exceeding max_auth_level L8", "finance", "L8"),
    ] {
        let id = Uuid::new_v4();
        let mut tx = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r_b = sqlx::query(
            "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
             VALUES ($1, $2, $3, $4, $5, 'NR Bkt', 'dept', 1, $6)",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("i6-nr-bkt-{}", id.simple()))
        .bind(format!("arn:openrag:org1:{dept_arn_seg}:dept:bucket:{level_seg}/{id}"))
        .execute(&mut **tx.as_mut())
        .await;
        assert_pg_err_code(r_b, &["42501"], &format!("{label} corpus.buckets"))?;
        tx.rollback().await?;

        let mut tx_w = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r_w = sqlx::query(
            "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'dept', 1, $5, 'k', $6, 'W', 'b', 'b', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:{dept_arn_seg}:dept:wiki:{level_seg}/{id}"))
        .bind(format!("i6-nr-w-{}", id.simple()))
        .execute(&mut **tx_w.as_mut())
        .await;
        assert_pg_err_code(r_w, &["42501"], &format!("{label} wiki.pages"))?;
        tx_w.rollback().await?;

        let mut tx_e = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r_e = sqlx::query(
            "INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, source_harness, summary_md, content_sha256)
             VALUES ($1, $2, $3, $4, 'private', 3, $5, 'claude_code', 'ep', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:{dept_arn_seg}:private:episode:{level_seg}/{id}"))
        .execute(&mut **tx_e.as_mut())
        .await;
        assert_pg_err_code(r_e, &["42501"], &format!("{label} memory.episodes"))?;
        tx_e.rollback().await?;

        let mut tx_m = app.begin_rls_tx(&bob_scoped_ctx).await?;
        let r_m = sqlx::query(
            "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
             VALUES ($1, $2, NULL, $3, $4, 'private', 3, $5, 'k', 'fact', 'm', 'm', 'sha')",
        )
        .bind(id)
        .bind(org1)
        .bind(fin_dept)
        .bind(bob)
        .bind(format!("arn:openrag:org1:{dept_arn_seg}:private:memory:{level_seg}/{id}"))
        .execute(&mut **tx_m.as_mut())
        .await;
        assert_pg_err_code(r_m, &["42501"], &format!("{label} memory.entries"))?;
        tx_m.rollback().await?;
    }

    // Cleanup
    sqlx::query("DELETE FROM iam.orgs WHERE id IN ($1, $2)")
        .bind(org1)
        .bind(org2)
        .execute(admin.inner())
        .await?;

    Ok(())
}
