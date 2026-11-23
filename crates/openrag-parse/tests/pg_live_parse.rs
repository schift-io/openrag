use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use openrag_core::{AccessScope, AuthLevel, OcrModeRule, ParseConfig, PrincipalContext, Sensitivity, ThumbnailConfig};
use openrag_db::DbPool;
use openrag_parse::{
    DocumentParser, InMemoryBlobStore, ParsePersister, ParsePolicyCtx, PersistParsedDocumentInput,
    Thumbnailer,
};
use sqlx::Row;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_parse_persist_under_rls_and_peer_isolation() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;

    let org_id = Uuid::new_v4();
    let org_slug = format!("org-parse-{}", org_id.simple());
    let dept_id = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let owner_id = Uuid::new_v4();
    let peer_id = Uuid::new_v4();
    let dept_admin_id = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Parse Test Org')")
        .bind(org_id)
        .bind(&org_slug)
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Engineering', 'eng')",
    )
    .bind(dept_id)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $4, $5, 'Owner', 6, false, false),
         ($2, $4, $6, 'Peer', 9, false, false),
         ($3, $4, $7, 'DeptAdmin', 6, false, false)",
    )
    .bind(owner_id)
    .bind(peer_id)
    .bind(dept_admin_id)
    .bind(org_id)
    .bind(format!("owner-{}@parse.test", org_id.simple()))
    .bind(format!("peer-{}@parse.test", org_id.simple()))
    .bind(format!("admin-{}@parse.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'member'),
         ($1, $2, $4, 'member'),
         ($1, $2, $5, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(owner_id)
    .bind(peer_id)
    .bind(dept_admin_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, name, slug, access_scope, min_auth_level, arn)
         VALUES ($1, $2, $3, $4, 'Eng Bucket', 'eng-bucket', 'dept', 1, $5)",
    )
    .bind(bucket_id)
    .bind(org_id)
    .bind(dept_id)
    .bind(owner_id)
    .bind(format!("arn:openrag:{org_slug}:eng:dept:bucket:L1/{bucket_id}"))
    .execute(admin.inner())
    .await?;

    // Build synthetic PNG with figure box and parse + thumbnail + persist under owner's RLS tx
    let mut img = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
    for y in 60..150 {
        for x in 80..240 {
            img.put_pixel(x, y, Rgb([10, 120, 40]));
        }
    }
    let mut png_cur = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(img).write_to(&mut png_cur, ImageFormat::Png)?;
    let png_bytes = png_cur.into_inner();

    let parse_cfg = ParseConfig {
        hwp_engine: "kordoc".into(),
        ocr_engine: "schift_ocr".into(),
        dewarp_enabled: true,
        pdf_native_enabled: true,
        default_ocr_mode: Sensitivity::Share,
        ocr_mode_by_auth_level: vec![OcrModeRule {
            min_auth_level: AuthLevel::new(5)?,
            required_mode: Sensitivity::Private,
        }],
        reject_zdr_with_batch: true,
    };
    let thumb_cfg = ThumbnailConfig {
        enabled: true,
        format: "webp".into(),
        max_edge_px: 1280,
        figure_max_edge_px: 1280,
        figure_max_bytes: 307_200,
    };

    let doc = DocumentParser::new()
        .parse(
            &png_bytes,
            Some("image/png"),
            Some("diagram.png"),
            &parse_cfg,
            &ParsePolicyCtx {
                auth_level: AuthLevel::new(6)?,
                sensitivity: Sensitivity::Private,
                ..Default::default()
            },
        )
        .await?;
    let thumbs = Thumbnailer::new().render(&doc, Some(&png_bytes), &thumb_cfg)?;

    let mut owner_ctx = PrincipalContext::new(org_id, owner_id, AuthLevel::new(6)?);
    owner_ctx.dept_ids = vec![dept_id];

    let blob_store = InMemoryBlobStore::new();
    let mut tx = app.begin_rls_tx(&owner_ctx).await?;
    let persisted = ParsePersister::persist_parsed_document(
        &mut tx,
        &PersistParsedDocumentInput {
            org_id,
            org_slug: org_slug.clone(),
            dept_id,
            dept_path: "eng".into(),
            bucket_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Private,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(3)?,
            sensitivity: Sensitivity::Private,
            title: "Private Diagram".into(),
            source_uri: Some("upload://diagram.png".into()),
        },
        &doc,
        &thumbs,
        &blob_store,
    )
    .await?;
    tx.commit().await?;

    assert_eq!(persisted.page_ids.len(), 1);
    assert!(!persisted.figure_ids.is_empty());

    // Owner sees the persisted page and its thumbnail URI (pages carry no vector)
    let mut owner_tx = app.begin_rls_tx(&owner_ctx).await?;
    let thumb_uri: Option<String> =
        sqlx::query_scalar("SELECT thumb_uri FROM corpus.pages WHERE id = $1")
            .bind(persisted.page_ids[0])
            .fetch_optional(&mut **owner_tx.tx_mut())
            .await?
            .flatten();
    assert!(thumb_uri.is_some_and(|u| !u.is_empty()));
    owner_tx.commit().await?;

    // Same-dept peer (even at higher auth_level=9) sees 0 rows under Fail-Closed RLS
    let mut peer_ctx = PrincipalContext::new(org_id, peer_id, AuthLevel::new(9)?);
    peer_ctx.dept_ids = vec![dept_id];
    let mut peer_tx = app.begin_rls_tx(&peer_ctx).await?;
    let peer_doc_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM corpus.documents WHERE id = $1")
            .bind(persisted.document_id)
            .fetch_one(&mut **peer_tx.tx_mut())
            .await?;
    assert_eq!(peer_doc_count, 0);
    peer_tx.commit().await?;

    // Dept admin of same dept CAN read the private document & figure
    let mut admin_ctx = PrincipalContext::new(org_id, dept_admin_id, AuthLevel::new(6)?);
    admin_ctx.dept_ids = vec![dept_id];
    admin_ctx.dept_admin_ids = vec![dept_id];
    let mut admin_tx = app.begin_rls_tx(&admin_ctx).await?;
    let fig_row = sqlx::query(
        "SELECT bbox_norm_1000 FROM corpus.page_figures WHERE id = $1",
    )
    .bind(persisted.figure_ids[0])
    .fetch_one(&mut **admin_tx.tx_mut())
    .await?;
    let bbox: Vec<i32> = fig_row.try_get("bbox_norm_1000")?;
    assert_eq!(bbox.len(), 4);
    assert!(bbox.iter().all(|&c| (0..=999).contains(&c)));
    admin_tx.commit().await?;

    Ok(())
}
