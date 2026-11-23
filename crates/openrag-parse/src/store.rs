use openrag_core::{AccessScope, Arn, ArnResourceType, AuthLevel, OpenRagError, Sensitivity};
use openrag_db::RlsTransaction;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::model::{ParsedDocument, RenderedPageThumbnail};
use crate::thumbnail::ThumbnailBlobStore;

#[derive(Debug, Clone)]
pub struct PersistParsedDocumentInput {
    pub org_id: Uuid,
    pub org_slug: String,
    pub dept_id: Uuid,
    pub dept_path: String,
    pub bucket_id: Uuid,
    pub owner_principal_id: Uuid,
    pub access_scope: AccessScope,
    pub allowed_group_ids: Vec<Uuid>,
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub sensitivity: Sensitivity,
    pub title: String,
    pub source_uri: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedDocumentIds {
    pub document_id: Uuid,
    pub document_arn: String,
    pub page_ids: Vec<Uuid>,
    pub figure_ids: Vec<Uuid>,
    pub thumb_uris: Vec<String>,
    pub figure_uris: Vec<String>,
}

pub struct ParsePersister;

impl ParsePersister {
    pub async fn persist_parsed_document(
        tx: &mut RlsTransaction<'_>,
        input: &PersistParsedDocumentInput,
        doc: &ParsedDocument,
        thumbs: &[RenderedPageThumbnail],
        blob_store: &dyn ThumbnailBlobStore,
    ) -> Result<PersistedDocumentIds, OpenRagError> {
        if !thumbs.is_empty() && doc.pages.len() != thumbs.len() {
            return Err(OpenRagError::Validation(format!(
                "pages count ({}) must match thumbnails count ({})",
                doc.pages.len(),
                thumbs.len()
            )));
        }

        let doc_id = Uuid::now_v7();
        let doc_arn = Arn::new(
            &input.org_slug,
            &input.dept_path,
            input.access_scope,
            ArnResourceType::Document,
            input.min_auth_level,
            doc_id.to_string(),
        )?
        .to_canonical();

        let acl_scope_key = format!(
            "{}:{}:{}:L{}",
            input.org_id,
            input.access_scope.as_str(),
            input.dept_id,
            input.min_auth_level.as_u8()
        );

        let metadata = serde_json::json!({
            "parser_engine": doc.parser_engine,
            "extracted_aliases": doc.extracted_aliases,
            "ocr_policy": doc.ocr_policy,
            "page_count": doc.pages.len(),
        });

        let title = if input.title.trim().is_empty() {
            doc.title.as_str()
        } else {
            input.title.as_str()
        };

        sqlx::query(
            "INSERT INTO corpus.documents (
                id, org_id, bucket_id, dept_id, owner_principal_id,
                access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                arn, acl_scope_key, title, source_uri, mime_type, content_sha256,
                sensitivity, status, metadata
             ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12, $13, $14, $15,
                $16, 'indexed', $17
             )",
        )
        .bind(doc_id)
        .bind(input.org_id)
        .bind(input.bucket_id)
        .bind(input.dept_id)
        .bind(input.owner_principal_id)
        .bind(input.access_scope.as_str())
        .bind(&input.allowed_group_ids)
        .bind(&input.denied_group_ids)
        .bind(input.min_auth_level.as_i16())
        .bind(&doc_arn)
        .bind(&acl_scope_key)
        .bind(title)
        .bind(&input.source_uri)
        .bind(&doc.mime_type)
        .bind(&doc.content_sha256)
        .bind(input.sensitivity.as_str())
        .bind(metadata)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("failed inserting corpus.documents: {e}"))
        })?;

        let mut page_ids = Vec::with_capacity(doc.pages.len());
        let mut figure_ids = Vec::new();
        let mut thumb_uris = Vec::with_capacity(doc.pages.len());
        let mut figure_uris = Vec::new();

        // `thumbs` is empty when thumbnails are off: pages are stored without them.
        for (idx, page) in doc.pages.iter().enumerate() {
            let thumb = thumbs.get(idx);
            let page_id = Uuid::now_v7();
            let page_arn = Arn::new(
                &input.org_slug,
                &input.dept_path,
                input.access_scope,
                ArnResourceType::Page,
                input.min_auth_level,
                page_id.to_string(),
            )?
            .to_canonical();

            let thumb_uri = match thumb {
                Some(t) => {
                    let key = format!("{}/{doc_id}/page-{:03}.webp", input.org_id, page.page_no);
                    Some(blob_store.put_bytes(&key, &t.webp_bytes).await?)
                }
                None => None,
            };

            let page_state = serde_json::json!({
                "tables_html": page.tables_html,
                "formulas": page.formulas,
                "fields": page.fields,
                "blocks": page.blocks,
                "extracted_aliases": page.extracted_aliases,
            });
            sqlx::query(
                "INSERT INTO corpus.pages (
                    id, org_id, document_id, dept_id, owner_principal_id,
                    access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                    arn, acl_scope_key, page_no, raw_markdown, redacted_markdown,
                    page_state, thumb_uri, content_sha256
                 ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11, $12, $13, $14,
                    $15, $16, $17
                 )",
            )
            .bind(page_id)
            .bind(input.org_id)
            .bind(doc_id)
            .bind(input.dept_id)
            .bind(input.owner_principal_id)
            .bind(input.access_scope.as_str())
            .bind(&input.allowed_group_ids)
            .bind(&input.denied_group_ids)
            .bind(input.min_auth_level.as_i16())
            .bind(&page_arn)
            .bind(&acl_scope_key)
            .bind(page.page_no)
            .bind(&page.raw_markdown)
            .bind(&page.raw_markdown)
            .bind(page_state)
            .bind(&thumb_uri)
            .bind(&page.content_sha256)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed inserting corpus.pages: {e}"))
            })?;

            page_ids.push(page_id);
            thumb_uris.extend(thumb_uri);

            for fig in thumb.map(|t| t.figures.as_slice()).unwrap_or_default() {
                let fig_id = Uuid::now_v7();
                let fig_arn = Arn::new(
                    &input.org_slug,
                    &input.dept_path,
                    input.access_scope,
                    ArnResourceType::Figure,
                    input.min_auth_level,
                    fig_id.to_string(),
                )?
                .to_canonical();

                let fig_key = format!("{}/{doc_id}/{}", input.org_id, fig.filename);
                let fig_uri = blob_store.put_bytes(&fig_key, &fig.png_bytes).await?;
                let bbox_arr: Option<Vec<i32>> =
                    fig.bbox_norm_1000.map(|b| b.to_array().to_vec());

                sqlx::query(
                    "INSERT INTO corpus.page_figures (
                        id, org_id, page_id, document_id, dept_id, owner_principal_id,
                        access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                        arn, acl_scope_key, figure_idx, kind, caption, bbox_norm_1000,
                        figure_uri
                     ) VALUES (
                        $1, $2, $3, $4, $5, $6,
                        $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16,
                        $17
                     )",
                )
                .bind(fig_id)
                .bind(input.org_id)
                .bind(page_id)
                .bind(doc_id)
                .bind(input.dept_id)
                .bind(input.owner_principal_id)
                .bind(input.access_scope.as_str())
                .bind(&input.allowed_group_ids)
                .bind(&input.denied_group_ids)
                .bind(input.min_auth_level.as_i16())
                .bind(&fig_arn)
                .bind(&acl_scope_key)
                .bind(fig.figure_idx)
                .bind(&fig.kind)
                .bind(&fig.caption)
                .bind(bbox_arr)
                .bind(&fig_uri)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| {
                    OpenRagError::Database(format!("failed inserting corpus.page_figures: {e}"))
                })?;

                figure_ids.push(fig_id);
                figure_uris.push(fig_uri);
            }
        }

        // Verify persisted document visibility inside the same RLS tx
        let row = sqlx::query("SELECT id FROM corpus.documents WHERE id = $1")
            .bind(doc_id)
            .fetch_optional(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed verifying document insert: {e}")))?;
        if row.is_none() {
            return Err(OpenRagError::Forbidden(
                "document inserted but hidden by Fail-Closed RLS policy".into(),
            ));
        }

        Ok(PersistedDocumentIds {
            document_id: doc_id,
            document_arn: doc_arn,
            page_ids,
            figure_ids,
            thumb_uris,
            figure_uris,
        })
    }
}
