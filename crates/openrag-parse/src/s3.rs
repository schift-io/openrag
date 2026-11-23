//! S3-compatible blob store for page thumbnails and figure crops: MinIO (or another S3 server)
//! next to OpenRAG, or AWS S3 and other hosted buckets. URIs are `s3://<bucket>/<key>`.

use std::time::Duration;

use openrag_core::OpenRagError;
use reqwest::{StatusCode, Url};
use rusty_s3::{Bucket, Credentials, S3Action, UrlStyle};

use crate::thumbnail::ThumbnailBlobStore;

const SIGN_FOR: Duration = Duration::from_secs(300);

#[derive(Debug, Clone)]
pub struct S3BlobConfig {
    /// `http://minio:9000`, `https://s3.ap-northeast-2.amazonaws.com`, ...
    pub endpoint: String,
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    /// Path-style URLs (`endpoint/bucket/key`), which MinIO and most S3-compatible servers expect.
    /// `false` uses virtual-hosted URLs (`bucket.endpoint/key`).
    pub path_style: bool,
}

#[derive(Debug, Clone)]
pub struct S3BlobStore {
    bucket: Bucket,
    credentials: Credentials,
    http: reqwest::Client,
}

fn s3_err(message: impl Into<String>) -> OpenRagError {
    OpenRagError::ExternalAdapter {
        adapter: "s3".into(),
        message: message.into(),
    }
}

fn content_type(key: &str) -> &'static str {
    match key.rsplit('.').next() {
        Some("webp") => "image/webp",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        _ => "application/octet-stream",
    }
}

impl S3BlobStore {
    pub fn new(cfg: S3BlobConfig) -> Result<Self, OpenRagError> {
        let endpoint: Url = cfg.endpoint.trim().parse().map_err(|e| {
            OpenRagError::Validation(format!("invalid S3 endpoint '{}': {e}", cfg.endpoint))
        })?;
        let style = if cfg.path_style {
            UrlStyle::Path
        } else {
            UrlStyle::VirtualHost
        };
        let bucket =
            Bucket::new(endpoint, style, cfg.bucket.clone(), cfg.region.clone()).map_err(|e| {
                OpenRagError::Validation(format!("invalid S3 bucket '{}': {e}", cfg.bucket))
            })?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| s3_err(format!("http client: {e}")))?;
        Ok(Self {
            bucket,
            credentials: Credentials::new(cfg.access_key_id, cfg.secret_access_key),
            http,
        })
    }

    pub fn bucket_name(&self) -> &str {
        self.bucket.name()
    }

    /// Create the bucket when it does not exist yet.
    pub async fn ensure_bucket(&self) -> Result<(), OpenRagError> {
        let head = self
            .bucket
            .head_bucket(Some(&self.credentials))
            .sign(SIGN_FOR);
        let resp = self
            .http
            .head(head)
            .send()
            .await
            .map_err(|e| s3_err(format!("HEAD bucket '{}': {e}", self.bucket.name())))?;
        if resp.status().is_success() {
            return Ok(());
        }
        if resp.status() != StatusCode::NOT_FOUND {
            return Err(s3_err(format!(
                "HEAD bucket '{}': HTTP {}",
                self.bucket.name(),
                resp.status()
            )));
        }
        let region = self.bucket.region();
        let body = if region == "us-east-1" {
            String::new()
        } else {
            format!(
                "<CreateBucketConfiguration xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><LocationConstraint>{region}</LocationConstraint></CreateBucketConfiguration>"
            )
        };
        let create = self.bucket.create_bucket(&self.credentials).sign(SIGN_FOR);
        let resp = self
            .http
            .put(create)
            .body(body)
            .send()
            .await
            .map_err(|e| s3_err(format!("create bucket '{}': {e}", self.bucket.name())))?;
        if resp.status().is_success() {
            return Ok(());
        }
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Err(s3_err(format!(
            "create bucket '{}': HTTP {status}: {}",
            self.bucket.name(),
            text.chars().take(300).collect::<String>()
        )))
    }

    fn uri_for(&self, key: &str) -> String {
        format!("s3://{}/{key}", self.bucket.name())
    }

    fn key_of<'u>(&self, uri: &'u str) -> Result<&'u str, OpenRagError> {
        uri.strip_prefix("s3://")
            .and_then(|rest| rest.strip_prefix(self.bucket.name()))
            .and_then(|rest| rest.strip_prefix('/'))
            .filter(|key| !key.is_empty())
            .ok_or_else(|| {
                OpenRagError::Validation(format!(
                    "'{uri}' is not an object in bucket '{}'",
                    self.bucket.name()
                ))
            })
    }

    async fn put_key(&self, key: &str, bytes: &[u8]) -> Result<(), OpenRagError> {
        let ct = content_type(key);
        let mut action = self.bucket.put_object(Some(&self.credentials), key);
        action.headers_mut().insert("content-type", ct);
        let url = action.sign(SIGN_FOR);
        let resp = self
            .http
            .put(url)
            .header("content-type", ct)
            .body(bytes.to_vec())
            .send()
            .await
            .map_err(|e| s3_err(format!("PUT '{key}': {e}")))?;
        if resp.status().is_success() {
            return Ok(());
        }
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Err(s3_err(format!(
            "PUT '{key}': HTTP {status}: {}",
            text.chars().take(300).collect::<String>()
        )))
    }
}

#[async_trait::async_trait]
impl ThumbnailBlobStore for S3BlobStore {
    async fn put_bytes(&self, key: &str, bytes: &[u8]) -> Result<String, OpenRagError> {
        let clean = key.trim_start_matches('/').replace("..", "_");
        self.put_key(&clean, bytes).await?;
        Ok(self.uri_for(&clean))
    }

    async fn put_uri(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError> {
        let key = self.key_of(uri)?;
        self.put_key(key, bytes).await
    }

    async fn get_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError> {
        let key = self.key_of(uri)?;
        let url = self
            .bucket
            .get_object(Some(&self.credentials), key)
            .sign(SIGN_FOR);
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| s3_err(format!("GET '{key}': {e}")))?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Err(OpenRagError::NotFound(format!("blob '{uri}' not found")));
        }
        if !resp.status().is_success() {
            return Err(s3_err(format!("GET '{key}': HTTP {}", resp.status())));
        }
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| s3_err(format!("GET '{key}' body: {e}")))
    }
}
