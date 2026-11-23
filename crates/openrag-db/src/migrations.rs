use std::path::Path;

use chrono::{DateTime, Utc};
use openrag_core::OpenRagError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigrationSpec {
    pub version: &'static str,
    pub filename: &'static str,
    pub sql: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationRecord {
    pub version: String,
    pub filename: String,
    pub checksum_sha256: String,
    pub applied_at: DateTime<Utc>,
}

pub const EMBEDDED_MIGRATIONS: &[MigrationSpec] = &[
    MigrationSpec {
        version: "000",
        filename: "000_init.sql",
        sql: include_str!("../../../migrations/000_init.sql"),
    },
    MigrationSpec {
        version: "001",
        filename: "001_iam_arn.sql",
        sql: include_str!("../../../migrations/001_iam_arn.sql"),
    },
    MigrationSpec {
        version: "002",
        filename: "002_vectors_corpus.sql",
        sql: include_str!("../../../migrations/002_vectors_corpus.sql"),
    },
    MigrationSpec {
        version: "003",
        filename: "003_pii_capsule.sql",
        sql: include_str!("../../../migrations/003_pii_capsule.sql"),
    },
    MigrationSpec {
        version: "004",
        filename: "004_llm_wiki.sql",
        sql: include_str!("../../../migrations/004_llm_wiki.sql"),
    },
    MigrationSpec {
        version: "005",
        filename: "005_gateway_plugins.sql",
        sql: include_str!("../../../migrations/005_gateway_plugins.sql"),
    },
    MigrationSpec {
        version: "006",
        filename: "006_eval_bench.sql",
        sql: include_str!("../../../migrations/006_eval_bench.sql"),
    },
    MigrationSpec {
        version: "007",
        filename: "007_memory_redaction_indexing.sql",
        sql: include_str!("../../../migrations/007_memory_redaction_indexing.sql"),
    },
    MigrationSpec {
        version: "008",
        filename: "008_concepts_tags.sql",
        sql: include_str!("../../../migrations/008_concepts_tags.sql"),
    },
    MigrationSpec {
        version: "009",
        filename: "009_cli_auth_codes.sql",
        sql: include_str!("../../../migrations/009_cli_auth_codes.sql"),
    },
    MigrationSpec {
        version: "010",
        filename: "010_guard_sk_key_pattern.sql",
        sql: include_str!("../../../migrations/010_guard_sk_key_pattern.sql"),
    },
    MigrationSpec {
        version: "011",
        filename: "011_vector_dim_256.sql",
        sql: include_str!("../../../migrations/011_vector_dim_256.sql"),
    },
    MigrationSpec {
        version: "012",
        filename: "012_drop_page_figure_vectors.sql",
        sql: include_str!("../../../migrations/012_drop_page_figure_vectors.sql"),
    },
];

pub fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

pub async fn run_migrations(pool: &PgPool) -> Result<Vec<MigrationRecord>, OpenRagError> {
    let items: Vec<(String, String, String)> = EMBEDDED_MIGRATIONS
        .iter()
        .map(|m| (m.version.to_string(), m.filename.to_string(), m.sql.to_string()))
        .collect();
    apply_migration_items(pool, &items).await
}

pub async fn run_migrations_from_dir(
    pool: &PgPool,
    dir: &Path,
) -> Result<Vec<MigrationRecord>, OpenRagError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        OpenRagError::Config(format!(
            "failed to read migrations directory '{}': {e}",
            dir.display()
        ))
    })?;

    let mut files: Vec<(String, String)> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            OpenRagError::Config(format!("failed to inspect migration entry: {e}"))
        })?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("sql") {
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            let version = name.split('_').next().unwrap_or(name).to_string();
            let sql = std::fs::read_to_string(&path).map_err(|e| {
                OpenRagError::Config(format!(
                    "failed to read migration '{}': {e}",
                    path.display()
                ))
            })?;
            files.push((name.to_string(), format!("{version}\0{sql}")));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let items: Vec<(String, String, String)> = files
        .into_iter()
        .filter_map(|(filename, packed)| {
            let (version, sql) = packed.split_once('\0')?;
            Some((version.to_string(), filename, sql.to_string()))
        })
        .collect();

    apply_migration_items(pool, &items).await
}

async fn apply_migration_items(
    pool: &PgPool,
    items: &[(String, String, String)],
) -> Result<Vec<MigrationRecord>, OpenRagError> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS public.openrag_schema_migrations (
            version         TEXT PRIMARY KEY,
            filename        TEXT NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            applied_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed to create schema_migrations table: {e}")))?;

    for (version, filename, sql) in items {
        let checksum = sha256_hex(sql);
        let existing = sqlx::query(
            "SELECT checksum_sha256 FROM public.openrag_schema_migrations WHERE version = $1",
        )
        .bind(version)
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!(
                "failed to check migration status for '{filename}': {e}"
            ))
        })?;

        if let Some(row) = existing {
            let recorded_checksum: String = row.try_get("checksum_sha256").map_err(|e| {
                OpenRagError::Database(format!("invalid checksum row for '{filename}': {e}"))
            })?;
            if recorded_checksum != checksum {
                return Err(OpenRagError::Conflict(format!(
                    "migration '{filename}' (version {version}) checksum mismatch: expected {recorded_checksum}, got {checksum}"
                )));
            }
            continue;
        }

        let mut tx = pool.begin().await.map_err(|e| {
            OpenRagError::Database(format!(
                "failed to start migration transaction for '{filename}': {e}"
            ))
        })?;

        sqlx::raw_sql(sql)
            .execute(&mut *tx)
            .await
            .map_err(|e| OpenRagError::Database(format!("migration '{filename}' failed: {e}")))?;

        sqlx::query(
            "INSERT INTO public.openrag_schema_migrations (version, filename, checksum_sha256)
             VALUES ($1, $2, $3)",
        )
        .bind(version)
        .bind(filename)
        .bind(&checksum)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!(
                "failed to record migration '{filename}' in schema_migrations: {e}"
            ))
        })?;

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!(
                "failed to commit migration '{filename}': {e}"
            ))
        })?;
    }

    let rows = sqlx::query(
        "SELECT version, filename, checksum_sha256, applied_at
         FROM public.openrag_schema_migrations
         ORDER BY version ASC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed to list applied migrations: {e}")))?;

    let mut records = Vec::with_capacity(rows.len());
    for row in rows {
        records.push(MigrationRecord {
            version: row
                .try_get("version")
                .map_err(|e| OpenRagError::Database(format!("invalid version column: {e}")))?,
            filename: row
                .try_get("filename")
                .map_err(|e| OpenRagError::Database(format!("invalid filename column: {e}")))?,
            checksum_sha256: row
                .try_get("checksum_sha256")
                .map_err(|e| OpenRagError::Database(format!("invalid checksum column: {e}")))?,
            applied_at: row
                .try_get("applied_at")
                .map_err(|e| OpenRagError::Database(format!("invalid applied_at column: {e}")))?,
        });
    }

    Ok(records)
}
