use openrag_core::OpenRagError;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub has_pg_pinecone: bool,
    pub has_pg_search: bool,
    pub has_pgvector: bool,
    pub has_pg_trgm: bool,
    pub pg_version_num: i32,
}

pub async fn detect_capabilities(pool: &PgPool) -> Result<Capabilities, OpenRagError> {
    let ext_rows = sqlx::query(
        "SELECT extname FROM pg_extension WHERE extname IN ('pg_pinecone', 'pg_search', 'vector', 'pg_trgm')",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed to query pg_extension: {e}")))?;

    let mut caps = Capabilities {
        has_pg_pinecone: false,
        has_pg_search: false,
        has_pgvector: false,
        has_pg_trgm: false,
        pg_version_num: 0,
    };

    for row in ext_rows {
        let name: String = row
            .try_get("extname")
            .map_err(|e| OpenRagError::Database(format!("invalid extname row: {e}")))?;
        match name.as_str() {
            "pg_pinecone" => caps.has_pg_pinecone = true,
            "pg_search" => caps.has_pg_search = true,
            "vector" => caps.has_pgvector = true,
            "pg_trgm" => caps.has_pg_trgm = true,
            _ => {}
        }
    }

    let ver_str: String = sqlx::query_scalar("SHOW server_version_num")
        .fetch_one(pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to read server_version_num: {e}")))?;

    caps.pg_version_num = ver_str.trim().parse::<i32>().map_err(|e| {
        OpenRagError::Database(format!("invalid server_version_num '{ver_str}': {e}"))
    })?;

    Ok(caps)
}
