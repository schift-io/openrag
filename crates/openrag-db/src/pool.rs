use std::path::Path;

use openrag_core::{OpenRagError, PrincipalContext};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::capabilities::{detect_capabilities, Capabilities};
use crate::migrations::{run_migrations, run_migrations_from_dir, MigrationRecord};
use crate::rls::{begin_rls_tx, RlsTransaction};

#[derive(Clone, Debug)]
pub struct DbPool {
    pool: PgPool,
}

impl DbPool {
    pub async fn connect(database_url: &str, max_connections: u32) -> Result<Self, OpenRagError> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(database_url)
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to connect to PostgreSQL: {e}")))?;
        Ok(Self { pool })
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn inner(&self) -> &PgPool {
        &self.pool
    }

    pub async fn health_check(&self) -> Result<(), OpenRagError> {
        sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| OpenRagError::Database(format!("database health check failed: {e}")))?;
        Ok(())
    }

    pub async fn capabilities(&self) -> Result<Capabilities, OpenRagError> {
        detect_capabilities(&self.pool).await
    }

    pub async fn run_migrations(&self) -> Result<Vec<MigrationRecord>, OpenRagError> {
        run_migrations(&self.pool).await
    }

    pub async fn run_migrations_from_dir(
        &self,
        dir: &Path,
    ) -> Result<Vec<MigrationRecord>, OpenRagError> {
        run_migrations_from_dir(&self.pool, dir).await
    }

    pub async fn begin_rls_tx<'a>(
        &'a self,
        ctx: &PrincipalContext,
    ) -> Result<RlsTransaction<'a>, OpenRagError> {
        begin_rls_tx(&self.pool, ctx).await
    }
}
