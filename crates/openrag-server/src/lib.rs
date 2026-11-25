#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub mod api_app;
pub mod api_core;
pub mod api_docs;
pub mod api_members;
pub mod api_studio;
pub mod auth;
pub mod chunking;
pub mod egress;
pub mod cli;
pub mod error;
pub mod llm_cli;
pub mod llm_router;
pub mod mcp;
pub mod state;
pub mod text_embed;
pub mod watch;

pub use state::{AppState, OpenRagMode};

const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

/// REST surface (`/health`, `/v1/*`, `/v2/*`, `/api/app/*`, `/api/studio/*`) without
/// transport layers. MCP tools and the CLI dispatch into this router in-process.
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(api_core::core_routes())
        .merge(api_docs::docs_routes())
        .merge(api_members::member_routes())
        .merge(api_app::app_routes())
        .merge(api_studio::studio_routes())
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
}

pub fn build_router(state: Arc<AppState>) -> Router {
    api_router()
        .merge(mcp::mcp_routes())
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
