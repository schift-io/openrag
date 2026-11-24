#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

use std::net::SocketAddr;

use clap::Parser;
use openrag_server::state::non_empty_env;
use openrag_server::{build_router, AppState};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "openrag-server", version, about = "OpenRAG v4 HTTP API Server")]
struct ServerArgs {
    #[arg(long, env = "OPENRAG_BIND")]
    bind: Option<String>,
}

fn resolve_bind_addr(cli_bind: Option<String>) -> Result<SocketAddr, anyhow::Error> {
    let raw = cli_bind
        .or_else(|| non_empty_env("OPENRAG_BIND"))
        .unwrap_or_else(|| {
            let host = non_empty_env("HOST").unwrap_or_else(|| "127.0.0.1".to_string());
            let port = non_empty_env("PORT").unwrap_or_else(|| "8080".to_string());
            format!("{host}:{port}")
        });
    raw.parse::<SocketAddr>()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{raw}': {e}"))
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let args = ServerArgs::parse();
    let bind_addr = resolve_bind_addr(args.bind)?;
    let state = AppState::from_env().await?;
    openrag_server::watch::start_from_env(&state)?;
    let router = build_router(state);

    let listener = TcpListener::bind(bind_addr).await?;
    tracing::info!(addr = %bind_addr, "openrag-server listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            let _ = sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
