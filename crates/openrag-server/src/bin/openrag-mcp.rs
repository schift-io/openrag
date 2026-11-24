#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

//! Standalone stdio MCP server: equivalent to `openrag mcp --stdio`.

use std::process::ExitCode;

use clap::Parser;
use openrag_server::mcp::{run_stdio, StdioOptions};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "openrag-mcp", version, about = "OpenRAG local MCP server (16 tools) over stdio")]
struct McpBinArgs {
    /// Accepted for parity with `openrag mcp --stdio` (stdio is the only transport here)
    #[arg(long)]
    stdio: bool,
    /// API key identifying the caller (default OPENRAG_API_KEY)
    #[arg(long, env = "OPENRAG_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Expected principal id; calls fail if the key belongs to someone else
    #[arg(long)]
    principal: Option<Uuid>,
}

fn main() -> ExitCode {
    if let Err(e) = openrag_server::cli::apply_config_file() {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let args = McpBinArgs::parse();
    let _ = args.stdio;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to start async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run_stdio(StdioOptions {
        api_key: args.api_key,
        principal_id: args.principal,
    })) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
