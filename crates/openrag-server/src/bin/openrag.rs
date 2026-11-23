#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

fn main() -> std::process::ExitCode {
    openrag_server::cli::main_entry()
}
