//! LLM providers backed by the Claude Code (`claude_cli`) or Codex (`codex_cli`) CLI on the server
//! host. Each completion runs the CLI once in a fresh empty directory, prompt on stdin.
//!
//! The CLI must already be logged in for the user the server runs as. Claude Code runs with no
//! tools and no MCP servers; Codex runs in its read-only sandbox (it has no switch that turns its
//! tools off).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use openrag_core::OpenRagError;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use uuid::Uuid;

const DEFAULT_TIMEOUT_SECS: u64 = 180;

fn adapter_err(provider_type: &str, message: impl Into<String>) -> OpenRagError {
    OpenRagError::ExternalAdapter {
        adapter: provider_type.to_string(),
        message: message.into(),
    }
}

/// Command-line arguments for one completion. Codex writes its final message to `last_message`.
pub fn cli_args(
    provider_type: &str,
    model_id: &str,
    system: &str,
    last_message: &Path,
) -> Result<Vec<String>, OpenRagError> {
    let model = model_id.trim();
    let mut args: Vec<String> = match provider_type {
        "claude_cli" => [
            "-p",
            "--output-format",
            "text",
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt",
            system,
        ]
        .map(String::from)
        .to_vec(),
        "codex_cli" => vec![
            "exec".into(),
            "--sandbox".into(),
            "read-only".into(),
            "--skip-git-repo-check".into(),
            "--ephemeral".into(),
            "--color".into(),
            "never".into(),
            "-o".into(),
            last_message.to_string_lossy().into_owned(),
        ],
        other => return Err(adapter_err(other, "not a CLI provider type")),
    };
    if !model.is_empty() {
        args.push(
            if provider_type == "claude_cli" {
                "--model"
            } else {
                "-m"
            }
            .into(),
        );
        args.push(model.into());
    }
    if provider_type == "codex_cli" {
        // read the prompt from stdin
        args.push("-".into());
    }
    Ok(args)
}

/// What goes to the CLI's stdin. Codex takes no separate system prompt, so it is prepended.
pub fn cli_stdin(provider_type: &str, system: &str, prompt: &str) -> String {
    if provider_type == "codex_cli" {
        format!("{system}\n\nQuestion:\n{prompt}")
    } else {
        prompt.to_string()
    }
}

/// Run one completion through the CLI named by `provider_type`.
///
/// `cfg_json` may set `command` (binary path; default `claude` / `codex`) and `timeout_secs`.
pub async fn run_cli_completion(
    provider_type: &str,
    model_id: &str,
    cfg_json: &Value,
    system: &str,
    prompt: &str,
) -> Result<String, OpenRagError> {
    let default_command = match provider_type {
        "claude_cli" => "claude",
        "codex_cli" => "codex",
        other => return Err(adapter_err(other, "not a CLI provider type")),
    };
    let command = cfg_json
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_command)
        .to_string();
    let timeout = Duration::from_secs(
        cfg_json
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .filter(|s| *s > 0)
            .unwrap_or(DEFAULT_TIMEOUT_SECS),
    );

    let workdir: PathBuf = std::env::temp_dir().join(format!("openrag-llm-{}", Uuid::new_v4()));
    tokio::fs::create_dir(&workdir).await.map_err(|e| {
        adapter_err(
            provider_type,
            format!("failed to create {}: {e}", workdir.display()),
        )
    })?;
    let result = run_in(
        provider_type,
        &command,
        model_id,
        system,
        prompt,
        &workdir,
        timeout,
    )
    .await;
    if let Err(e) = tokio::fs::remove_dir_all(&workdir).await {
        tracing::warn!("failed to clean up {}: {e}", workdir.display());
    }
    result
}

async fn run_in(
    provider_type: &str,
    command: &str,
    model_id: &str,
    system: &str,
    prompt: &str,
    workdir: &Path,
    timeout: Duration,
) -> Result<String, OpenRagError> {
    let last_message = workdir.join("last_message.txt");
    let args = cli_args(provider_type, model_id, system, &last_message)?;
    let mut child = Command::new(command)
        .args(&args)
        .current_dir(workdir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| adapter_err(provider_type, format!("failed to start '{command}': {e}")))?;

    let input = cli_stdin(provider_type, system, prompt);
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(input.as_bytes())
            .await
            .map_err(|e| adapter_err(provider_type, format!("failed to write prompt: {e}")))?;
    }

    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| {
            adapter_err(
                provider_type,
                format!("'{command}' did not finish within {}s", timeout.as_secs()),
            )
        })?
        .map_err(|e| adapter_err(provider_type, format!("'{command}' failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .trim()
            .chars()
            .rev()
            .take(400)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(adapter_err(
            provider_type,
            format!("'{command}' exited with {}: {tail}", output.status),
        ));
    }

    let answer = if provider_type == "codex_cli" {
        tokio::fs::read_to_string(&last_message)
            .await
            .unwrap_or_default()
    } else {
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let answer = answer.trim();
    if answer.is_empty() {
        return Err(adapter_err(
            provider_type,
            format!("'{command}' returned no answer"),
        ));
    }
    Ok(answer.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    /// A stand-in CLI: records its arguments and stdin next to itself, then answers like the
    /// real one (Claude on stdout, Codex into the `-o` file).
    fn fake_cli(dir: &Path, name: &str, body: &str) -> anyhow::Result<PathBuf> {
        let path = dir.join(name);
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"{rec}/args\"\ncat > \"{rec}/stdin\"\n{body}\n",
                rec = dir.display()
            ),
        )?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
        Ok(path)
    }

    fn expect_adapter_err(res: Result<String, OpenRagError>, needle: &str) -> anyhow::Result<()> {
        match res {
            Err(OpenRagError::ExternalAdapter { message, .. }) if message.contains(needle) => {
                Ok(())
            }
            other => anyhow::bail!("expected ExternalAdapter containing {needle:?}, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn claude_cli_gets_no_tools_and_reads_the_answer_from_stdout() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let cli = fake_cli(dir.path(), "claude", "echo '  연차는 15일입니다.  '")?;
        let cfg = json!({ "command": cli.to_string_lossy() });
        let answer =
            run_cli_completion("claude_cli", "sonnet", &cfg, "SYS ctx", "연차 며칠?").await?;
        assert_eq!(answer, "연차는 15일입니다.");
        let args = std::fs::read_to_string(dir.path().join("args"))?;
        let args: Vec<&str> = args.lines().collect();
        assert_eq!(
            args,
            [
                "-p",
                "--output-format",
                "text",
                "--tools",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--system-prompt",
                "SYS ctx",
                "--model",
                "sonnet"
            ]
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("stdin"))?,
            "연차 며칠?"
        );
        Ok(())
    }

    #[tokio::test]
    async fn codex_cli_reads_the_last_message_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        // the path after -o is the 9th argument; stdout carries only progress
        let cli = fake_cli(
            dir.path(),
            "codex",
            "echo progress; echo '출장비는 12만 원' > \"$9\"",
        )?;
        let cfg = json!({ "command": cli.to_string_lossy() });
        let answer = run_cli_completion("codex_cli", "", &cfg, "SYS ctx", "호텔비?").await?;
        assert_eq!(answer, "출장비는 12만 원");
        let args = std::fs::read_to_string(dir.path().join("args"))?;
        let args: Vec<&str> = args.lines().collect();
        assert_eq!(
            &args[..8],
            [
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "-o"
            ]
        );
        assert!(args[8].ends_with("last_message.txt"));
        assert_eq!(args[9], "-");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("stdin"))?,
            "SYS ctx\n\nQuestion:\n호텔비?"
        );
        Ok(())
    }

    #[tokio::test]
    async fn failures_surface_as_external_adapter_errors() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let failing = fake_cli(dir.path(), "failing", "echo 'not logged in' >&2; exit 3")?;
        let cfg = json!({ "command": failing.to_string_lossy() });
        expect_adapter_err(
            run_cli_completion("claude_cli", "", &cfg, "s", "q").await,
            "not logged in",
        )?;

        let cfg = json!({ "command": "/nonexistent/codex" });
        expect_adapter_err(
            run_cli_completion("codex_cli", "", &cfg, "s", "q").await,
            "failed to start",
        )?;

        let slow = fake_cli(dir.path(), "slow", "sleep 5; echo late")?;
        let cfg = json!({ "command": slow.to_string_lossy(), "timeout_secs": 1 });
        expect_adapter_err(
            run_cli_completion("claude_cli", "", &cfg, "s", "q").await,
            "did not finish",
        )?;
        Ok(())
    }
}
