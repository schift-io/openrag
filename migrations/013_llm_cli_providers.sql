-- OpenRAG Migration 013: LLM providers backed by the Claude Code or Codex CLI on the server host.
--
-- claude_cli / codex_cli run `claude -p` / `codex exec` as a subprocess instead of calling an HTTP
-- API. Both send the prompt to Anthropic / OpenAI, so they are seeded as external providers.
--
-- Idempotent.

ALTER TABLE gateway.providers DROP CONSTRAINT IF EXISTS providers_provider_type_check;
ALTER TABLE gateway.providers ADD CONSTRAINT providers_provider_type_check CHECK (
    provider_type IN ('openrouter', 'openai', 'vllm_local', 'schift_ocr', 'mock', 'claude_cli', 'codex_cli')
);
