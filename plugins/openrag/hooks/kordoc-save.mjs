#!/usr/bin/env node
// OpenRAG plugin hook: after kordoc writes a document, put that file into OpenRAG right away.
//
// Runs as a PostToolUse hook in Claude Code and Codex (same event JSON on stdin) after kordoc's
// writing tools: generate_document, fill_form, patch_document, convert_to_hwpx, place_seal,
// add_memo, redact_document. Saving the same path again replaces the document with a new
// revision; unchanged content is left as it is.
//
// Settings, from the Claude Code plugin options or the environment:
//   OPENRAG_URL       server address (default http://127.0.0.1:8080)
//   OPENRAG_API_KEY   API key (orag_…); without it the hook does nothing
//   OPENRAG_BUCKET    bucket to save into (default docs)
//   OPENRAG_DOCS_DIR  the server's watched documents folder, if this PC writes into it: files
//                     there are left to the folder watch so they are not saved twice
//   OPENRAG_KORDOC_AUTOSAVE  false/0/off turns the hook off (default on)
//
// The hook never blocks kordoc: any failure is reported as a warning and the tool result stays.
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const SAVED_TYPES = new Set([".hwpx", ".hwp", ".md", ".pdf", ".docx", ".pptx", ".html", ".xlsx", ".csv"]);

// Claude Code passes plugin options to hooks as CLAUDE_PLUGIN_OPTION_<KEY> (url, api_key, bucket,
// kordoc_autosave, docs_dir); an option left empty falls back to the OPENRAG_* variable (Codex).
function setting(name) {
  for (const v of [process.env[`CLAUDE_PLUGIN_OPTION_${name.replace(/^OPENRAG_/, "")}`], process.env[name]]) {
    const value = (v ?? "").trim();
    if (value !== "") return value;
  }
  return undefined;
}

function say(message, context) {
  const out = { systemMessage: message };
  if (context) out.hookSpecificOutput = { hookEventName: "PostToolUse", additionalContext: context };
  process.stdout.write(JSON.stringify(out));
}

async function readStdin() {
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

function responseText(resp) {
  if (resp == null) return "";
  if (typeof resp === "string") return resp;
  const items = Array.isArray(resp) ? resp : Array.isArray(resp.content) ? resp.content : [];
  const text = items.map((c) => (typeof c === "string" ? c : c?.text ?? "")).join("\n");
  return text || JSON.stringify(resp);
}

function failed(resp) {
  return Boolean(resp && typeof resp === "object" && !Array.isArray(resp) && resp.isError);
}

// The file kordoc wrote: paths named in its result first (they are resolved), then output_path,
// then convert_to_hwpx's default (the input next to itself as .hwpx).
function candidates(event) {
  const input = event.tool_input ?? {};
  const out = [];
  const text = responseText(event.tool_response);
  for (const m of text.matchAll(/(?:→|저장:)\s*([^\n]+?\.(?:hwpx|hwp|md|pdf|docx|pptx|html|xlsx|csv))(?=\s|$|\))/gi)) {
    out.push(m[1].trim());
  }
  if (typeof input.output_path === "string") out.push(input.output_path);
  if (String(event.tool_name ?? "").endsWith("convert_to_hwpx") && typeof input.file_path === "string") {
    out.push(input.file_path.replace(/\.[^./\\]+$/, "") + ".hwpx");
  }
  const cwd = event.cwd || process.cwd();
  return [...new Set(out.map((p) => path.resolve(cwd, p)))];
}

async function existing(paths) {
  for (const p of paths) {
    try {
      if ((await stat(p)).isFile()) return p;
    } catch {
      // not this one
    }
  }
  return undefined;
}

function inside(file, dir) {
  const rel = path.relative(path.resolve(dir), file);
  return rel !== "" && !rel.startsWith("..") && !path.isAbsolute(rel);
}

async function main() {
  const event = await readStdin();
  if (failed(event.tool_response)) return;
  if (/^(false|0|off|no)$/i.test(setting("OPENRAG_KORDOC_AUTOSAVE") ?? "")) return;
  const key = setting("OPENRAG_API_KEY");
  if (!key) return;
  const file = await existing(candidates(event));
  if (!file || !SAVED_TYPES.has(path.extname(file).toLowerCase())) return;
  const docsDir = setting("OPENRAG_DOCS_DIR");
  if (docsDir && inside(file, docsDir)) {
    say(`OpenRAG: ${path.basename(file)}은(는) 문서 폴더 안에 있어서 폴더 감시가 저장해요.`);
    return;
  }
  const base = (setting("OPENRAG_URL") ?? "http://127.0.0.1:8080").replace(/\/+$/, "");
  const bucket = setting("OPENRAG_BUCKET") ?? "docs";
  const form = new FormData();
  form.append("file", new Blob([await readFile(file)]), path.basename(file));
  // The path identifies the file: saving it again replaces the document with a new revision.
  form.append("external_key", `kordoc:${file}`);
  form.append("source_uri", `file://${file}`);
  let res;
  try {
    res = await fetch(`${base}/v2/buckets/${encodeURIComponent(bucket)}/documents`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
      body: form,
      signal: AbortSignal.timeout(110_000),
    });
  } catch (e) {
    say(`OpenRAG: ${path.basename(file)}을(를) 저장하지 못했어요 (${e?.cause?.code ?? e?.name ?? e}). ${base} 서버가 켜져 있는지 확인하세요.`);
    return;
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    say(`OpenRAG: ${path.basename(file)} 저장 실패 (${res.status} ${body.code ?? ""} ${body.message ?? ""}).`.replace(/\s+\)/, ")"));
    return;
  }
  const what = body.unchanged ? "이미 같은 내용으로 저장돼 있어요" : "저장했어요";
  say(
    `OpenRAG: ${path.basename(file)} ${what}. 버킷 ${bucket}`,
    `${file} is in OpenRAG bucket "${bucket}" as document ${body.document_id} ("${body.title ?? path.basename(file)}"); rag_search can find it now.`,
  );
}

main().catch((e) => say(`OpenRAG: kordoc 저장 훅 오류: ${e?.message ?? e}`));
