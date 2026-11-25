//! Embedded Agent Skills and MCP Resources (`io.modelcontextprotocol/skills`).
//!
//! To ship an additional skill with the server, drop its directory under
//! `crates/openrag-server/skills/<name>/` (with a `SKILL.md` at the root) and add one
//! entry to [`EMBEDDED_SKILLS`]. Digests, byte lengths, and YAML frontmatter are derived
//! lazily from the embedded bytes.

use std::sync::OnceLock;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

struct EmbeddedSkillFile {
    /// Path relative to the skill root (e.g. `"SKILL.md"` or `"references/guide.md"`).
    rel_path: &'static str,
    mime_type: &'static str,
    content: &'static str,
}

struct EmbeddedSkill {
    /// Skill path under `skill://` (its final segment must match frontmatter `name`).
    skill_path: &'static str,
    files: &'static [EmbeddedSkillFile],
}

/// All skills compiled into the binary.
const EMBEDDED_SKILLS: &[EmbeddedSkill] = &[EmbeddedSkill {
    skill_path: "openrag-search",
    files: &[EmbeddedSkillFile {
        rel_path: "SKILL.md",
        mime_type: "text/markdown",
        content: include_str!("../skills/openrag-search/SKILL.md"),
    }],
}];

#[derive(Debug, Clone)]
struct LoadedResource {
    uri: String,
    name: String,
    description: Option<String>,
    mime_type: &'static str,
    text: &'static str,
    digest: String,
    size: usize,
}

#[derive(Debug, Clone)]
struct LoadedSkill {
    uri: String,
    frontmatter: Value,
    resources: Vec<LoadedResource>,
}

#[derive(Debug)]
struct SkillCatalog {
    skills: Vec<LoadedSkill>,
}

fn catalog() -> Result<&'static SkillCatalog, String> {
    static CATALOG: OnceLock<Result<SkillCatalog, String>> = OnceLock::new();
    match CATALOG.get_or_init(build_catalog) {
        Ok(cat) => Ok(cat),
        Err(msg) => Err(msg.clone()),
    }
}

fn build_catalog() -> Result<SkillCatalog, String> {
    let mut skills = Vec::with_capacity(EMBEDDED_SKILLS.len());
    for embedded in EMBEDDED_SKILLS {
        let skill_md = embedded
            .files
            .iter()
            .find(|f| f.rel_path == "SKILL.md")
            .ok_or_else(|| format!("embedded skill '{}' is missing SKILL.md", embedded.skill_path))?;
        let frontmatter = parse_frontmatter(skill_md.content)
            .map_err(|e| format!("embedded skill '{}' frontmatter: {e}", embedded.skill_path))?;

        let fm_name = frontmatter
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                format!(
                    "embedded skill '{}' frontmatter missing string 'name'",
                    embedded.skill_path
                )
            })?;
        let fm_desc = frontmatter
            .get("description")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                format!(
                    "embedded skill '{}' frontmatter missing string 'description'",
                    embedded.skill_path
                )
            })?;

        let mut resources = Vec::with_capacity(embedded.files.len());
        for file in embedded.files {
            let uri = format!("skill://{}/{}", embedded.skill_path, file.rel_path);
            let bytes = file.content.as_bytes();
            let digest = format!("sha256:{}", hex::encode(Sha256::digest(bytes)));
            let is_root_skill_md = file.rel_path == "SKILL.md";
            let name = if is_root_skill_md {
                fm_name.to_string()
            } else {
                file.rel_path
                    .rsplit('/')
                    .next()
                    .unwrap_or(file.rel_path)
                    .to_string()
            };
            let description = is_root_skill_md.then(|| fm_desc.to_string());
            resources.push(LoadedResource {
                uri,
                name,
                description,
                mime_type: file.mime_type,
                text: file.content,
                digest,
                size: bytes.len(),
            });
        }

        skills.push(LoadedSkill {
            uri: format!("skill://{}/SKILL.md", embedded.skill_path),
            frontmatter,
            resources,
        });
    }
    Ok(SkillCatalog { skills })
}

/// Extracts and parses the leading YAML frontmatter block (`---\n...\n---`) of a `SKILL.md`.
fn parse_frontmatter(markdown: &str) -> Result<Value, String> {
    let rest = markdown
        .strip_prefix("---\r\n")
        .or_else(|| markdown.strip_prefix("---\n"))
        .ok_or_else(|| "file must start with '---' frontmatter delimiter".to_string())?;

    let mut offset = 0usize;
    let mut end_offset = None;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            end_offset = Some(offset);
            break;
        }
        offset += line.len();
    }
    let end = end_offset.ok_or_else(|| "unterminated YAML frontmatter".to_string())?;
    let yaml_block = &rest[..end];
    let parsed: Value =
        serde_yaml_ng::from_str(yaml_block).map_err(|e| format!("invalid YAML: {e}"))?;
    if !parsed.is_object() {
        return Err("frontmatter must be a YAML mapping".to_string());
    }
    Ok(parsed)
}

fn skill_to_json(skill: &LoadedSkill) -> Value {
    let manifest: Vec<Value> = skill
        .resources
        .iter()
        .map(|r| {
            json!({
                "uri": r.uri,
                "digest": r.digest,
                "size": r.size,
            })
        })
        .collect();
    json!({
        "uri": skill.uri,
        "frontmatter": skill.frontmatter,
        "resources": manifest,
    })
}

fn resource_to_json(res: &LoadedResource) -> Value {
    let mut out = json!({
        "uri": res.uri,
        "name": res.name,
        "mimeType": res.mime_type,
        "size": res.size,
    });
    if let (Some(obj), Some(desc)) = (out.as_object_mut(), &res.description) {
        obj.insert("description".into(), json!(desc));
    }
    out
}

pub(crate) fn list_skills() -> Result<Value, String> {
    let cat = catalog()?;
    let skills: Vec<Value> = cat.skills.iter().map(skill_to_json).collect();
    Ok(json!({ "skills": skills }))
}

pub(crate) fn get_skill(uri: &str) -> Result<Option<Value>, String> {
    let cat = catalog()?;
    Ok(cat
        .skills
        .iter()
        .find(|s| s.uri == uri)
        .map(|s| json!({ "skill": skill_to_json(s) })))
}

pub(crate) fn list_resources() -> Result<Value, String> {
    let cat = catalog()?;
    let resources: Vec<Value> = cat
        .skills
        .iter()
        .flat_map(|s| s.resources.iter())
        .map(resource_to_json)
        .collect();
    Ok(json!({ "resources": resources }))
}

pub(crate) fn list_resource_templates() -> Value {
    json!({ "resourceTemplates": [] })
}

pub(crate) fn read_resource(uri: &str) -> Result<Option<Value>, String> {
    let cat = catalog()?;
    let found = cat
        .skills
        .iter()
        .flat_map(|s| s.resources.iter())
        .find(|r| r.uri == uri);
    Ok(found.map(|r| {
        json!({
            "contents": [{
                "uri": r.uri,
                "mimeType": r.mime_type,
                "text": r.text,
            }]
        })
    }))
}
