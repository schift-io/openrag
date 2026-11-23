use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::access::{AccessScope, AuthLevel};
use crate::error::OpenRagError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArnResourceType {
    Bucket,
    Document,
    Page,
    Figure,
    Chunk,
    Memory,
    Episode,
    Wiki,
    Vault,
    Policy,
}

impl ArnResourceType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bucket => "bucket",
            Self::Document => "document",
            Self::Page => "page",
            Self::Figure => "figure",
            Self::Chunk => "chunk",
            Self::Memory => "memory",
            Self::Episode => "episode",
            Self::Wiki => "wiki",
            Self::Vault => "vault",
            Self::Policy => "policy",
        }
    }
}

impl fmt::Display for ArnResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ArnResourceType {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bucket" => Ok(Self::Bucket),
            "document" => Ok(Self::Document),
            "page" => Ok(Self::Page),
            "figure" => Ok(Self::Figure),
            "chunk" => Ok(Self::Chunk),
            "memory" => Ok(Self::Memory),
            "episode" => Ok(Self::Episode),
            "wiki" => Ok(Self::Wiki),
            "vault" => Ok(Self::Vault),
            "policy" => Ok(Self::Policy),
            other => Err(OpenRagError::Validation(format!(
                "invalid ARN resource_type '{other}'"
            ))),
        }
    }
}

/// Canonical OpenRAG ARN:
/// `arn:openrag:<org>:<dept_path>:<scope>:<type>:L<1..10>/<id>`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Arn {
    pub org: String,
    pub dept_path: String,
    pub scope: AccessScope,
    pub resource_type: ArnResourceType,
    pub auth_level: AuthLevel,
    pub resource_id: String,
}

impl Arn {
    pub fn new(
        org: impl Into<String>,
        dept_path: impl Into<String>,
        scope: AccessScope,
        resource_type: ArnResourceType,
        auth_level: AuthLevel,
        resource_id: impl Into<String>,
    ) -> Result<Self, OpenRagError> {
        let org = org.into();
        let dept_path = dept_path.into();
        let resource_id = resource_id.into();

        if org.trim().is_empty() || org.contains(':') {
            return Err(OpenRagError::Validation("invalid ARN org segment".into()));
        }
        if dept_path.trim().is_empty() || dept_path.contains(':') {
            return Err(OpenRagError::Validation(
                "invalid ARN dept_path segment".into(),
            ));
        }
        if resource_id.trim().is_empty() || resource_id.contains(':') {
            return Err(OpenRagError::Validation(
                "invalid ARN resource_id segment".into(),
            ));
        }

        Ok(Self {
            org,
            dept_path,
            scope,
            resource_type,
            auth_level,
            resource_id,
        })
    }

    pub fn parse(raw: &str) -> Result<Self, OpenRagError> {
        let parts: Vec<&str> = raw.split(':').collect();
        if parts.len() != 7 || parts[0] != "arn" || parts[1] != "openrag" {
            return Err(OpenRagError::Validation(format!(
                "invalid OpenRAG ARN format '{raw}'; expected arn:openrag:<org>:<dept_path>:<scope>:<type>:L<n>/<id>"
            )));
        }

        let org = parts[2];
        let dept_path = parts[3];
        let scope = AccessScope::from_str(parts[4])?;
        let resource_type = ArnResourceType::from_str(parts[5])?;

        let (level_str, resource_id) = parts[6].split_once('/').ok_or_else(|| {
            OpenRagError::Validation(format!(
                "invalid ARN level/id suffix '{}'; expected L<n>/<id>",
                parts[6]
            ))
        })?;

        let level_digits = level_str.strip_prefix('L').ok_or_else(|| {
            OpenRagError::Validation(format!(
                "invalid ARN level prefix '{level_str}'; expected L<1..10>"
            ))
        })?;
        let level_num: u8 = level_digits.parse().map_err(|_| {
            OpenRagError::Validation(format!("invalid ARN auth_level '{level_str}'"))
        })?;
        let auth_level = AuthLevel::new(level_num)?;

        Self::new(
            org,
            dept_path,
            scope,
            resource_type,
            auth_level,
            resource_id,
        )
    }

    pub fn to_canonical(&self) -> String {
        format!(
            "arn:openrag:{}:{}:{}:{}:L{}/{}",
            self.org,
            self.dept_path,
            self.scope.as_str(),
            self.resource_type.as_str(),
            self.auth_level.as_u8(),
            self.resource_id
        )
    }

    /// Glob `*` matching aligned 1:1 with PostgreSQL `iam.arn_matches`.
    pub fn matches_pattern(&self, pattern: &str) -> bool {
        arn_pattern_matches(pattern, &self.to_canonical())
    }
}

impl fmt::Display for Arn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_canonical())
    }
}

impl FromStr for Arn {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

pub fn arn_pattern_matches(pattern: &str, arn: &str) -> bool {
    if pattern.is_empty() || arn.is_empty() {
        return false;
    }
    if pattern == "*" {
        return true;
    }

    // Iterative glob matching supporting multiple '*' wildcards without regex compilation overhead.
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = arn.chars().collect();
    let (mut p_idx, mut t_idx) = (0usize, 0usize);
    let mut star_idx: Option<usize> = None;
    let mut match_idx = 0usize;

    while t_idx < txt.len() {
        if p_idx < pat.len() && pat[p_idx] == txt[t_idx] {
            p_idx += 1;
            t_idx += 1;
        } else if p_idx < pat.len() && pat[p_idx] == '*' {
            star_idx = Some(p_idx);
            match_idx = t_idx;
            p_idx += 1;
        } else if let Some(s) = star_idx {
            p_idx = s + 1;
            match_idx += 1;
            t_idx = match_idx;
        } else {
            return false;
        }
    }

    while p_idx < pat.len() && pat[p_idx] == '*' {
        p_idx += 1;
    }

    p_idx == pat.len()
}
