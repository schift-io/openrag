//! Outbound HTTP allowlist for the air-gapped profile.
//!
//! An entry is a host (`vllm.site.local`, any port), a host and port (`10.0.5.7:8000`) or a URL
//! (`http://10.0.5.7:8000/v1`, its host and port). A destination URL is allowed when its host
//! matches an entry and the entry names no port or the same port.

use serde::{Deserialize, Serialize};

use crate::error::OpenRagError;

/// Hosts the server may call in the air-gapped profile besides the endpoints it is configured
/// with (LLM, embeddings, OCR, S3), which are always allowed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EgressConfig {
    #[serde(default)]
    pub allow: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgressTarget {
    pub host: String,
    /// `None`: any port.
    pub port: Option<u16>,
}

fn split_host_port(authority: &str) -> Result<(String, Option<u16>), OpenRagError> {
    let bad = || OpenRagError::Config(format!("invalid egress host '{authority}'"));
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let (h, after) = rest.split_once(']').ok_or_else(bad)?;
        (h, after.strip_prefix(':'))
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        }
    };
    if host.is_empty() {
        return Err(bad());
    }
    let port = match port {
        Some(p) => Some(p.parse::<u16>().map_err(|_| bad())?),
        None => None,
    };
    Ok((host.to_ascii_lowercase(), port))
}

impl EgressTarget {
    /// Parse an allowlist entry: `host`, `host:port` or a URL.
    pub fn parse_entry(raw: &str) -> Result<Self, OpenRagError> {
        let raw = raw.trim();
        if raw.contains("://") {
            return Self::from_url(raw);
        }
        let (host, port) = split_host_port(raw.split('/').next().unwrap_or(raw))?;
        Ok(Self { host, port })
    }

    /// Host and effective port of a URL (scheme default when the URL names none).
    pub fn from_url(url: &str) -> Result<Self, OpenRagError> {
        let (scheme, rest) = url
            .trim()
            .split_once("://")
            .ok_or_else(|| OpenRagError::Config(format!("'{url}' is not a URL")))?;
        let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
        let (host, port) = split_host_port(authority)?;
        let port = port.or(match scheme.to_ascii_lowercase().as_str() {
            "http" => Some(80),
            "https" => Some(443),
            _ => None,
        });
        Ok(Self { host, port })
    }

    /// Whether `dest` (a destination parsed with [`EgressTarget::from_url`]) is covered.
    pub fn covers(&self, dest: &EgressTarget) -> bool {
        self.host == dest.host && (self.port.is_none() || self.port == dest.port)
    }
}

impl EgressConfig {
    pub fn targets(&self) -> Result<Vec<EgressTarget>, OpenRagError> {
        self.allow
            .iter()
            .map(|e| EgressTarget::parse_entry(e))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(entries: &[&str], url: &str) -> Result<bool, OpenRagError> {
        let cfg = EgressConfig {
            allow: entries.iter().map(|s| s.to_string()).collect(),
        };
        let dest = EgressTarget::from_url(url)?;
        Ok(cfg.targets()?.iter().any(|t| t.covers(&dest)))
    }

    #[test]
    fn entries_match_host_and_port() -> Result<(), OpenRagError> {
        assert!(allowed(
            &["vllm.site.local"],
            "http://vllm.site.local:8000/v1/chat/completions"
        )?);
        assert!(allowed(&["10.0.5.7:8000"], "http://10.0.5.7:8000/v1")?);
        assert!(!allowed(&["10.0.5.7:8000"], "http://10.0.5.7:9000/v1")?);
        assert!(allowed(
            &["http://10.0.5.7:8000/v1"],
            "http://10.0.5.7:8000/v1/chat/completions"
        )?);
        // scheme default ports
        assert!(allowed(
            &["https://llm.site.local"],
            "https://llm.site.local/v1"
        )?);
        assert!(!allowed(
            &["https://llm.site.local"],
            "http://llm.site.local/v1"
        )?);
        assert!(allowed(&["[fd00::5]:8000"], "http://[fd00::5]:8000/v1")?);
        assert!(allowed(&["LLM.Site.Local"], "http://llm.site.local:8000")?);
        // no partial host matches, userinfo ignored
        assert!(!allowed(&["site.local"], "http://evil.site.local/")?);
        assert!(!allowed(
            &["vllm.site.local"],
            "http://vllm.site.local.evil.com/"
        )?);
        assert!(!allowed(
            &["vllm.site.local"],
            "http://vllm.site.local@evil.com/"
        )?);
        assert!(!allowed(&[], "https://api.openai.com/v1")?);
        Ok(())
    }

    #[test]
    fn bad_entries_are_config_errors() {
        for bad in ["", ":8000", "host:port", "http://", "[fd00::5"] {
            assert!(EgressTarget::parse_entry(bad).is_err(), "{bad:?}");
        }
    }
}
