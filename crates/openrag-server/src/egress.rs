//! Outbound HTTP allowlist, enforced in the air-gapped profile.
//!
//! The endpoints the server is configured with through the environment are always allowed:
//! `OPENRAG_LOCAL_LLM_URL`, `OPENRAG_TEXT_EMBED_URL`, `OPENRAG_SCHIFT_OCR_URL` and the S3 bucket
//! (`OPENRAG_S3_ENDPOINT`). `egress.allow` in the pipeline config adds more. Anything else is
//! refused before a request leaves — for example a `gateway.providers` row whose `base_url` an
//! admin pointed at an outside host.

use openrag_core::{EgressTarget, OpenRagError, PipelineConfig, PipelineProfile};

use crate::state::non_empty_env;

/// Destinations configured through the environment.
pub fn configured_targets() -> Vec<EgressTarget> {
    let mut out = Vec::new();
    for name in [
        "OPENRAG_LOCAL_LLM_URL",
        "OPENRAG_TEXT_EMBED_URL",
        "OPENRAG_SCHIFT_OCR_URL",
        "OPENRAG_S3_ENDPOINT",
    ] {
        let Some(url) = non_empty_env(name) else {
            continue;
        };
        match EgressTarget::from_url(&url) {
            Ok(t) => {
                if name == "OPENRAG_S3_ENDPOINT"
                    && non_empty_env("OPENRAG_S3_URL_STYLE").as_deref() == Some("virtual")
                {
                    if let Some(bucket) = non_empty_env("OPENRAG_S3_BUCKET") {
                        out.push(EgressTarget {
                            host: format!("{}.{}", bucket.to_ascii_lowercase(), t.host),
                            port: t.port,
                        });
                    }
                }
                out.push(t);
            }
            Err(e) => tracing::warn!("{name} is not a URL, not allowlisted: {e}"),
        }
    }
    out
}

/// Refuse `url` unless the profile is not air-gapped or the destination is allowlisted.
pub fn check(pipeline: &PipelineConfig, url: &str) -> Result<(), OpenRagError> {
    if pipeline.profile != PipelineProfile::Airgap {
        return Ok(());
    }
    let dest = EgressTarget::from_url(url)
        .map_err(|_| OpenRagError::Forbidden(format!("egress to '{url}' refused: not a URL")))?;
    let extra = pipeline.egress.targets()?;
    if configured_targets()
        .iter()
        .chain(extra.iter())
        .any(|t| t.covers(&dest))
    {
        return Ok(());
    }
    let port = dest.port.map(|p| format!(":{p}")).unwrap_or_default();
    Err(OpenRagError::Forbidden(format!(
        "egress to {}{port} is not allowed in the air-gapped profile; add it to egress.allow in the pipeline config",
        dest.host
    )))
}
