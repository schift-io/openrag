use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::guard::GuardSurface;

pub type Result<T> = std::result::Result<T, OpenRagError>;

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenRagError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("unauthorized: {0}")]
    Unauthorized(String),

    #[error("forbidden: {0}")]
    Forbidden(String),

    #[error("guard blocked on surface {surface}: {reason}")]
    GuardBlocked {
        surface: GuardSurface,
        reason: String,
        matched_rule_ids: Vec<Uuid>,
        matched_categories: Vec<String>,
    },

    #[error("not found: {0}")]
    NotFound(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("external adapter error ({adapter}): {message}")]
    ExternalAdapter { adapter: String, message: String },

    #[error("internal error: {0}")]
    Internal(String),
}
