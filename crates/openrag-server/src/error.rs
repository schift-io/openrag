use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openrag_core::{GuardSurface, OpenRagError};
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: String,
    pub message: String,
    pub details: Option<Value>,
}

impl ApiError {
    pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized".into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn forbidden(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found".into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "conflict".into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "service_unavailable".into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn guard_blocked(
        surface: GuardSurface,
        reason: String,
        matched_rule_ids: Vec<Uuid>,
        matched_categories: Vec<String>,
    ) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "guard_blocked".into(),
            message: format!("guard_blocked on {}: {reason}", surface.as_str()),
            details: Some(json!({
                "surface": surface.as_str(),
                "reason": reason,
                "matched_rule_ids": matched_rule_ids,
                "matched_categories": matched_categories,
            })),
        }
    }
}

impl From<OpenRagError> for ApiError {
    fn from(err: OpenRagError) -> Self {
        match err {
            OpenRagError::Validation(msg) => Self::bad_request("validation_error", msg),
            OpenRagError::Config(msg) => Self::bad_request("config_error", msg),
            OpenRagError::Unauthorized(msg) => Self::unauthorized(msg),
            OpenRagError::Forbidden(msg) => Self::forbidden("forbidden", msg),
            OpenRagError::GuardBlocked {
                surface,
                reason,
                matched_rule_ids,
                matched_categories,
            } => Self::guard_blocked(surface, reason, matched_rule_ids, matched_categories),
            OpenRagError::NotFound(msg) => Self::not_found(msg),
            OpenRagError::Conflict(msg) => Self::conflict(msg),
            OpenRagError::RemoteOcrConsentRequired {
                pages_needing_ocr,
                total_pages,
            } => Self {
                status: StatusCode::CONFLICT,
                code: "remote_ocr_consent_required".into(),
                message: format!(
                    "{pages_needing_ocr} of {total_pages} pages require consent before sending to remote OCR"
                ),
                details: Some(json!({
                    "pages_needing_ocr": pages_needing_ocr,
                    "total_pages": total_pages,
                })),
            },
            OpenRagError::ExternalAdapter { adapter, message } => {
                if message.starts_with("403") {
                    Self::forbidden("external_adapter_forbidden", format!("{adapter}: {message}"))
                } else {
                    Self::service_unavailable(format!("{adapter}: {message}"))
                }
            }
            OpenRagError::Database(msg) => {
                let lower = msg.to_ascii_lowercase();
                if lower.contains("42501")
                    || lower.contains("rls")
                    || lower.contains("row-level security")
                {
                    Self::forbidden("rls_denied", msg)
                } else if lower.contains("23505")
                    || lower.contains("duplicate key value violates unique constraint")
                {
                    Self::conflict(msg)
                } else {
                    Self {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        code: "database_error".into(),
                        message: msg,
                        details: None,
                    }
                }
            }
            OpenRagError::Internal(msg) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "internal_error".into(),
                message: msg,
                details: None,
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({
            "error": self.code,
            "code": self.code,
            "message": self.message,
        });
        if let (Some(obj), Some(extra)) = (body.as_object_mut(), self.details) {
            if let Some(extra_map) = extra.as_object() {
                for (k, v) in extra_map {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }
        (self.status, Json(body)).into_response()
    }
}
