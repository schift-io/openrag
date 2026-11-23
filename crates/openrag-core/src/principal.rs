use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::access::AuthLevel;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrincipalContext {
    pub org_id: Uuid,
    pub principal_id: Uuid,
    pub auth_level: AuthLevel,
    #[serde(default)]
    pub dept_ids: Vec<Uuid>,
    #[serde(default)]
    pub dept_admin_ids: Vec<Uuid>,
    #[serde(default)]
    pub group_ids: Vec<Uuid>,
    #[serde(default)]
    pub can_unmask_pii: bool,
    #[serde(default)]
    pub is_org_admin: bool,
    #[serde(default)]
    pub api_key_id: Option<Uuid>,
    #[serde(default)]
    pub scoped_arns: Vec<String>,
    #[serde(default)]
    pub scoped_group_ids: Vec<Uuid>,
    /// Always forced to `false` by `openrag_db::begin_rls_tx`.
    #[serde(default)]
    pub bypass_rls: bool,
}

impl PrincipalContext {
    pub fn new(org_id: Uuid, principal_id: Uuid, auth_level: AuthLevel) -> Self {
        Self {
            org_id,
            principal_id,
            auth_level,
            dept_ids: Vec::new(),
            dept_admin_ids: Vec::new(),
            group_ids: Vec::new(),
            can_unmask_pii: false,
            is_org_admin: false,
            api_key_id: None,
            scoped_arns: Vec::new(),
            scoped_group_ids: Vec::new(),
            bypass_rls: false,
        }
    }
}
