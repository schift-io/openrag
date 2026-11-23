use openrag_core::{AccessScope, Arn, ArnResourceType, AuthLevel, IamAction};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDescriptor {
    pub org_id: Uuid,
    pub resource_type: ArnResourceType,
    pub resource_id: Uuid,
    pub dept_id: Uuid,
    pub owner_principal_id: Uuid,
    pub access_scope: AccessScope,
    #[serde(default)]
    pub allowed_group_ids: Vec<Uuid>,
    #[serde(default)]
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub arn: Arn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupDefinition {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub min_auth_level: AuthLevel,
    pub max_auth_level: AuthLevel,
    pub allowed_actions: Vec<IamAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupMembership {
    pub group_id: Uuid,
    pub principal_id: Uuid,
    #[serde(default)]
    pub action_overrides: Option<Vec<IamAction>>,
    #[serde(default)]
    pub level_override: Option<AuthLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PolicyEffect {
    Allow,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceGroupBinding {
    pub resource_type: ArnResourceType,
    pub resource_id: Uuid,
    pub group_id: Uuid,
    pub effect: PolicyEffect,
    pub actions: Vec<IamAction>,
    #[serde(default)]
    pub min_auth_level: Option<AuthLevel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyCondition {
    #[serde(default)]
    pub min_auth_level: Option<AuthLevel>,
    #[serde(default)]
    pub max_auth_level: Option<AuthLevel>,
    #[serde(default)]
    pub required_group_ids: Option<Vec<Uuid>>,
    #[serde(default)]
    pub allowed_scopes: Option<Vec<AccessScope>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyStatement {
    #[serde(default)]
    pub sid: Option<String>,
    pub effect: PolicyEffect,
    pub actions: Vec<IamAction>,
    pub resources: Vec<String>,
    #[serde(default)]
    pub condition: Option<PolicyCondition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IamPolicy {
    pub id: Uuid,
    pub name: String,
    pub statements: Vec<PolicyStatement>,
    #[serde(default)]
    pub min_auth_level_override: Option<AuthLevel>,
}
