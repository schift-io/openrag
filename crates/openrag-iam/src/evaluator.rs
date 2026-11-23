use openrag_core::{AccessScope, AuthLevel, IamAction, PrincipalContext};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::model::{
    GroupDefinition, GroupMembership, IamPolicy, PolicyCondition, PolicyEffect, ResourceDescriptor,
    ResourceGroupBinding,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchedRule {
    pub source: String,
    #[serde(default)]
    pub rule_id: Option<Uuid>,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceStep {
    pub stage: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessDecision {
    pub allowed: bool,
    pub reason: String,
    pub matched_denies: Vec<MatchedRule>,
    pub matched_allows: Vec<MatchedRule>,
    pub effective_min_level: AuthLevel,
    pub effective_actions: Vec<IamAction>,
    pub trace: Vec<TraceStep>,
}

pub struct IamEvaluator;

impl IamEvaluator {
    pub fn evaluate(
        principal: &PrincipalContext,
        resource: &ResourceDescriptor,
        action: IamAction,
        groups: &[GroupDefinition],
        memberships: &[GroupMembership],
        bindings: &[ResourceGroupBinding],
        policies: &[IamPolicy],
    ) -> AccessDecision {
        let mut decision = Self::evaluate_single(
            principal,
            resource,
            action,
            groups,
            memberships,
            bindings,
            policies,
        );

        let all_actions = [
            IamAction::Read,
            IamAction::Write,
            IamAction::Search,
            IamAction::PiiUnmask,
            IamAction::Admin,
        ];
        let mut effective_actions = Vec::new();
        for candidate in all_actions {
            let sub = Self::evaluate_single(
                principal,
                resource,
                candidate,
                groups,
                memberships,
                bindings,
                policies,
            );
            if sub.allowed {
                effective_actions.push(candidate);
            }
        }
        decision.effective_actions = effective_actions;
        decision
    }

    /// Simulator entrypoint returning full `AccessDecision` trace for `/studio/iam`.
    pub fn simulate(
        principal: &PrincipalContext,
        resource: &ResourceDescriptor,
        action: IamAction,
        groups: &[GroupDefinition],
        memberships: &[GroupMembership],
        bindings: &[ResourceGroupBinding],
        policies: &[IamPolicy],
    ) -> AccessDecision {
        Self::evaluate(
            principal,
            resource,
            action,
            groups,
            memberships,
            bindings,
            policies,
        )
    }

    fn evaluate_single(
        principal: &PrincipalContext,
        resource: &ResourceDescriptor,
        action: IamAction,
        groups: &[GroupDefinition],
        memberships: &[GroupMembership],
        bindings: &[ResourceGroupBinding],
        policies: &[IamPolicy],
    ) -> AccessDecision {
        let mut trace = Vec::new();
        let mut matched_denies = Vec::new();
        let mut matched_allows = Vec::new();
        let mut effective_min_level = resource.min_auth_level;

        // 1. Org isolation & API Key ARN scoping
        if principal.org_id != resource.org_id {
            trace.push(TraceStep {
                stage: "org_isolation".into(),
                passed: false,
                detail: format!(
                    "principal org {} != resource org {}",
                    principal.org_id, resource.org_id
                ),
            });
            return AccessDecision {
                allowed: false,
                reason: "org_mismatch".into(),
                matched_denies,
                matched_allows,
                effective_min_level,
                effective_actions: Vec::new(),
                trace,
            };
        }
        trace.push(TraceStep {
            stage: "org_isolation".into(),
            passed: true,
            detail: format!("org {} matched", principal.org_id),
        });

        if !principal.scoped_arns.is_empty()
            && !principal
                .scoped_arns
                .iter()
                .any(|pat| resource.arn.matches_pattern(pat))
        {
            trace.push(TraceStep {
                stage: "api_key_scope".into(),
                passed: false,
                detail: format!(
                    "resource ARN {} outside api_key scoped_arns",
                    resource.arn
                ),
            });
            return AccessDecision {
                allowed: false,
                reason: "api_key_arn_scope_denied".into(),
                matched_denies,
                matched_allows,
                effective_min_level,
                effective_actions: Vec::new(),
                trace,
            };
        }

        if action == IamAction::PiiUnmask && !principal.can_unmask_pii {
            trace.push(TraceStep {
                stage: "pii_unmask_capability".into(),
                passed: false,
                detail: "principal.can_unmask_pii is false".into(),
            });
            return AccessDecision {
                allowed: false,
                reason: "principal_cannot_unmask_pii".into(),
                matched_denies,
                matched_allows,
                effective_min_level,
                effective_actions: Vec::new(),
                trace,
            };
        }

        // Full group set for Explicit Deny (never narrowed by scoped_group_ids)
        let mut deny_group_ids = principal.group_ids.clone();
        for membership in memberships {
            if membership.principal_id == principal.principal_id
                && !deny_group_ids.contains(&membership.group_id)
            {
                deny_group_ids.push(membership.group_id);
            }
        }

        // Narrowed group set for Allow evaluation only
        let allow_group_ids: Vec<Uuid> = if principal.scoped_group_ids.is_empty() {
            principal.group_ids.clone()
        } else {
            principal
                .group_ids
                .iter()
                .copied()
                .filter(|g| principal.scoped_group_ids.contains(g))
                .collect()
        };

        // 2. Explicit Deny > Allow (Group Deny + Resource Group Binding Deny)
        for gid in &deny_group_ids {
            if resource.denied_group_ids.contains(gid) {
                matched_denies.push(MatchedRule {
                    source: "resource_denied_group".into(),
                    rule_id: Some(*gid),
                    description: format!("principal group {gid} is in resource.denied_group_ids"),
                });
            }
        }

        for binding in bindings {
            if binding.effect == PolicyEffect::Deny
                && binding.resource_type == resource.resource_type
                && binding.resource_id == resource.resource_id
                && deny_group_ids.contains(&binding.group_id)
                && binding.actions.iter().any(|a| a.matches(action))
            {
                matched_denies.push(MatchedRule {
                    source: "resource_group_binding".into(),
                    rule_id: Some(binding.group_id),
                    description: format!(
                        "explicit deny binding on group {} for action {}",
                        binding.group_id, action
                    ),
                });
            }
        }

        let group_deny_passed = matched_denies.is_empty();
        trace.push(TraceStep {
            stage: "explicit_group_deny".into(),
            passed: group_deny_passed,
            detail: if group_deny_passed {
                "no explicit group deny".into()
            } else {
                format!("{} group deny rule(s) matched", matched_denies.len())
            },
        });

        // Explicit ARN Policy Deny
        let mut arn_denies = 0usize;
        for policy in policies {
            for stmt in &policy.statements {
                if stmt.effect == PolicyEffect::Deny
                    && stmt.actions.iter().any(|a| a.matches(action))
                    && stmt
                        .resources
                        .iter()
                        .any(|pat| resource.arn.matches_pattern(pat))
                    && condition_matches(
                        stmt.condition.as_ref(),
                        principal.auth_level,
                        &deny_group_ids,
                        resource.access_scope,
                    )
                {
                    arn_denies += 1;
                    matched_denies.push(MatchedRule {
                        source: "iam_policy".into(),
                        rule_id: Some(policy.id),
                        description: format!(
                            "explicit deny policy '{}' ({})",
                            policy.name,
                            stmt.sid.as_deref().unwrap_or("unnamed")
                        ),
                    });
                }
            }
        }
        trace.push(TraceStep {
            stage: "explicit_arn_deny".into(),
            passed: arn_denies == 0,
            detail: if arn_denies == 0 {
                "no explicit ARN deny".into()
            } else {
                format!("{arn_denies} ARN deny statement(s) matched")
            },
        });

        if !matched_denies.is_empty() {
            return AccessDecision {
                allowed: false,
                reason: "explicit_deny".into(),
                matched_denies,
                matched_allows,
                effective_min_level,
                effective_actions: Vec::new(),
                trace,
            };
        }

        // 3. Base Resource Permission Level
        if principal.auth_level < resource.min_auth_level {
            trace.push(TraceStep {
                stage: "permission_level".into(),
                passed: false,
                detail: format!(
                    "caller level {} < resource min_auth_level {}",
                    principal.auth_level, resource.min_auth_level
                ),
            });
            return AccessDecision {
                allowed: false,
                reason: "insufficient_auth_level".into(),
                matched_denies,
                matched_allows,
                effective_min_level,
                effective_actions: Vec::new(),
                trace,
            };
        }

        // 4. Private Scope Strict Isolation (owner or same-dept dept_admin ONLY)
        if resource.access_scope == AccessScope::Private {
            if principal.principal_id == resource.owner_principal_id {
                matched_allows.push(MatchedRule {
                    source: "scope_owner".into(),
                    rule_id: Some(principal.principal_id),
                    description: "owner access to private resource".into(),
                });
                trace.push(TraceStep {
                    stage: "permission_level".into(),
                    passed: true,
                    detail: format!("caller {} >= {}", principal.auth_level, effective_min_level),
                });
                trace.push(TraceStep {
                    stage: "scope_and_allow".into(),
                    passed: true,
                    detail: "private scope allowed for owner".into(),
                });
                return AccessDecision {
                    allowed: true,
                    reason: "allowed_private_owner".into(),
                    matched_denies,
                    matched_allows,
                    effective_min_level,
                    effective_actions: Vec::new(),
                    trace,
                };
            }

            if principal.dept_admin_ids.contains(&resource.dept_id) {
                matched_allows.push(MatchedRule {
                    source: "scope_dept_admin".into(),
                    rule_id: Some(resource.dept_id),
                    description: "dept_admin access to private resource in own dept".into(),
                });
                trace.push(TraceStep {
                    stage: "permission_level".into(),
                    passed: true,
                    detail: format!("caller {} >= {}", principal.auth_level, effective_min_level),
                });
                trace.push(TraceStep {
                    stage: "scope_and_allow".into(),
                    passed: true,
                    detail: "private scope allowed for dept_admin".into(),
                });
                return AccessDecision {
                    allowed: true,
                    reason: "allowed_private_dept_admin".into(),
                    matched_denies,
                    matched_allows,
                    effective_min_level,
                    effective_actions: Vec::new(),
                    trace,
                };
            }

            trace.push(TraceStep {
                stage: "permission_level".into(),
                passed: true,
                detail: format!("caller {} >= {}", principal.auth_level, effective_min_level),
            });
            trace.push(TraceStep {
                stage: "scope_and_allow".into(),
                passed: false,
                detail: "private scope blocks peers and external dept_admins".into(),
            });
            return AccessDecision {
                allowed: false,
                reason: "private_scope_denied".into(),
                matched_denies,
                matched_allows,
                effective_min_level,
                effective_actions: Vec::new(),
                trace,
            };
        }

        // 5. Evaluate Group Allow & ARN Policy Allow (tracking effective_min_level)
        let mut group_allow_blocked_by_level = false;
        let mut passed_allow_levels: Vec<AuthLevel> = Vec::new();
        let mut blocked_allow_levels: Vec<AuthLevel> = Vec::new();

        for group in groups {
            if group.org_id != principal.org_id || !allow_group_ids.contains(&group.id) {
                continue;
            }
            let Some(membership) = memberships
                .iter()
                .find(|m| m.group_id == group.id && m.principal_id == principal.principal_id)
            else {
                continue;
            };

            let group_actions = membership
                .action_overrides
                .as_deref()
                .unwrap_or(&group.allowed_actions);
            let caller_group_level = membership
                .level_override
                .map_or(principal.auth_level, |ov| principal.auth_level.min(ov));

            // G1: allowed_group_ids on resource
            if resource.allowed_group_ids.contains(&group.id)
                && group_actions.iter().any(|a| a.matches(action))
            {
                let req_level = resource.min_auth_level.max(group.min_auth_level);
                if caller_group_level >= req_level && resource.min_auth_level <= group.max_auth_level
                {
                    passed_allow_levels.push(req_level);
                    matched_allows.push(MatchedRule {
                        source: "scope_group".into(),
                        rule_id: Some(group.id),
                        description: format!(
                            "group '{}' ({}) allows action {}",
                            group.name, group.id, action
                        ),
                    });
                } else {
                    blocked_allow_levels.push(req_level);
                    group_allow_blocked_by_level = true;
                }
            }

            // G2: resource_group_bindings 'allow'
            for binding in bindings {
                if binding.effect == PolicyEffect::Allow
                    && binding.resource_type == resource.resource_type
                    && binding.resource_id == resource.resource_id
                    && binding.group_id == group.id
                    && binding.actions.iter().any(|a| a.matches(action))
                    && group_actions.iter().any(|a| a.matches(action))
                {
                    let req_level = resource
                        .min_auth_level
                        .max(group.min_auth_level)
                        .max(binding.min_auth_level.unwrap_or(AuthLevel::MIN));
                    if caller_group_level >= req_level
                        && resource.min_auth_level <= group.max_auth_level
                    {
                        passed_allow_levels.push(req_level);
                        matched_allows.push(MatchedRule {
                            source: "resource_group_binding".into(),
                            rule_id: Some(group.id),
                            description: format!(
                                "resource_group_binding '{}' allows action {}",
                                group.name, action
                            ),
                        });
                    } else {
                        blocked_allow_levels.push(req_level);
                        group_allow_blocked_by_level = true;
                    }
                }
            }
        }

        let mut policy_allow_blocked_by_level = false;
        for policy in policies {
            let pol_override = policy.min_auth_level_override.unwrap_or(AuthLevel::MIN);
            for stmt in &policy.statements {
                if stmt.effect == PolicyEffect::Allow
                    && stmt.actions.iter().any(|a| a.matches(action))
                    && stmt
                        .resources
                        .iter()
                        .any(|pat| resource.arn.matches_pattern(pat))
                {
                    let cond_min = stmt
                        .condition
                        .as_ref()
                        .and_then(|c| c.min_auth_level)
                        .unwrap_or(AuthLevel::MIN);
                    let req_level = resource.min_auth_level.max(pol_override).max(cond_min);
                    if condition_matches_ignoring_min(
                        stmt.condition.as_ref(),
                        principal.auth_level,
                        &allow_group_ids,
                        resource.access_scope,
                    ) {
                        if principal.auth_level < req_level {
                            blocked_allow_levels.push(req_level);
                            policy_allow_blocked_by_level = true;
                            continue;
                        }
                        passed_allow_levels.push(req_level);
                        matched_allows.push(MatchedRule {
                            source: "iam_policy".into(),
                            rule_id: Some(policy.id),
                            description: format!(
                                "ARN policy '{}' ({}) allows action {}",
                                policy.name,
                                stmt.sid.as_deref().unwrap_or("unnamed"),
                                action
                            ),
                        });
                    }
                }
            }
        }

        // 6. Scope Resolution (Dept | Group | Org)
        let is_read_or_search = matches!(action, IamAction::Read | IamAction::Search);
        let is_owner = principal.principal_id == resource.owner_principal_id;
        let is_dept_member = principal.dept_ids.contains(&resource.dept_id);
        let is_dept_admin = principal.dept_admin_ids.contains(&resource.dept_id);

        if is_owner {
            passed_allow_levels.push(resource.min_auth_level);
            matched_allows.push(MatchedRule {
                source: "scope_owner".into(),
                rule_id: Some(principal.principal_id),
                description: "resource owner".into(),
            });
        }
        if is_dept_admin
            && !is_read_or_search
            && matches!(
                resource.access_scope,
                AccessScope::Dept | AccessScope::Org
            )
        {
            passed_allow_levels.push(resource.min_auth_level);
            matched_allows.push(MatchedRule {
                source: "scope_dept_admin".into(),
                rule_id: Some(resource.dept_id),
                description: "dept_admin of resource department".into(),
            });
        }

        let scope_allowed = match resource.access_scope {
            AccessScope::Dept => {
                if is_read_or_search {
                    if is_dept_member {
                        passed_allow_levels.push(resource.min_auth_level);
                        matched_allows.push(MatchedRule {
                            source: "scope_dept".into(),
                            rule_id: Some(resource.dept_id),
                            description: "same department member".into(),
                        });
                    }
                    is_owner || is_dept_member || !matched_allows.is_empty()
                } else {
                    is_owner || is_dept_admin || !matched_allows.is_empty()
                }
            }
            AccessScope::Group => is_owner || !matched_allows.is_empty(),
            AccessScope::Org => {
                if is_read_or_search {
                    passed_allow_levels.push(resource.min_auth_level);
                    matched_allows.push(MatchedRule {
                        source: "scope_org".into(),
                        rule_id: Some(resource.org_id),
                        description: "org-wide readable resource".into(),
                    });
                    true
                } else {
                    if principal.is_org_admin {
                        passed_allow_levels.push(resource.min_auth_level);
                        matched_allows.push(MatchedRule {
                            source: "scope_org_admin".into(),
                            rule_id: Some(principal.principal_id),
                            description: "org admin write access".into(),
                        });
                    }
                    is_owner || is_dept_admin || principal.is_org_admin || !matched_allows.is_empty()
                }
            }
            AccessScope::Private => false,
        };

        effective_min_level = if scope_allowed {
            passed_allow_levels
                .into_iter()
                .min()
                .unwrap_or(resource.min_auth_level)
        } else if !blocked_allow_levels.is_empty() {
            blocked_allow_levels
                .into_iter()
                .min()
                .unwrap_or(resource.min_auth_level)
        } else {
            resource.min_auth_level
        };

        let level_passed = !((group_allow_blocked_by_level || policy_allow_blocked_by_level)
            && !scope_allowed);
        trace.push(TraceStep {
            stage: "permission_level".into(),
            passed: level_passed,
            detail: format!(
                "caller {} vs effective_min_level {}",
                principal.auth_level, effective_min_level
            ),
        });
        trace.push(TraceStep {
            stage: "scope_and_allow".into(),
            passed: scope_allowed,
            detail: if scope_allowed {
                format!("{} allow rule(s) matched", matched_allows.len())
            } else {
                format!("no allow path matched for scope {}", resource.access_scope)
            },
        });

        let reason = if scope_allowed {
            format!("allowed_{}", resource.access_scope.as_str())
        } else if group_allow_blocked_by_level || policy_allow_blocked_by_level {
            "insufficient_auth_level_or_group_ceiling".into()
        } else {
            format!("{}_scope_denied", resource.access_scope.as_str())
        };

        AccessDecision {
            allowed: scope_allowed,
            reason,
            matched_denies,
            matched_allows,
            effective_min_level,
            effective_actions: Vec::new(),
            trace,
        }
    }
}

fn condition_matches(
    condition: Option<&PolicyCondition>,
    effective_level: AuthLevel,
    group_ids: &[Uuid],
    access_scope: AccessScope,
) -> bool {
    let Some(cond) = condition else {
        return true;
    };
    if let Some(min_lvl) = cond.min_auth_level {
        if effective_level < min_lvl {
            return false;
        }
    }
    condition_matches_ignoring_min(Some(cond), effective_level, group_ids, access_scope)
}

fn condition_matches_ignoring_min(
    condition: Option<&PolicyCondition>,
    effective_level: AuthLevel,
    group_ids: &[Uuid],
    access_scope: AccessScope,
) -> bool {
    let Some(cond) = condition else {
        return true;
    };
    if let Some(max_lvl) = cond.max_auth_level {
        if effective_level > max_lvl {
            return false;
        }
    }
    if let Some(req_groups) = &cond.required_group_ids {
        if !req_groups.is_empty() && !req_groups.iter().any(|g| group_ids.contains(g)) {
            return false;
        }
    }
    if let Some(scopes) = &cond.allowed_scopes {
        if !scopes.is_empty() && !scopes.contains(&access_scope) {
            return false;
        }
    }
    true
}
