use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, IamAction, PrincipalContext,
};
use openrag_iam::{
    GroupDefinition, GroupMembership, IamEvaluator, IamPolicy, PolicyEffect, PolicyStatement,
    ResourceDescriptor, ResourceGroupBinding,
};
use uuid::Uuid;

fn make_resource(
    org_id: Uuid,
    dept_id: Uuid,
    owner_id: Uuid,
    scope: AccessScope,
    min_level: u8,
) -> Result<ResourceDescriptor, Box<dyn std::error::Error>> {
    let res_id = Uuid::new_v4();
    let arn = Arn::new(
        "acme",
        "finance",
        scope,
        ArnResourceType::Document,
        AuthLevel::new(min_level)?,
        res_id.to_string(),
    )?;
    Ok(ResourceDescriptor {
        org_id,
        resource_type: ArnResourceType::Document,
        resource_id: res_id,
        dept_id,
        owner_principal_id: owner_id,
        access_scope: scope,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        min_auth_level: AuthLevel::new(min_level)?,
        arn,
    })
}

#[test]
fn private_resource_blocks_peer_and_other_dept_admin_even_with_allow_star(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let fin_dept = Uuid::new_v4();
    let hr_dept = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let carol_fin_admin = Uuid::new_v4();
    let dave_hr_admin = Uuid::new_v4();

    let res = make_resource(org_id, fin_dept, alice, AccessScope::Private, 5)?;

    let allow_all = IamPolicy {
        id: Uuid::new_v4(),
        name: "org-wide-allow".into(),
        statements: vec![PolicyStatement {
            sid: Some("allow-all".into()),
            effect: PolicyEffect::Allow,
            actions: vec![IamAction::Wildcard],
            resources: vec!["*".into()],
            condition: None,
        }],
        min_auth_level_override: None,
    };

    // Owner Alice allowed
    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(5)?);
    alice_ctx.dept_ids = vec![fin_dept];
    assert!(
        IamEvaluator::evaluate(
            &alice_ctx,
            &res,
            IamAction::Read,
            &[],
            &[],
            &[],
            std::slice::from_ref(&allow_all)
        )
        .allowed
    );

    // Same-dept peer Bob (even at higher auth_level + Allow *) strictly blocked!
    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(9)?);
    bob_ctx.dept_ids = vec![fin_dept];
    let bob_dec = IamEvaluator::simulate(
        &bob_ctx,
        &res,
        IamAction::Read,
        &[],
        &[],
        &[],
        std::slice::from_ref(&allow_all),
    );
    assert!(!bob_dec.allowed);
    assert_eq!(bob_dec.reason, "private_scope_denied");

    // Same-dept dept_admin Carol allowed
    let mut carol_ctx = PrincipalContext::new(org_id, carol_fin_admin, AuthLevel::new(7)?);
    carol_ctx.dept_ids = vec![fin_dept];
    carol_ctx.dept_admin_ids = vec![fin_dept];
    assert!(
        IamEvaluator::evaluate(&carol_ctx, &res, IamAction::Read, &[], &[], &[], &[]).allowed
    );

    // Different-dept dept_admin Dave blocked
    let mut dave_ctx = PrincipalContext::new(org_id, dave_hr_admin, AuthLevel::new(9)?);
    dave_ctx.dept_ids = vec![hr_dept];
    dave_ctx.dept_admin_ids = vec![hr_dept];
    assert!(
        !IamEvaluator::evaluate(&dave_ctx, &res, IamAction::Read, &[], &[], &[], &[allow_all])
            .allowed
    );

    Ok(())
}

#[test]
fn explicit_deny_overrides_group_and_policy_allows() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let allow_group = Uuid::new_v4();
    let deny_group = Uuid::new_v4();

    let mut res = make_resource(org_id, dept_id, owner, AccessScope::Group, 3)?;
    res.allowed_group_ids = vec![allow_group];
    res.denied_group_ids = vec![deny_group];

    let groups = vec![
        GroupDefinition {
            id: allow_group,
            org_id,
            name: "audit-tf".into(),
            min_auth_level: AuthLevel::new(1)?,
            max_auth_level: AuthLevel::new(10)?,
            allowed_actions: vec![IamAction::Read, IamAction::Search],
        },
        GroupDefinition {
            id: deny_group,
            org_id,
            name: "blocked-group".into(),
            min_auth_level: AuthLevel::new(1)?,
            max_auth_level: AuthLevel::new(10)?,
            allowed_actions: vec![IamAction::Read],
        },
    ];
    let memberships = vec![GroupMembership {
        group_id: allow_group,
        principal_id: bob,
        action_overrides: None,
        level_override: None,
    }];

    // Bob in both allowed_group and denied_group -> denied
    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(8)?);
    bob_ctx.group_ids = vec![allow_group, deny_group];
    let dec = IamEvaluator::evaluate(&bob_ctx, &res, IamAction::Read, &groups, &memberships, &[], &[]);
    assert!(!dec.allowed);
    assert_eq!(dec.reason, "explicit_deny");
    assert_eq!(dec.matched_denies.len(), 1);

    // Without deny_group, Bob is allowed via allow_group
    bob_ctx.group_ids = vec![allow_group];
    let dec_ok = IamEvaluator::evaluate(&bob_ctx, &res, IamAction::Read, &groups, &memberships, &[], &[]);
    assert!(dec_ok.allowed);

    // ResourceGroupBinding Deny overrides allow
    let deny_binding = ResourceGroupBinding {
        resource_type: res.resource_type,
        resource_id: res.resource_id,
        group_id: allow_group,
        effect: PolicyEffect::Deny,
        actions: vec![IamAction::Read],
        min_auth_level: None,
    };
    let dec_binding_deny = IamEvaluator::evaluate(
        &bob_ctx,
        &res,
        IamAction::Read,
        &groups,
        &memberships,
        &[deny_binding],
        &[],
    );
    assert!(!dec_binding_deny.allowed);
    assert_eq!(dec_binding_deny.reason, "explicit_deny");

    // ARN Policy Deny overrides allow
    let deny_policy = IamPolicy {
        id: Uuid::new_v4(),
        name: "deny-doc".into(),
        statements: vec![PolicyStatement {
            sid: Some("deny-finance".into()),
            effect: PolicyEffect::Deny,
            actions: vec![IamAction::Read],
            resources: vec!["arn:openrag:acme:finance:*".into()],
            condition: None,
        }],
        min_auth_level_override: None,
    };
    let dec_pol_deny = IamEvaluator::evaluate(
        &bob_ctx,
        &res,
        IamAction::Read,
        &groups,
        &memberships,
        &[],
        &[deny_policy],
    );
    assert!(!dec_pol_deny.allowed);
    assert_eq!(dec_pol_deny.reason, "explicit_deny");

    Ok(())
}

#[test]
fn permission_level_ceilings_and_overrides_enforced() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let carol = Uuid::new_v4();
    let group_id = Uuid::new_v4();

    // 1. Group max_auth_level = 8 blocks L9 doc even for L9 member
    let mut res_l9 = make_resource(org_id, dept_id, owner, AccessScope::Group, 9)?;
    res_l9.allowed_group_ids = vec![group_id];
    let groups_capped = vec![GroupDefinition {
        id: group_id,
        org_id,
        name: "capped-group".into(),
        min_auth_level: AuthLevel::new(1)?,
        max_auth_level: AuthLevel::new(8)?,
        allowed_actions: vec![IamAction::Read],
    }];
    let memberships = vec![GroupMembership {
        group_id,
        principal_id: carol,
        action_overrides: None,
        level_override: None,
    }];
    let mut carol_l9 = PrincipalContext::new(org_id, carol, AuthLevel::new(9)?);
    carol_l9.group_ids = vec![group_id];
    assert!(
        !IamEvaluator::evaluate(&carol_l9, &res_l9, IamAction::Read, &groups_capped, &memberships, &[], &[])
            .allowed
    );

    // 2. Group membership level_override = 5 blocks Carol (L7) on binding requiring min_auth_level = 6
    let res_l4 = make_resource(org_id, dept_id, owner, AccessScope::Group, 4)?;
    let groups_l10 = vec![GroupDefinition {
        id: group_id,
        org_id,
        name: "tf-group".into(),
        min_auth_level: AuthLevel::new(1)?,
        max_auth_level: AuthLevel::new(10)?,
        allowed_actions: vec![IamAction::Read, IamAction::Search],
    }];
    let memberships_capped = vec![GroupMembership {
        group_id,
        principal_id: carol,
        action_overrides: None,
        level_override: Some(AuthLevel::new(5)?),
    }];
    let bindings = vec![ResourceGroupBinding {
        resource_type: res_l4.resource_type,
        resource_id: res_l4.resource_id,
        group_id,
        effect: PolicyEffect::Allow,
        actions: vec![IamAction::Read, IamAction::Search],
        min_auth_level: Some(AuthLevel::new(6)?),
    }];
    let mut carol_l7 = PrincipalContext::new(org_id, carol, AuthLevel::new(7)?);
    carol_l7.group_ids = vec![group_id];
    let dec_ov = IamEvaluator::evaluate(
        &carol_l7,
        &res_l4,
        IamAction::Read,
        &groups_l10,
        &memberships_capped,
        &bindings,
        &[],
    );
    assert!(!dec_ov.allowed);
    assert_eq!(dec_ov.effective_min_level, AuthLevel::new(6)?);

    // 3. PolicyAttachment min_auth_level_override = 9 blocks L8 and allows L9
    let pol = IamPolicy {
        id: Uuid::new_v4(),
        name: "strict-arn".into(),
        statements: vec![PolicyStatement {
            sid: None,
            effect: PolicyEffect::Allow,
            actions: vec![IamAction::Read],
            resources: vec!["arn:openrag:acme:finance:*".into()],
            condition: None,
        }],
        min_auth_level_override: Some(AuthLevel::new(9)?),
    };
    let user_l8 = PrincipalContext::new(org_id, carol, AuthLevel::new(8)?);
    assert!(
        !IamEvaluator::evaluate(
            &user_l8,
            &res_l4,
            IamAction::Read,
            &[],
            &[],
            &[],
            std::slice::from_ref(&pol)
        )
        .allowed
    );
    let user_l9 = PrincipalContext::new(org_id, carol, AuthLevel::new(9)?);
    assert!(IamEvaluator::evaluate(&user_l9, &res_l4, IamAction::Read, &[], &[], &[], &[pol]).allowed);

    Ok(())
}

#[test]
fn per_action_pii_unmask_and_api_key_scoped_arns_enforced() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let res = make_resource(org_id, dept_id, alice, AccessScope::Org, 3)?;

    // Without can_unmask_pii, even owner cannot unmask PII
    let mut ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(8)?);
    ctx.can_unmask_pii = false;
    let dec_no_unmask = IamEvaluator::evaluate(&ctx, &res, IamAction::PiiUnmask, &[], &[], &[], &[]);
    assert!(!dec_no_unmask.allowed);
    assert_eq!(dec_no_unmask.reason, "principal_cannot_unmask_pii");
    assert!(dec_no_unmask.effective_actions.contains(&IamAction::Read));
    assert!(dec_no_unmask.effective_actions.contains(&IamAction::Write));
    assert!(!dec_no_unmask.effective_actions.contains(&IamAction::PiiUnmask));

    // With can_unmask_pii = true, owner can unmask
    ctx.can_unmask_pii = true;
    assert!(IamEvaluator::evaluate(&ctx, &res, IamAction::PiiUnmask, &[], &[], &[], &[]).allowed);

    // API key scoped_arns ceiling blocks out-of-scope ARN
    ctx.scoped_arns = vec!["arn:openrag:acme:hr/*".into()];
    let dec_ak = IamEvaluator::evaluate(&ctx, &res, IamAction::Read, &[], &[], &[], &[]);
    assert!(!dec_ak.allowed);
    assert_eq!(dec_ak.reason, "api_key_arn_scope_denied");

    Ok(())
}

#[test]
fn api_key_scoped_group_ids_never_bypasses_explicit_deny_and_narrows_allow(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let g_audit = Uuid::new_v4();
    let g_blocked = Uuid::new_v4();
    let g_other = Uuid::new_v4();

    let mut res = make_resource(org_id, dept_id, owner, AccessScope::Group, 3)?;
    res.allowed_group_ids = vec![g_audit];
    res.denied_group_ids = vec![g_blocked];

    let groups = vec![
        GroupDefinition {
            id: g_audit,
            org_id,
            name: "audit".into(),
            min_auth_level: AuthLevel::new(1)?,
            max_auth_level: AuthLevel::new(10)?,
            allowed_actions: vec![IamAction::Read],
        },
        GroupDefinition {
            id: g_blocked,
            org_id,
            name: "blocked".into(),
            min_auth_level: AuthLevel::new(1)?,
            max_auth_level: AuthLevel::new(10)?,
            allowed_actions: vec![IamAction::Read],
        },
    ];
    let memberships = vec![
        GroupMembership {
            group_id: g_audit,
            principal_id: bob,
            action_overrides: None,
            level_override: None,
        },
        GroupMembership {
            group_id: g_blocked,
            principal_id: bob,
            action_overrides: None,
            level_override: None,
        },
    ];

    // Bob scopes API key solely to g_audit -> explicit deny on g_blocked STILL blocks!
    let mut bob_scoped = PrincipalContext::new(org_id, bob, AuthLevel::new(8)?);
    bob_scoped.group_ids = vec![g_audit, g_blocked];
    bob_scoped.scoped_group_ids = vec![g_audit];
    let dec_deny = IamEvaluator::evaluate(
        &bob_scoped,
        &res,
        IamAction::Read,
        &groups,
        &memberships,
        &[],
        &[],
    );
    assert!(!dec_deny.allowed);
    assert_eq!(dec_deny.reason, "explicit_deny");

    // Even if group_ids omitted g_blocked, membership in g_blocked still enforces Explicit Deny
    bob_scoped.group_ids = vec![g_audit];
    let dec_member_deny = IamEvaluator::evaluate(
        &bob_scoped,
        &res,
        IamAction::Read,
        &groups,
        &memberships,
        &[],
        &[],
    );
    assert!(!dec_member_deny.allowed);
    assert_eq!(dec_member_deny.reason, "explicit_deny");

    // When resource does not deny g_blocked, scoping API key to unrelated g_other narrows Allow -> denied
    res.denied_group_ids.clear();
    let mut bob_narrow = PrincipalContext::new(org_id, bob, AuthLevel::new(8)?);
    bob_narrow.group_ids = vec![g_audit];
    bob_narrow.scoped_group_ids = vec![g_other];
    let dec_narrow = IamEvaluator::evaluate(
        &bob_narrow,
        &res,
        IamAction::Read,
        &groups,
        &memberships,
        &[],
        &[],
    );
    assert!(!dec_narrow.allowed);
    assert_eq!(dec_narrow.reason, "group_scope_denied");

    Ok(())
}

#[test]
fn effective_min_level_and_dept_read_parity_are_exact() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let g_high = Uuid::new_v4();

    // 1. Dept L3 resource: L3 member in dept AND member of L8 group -> allowed via scope_dept,
    // effective_min_level MUST remain L3 (not inflated to L8 by the unselected group).
    let mut dept_res = make_resource(org_id, dept_id, owner, AccessScope::Dept, 3)?;
    dept_res.allowed_group_ids = vec![g_high];
    let groups = vec![GroupDefinition {
        id: g_high,
        org_id,
        name: "l8-group".into(),
        min_auth_level: AuthLevel::new(8)?,
        max_auth_level: AuthLevel::new(10)?,
        allowed_actions: vec![IamAction::Read],
    }];
    let memberships = vec![GroupMembership {
        group_id: g_high,
        principal_id: alice,
        action_overrides: None,
        level_override: None,
    }];
    let mut alice_dept = PrincipalContext::new(org_id, alice, AuthLevel::new(3)?);
    alice_dept.dept_ids = vec![dept_id];
    alice_dept.group_ids = vec![g_high];

    let dec_dept = IamEvaluator::evaluate(
        &alice_dept,
        &dept_res,
        IamAction::Read,
        &groups,
        &memberships,
        &[],
        &[],
    );
    assert!(dec_dept.allowed);
    assert_eq!(dec_dept.effective_min_level, AuthLevel::new(3)?);

    // 2. Synthetic principal with dept_admin_ids ONLY (not in dept_ids) is denied Read on Dept scope
    // (matching SQL iam.rls_can_access) while allowed Write.
    let mut synth_admin = PrincipalContext::new(org_id, alice, AuthLevel::new(5)?);
    synth_admin.dept_admin_ids = vec![dept_id];
    let dec_read = IamEvaluator::evaluate(
        &synth_admin,
        &dept_res,
        IamAction::Read,
        &[],
        &[],
        &[],
        &[],
    );
    assert!(!dec_read.allowed);
    assert!(
        IamEvaluator::evaluate(
            &synth_admin,
            &dept_res,
            IamAction::Write,
            &[],
            &[],
            &[],
            &[],
        )
        .allowed
    );

    // 3. IamPolicy Allow with higher min_auth_level_override updates effective_min_level
    let group_res = make_resource(org_id, dept_id, owner, AccessScope::Group, 4)?;
    let pol_l9 = IamPolicy {
        id: Uuid::new_v4(),
        name: "l9-allow".into(),
        statements: vec![PolicyStatement {
            sid: None,
            effect: PolicyEffect::Allow,
            actions: vec![IamAction::Read],
            resources: vec!["*".into()],
            condition: None,
        }],
        min_auth_level_override: Some(AuthLevel::new(9)?),
    };
    let alice_l9 = PrincipalContext::new(org_id, alice, AuthLevel::new(9)?);
    let dec_pol = IamEvaluator::evaluate(
        &alice_l9,
        &group_res,
        IamAction::Read,
        &[],
        &[],
        &[],
        &[pol_l9],
    );
    assert!(dec_pol.allowed);
    assert_eq!(dec_pol.effective_min_level, AuthLevel::new(9)?);

    Ok(())
}
