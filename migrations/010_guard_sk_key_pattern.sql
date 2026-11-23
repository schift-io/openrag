-- OpenRAG Migration 010: Widen the seeded sk- API key guard rule on existing orgs
--
-- Orgs seeded before this release carry the original default rule `(?i)sk-[a-z0-9]{20,}`,
-- which misses hyphenated keys (sk-proj-..., sk-ant-api03-...). New orgs are seeded from
-- configs/pipeline.default.yaml with the widened pattern; this brings existing orgs in line.
--
-- Only rows that still hold the exact original seeded value are touched, so rules an
-- operator edited by hand are left alone. Idempotent: a second run matches nothing.

UPDATE pii.guard_rules
   SET pattern_or_category = '(?i)\bsk-[a-z0-9][a-z0-9_-]{15,}'
 WHERE rule_kind = 'regex'
   AND pattern_or_category = '(?i)sk-[a-z0-9]{20,}';
