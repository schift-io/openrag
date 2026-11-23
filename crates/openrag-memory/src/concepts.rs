use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use chrono::Utc;
use openrag_core::OpenRagError;
use openrag_db::RlsTransaction;
use openrag_store::{cosine_similarity, validate_vector};
use pgvector::Vector;
use regex::Regex;
use sqlx::Row;
use tokio::sync::RwLock;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::embed::DeterministicTextEmbedder;
use crate::types::{
    Concept, ConceptAlias, ConceptCategory, ConceptSearchMatch, ConceptSource, ResolvedConcept,
};

pub fn normalize_term(raw: &str) -> String {
    let nfkc: String = raw.nfkc().collect();
    nfkc.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn compact_term(raw: &str) -> String {
    normalize_term(raw)
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

pub fn detect_lang_hint(term: &str) -> &'static str {
    let has_hangul = term.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
    let has_ascii = term.chars().any(|c| c.is_ascii_alphabetic());
    match (has_hangul, has_ascii) {
        (true, false) => "ko",
        (false, true) => "en",
        (true, true) => "mul",
        (false, false) => "und",
    }
}

/// Automatically extracts cross-lingual parenthetical pairs such as:
/// `"사과(apple)"`, `"제안요청서(RFP)"`, `"LLM Wiki(세컨드 브레인)"`, `"세컨드 브레인(llmwiki)"`.
pub fn extract_parenthetical_pairs(text: &str) -> Vec<(String, String)> {
    let Ok(re) = Regex::new(
        r"((?:[가-힣]{2,}\s+)?[가-힣]+|[A-Za-z0-9][A-Za-z0-9\-_]*(?:\s+[A-Za-z0-9][A-Za-z0-9\-_]*){0,2})\(([가-힣A-Za-z0-9][가-힣A-Za-z0-9\s\-_]{0,40})\)",
    ) else {
        return Vec::new();
    };
    let mut pairs = Vec::new();
    let mut seen = BTreeSet::new();

    for cap in re.captures_iter(text) {
        let Some(left_m) = cap.get(1) else {
            continue;
        };
        let Some(right_m) = cap.get(2) else {
            continue;
        };
        let mut left = left_m.as_str().trim();
        if let Some((first, second)) = left.split_once(char::is_whitespace) {
            if is_hangul_only(first) && ends_with_korean_particle(first) {
                left = second.trim();
            }
        }
        let right = right_m.as_str().trim();
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let l_hangul = left.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
        let r_hangul = right.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
        let l_ascii = left.chars().any(|c| c.is_ascii_alphabetic());
        let r_ascii = right.chars().any(|c| c.is_ascii_alphabetic());

        if (l_hangul && r_ascii) || (l_ascii && r_hangul) {
            let key = format!("{}={}", normalize_term(left), normalize_term(right));
            if seen.insert(key) {
                pairs.push((left.to_string(), right.to_string()));
            }
        }
    }
    pairs
}

fn is_hangul_only(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}

fn ends_with_korean_particle(word: &str) -> bool {
    const PARTICLES: &[char] = &['은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '로', '도', '만'];
    word.chars().last().is_some_and(|c| PARTICLES.contains(&c))
}

#[derive(Debug, Default)]
struct InMemConceptOrgState {
    concepts: HashMap<Uuid, Concept>,
    concept_vectors: HashMap<Uuid, Vec<f32>>,
    aliases: HashMap<Uuid, ConceptAlias>,
}

/// Offline / Kumiho-compatible in-memory Concept / Alias graph.
#[derive(Debug, Clone, Default)]
pub struct InMemoryConceptStore {
    orgs: Arc<RwLock<HashMap<Uuid, InMemConceptOrgState>>>,
}

impl InMemoryConceptStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn link_terms(
        &self,
        org_id: Uuid,
        principal_id: Option<Uuid>,
        term_a: &str,
        term_b: &str,
        category: ConceptCategory,
        source: ConceptSource,
    ) -> Result<ResolvedConcept, OpenRagError> {
        let norm_a = normalize_term(term_a);
        let norm_b = normalize_term(term_b);
        let comp_a = compact_term(term_a);
        let comp_b = compact_term(term_b);
        if norm_a.is_empty() || norm_b.is_empty() {
            return Err(OpenRagError::Validation(
                "concept terms must not be empty".into(),
            ));
        }

        let mut guard = self.orgs.write().await;
        let state = guard.entry(org_id).or_default();

        let cid_a = find_in_mem_concept_id(state, &norm_a, &comp_a);
        let cid_b = find_in_mem_concept_id(state, &norm_b, &comp_b);
        let now = Utc::now();

        let target_cid = match (cid_a, cid_b) {
            (Some(a), Some(b)) if a == b => a,
            (Some(a), Some(b)) => {
                // Merge concept b into concept a
                let moving_ids: Vec<Uuid> = state
                    .aliases
                    .values()
                    .filter(|al| al.concept_id == b)
                    .map(|al| al.id)
                    .collect();
                let existing_norms: BTreeSet<String> = state
                    .aliases
                    .values()
                    .filter(|al| al.concept_id == a)
                    .map(|al| al.normalized_alias.clone())
                    .collect();
                for aid in moving_ids {
                    if let Some(al) = state.aliases.get(&aid) {
                        if existing_norms.contains(&al.normalized_alias) {
                            state.aliases.remove(&aid);
                            continue;
                        }
                    }
                    if let Some(al) = state.aliases.get_mut(&aid) {
                        al.concept_id = a;
                    }
                }
                state.concepts.remove(&b);
                state.concept_vectors.remove(&b);
                a
            }
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => {
                let cid = Uuid::new_v4();
                let vec = DeterministicTextEmbedder::embed_sync(&format!("{term_a} {term_b}"))?;
                state.concepts.insert(
                    cid,
                    Concept {
                        id: cid,
                        org_id,
                        canonical_key: norm_a.clone(),
                        display_name: term_a.trim().to_string(),
                        category,
                        description: None,
                        wiki_page_id: None,
                        created_by: principal_id,
                        created_at: now,
                        updated_at: now,
                    },
                );
                state.concept_vectors.insert(cid, vec);
                cid
            }
        };

        for (raw, norm, comp) in [
            (term_a.trim(), norm_a, comp_a),
            (term_b.trim(), norm_b, comp_b),
        ] {
            let exists = state
                .aliases
                .values()
                .any(|al| al.concept_id == target_cid && al.normalized_alias == norm);
            if !exists {
                let aid = Uuid::new_v4();
                state.aliases.insert(
                    aid,
                    ConceptAlias {
                        id: aid,
                        org_id,
                        concept_id: target_cid,
                        alias_text: raw.to_string(),
                        normalized_alias: norm,
                        compact_alias: comp,
                        lang: detect_lang_hint(raw).into(),
                        source,
                        confidence: 1.0,
                        created_by: principal_id,
                        created_at: now,
                    },
                );
            }
        }

        if let Some(c) = state.concepts.get_mut(&target_cid) {
            c.updated_at = now;
            if category == ConceptCategory::SafewordAlias {
                c.category = ConceptCategory::SafewordAlias;
            }
        }

        build_resolved_in_mem(state, target_cid)
            .ok_or_else(|| OpenRagError::Internal("missing target concept after link".into()))
    }

    pub async fn unlink_alias(&self, org_id: Uuid, alias_or_id: &str) -> Result<bool, OpenRagError> {
        let mut guard = self.orgs.write().await;
        let Some(state) = guard.get_mut(&org_id) else {
            return Ok(false);
        };
        if let Ok(uid) = Uuid::parse_str(alias_or_id.trim()) {
            return Ok(state.aliases.remove(&uid).is_some());
        }
        let norm = normalize_term(alias_or_id);
        let comp = compact_term(alias_or_id);
        let ids: Vec<Uuid> = state
            .aliases
            .values()
            .filter(|a| a.normalized_alias == norm || (!comp.is_empty() && a.compact_alias == comp))
            .map(|a| a.id)
            .collect();
        let removed = !ids.is_empty();
        for id in ids {
            state.aliases.remove(&id);
        }
        Ok(removed)
    }

    pub async fn resolve(&self, org_id: Uuid, term: &str) -> Option<ResolvedConcept> {
        let norm = normalize_term(term);
        let comp = compact_term(term);
        if norm.is_empty() {
            return None;
        }
        let guard = self.orgs.read().await;
        let state = guard.get(&org_id)?;
        let cid = find_in_mem_concept_id(state, &norm, &comp)?;
        build_resolved_in_mem(state, cid)
    }

    pub async fn expand_terms(&self, org_id: Uuid, terms: &[String]) -> Vec<String> {
        let mut out = BTreeSet::new();
        let guard = self.orgs.read().await;
        let state = guard.get(&org_id);
        for t in terms {
            let norm = normalize_term(t);
            let comp = compact_term(t);
            if norm.is_empty() {
                continue;
            }
            out.insert(norm.clone());
            if let Some(st) = state {
                if let Some(cid) = find_in_mem_concept_id(st, &norm, &comp) {
                    if let Some(c) = st.concepts.get(&cid) {
                        out.insert(c.canonical_key.clone());
                    }
                    for al in st.aliases.values().filter(|a| a.concept_id == cid) {
                        out.insert(al.alias_text.clone());
                        out.insert(al.normalized_alias.clone());
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    pub async fn list_concepts(&self, org_id: Uuid) -> Vec<ResolvedConcept> {
        let guard = self.orgs.read().await;
        let Some(state) = guard.get(&org_id) else {
            return Vec::new();
        };
        let mut out: Vec<ResolvedConcept> = state
            .concepts
            .keys()
            .filter_map(|cid| build_resolved_in_mem(state, *cid))
            .collect();
        out.sort_by(|a, b| a.concept.canonical_key.cmp(&b.concept.canonical_key));
        out
    }

    pub async fn find_similar_by_vector(
        &self,
        org_id: Uuid,
        query_vec: &[f32],
        min_cosine: f32,
        limit: usize,
    ) -> Result<Vec<ConceptSearchMatch>, OpenRagError> {
        validate_vector(query_vec)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let guard = self.orgs.read().await;
        let Some(state) = guard.get(&org_id) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (cid, concept) in &state.concepts {
            if let Some(cvec) = state.concept_vectors.get(cid) {
                let sim = cosine_similarity(query_vec, cvec)?;
                if sim >= min_cosine {
                    out.push(ConceptSearchMatch {
                        concept: concept.clone(),
                        cosine_sim: sim,
                    });
                }
            }
        }
        out.sort_by(|a, b| {
            b.cosine_sim
                .partial_cmp(&a.cosine_sim)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit);
        Ok(out)
    }
}

fn find_in_mem_concept_id(
    state: &InMemConceptOrgState,
    norm: &str,
    comp: &str,
) -> Option<Uuid> {
    if let Some(al) = state
        .aliases
        .values()
        .find(|a| a.normalized_alias == norm || (!comp.is_empty() && a.compact_alias == comp))
    {
        return Some(al.concept_id);
    }
    state
        .concepts
        .values()
        .find(|c| c.canonical_key == norm || (!comp.is_empty() && compact_term(&c.canonical_key) == comp))
        .map(|c| c.id)
}

fn build_resolved_in_mem(state: &InMemConceptOrgState, cid: Uuid) -> Option<ResolvedConcept> {
    let concept = state.concepts.get(&cid)?.clone();
    let mut aliases: Vec<ConceptAlias> = state
        .aliases
        .values()
        .filter(|a| a.concept_id == cid)
        .cloned()
        .collect();
    aliases.sort_by(|a, b| a.normalized_alias.cmp(&b.normalized_alias));
    Some(ResolvedConcept { concept, aliases })
}

/// PostgreSQL-backed Concept / Alias graph under Fail-Closed RLS.
pub struct ConceptGraph;

impl ConceptGraph {
    pub async fn link_terms_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        principal_id: Option<Uuid>,
        term_a: &str,
        term_b: &str,
        category: ConceptCategory,
        source: ConceptSource,
    ) -> Result<ResolvedConcept, OpenRagError> {
        let norm_a = normalize_term(term_a);
        let norm_b = normalize_term(term_b);
        let comp_a = compact_term(term_a);
        let comp_b = compact_term(term_b);
        if norm_a.is_empty() || norm_b.is_empty() {
            return Err(OpenRagError::Validation(
                "concept terms must not be empty".into(),
            ));
        }

        let cid_a = Self::lookup_concept_id_rls(tx, org_id, &norm_a, &comp_a).await?;
        let cid_b = Self::lookup_concept_id_rls(tx, org_id, &norm_b, &comp_b).await?;

        let target_cid = match (cid_a, cid_b) {
            (Some(a), Some(b)) if a == b => a,
            (Some(a), Some(b)) => {
                // Move non-colliding aliases from b -> a and delete absorbed concept b
                sqlx::query(
                    "DELETE FROM wiki.concept_aliases AS b_al
                     WHERE b_al.org_id = $1
                       AND b_al.concept_id = $2
                       AND EXISTS (
                           SELECT 1 FROM wiki.concept_aliases AS a_al
                           WHERE a_al.org_id = $1
                             AND a_al.concept_id = $3
                             AND a_al.normalized_alias = b_al.normalized_alias
                       )",
                )
                .bind(org_id)
                .bind(b)
                .bind(a)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| {
                    OpenRagError::Database(format!("failed to deduplicate merged aliases: {e}"))
                })?;

                sqlx::query(
                    "UPDATE wiki.concept_aliases SET concept_id = $1 WHERE org_id = $2 AND concept_id = $3",
                )
                .bind(a)
                .bind(org_id)
                .bind(b)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("failed to move merged aliases: {e}")))?;

                sqlx::query("DELETE FROM wiki.concepts WHERE org_id = $1 AND id = $2")
                    .bind(org_id)
                    .bind(b)
                    .execute(&mut **tx.tx_mut())
                    .await
                    .map_err(|e| {
                        OpenRagError::Database(format!("failed to delete absorbed concept: {e}"))
                    })?;
                a
            }
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => {
                let vec = DeterministicTextEmbedder::embed_sync(&format!("{term_a} {term_b}"))?;
                sqlx::query_scalar::<_, Uuid>(
                    "INSERT INTO wiki.concepts
                     (org_id, canonical_key, display_name, category, embedding, created_by)
                     VALUES ($1, $2, $3, $4, $5, $6)
                     ON CONFLICT (org_id, canonical_key)
                     DO UPDATE SET updated_at = now()
                     RETURNING id",
                )
                .bind(org_id)
                .bind(&norm_a)
                .bind(term_a.trim())
                .bind(category.as_str())
                .bind(Vector::from(vec))
                .bind(principal_id)
                .fetch_one(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("failed to insert concept: {e}")))?
            }
        };

        for (raw, norm, comp) in [
            (term_a.trim(), &norm_a, &comp_a),
            (term_b.trim(), &norm_b, &comp_b),
        ] {
            sqlx::query(
                "INSERT INTO wiki.concept_aliases
                 (org_id, concept_id, alias_text, normalized_alias, compact_alias, lang, source, created_by)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                 ON CONFLICT (org_id, concept_id, normalized_alias) DO NOTHING",
            )
            .bind(org_id)
            .bind(target_cid)
            .bind(raw)
            .bind(norm)
            .bind(comp)
            .bind(detect_lang_hint(raw))
            .bind(source.as_str())
            .bind(principal_id)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to insert concept alias: {e}")))?;
        }

        Self::get_resolved_by_id_rls(tx, org_id, target_cid)
            .await?
            .ok_or_else(|| OpenRagError::Internal("concept disappeared after link".into()))
    }

    pub async fn unlink_alias_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        alias_or_id: &str,
    ) -> Result<bool, OpenRagError> {
        let rows = if let Ok(uid) = Uuid::parse_str(alias_or_id.trim()) {
            sqlx::query("DELETE FROM wiki.concept_aliases WHERE org_id = $1 AND id = $2")
                .bind(org_id)
                .bind(uid)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("failed to unlink alias id: {e}")))?
                .rows_affected()
        } else {
            let norm = normalize_term(alias_or_id);
            let comp = compact_term(alias_or_id);
            sqlx::query(
                "DELETE FROM wiki.concept_aliases
                 WHERE org_id = $1 AND (normalized_alias = $2 OR compact_alias = $3)",
            )
            .bind(org_id)
            .bind(&norm)
            .bind(&comp)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to unlink alias term: {e}")))?
            .rows_affected()
        };
        Ok(rows > 0)
    }

    pub async fn resolve_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        term: &str,
    ) -> Result<Option<ResolvedConcept>, OpenRagError> {
        let norm = normalize_term(term);
        let comp = compact_term(term);
        if norm.is_empty() {
            return Ok(None);
        }
        let Some(cid) = Self::lookup_concept_id_rls(tx, org_id, &norm, &comp).await? else {
            return Ok(None);
        };
        Self::get_resolved_by_id_rls(tx, org_id, cid).await
    }

    pub async fn expand_terms_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        terms: &[String],
    ) -> Result<Vec<String>, OpenRagError> {
        let mut out = BTreeSet::new();
        for t in terms {
            let norm = normalize_term(t);
            if norm.is_empty() {
                continue;
            }
            out.insert(norm.clone());
            if let Some(resolved) = Self::resolve_rls(tx, org_id, &norm).await? {
                out.insert(resolved.concept.canonical_key);
                for al in resolved.aliases {
                    out.insert(al.alias_text);
                    out.insert(al.normalized_alias);
                }
            }
        }
        Ok(out.into_iter().collect())
    }

    pub async fn list_concepts_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
    ) -> Result<Vec<ResolvedConcept>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, canonical_key, display_name, category, description, wiki_page_id, created_by, created_at, updated_at
             FROM wiki.concepts
             WHERE org_id = $1
             ORDER BY canonical_key ASC",
        )
        .bind(org_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list concepts: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let concept = row_to_concept(&row)?;
            let aliases = Self::fetch_aliases_for_concept_rls(tx, org_id, concept.id).await?;
            out.push(ResolvedConcept { concept, aliases });
        }
        Ok(out)
    }

    pub async fn find_similar_by_vector_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        query_vec: &[f32],
        min_cosine: f32,
        limit: i64,
    ) -> Result<Vec<ConceptSearchMatch>, OpenRagError> {
        validate_vector(query_vec)?;
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let qvec = Vector::from(query_vec.to_vec());
        let rows = sqlx::query(
            "SELECT id, org_id, canonical_key, display_name, category, description, wiki_page_id, created_by, created_at, updated_at,
                    (1.0 - (embedding <=> $2))::real AS cosine_sim
             FROM wiki.concepts
             WHERE org_id = $1
               AND embedding IS NOT NULL
               AND (1.0 - (embedding <=> $2))::real >= $3
             ORDER BY embedding <=> $2 ASC
             LIMIT $4",
        )
        .bind(org_id)
        .bind(qvec)
        .bind(min_cosine)
        .bind(limit)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("concept vector search failed: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let cosine_sim: f32 = row
                .try_get("cosine_sim")
                .map_err(|e| OpenRagError::Database(format!("invalid cosine_sim: {e}")))?;
            out.push(ConceptSearchMatch {
                concept: row_to_concept(&row)?,
                cosine_sim,
            });
        }
        Ok(out)
    }

    pub async fn ingest_parenthetical_pairs_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        principal_id: Option<Uuid>,
        text: &str,
    ) -> Result<Vec<(String, String)>, OpenRagError> {
        let pairs = extract_parenthetical_pairs(text);
        for (a, b) in &pairs {
            let _ = Self::link_terms_rls(
                tx,
                org_id,
                principal_id,
                a,
                b,
                ConceptCategory::Domain,
                ConceptSource::IngestParenthetical,
            )
            .await?;
        }
        Ok(pairs)
    }

    async fn lookup_concept_id_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        norm: &str,
        comp: &str,
    ) -> Result<Option<Uuid>, OpenRagError> {
        let by_alias: Option<Uuid> = sqlx::query_scalar(
            "SELECT concept_id
             FROM wiki.concept_aliases
             WHERE org_id = $1 AND (normalized_alias = $2 OR compact_alias = $3)
             LIMIT 1",
        )
        .bind(org_id)
        .bind(norm)
        .bind(comp)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to lookup concept alias: {e}")))?;
        if by_alias.is_some() {
            return Ok(by_alias);
        }

        let by_key: Option<Uuid> = sqlx::query_scalar(
            "SELECT id FROM wiki.concepts WHERE org_id = $1 AND canonical_key = $2 LIMIT 1",
        )
        .bind(org_id)
        .bind(norm)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to lookup concept key: {e}")))?;
        Ok(by_key)
    }

    async fn get_resolved_by_id_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        concept_id: Uuid,
    ) -> Result<Option<ResolvedConcept>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, canonical_key, display_name, category, description, wiki_page_id, created_by, created_at, updated_at
             FROM wiki.concepts
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(concept_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to load concept: {e}")))?;
        let Some(row) = row else {
            return Ok(None);
        };
        let concept = row_to_concept(&row)?;
        let aliases = Self::fetch_aliases_for_concept_rls(tx, org_id, concept_id).await?;
        Ok(Some(ResolvedConcept { concept, aliases }))
    }

    async fn fetch_aliases_for_concept_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        concept_id: Uuid,
    ) -> Result<Vec<ConceptAlias>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, concept_id, alias_text, normalized_alias, compact_alias, lang, source, confidence, created_by, created_at
             FROM wiki.concept_aliases
             WHERE org_id = $1 AND concept_id = $2
             ORDER BY normalized_alias ASC",
        )
        .bind(org_id)
        .bind(concept_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to load concept aliases: {e}")))?;

        let mut aliases = Vec::with_capacity(rows.len());
        for r in rows {
            let src_str: String = r
                .try_get("source")
                .map_err(|e| OpenRagError::Database(format!("invalid alias source: {e}")))?;
            aliases.push(ConceptAlias {
                id: r
                    .try_get("id")
                    .map_err(|e| OpenRagError::Database(format!("invalid alias id: {e}")))?,
                org_id: r
                    .try_get("org_id")
                    .map_err(|e| OpenRagError::Database(format!("invalid alias org_id: {e}")))?,
                concept_id: r
                    .try_get("concept_id")
                    .map_err(|e| OpenRagError::Database(format!("invalid concept_id: {e}")))?,
                alias_text: r
                    .try_get("alias_text")
                    .map_err(|e| OpenRagError::Database(format!("invalid alias_text: {e}")))?,
                normalized_alias: r
                    .try_get("normalized_alias")
                    .map_err(|e| OpenRagError::Database(format!("invalid normalized_alias: {e}")))?,
                compact_alias: r
                    .try_get("compact_alias")
                    .map_err(|e| OpenRagError::Database(format!("invalid compact_alias: {e}")))?,
                lang: r
                    .try_get("lang")
                    .map_err(|e| OpenRagError::Database(format!("invalid lang: {e}")))?,
                source: src_str.parse()?,
                confidence: r
                    .try_get("confidence")
                    .map_err(|e| OpenRagError::Database(format!("invalid confidence: {e}")))?,
                created_by: r
                    .try_get("created_by")
                    .map_err(|e| OpenRagError::Database(format!("invalid created_by: {e}")))?,
                created_at: r
                    .try_get("created_at")
                    .map_err(|e| OpenRagError::Database(format!("invalid created_at: {e}")))?,
            });
        }
        Ok(aliases)
    }
}

fn row_to_concept(row: &sqlx::postgres::PgRow) -> Result<Concept, OpenRagError> {
    let cat_str: String = row
        .try_get("category")
        .map_err(|e| OpenRagError::Database(format!("invalid concept category: {e}")))?;
    Ok(Concept {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid concept id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("invalid concept org_id: {e}")))?,
        canonical_key: row
            .try_get("canonical_key")
            .map_err(|e| OpenRagError::Database(format!("invalid canonical_key: {e}")))?,
        display_name: row
            .try_get("display_name")
            .map_err(|e| OpenRagError::Database(format!("invalid display_name: {e}")))?,
        category: cat_str.parse()?,
        description: row
            .try_get("description")
            .map_err(|e| OpenRagError::Database(format!("invalid description: {e}")))?,
        wiki_page_id: row
            .try_get("wiki_page_id")
            .map_err(|e| OpenRagError::Database(format!("invalid wiki_page_id: {e}")))?,
        created_by: row
            .try_get("created_by")
            .map_err(|e| OpenRagError::Database(format!("invalid created_by: {e}")))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("invalid created_at: {e}")))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(format!("invalid updated_at: {e}")))?,
    })
}
