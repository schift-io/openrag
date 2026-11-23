use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use openrag_core::OpenRagError;
use openrag_db::RlsTransaction;
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, QueryBuilder, Row};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagTable {
    CorpusDocuments,
    CorpusChunks,
    WikiPages,
    MemoryEntries,
}

impl TagTable {
    pub const ALL: [Self; 4] = [
        Self::CorpusDocuments,
        Self::CorpusChunks,
        Self::WikiPages,
        Self::MemoryEntries,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CorpusDocuments => "corpus.documents",
            Self::CorpusChunks => "corpus.chunks",
            Self::WikiPages => "wiki.pages",
            Self::MemoryEntries => "memory.entries",
        }
    }

    pub const fn has_status_active(self) -> bool {
        matches!(self, Self::WikiPages | Self::MemoryEntries)
    }
}

impl fmt::Display for TagTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TagTable {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "corpus.documents" => Ok(Self::CorpusDocuments),
            "corpus.chunks" => Ok(Self::CorpusChunks),
            "wiki.pages" => Ok(Self::WikiPages),
            "memory.entries" => Ok(Self::MemoryEntries),
            other => Err(OpenRagError::Validation(format!(
                "table '{other}' does not support first-class tags"
            ))),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TagFilter {
    #[serde(default)]
    pub tags_any: Vec<String>,
    #[serde(default)]
    pub tags_all: Vec<String>,
    #[serde(default)]
    pub tags_exclude: Vec<String>,
    #[serde(default)]
    pub expand_tag_aliases: bool,
}

impl TagFilter {
    pub fn is_empty(&self) -> bool {
        self.tags_any.is_empty() && self.tags_all.is_empty() && self.tags_exclude.is_empty()
    }

    pub fn normalized(&self) -> Self {
        Self {
            tags_any: dedup_normalized(&self.tags_any),
            tags_all: dedup_normalized(&self.tags_all),
            tags_exclude: dedup_normalized(&self.tags_exclude),
            expand_tag_aliases: self.expand_tag_aliases,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedTagQuery {
    pub clean_query: String,
    pub inline_tags: Vec<String>,
    #[serde(default)]
    pub excluded_tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TagFacet {
    pub tag: String,
    pub count: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpandedTagFilter {
    pub tags_any: Vec<String>,
    pub tags_all_groups: Vec<Vec<String>>,
    pub tags_exclude: Vec<String>,
}

impl ExpandedTagFilter {
    pub fn is_empty(&self) -> bool {
        self.tags_any.is_empty()
            && self.tags_all_groups.is_empty()
            && self.tags_exclude.is_empty()
    }

    pub fn matches_tags(&self, raw_tags: &[String]) -> bool {
        let norm: BTreeSet<String> = raw_tags
            .iter()
            .map(|t| normalize_tag(t))
            .filter(|t| !t.is_empty())
            .collect();
        let compact: BTreeSet<String> = norm.iter().map(|t| compact_tag(t)).collect();

        let has_tag = |candidate: &str| -> bool {
            let n = normalize_tag(candidate);
            let c = compact_tag(&n);
            norm.contains(&n) || (!c.is_empty() && compact.contains(&c))
        };

        if !self.tags_any.is_empty() && !self.tags_any.iter().any(|t| has_tag(t)) {
            return false;
        }
        for group in &self.tags_all_groups {
            if !group.is_empty() && !group.iter().any(|t| has_tag(t)) {
                return false;
            }
        }
        if self.tags_exclude.iter().any(|t| has_tag(t)) {
            return false;
        }
        true
    }
}

pub fn normalize_tag(raw: &str) -> String {
    let mut s = raw.trim();
    if let Some(stripped) = s.strip_prefix('#') {
        s = stripped;
    } else if s
        .as_bytes()
        .get(..4)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"tag:"))
    {
        s = &s[4..];
    }
    let nfkc: String = s.nfkc().collect();
    nfkc.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn compact_tag(raw: &str) -> String {
    normalize_tag(raw)
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn dedup_normalized(items: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for item in items {
        let norm = normalize_tag(item);
        if !norm.is_empty() && seen.insert(norm.clone()) {
            out.push(norm);
        }
    }
    out
}

/// Parses inline `#tag`, `tag:foo`, `-#neg`, and `-tag:neg` tokens from a search query.
pub fn parse_inline_tags(query: &str) -> ParsedTagQuery {
    let mut clean_tokens = Vec::new();
    let mut inline_tags = Vec::new();
    let mut excluded_tags = Vec::new();

    for token in query.split_whitespace() {
        let (is_neg, body) = if let Some(rest) = token.strip_prefix("-#") {
            (true, Some(rest))
        } else if token
            .as_bytes()
            .get(..5)
            .is_some_and(|p| p.eq_ignore_ascii_case(b"-tag:"))
        {
            (true, Some(&token[5..]))
        } else if let Some(rest) = token.strip_prefix('#') {
            (false, Some(rest))
        } else if token
            .as_bytes()
            .get(..4)
            .is_some_and(|p| p.eq_ignore_ascii_case(b"tag:"))
        {
            (false, Some(&token[4..]))
        } else {
            (false, None)
        };

        if let Some(raw_tag) = body {
            let norm = normalize_tag(raw_tag);
            if !norm.is_empty() {
                if is_neg {
                    if !excluded_tags.contains(&norm) {
                        excluded_tags.push(norm);
                    }
                } else if !inline_tags.contains(&norm) {
                    inline_tags.push(norm);
                }
                continue;
            }
        }
        clean_tokens.push(token);
    }

    ParsedTagQuery {
        clean_query: clean_tokens.join(" "),
        inline_tags,
        excluded_tags,
    }
}

pub async fn resolve_tag_aliases_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    raw_tag: &str,
) -> Result<Vec<String>, OpenRagError> {
    let norm = normalize_tag(raw_tag);
    if norm.is_empty() {
        return Ok(Vec::new());
    }
    let compact = compact_tag(&norm);

    let rows = sqlx::query(
        "WITH matched_concepts AS (
             SELECT DISTINCT ca.concept_id
             FROM wiki.concept_aliases AS ca
             WHERE ca.org_id = $1
               AND (ca.normalized_alias = $2 OR ca.compact_alias = $3)
             UNION
             SELECT c.id AS concept_id
             FROM wiki.concepts AS c
             WHERE c.org_id = $1
               AND c.canonical_key = $2
         )
         SELECT DISTINCT term
         FROM (
             SELECT ca.normalized_alias AS term
             FROM wiki.concept_aliases AS ca
             JOIN matched_concepts AS mc ON mc.concept_id = ca.concept_id
             WHERE ca.org_id = $1
             UNION
             SELECT ca.compact_alias AS term
             FROM wiki.concept_aliases AS ca
             JOIN matched_concepts AS mc ON mc.concept_id = ca.concept_id
             WHERE ca.org_id = $1
             UNION
             SELECT c.canonical_key AS term
             FROM wiki.concepts AS c
             JOIN matched_concepts AS mc ON mc.concept_id = c.id
             WHERE c.org_id = $1
         ) AS all_terms
         ORDER BY term ASC",
    )
    .bind(org_id)
    .bind(&norm)
    .bind(&compact)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed to resolve tag aliases: {e}")))?;

    let mut out = BTreeSet::new();
    out.insert(norm);
    if !compact.is_empty() {
        out.insert(compact);
    }
    for row in rows {
        let term: String = row.try_get("term").map_err(|e| {
            OpenRagError::Database(format!("invalid term in resolve_tag_aliases_rls: {e}"))
        })?;
        let t = normalize_tag(&term);
        if !t.is_empty() {
            out.insert(t);
        }
    }
    Ok(out.into_iter().collect())
}

pub async fn expand_tag_filter_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    filter: &TagFilter,
) -> Result<ExpandedTagFilter, OpenRagError> {
    let cleaned = filter.normalized();
    if !cleaned.expand_tag_aliases {
        return Ok(ExpandedTagFilter {
            tags_any: cleaned.tags_any,
            tags_all_groups: cleaned.tags_all.into_iter().map(|t| vec![t]).collect(),
            tags_exclude: cleaned.tags_exclude,
        });
    }

    let mut any_set = BTreeSet::new();
    for t in &cleaned.tags_any {
        for expanded in resolve_tag_aliases_rls(tx, org_id, t).await? {
            any_set.insert(expanded);
        }
    }

    let mut all_groups = Vec::with_capacity(cleaned.tags_all.len());
    for t in &cleaned.tags_all {
        all_groups.push(resolve_tag_aliases_rls(tx, org_id, t).await?);
    }

    let mut excl_set = BTreeSet::new();
    for t in &cleaned.tags_exclude {
        for expanded in resolve_tag_aliases_rls(tx, org_id, t).await? {
            excl_set.insert(expanded);
        }
    }

    Ok(ExpandedTagFilter {
        tags_any: any_set.into_iter().collect(),
        tags_all_groups: all_groups,
        tags_exclude: excl_set.into_iter().collect(),
    })
}

pub fn expand_tag_filter_in_memory<F>(filter: &TagFilter, alias_lookup: F) -> ExpandedTagFilter
where
    F: Fn(&str) -> Vec<String>,
{
    let cleaned = filter.normalized();
    if !cleaned.expand_tag_aliases {
        return ExpandedTagFilter {
            tags_any: cleaned.tags_any,
            tags_all_groups: cleaned.tags_all.into_iter().map(|t| vec![t]).collect(),
            tags_exclude: cleaned.tags_exclude,
        };
    }

    let expand_one = |t: &str| -> Vec<String> {
        let mut set = BTreeSet::new();
        let n = normalize_tag(t);
        if !n.is_empty() {
            set.insert(n);
        }
        for a in alias_lookup(t) {
            let na = normalize_tag(&a);
            if !na.is_empty() {
                set.insert(na);
            }
        }
        set.into_iter().collect()
    };

    let mut any_set = BTreeSet::new();
    for t in &cleaned.tags_any {
        for exp in expand_one(t) {
            any_set.insert(exp);
        }
    }
    let all_groups = cleaned.tags_all.iter().map(|t| expand_one(t)).collect();
    let mut excl_set = BTreeSet::new();
    for t in &cleaned.tags_exclude {
        for exp in expand_one(t) {
            excl_set.insert(exp);
        }
    }

    ExpandedTagFilter {
        tags_any: any_set.into_iter().collect(),
        tags_all_groups: all_groups,
        tags_exclude: excl_set.into_iter().collect(),
    }
}

pub fn push_tag_filter_predicates<'a>(
    qb: &mut QueryBuilder<'a, Postgres>,
    col_expr: &str,
    expanded: &'a ExpandedTagFilter,
) {
    if !expanded.tags_any.is_empty() {
        qb.push(" AND (");
        qb.push(col_expr);
        qb.push(" && ");
        qb.push_bind(&expanded.tags_any);
        qb.push("::text[])");
    }
    for group in &expanded.tags_all_groups {
        if !group.is_empty() {
            qb.push(" AND (");
            qb.push(col_expr);
            qb.push(" && ");
            qb.push_bind(group);
            qb.push("::text[])");
        }
    }
    if !expanded.tags_exclude.is_empty() {
        qb.push(" AND NOT (");
        qb.push(col_expr);
        qb.push(" && ");
        qb.push_bind(&expanded.tags_exclude);
        qb.push("::text[])");
    }
}

pub async fn add_tags_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    table: TagTable,
    target_id: Uuid,
    tags: &[String],
    propagate_to_chunks: bool,
) -> Result<Vec<String>, OpenRagError> {
    let norm_tags = dedup_normalized(tags);
    let sql = format!(
        "UPDATE {}
         SET tags = ARRAY(
             SELECT DISTINCT t
             FROM unnest(tags || $1::text[]) AS t
             WHERE length(btrim(t)) > 0
             ORDER BY t ASC
         )
         WHERE org_id = $2 AND id = $3
         RETURNING tags",
        table.as_str()
    );
    let updated: Option<Vec<String>> = sqlx::query_scalar(&sql)
        .bind(&norm_tags)
        .bind(org_id)
        .bind(target_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to add tags on {table}: {e}")))?;

    let result_tags = updated.ok_or_else(|| {
        OpenRagError::NotFound(format!("{table} row {target_id} not found or inaccessible"))
    })?;

    if table == TagTable::CorpusDocuments && propagate_to_chunks && !norm_tags.is_empty() {
        sqlx::query(
            "UPDATE corpus.chunks
             SET tags = ARRAY(
                 SELECT DISTINCT t
                 FROM unnest(tags || $1::text[]) AS t
                 WHERE length(btrim(t)) > 0
                 ORDER BY t ASC
             )
             WHERE org_id = $2 AND document_id = $3",
        )
        .bind(&norm_tags)
        .bind(org_id)
        .bind(target_id)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to propagate tags to chunks: {e}")))?;
    }

    Ok(result_tags)
}

pub async fn remove_tags_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    table: TagTable,
    target_id: Uuid,
    tags: &[String],
    propagate_to_chunks: bool,
) -> Result<Vec<String>, OpenRagError> {
    let norm_tags = dedup_normalized(tags);
    let sql = format!(
        "UPDATE {}
         SET tags = ARRAY(
             SELECT t
             FROM unnest(tags) AS t
             WHERE NOT (t = ANY($1::text[]))
             ORDER BY t ASC
         )
         WHERE org_id = $2 AND id = $3
         RETURNING tags",
        table.as_str()
    );
    let updated: Option<Vec<String>> = sqlx::query_scalar(&sql)
        .bind(&norm_tags)
        .bind(org_id)
        .bind(target_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to remove tags on {table}: {e}")))?;

    let result_tags = updated.ok_or_else(|| {
        OpenRagError::NotFound(format!("{table} row {target_id} not found or inaccessible"))
    })?;

    if table == TagTable::CorpusDocuments && propagate_to_chunks && !norm_tags.is_empty() {
        sqlx::query(
            "UPDATE corpus.chunks
             SET tags = ARRAY(
                 SELECT t
                 FROM unnest(tags) AS t
                 WHERE NOT (t = ANY($1::text[]))
                 ORDER BY t ASC
             )
             WHERE org_id = $2 AND document_id = $3",
        )
        .bind(&norm_tags)
        .bind(org_id)
        .bind(target_id)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("failed to propagate tag removal to chunks: {e}"))
        })?;
    }

    Ok(result_tags)
}

pub async fn compute_tag_facets_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    tables: &[TagTable],
    bucket_id: Option<Uuid>,
    limit: i64,
) -> Result<Vec<TagFacet>, OpenRagError> {
    if tables.is_empty() || limit <= 0 {
        return Ok(Vec::new());
    }

    let mut qb: QueryBuilder<'_, Postgres> =
        QueryBuilder::new("SELECT tag, count(*)::bigint AS cnt FROM (");
    for (idx, table) in tables.iter().enumerate() {
        if idx > 0 {
            qb.push(" UNION ALL ");
        }
        match table {
            TagTable::CorpusDocuments => {
                qb.push("SELECT unnest(d.tags) AS tag FROM corpus.documents AS d WHERE d.org_id = ");
                qb.push_bind(org_id);
                if let Some(bid) = bucket_id {
                    qb.push(" AND d.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
            TagTable::CorpusChunks => {
                qb.push("SELECT unnest(c.tags) AS tag FROM corpus.chunks AS c");
                if bucket_id.is_some() {
                    qb.push(" JOIN corpus.documents AS cd ON cd.id = c.document_id AND cd.org_id = c.org_id");
                }
                qb.push(" WHERE c.org_id = ");
                qb.push_bind(org_id);
                if let Some(bid) = bucket_id {
                    qb.push(" AND cd.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
            TagTable::WikiPages => {
                qb.push("SELECT unnest(w.tags) AS tag FROM wiki.pages AS w WHERE w.org_id = ");
                qb.push_bind(org_id);
                qb.push(" AND w.status = 'active'");
                if let Some(bid) = bucket_id {
                    qb.push(" AND w.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
            TagTable::MemoryEntries => {
                qb.push("SELECT unnest(m.tags) AS tag FROM memory.entries AS m WHERE m.org_id = ");
                qb.push_bind(org_id);
                qb.push(" AND m.status = 'active'");
                if let Some(bid) = bucket_id {
                    qb.push(" AND m.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
        }
    }
    qb.push(") AS all_tags GROUP BY tag ORDER BY cnt DESC, tag ASC LIMIT ");
    qb.push_bind(limit);

    let rows = qb
        .build()
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to compute tag facets: {e}")))?;

    let mut facets = Vec::with_capacity(rows.len());
    for row in rows {
        let tag: String = row
            .try_get("tag")
            .map_err(|e| OpenRagError::Database(format!("invalid facet tag row: {e}")))?;
        let count: i64 = row
            .try_get("cnt")
            .map_err(|e| OpenRagError::Database(format!("invalid facet cnt row: {e}")))?;
        facets.push(TagFacet { tag, count });
    }
    Ok(facets)
}
