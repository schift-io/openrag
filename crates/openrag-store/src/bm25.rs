use std::collections::{BTreeMap, HashMap};

use openrag_core::OpenRagError;
use openrag_db::{Capabilities, RlsTransaction};
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, QueryBuilder, Row};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::tags::{expand_tag_filter_rls, push_tag_filter_predicates, ExpandedTagFilter, TagFilter};
use crate::types::VectorTable;

pub const BM25_K1: f64 = 1.2;
pub const BM25_B: f64 = 0.0;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bm25DocInput {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bm25Query {
    pub org_id: Uuid,
    pub table: VectorTable,
    pub query_text: String,
    #[serde(default)]
    pub expanded_terms: Vec<String>,
    #[serde(default)]
    pub bucket_id: Option<Uuid>,
    #[serde(default)]
    pub tag_filter: Option<TagFilter>,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bm25Hit {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub table: VectorTable,
    pub score: f64,
    pub snippet: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Tokenizes text for BM25 matching with Korean legal clause normalization
/// (e.g., `"제3조"` emits `["제3조", "3조"]` so `"3조 출장 여비"` matches `"제3조(출장 여비 지급 기준)"`).
pub fn tokenize_bm25(text: &str) -> Vec<String> {
    let nfkc: String = text.nfkc().collect();
    let mut out = Vec::new();
    for raw in nfkc.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        let lower = raw.to_lowercase();
        out.push(lower.clone());

        if let Some(stripped) = strip_korean_clause_prefix(&lower) {
            if stripped != lower && !stripped.is_empty() {
                out.push(stripped);
            }
        }
    }
    out
}

fn strip_korean_clause_prefix(token: &str) -> Option<String> {
    let rest = token.strip_prefix('제')?;
    let mut chars = rest.chars().peekable();
    let mut digits = String::new();
    while let Some(&ch) = chars.peek() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    if digits.is_empty() {
        return None;
    }
    let tail: String = chars.collect();
    if tail.starts_with('조') || tail.starts_with('항') || tail.starts_with('호') || tail.starts_with('목') {
        Some(format!("{digits}{tail}"))
    } else {
        None
    }
}

/// Exact in-memory BM25 index matching `vectors/src/keyword.rs`:
/// `k1 = 1.2`, `b = 0.0` (length-invariant), `idf = ((N - df + 0.5) / (df + 0.5)).ln_1p()`.
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    docs: Vec<IndexedBm25Doc>,
    df: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct IndexedBm25Doc {
    id: Uuid,
    document_id: Option<Uuid>,
    text: String,
    tags: Vec<String>,
    tf: HashMap<String, usize>,
}

impl Bm25Index {
    pub fn from_docs(items: &[Bm25DocInput]) -> Self {
        let mut docs = Vec::with_capacity(items.len());
        let mut df: HashMap<String, usize> = HashMap::new();

        for item in items {
            let mut tf: HashMap<String, usize> = HashMap::new();
            for tok in tokenize_bm25(&item.text) {
                *tf.entry(tok).or_insert(0) += 1;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            docs.push(IndexedBm25Doc {
                id: item.id,
                document_id: item.document_id,
                text: item.text.clone(),
                tags: item.tags.clone(),
                tf,
            });
        }

        Self { docs, df }
    }

    pub fn search(&self, table: VectorTable, query: &str, limit: usize) -> Vec<Bm25Hit> {
        self.search_with_expansion(table, query, &[], None, limit)
    }

    pub fn search_with_expansion(
        &self,
        table: VectorTable,
        query: &str,
        expanded_terms: &[String],
        tag_filter: Option<&ExpandedTagFilter>,
        limit: usize,
    ) -> Vec<Bm25Hit> {
        if limit == 0 || self.docs.is_empty() {
            return Vec::new();
        }
        let mut query_terms: BTreeMap<String, f64> = BTreeMap::new();
        for tok in tokenize_bm25(query) {
            query_terms.entry(tok).or_insert(1.0);
        }
        if query_terms.is_empty() && expanded_terms.is_empty() {
            return Vec::new();
        }
        for extra in expanded_terms {
            for tok in tokenize_bm25(extra) {
                query_terms.entry(tok).or_insert(0.85);
            }
        }
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n_docs = self.docs.len() as f64;
        let mut hits = Vec::new();

        for doc in &self.docs {
            if let Some(tf_filter) = tag_filter {
                if !tf_filter.matches_tags(&doc.tags) {
                    continue;
                }
            }
            let mut score = 0.0_f64;
            for (term, weight) in &query_terms {
                let Some(&tf_raw) = doc.tf.get(term) else {
                    continue;
                };
                let tf = tf_raw as f64;
                let df = *self.df.get(term).unwrap_or(&0) as f64;
                let idf = ((n_docs - df + 0.5) / (df + 0.5)).max(0.0).ln_1p();
                let term_score = weight * idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1);
                score += term_score;
            }
            if score > 0.0 {
                hits.push(Bm25Hit {
                    id: doc.id,
                    document_id: doc.document_id,
                    table,
                    score,
                    snippet: make_snippet(&doc.text, 240),
                    tags: doc.tags.clone(),
                });
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(limit);
        hits
    }
}

fn make_snippet(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let prefix: String = trimmed.chars().take(max_chars).collect();
        format!("{prefix}...")
    }
}

pub struct Bm25Store;

impl Bm25Store {
    pub async fn search_rls(
        tx: &mut RlsTransaction<'_>,
        caps: &Capabilities,
        query: &Bm25Query,
    ) -> Result<Vec<Bm25Hit>, OpenRagError> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let base_tokens = tokenize_bm25(&query.query_text);
        if base_tokens.is_empty() && query.expanded_terms.is_empty() {
            return Ok(Vec::new());
        }

        let expanded_tags = if let Some(tf) = &query.tag_filter {
            Some(expand_tag_filter_rls(tx, query.org_id, tf).await?)
        } else {
            None
        };

        if caps.has_pg_search && query.table.has_tags_column() {
            if let Ok(hits) = Self::search_pg_search(tx, query, expanded_tags.as_ref()).await {
                return Ok(hits);
            }
        }

        Self::search_pg_trgm_fallback(tx, query, expanded_tags.as_ref()).await
    }

    async fn search_pg_search(
        tx: &mut RlsTransaction<'_>,
        query: &Bm25Query,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<Bm25Hit>, OpenRagError> {
        let mut combined_query = query.query_text.clone();
        if !query.expanded_terms.is_empty() {
            combined_query.push(' ');
            combined_query.push_str(&query.expanded_terms.join(" "));
        }
        let clean_search = sanitize_pg_search_query(&combined_query);
        if clean_search.is_empty() {
            return Ok(Vec::new());
        }

        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT t.id, ");
        if query.table.has_document_id() {
            qb.push("t.document_id");
        } else {
            qb.push("NULL::uuid AS document_id");
        }
        if query.table.has_tags_column() {
            qb.push(", t.tags");
        } else {
            qb.push(", ARRAY[]::text[] AS tags");
        }
        qb.push(", t.bm25_text, paradedb.score(t.id)::double precision AS bm25_score FROM ");
        qb.push(query.table.as_str());
        qb.push(" AS t");
        if query.table == VectorTable::CorpusChunks && query.bucket_id.is_some() {
            qb.push(" JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id");
        }
        qb.push(" WHERE t.org_id = ");
        qb.push_bind(query.org_id);
        qb.push(" AND t.bm25_text @@@ ");
        qb.push_bind(clean_search);

        if query.table.has_status_active() {
            qb.push(" AND t.status = 'active'");
        }
        if let Some(bid) = query.bucket_id {
            match query.table {
                VectorTable::CorpusChunks => {
                    qb.push(" AND d.bucket_id = ");
                    qb.push_bind(bid);
                }
                VectorTable::WikiPages | VectorTable::MemoryEntries => {
                    qb.push(" AND t.bucket_id = ");
                    qb.push_bind(bid);
                }
                VectorTable::CorpusPages | VectorTable::CorpusPageFigures => {}
            }
        }
        if let Some(tf) = expanded_tags {
            if query.table.has_tags_column() {
                push_tag_filter_predicates(&mut qb, "t.tags", tf);
            }
        }
        qb.push(" ORDER BY bm25_score DESC, t.id ASC LIMIT ");
        qb.push_bind(query.limit as i64);

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("pg_search query failed: {e}")))?;

        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid id row: {e}")))?;
            let document_id: Option<Uuid> = row
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(format!("invalid document_id row: {e}")))?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid tags row: {e}")))?;
            let bm25_text: String = row
                .try_get("bm25_text")
                .map_err(|e| OpenRagError::Database(format!("invalid bm25_text row: {e}")))?;
            let score: f64 = row
                .try_get("bm25_score")
                .map_err(|e| OpenRagError::Database(format!("invalid bm25_score row: {e}")))?;
            hits.push(Bm25Hit {
                id,
                document_id,
                table: query.table,
                score,
                snippet: make_snippet(&bm25_text, 240),
                tags,
            });
        }
        Ok(hits)
    }

    async fn search_pg_trgm_fallback(
        tx: &mut RlsTransaction<'_>,
        query: &Bm25Query,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<Bm25Hit>, OpenRagError> {
        // CorpusChunks, WikiPages, MemoryEntries have bm25_text; CorpusPages uses redacted/raw markdown
        let text_col = match query.table {
            VectorTable::CorpusChunks | VectorTable::WikiPages | VectorTable::MemoryEntries => {
                "t.bm25_text"
            }
            VectorTable::CorpusPages => "COALESCE(t.redacted_markdown, t.raw_markdown, '')",
            VectorTable::CorpusPageFigures => "COALESCE(t.caption, '')",
        };

        let mut all_tokens: Vec<String> = tokenize_bm25(&query.query_text);
        for extra in &query.expanded_terms {
            all_tokens.extend(tokenize_bm25(extra));
        }
        let mut patterns: Vec<String> = Vec::new();
        for tok in &all_tokens {
            let pat = format!("%{}%", escape_like(tok));
            if !patterns.contains(&pat) {
                patterns.push(pat);
            }
        }
        if patterns.is_empty() {
            return Ok(Vec::new());
        }

        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT t.id, ");
        if query.table.has_document_id() {
            qb.push("t.document_id");
        } else {
            qb.push("NULL::uuid AS document_id");
        }
        if query.table.has_tags_column() {
            qb.push(", t.tags");
        } else {
            qb.push(", ARRAY[]::text[] AS tags");
        }
        qb.push(", ");
        qb.push(text_col);
        qb.push(" AS text_body FROM ");
        qb.push(query.table.as_str());
        qb.push(" AS t");
        if matches!(
            query.table,
            VectorTable::CorpusChunks | VectorTable::CorpusPages | VectorTable::CorpusPageFigures
        ) && query.bucket_id.is_some()
        {
            qb.push(" JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id");
        }
        qb.push(" WHERE t.org_id = ");
        qb.push_bind(query.org_id);
        if query.table.has_status_active() {
            qb.push(" AND t.status = 'active'");
        }
        if let Some(bid) = query.bucket_id {
            match query.table {
                VectorTable::CorpusChunks
                | VectorTable::CorpusPages
                | VectorTable::CorpusPageFigures => {
                    qb.push(" AND d.bucket_id = ");
                    qb.push_bind(bid);
                }
                VectorTable::WikiPages | VectorTable::MemoryEntries => {
                    qb.push(" AND t.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
        }
        if let Some(tf) = expanded_tags {
            if query.table.has_tags_column() {
                push_tag_filter_predicates(&mut qb, "t.tags", tf);
            }
        }

        // Fetch RLS-visible candidates matching any token OR trigram similarity so exact BM25 IDF/TF scoring runs over real docs
        qb.push(" AND (");
        qb.push(text_col);
        qb.push(" ILIKE ANY(");
        qb.push_bind(patterns);
        qb.push("::text[]) OR ");
        qb.push(text_col);
        qb.push(" % ");
        qb.push_bind(&query.query_text);
        qb.push(") LIMIT 500");

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("pg_trgm fallback query failed: {e}")))?;

        let mut docs = Vec::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid id row: {e}")))?;
            let document_id: Option<Uuid> = row
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(format!("invalid document_id row: {e}")))?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid tags row: {e}")))?;
            let text_body: String = row
                .try_get("text_body")
                .map_err(|e| OpenRagError::Database(format!("invalid text_body row: {e}")))?;
            docs.push(Bm25DocInput {
                id,
                document_id,
                text: text_body,
                tags,
            });
        }

        let index = Bm25Index::from_docs(&docs);
        Ok(index.search_with_expansion(
            query.table,
            &query.query_text,
            &query.expanded_terms,
            expanded_tags,
            query.limit,
        ))
    }
}

fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

fn sanitize_pg_search_query(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}
