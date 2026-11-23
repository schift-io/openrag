use async_trait::async_trait;
use openrag_core::OpenRagError;
use openrag_store::{l2_normalize, VECTOR_DIM};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

#[async_trait]
pub trait TextEmbedder: Send + Sync {
    fn model_name(&self) -> &str;
    /// Embeds stored content (chunks, pages, wiki pages, memories).
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, OpenRagError>;
    /// Embeds a search query. Asymmetric models use a different instruction prefix for
    /// queries; symmetric embedders keep the default, which is the document embedding.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, OpenRagError> {
        self.embed_text(text).await
    }
}

/// Deterministic offline 256d feature-hashed embedder over NFKC-normalized tokens and char trigrams.
#[derive(Debug, Clone)]
pub struct DeterministicTextEmbedder {
    model: String,
}

impl Default for DeterministicTextEmbedder {
    fn default() -> Self {
        Self {
            model: "deterministic-hash-256d-v1".into(),
        }
    }
}

impl DeterministicTextEmbedder {
    pub fn embed_sync(text: &str) -> Result<Vec<f32>, OpenRagError> {
        let mut vec = vec![0.0_f32; VECTOR_DIM];
        let nfkc: String = text.nfkc().collect();
        let lower = nfkc.to_lowercase();

        for token in lower.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
            hash_feature(&mut vec, token.as_bytes(), 2.0);
            let chars: Vec<char> = token.chars().collect();
            if chars.len() >= 3 {
                for win in chars.windows(3) {
                    let trigram: String = win.iter().collect();
                    hash_feature(&mut vec, trigram.as_bytes(), 1.0);
                }
            }
        }

        if vec.iter().all(|v| *v == 0.0) {
            // Ensure empty/punctuation-only text still yields a deterministic non-zero unit vector
            hash_feature(&mut vec, lower.as_bytes(), 1.0);
            if vec.iter().all(|v| *v == 0.0) {
                vec[0] = 1.0;
            }
        }
        l2_normalize(&vec)
    }
}

fn hash_feature(vec: &mut [f32], bytes: &[u8], weight: f32) {
    let digest = Sha256::digest(bytes);
    let idx = usize::from(digest[0]) % VECTOR_DIM;
    let sign = if digest[1] & 1 == 0 { 1.0 } else { -1.0 };
    vec[idx] += sign * weight;
}

#[async_trait]
impl TextEmbedder for DeterministicTextEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, OpenRagError> {
        Self::embed_sync(text)
    }
}
