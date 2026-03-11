use std::hash::{Hash, Hasher};

use mdbase_core::Result;

use crate::provider::EmbeddingProvider;

/// A lightweight embedding provider based on the *hashing trick* (feature hashing).
///
/// Each input text is tokenized into words, and each word is hashed into one of
/// `dimension` buckets. A second hash bit selects the sign (+1 / -1) so that
/// inner-product expectations are preserved.  The resulting vector is L2-normalized
/// to unit length, making cosine similarity equivalent to a dot product.
///
/// This technique is the same one used by scikit-learn's `HashingVectorizer` and
/// provides reasonable semantic similarity without any external model or vocabulary.
pub struct TfIdfEmbedder {
    dimension: usize,
}

impl TfIdfEmbedder {
    /// Create a new embedder with the given vector dimension.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` is zero.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "embedding dimension must be positive");
        Self { dimension }
    }

    /// Tokenize text into lowercase word tokens.
    ///
    /// Splits on whitespace and strips leading/trailing ASCII punctuation from
    /// each token, discarding any token that becomes empty after stripping.
    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter_map(|word| {
                let stripped = word
                    .trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase();
                if stripped.is_empty() {
                    None
                } else {
                    Some(stripped)
                }
            })
            .collect()
    }

    /// Deterministic hash for a word using a FNV-1a-style approach.
    ///
    /// We use the standard library's `DefaultHasher` which is SipHash-based and
    /// deterministic within a single binary build.  This is sufficient because we
    /// only need consistency within a single index lifetime, and the model_id
    /// captures the algorithm version.
    fn word_hash(word: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for TfIdfEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl EmbeddingProvider for TfIdfEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        "tfidf-hash-v1"
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = Self::tokenize(text);
        let mut vector = vec![0.0f32; self.dimension];

        for token in &tokens {
            let h = Self::word_hash(token);
            let dim_index = (h as usize) % self.dimension;
            // Use a separate bit to decide sign: improves quality by reducing
            // hash-collision bias (same idea as signed random projections).
            let sign = if (h >> 32) & 1 == 0 { 1.0f32 } else { -1.0f32 };
            vector[dim_index] += sign;
        }

        // L2-normalize so that cosine similarity == dot product.
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut vector {
                *v /= norm;
            }
        }

        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_produces_correct_dimension() {
        let embedder = TfIdfEmbedder::new(128);
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 128);

        let embedder_default = TfIdfEmbedder::default();
        let vec_default = embedder_default.embed("hello world").unwrap();
        assert_eq!(vec_default.len(), 256);
    }

    #[test]
    fn embed_produces_normalized_vectors() {
        let embedder = TfIdfEmbedder::default();
        let vec = embedder.embed("the quick brown fox jumps over the lazy dog").unwrap();

        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit norm, got {}",
            norm
        );
    }

    #[test]
    fn empty_text_returns_zero_vector() {
        let embedder = TfIdfEmbedder::default();
        let vec = embedder.embed("").unwrap();
        assert_eq!(vec.len(), 256);
        assert!(vec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn punctuation_only_text_returns_zero_vector() {
        let embedder = TfIdfEmbedder::default();
        let vec = embedder.embed("... --- !!!").unwrap();
        assert_eq!(vec.len(), 256);
        assert!(vec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn batch_consistency() {
        let embedder = TfIdfEmbedder::default();
        let texts = ["hello world", "rust is great", "semantic search"];

        let batch = embedder.embed_batch(&texts).unwrap();
        assert_eq!(batch.len(), 3);

        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            assert_eq!(batch[i], single, "batch[{}] differs from single embed", i);
        }
    }

    #[test]
    fn deterministic_across_calls() {
        let embedder = TfIdfEmbedder::default();
        let v1 = embedder.embed("reproducibility matters").unwrap();
        let v2 = embedder.embed("reproducibility matters").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn similar_texts_have_higher_similarity_than_unrelated() {
        let embedder = TfIdfEmbedder::default();
        let va = embedder.embed("rust programming language").unwrap();
        let vb = embedder.embed("rust programming systems").unwrap();
        let vc = embedder.embed("chocolate cake recipe").unwrap();

        let sim_ab: f32 = va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum();
        let sim_ac: f32 = va.iter().zip(vc.iter()).map(|(a, b)| a * b).sum();

        assert!(
            sim_ab > sim_ac,
            "expected similar texts to score higher: sim(a,b)={} vs sim(a,c)={}",
            sim_ab,
            sim_ac
        );
    }

    #[test]
    #[should_panic(expected = "embedding dimension must be positive")]
    fn zero_dimension_panics() {
        TfIdfEmbedder::new(0);
    }
}
