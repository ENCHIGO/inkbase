use std::collections::HashMap;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use tracing::debug;

use mdbase_core::error::MdbaseError;

use crate::vector_types::VectorSearchResult;

/// Maximum number of characters stored in `text_preview`.
const TEXT_PREVIEW_LIMIT: usize = 200;

/// In-memory vector index using brute-force cosine similarity.
///
/// All vectors share a fixed dimensionality set at construction time. This
/// implementation is intentionally simple — a linear scan over all stored
/// vectors — which is more than fast enough for knowledge bases with fewer
/// than 100K entries. The internal storage is protected by a `RwLock` so the
/// index can be shared across threads safely.
pub struct VectorIndex {
    inner: RwLock<VectorStore>,
}

struct VectorStore {
    /// Map from composite key `"{doc_id}:{block_id}"` to stored vector.
    vectors: HashMap<String, StoredVector>,
    /// Expected dimensionality of every vector.
    dimension: usize,
}

#[derive(Clone)]
struct StoredVector {
    doc_id: String,
    block_id: String,
    path: String,
    text_preview: String,
    vector: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Cosine similarity helper
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two vectors of equal length.
///
/// Returns 0.0 when either vector has zero magnitude, avoiding a division by
/// zero.  The caller must ensure `a.len() == b.len()`.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    dot / denom
}

// ---------------------------------------------------------------------------
// VectorIndex implementation
// ---------------------------------------------------------------------------

impl VectorIndex {
    /// Create a new, empty vector index that expects vectors of `dimension`
    /// components.
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: RwLock::new(VectorStore {
                vectors: HashMap::new(),
                dimension,
            }),
        }
    }

    /// Insert (or overwrite) a vector for the given document block.
    ///
    /// `text` is truncated to [`TEXT_PREVIEW_LIMIT`] characters and stored as
    /// a preview for search results.  Returns an error if the vector dimension
    /// does not match the index dimension.
    pub fn insert(
        &self,
        doc_id: &str,
        block_id: &str,
        path: &str,
        text: &str,
        vector: Vec<f32>,
    ) -> mdbase_core::Result<()> {
        let mut store = self.inner.write();

        if vector.len() != store.dimension {
            return Err(MdbaseError::IndexError(format!(
                "vector dimension mismatch: expected {}, got {}",
                store.dimension,
                vector.len()
            )));
        }

        let key = composite_key(doc_id, block_id);
        let preview = truncate_text(text, TEXT_PREVIEW_LIMIT);

        store.vectors.insert(
            key,
            StoredVector {
                doc_id: doc_id.to_owned(),
                block_id: block_id.to_owned(),
                path: path.to_owned(),
                text_preview: preview,
                vector,
            },
        );

        debug!(doc_id, block_id, "inserted vector");
        Ok(())
    }

    /// Remove all vectors belonging to the given document.
    ///
    /// Vectors are identified by a `"{doc_id}:"` prefix on their composite
    /// key.
    pub fn delete_document(&self, doc_id: &str) -> mdbase_core::Result<()> {
        let mut store = self.inner.write();
        let prefix = format!("{doc_id}:");
        store.vectors.retain(|k, _| !k.starts_with(&prefix));
        debug!(doc_id, "deleted vectors for document");
        Ok(())
    }

    /// Find the top `limit` vectors most similar to `query_vector`, ranked by
    /// cosine similarity in descending order.
    ///
    /// Returns an empty list when the index contains no vectors. Returns an
    /// error if the query vector dimension does not match the index dimension.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> mdbase_core::Result<Vec<VectorSearchResult>> {
        let store = self.inner.read();

        if query_vector.len() != store.dimension {
            return Err(MdbaseError::IndexError(format!(
                "query vector dimension mismatch: expected {}, got {}",
                store.dimension,
                query_vector.len()
            )));
        }

        if store.vectors.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        // Score every stored vector.
        let mut scored: Vec<(OrderedFloat<f32>, &StoredVector)> = store
            .vectors
            .values()
            .map(|sv| {
                let score = cosine_similarity(query_vector, &sv.vector);
                (OrderedFloat(score), sv)
            })
            .collect();

        // Sort descending by score.
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let results = scored
            .into_iter()
            .take(limit)
            .map(|(score, sv)| VectorSearchResult {
                doc_id: sv.doc_id.clone(),
                block_id: sv.block_id.clone(),
                path: sv.path.clone(),
                text_preview: sv.text_preview.clone(),
                score: score.into_inner(),
            })
            .collect();

        Ok(results)
    }

    /// Return the number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.inner.read().vectors.len()
    }

    /// Return `true` when the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.read().vectors.is_empty()
    }

    /// Remove all stored vectors, leaving the index empty.
    pub fn clear(&self) {
        self.inner.write().vectors.clear();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the composite key used as the HashMap key.
fn composite_key(doc_id: &str, block_id: &str) -> String {
    format!("{doc_id}:{block_id}")
}

/// Truncate `text` to at most `limit` characters, respecting char boundaries.
fn truncate_text(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        // Fast path — most strings are ASCII and `len()` equals char count.
        return text.to_owned();
    }
    // Slow path: find the char boundary at or before `limit` bytes.
    match text.char_indices().nth(limit) {
        Some((byte_idx, _)) => text[..byte_idx].to_owned(),
        None => text.to_owned(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple unit vector along a single axis.
    fn axis_vector(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        v
    }

    // -----------------------------------------------------------------------
    // cosine_similarity
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have similarity ~1.0");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have similarity ~0.0");
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "opposite vectors should have similarity ~-1.0");
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&b, &a), 0.0);
        assert_eq!(cosine_similarity(&a, &a), 0.0);
    }

    // -----------------------------------------------------------------------
    // VectorIndex — insert and search
    // -----------------------------------------------------------------------

    #[test]
    fn insert_and_search_returns_best_match() {
        let idx = VectorIndex::new(3);

        idx.insert("doc1", "b1", "a.md", "about cats", vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert("doc1", "b2", "a.md", "about dogs", vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert("doc2", "b3", "b.md", "about fish", vec![0.0, 0.0, 1.0]).unwrap();

        assert_eq!(idx.len(), 3);

        // Query is aligned with the first axis — "about cats" should be the best match.
        let results = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].block_id, "b1");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn insert_overwrites_existing_key() {
        let idx = VectorIndex::new(2);

        idx.insert("d1", "b1", "a.md", "old text", vec![1.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);

        idx.insert("d1", "b1", "a.md", "new text", vec![0.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 1);

        // The overwritten vector should point along the second axis.
        let results = idx.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(results[0].text_preview, "new text");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // VectorIndex — dimension validation
    // -----------------------------------------------------------------------

    #[test]
    fn insert_wrong_dimension_fails() {
        let idx = VectorIndex::new(3);
        let err = idx.insert("d1", "b1", "x.md", "text", vec![1.0, 2.0]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dimension mismatch"), "error was: {msg}");
    }

    #[test]
    fn search_wrong_dimension_fails() {
        let idx = VectorIndex::new(3);
        idx.insert("d1", "b1", "x.md", "text", vec![1.0, 0.0, 0.0]).unwrap();
        let err = idx.search(&[1.0], 5).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dimension mismatch"), "error was: {msg}");
    }

    // -----------------------------------------------------------------------
    // VectorIndex — delete_document
    // -----------------------------------------------------------------------

    #[test]
    fn delete_document_removes_all_blocks() {
        let idx = VectorIndex::new(2);

        idx.insert("doc1", "b1", "a.md", "block one", vec![1.0, 0.0]).unwrap();
        idx.insert("doc1", "b2", "a.md", "block two", vec![0.0, 1.0]).unwrap();
        idx.insert("doc2", "b3", "b.md", "other doc", vec![0.5, 0.5]).unwrap();

        assert_eq!(idx.len(), 3);

        idx.delete_document("doc1").unwrap();

        assert_eq!(idx.len(), 1);

        // Only doc2's vector remains.
        let results = idx.search(&[1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc2");
    }

    #[test]
    fn delete_nonexistent_document_is_noop() {
        let idx = VectorIndex::new(2);
        idx.insert("doc1", "b1", "a.md", "hello", vec![1.0, 0.0]).unwrap();
        idx.delete_document("no_such_doc").unwrap();
        assert_eq!(idx.len(), 1);
    }

    // -----------------------------------------------------------------------
    // VectorIndex — search edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn search_empty_index_returns_empty() {
        let idx = VectorIndex::new(3);
        let results = idx.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_with_limit_zero_returns_empty() {
        let idx = VectorIndex::new(2);
        idx.insert("d1", "b1", "a.md", "x", vec![1.0, 0.0]).unwrap();
        let results = idx.search(&[1.0, 0.0], 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_limit_exceeds_count() {
        let idx = VectorIndex::new(2);
        idx.insert("d1", "b1", "a.md", "only one", vec![1.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 100).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_results_sorted_descending() {
        let dim = 4;
        let idx = VectorIndex::new(dim);

        // Insert vectors at known angles to the query.
        idx.insert("d1", "b1", "a.md", "best", axis_vector(dim, 0)).unwrap();
        idx.insert("d2", "b2", "b.md", "mid", vec![0.7, 0.7, 0.0, 0.0]).unwrap();
        idx.insert("d3", "b3", "c.md", "worst", axis_vector(dim, 2)).unwrap();

        let results = idx.search(&axis_vector(dim, 0), 3).unwrap();
        assert_eq!(results.len(), 3);

        // Verify descending score order.
        for pair in results.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "results not sorted: {} < {}",
                pair[0].score,
                pair[1].score,
            );
        }

        assert_eq!(results[0].text_preview, "best");
    }

    // -----------------------------------------------------------------------
    // VectorIndex — clear and is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn clear_removes_all_vectors() {
        let idx = VectorIndex::new(2);
        idx.insert("d1", "b1", "a.md", "x", vec![1.0, 0.0]).unwrap();
        idx.insert("d2", "b2", "b.md", "y", vec![0.0, 1.0]).unwrap();
        assert!(!idx.is_empty());

        idx.clear();

        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        let results = idx.search(&[1.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Text preview truncation
    // -----------------------------------------------------------------------

    #[test]
    fn text_preview_truncated_to_limit() {
        let idx = VectorIndex::new(2);
        let long_text = "a".repeat(500);
        idx.insert("d1", "b1", "a.md", &long_text, vec![1.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].text_preview.len(), TEXT_PREVIEW_LIMIT);
    }

    #[test]
    fn text_preview_preserves_short_text() {
        let idx = VectorIndex::new(2);
        idx.insert("d1", "b1", "a.md", "short", vec![1.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].text_preview, "short");
    }

    #[test]
    fn text_preview_handles_multibyte_chars() {
        let idx = VectorIndex::new(2);
        // Each character is 3 bytes in UTF-8; the truncation should be
        // by *character count*, not byte count.
        let text: String = std::iter::repeat('\u{4e16}').take(300).collect();
        idx.insert("d1", "b1", "a.md", &text, vec![1.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].text_preview.chars().count(), TEXT_PREVIEW_LIMIT);
    }
}
