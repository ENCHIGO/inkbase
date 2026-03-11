use mdbase_core::Result;

/// A provider that converts text into dense float vectors for semantic search.
///
/// Implementations range from simple bag-of-words hashing (built-in) to neural
/// model inference via ONNX Runtime or external APIs.
pub trait EmbeddingProvider: Send + Sync {
    /// Returns the dimensionality of the embedding vectors produced by this provider.
    fn dimension(&self) -> usize;

    /// Returns a stable identifier for the model or algorithm (e.g. "tfidf-hash-256").
    fn model_id(&self) -> &str;

    /// Embed a single text string into a dense vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in one call.
    ///
    /// The default implementation calls [`embed`](Self::embed) in a loop.
    /// Providers that support true batching should override this for efficiency.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
