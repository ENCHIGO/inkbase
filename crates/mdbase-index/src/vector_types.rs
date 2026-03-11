/// A single result from a vector similarity search.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VectorSearchResult {
    /// The document that contains this match.
    pub doc_id: String,
    /// The specific block within the document.
    pub block_id: String,
    /// File path of the matched document.
    pub path: String,
    /// First 200 characters of the block text.
    pub text_preview: String,
    /// Cosine similarity score (higher is more similar).
    pub score: f32,
}
