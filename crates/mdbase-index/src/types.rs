/// A single result from a full-text search query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    /// The document that contains this match.
    pub doc_id: String,
    /// The specific block within the document, if the match is block-level.
    pub block_id: Option<String>,
    /// File path of the matched document.
    pub path: String,
    /// Document title, if available.
    pub title: Option<String>,
    /// Block type tag (e.g. "paragraph", "heading"), if block-level.
    pub block_type: Option<String>,
    /// A text snippet highlighting the matched region.
    pub snippet: String,
    /// Tantivy relevance score.
    pub score: f32,
}
