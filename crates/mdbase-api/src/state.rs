use std::sync::Arc;

use mdbase_embedding::EmbeddingPipeline;
use mdbase_graph::KnowledgeGraph;
use mdbase_index::{FullTextIndex, VectorIndex};
use mdbase_storage::Storage;

/// Shared application state passed to every handler via axum's `State` extractor.
///
/// All fields are wrapped in `Arc` so cloning is cheap (axum clones state for
/// each request). The storage backend uses a trait object to remain agnostic
/// of the concrete implementation.
#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<dyn Storage>,
    pub index: Arc<FullTextIndex>,
    pub graph: Arc<KnowledgeGraph>,
    pub embedder: Arc<EmbeddingPipeline>,
    pub vector_index: Arc<VectorIndex>,
}
