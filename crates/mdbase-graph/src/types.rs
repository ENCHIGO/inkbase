use mdbase_core::types::LinkType;
use serde::Serialize;
use utoipa::ToSchema;

/// A resolved link between two documents in the knowledge graph.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LinkInfo {
    pub source_path: String,
    pub target_path: String,
    pub link_type: LinkType,
    pub anchor_text: String,
}

/// Summary statistics for the knowledge graph.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
}

/// A single node's PageRank score.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PageRankScore {
    pub path: String,
    pub score: f64,
}
