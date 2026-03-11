use utoipa::OpenApi;

use crate::handlers;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Inkbase REST API",
        version = "0.1.0",
        description = "Markdown-oriented database middleware for AI agents. Parses Markdown into structural blocks, stores them with full-text/vector/graph indexes.",
        license(name = "MIT")
    ),
    paths(
        handlers::health_check,
        handlers::database_stats,
        handlers::ingest_document,
        handlers::list_documents,
        handlers::get_document,
        handlers::delete_document,
        handlers::get_blocks,
        handlers::get_links,
        handlers::get_backlinks,
        handlers::fulltext_search,
        handlers::semantic_search,
        handlers::graph_stats,
        handlers::shortest_path,
        handlers::graph_pagerank,
        handlers::graph_components,
    ),
    components(schemas(
        handlers::IngestRequest,
        handlers::SearchRequest,
        handlers::ShortestPathQuery,
        handlers::PageRankParams,
        inkbase_core::DocumentRecord,
        inkbase_core::BlockRecord,
        inkbase_core::BlockType,
        inkbase_core::LinkRecord,
        inkbase_core::LinkType,
        inkbase_core::TagRecord,
        inkbase_core::EmbeddingRecord,
        inkbase_graph::GraphStats,
        inkbase_graph::LinkInfo,
        inkbase_graph::PageRankScore,
        inkbase_index::SearchResult,
        inkbase_index::VectorSearchResult,
    )),
    tags(
        (name = "documents", description = "Document CRUD and sub-resources"),
        (name = "search", description = "Full-text and semantic search"),
        (name = "graph", description = "Knowledge graph analysis"),
        (name = "operational", description = "Health and statistics")
    )
)]
pub struct ApiDoc;
