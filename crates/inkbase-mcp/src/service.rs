use std::fmt;
use std::future::Future;
use std::sync::Arc;

use rmcp::{
    handler::server::router::tool::ToolRouter,
    model::*,
    schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use serde::Deserialize;
use tracing::{debug, error, info, instrument};

use inkbase_embedding::EmbeddingPipeline;
use inkbase_graph::KnowledgeGraph;
use inkbase_index::{FullTextIndex, VectorIndex};
use inkbase_parser::parse_markdown;
use inkbase_storage::Storage;

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// MCP server exposing Inkbase operations as tools for AI agents.
///
/// Holds a shared reference to the storage backend so all tool calls operate
/// on the same underlying data.
#[derive(Clone)]
pub struct InkbaseService {
    storage: Arc<dyn Storage>,
    index: Option<Arc<FullTextIndex>>,
    graph: Option<Arc<KnowledgeGraph>>,
    embedder: Option<Arc<EmbeddingPipeline>>,
    vector_index: Option<Arc<VectorIndex>>,
    tool_router: ToolRouter<Self>,
}

impl fmt::Debug for InkbaseService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InkbaseService")
            .field("tool_router", &self.tool_router)
            .field("has_index", &self.index.is_some())
            .field("has_graph", &self.graph.is_some())
            .field("has_embedder", &self.embedder.is_some())
            .field("has_vector_index", &self.vector_index.is_some())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Request types — one per tool, each deriving JsonSchema for MCP discovery
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IngestDocumentRequest {
    #[schemars(description = "Relative path of the document (e.g. \"notes/rust.md\")")]
    pub path: String,

    #[schemars(description = "Raw Markdown content of the document, including any YAML frontmatter")]
    pub content: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetDocumentRequest {
    #[schemars(description = "Relative path of the document to retrieve")]
    pub path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct QueryBlocksRequest {
    #[schemars(
        description = "Filter by block type: heading, paragraph, code_block, list, list_item, table, block_quote, thematic_break, image, html, footnote_definition"
    )]
    pub block_type: Option<String>,

    #[schemars(description = "For heading blocks, filter by exact heading level (1-6)")]
    pub heading_level: Option<u8>,

    #[schemars(description = "For code blocks, filter by programming language")]
    pub language: Option<String>,

    #[schemars(description = "Restrict query to blocks within the document at this path")]
    pub doc_path: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DeleteDocumentRequest {
    #[schemars(description = "Relative path of the document to delete")]
    pub path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchFulltextRequest {
    #[schemars(description = "Full-text search query string")]
    pub query: String,

    #[schemars(description = "Maximum number of results to return (default: 10)")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetLinksRequest {
    #[schemars(description = "Document path to get outgoing links from")]
    pub path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetBacklinksRequest {
    #[schemars(description = "Document path to get incoming backlinks for")]
    pub path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchSemanticRequest {
    #[schemars(description = "Natural language query for semantic similarity search")]
    pub query: String,

    #[schemars(description = "Maximum number of results to return (default: 10)")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GraphPageRankRequest {
    #[schemars(description = "Damping factor for PageRank (default: 0.85). \
        Higher values give more weight to link structure vs. random surfing.")]
    pub damping: Option<f64>,

    #[schemars(description = "Number of power iterations (default: 20). \
        More iterations yield more precise scores.")]
    pub iterations: Option<usize>,
}

// No parameters needed for connected_components, but rmcp tools require a
// parameter type, so we use an empty struct.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GraphComponentsRequest {}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

#[tool_router]
impl InkbaseService {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            storage,
            index: None,
            graph: None,
            embedder: None,
            vector_index: None,
            tool_router: Self::tool_router(),
        }
    }

    /// Create a fully-configured service with all subsystems.
    pub fn full(
        storage: Arc<dyn Storage>,
        index: Arc<FullTextIndex>,
        graph: Arc<KnowledgeGraph>,
        embedder: Arc<EmbeddingPipeline>,
        vector_index: Arc<VectorIndex>,
    ) -> Self {
        Self {
            storage,
            index: Some(index),
            graph: Some(graph),
            embedder: Some(embedder),
            vector_index: Some(vector_index),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Parse and ingest a Markdown document into the database. \
        Extracts structural blocks, links, and frontmatter tags. \
        If a document at the same path already exists it will be replaced.")]
    #[instrument(skip(self, req), fields(path = %req.path))]
    async fn ingest_document(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            IngestDocumentRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        info!("ingesting document");

        let parse_result = parse_markdown(&req.path, &req.content).map_err(|e| {
            error!(error = %e, "markdown parsing failed");
            internal_error(format!("failed to parse markdown: {e}"))
        })?;

        let doc_id = parse_result.document.doc_id;
        let block_count = parse_result.blocks.len();
        let link_count = parse_result.links.len();
        let tag_count = parse_result.tags.len();

        // Save copies for index/graph before moving into spawn_blocking
        let parse_result_for_index = (parse_result.document.clone(), parse_result.blocks.clone());
        let parse_result_for_graph = parse_result.links.clone();

        // Capture the path before moving req into the closure.
        let path = req.path;
        let path_for_response = path.clone();
        let storage = self.storage.clone();

        // Storage operations are synchronous — run them on the blocking thread
        // pool to avoid starving the async runtime.
        tokio::task::spawn_blocking(move || -> Result<(), ErrorData> {
            // If a document at this path already exists, delete it first so we
            // get a clean replacement (cascade removes old blocks/links/tags).
            if let Ok(Some(existing)) = storage.get_document_by_path(&path) {
                debug!(old_doc_id = %existing.doc_id, "replacing existing document");
                storage.delete_document(&existing.doc_id).map_err(|e| {
                    internal_error(format!("failed to delete existing document: {e}"))
                })?;
            }

            storage
                .insert_document(&parse_result.document)
                .map_err(|e| internal_error(format!("failed to insert document: {e}")))?;

            storage
                .insert_blocks(&parse_result.blocks)
                .map_err(|e| internal_error(format!("failed to insert blocks: {e}")))?;

            storage
                .insert_links(&parse_result.links)
                .map_err(|e| internal_error(format!("failed to insert links: {e}")))?;

            storage
                .insert_tags(&parse_result.tags)
                .map_err(|e| internal_error(format!("failed to insert tags: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| internal_error(format!("blocking task panicked: {e}")))?
        .map_err(|e| {
            error!(error = ?e, "storage operations failed");
            e
        })?;

        // Index in full-text search if available
        if let Some(index) = &self.index {
            let index = index.clone();
            let doc = parse_result_for_index.0.clone();
            let blocks = parse_result_for_index.1.clone();
            if let Err(e) = index.index_document(&doc, &blocks) {
                error!(error = %e, "full-text indexing failed (non-fatal)");
            }
        }

        // Add links to knowledge graph if available
        if let Some(graph) = &self.graph {
            let links_for_graph = parse_result_for_graph.clone();
            if let Err(e) = graph.add_links(&path_for_response, &links_for_graph) {
                error!(error = %e, "graph update failed (non-fatal)");
            }
        }

        // Embed blocks and add to vector index if available
        if let (Some(embedder), Some(vi)) = (&self.embedder, &self.vector_index) {
            let blocks = &parse_result_for_index.1;
            match embedder.embed_blocks(blocks) {
                Ok(embeddings) => {
                    for (block_id, vector) in &embeddings {
                        let block = blocks.iter().find(|b| &b.block_id == block_id);
                        let text = block.map(|b| b.text_content.as_str()).unwrap_or("");
                        if let Err(e) = vi.insert(
                            &doc_id.to_string(),
                            &block_id.to_string(),
                            &path_for_response,
                            text,
                            vector.clone(),
                        ) {
                            error!(error = %e, "vector indexing failed (non-fatal)");
                        }
                    }
                    debug!(count = embeddings.len(), "embedded blocks for vector search");
                }
                Err(e) => error!(error = %e, "block embedding failed (non-fatal)"),
            }
        }

        info!(%doc_id, block_count, link_count, tag_count, "document ingested");

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Document ingested successfully.\n\
             doc_id: {doc_id}\n\
             path: {path_for_response}\n\
             blocks: {block_count}\n\
             links: {link_count}\n\
             tags: {tag_count}",
        ))]))
    }

    #[tool(description = "Retrieve a document and its structural blocks by file path. \
        Returns document metadata and all blocks as JSON.")]
    #[instrument(skip(self, req), fields(path = %req.path))]
    async fn get_document(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            GetDocumentRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        debug!("looking up document");

        let storage = self.storage.clone();
        let path = req.path;

        let (doc, blocks, tags) =
            tokio::task::spawn_blocking(move || -> Result<_, ErrorData> {
                let doc = storage
                    .get_document_by_path(&path)
                    .map_err(|e| internal_error(format!("storage error: {e}")))?
                    .ok_or_else(|| {
                        ErrorData::new(
                            ErrorCode::INVALID_PARAMS,
                            format!("document not found: {path}"),
                            None,
                        )
                    })?;

                let blocks = storage
                    .get_blocks_by_doc(&doc.doc_id)
                    .map_err(|e| internal_error(format!("failed to fetch blocks: {e}")))?;

                let tags = storage
                    .get_tags_by_doc(&doc.doc_id)
                    .map_err(|e| internal_error(format!("failed to fetch tags: {e}")))?;

                Ok((doc, blocks, tags))
            })
            .await
            .map_err(|e| internal_error(format!("blocking task panicked: {e}")))?
            ?;

        debug!(doc_id = %doc.doc_id, block_count = blocks.len(), "document retrieved");

        let response = serde_json::json!({
            "document": doc,
            "blocks": blocks,
            "tags": tags,
        });

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Query structural blocks across documents with optional filters. \
        Filters are ANDed together. Returns matching blocks as a JSON array.")]
    #[instrument(skip(self, req))]
    async fn query_blocks(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            QueryBlocksRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        debug!(?req, "querying blocks");

        let storage = self.storage.clone();

        let blocks = tokio::task::spawn_blocking(move || -> Result<_, ErrorData> {
            // If a doc_path filter is provided, resolve it to a doc_id first.
            let doc_id = match &req.doc_path {
                Some(path) => {
                    let doc = storage
                        .get_document_by_path(path)
                        .map_err(|e| internal_error(format!("storage error: {e}")))?
                        .ok_or_else(|| {
                            ErrorData::new(
                                ErrorCode::INVALID_PARAMS,
                                format!("document not found: {path}"),
                                None,
                            )
                        })?;
                    Some(doc.doc_id)
                }
                None => None,
            };

            storage
                .query_blocks(
                    doc_id.as_ref(),
                    req.block_type.as_deref(),
                    req.heading_level,
                    req.language.as_deref(),
                )
                .map_err(|e| internal_error(format!("query failed: {e}")))
        })
        .await
        .map_err(|e| internal_error(format!("blocking task panicked: {e}")))?
        ?;

        debug!(count = blocks.len(), "query returned blocks");

        let json = serde_json::to_string_pretty(&blocks)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "List all documents in the database. \
        Returns an array of document records with metadata (no block content).")]
    async fn list_documents(&self) -> Result<CallToolResult, ErrorData> {
        debug!("listing documents");

        let storage = self.storage.clone();

        let docs = tokio::task::spawn_blocking(move || {
            storage
                .list_documents()
                .map_err(|e| internal_error(format!("storage error: {e}")))
        })
        .await
        .map_err(|e| internal_error(format!("blocking task panicked: {e}")))?
        ?;

        info!(count = docs.len(), "listed documents");

        let json = serde_json::to_string_pretty(&docs)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Delete a document and all of its associated blocks, links, and tags.")]
    #[instrument(skip(self, req), fields(path = %req.path))]
    async fn delete_document(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            DeleteDocumentRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        info!("deleting document");

        let storage = self.storage.clone();
        let path = req.path;
        let path_for_response = path.clone();

        tokio::task::spawn_blocking(move || -> Result<(), ErrorData> {
            let doc = storage
                .get_document_by_path(&path)
                .map_err(|e| internal_error(format!("storage error: {e}")))?
                .ok_or_else(|| {
                    ErrorData::new(
                        ErrorCode::INVALID_PARAMS,
                        format!("document not found: {path}"),
                        None,
                    )
                })?;

            storage
                .delete_document(&doc.doc_id)
                .map_err(|e| internal_error(format!("failed to delete document: {e}")))?;

            info!(doc_id = %doc.doc_id, %path, "document deleted");
            Ok(())
        })
        .await
        .map_err(|e| internal_error(format!("blocking task panicked: {e}")))?
        .map_err(|e| {
            error!(error = ?e, "delete failed");
            e
        })?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Document at '{path_for_response}' deleted successfully.",
        ))]))
    }

    #[tool(description = "Full-text search across all ingested documents. \
        Returns ranked results with text snippets highlighting matches.")]
    #[instrument(skip(self, req), fields(query = %req.query))]
    async fn search_fulltext(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            SearchFulltextRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        let index = self.index.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "full-text search index not available".to_string(),
                None,
            )
        })?;

        let limit = req.limit.unwrap_or(10);
        let results = index.search(&req.query, limit).map_err(|e| {
            error!(error = %e, "full-text search failed");
            internal_error(format!("search failed: {e}"))
        })?;

        info!(count = results.len(), "search completed");

        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get all outgoing links from a document. \
        Returns links with their targets, types, and anchor text.")]
    #[instrument(skip(self, req), fields(path = %req.path))]
    async fn get_links(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            GetLinksRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "knowledge graph not available".to_string(),
                None,
            )
        })?;

        let links = graph.get_links(&req.path).map_err(|e| {
            error!(error = %e, "get_links failed");
            internal_error(format!("graph query failed: {e}"))
        })?;

        debug!(count = links.len(), "links retrieved");

        let json = serde_json::to_string_pretty(&links)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get all incoming backlinks to a document — \
        i.e., documents that link to this one.")]
    #[instrument(skip(self, req), fields(path = %req.path))]
    async fn get_backlinks(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            GetBacklinksRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "knowledge graph not available".to_string(),
                None,
            )
        })?;

        let backlinks = graph.get_backlinks(&req.path).map_err(|e| {
            error!(error = %e, "get_backlinks failed");
            internal_error(format!("graph query failed: {e}"))
        })?;

        debug!(count = backlinks.len(), "backlinks retrieved");

        let json = serde_json::to_string_pretty(&backlinks)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Semantic similarity search — finds blocks whose meaning is \
        similar to the query, even if they don't share exact keywords.")]
    #[instrument(skip(self, req), fields(query = %req.query))]
    async fn search_semantic(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            SearchSemanticRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "embedding pipeline not available".to_string(),
                None,
            )
        })?;
        let vi = self.vector_index.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "vector index not available".to_string(),
                None,
            )
        })?;

        let query_vector = embedder.embed_text(&req.query).map_err(|e| {
            error!(error = %e, "query embedding failed");
            internal_error(format!("failed to embed query: {e}"))
        })?;

        let limit = req.limit.unwrap_or(10);
        let results = vi.search(&query_vector, limit).map_err(|e| {
            error!(error = %e, "vector search failed");
            internal_error(format!("vector search failed: {e}"))
        })?;

        info!(count = results.len(), "semantic search completed");

        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Compute PageRank scores for all documents in the knowledge graph. \
        Returns documents ranked by importance based on how many other documents link to them. \
        Useful for identifying the most central/important notes in a knowledge base.")]
    async fn graph_pagerank(
        &self,
        rmcp::handler::server::tool::Parameters(req): rmcp::handler::server::tool::Parameters<
            GraphPageRankRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "knowledge graph not available".to_string(),
                None,
            )
        })?;

        let damping = req.damping.unwrap_or(0.85);
        let iterations = req.iterations.unwrap_or(20);

        let scores = graph.pagerank(damping, iterations).map_err(|e| {
            error!(error = %e, "pagerank computation failed");
            internal_error(format!("pagerank failed: {e}"))
        })?;

        info!(count = scores.len(), damping, iterations, "pagerank computed");

        let json = serde_json::to_string_pretty(&scores)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Find connected components in the knowledge graph (treating links \
        as undirected). Returns groups of documents that are reachable from each other \
        through any chain of links. Useful for discovering isolated document clusters.")]
    async fn graph_components(
        &self,
        rmcp::handler::server::tool::Parameters(_req): rmcp::handler::server::tool::Parameters<
            GraphComponentsRequest,
        >,
    ) -> Result<CallToolResult, ErrorData> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            ErrorData::new(
                ErrorCode::INTERNAL_ERROR,
                "knowledge graph not available".to_string(),
                None,
            )
        })?;

        let components = graph.connected_components().map_err(|e| {
            error!(error = %e, "connected_components failed");
            internal_error(format!("connected components failed: {e}"))
        })?;

        info!(count = components.len(), "connected components found");

        let response = serde_json::json!({
            "count": components.len(),
            "components": components,
        });

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| internal_error(format!("serialization error: {e}")))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

// ---------------------------------------------------------------------------
// ServerHandler — MCP server metadata and capability declaration
// ---------------------------------------------------------------------------

#[tool_handler]
impl ServerHandler for InkbaseService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "inkbase".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(
                "Inkbase MCP server — a Markdown-oriented database for AI agents. \
                 Ingest Markdown documents, query structural blocks, and manage a \
                 knowledge base with links and tags."
                    .to_string(),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Start the MCP server on stdio transport.
///
/// This is the main entry point for running the MCP server as a subprocess
/// of an AI agent host. The server reads JSON-RPC messages from stdin and
/// writes responses to stdout.
pub async fn run_stdio(
    storage: Arc<dyn Storage>,
    index: Option<Arc<FullTextIndex>>,
    graph: Option<Arc<KnowledgeGraph>>,
    embedder: Option<Arc<EmbeddingPipeline>>,
    vector_index: Option<Arc<VectorIndex>>,
) -> anyhow::Result<()> {
    info!("starting MCP server on stdio transport");

    let service = match (index, graph, embedder, vector_index) {
        (Some(idx), Some(g), Some(emb), Some(vi)) => {
            InkbaseService::full(storage, idx, g, emb, vi)
        }
        _ => InkbaseService::new(storage),
    };
    let server = service.serve(rmcp::transport::stdio()).await?;

    info!("MCP server running, waiting for client connection");
    server.waiting().await?;

    info!("MCP server shut down");
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP entry point
// ---------------------------------------------------------------------------

/// Start the MCP server on HTTP streamable transport.
///
/// This runs an HTTP server that speaks the MCP streamable-HTTP protocol,
/// allowing remote AI agents to connect over HTTP instead of stdio.
pub async fn run_http(
    storage: Arc<dyn Storage>,
    index: Option<Arc<FullTextIndex>>,
    graph: Option<Arc<KnowledgeGraph>>,
    embedder: Option<Arc<EmbeddingPipeline>>,
    vector_index: Option<Arc<VectorIndex>>,
    addr: &str,
) -> anyhow::Result<()> {
    info!(addr = %addr, "starting MCP server on HTTP transport");

    let service_factory = {
        let storage = storage.clone();
        let index = index.clone();
        let graph = graph.clone();
        let embedder = embedder.clone();
        let vector_index = vector_index.clone();
        move || -> Result<InkbaseService, std::io::Error> {
            let service = match (&index, &graph, &embedder, &vector_index) {
                (Some(idx), Some(g), Some(emb), Some(vi)) => InkbaseService::full(
                    storage.clone(),
                    idx.clone(),
                    g.clone(),
                    emb.clone(),
                    vi.clone(),
                ),
                _ => InkbaseService::new(storage.clone()),
            };
            Ok(service)
        }
    };

    let config = rmcp::transport::StreamableHttpServerConfig::default();
    let session_manager = Arc::new(
        rmcp::transport::streamable_http_server::session::local::LocalSessionManager::default(),
    );
    let mcp_service =
        rmcp::transport::StreamableHttpService::new(service_factory, session_manager, config);

    let app = axum::Router::new().route(
        "/mcp",
        axum::routing::any_service(mcp_service),
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(addr = %addr, "MCP HTTP server listening");
    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Shorthand for constructing an INTERNAL_ERROR response.
fn internal_error(message: String) -> ErrorData {
    ErrorData::new(ErrorCode::INTERNAL_ERROR, message, None)
}
