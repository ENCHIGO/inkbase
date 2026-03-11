use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, error, info};

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Error response helper
// ---------------------------------------------------------------------------

/// Map a `JoinError` (from `spawn_blocking`) to a 500 response.
fn join_error(err: tokio::task::JoinError) -> (StatusCode, Json<Value>) {
    error!(error = %err, "blocking task panicked");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": format!("internal error: {err}") })),
    )
}

// ---------------------------------------------------------------------------
// Request / query types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ShortestPathQuery {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Deserialize)]
pub struct PageRankParams {
    pub damping: Option<f64>,
    pub iterations: Option<usize>,
}

// ---------------------------------------------------------------------------
// Health & stats
// ---------------------------------------------------------------------------

/// `GET /api/v1/health`
pub async fn health_check() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// `GET /api/v1/stats` -- database-level statistics.
pub async fn database_stats(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let storage = state.storage.clone();

    let docs = tokio::task::spawn_blocking(move || storage.list_documents())
        .await
        .map_err(join_error)?
        .map_err(|e| {
            error!(error = %e, "failed to list documents for stats");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("storage error: {e}") })),
            )
        })?;

    let graph_stats = state.graph.stats();

    Ok(Json(json!({
        "document_count": docs.len(),
        "graph": graph_stats,
    })))
}

// ---------------------------------------------------------------------------
// Document CRUD
// ---------------------------------------------------------------------------

/// `POST /api/v1/documents` -- ingest (parse + store + index + graph) a document.
pub async fn ingest_document(
    State(state): State<AppState>,
    Json(req): Json<IngestRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<Value>)> {
    // Parse markdown.
    let parse_result = mdbase_parser::parse_markdown(&req.path, &req.content).map_err(|e| {
        error!(error = %e, path = %req.path, "markdown parsing failed");
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("parse error: {e}") })),
        )
    })?;

    let doc_id = parse_result.document.doc_id;
    let block_count = parse_result.blocks.len();
    let link_count = parse_result.links.len();
    let tag_count = parse_result.tags.len();
    let path = req.path.clone();

    // Store in the storage backend (synchronous -- use spawn_blocking).
    let storage = state.storage.clone();
    let doc_for_index = parse_result.document.clone();
    let blocks_for_index = parse_result.blocks.clone();
    let links_for_graph = parse_result.links.clone();

    tokio::task::spawn_blocking(move || -> mdbase_core::Result<()> {
        // If a document at this path already exists, delete it first for a
        // clean replacement.
        if let Some(existing) = storage.get_document_by_path(&path)? {
            debug!(old_doc_id = %existing.doc_id, "replacing existing document");
            storage.delete_document(&existing.doc_id)?;
        }

        storage.insert_document(&parse_result.document)?;
        storage.insert_blocks(&parse_result.blocks)?;
        storage.insert_links(&parse_result.links)?;
        storage.insert_tags(&parse_result.tags)?;
        Ok(())
    })
    .await
    .map_err(join_error)?
    .map_err(|e| {
        error!(error = %e, "storage write failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("storage error: {e}") })),
        )
    })?;

    // Clone blocks for embedding before moving into index closure
    let blocks_for_embed = blocks_for_index.clone();

    // Index for full-text search.
    let index = state.index.clone();
    tokio::task::spawn_blocking(move || {
        index.index_document(&doc_for_index, &blocks_for_index)
    })
    .await
    .map_err(join_error)?
    .map_err(|e| {
        error!(error = %e, "index write failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("index error: {e}") })),
        )
    })?;

    // Add links to the knowledge graph.
    let source_path = req.path.clone();
    state.graph.add_links(&source_path, &links_for_graph).map_err(|e| {
        error!(error = %e, "graph update failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("graph error: {e}") })),
        )
    })?;

    // Embed blocks for vector search.
    let embedder = state.embedder.clone();
    let vi = state.vector_index.clone();
    let embed_doc_id = doc_id.to_string();
    let embed_path = req.path.clone();
    tokio::task::spawn_blocking(move || {
        if let Ok(embeddings) = embedder.embed_blocks(&blocks_for_embed) {
            for (block_id, vector) in embeddings {
                let block = blocks_for_embed.iter().find(|b| b.block_id == block_id);
                let text = block.map(|b| b.text_content.as_str()).unwrap_or("");
                let _ = vi.insert(&embed_doc_id, &block_id.to_string(), &embed_path, text, vector);
            }
        }
    })
    .await
    .map_err(join_error)?;

    info!(
        %doc_id,
        path = %req.path,
        block_count,
        link_count,
        tag_count,
        "document ingested"
    );

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "doc_id": doc_id,
            "path": req.path,
            "blocks": block_count,
            "links": link_count,
            "tags": tag_count,
        })),
    ))
}

/// `GET /api/v1/documents` -- list all documents.
pub async fn list_documents(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let storage = state.storage.clone();

    let docs = tokio::task::spawn_blocking(move || storage.list_documents())
        .await
        .map_err(join_error)?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("storage error: {e}") })),
            )
        })?;

    info!(count = docs.len(), "listed documents");
    Ok(Json(json!({ "documents": docs })))
}

/// `GET /api/v1/documents/:path` -- get a single document by path.
pub async fn get_document(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let storage = state.storage.clone();
    let lookup_path = path.clone();

    let (doc, tags) = tokio::task::spawn_blocking(move || -> mdbase_core::Result<_> {
        let doc = storage
            .get_document_by_path(&lookup_path)?
            .ok_or_else(|| {
                mdbase_core::MdbaseError::DocumentNotFound(lookup_path.clone())
            })?;
        let tags = storage.get_tags_by_doc(&doc.doc_id)?;
        Ok((doc, tags))
    })
    .await
    .map_err(join_error)?
    .map_err(|e| match &e {
        mdbase_core::MdbaseError::DocumentNotFound(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("storage error: {e}") })),
        ),
    })?;

    Ok(Json(json!({
        "document": doc,
        "tags": tags,
    })))
}

/// `DELETE /api/v1/documents/:path` -- delete a document and cascade.
pub async fn delete_document(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let storage = state.storage.clone();
    let lookup_path = path.clone();

    let doc_id = tokio::task::spawn_blocking(move || -> mdbase_core::Result<_> {
        let doc = storage
            .get_document_by_path(&lookup_path)?
            .ok_or_else(|| {
                mdbase_core::MdbaseError::DocumentNotFound(lookup_path.clone())
            })?;
        let doc_id = doc.doc_id;
        storage.delete_document(&doc_id)?;
        Ok(doc_id)
    })
    .await
    .map_err(join_error)?
    .map_err(|e| match &e {
        mdbase_core::MdbaseError::DocumentNotFound(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("storage error: {e}") })),
        ),
    })?;

    // Remove from full-text index.
    let index = state.index.clone();
    let id_for_index = doc_id;
    tokio::task::spawn_blocking(move || index.delete_document(&id_for_index))
        .await
        .map_err(join_error)?
        .map_err(|e| {
            error!(error = %e, "failed to remove document from index");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("index error: {e}") })),
            )
        })?;

    // Remove outgoing edges from the knowledge graph.
    state.graph.remove_document(&path).map_err(|e| {
        error!(error = %e, "failed to remove document from graph");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("graph error: {e}") })),
        )
    })?;

    // Remove from vector index.
    let _ = state.vector_index.delete_document(&doc_id.to_string());

    info!(%doc_id, path = %path, "document deleted");

    Ok(Json(json!({
        "deleted": true,
        "doc_id": doc_id,
        "path": path,
    })))
}

// ---------------------------------------------------------------------------
// Sub-resource endpoints
// ---------------------------------------------------------------------------

/// `GET /api/v1/documents/:path/blocks` -- get all blocks for a document.
pub async fn get_blocks(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let storage = state.storage.clone();

    let blocks = tokio::task::spawn_blocking(move || -> mdbase_core::Result<_> {
        let doc = storage.get_document_by_path(&path)?.ok_or_else(|| {
            mdbase_core::MdbaseError::DocumentNotFound(path.clone())
        })?;
        storage.get_blocks_by_doc(&doc.doc_id)
    })
    .await
    .map_err(join_error)?
    .map_err(|e| match &e {
        mdbase_core::MdbaseError::DocumentNotFound(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": e.to_string() })),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("storage error: {e}") })),
        ),
    })?;

    Ok(Json(json!({ "blocks": blocks })))
}

/// `GET /api/v1/documents/:path/links` -- outgoing links from a document.
pub async fn get_links(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let links = state.graph.get_links(&path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("graph error: {e}") })),
        )
    })?;

    Ok(Json(json!({ "links": links })))
}

/// `GET /api/v1/documents/:path/backlinks` -- incoming links to a document.
pub async fn get_backlinks(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let backlinks = state.graph.get_backlinks(&path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("graph error: {e}") })),
        )
    })?;

    Ok(Json(json!({ "backlinks": backlinks })))
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// `POST /api/v1/search/fulltext` -- full-text search.
pub async fn fulltext_search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let limit = req.limit.unwrap_or(20);
    let query = req.query.clone();
    let index = state.index.clone();

    let results = tokio::task::spawn_blocking(move || index.search(&query, limit))
        .await
        .map_err(join_error)?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("search error: {e}") })),
            )
        })?;

    debug!(query = %req.query, hits = results.len(), "fulltext search");

    Ok(Json(json!({
        "query": req.query,
        "limit": limit,
        "results": results,
    })))
}

/// `POST /api/v1/search/semantic` -- semantic similarity search.
pub async fn semantic_search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let limit = req.limit.unwrap_or(10);

    // Embed the query text.
    let query_vector = state.embedder.embed_text(&req.query).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("embedding error: {e}") })),
        )
    })?;

    // Search the vector index.
    let results = state.vector_index.search(&query_vector, limit).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("vector search error: {e}") })),
        )
    })?;

    debug!(query = %req.query, hits = results.len(), "semantic search");

    Ok(Json(json!({
        "query": req.query,
        "limit": limit,
        "results": results,
    })))
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// `GET /api/v1/graph/stats` -- knowledge graph statistics.
pub async fn graph_stats(State(state): State<AppState>) -> Json<Value> {
    let stats = state.graph.stats();
    Json(json!(stats))
}

/// `GET /api/v1/graph/shortest-path?from=X&to=Y`
pub async fn shortest_path(
    State(state): State<AppState>,
    Query(params): Query<ShortestPathQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let path = state
        .graph
        .shortest_path(&params.from, &params.to)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("graph error: {e}") })),
            )
        })?;

    Ok(Json(json!({
        "from": params.from,
        "to": params.to,
        "path": path,
    })))
}

/// `GET /api/v1/graph/pagerank?damping=0.85&iterations=20`
pub async fn graph_pagerank(
    State(state): State<AppState>,
    Query(params): Query<PageRankParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let damping = params.damping.unwrap_or(0.85);
    let iterations = params.iterations.unwrap_or(20);

    let scores = state
        .graph
        .pagerank(damping, iterations)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("graph error: {e}") })),
            )
        })?;

    Ok(Json(json!({
        "damping": damping,
        "iterations": iterations,
        "scores": scores,
    })))
}

/// `GET /api/v1/graph/components`
pub async fn graph_components(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let components = state
        .graph
        .connected_components()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("graph error: {e}") })),
            )
        })?;

    Ok(Json(json!({
        "count": components.len(),
        "components": components,
    })))
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Create and run the HTTP server.
///
/// Binds to `addr` (e.g. `"0.0.0.0:3000"`) and serves until shutdown.
pub async fn run_server(state: AppState, addr: &str) -> anyhow::Result<()> {
    let app = crate::router::build_router(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(addr = %addr, "API server listening");

    axum::serve(listener, app).await?;

    Ok(())
}
