use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::AppState;

/// Build the application router with all API endpoints, CORS, and tracing.
pub fn build_router(state: AppState) -> Router {
    let api = Router::new()
        // Document endpoints
        .route("/documents", post(handlers::ingest_document))
        .route("/documents", get(handlers::list_documents))
        .route("/documents/{path}", get(handlers::get_document))
        .route("/documents/{path}", delete(handlers::delete_document))
        .route("/documents/{path}/blocks", get(handlers::get_blocks))
        .route("/documents/{path}/links", get(handlers::get_links))
        .route(
            "/documents/{path}/backlinks",
            get(handlers::get_backlinks),
        )
        // Search
        .route("/search/fulltext", post(handlers::fulltext_search))
        .route("/search/semantic", post(handlers::semantic_search))
        // Graph
        .route("/graph/stats", get(handlers::graph_stats))
        .route("/graph/shortest-path", get(handlers::shortest_path))
        .route("/graph/pagerank", get(handlers::graph_pagerank))
        .route("/graph/components", get(handlers::graph_components))
        // Operational
        .route("/health", get(handlers::health_check))
        .route("/stats", get(handlers::database_stats));

    Router::new()
        .nest("/api/v1", api)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
