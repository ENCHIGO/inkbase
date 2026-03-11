use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use inkbase_embedding::EmbeddingPipeline;
use inkbase_graph::KnowledgeGraph;
use inkbase_index::{FullTextIndex, VectorIndex};
use inkbase_parser::parse_markdown;
use inkbase_storage::{CustomStorageEngine, SledStorage, Storage};

#[derive(Clone, clap::ValueEnum)]
enum EngineType {
    Sled,
    Custom,
}

#[derive(Parser)]
#[command(name = "inkbase")]
#[command(about = "Markdown-oriented database middleware for AI agents")]
#[command(version)]
struct Cli {
    /// Data directory path
    #[arg(short, long, default_value = "./data")]
    data_dir: PathBuf,

    /// Storage engine backend (sled or custom)
    #[arg(short, long, default_value = "sled")]
    engine: EngineType,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server over stdio
    Serve,
    /// Start the REST API server
    Api {
        /// Address to bind the HTTP server to
        #[arg(short, long, default_value = "127.0.0.1:3000")]
        bind: String,
    },
    /// Start the MCP server over HTTP (streamable-http transport)
    McpHttp {
        /// Address to bind the MCP HTTP server to
        #[arg(short, long, default_value = "127.0.0.1:3001")]
        bind: String,
    },
    /// Ingest a markdown file or directory
    Ingest {
        /// Path to markdown file or directory
        path: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing to stderr (stdout is used for MCP JSON-RPC)
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .with_writer(std::io::stderr)
        .init();

    tracing::info!("inkbase starting, data_dir={}", cli.data_dir.display());

    // Ensure data directory exists
    std::fs::create_dir_all(&cli.data_dir)?;

    let storage: Arc<dyn Storage> = match cli.engine {
        EngineType::Sled => {
            tracing::info!("Using sled storage engine");
            Arc::new(SledStorage::new(&cli.data_dir)?)
        }
        EngineType::Custom => {
            tracing::info!("Using custom B+Tree storage engine");
            Arc::new(CustomStorageEngine::new(&cli.data_dir)?)
        }
    };

    // Initialize full-text index
    let index_dir = cli.data_dir.join("index");
    std::fs::create_dir_all(&index_dir)?;
    let index = Arc::new(FullTextIndex::new(&index_dir)?);

    // Initialize knowledge graph
    let graph = Arc::new(KnowledgeGraph::new());

    // Initialize embedding pipeline and vector index
    let embedder = Arc::new(EmbeddingPipeline::default_tfidf());
    let vector_index = Arc::new(VectorIndex::new(embedder.dimension()));

    match cli.command {
        Commands::Serve => {
            tracing::info!("Starting MCP server on stdio...");
            inkbase_mcp::run_stdio(
                storage,
                Some(index),
                Some(graph),
                Some(embedder),
                Some(vector_index),
            )
            .await?;
            Ok(())
        }
        Commands::Api { bind } => {
            tracing::info!("Starting REST API server on {bind}...");
            let state = inkbase_api::AppState {
                storage,
                index,
                graph,
                embedder,
                vector_index,
            };
            inkbase_api::run_server(state, &bind).await?;
            Ok(())
        }
        Commands::McpHttp { bind } => {
            tracing::info!("Starting MCP HTTP server on {bind}...");
            inkbase_mcp::run_http(
                storage,
                Some(index),
                Some(graph),
                Some(embedder),
                Some(vector_index),
                &bind,
            )
            .await?;
            Ok(())
        }
        Commands::Ingest { path } => {
            tracing::info!("Ingesting from {}...", path.display());
            ingest_path(&path, &storage, &index, &graph, &embedder, &vector_index)?;
            Ok(())
        }
    }
}

fn ingest_path(
    path: &PathBuf,
    storage: &Arc<dyn Storage>,
    index: &Arc<FullTextIndex>,
    graph: &Arc<KnowledgeGraph>,
    embedder: &Arc<EmbeddingPipeline>,
    vector_index: &Arc<VectorIndex>,
) -> Result<()> {
    if path.is_dir() {
        let mut count = 0;
        for entry in walkdir(path)? {
            ingest_file(&entry, storage, index, graph, embedder, vector_index)?;
            count += 1;
        }
        tracing::info!(count, "ingested all markdown files from directory");
    } else {
        ingest_file(path, storage, index, graph, embedder, vector_index)?;
    }
    Ok(())
}

fn ingest_file(
    path: &PathBuf,
    storage: &Arc<dyn Storage>,
    index: &Arc<FullTextIndex>,
    graph: &Arc<KnowledgeGraph>,
    embedder: &Arc<EmbeddingPipeline>,
    vector_index: &Arc<VectorIndex>,
) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let rel_path = path.to_string_lossy().to_string();

    let result = parse_markdown(&rel_path, &content)?;
    let doc_id = result.document.doc_id;

    // Delete existing document at this path if any
    if let Ok(Some(existing)) = storage.get_document_by_path(&rel_path) {
        storage.delete_document(&existing.doc_id)?;
        let _ = index.delete_document(&existing.doc_id);
        let _ = graph.remove_document(&rel_path);
        let _ = vector_index.delete_document(&existing.doc_id.to_string());
    }

    storage.insert_document(&result.document)?;
    storage.insert_blocks(&result.blocks)?;
    storage.insert_links(&result.links)?;
    storage.insert_tags(&result.tags)?;

    // Index for full-text search
    index.index_document(&result.document, &result.blocks)?;

    // Add links to knowledge graph
    graph.add_links(&rel_path, &result.links)?;

    // Embed blocks for vector search
    if let Ok(embeddings) = embedder.embed_blocks(&result.blocks) {
        for (block_id, vector) in embeddings {
            let block = result.blocks.iter().find(|b| b.block_id == block_id);
            let text = block.map(|b| b.text_content.as_str()).unwrap_or("");
            let _ = vector_index.insert(
                &doc_id.to_string(),
                &block_id.to_string(),
                &rel_path,
                text,
                vector,
            );
        }
    }

    tracing::info!(
        path = %rel_path,
        doc_id = %doc_id,
        blocks = result.blocks.len(),
        links = result.links.len(),
        tags = result.tags.len(),
        "ingested document"
    );
    Ok(())
}

/// Recursively find all .md files in a directory.
fn walkdir(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    walk_recursive(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk_recursive(dir: &PathBuf, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_recursive(&path, files)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            files.push(path);
        }
    }
    Ok(())
}
