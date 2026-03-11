# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Markdotabase

A Markdown-oriented database middleware for AI agents, written in Rust. Parses Markdown into structural blocks, stores them with full-text/vector/graph indexes, and exposes everything via MCP (stdio + HTTP) and REST API.

## Build & Test Commands

```bash
cargo build                                    # build all crates
cargo build -p mdbase-parser                   # build a single crate
cargo test --workspace                         # run all tests (unit + integration)
cargo test -p mdbase-integration-tests         # run only integration tests
cargo test -p mdbase-parser                    # run tests for a single crate
cargo test full_pipeline_sled                  # run a single test by name
cargo bench -p mdbase-parser                   # run benchmarks for a crate
```

## Running the Server

```bash
cargo run -- serve                             # MCP server on stdio
cargo run -- api --bind 127.0.0.1:3000         # REST API server
cargo run -- mcp-http --bind 127.0.0.1:3001    # MCP over HTTP (streamable-http)
cargo run -- ingest path/to/file.md            # ingest a markdown file or directory
cargo run -- --engine custom serve             # use custom B+Tree engine instead of sled
```

Tracing logs go to stderr (stdout is reserved for MCP JSON-RPC). Set log level with `--log-level debug` or `RUST_LOG=debug`.

## Workspace Architecture

Cargo workspace with 11 crates under `crates/`. Data flows top-down through these layers:

**Interface layer** (how clients connect):
- `mdbase-server` — binary entry point, CLI via clap. Wires everything together.
- `mdbase-mcp` — MCP server (rmcp). Tools: `ingest_document`, `get_document`, `query_blocks`, `list_documents`, `delete_document`, `search_fulltext`, `search_semantic`, `get_links`, `get_backlinks`, `graph_pagerank`, `graph_components`.
- `mdbase-api` — REST API (axum). Routes in `router.rs`, handlers in `handlers.rs`, shared state in `state.rs` (`AppState`).

**Domain layer** (processing logic):
- `mdbase-parser` — Markdown parsing via comrak. Produces `ParseResult` containing `DocumentRecord`, `BlockRecord`s, `LinkRecord`s, `TagRecord`s. Handles frontmatter (YAML), wiki-links (`[[target]]`), and standard markdown links.
- `mdbase-graph` — Knowledge graph (petgraph). Links/backlinks traversal, shortest path, PageRank, connected components.
- `mdbase-embedding` — Embedding pipeline with pluggable providers via `EmbeddingProvider` trait. Implementations: `TfIdfEmbedder` (default, no external deps), `OpenAiEmbedder`, `OllamaEmbedder`.
- `mdbase-query` — MQL query language parsed with pest. AST in `ast.rs`, parser in `parser.rs`. Supports `SELECT documents/blocks/links/tags WHERE ... LIMIT n` and `SEARCH fulltext/semantic '...'`.

**Index layer**:
- `mdbase-index` — Full-text search (Tantivy) and vector search (in-memory HNSW). Types in `types.rs`, `vector_types.rs`.

**Storage layer**:
- `mdbase-storage` — `Storage` trait in `traits.rs` abstracts all persistence. Two implementations:
  - `SledStorage` — sled-based (Phase 1 interim).
  - `CustomStorageEngine` — custom B+Tree engine with page storage, WAL, buffer pool, and MVCC snapshots. Files under `engine/`: `storage_engine.rs`, `btree.rs`, `page.rs`, `buffer_pool.rs`, `wal.rs`, `mvcc.rs`, `header.rs`.

**Shared types**:
- `mdbase-core` — Core types (`DocumentRecord`, `BlockRecord`, `LinkRecord`, `EmbeddingRecord`, `TagRecord`), `MdbaseError` enum, `MdbaseConfig`. All other crates depend on this.

**Tests**:
- `mdbase-integration-tests` — End-to-end tests spanning multiple crates. Tests both storage backends, graph analysis, MQL parsing, MVCC snapshots, cascade deletes, embedding pipeline, and parser edge cases.

## Key Patterns

- All storage operations are synchronous. In async contexts (MCP/API handlers), they're wrapped in `tokio::task::spawn_blocking`.
- `Storage` trait requires `Send + Sync` — implementations are shared via `Arc<dyn Storage>`.
- The custom storage engine uses file magic `MDOTADB\0`, 8KB pages, and slotted page layout.
- `EmbeddingPipeline::default_tfidf()` is used by default — no external API keys or models needed.
- MCP tool request types derive `schemars::JsonSchema` for automatic MCP tool discovery.
- Error handling: `mdbase_core::MdbaseError` with `thiserror` for library crates; `anyhow::Result` at the binary level.
- Document identity: UUID v4 for `doc_id`, BLAKE3 hash for content deduplication.
