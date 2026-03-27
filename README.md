# Inkbase

Inkbase is a Markdown-oriented database middleware for AI agents, written in Rust. It parses Markdown into structured records, stores them in a pluggable backend, builds full-text, vector, and graph indexes, and exposes the result over REST and MCP.

## Features

- Parse Markdown, YAML frontmatter, wiki-links, and standard links into documents, blocks, links, and tags.
- Store content in either `sled` or the custom B+Tree storage engine.
- Query data through full-text search, semantic search, and graph traversal.
- Expose the same knowledge base through REST API, MCP over stdio, and MCP over HTTP.
- Run without external model dependencies by default with the built-in TF-IDF embedding pipeline.

## Workspace Layout

- `crates/inkbase-core`: shared types, config, and error definitions.
- `crates/inkbase-parser`: Markdown and frontmatter parsing.
- `crates/inkbase-storage`: storage trait plus `sled` and custom engine implementations.
- `crates/inkbase-index`: Tantivy full-text search and vector index.
- `crates/inkbase-graph`: knowledge graph traversal and ranking.
- `crates/inkbase-embedding`: TF-IDF, OpenAI, and Ollama embedding providers.
- `crates/inkbase-query`: MQL parser and AST.
- `crates/inkbase-api`: Axum REST API and Swagger/OpenAPI wiring.
- `crates/inkbase-mcp`: MCP tools for agent access.
- `crates/inkbase-server`: CLI entry point and runtime wiring.
- `crates/inkbase-integration-tests`: end-to-end tests across crates.

## Getting Started

Prerequisite: a recent Rust toolchain with Cargo.

```bash
cargo build --workspace
cargo test --workspace
```

Start the REST API with the helper script:

```bash
./start.sh
```

Or run the binary directly:

```bash
cargo run -p inkbase-server -- api --bind 127.0.0.1:3000
```

Useful variants:

```bash
cargo run -p inkbase-server -- serve
cargo run -p inkbase-server -- mcp-http --bind 127.0.0.1:3001
cargo run -p inkbase-server -- ingest path/to/file-or-directory.md
cargo run -p inkbase-server -- --engine custom api --bind 127.0.0.1:3000
```

By default, runtime data is written to `./data`. `start.sh` also accepts `HOST`, `PORT`, `LOG_LEVEL`, and `ENGINE` environment variables.

## REST API

When the API server is running:

- Swagger UI: `http://127.0.0.1:3000/swagger-ui/`
- OpenAPI JSON: `http://127.0.0.1:3000/api-docs/openapi.json`
- Health check: `GET /api/v1/health`

Core endpoints include:

- `POST /api/v1/documents` to ingest Markdown content.
- `GET /api/v1/documents` and `GET /api/v1/documents/{path}` to inspect stored documents.
- `POST /api/v1/search/fulltext` and `POST /api/v1/search/semantic` for search.
- `GET /api/v1/graph/stats`, `/graph/shortest-path`, `/graph/pagerank`, and `/graph/components` for graph analysis.

## Development

Use the workspace root for all commands:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test -p inkbase-integration-tests
cargo bench -p inkbase-parser
```

CI currently validates `cargo build` and `cargo test` on pushes and pull requests to `main`.
