# Repository Guidelines

## Project Structure & Module Organization
`Cargo.toml` defines a Rust 2021 workspace. All production code lives in `crates/`, split by subsystem: `inkbase-core` for shared types/errors/config, `inkbase-parser` for Markdown parsing, `inkbase-storage` for persistence, `inkbase-index` for search, `inkbase-graph` for link analysis, `inkbase-embedding` for vector generation, `inkbase-query` for MQL parsing, `inkbase-api` for REST, `inkbase-mcp` for MCP, and `inkbase-server` for the `inkbase` binary. End-to-end coverage lives in `crates/inkbase-integration-tests/tests/end_to_end.rs`; Criterion benches live in `crates/*/benches`. Runtime artifacts under `data/` and build output under `target/` are ignored.

## Build, Test, and Development Commands
Use Cargo from the repository root:

- `cargo build --workspace`: build every crate.
- `cargo test --workspace`: run unit and integration tests; this matches CI.
- `cargo test -p inkbase-integration-tests`: focus on cross-crate scenarios.
- `cargo bench -p inkbase-parser`: run a crate benchmark; swap in other bench crates as needed.
- `cargo run -p inkbase-server -- api --bind 127.0.0.1:3000`: start the REST API.
- `cargo run -p inkbase-server -- serve`: start MCP over stdio.
- `cargo run -p inkbase-server -- mcp-http --bind 127.0.0.1:3001`: start MCP over HTTP.
- `./start.sh`: convenience wrapper for the API server with `HOST`, `PORT`, `LOG_LEVEL`, and `ENGINE` env vars.

## Coding Style & Naming Conventions
Follow standard Rust formatting with `cargo fmt --all`; no custom `rustfmt.toml` is checked in. Use 4-space indentation, `snake_case` for functions/modules/files, `CamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Keep crate APIs narrow by re-exporting intentional entry points from `lib.rs`, and prefer small focused modules over cross-crate leakage. `cargo clippy --workspace --all-targets` is the expected lint pass before review.

## Testing Guidelines
Keep fast unit tests next to implementation with `#[cfg(test)]`. Use the integration-test crate for pipelines that span parser, storage, search, graph, or embedding behavior. Name tests by behavior, such as `full_pipeline_sled`. When changing parser rules, storage semantics, or API/MCP behavior, add or update both focused unit coverage and one end-to-end path when the change crosses crate boundaries.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects like `Add Swagger UI API docs and local startup script`. Keep that style: capitalized verb, narrow scope, one line. Pull requests should summarize the behavior change, list validation commands, link related issues, and include request/response samples or screenshots when API or docs behavior changes.
