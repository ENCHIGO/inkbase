//! End-to-end integration tests for the Markdotabase workspace.
//!
//! These tests exercise complete pipelines that span multiple crates:
//! parser -> storage -> search/graph/embedding, verifying that the pieces
//! compose correctly in realistic scenarios.

use uuid::Uuid;

use mdbase_core::types::{BlockType, LinkRecord, LinkType};
use mdbase_embedding::EmbeddingPipeline;
use mdbase_graph::KnowledgeGraph;
use mdbase_index::{FullTextIndex, VectorIndex};
use mdbase_parser::parse_markdown;
use mdbase_query::{
    parse_query, CompareOp, Entity, Query, SearchType, Value,
};
use mdbase_storage::{CustomStorageEngine, SledStorage, Storage};

// ---------------------------------------------------------------------------
// Shared test data
// ---------------------------------------------------------------------------

const RUST_DOC: &str = r#"---
title: Rust Programming
tags: [rust, programming, systems]
category: languages
---

# Rust Programming Guide

Rust is a systems programming language focused on safety and performance.

## Memory Safety

Rust prevents null pointer dereferences and data races at compile time.

```rust
fn main() {
    let x = vec![1, 2, 3];
    println!("{:?}", x);
}
```

## Links

Check out [[python-guide]] for comparison.
See also [Go Guide](go-guide.md) for another systems language.

## Tables

| Feature | Rust | Go |
|---------|------|-----|
| Memory Safety | Yes | Partial |
| Generics | Yes | Yes |
"#;

const PYTHON_DOC: &str = r#"---
title: Python Guide
tags: [python, programming, scripting]
category: languages
---

# Python Programming Guide

Python is a high-level interpreted language used for web development and data science.

## Dynamic Typing

Python uses duck typing and dynamic dispatch.

```python
def greet(name):
    print(f"Hello, {name}")
```

## Links

See [[rust-programming]] for a compiled systems language.
Check out [[go-guide]] for another option.
"#;

const GO_DOC: &str = r#"---
title: Go Guide
tags: [go, programming, systems]
category: languages
---

# Go Programming Guide

Go is a compiled language designed for simplicity and concurrency.

## Goroutines

Go uses goroutines for lightweight concurrency.

```go
func main() {
    go fmt.Println("hello")
}
```

## Links

Also see [[rust-programming]] for a memory-safe systems language.
"#;

// ===========================================================================
// Helper: run the full ingest pipeline on a storage backend
// ===========================================================================

/// Parse a markdown document and ingest its records into the given storage.
/// Returns the ParseResult for further assertions.
fn ingest_document(
    storage: &dyn Storage,
    path: &str,
    content: &str,
) -> mdbase_parser::ParseResult {
    let result = parse_markdown(path, content).expect("parse should succeed");
    storage
        .insert_document(&result.document)
        .expect("insert_document should succeed");
    storage
        .insert_blocks(&result.blocks)
        .expect("insert_blocks should succeed");
    storage
        .insert_links(&result.links)
        .expect("insert_links should succeed");
    storage
        .insert_tags(&result.tags)
        .expect("insert_tags should succeed");
    result
}

// ===========================================================================
// Test 1: Full pipeline with SledStorage
// ===========================================================================

#[test]
fn full_pipeline_sled() {
    let tmp = tempfile::tempdir().unwrap();
    let sled_dir = tmp.path().join("sled");
    let index_dir = tmp.path().join("index");

    // Create infrastructure.
    let storage = SledStorage::new(&sled_dir).unwrap();
    let fts = FullTextIndex::new(&index_dir).unwrap();
    let graph = KnowledgeGraph::new();
    let pipeline = EmbeddingPipeline::default_tfidf();
    let vec_index = VectorIndex::new(pipeline.dimension());

    // Ingest the Rust document.
    let rust_result = ingest_document(&storage, "notes/rust.md", RUST_DOC);
    let doc_id = rust_result.document.doc_id;

    // Index for full-text search.
    fts.index_document(&rust_result.document, &rust_result.blocks)
        .unwrap();

    // Add to graph.
    graph
        .add_links("notes/rust.md", &rust_result.links)
        .unwrap();

    // Generate and store embeddings.
    let embeddings = pipeline.embed_blocks(&rust_result.blocks).unwrap();
    for (block_id, vector) in &embeddings {
        // Find the matching block to get its text.
        let block = rust_result
            .blocks
            .iter()
            .find(|b| b.block_id == *block_id)
            .unwrap();
        vec_index
            .insert(
                &doc_id.to_string(),
                &block_id.to_string(),
                "notes/rust.md",
                &block.text_content,
                vector.clone(),
            )
            .unwrap();
    }

    // ---- Verify storage ----

    // Document lookup by path.
    let fetched = storage
        .get_document_by_path("notes/rust.md")
        .unwrap()
        .unwrap();
    assert_eq!(fetched.doc_id, doc_id);
    assert_eq!(
        fetched.title,
        Some("Rust Programming".to_string())
    );

    // Blocks are all stored.
    let blocks = storage.get_blocks_by_doc(&doc_id).unwrap();
    assert!(
        !blocks.is_empty(),
        "should have stored blocks for the document"
    );

    // Query blocks by type filter.
    let headings = storage
        .query_blocks(Some(&doc_id), Some("heading"), None, None)
        .unwrap();
    assert!(headings.len() >= 2, "should have at least 2 headings");

    let code_blocks = storage
        .query_blocks(Some(&doc_id), Some("code_block"), None, Some("rust"))
        .unwrap();
    assert_eq!(code_blocks.len(), 1, "should have exactly 1 Rust code block");

    // Tags extracted from frontmatter.
    let tags = storage.get_tags_by_doc(&doc_id).unwrap();
    let tag_keys: Vec<&str> = tags.iter().map(|t| t.key.as_str()).collect();
    assert!(tag_keys.contains(&"tags"), "should have 'tags' key");
    assert!(tag_keys.contains(&"category"), "should have 'category' key");

    // Flattened array tags.
    let tag_values: Vec<_> = tags
        .iter()
        .filter(|t| t.key == "tags")
        .map(|t| &t.value)
        .collect();
    assert_eq!(tag_values.len(), 3, "tags array should be flattened to 3");

    // Links stored.
    let links = storage.get_links_from(&doc_id).unwrap();
    assert!(!links.is_empty(), "should have stored links");

    // Wiki-link detection.
    let wiki_links: Vec<_> = links
        .iter()
        .filter(|l| l.link_type == LinkType::WikiLink)
        .collect();
    assert!(
        wiki_links.iter().any(|l| l.target == "python-guide"),
        "should detect [[python-guide]] wiki-link"
    );

    // Markdown link detection.
    let md_links: Vec<_> = links
        .iter()
        .filter(|l| l.link_type == LinkType::MarkdownLink)
        .collect();
    assert!(
        md_links.iter().any(|l| l.target == "go-guide.md"),
        "should detect [Go Guide](go-guide.md) link"
    );

    // ---- Verify full-text search ----

    let search_results = fts.search("safety performance", 10).unwrap();
    assert!(
        !search_results.is_empty(),
        "full-text search should find results for 'safety performance'"
    );
    assert_eq!(
        search_results[0].doc_id,
        doc_id.to_string(),
        "top result should be the Rust doc"
    );

    let no_results = fts.search("xylophone_nonexistent_term", 10).unwrap();
    assert!(
        no_results.is_empty(),
        "search for nonsense term should return empty"
    );

    // ---- Verify vector/semantic search ----

    let query_vec = pipeline.embed_text("memory safety systems programming").unwrap();
    let vec_results = vec_index.search(&query_vec, 5).unwrap();
    assert!(
        !vec_results.is_empty(),
        "vector search should find results"
    );
    assert_eq!(
        vec_results[0].doc_id,
        doc_id.to_string(),
        "top vector result should belong to the Rust doc"
    );

    // ---- Verify graph ----

    let outgoing = graph.get_links("notes/rust.md").unwrap();
    assert!(
        !outgoing.is_empty(),
        "graph should have outgoing links from Rust doc"
    );

    let stats = graph.stats();
    assert!(
        stats.node_count >= 3,
        "graph should have at least 3 nodes (source + 2 targets)"
    );
}

// ===========================================================================
// Test 2: Full pipeline with CustomStorageEngine + persistence
// ===========================================================================

#[test]
fn full_pipeline_custom_engine() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");

    let doc_id;
    let block_count;

    // Phase 1: ingest and verify.
    {
        let engine = CustomStorageEngine::new(&data_dir).unwrap();
        let result = ingest_document(&engine, "notes/rust.md", RUST_DOC);
        doc_id = result.document.doc_id;
        block_count = result.blocks.len();

        // Verify data is accessible.
        let fetched = engine.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(fetched.path, "notes/rust.md");

        let blocks = engine.get_blocks_by_doc(&doc_id).unwrap();
        assert_eq!(blocks.len(), block_count);

        let links = engine.get_links_from(&doc_id).unwrap();
        assert!(!links.is_empty());

        let tags = engine.get_tags_by_doc(&doc_id).unwrap();
        assert!(!tags.is_empty());
    }

    // Phase 2: reopen engine and verify persistence.
    {
        let engine = CustomStorageEngine::new(&data_dir).unwrap();

        let doc = engine.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(doc.path, "notes/rust.md");
        assert_eq!(
            doc.title,
            Some("Rust Programming".to_string())
        );

        let by_path = engine
            .get_document_by_path("notes/rust.md")
            .unwrap()
            .unwrap();
        assert_eq!(by_path.doc_id, doc_id);

        let blocks = engine.get_blocks_by_doc(&doc_id).unwrap();
        assert_eq!(
            blocks.len(),
            block_count,
            "block count should match after reopen"
        );

        let links = engine.get_links_from(&doc_id).unwrap();
        assert!(!links.is_empty(), "links should survive reopen");

        let tags = engine.get_tags_by_doc(&doc_id).unwrap();
        assert!(!tags.is_empty(), "tags should survive reopen");

        // Verify backlinks by target path.
        let wiki_links: Vec<_> = links
            .iter()
            .filter(|l| l.link_type == LinkType::WikiLink)
            .collect();
        if let Some(wl) = wiki_links.first() {
            let backlinks = engine.get_links_to(&wl.target).unwrap();
            assert!(
                !backlinks.is_empty(),
                "backlinks index should survive reopen"
            );
        }
    }
}

// ===========================================================================
// Test 3: Multi-document graph analysis
// ===========================================================================

#[test]
fn multi_document_graph() {
    let tmp = tempfile::tempdir().unwrap();
    let sled_dir = tmp.path().join("sled");

    let storage = SledStorage::new(&sled_dir).unwrap();
    let graph = KnowledgeGraph::new();

    // Ingest three interconnected documents.
    let rust_result = ingest_document(&storage, "rust.md", RUST_DOC);
    let python_result = ingest_document(&storage, "python.md", PYTHON_DOC);
    let go_result = ingest_document(&storage, "go.md", GO_DOC);

    // Build graph edges.
    // Rust doc links to: python-guide, go-guide.md
    // Python doc links to: rust-programming, go-guide
    // Go doc links to: rust-programming
    //
    // We use the raw link targets from parsing. Build a graph where paths
    // are the node identifiers matching the actual document paths.
    // For a realistic test, we re-map wiki-link targets to document paths.

    // Add links using the actual parsed link records.
    graph.add_links("rust.md", &rust_result.links).unwrap();
    graph.add_links("python.md", &python_result.links).unwrap();
    graph.add_links("go.md", &go_result.links).unwrap();

    let stats = graph.stats();
    assert!(
        stats.node_count >= 3,
        "graph should have at least 3 nodes, got {}",
        stats.node_count
    );
    assert!(
        stats.edge_count >= 4,
        "graph should have at least 4 edges, got {}",
        stats.edge_count
    );

    // Verify outgoing links.
    let rust_links = graph.get_links("rust.md").unwrap();
    assert!(
        !rust_links.is_empty(),
        "rust.md should have outgoing links"
    );

    let python_links = graph.get_links("python.md").unwrap();
    assert!(
        !python_links.is_empty(),
        "python.md should have outgoing links"
    );

    // Verify backlinks: rust-programming should have backlinks from python
    // and go (since they link to [[rust-programming]]).
    let rust_backlinks = graph.get_backlinks("rust-programming").unwrap();
    assert!(
        rust_backlinks.len() >= 2,
        "rust-programming should have at least 2 backlinks (from python and go), got {}",
        rust_backlinks.len()
    );

    // Shortest path: rust.md -> python-guide should be length 2 (direct link).
    let path_to_python = graph.shortest_path("rust.md", "python-guide").unwrap();
    assert!(
        path_to_python.is_some(),
        "should find a path from rust.md to python-guide"
    );
    let path = path_to_python.unwrap();
    assert_eq!(
        path.len(),
        2,
        "direct link should give path of length 2, got {:?}",
        path
    );
    assert_eq!(path[0], "rust.md");
    assert_eq!(path[1], "python-guide");

    // PageRank: nodes that receive more inbound links should rank higher.
    let scores = graph.pagerank(0.85, 50).unwrap();
    assert!(
        !scores.is_empty(),
        "pagerank should return scores for all nodes"
    );

    // rust-programming is linked from both python.md and go.md, so it
    // should have a high score (likely highest among the wiki-link targets).
    let rust_score = scores.iter().find(|s| s.path == "rust-programming");
    assert!(
        rust_score.is_some(),
        "rust-programming should appear in pagerank results"
    );

    // Connected components: The graph has two disconnected components
    // because wiki-link targets are bare names (e.g. "go-guide") while
    // the markdown link target is "go-guide.md" -- these are distinct
    // nodes. The two components are:
    //   1. {rust.md, python-guide, go-guide.md}
    //   2. {python.md, rust-programming, go-guide, go.md}
    let components = graph.connected_components().unwrap();
    assert_eq!(
        components.len(),
        2,
        "should have 2 connected components (wiki-link targets differ from md-link targets), got {}",
        components.len()
    );

    // The larger component should have 4 nodes.
    assert_eq!(
        components[0].len(),
        4,
        "larger component should have 4 nodes, got {}",
        components[0].len()
    );
    assert_eq!(
        components[1].len(),
        3,
        "smaller component should have 3 nodes, got {}",
        components[1].len()
    );
}

// ===========================================================================
// Test 4: MQL query integration
// ===========================================================================

#[test]
fn mql_query_integration() {
    // SELECT documents.
    let q = parse_query("SELECT documents").unwrap();
    match q {
        Query::Select(sel) => {
            assert_eq!(sel.entity, Entity::Documents);
            assert!(sel.conditions.is_empty());
            assert_eq!(sel.limit, None);
        }
        _ => panic!("expected Select query"),
    }

    // SELECT blocks with conditions and limit.
    let q = parse_query(
        "SELECT blocks WHERE block_type = 'heading' AND level >= 2 LIMIT 10",
    )
    .unwrap();
    match q {
        Query::Select(sel) => {
            assert_eq!(sel.entity, Entity::Blocks);
            assert_eq!(sel.conditions.len(), 2);
            assert_eq!(sel.conditions[0].field, "block_type");
            assert_eq!(sel.conditions[0].op, CompareOp::Eq);
            assert_eq!(
                sel.conditions[0].value,
                Value::String("heading".into())
            );
            assert_eq!(sel.conditions[1].field, "level");
            assert_eq!(sel.conditions[1].op, CompareOp::Gte);
            assert_eq!(sel.conditions[1].value, Value::Number(2));
            assert_eq!(sel.limit, Some(10));
        }
        _ => panic!("expected Select query"),
    }

    // SELECT with UUID value.
    let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
    let q = parse_query(&format!(
        "SELECT blocks WHERE doc_id = {}",
        uuid_str
    ))
    .unwrap();
    match q {
        Query::Select(sel) => {
            let expected = Uuid::parse_str(uuid_str).unwrap();
            assert_eq!(sel.conditions[0].value, Value::Uuid(expected));
        }
        _ => panic!("expected Select query"),
    }

    // SEARCH fulltext.
    let q = parse_query("SEARCH fulltext 'rust programming'").unwrap();
    match q {
        Query::Search(s) => {
            assert_eq!(s.search_type, SearchType::Fulltext);
            assert_eq!(s.query_text, "rust programming");
            assert_eq!(s.limit, None);
        }
        _ => panic!("expected Search query"),
    }

    // SEARCH semantic with limit.
    let q = parse_query("SEARCH semantic 'memory safety' LIMIT 5").unwrap();
    match q {
        Query::Search(s) => {
            assert_eq!(s.search_type, SearchType::Semantic);
            assert_eq!(s.query_text, "memory safety");
            assert_eq!(s.limit, Some(5));
        }
        _ => panic!("expected Search query"),
    }

    // SELECT with all comparison operators.
    for (op_str, expected_op) in [
        ("=", CompareOp::Eq),
        ("!=", CompareOp::Neq),
        (">", CompareOp::Gt),
        ("<", CompareOp::Lt),
        (">=", CompareOp::Gte),
        ("<=", CompareOp::Lte),
    ] {
        let q = parse_query(&format!(
            "SELECT blocks WHERE level {} 3",
            op_str
        ))
        .unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(
                    sel.conditions[0].op, expected_op,
                    "operator '{}' should parse correctly",
                    op_str
                );
            }
            _ => panic!("expected Select query for operator {}", op_str),
        }
    }

    // All entity types.
    for (entity_str, expected) in [
        ("documents", Entity::Documents),
        ("blocks", Entity::Blocks),
        ("links", Entity::Links),
        ("tags", Entity::Tags),
    ] {
        let q = parse_query(&format!("SELECT {}", entity_str)).unwrap();
        match q {
            Query::Select(sel) => {
                assert_eq!(
                    sel.entity, expected,
                    "entity '{}' should parse correctly",
                    entity_str
                );
            }
            _ => panic!("expected Select for entity {}", entity_str),
        }
    }

    // Case insensitivity of keywords.
    let q = parse_query("select DOCUMENTS where path = 'test.md'").unwrap();
    match q {
        Query::Select(sel) => {
            assert_eq!(sel.entity, Entity::Documents);
            assert_eq!(sel.conditions.len(), 1);
        }
        _ => panic!("expected Select query"),
    }

    // Invalid query returns error.
    assert!(
        parse_query("INVALID query text").is_err(),
        "invalid query should return error"
    );
    assert!(
        parse_query("").is_err(),
        "empty query should return error"
    );
}

// ===========================================================================
// Test 5: MVCC snapshot isolation
// ===========================================================================

#[test]
fn mvcc_snapshot_isolation() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");

    let engine = CustomStorageEngine::new(&data_dir).unwrap();

    // Insert document A.
    let doc_a = parse_markdown("a.md", "# Document A\n\nContent of A.\n").unwrap();
    engine.insert_document(&doc_a.document).unwrap();
    engine.insert_blocks(&doc_a.blocks).unwrap();

    // Take snapshot.
    let snapshot = engine.snapshot().unwrap();

    // Insert document B after the snapshot.
    let doc_b = parse_markdown("b.md", "# Document B\n\nContent of B.\n").unwrap();
    engine.insert_document(&doc_b.document).unwrap();
    engine.insert_blocks(&doc_b.blocks).unwrap();

    // Snapshot should only see document A.
    assert_eq!(
        snapshot.list_documents().unwrap().len(),
        1,
        "snapshot should see exactly 1 document"
    );
    assert!(
        snapshot
            .get_document(&doc_a.document.doc_id)
            .unwrap()
            .is_some(),
        "snapshot should see document A"
    );
    assert!(
        snapshot
            .get_document(&doc_b.document.doc_id)
            .unwrap()
            .is_none(),
        "snapshot should NOT see document B"
    );
    assert!(
        snapshot
            .get_document_by_path("a.md")
            .unwrap()
            .is_some(),
        "snapshot should resolve path a.md"
    );
    assert!(
        snapshot
            .get_document_by_path("b.md")
            .unwrap()
            .is_none(),
        "snapshot should NOT resolve path b.md"
    );

    // Snapshot blocks should only include A's blocks.
    let snap_blocks_a = snapshot
        .get_blocks_by_doc(&doc_a.document.doc_id)
        .unwrap();
    assert!(
        !snap_blocks_a.is_empty(),
        "snapshot should have blocks for A"
    );
    let snap_blocks_b = snapshot
        .get_blocks_by_doc(&doc_b.document.doc_id)
        .unwrap();
    assert!(
        snap_blocks_b.is_empty(),
        "snapshot should NOT have blocks for B"
    );

    // Engine should see both documents.
    assert_eq!(
        engine.list_documents().unwrap().len(),
        2,
        "engine should see both documents"
    );
    assert!(
        engine
            .get_document(&doc_b.document.doc_id)
            .unwrap()
            .is_some(),
        "engine should see document B"
    );

    // Version ordering.
    assert!(
        engine.version() > snapshot.version(),
        "engine version should be higher than snapshot version"
    );

    // Snapshot write methods should return errors.
    let write_err = snapshot.insert_document(&doc_b.document);
    assert!(write_err.is_err(), "snapshot writes should fail");
    let err_msg = write_err.unwrap_err().to_string();
    assert!(
        err_msg.contains("read-only"),
        "error should mention read-only, got: {}",
        err_msg
    );
}

// ===========================================================================
// Test 6: Parse + Store + Delete cascade
// ===========================================================================

#[test]
fn parse_store_delete_cascade() {
    let tmp = tempfile::tempdir().unwrap();
    let sled_dir = tmp.path().join("sled");

    let storage = SledStorage::new(&sled_dir).unwrap();
    let result = ingest_document(&storage, "doomed.md", RUST_DOC);
    let doc_id = result.document.doc_id;

    // Verify everything was stored.
    assert!(storage.get_document(&doc_id).unwrap().is_some());
    assert!(!storage.get_blocks_by_doc(&doc_id).unwrap().is_empty());
    assert!(!storage.get_links_from(&doc_id).unwrap().is_empty());
    assert!(!storage.get_tags_by_doc(&doc_id).unwrap().is_empty());

    // Delete and verify cascade.
    storage.delete_document(&doc_id).unwrap();

    assert!(storage.get_document(&doc_id).unwrap().is_none());
    assert!(
        storage
            .get_document_by_path("doomed.md")
            .unwrap()
            .is_none()
    );
    assert!(storage.get_blocks_by_doc(&doc_id).unwrap().is_empty());
    assert!(storage.get_links_from(&doc_id).unwrap().is_empty());
    assert!(storage.get_tags_by_doc(&doc_id).unwrap().is_empty());

    // Verify backlink index is also cleaned up for wiki-link targets.
    assert!(
        storage.get_links_to("python-guide").unwrap().is_empty(),
        "backlinks to python-guide should be removed after cascade delete"
    );
}

// ===========================================================================
// Test 7: Embedding pipeline end-to-end with vector search
// ===========================================================================

#[test]
fn embedding_pipeline_end_to_end() {
    let pipeline = EmbeddingPipeline::default_tfidf();
    let vec_index = VectorIndex::new(pipeline.dimension());

    // Parse two documents with different topics.
    let rust_result = parse_markdown("rust.md", RUST_DOC).unwrap();
    let python_result = parse_markdown("python.md", PYTHON_DOC).unwrap();

    // Embed and index blocks from both documents.
    for result in [&rust_result, &python_result] {
        let embeddings = pipeline.embed_blocks(&result.blocks).unwrap();
        for (block_id, vector) in &embeddings {
            let block = result
                .blocks
                .iter()
                .find(|b| b.block_id == *block_id)
                .unwrap();
            vec_index
                .insert(
                    &result.document.doc_id.to_string(),
                    &block_id.to_string(),
                    &result.document.path,
                    &block.text_content,
                    vector.clone(),
                )
                .unwrap();
        }
    }

    assert!(
        vec_index.len() >= 2,
        "vector index should have entries from both docs"
    );

    // Search for "systems programming memory safety" -- should favor Rust doc.
    let query_vec = pipeline
        .embed_text("systems programming memory safety")
        .unwrap();
    let results = vec_index.search(&query_vec, 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(
        results[0].path, "rust.md",
        "top result for 'systems programming memory safety' should be from rust.md"
    );

    // Search for "dynamic typing scripting" -- should favor Python doc.
    let query_vec2 = pipeline
        .embed_text("dynamic typing scripting interpreted")
        .unwrap();
    let results2 = vec_index.search(&query_vec2, 5).unwrap();
    assert!(!results2.is_empty());
    assert_eq!(
        results2[0].path, "python.md",
        "top result for 'dynamic typing scripting' should be from python.md"
    );
}

// ===========================================================================
// Test 8: Full-text search across multiple documents
// ===========================================================================

#[test]
fn fulltext_search_multi_document() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");

    let fts = FullTextIndex::new(&index_dir).unwrap();

    let rust_result = parse_markdown("rust.md", RUST_DOC).unwrap();
    let python_result = parse_markdown("python.md", PYTHON_DOC).unwrap();
    let go_result = parse_markdown("go.md", GO_DOC).unwrap();

    fts.index_document(&rust_result.document, &rust_result.blocks)
        .unwrap();
    fts.index_document(&python_result.document, &python_result.blocks)
        .unwrap();
    fts.index_document(&go_result.document, &go_result.blocks)
        .unwrap();

    // Search for "goroutines concurrency" -- should find Go doc.
    let results = fts.search("goroutines concurrency", 10).unwrap();
    assert!(
        !results.is_empty(),
        "should find results for 'goroutines concurrency'"
    );
    assert_eq!(
        results[0].path, "go.md",
        "top result should be go.md"
    );

    // Search for "duck typing" -- should find Python doc.
    let results2 = fts.search("duck typing", 10).unwrap();
    assert!(
        !results2.is_empty(),
        "should find results for 'duck typing'"
    );
    assert_eq!(
        results2[0].path, "python.md",
        "top result should be python.md"
    );

    // Search for "memory safety" -- should find Rust doc.
    let results3 = fts.search("memory safety", 10).unwrap();
    assert!(
        !results3.is_empty(),
        "should find results for 'memory safety'"
    );
    assert_eq!(
        results3[0].path, "rust.md",
        "top result should be rust.md"
    );

    // Delete one document and verify it is no longer searchable.
    fts.delete_document(&rust_result.document.doc_id).unwrap();
    let results4 = fts.search("memory safety", 10).unwrap();
    for r in &results4 {
        assert_ne!(
            r.doc_id,
            rust_result.document.doc_id.to_string(),
            "deleted document should not appear in search results"
        );
    }
}

// ===========================================================================
// Test 9: Custom engine CRUD + query_blocks
// ===========================================================================

#[test]
fn custom_engine_crud_operations() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");

    let engine = CustomStorageEngine::new(&data_dir).unwrap();

    // Parse and ingest all three documents.
    let rust = ingest_document(&engine, "rust.md", RUST_DOC);
    let python = ingest_document(&engine, "python.md", PYTHON_DOC);
    let go = ingest_document(&engine, "go.md", GO_DOC);

    // Verify list_documents.
    let docs = engine.list_documents().unwrap();
    assert_eq!(docs.len(), 3);

    // Query all headings across all documents (no doc_id filter).
    let all_headings = engine
        .query_blocks(None, Some("heading"), None, None)
        .unwrap();
    assert!(
        all_headings.len() >= 6,
        "should have at least 6 headings across 3 docs, got {}",
        all_headings.len()
    );

    // Query code blocks by language across all documents.
    let rust_code = engine
        .query_blocks(None, Some("code_block"), None, Some("rust"))
        .unwrap();
    assert_eq!(rust_code.len(), 1, "should have exactly 1 Rust code block");

    let python_code = engine
        .query_blocks(None, Some("code_block"), None, Some("python"))
        .unwrap();
    assert_eq!(
        python_code.len(),
        1,
        "should have exactly 1 Python code block"
    );

    let go_code = engine
        .query_blocks(None, Some("code_block"), None, Some("go"))
        .unwrap();
    assert_eq!(go_code.len(), 1, "should have exactly 1 Go code block");

    // Query H1 headings specifically.
    let h1_blocks = engine
        .query_blocks(None, Some("heading"), Some(1), None)
        .unwrap();
    assert_eq!(
        h1_blocks.len(),
        3,
        "each document should have exactly one H1"
    );

    // Delete one document and verify isolation.
    engine.delete_document(&python.document.doc_id).unwrap();
    let docs_after = engine.list_documents().unwrap();
    assert_eq!(docs_after.len(), 2);

    let remaining_headings = engine
        .query_blocks(None, Some("heading"), None, None)
        .unwrap();
    // Should no longer include Python's headings.
    for h in &remaining_headings {
        assert_ne!(
            h.doc_id, python.document.doc_id,
            "Python blocks should be cascade-deleted"
        );
    }

    // Verify the other two documents are still intact.
    assert!(
        engine
            .get_document(&rust.document.doc_id)
            .unwrap()
            .is_some()
    );
    assert!(
        engine
            .get_document(&go.document.doc_id)
            .unwrap()
            .is_some()
    );
}

// ===========================================================================
// Test 10: Graph disconnected components
// ===========================================================================

#[test]
fn graph_disconnected_components() {
    let graph = KnowledgeGraph::new();

    // Create two isolated clusters.
    // Cluster 1: a -> b -> c
    let links_a = vec![LinkRecord {
        link_id: Uuid::new_v4(),
        source_doc_id: Uuid::new_v4(),
        source_block_id: None,
        target: "b.md".to_string(),
        target_doc_id: None,
        link_type: LinkType::WikiLink,
        anchor_text: "b".to_string(),
    }];
    graph.add_links("a.md", &links_a).unwrap();

    let links_b = vec![LinkRecord {
        link_id: Uuid::new_v4(),
        source_doc_id: Uuid::new_v4(),
        source_block_id: None,
        target: "c.md".to_string(),
        target_doc_id: None,
        link_type: LinkType::WikiLink,
        anchor_text: "c".to_string(),
    }];
    graph.add_links("b.md", &links_b).unwrap();

    // Cluster 2: x -> y
    let links_x = vec![LinkRecord {
        link_id: Uuid::new_v4(),
        source_doc_id: Uuid::new_v4(),
        source_block_id: None,
        target: "y.md".to_string(),
        target_doc_id: None,
        link_type: LinkType::MarkdownLink,
        anchor_text: "y".to_string(),
    }];
    graph.add_links("x.md", &links_x).unwrap();

    // Verify connected components.
    let components = graph.connected_components().unwrap();
    assert_eq!(
        components.len(),
        2,
        "should have 2 connected components"
    );

    // Larger component first (3 nodes), then smaller (2 nodes).
    assert_eq!(components[0].len(), 3);
    assert_eq!(components[1].len(), 2);

    // Shortest path within a cluster.
    let path = graph.shortest_path("a.md", "c.md").unwrap();
    assert_eq!(
        path,
        Some(vec!["a.md".to_owned(), "b.md".to_owned(), "c.md".to_owned()])
    );

    // No path between clusters.
    let no_path = graph.shortest_path("a.md", "y.md").unwrap();
    assert_eq!(no_path, None, "no path should exist between clusters");

    // PageRank: c.md should rank higher than a.md within cluster 1
    // because it receives an inbound link from b.md.
    let scores = graph.pagerank(0.85, 50).unwrap();
    let score_of = |path: &str| -> f64 {
        scores.iter().find(|s| s.path == path).unwrap().score
    };
    assert!(
        score_of("c.md") > score_of("a.md"),
        "c.md ({}) should outrank a.md ({})",
        score_of("c.md"),
        score_of("a.md")
    );
}

// ===========================================================================
// Test 11: Reindex full-text search
// ===========================================================================

#[test]
fn fulltext_reindex_replaces_all_data() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");

    let fts = FullTextIndex::new(&index_dir).unwrap();

    // Index the Rust document.
    let rust_result = parse_markdown("rust.md", RUST_DOC).unwrap();
    fts.index_document(&rust_result.document, &rust_result.blocks)
        .unwrap();

    // Verify it is searchable.
    let before = fts.search("memory safety", 10).unwrap();
    assert!(!before.is_empty());

    // Reindex with only the Python document.
    let python_result = parse_markdown("python.md", PYTHON_DOC).unwrap();
    fts.reindex(&[(python_result.document.clone(), python_result.blocks.clone())])
        .unwrap();

    // Rust doc should be gone.
    let after_rust = fts.search("memory safety", 10).unwrap();
    for r in &after_rust {
        assert_ne!(r.path, "rust.md", "rust.md should be gone after reindex");
    }

    // Python doc should be present.
    let after_python = fts.search("dynamic typing", 10).unwrap();
    assert!(!after_python.is_empty());
    assert_eq!(after_python[0].path, "python.md");
}

// ===========================================================================
// Test 12: Parser edge cases fed through storage
// ===========================================================================

#[test]
fn parser_edge_cases_through_storage() {
    let tmp = tempfile::tempdir().unwrap();
    let sled_dir = tmp.path().join("sled");
    let storage = SledStorage::new(&sled_dir).unwrap();

    // Empty document.
    let empty = ingest_document(&storage, "empty.md", "");
    assert!(empty.blocks.is_empty());
    assert!(empty.tags.is_empty());
    assert!(empty.links.is_empty());
    assert!(empty.document.title.is_none());
    assert!(
        storage
            .get_document_by_path("empty.md")
            .unwrap()
            .is_some()
    );

    // Document with only a heading (no frontmatter).
    let heading_only = ingest_document(
        &storage,
        "heading.md",
        "# Just A Heading\n",
    );
    assert_eq!(
        heading_only.document.title,
        Some("Just A Heading".to_string())
    );
    assert!(heading_only.document.frontmatter.is_null());

    // Document with nested lists.
    let nested_list = ingest_document(
        &storage,
        "nested.md",
        "- outer\n  - inner\n    - deep\n",
    );
    let list_items: Vec<_> = nested_list
        .blocks
        .iter()
        .filter(|b| matches!(b.block_type, BlockType::ListItem))
        .collect();
    assert!(
        list_items.len() >= 3,
        "should have at least 3 list items, got {}",
        list_items.len()
    );
    // At least one should have a parent.
    assert!(
        list_items.iter().any(|b| b.parent_block_id.is_some()),
        "nested list items should have parent_block_id set"
    );

    // Document with only frontmatter, no body.
    let frontmatter_only = ingest_document(
        &storage,
        "frontmatter.md",
        "---\ntitle: Just Frontmatter\nstatus: draft\n---\n",
    );
    assert_eq!(
        frontmatter_only.document.title,
        Some("Just Frontmatter".to_string())
    );
    let fm_tags = storage
        .get_tags_by_doc(&frontmatter_only.document.doc_id)
        .unwrap();
    assert!(
        !fm_tags.is_empty(),
        "frontmatter-only doc should still have tags"
    );
}

// ===========================================================================
// Test 13: Multiple snapshots at different points in time
// ===========================================================================

#[test]
fn multiple_snapshots_timeline() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");

    let engine = CustomStorageEngine::new(&data_dir).unwrap();

    // Snapshot 0: empty.
    let snap0 = engine.snapshot().unwrap();

    // Insert document A.
    let doc_a = parse_markdown("a.md", "# Alpha\n\nContent A.\n").unwrap();
    engine.insert_document(&doc_a.document).unwrap();
    engine.insert_blocks(&doc_a.blocks).unwrap();

    // Snapshot 1: one document.
    let snap1 = engine.snapshot().unwrap();

    // Insert document B.
    let doc_b = parse_markdown("b.md", "# Beta\n\nContent B.\n").unwrap();
    engine.insert_document(&doc_b.document).unwrap();
    engine.insert_blocks(&doc_b.blocks).unwrap();

    // Snapshot 2: two documents.
    let snap2 = engine.snapshot().unwrap();

    // Insert document C.
    let doc_c = parse_markdown("c.md", "# Gamma\n\nContent C.\n").unwrap();
    engine.insert_document(&doc_c.document).unwrap();
    engine.insert_blocks(&doc_c.blocks).unwrap();

    // Verify each snapshot sees the correct number of documents.
    assert_eq!(snap0.list_documents().unwrap().len(), 0);
    assert_eq!(snap1.list_documents().unwrap().len(), 1);
    assert_eq!(snap2.list_documents().unwrap().len(), 2);
    assert_eq!(engine.list_documents().unwrap().len(), 3);

    // Verify version ordering.
    assert!(snap0.version() < snap1.version());
    assert!(snap1.version() < snap2.version());
    assert!(snap2.version() < engine.version());

    // Each snapshot is immutable: deleting from engine does not affect any.
    engine.delete_document(&doc_a.document.doc_id).unwrap();
    assert_eq!(engine.list_documents().unwrap().len(), 2);

    // snap1 still sees document A.
    assert!(
        snap1
            .get_document(&doc_a.document.doc_id)
            .unwrap()
            .is_some()
    );
    // snap2 also still sees document A.
    assert!(
        snap2
            .get_document(&doc_a.document.doc_id)
            .unwrap()
            .is_some()
    );
}
