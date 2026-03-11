use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};
use tracing::debug;
use uuid::Uuid;

use mdbase_core::error::MdbaseError;
use mdbase_core::types::{BlockRecord, DocumentRecord};

use crate::types::SearchResult;

/// Heap size budget for the Tantivy index writer (50 MB).
const WRITER_HEAP_SIZE: usize = 50_000_000;

/// Full-text search index backed by Tantivy.
///
/// Each indexed document produces one "document-level" entry (carrying the
/// title and path) plus one entry per block (carrying the block's plain-text
/// content). All entries share the same `doc_id` so that deleting a document
/// removes every associated entry in a single pass.
pub struct FullTextIndex {
    index: Index,
    reader: IndexReader,
    doc_id_field: Field,
    block_id_field: Field,
    path_field: Field,
    title_field: Field,
    content_field: Field,
    block_type_field: Field,
}

impl FullTextIndex {
    /// Open or create a Tantivy index at `index_dir`.
    ///
    /// The directory is created if it does not exist. If an index already
    /// exists there its schema is validated against the expected layout.
    pub fn new(index_dir: &Path) -> mdbase_core::Result<Self> {
        let mut schema_builder = Schema::builder();

        let doc_id_field = schema_builder.add_text_field("doc_id", STRING | STORED);
        let block_id_field = schema_builder.add_text_field("block_id", STRING | STORED);
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let block_type_field = schema_builder.add_text_field("block_type", STRING | STORED);

        let schema = schema_builder.build();

        std::fs::create_dir_all(index_dir).map_err(|e| {
            MdbaseError::IndexError(format!("failed to create index directory: {e}"))
        })?;

        let index = Index::open_or_create(
            tantivy::directory::MmapDirectory::open(index_dir)
                .map_err(|e| MdbaseError::IndexError(e.to_string()))?,
            schema,
        )
        .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e: tantivy::TantivyError| MdbaseError::IndexError(e.to_string()))?;

        debug!("opened full-text index at {}", index_dir.display());

        Ok(Self {
            index,
            reader,
            doc_id_field,
            block_id_field,
            path_field,
            title_field,
            content_field,
            block_type_field,
        })
    }

    // ------------------------------------------------------------------
    // Indexing
    // ------------------------------------------------------------------

    /// Index a single document and all of its blocks.
    ///
    /// A "document-level" entry is created with the title and path so that
    /// title-only queries still return results. Then one entry per block is
    /// added with the block's plain-text content.
    ///
    /// Any previous entries for this `doc_id` are deleted first to avoid
    /// duplicates on re-index.
    pub fn index_document(
        &self,
        doc: &DocumentRecord,
        blocks: &[BlockRecord],
    ) -> mdbase_core::Result<()> {
        let mut writer = self.writer()?;
        self.add_document_to_writer(&mut writer, doc, blocks)?;
        writer
            .commit()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;
        self.reader
            .reload()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        debug!(
            doc_id = %doc.doc_id,
            blocks = blocks.len(),
            "indexed document"
        );

        Ok(())
    }

    /// Delete every index entry belonging to the given document.
    pub fn delete_document(&self, doc_id: &Uuid) -> mdbase_core::Result<()> {
        let mut writer = self.writer()?;
        let term = Term::from_field_text(self.doc_id_field, &doc_id.to_string());
        writer.delete_term(term);
        writer
            .commit()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        self.reader
            .reload()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        debug!(%doc_id, "deleted document from index");
        Ok(())
    }

    /// Drop all existing entries and re-index the given set of documents.
    pub fn reindex(
        &self,
        docs: &[(DocumentRecord, Vec<BlockRecord>)],
    ) -> mdbase_core::Result<()> {
        let mut writer = self.writer()?;
        writer
            .delete_all_documents()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        for (doc, blocks) in docs {
            self.add_document_to_writer(&mut writer, doc, blocks)?;
        }

        writer
            .commit()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;
        self.reader
            .reload()
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        debug!(count = docs.len(), "full reindex complete");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Searching
    // ------------------------------------------------------------------

    /// Execute a full-text query and return the top `limit` results.
    ///
    /// The query targets the `content` and `title` fields. Tantivy's default
    /// query syntax is supported (boolean operators, phrase queries, etc.).
    pub fn search(
        &self,
        query_str: &str,
        limit: usize,
    ) -> mdbase_core::Result<Vec<SearchResult>> {
        let searcher = self.reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.content_field, self.title_field]);

        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| MdbaseError::IndexError(format!("query parse error: {e}")))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        // Build a snippet generator for the content field.
        let snippet_generator =
            tantivy::SnippetGenerator::create(&searcher, &query, self.content_field)
                .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            let retrieved = searcher
                .doc(doc_address)
                .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

            let doc_id = field_text(&retrieved, self.doc_id_field).unwrap_or_default();
            let block_id_raw = field_text(&retrieved, self.block_id_field).unwrap_or_default();
            let path = field_text(&retrieved, self.path_field).unwrap_or_default();
            let title_raw = field_text(&retrieved, self.title_field).unwrap_or_default();
            let block_type_raw =
                field_text(&retrieved, self.block_type_field).unwrap_or_default();

            let snippet_markup = snippet_generator.snippet_from_doc(&retrieved);
            let snippet_text = snippet_markup.to_html();
            // If the snippet generator returned nothing useful (e.g. the match was
            // in the title rather than the content), fall back to the stored content.
            let snippet = if snippet_text.trim().is_empty() {
                field_text(&retrieved, self.content_field).unwrap_or_default()
            } else {
                snippet_text
            };

            results.push(SearchResult {
                doc_id,
                block_id: if block_id_raw.is_empty() {
                    None
                } else {
                    Some(block_id_raw)
                },
                path,
                title: if title_raw.is_empty() {
                    None
                } else {
                    Some(title_raw)
                },
                block_type: if block_type_raw.is_empty() {
                    None
                } else {
                    Some(block_type_raw)
                },
                snippet,
                score,
            });
        }

        debug!(query = query_str, hits = results.len(), "search executed");
        Ok(results)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Add a single document (plus its blocks) to the given writer.
    ///
    /// Deletes any existing entries for this `doc_id` first, then adds a
    /// document-level entry and one entry per block. Does **not** commit —
    /// the caller is responsible for committing and reloading the reader.
    fn add_document_to_writer(
        &self,
        writer: &mut IndexWriter,
        doc: &DocumentRecord,
        blocks: &[BlockRecord],
    ) -> mdbase_core::Result<()> {
        let doc_id_str = doc.doc_id.to_string();

        // Remove stale entries for this document.
        let delete_term = Term::from_field_text(self.doc_id_field, &doc_id_str);
        writer.delete_term(delete_term);

        // Document-level entry (no block_id, content is the title if present).
        let title_text = doc.title.as_deref().unwrap_or("");
        writer
            .add_document(doc!(
                self.doc_id_field => doc_id_str.as_str(),
                self.block_id_field => "",
                self.path_field => doc.path.as_str(),
                self.title_field => title_text,
                self.content_field => title_text,
                self.block_type_field => "",
            ))
            .map_err(|e| MdbaseError::IndexError(e.to_string()))?;

        // One entry per block.
        for block in blocks {
            let block_type_tag = block_type_to_tag(&block.block_type);
            writer
                .add_document(doc!(
                    self.doc_id_field => doc_id_str.as_str(),
                    self.block_id_field => block.block_id.to_string(),
                    self.path_field => doc.path.as_str(),
                    self.title_field => title_text,
                    self.content_field => block.text_content.as_str(),
                    self.block_type_field => block_type_tag,
                ))
                .map_err(|e| MdbaseError::IndexError(e.to_string()))?;
        }

        Ok(())
    }

    /// Create a fresh `IndexWriter` with a fixed heap budget.
    fn writer(&self) -> mdbase_core::Result<IndexWriter> {
        self.index
            .writer(WRITER_HEAP_SIZE)
            .map_err(|e| MdbaseError::IndexError(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

/// Extract the first text value for `field` from a Tantivy document.
fn field_text(doc: &tantivy::TantivyDocument, field: Field) -> Option<String> {
    doc.get_first(field).and_then(|v| {
        if let tantivy::schema::OwnedValue::Str(s) = v {
            Some(s.clone())
        } else {
            None
        }
    })
}

/// Map a `BlockType` variant to a short string tag for storage in the index.
fn block_type_to_tag(bt: &mdbase_core::types::BlockType) -> &'static str {
    use mdbase_core::types::BlockType;
    match bt {
        BlockType::Heading { .. } => "heading",
        BlockType::Paragraph => "paragraph",
        BlockType::CodeBlock { .. } => "code_block",
        BlockType::List { .. } => "list",
        BlockType::ListItem => "list_item",
        BlockType::Table => "table",
        BlockType::BlockQuote => "block_quote",
        BlockType::ThematicBreak => "thematic_break",
        BlockType::Image { .. } => "image",
        BlockType::Html => "html",
        BlockType::FootnoteDefinition => "footnote_definition",
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use mdbase_core::types::{BlockType, DocumentRecord};

    /// Create a minimal `DocumentRecord` for testing.
    fn make_doc(title: &str, path: &str) -> DocumentRecord {
        DocumentRecord {
            doc_id: Uuid::new_v4(),
            path: path.to_string(),
            title: Some(title.to_string()),
            frontmatter: serde_json::json!({}),
            raw_content_hash: "abc123".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
        }
    }

    /// Create a paragraph `BlockRecord` tied to `doc_id`.
    fn make_block(doc_id: Uuid, text: &str) -> BlockRecord {
        BlockRecord {
            block_id: Uuid::new_v4(),
            doc_id,
            block_type: BlockType::Paragraph,
            depth: 0,
            ordinal: 0,
            parent_block_id: None,
            text_content: text.to_string(),
            raw_markdown: text.to_string(),
        }
    }

    #[test]
    fn index_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let idx = FullTextIndex::new(dir.path()).unwrap();

        let doc = make_doc("Rust Ownership", "notes/rust.md");
        let blocks = vec![
            make_block(
                doc.doc_id,
                "Rust uses an ownership model to manage memory safely without a garbage collector.",
            ),
            make_block(
                doc.doc_id,
                "Each value in Rust has a single owner at any point in time.",
            ),
        ];

        idx.index_document(&doc, &blocks).unwrap();

        // Search for a term that appears in one of the blocks.
        let results = idx.search("ownership", 10).unwrap();
        assert!(
            !results.is_empty(),
            "expected at least one result for 'ownership'"
        );

        // Every result should belong to the document we indexed.
        for r in &results {
            assert_eq!(r.doc_id, doc.doc_id.to_string());
            assert_eq!(r.path, "notes/rust.md");
        }
    }

    #[test]
    fn delete_removes_all_entries() {
        let dir = tempfile::tempdir().unwrap();
        let idx = FullTextIndex::new(dir.path()).unwrap();

        let doc = make_doc("Concurrency", "notes/concurrency.md");
        let blocks = vec![make_block(
            doc.doc_id,
            "Fearless concurrency is one of Rust's marquee features.",
        )];

        idx.index_document(&doc, &blocks).unwrap();

        // Sanity check: search should return something.
        let before = idx.search("concurrency", 10).unwrap();
        assert!(!before.is_empty());

        idx.delete_document(&doc.doc_id).unwrap();

        let after = idx.search("concurrency", 10).unwrap();
        assert!(after.is_empty(), "expected no results after delete");
    }

    #[test]
    fn search_no_matches_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let idx = FullTextIndex::new(dir.path()).unwrap();

        let doc = make_doc("Hello", "notes/hello.md");
        let blocks = vec![make_block(doc.doc_id, "This is a simple greeting note.")];

        idx.index_document(&doc, &blocks).unwrap();

        let results = idx.search("xylophone", 10).unwrap();
        assert!(results.is_empty(), "expected no results for unrelated term");
    }

    #[test]
    fn reindex_replaces_all_data() {
        let dir = tempfile::tempdir().unwrap();
        let idx = FullTextIndex::new(dir.path()).unwrap();

        // Index an initial document.
        let doc_a = make_doc("Alpha", "alpha.md");
        let blocks_a = vec![make_block(doc_a.doc_id, "Alpha content about databases.")];
        idx.index_document(&doc_a, &blocks_a).unwrap();

        // Now reindex with a completely different document set.
        let doc_b = make_doc("Beta", "beta.md");
        let blocks_b = vec![make_block(doc_b.doc_id, "Beta content about networking.")];
        idx.reindex(&[(doc_b.clone(), blocks_b)]).unwrap();

        // The old document should be gone.
        let old = idx.search("databases", 10).unwrap();
        assert!(old.is_empty(), "old document should have been removed");

        // The new document should be findable.
        let new = idx.search("networking", 10).unwrap();
        assert!(!new.is_empty(), "new document should be searchable");
        assert_eq!(new[0].doc_id, doc_b.doc_id.to_string());
    }
}
