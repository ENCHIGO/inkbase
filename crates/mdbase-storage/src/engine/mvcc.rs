//! MVCC read snapshots for the custom storage engine.
//!
//! A [`StorageSnapshot`] is a materialized, read-only copy of the storage state
//! at a specific version. It is created by calling
//! [`CustomStorageEngine::snapshot()`] and captures all current data into
//! in-memory collections.
//!
//! The snapshot implements the [`Storage`] trait. Read methods serve data from
//! the cached collections; write methods return a read-only error.

use std::collections::HashMap;

use uuid::Uuid;

use mdbase_core::types::{BlockRecord, BlockType, DocumentRecord, LinkRecord, TagRecord};
use mdbase_core::{MdbaseError, Result};

use crate::Storage;

// ---------------------------------------------------------------------------
// Block type matching (duplicated from storage_engine to keep this module
// self-contained — the function is trivial)
// ---------------------------------------------------------------------------

/// Return the serde tag name for a `BlockType` variant.
fn block_type_name(bt: &BlockType) -> &'static str {
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

/// Helper to produce a consistent read-only error.
fn read_only_error() -> MdbaseError {
    MdbaseError::StorageError("snapshot is read-only".to_string())
}

// ---------------------------------------------------------------------------
// StorageSnapshot
// ---------------------------------------------------------------------------

/// A read-only, point-in-time snapshot of the storage state.
///
/// Created by [`CustomStorageEngine::snapshot()`]. All data is materialized
/// into memory at creation time so that subsequent writes to the engine do
/// not affect the snapshot.
pub struct StorageSnapshot {
    /// The engine version at which this snapshot was taken.
    version: u64,

    /// All documents, keyed by `doc_id`.
    documents: HashMap<Uuid, DocumentRecord>,
    /// Reverse index: file path to `doc_id`.
    documents_by_path: HashMap<String, Uuid>,

    /// Blocks grouped by owning `doc_id`.
    blocks_by_doc: HashMap<Uuid, Vec<BlockRecord>>,

    /// Outgoing links grouped by source `doc_id`.
    links_by_source: HashMap<Uuid, Vec<LinkRecord>>,
    /// Incoming links grouped by target path.
    links_by_target: HashMap<String, Vec<LinkRecord>>,

    /// Tags grouped by `doc_id`.
    tags_by_doc: HashMap<Uuid, Vec<TagRecord>>,
}

impl StorageSnapshot {
    /// Build a snapshot from data read out of the engine.
    ///
    /// This is `pub(crate)` — only `CustomStorageEngine` should construct
    /// snapshots.
    pub(crate) fn new(
        version: u64,
        documents: Vec<DocumentRecord>,
        blocks: Vec<BlockRecord>,
        links: Vec<LinkRecord>,
        tags: Vec<TagRecord>,
    ) -> Self {
        // Index documents.
        let mut doc_map = HashMap::with_capacity(documents.len());
        let mut doc_path_map = HashMap::with_capacity(documents.len());
        for doc in documents {
            doc_path_map.insert(doc.path.clone(), doc.doc_id);
            doc_map.insert(doc.doc_id, doc);
        }

        // Index blocks by doc_id.
        let mut blocks_by_doc: HashMap<Uuid, Vec<BlockRecord>> = HashMap::new();
        for block in blocks {
            blocks_by_doc
                .entry(block.doc_id)
                .or_default()
                .push(block);
        }

        // Index links by source and target.
        let mut links_by_source: HashMap<Uuid, Vec<LinkRecord>> = HashMap::new();
        let mut links_by_target: HashMap<String, Vec<LinkRecord>> = HashMap::new();
        for link in links {
            links_by_source
                .entry(link.source_doc_id)
                .or_default()
                .push(link.clone());
            links_by_target
                .entry(link.target.clone())
                .or_default()
                .push(link);
        }

        // Index tags by doc_id.
        let mut tags_by_doc_map: HashMap<Uuid, Vec<TagRecord>> = HashMap::new();
        for tag in tags {
            tags_by_doc_map.entry(tag.doc_id).or_default().push(tag);
        }

        Self {
            version,
            documents: doc_map,
            documents_by_path: doc_path_map,
            blocks_by_doc,
            links_by_source,
            links_by_target,
            tags_by_doc: tags_by_doc_map,
        }
    }

    /// The engine version at which this snapshot was taken.
    pub fn version(&self) -> u64 {
        self.version
    }
}

impl Storage for StorageSnapshot {
    // -----------------------------------------------------------------------
    // Documents (read)
    // -----------------------------------------------------------------------

    fn get_document(&self, doc_id: &Uuid) -> Result<Option<DocumentRecord>> {
        Ok(self.documents.get(doc_id).cloned())
    }

    fn get_document_by_path(&self, path: &str) -> Result<Option<DocumentRecord>> {
        let doc_id = match self.documents_by_path.get(path) {
            Some(id) => id,
            None => return Ok(None),
        };
        Ok(self.documents.get(doc_id).cloned())
    }

    fn list_documents(&self) -> Result<Vec<DocumentRecord>> {
        Ok(self.documents.values().cloned().collect())
    }

    // -----------------------------------------------------------------------
    // Documents (write — rejected)
    // -----------------------------------------------------------------------

    fn insert_document(&self, _doc: &DocumentRecord) -> Result<()> {
        Err(read_only_error())
    }

    fn delete_document(&self, _doc_id: &Uuid) -> Result<()> {
        Err(read_only_error())
    }

    // -----------------------------------------------------------------------
    // Blocks (read)
    // -----------------------------------------------------------------------

    fn get_blocks_by_doc(&self, doc_id: &Uuid) -> Result<Vec<BlockRecord>> {
        Ok(self
            .blocks_by_doc
            .get(doc_id)
            .cloned()
            .unwrap_or_default())
    }

    fn query_blocks(
        &self,
        doc_id: Option<&Uuid>,
        block_type: Option<&str>,
        heading_level: Option<u8>,
        language: Option<&str>,
    ) -> Result<Vec<BlockRecord>> {
        let candidates: Vec<&BlockRecord> = if let Some(id) = doc_id {
            match self.blocks_by_doc.get(id) {
                Some(blocks) => blocks.iter().collect(),
                None => Vec::new(),
            }
        } else {
            self.blocks_by_doc.values().flat_map(|v| v.iter()).collect()
        };

        let results = candidates
            .into_iter()
            .filter(|block| {
                if let Some(bt_name) = block_type {
                    if block_type_name(&block.block_type) != bt_name {
                        return false;
                    }
                }
                if let Some(level) = heading_level {
                    match &block.block_type {
                        BlockType::Heading { level: l } if *l == level => {}
                        _ => return false,
                    }
                }
                if let Some(lang) = language {
                    match &block.block_type {
                        BlockType::CodeBlock {
                            language: Some(l),
                        } if l == lang => {}
                        _ => return false,
                    }
                }
                true
            })
            .cloned()
            .collect();

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Blocks (write — rejected)
    // -----------------------------------------------------------------------

    fn insert_blocks(&self, _blocks: &[BlockRecord]) -> Result<()> {
        Err(read_only_error())
    }

    // -----------------------------------------------------------------------
    // Links (read)
    // -----------------------------------------------------------------------

    fn get_links_from(&self, doc_id: &Uuid) -> Result<Vec<LinkRecord>> {
        Ok(self
            .links_by_source
            .get(doc_id)
            .cloned()
            .unwrap_or_default())
    }

    fn get_links_to(&self, target_path: &str) -> Result<Vec<LinkRecord>> {
        Ok(self
            .links_by_target
            .get(target_path)
            .cloned()
            .unwrap_or_default())
    }

    // -----------------------------------------------------------------------
    // Links (write — rejected)
    // -----------------------------------------------------------------------

    fn insert_links(&self, _links: &[LinkRecord]) -> Result<()> {
        Err(read_only_error())
    }

    // -----------------------------------------------------------------------
    // Tags (read)
    // -----------------------------------------------------------------------

    fn get_tags_by_doc(&self, doc_id: &Uuid) -> Result<Vec<TagRecord>> {
        Ok(self
            .tags_by_doc
            .get(doc_id)
            .cloned()
            .unwrap_or_default())
    }

    // -----------------------------------------------------------------------
    // Tags (write — rejected)
    // -----------------------------------------------------------------------

    fn insert_tags(&self, _tags: &[TagRecord]) -> Result<()> {
        Err(read_only_error())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use mdbase_core::types::LinkType;

    use crate::engine::storage_engine::CustomStorageEngine;
    use crate::Storage;

    /// Create a temporary storage engine for testing.
    fn temp_engine() -> (CustomStorageEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let engine = CustomStorageEngine::new(dir.path()).unwrap();
        (engine, dir)
    }

    fn make_document(path: &str) -> DocumentRecord {
        DocumentRecord {
            doc_id: Uuid::new_v4(),
            path: path.to_string(),
            title: Some(format!("Title for {path}")),
            frontmatter: serde_json::json!({}),
            raw_content_hash: "abc123".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
        }
    }

    fn make_block(doc_id: Uuid, ordinal: u32, bt: BlockType) -> BlockRecord {
        BlockRecord {
            block_id: Uuid::new_v4(),
            doc_id,
            block_type: bt,
            depth: 0,
            ordinal,
            parent_block_id: None,
            text_content: format!("block {ordinal}"),
            raw_markdown: format!("block {ordinal}"),
        }
    }

    fn make_link(source_doc_id: Uuid, target: &str) -> LinkRecord {
        LinkRecord {
            link_id: Uuid::new_v4(),
            source_doc_id,
            source_block_id: None,
            target: target.to_string(),
            target_doc_id: None,
            link_type: LinkType::WikiLink,
            anchor_text: format!("link to {target}"),
        }
    }

    fn make_tag(doc_id: Uuid, key: &str, value: serde_json::Value) -> TagRecord {
        TagRecord {
            tag_id: Uuid::new_v4(),
            doc_id,
            key: key.to_string(),
            value,
        }
    }

    // 1. Snapshot isolation: writes after snapshot do not affect snapshot.
    #[test]
    fn snapshot_isolation_from_later_writes() {
        let (engine, _dir) = temp_engine();

        let doc = make_document("snap.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();

        let blocks = vec![make_block(doc_id, 0, BlockType::Paragraph)];
        engine.insert_blocks(&blocks).unwrap();

        let links = vec![make_link(doc_id, "target.md")];
        engine.insert_links(&links).unwrap();

        let tags = vec![make_tag(doc_id, "status", serde_json::json!("draft"))];
        engine.insert_tags(&tags).unwrap();

        // Take a snapshot.
        let snap = engine.snapshot().unwrap();

        // Mutate the engine after the snapshot.
        let doc2 = make_document("new.md");
        engine.insert_document(&doc2).unwrap();
        engine
            .insert_blocks(&[make_block(doc2.doc_id, 0, BlockType::Paragraph)])
            .unwrap();
        engine
            .insert_links(&[make_link(doc2.doc_id, "other.md")])
            .unwrap();
        engine
            .insert_tags(&[make_tag(doc2.doc_id, "key", serde_json::json!("val"))])
            .unwrap();

        // Snapshot should NOT see the new document.
        assert_eq!(snap.list_documents().unwrap().len(), 1);
        assert!(snap.get_document(&doc_id).unwrap().is_some());
        assert!(snap.get_document(&doc2.doc_id).unwrap().is_none());

        // Snapshot should still see old blocks/links/tags.
        assert_eq!(snap.get_blocks_by_doc(&doc_id).unwrap().len(), 1);
        assert_eq!(snap.get_links_from(&doc_id).unwrap().len(), 1);
        assert_eq!(snap.get_links_to("target.md").unwrap().len(), 1);
        assert_eq!(snap.get_tags_by_doc(&doc_id).unwrap().len(), 1);

        // Snapshot should NOT see blocks/links/tags for the new doc.
        assert!(snap.get_blocks_by_doc(&doc2.doc_id).unwrap().is_empty());
        assert!(snap.get_links_from(&doc2.doc_id).unwrap().is_empty());
        assert!(snap.get_tags_by_doc(&doc2.doc_id).unwrap().is_empty());

        // Engine version should have advanced beyond the snapshot.
        assert!(engine.version() > snap.version());
    }

    // 2. Snapshot read methods return correct data.
    #[test]
    fn snapshot_read_methods_return_correct_data() {
        let (engine, _dir) = temp_engine();

        let doc = make_document("read.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::CodeBlock {
                language: Some("rust".to_string()),
            }),
            make_block(doc_id, 2, BlockType::Paragraph),
        ];
        engine.insert_blocks(&blocks).unwrap();

        let links = vec![
            make_link(doc_id, "a.md"),
            make_link(doc_id, "b.md"),
        ];
        engine.insert_links(&links).unwrap();

        let tags = vec![
            make_tag(doc_id, "status", serde_json::json!("published")),
            make_tag(doc_id, "priority", serde_json::json!(1)),
        ];
        engine.insert_tags(&tags).unwrap();

        let snap = engine.snapshot().unwrap();

        // get_document
        let fetched = snap.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(fetched.path, "read.md");

        // get_document_by_path
        let by_path = snap.get_document_by_path("read.md").unwrap().unwrap();
        assert_eq!(by_path.doc_id, doc_id);
        assert!(snap.get_document_by_path("nonexistent.md").unwrap().is_none());

        // list_documents
        assert_eq!(snap.list_documents().unwrap().len(), 1);

        // get_blocks_by_doc
        assert_eq!(snap.get_blocks_by_doc(&doc_id).unwrap().len(), 3);

        // query_blocks — by type
        let headings = snap
            .query_blocks(Some(&doc_id), Some("heading"), None, None)
            .unwrap();
        assert_eq!(headings.len(), 1);

        // query_blocks — by language
        let rust_blocks = snap
            .query_blocks(Some(&doc_id), Some("code_block"), None, Some("rust"))
            .unwrap();
        assert_eq!(rust_blocks.len(), 1);

        // query_blocks — no filters
        let all = snap.query_blocks(None, None, None, None).unwrap();
        assert_eq!(all.len(), 3);

        // get_links_from
        assert_eq!(snap.get_links_from(&doc_id).unwrap().len(), 2);

        // get_links_to
        assert_eq!(snap.get_links_to("a.md").unwrap().len(), 1);
        assert_eq!(snap.get_links_to("b.md").unwrap().len(), 1);
        assert!(snap.get_links_to("nonexistent.md").unwrap().is_empty());

        // get_tags_by_doc
        let fetched_tags = snap.get_tags_by_doc(&doc_id).unwrap();
        assert_eq!(fetched_tags.len(), 2);
        let keys: Vec<&str> = fetched_tags.iter().map(|t| t.key.as_str()).collect();
        assert!(keys.contains(&"status"));
        assert!(keys.contains(&"priority"));
    }

    // 3. Snapshot write methods return read-only error.
    #[test]
    fn snapshot_write_methods_return_error() {
        let (engine, _dir) = temp_engine();
        let snap = engine.snapshot().unwrap();

        let doc = make_document("fail.md");
        assert!(snap.insert_document(&doc).is_err());
        assert!(snap.delete_document(&doc.doc_id).is_err());
        assert!(snap.insert_blocks(&[]).is_err());
        assert!(snap.insert_links(&[]).is_err());
        assert!(snap.insert_tags(&[]).is_err());

        // Verify the error message.
        let err = snap.insert_document(&doc).unwrap_err();
        assert!(
            err.to_string().contains("read-only"),
            "error should mention read-only, got: {err}"
        );
    }

    // 4. Multiple snapshots are independent.
    #[test]
    fn multiple_snapshots_are_independent() {
        let (engine, _dir) = temp_engine();

        // Snapshot 1: empty.
        let snap1 = engine.snapshot().unwrap();

        // Insert a document.
        let doc = make_document("multi.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();

        // Snapshot 2: one document.
        let snap2 = engine.snapshot().unwrap();

        // Insert another document.
        let doc2 = make_document("multi2.md");
        engine.insert_document(&doc2).unwrap();

        // Snapshot 3: two documents.
        let snap3 = engine.snapshot().unwrap();

        assert_eq!(snap1.list_documents().unwrap().len(), 0);
        assert_eq!(snap2.list_documents().unwrap().len(), 1);
        assert_eq!(snap3.list_documents().unwrap().len(), 2);

        assert!(snap1.get_document(&doc_id).unwrap().is_none());
        assert!(snap2.get_document(&doc_id).unwrap().is_some());
        assert!(snap3.get_document(&doc_id).unwrap().is_some());

        // Version ordering.
        assert!(snap1.version() < snap2.version());
        assert!(snap2.version() < snap3.version());
    }

    // 5. Empty engine snapshot works.
    #[test]
    fn empty_engine_snapshot() {
        let (engine, _dir) = temp_engine();
        let snap = engine.snapshot().unwrap();

        assert_eq!(snap.version(), 0);
        assert!(snap.list_documents().unwrap().is_empty());
        assert!(snap.get_document(&Uuid::new_v4()).unwrap().is_none());
        assert!(snap.get_document_by_path("any.md").unwrap().is_none());
        assert!(snap.get_blocks_by_doc(&Uuid::new_v4()).unwrap().is_empty());
        assert!(snap.query_blocks(None, None, None, None).unwrap().is_empty());
        assert!(snap.get_links_from(&Uuid::new_v4()).unwrap().is_empty());
        assert!(snap.get_links_to("any.md").unwrap().is_empty());
        assert!(snap.get_tags_by_doc(&Uuid::new_v4()).unwrap().is_empty());
    }

    // 6. Snapshot survives engine deletion of data it references.
    #[test]
    fn snapshot_survives_engine_deletion() {
        let (engine, _dir) = temp_engine();

        let doc = make_document("doomed.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();
        engine
            .insert_blocks(&[make_block(doc_id, 0, BlockType::Paragraph)])
            .unwrap();
        engine
            .insert_links(&[make_link(doc_id, "target.md")])
            .unwrap();
        engine
            .insert_tags(&[make_tag(doc_id, "k", serde_json::json!("v"))])
            .unwrap();

        let snap = engine.snapshot().unwrap();

        // Delete everything from the engine.
        engine.delete_document(&doc_id).unwrap();

        // Engine sees nothing.
        assert!(engine.get_document(&doc_id).unwrap().is_none());
        assert!(engine.list_documents().unwrap().is_empty());

        // Snapshot still sees everything.
        assert!(snap.get_document(&doc_id).unwrap().is_some());
        assert_eq!(snap.list_documents().unwrap().len(), 1);
        assert_eq!(snap.get_blocks_by_doc(&doc_id).unwrap().len(), 1);
        assert_eq!(snap.get_links_from(&doc_id).unwrap().len(), 1);
        assert_eq!(snap.get_links_to("target.md").unwrap().len(), 1);
        assert_eq!(snap.get_tags_by_doc(&doc_id).unwrap().len(), 1);
    }
}
