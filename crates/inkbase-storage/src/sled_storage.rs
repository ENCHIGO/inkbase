use std::path::Path;

use inkbase_core::error::InkbaseError;
use inkbase_core::types::{BlockRecord, BlockType, DocumentRecord, LinkRecord, TagRecord};
use sled::{Db, Tree};
use tracing::debug;
use uuid::Uuid;

use crate::traits::Storage;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map any sled error into our domain error type.
fn sled_err(err: sled::Error) -> InkbaseError {
    InkbaseError::StorageError(err.to_string())
}

/// Build a composite key of the form `{parent_uuid}:{child_uuid}`.
///
/// Using the hyphenated UUID string representation keeps keys human-readable
/// in debug tools while still being unique and prefix-scannable.
fn composite_key(parent: &Uuid, child: &Uuid) -> String {
    format!("{}:{}", parent, child)
}

/// Return the prefix used to scan all children of a given parent.
fn prefix_for(parent: &Uuid) -> String {
    format!("{}:", parent)
}

// ---------------------------------------------------------------------------
// SledStorage
// ---------------------------------------------------------------------------

/// A sled-backed implementation of the [`Storage`] trait.
///
/// Each logical index lives in its own sled `Tree` so key spaces never
/// collide and prefix scans stay efficient.
pub struct SledStorage {
    _db: Db,

    // Primary stores — keyed by the record's own id.
    documents: Tree,
    blocks: Tree,
    links: Tree,
    tags: Tree,

    // Secondary indexes — different key schemes for efficient lookups.
    documents_by_path: Tree,  // path -> doc_id bytes
    blocks_by_doc: Tree,      // "{doc_id}:{block_id}" -> block JSON
    links_by_source: Tree,    // "{source_doc_id}:{link_id}" -> link JSON
    links_by_target: Tree,    // "{target_path}:{link_id}" -> link JSON
    tags_by_doc: Tree,        // "{doc_id}:{tag_id}" -> tag JSON
}

impl SledStorage {
    /// Open (or create) a sled database at `path`.
    pub fn new(path: &Path) -> inkbase_core::Result<Self> {
        let db = sled::open(path).map_err(sled_err)?;

        let documents = db.open_tree("documents").map_err(sled_err)?;
        let blocks = db.open_tree("blocks").map_err(sled_err)?;
        let links = db.open_tree("links").map_err(sled_err)?;
        let tags = db.open_tree("tags").map_err(sled_err)?;

        let documents_by_path = db.open_tree("documents_by_path").map_err(sled_err)?;
        let blocks_by_doc = db.open_tree("blocks_by_doc").map_err(sled_err)?;
        let links_by_source = db.open_tree("links_by_source").map_err(sled_err)?;
        let links_by_target = db.open_tree("links_by_target").map_err(sled_err)?;
        let tags_by_doc = db.open_tree("tags_by_doc").map_err(sled_err)?;

        debug!("SledStorage opened at {}", path.display());

        Ok(Self {
            _db: db,
            documents,
            blocks,
            links,
            tags,
            documents_by_path,
            blocks_by_doc,
            links_by_source,
            links_by_target,
            tags_by_doc,
        })
    }

    // -- internal helpers ----------------------------------------------------

    /// Serialize a value to JSON bytes for storage.
    fn serialize<T: serde::Serialize>(value: &T) -> inkbase_core::Result<Vec<u8>> {
        serde_json::to_vec(value).map_err(|e| InkbaseError::SerializationError(e.to_string()))
    }

    /// Deserialize JSON bytes back into a typed value.
    fn deserialize<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> inkbase_core::Result<T> {
        serde_json::from_slice(bytes).map_err(|e| InkbaseError::SerializationError(e.to_string()))
    }

    /// Collect all values whose key starts with `prefix` from `tree`,
    /// deserializing each into `T`.
    fn scan_prefix<T: serde::de::DeserializeOwned>(
        tree: &Tree,
        prefix: &str,
    ) -> inkbase_core::Result<Vec<T>> {
        let mut results = Vec::new();
        for entry in tree.scan_prefix(prefix.as_bytes()) {
            let (_key, value) = entry.map_err(sled_err)?;
            results.push(Self::deserialize(&value)?);
        }
        Ok(results)
    }

}

impl Storage for SledStorage {
    // -----------------------------------------------------------------------
    // Documents
    // -----------------------------------------------------------------------

    fn insert_document(&self, doc: &DocumentRecord) -> inkbase_core::Result<()> {
        let id_key = doc.doc_id.to_string();
        let data = Self::serialize(doc)?;

        // If this doc_id already exists, clean up the old path index entry
        // (the path may have changed).
        if let Some(old_bytes) = self.documents.get(id_key.as_bytes()).map_err(sled_err)? {
            let old_doc: DocumentRecord = Self::deserialize(&old_bytes)?;
            if old_doc.path != doc.path {
                self.documents_by_path
                    .remove(old_doc.path.as_bytes())
                    .map_err(sled_err)?;
            }
        }

        self.documents
            .insert(id_key.as_bytes(), data)
            .map_err(sled_err)?;

        // Secondary index: path -> doc_id.
        self.documents_by_path
            .insert(doc.path.as_bytes(), doc.doc_id.as_bytes().as_slice())
            .map_err(sled_err)?;

        debug!(doc_id = %doc.doc_id, path = %doc.path, "inserted document");
        Ok(())
    }

    fn get_document(&self, doc_id: &Uuid) -> inkbase_core::Result<Option<DocumentRecord>> {
        let key = doc_id.to_string();
        match self.documents.get(key.as_bytes()).map_err(sled_err)? {
            Some(bytes) => Ok(Some(Self::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    fn get_document_by_path(&self, path: &str) -> inkbase_core::Result<Option<DocumentRecord>> {
        let id_bytes = match self
            .documents_by_path
            .get(path.as_bytes())
            .map_err(sled_err)?
        {
            Some(b) => b,
            None => return Ok(None),
        };

        let doc_id = Uuid::from_slice(&id_bytes)
            .map_err(|e| InkbaseError::StorageError(format!("corrupt doc_id in path index: {e}")))?;

        self.get_document(&doc_id)
    }

    fn list_documents(&self) -> inkbase_core::Result<Vec<DocumentRecord>> {
        let mut docs = Vec::new();
        for entry in self.documents.iter() {
            let (_key, value) = entry.map_err(sled_err)?;
            docs.push(Self::deserialize(&value)?);
        }
        debug!(count = docs.len(), "listed documents");
        Ok(docs)
    }

    fn delete_document(&self, doc_id: &Uuid) -> inkbase_core::Result<()> {
        let id_key = doc_id.to_string();
        let prefix = prefix_for(doc_id);

        // Remove path index entry before deleting the document itself.
        if let Some(doc_bytes) = self.documents.get(id_key.as_bytes()).map_err(sled_err)? {
            let doc: DocumentRecord = Self::deserialize(&doc_bytes)?;
            self.documents_by_path
                .remove(doc.path.as_bytes())
                .map_err(sled_err)?;
        }

        // Cascade: collect child ids from the secondary indexes, then remove
        // from both primary and secondary stores.

        // Blocks
        let block_keys: Vec<(String, sled::IVec)> = self
            .blocks_by_doc
            .scan_prefix(prefix.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .map_err(sled_err)?
            .into_iter()
            .map(|(k, v)| (String::from_utf8_lossy(&k).to_string(), v))
            .collect();
        for (comp_key, _val) in &block_keys {
            // The child id is the portion after the ':'.
            if let Some(block_id_str) = comp_key.split(':').nth(1) {
                self.blocks
                    .remove(block_id_str.as_bytes())
                    .map_err(sled_err)?;
            }
            self.blocks_by_doc
                .remove(comp_key.as_bytes())
                .map_err(sled_err)?;
        }

        // Links (by source)
        let link_entries: Vec<(String, Vec<u8>)> = self
            .links_by_source
            .scan_prefix(prefix.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .map_err(sled_err)?
            .into_iter()
            .map(|(k, v)| (String::from_utf8_lossy(&k).to_string(), v.to_vec()))
            .collect();
        for (comp_key, val) in &link_entries {
            if let Some(link_id_str) = comp_key.split(':').nth(1) {
                self.links
                    .remove(link_id_str.as_bytes())
                    .map_err(sled_err)?;
            }
            self.links_by_source
                .remove(comp_key.as_bytes())
                .map_err(sled_err)?;

            // Also clean the links_by_target index for this link.
            let link: LinkRecord = Self::deserialize(val)?;
            let target_key = format!("{}:{}", link.target, link.link_id);
            self.links_by_target
                .remove(target_key.as_bytes())
                .map_err(sled_err)?;
        }

        // Tags
        let tag_keys: Vec<String> = self
            .tags_by_doc
            .scan_prefix(prefix.as_bytes())
            .keys()
            .collect::<Result<Vec<_>, _>>()
            .map_err(sled_err)?
            .into_iter()
            .map(|k| String::from_utf8_lossy(&k).to_string())
            .collect();
        for comp_key in &tag_keys {
            if let Some(tag_id_str) = comp_key.split(':').nth(1) {
                self.tags
                    .remove(tag_id_str.as_bytes())
                    .map_err(sled_err)?;
            }
            self.tags_by_doc
                .remove(comp_key.as_bytes())
                .map_err(sled_err)?;
        }

        // Finally remove the document record itself.
        self.documents
            .remove(id_key.as_bytes())
            .map_err(sled_err)?;

        debug!(%doc_id, "deleted document with cascade");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Blocks
    // -----------------------------------------------------------------------

    fn insert_blocks(&self, blocks: &[BlockRecord]) -> inkbase_core::Result<()> {
        for block in blocks {
            let data = Self::serialize(block)?;
            let id_key = block.block_id.to_string();

            self.blocks
                .insert(id_key.as_bytes(), data.clone())
                .map_err(sled_err)?;

            let comp = composite_key(&block.doc_id, &block.block_id);
            self.blocks_by_doc
                .insert(comp.as_bytes(), data)
                .map_err(sled_err)?;
        }
        debug!(count = blocks.len(), "inserted blocks");
        Ok(())
    }

    fn get_blocks_by_doc(&self, doc_id: &Uuid) -> inkbase_core::Result<Vec<BlockRecord>> {
        let prefix = prefix_for(doc_id);
        Self::scan_prefix(&self.blocks_by_doc, &prefix)
    }

    fn query_blocks(
        &self,
        doc_id: Option<&Uuid>,
        block_type: Option<&str>,
        heading_level: Option<u8>,
        language: Option<&str>,
    ) -> inkbase_core::Result<Vec<BlockRecord>> {
        // When a doc_id filter is present we can narrow the scan to a single
        // prefix; otherwise we must iterate all blocks.
        let candidates: Vec<BlockRecord> = if let Some(id) = doc_id {
            let prefix = prefix_for(id);
            Self::scan_prefix(&self.blocks_by_doc, &prefix)?
        } else {
            let mut all = Vec::new();
            for entry in self.blocks.iter() {
                let (_key, value) = entry.map_err(sled_err)?;
                all.push(Self::deserialize(&value)?);
            }
            all
        };

        let results = candidates
            .into_iter()
            .filter(|block| {
                if let Some(bt) = block_type {
                    if !block_type_matches(&block.block_type, bt) {
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
            .collect();

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Links
    // -----------------------------------------------------------------------

    fn insert_links(&self, links: &[LinkRecord]) -> inkbase_core::Result<()> {
        for link in links {
            let data = Self::serialize(link)?;
            let id_key = link.link_id.to_string();

            self.links
                .insert(id_key.as_bytes(), data.clone())
                .map_err(sled_err)?;

            // Secondary: source_doc_id -> link
            let source_comp = composite_key(&link.source_doc_id, &link.link_id);
            self.links_by_source
                .insert(source_comp.as_bytes(), data.clone())
                .map_err(sled_err)?;

            // Secondary: target path -> link
            let target_key = format!("{}:{}", link.target, link.link_id);
            self.links_by_target
                .insert(target_key.as_bytes(), data)
                .map_err(sled_err)?;
        }
        debug!(count = links.len(), "inserted links");
        Ok(())
    }

    fn get_links_from(&self, doc_id: &Uuid) -> inkbase_core::Result<Vec<LinkRecord>> {
        let prefix = prefix_for(doc_id);
        Self::scan_prefix(&self.links_by_source, &prefix)
    }

    fn get_links_to(&self, target_path: &str) -> inkbase_core::Result<Vec<LinkRecord>> {
        let prefix = format!("{target_path}:");
        Self::scan_prefix(&self.links_by_target, &prefix)
    }

    // -----------------------------------------------------------------------
    // Tags
    // -----------------------------------------------------------------------

    fn insert_tags(&self, tags: &[TagRecord]) -> inkbase_core::Result<()> {
        for tag in tags {
            let data = Self::serialize(tag)?;
            let id_key = tag.tag_id.to_string();

            self.tags
                .insert(id_key.as_bytes(), data.clone())
                .map_err(sled_err)?;

            let comp = composite_key(&tag.doc_id, &tag.tag_id);
            self.tags_by_doc
                .insert(comp.as_bytes(), data)
                .map_err(sled_err)?;
        }
        debug!(count = tags.len(), "inserted tags");
        Ok(())
    }

    fn get_tags_by_doc(&self, doc_id: &Uuid) -> inkbase_core::Result<Vec<TagRecord>> {
        let prefix = prefix_for(doc_id);
        Self::scan_prefix(&self.tags_by_doc, &prefix)
    }
}

// ---------------------------------------------------------------------------
// BlockType matching
// ---------------------------------------------------------------------------

/// Match a `BlockType` variant against a user-supplied string name.
///
/// The comparison is case-insensitive and accepts both snake_case and the
/// natural variant names: `"heading"`, `"paragraph"`, `"code_block"`, etc.
fn block_type_matches(bt: &BlockType, name: &str) -> bool {
    let name_lower = name.to_ascii_lowercase();
    match bt {
        BlockType::Heading { .. } => name_lower == "heading",
        BlockType::Paragraph => name_lower == "paragraph",
        BlockType::CodeBlock { .. } => name_lower == "code_block" || name_lower == "codeblock",
        BlockType::List { .. } => name_lower == "list",
        BlockType::ListItem => name_lower == "list_item" || name_lower == "listitem",
        BlockType::Table => name_lower == "table",
        BlockType::BlockQuote => name_lower == "block_quote" || name_lower == "blockquote",
        BlockType::ThematicBreak => name_lower == "thematic_break" || name_lower == "thematicbreak",
        BlockType::Image { .. } => name_lower == "image",
        BlockType::Html => name_lower == "html",
        BlockType::FootnoteDefinition => {
            name_lower == "footnote_definition" || name_lower == "footnotedefinition"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use inkbase_core::types::{BlockType, LinkType};
    use uuid::Uuid;

    /// Create a temporary sled database for testing.
    fn temp_storage() -> SledStorage {
        let dir = tempfile::tempdir().unwrap();
        SledStorage::new(dir.path()).unwrap()
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

    #[test]
    fn document_roundtrip() {
        let storage = temp_storage();
        let doc = make_document("notes/test.md");
        let doc_id = doc.doc_id;

        storage.insert_document(&doc).unwrap();

        let fetched = storage.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(fetched.doc_id, doc_id);
        assert_eq!(fetched.path, "notes/test.md");
    }

    #[test]
    fn document_lookup_by_path() {
        let storage = temp_storage();
        let doc = make_document("notes/lookup.md");
        let doc_id = doc.doc_id;

        storage.insert_document(&doc).unwrap();

        let fetched = storage
            .get_document_by_path("notes/lookup.md")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.doc_id, doc_id);

        assert!(storage
            .get_document_by_path("nonexistent.md")
            .unwrap()
            .is_none());
    }

    #[test]
    fn document_path_update() {
        let storage = temp_storage();
        let mut doc = make_document("old/path.md");

        storage.insert_document(&doc).unwrap();

        // Update the path and re-insert.
        doc.path = "new/path.md".to_string();
        storage.insert_document(&doc).unwrap();

        // Old path should no longer resolve.
        assert!(storage
            .get_document_by_path("old/path.md")
            .unwrap()
            .is_none());

        // New path should work.
        assert!(storage
            .get_document_by_path("new/path.md")
            .unwrap()
            .is_some());
    }

    #[test]
    fn list_documents() {
        let storage = temp_storage();

        storage.insert_document(&make_document("a.md")).unwrap();
        storage.insert_document(&make_document("b.md")).unwrap();
        storage.insert_document(&make_document("c.md")).unwrap();

        let docs = storage.list_documents().unwrap();
        assert_eq!(docs.len(), 3);
    }

    #[test]
    fn delete_document_cascades() {
        let storage = temp_storage();
        let doc = make_document("cascade.md");
        let doc_id = doc.doc_id;

        storage.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::Paragraph),
        ];
        storage.insert_blocks(&blocks).unwrap();

        let links = vec![LinkRecord {
            link_id: Uuid::new_v4(),
            source_doc_id: doc_id,
            source_block_id: None,
            target: "other.md".to_string(),
            target_doc_id: None,
            link_type: LinkType::WikiLink,
            anchor_text: "other".to_string(),
        }];
        storage.insert_links(&links).unwrap();

        let tags = vec![TagRecord {
            tag_id: Uuid::new_v4(),
            doc_id,
            key: "status".to_string(),
            value: serde_json::json!("draft"),
        }];
        storage.insert_tags(&tags).unwrap();

        // Delete should remove everything.
        storage.delete_document(&doc_id).unwrap();

        assert!(storage.get_document(&doc_id).unwrap().is_none());
        assert!(storage
            .get_document_by_path("cascade.md")
            .unwrap()
            .is_none());
        assert!(storage.get_blocks_by_doc(&doc_id).unwrap().is_empty());
        assert!(storage.get_links_from(&doc_id).unwrap().is_empty());
        assert!(storage.get_links_to("other.md").unwrap().is_empty());
        assert!(storage.get_tags_by_doc(&doc_id).unwrap().is_empty());
    }

    #[test]
    fn blocks_by_doc() {
        let storage = temp_storage();
        let doc = make_document("blocks.md");
        let doc_id = doc.doc_id;
        storage.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::Paragraph),
            make_block(doc_id, 2, BlockType::CodeBlock { language: Some("rust".to_string()) }),
        ];
        storage.insert_blocks(&blocks).unwrap();

        let fetched = storage.get_blocks_by_doc(&doc_id).unwrap();
        assert_eq!(fetched.len(), 3);
    }

    #[test]
    fn query_blocks_by_type() {
        let storage = temp_storage();
        let doc = make_document("query.md");
        let doc_id = doc.doc_id;
        storage.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::Heading { level: 2 }),
            make_block(doc_id, 2, BlockType::Paragraph),
            make_block(doc_id, 3, BlockType::CodeBlock { language: Some("rust".to_string()) }),
            make_block(doc_id, 4, BlockType::CodeBlock { language: Some("python".to_string()) }),
        ];
        storage.insert_blocks(&blocks).unwrap();

        // All headings.
        let headings = storage
            .query_blocks(Some(&doc_id), Some("heading"), None, None)
            .unwrap();
        assert_eq!(headings.len(), 2);

        // Only H2 headings.
        let h2 = storage
            .query_blocks(Some(&doc_id), Some("heading"), Some(2), None)
            .unwrap();
        assert_eq!(h2.len(), 1);

        // Rust code blocks.
        let rust = storage
            .query_blocks(Some(&doc_id), Some("code_block"), None, Some("rust"))
            .unwrap();
        assert_eq!(rust.len(), 1);

        // All paragraphs.
        let paragraphs = storage
            .query_blocks(Some(&doc_id), Some("paragraph"), None, None)
            .unwrap();
        assert_eq!(paragraphs.len(), 1);
    }

    #[test]
    fn links_from_and_to() {
        let storage = temp_storage();
        let doc_a = make_document("a.md");
        let doc_b = make_document("b.md");
        storage.insert_document(&doc_a).unwrap();
        storage.insert_document(&doc_b).unwrap();

        let links = vec![
            LinkRecord {
                link_id: Uuid::new_v4(),
                source_doc_id: doc_a.doc_id,
                source_block_id: None,
                target: "b.md".to_string(),
                target_doc_id: Some(doc_b.doc_id),
                link_type: LinkType::WikiLink,
                anchor_text: "link to b".to_string(),
            },
            LinkRecord {
                link_id: Uuid::new_v4(),
                source_doc_id: doc_a.doc_id,
                source_block_id: None,
                target: "external.com".to_string(),
                target_doc_id: None,
                link_type: LinkType::AutoLink,
                anchor_text: "external".to_string(),
            },
        ];
        storage.insert_links(&links).unwrap();

        let from_a = storage.get_links_from(&doc_a.doc_id).unwrap();
        assert_eq!(from_a.len(), 2);

        let to_b = storage.get_links_to("b.md").unwrap();
        assert_eq!(to_b.len(), 1);
        assert_eq!(to_b[0].source_doc_id, doc_a.doc_id);

        // No links point to a.md.
        let to_a = storage.get_links_to("a.md").unwrap();
        assert!(to_a.is_empty());
    }

    #[test]
    fn tags_by_doc() {
        let storage = temp_storage();
        let doc = make_document("tags.md");
        let doc_id = doc.doc_id;
        storage.insert_document(&doc).unwrap();

        let tags = vec![
            TagRecord {
                tag_id: Uuid::new_v4(),
                doc_id,
                key: "category".to_string(),
                value: serde_json::json!("notes"),
            },
            TagRecord {
                tag_id: Uuid::new_v4(),
                doc_id,
                key: "priority".to_string(),
                value: serde_json::json!(1),
            },
        ];
        storage.insert_tags(&tags).unwrap();

        let fetched = storage.get_tags_by_doc(&doc_id).unwrap();
        assert_eq!(fetched.len(), 2);
    }

    #[test]
    fn get_nonexistent_document_returns_none() {
        let storage = temp_storage();
        let result = storage.get_document(&Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn delete_nonexistent_document_is_noop() {
        let storage = temp_storage();
        // Should not error.
        storage.delete_document(&Uuid::new_v4()).unwrap();
    }
}
