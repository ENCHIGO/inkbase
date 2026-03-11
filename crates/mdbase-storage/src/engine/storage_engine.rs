//! Custom storage engine implementing the [`Storage`] trait using B+Trees.
//!
//! This replaces the sled backend with our own page-based storage engine.
//! Nine B+Trees share a single [`BufferPool`], providing primary stores and
//! secondary indexes for documents, blocks, links, and tags.
//!
//! ## Tree Layout
//!
//! | Tree              | Key                                    | Value               |
//! |-------------------|----------------------------------------|----------------------|
//! | `docs`            | `doc_id` (16 bytes)                    | JSON DocumentRecord  |
//! | `docs_by_path`    | `path` bytes                           | `doc_id` (16 bytes)  |
//! | `blocks`          | `block_id` (16 bytes)                  | JSON BlockRecord     |
//! | `blocks_by_doc`   | `doc_id(16) ++ block_id(16)`           | `[]` (empty)         |
//! | `links`           | `link_id` (16 bytes)                   | JSON LinkRecord      |
//! | `links_by_source` | `source_doc_id(16) ++ link_id(16)`     | `[]` (empty)         |
//! | `links_by_target` | `target_bytes ++ \0 ++ link_id(16)`    | `[]` (empty)         |
//! | `tags`            | `tag_id` (16 bytes)                    | JSON TagRecord       |
//! | `tags_by_doc`     | `doc_id(16) ++ tag_id(16)`             | `[]` (empty)         |

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use uuid::Uuid;

use mdbase_core::types::{BlockRecord, BlockType, DocumentRecord, LinkRecord, TagRecord};
use mdbase_core::{MdbaseError, Result};

use super::btree::BTree;
use super::buffer_pool::BufferPool;
use super::mvcc::StorageSnapshot;
use crate::Storage;

// ---------------------------------------------------------------------------
// Metadata persistence
// ---------------------------------------------------------------------------

/// Names for the nine B+Trees stored in the metadata file.
const TREE_NAMES: &[&str] = &[
    "docs",
    "docs_by_path",
    "blocks",
    "blocks_by_doc",
    "links",
    "links_by_source",
    "links_by_target",
    "tags",
    "tags_by_doc",
];

/// On-disk representation of tree metadata: maps tree name to root page ID.
type TreeMeta = HashMap<String, u32>;

fn meta_path(dir: &Path) -> PathBuf {
    dir.join("meta.json")
}

fn load_meta(dir: &Path) -> Result<Option<TreeMeta>> {
    let path = meta_path(dir);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    let meta: TreeMeta = serde_json::from_slice(&bytes)?;
    Ok(Some(meta))
}

fn save_meta(dir: &Path, meta: &TreeMeta) -> Result<()> {
    let path = meta_path(dir);
    let bytes = serde_json::to_vec_pretty(meta)?;
    fs::write(&path, bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn serialize<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| MdbaseError::SerializationError(e.to_string()))
}

fn deserialize<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| MdbaseError::SerializationError(e.to_string()))
}

// ---------------------------------------------------------------------------
// Key construction helpers
// ---------------------------------------------------------------------------

/// 32-byte composite key: `parent_uuid(16) ++ child_uuid(16)`.
fn composite_key(parent: &Uuid, child: &Uuid) -> [u8; 32] {
    let mut key = [0u8; 32];
    key[..16].copy_from_slice(parent.as_bytes());
    key[16..].copy_from_slice(child.as_bytes());
    key
}

/// Build the `links_by_target` key: `target_bytes ++ \0 ++ link_id(16)`.
fn target_index_key(target: &str, link_id: &Uuid) -> Vec<u8> {
    let target_bytes = target.as_bytes();
    let mut key = Vec::with_capacity(target_bytes.len() + 1 + 16);
    key.extend_from_slice(target_bytes);
    key.push(0);
    key.extend_from_slice(link_id.as_bytes());
    key
}

/// Build the prefix for scanning `links_by_target` by target path:
/// `target_bytes ++ \0`.
fn target_index_prefix(target: &str) -> Vec<u8> {
    let target_bytes = target.as_bytes();
    let mut prefix = Vec::with_capacity(target_bytes.len() + 1);
    prefix.extend_from_slice(target_bytes);
    prefix.push(0);
    prefix
}

// ---------------------------------------------------------------------------
// Block type matching
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

// ---------------------------------------------------------------------------
// CustomStorageEngine
// ---------------------------------------------------------------------------

/// Buffer pool capacity (pages held in memory).
const POOL_CAPACITY: usize = 2048; // 16 MB at 8 KB/page

/// Custom storage engine backed by nine B+Trees sharing one [`BufferPool`].
///
/// Implements the [`Storage`] trait and is designed as a drop-in replacement
/// for the sled-backed storage.
pub struct CustomStorageEngine {
    pool: Arc<BufferPool>,
    data_dir: PathBuf,

    /// Monotonically increasing version counter, bumped on each write.
    version: AtomicU64,

    // Primary stores
    docs: RwLock<BTree>,
    blocks: RwLock<BTree>,
    links: RwLock<BTree>,
    tags: RwLock<BTree>,

    // Secondary indexes
    docs_by_path: RwLock<BTree>,
    blocks_by_doc: RwLock<BTree>,
    links_by_source: RwLock<BTree>,
    links_by_target: RwLock<BTree>,
    tags_by_doc: RwLock<BTree>,
}

impl CustomStorageEngine {
    /// Open or create a custom storage engine rooted at `data_dir`.
    ///
    /// The engine stores its data file at `{data_dir}/custom_engine/custom.db`
    /// and metadata at `{data_dir}/custom_engine/meta.json`.
    pub fn new(data_dir: &Path) -> Result<Self> {
        let engine_dir = data_dir.join("custom_engine");
        fs::create_dir_all(&engine_dir)?;

        let db_path = engine_dir.join("custom.db");
        let pool = Arc::new(BufferPool::open(&db_path, POOL_CAPACITY)?);

        let existing_meta = load_meta(&engine_dir)?;

        let trees = if let Some(meta) = existing_meta {
            let mut trees = HashMap::new();
            for &name in TREE_NAMES {
                let root_id = meta.get(name).copied().ok_or_else(|| {
                    MdbaseError::StorageError(format!(
                        "missing tree '{}' in metadata file",
                        name
                    ))
                })?;
                trees.insert(name, BTree::open(pool.clone(), root_id));
            }
            trees
        } else {
            let mut trees = HashMap::new();
            let mut meta = TreeMeta::new();
            for &name in TREE_NAMES {
                let btree = BTree::new(pool.clone())?;
                meta.insert(name.to_string(), btree.root_page_id());
                trees.insert(name, btree);
            }
            pool.flush()?;
            save_meta(&engine_dir, &meta)?;
            trees
        };

        fn take_tree(
            trees: &mut HashMap<&str, BTree>,
            name: &str,
        ) -> Result<RwLock<BTree>> {
            trees.remove(name).map(RwLock::new).ok_or_else(|| {
                MdbaseError::StorageError(format!("internal error: missing tree '{}'", name))
            })
        }

        let mut trees = trees;

        Ok(Self {
            pool,
            data_dir: engine_dir,
            version: AtomicU64::new(0),
            docs: take_tree(&mut trees, "docs")?,
            docs_by_path: take_tree(&mut trees, "docs_by_path")?,
            blocks: take_tree(&mut trees, "blocks")?,
            blocks_by_doc: take_tree(&mut trees, "blocks_by_doc")?,
            links: take_tree(&mut trees, "links")?,
            links_by_source: take_tree(&mut trees, "links_by_source")?,
            links_by_target: take_tree(&mut trees, "links_by_target")?,
            tags: take_tree(&mut trees, "tags")?,
            tags_by_doc: take_tree(&mut trees, "tags_by_doc")?,
        })
    }

    /// Write the current B+Tree root page IDs to the metadata file.
    fn save_metadata(&self) -> Result<()> {
        let mut meta = TreeMeta::new();

        meta.insert("docs".into(), self.docs.read().root_page_id());
        meta.insert("docs_by_path".into(), self.docs_by_path.read().root_page_id());
        meta.insert("blocks".into(), self.blocks.read().root_page_id());
        meta.insert("blocks_by_doc".into(), self.blocks_by_doc.read().root_page_id());
        meta.insert("links".into(), self.links.read().root_page_id());
        meta.insert("links_by_source".into(), self.links_by_source.read().root_page_id());
        meta.insert("links_by_target".into(), self.links_by_target.read().root_page_id());
        meta.insert("tags".into(), self.tags.read().root_page_id());
        meta.insert("tags_by_doc".into(), self.tags_by_doc.read().root_page_id());

        save_meta(&self.data_dir, &meta)
    }

    /// Persist metadata and flush dirty pages to disk.
    fn persist(&self) -> Result<()> {
        self.save_metadata()?;
        self.pool.flush()
    }

    /// Increment the version counter and return the new value.
    fn bump_version(&self) -> u64 {
        self.version.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Return the current version counter.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// Create a read-only snapshot of the current storage state.
    ///
    /// The snapshot materializes all data into memory and is isolated from
    /// subsequent writes to the engine.
    pub fn snapshot(&self) -> Result<StorageSnapshot> {
        // Acquire read locks on all primary trees simultaneously to get a
        // consistent view. We lock in a fixed order to avoid deadlocks, though
        // read locks cannot deadlock with each other in parking_lot.
        let docs = self.docs.read();
        let blocks_tree = self.blocks.read();
        let links_tree = self.links.read();
        let tags_tree = self.tags.read();

        let version = self.version.load(Ordering::SeqCst);

        // Read all documents.
        let doc_entries = docs.range_scan(None, None)?;
        let mut documents = Vec::with_capacity(doc_entries.len());
        for (_key, value) in doc_entries {
            documents.push(deserialize(&value)?);
        }

        // Read all blocks.
        let block_entries = blocks_tree.range_scan(None, None)?;
        let mut all_blocks = Vec::with_capacity(block_entries.len());
        for (_key, value) in block_entries {
            all_blocks.push(deserialize(&value)?);
        }

        // Read all links.
        let link_entries = links_tree.range_scan(None, None)?;
        let mut all_links = Vec::with_capacity(link_entries.len());
        for (_key, value) in link_entries {
            all_links.push(deserialize(&value)?);
        }

        // Read all tags.
        let tag_entries = tags_tree.range_scan(None, None)?;
        let mut all_tags = Vec::with_capacity(tag_entries.len());
        for (_key, value) in tag_entries {
            all_tags.push(deserialize(&value)?);
        }

        // Release all locks before constructing the snapshot.
        drop(docs);
        drop(blocks_tree);
        drop(links_tree);
        drop(tags_tree);

        Ok(StorageSnapshot::new(
            version,
            documents,
            all_blocks,
            all_links,
            all_tags,
        ))
    }
}

impl Storage for CustomStorageEngine {
    // -----------------------------------------------------------------------
    // Documents
    // -----------------------------------------------------------------------

    fn insert_document(&self, doc: &DocumentRecord) -> Result<()> {
        self.bump_version();

        let id_key = *doc.doc_id.as_bytes();
        let data = serialize(doc)?;

        let mut docs = self.docs.write();
        let mut docs_by_path = self.docs_by_path.write();

        // If this doc_id already exists, clean up the old path index entry
        // in case the path changed.
        if let Some(old_bytes) = docs.get(&id_key)? {
            let old_doc: DocumentRecord = deserialize(&old_bytes)?;
            if old_doc.path != doc.path {
                docs_by_path.delete(old_doc.path.as_bytes())?;
            }
        }

        docs.insert(&id_key, &data)?;
        docs_by_path.insert(doc.path.as_bytes(), doc.doc_id.as_bytes().as_slice())?;

        drop(docs);
        drop(docs_by_path);
        self.persist()
    }

    fn get_document(&self, doc_id: &Uuid) -> Result<Option<DocumentRecord>> {
        let docs = self.docs.read();
        match docs.get(doc_id.as_bytes())? {
            Some(bytes) => Ok(Some(deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    fn get_document_by_path(&self, path: &str) -> Result<Option<DocumentRecord>> {
        let docs_by_path = self.docs_by_path.read();
        let id_bytes = match docs_by_path.get(path.as_bytes())? {
            Some(b) => b,
            None => return Ok(None),
        };
        drop(docs_by_path);

        let doc_id = Uuid::from_slice(&id_bytes).map_err(|e| {
            MdbaseError::StorageError(format!("corrupt doc_id in path index: {e}"))
        })?;
        self.get_document(&doc_id)
    }

    fn list_documents(&self) -> Result<Vec<DocumentRecord>> {
        let docs = self.docs.read();
        let entries = docs.range_scan(None, None)?;
        let mut results = Vec::with_capacity(entries.len());
        for (_key, value) in entries {
            results.push(deserialize(&value)?);
        }
        Ok(results)
    }

    fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        self.bump_version();

        let id_key = *doc_id.as_bytes();

        // Remove the path index entry before deleting the document.
        {
            let docs = self.docs.read();
            if let Some(doc_bytes) = docs.get(&id_key)? {
                let doc: DocumentRecord = deserialize(&doc_bytes)?;
                drop(docs);
                let mut docs_by_path = self.docs_by_path.write();
                docs_by_path.delete(doc.path.as_bytes())?;
            }
        }

        // Cascade: delete all blocks belonging to this document.
        {
            let blocks_by_doc = self.blocks_by_doc.read();
            let entries = blocks_by_doc.prefix_scan(&id_key)?;
            drop(blocks_by_doc);

            if !entries.is_empty() {
                let mut blocks = self.blocks.write();
                let mut blocks_by_doc = self.blocks_by_doc.write();
                for (comp_key, _) in &entries {
                    // The composite key is doc_id(16) ++ block_id(16).
                    if comp_key.len() >= 32 {
                        let block_id_bytes = &comp_key[16..32];
                        blocks.delete(block_id_bytes)?;
                    }
                    blocks_by_doc.delete(comp_key)?;
                }
            }
        }

        // Cascade: delete all links originating from this document.
        {
            let links_by_source = self.links_by_source.read();
            let entries = links_by_source.prefix_scan(&id_key)?;
            drop(links_by_source);

            if !entries.is_empty() {
                let mut links = self.links.write();
                let mut links_by_source = self.links_by_source.write();
                let mut links_by_target = self.links_by_target.write();

                for (comp_key, _) in &entries {
                    // Extract the link_id from the composite key.
                    if comp_key.len() >= 32 {
                        let link_id_bytes = &comp_key[16..32];
                        // Look up the full link record to get the target for
                        // cleaning the links_by_target index.
                        if let Some(link_bytes) = links.get(link_id_bytes)? {
                            let link: LinkRecord = deserialize(&link_bytes)?;
                            let link_id = Uuid::from_slice(link_id_bytes).map_err(|e| {
                                MdbaseError::StorageError(format!("corrupt link_id: {e}"))
                            })?;
                            let tkey = target_index_key(&link.target, &link_id);
                            links_by_target.delete(&tkey)?;
                        }
                        links.delete(link_id_bytes)?;
                    }
                    links_by_source.delete(comp_key)?;
                }
            }
        }

        // Cascade: delete all tags belonging to this document.
        {
            let tags_by_doc = self.tags_by_doc.read();
            let entries = tags_by_doc.prefix_scan(&id_key)?;
            drop(tags_by_doc);

            if !entries.is_empty() {
                let mut tags = self.tags.write();
                let mut tags_by_doc = self.tags_by_doc.write();
                for (comp_key, _) in &entries {
                    if comp_key.len() >= 32 {
                        let tag_id_bytes = &comp_key[16..32];
                        tags.delete(tag_id_bytes)?;
                    }
                    tags_by_doc.delete(comp_key)?;
                }
            }
        }

        // Finally remove the document record itself.
        {
            let mut docs = self.docs.write();
            docs.delete(&id_key)?;
        }

        self.persist()
    }

    // -----------------------------------------------------------------------
    // Blocks
    // -----------------------------------------------------------------------

    fn insert_blocks(&self, blocks: &[BlockRecord]) -> Result<()> {
        self.bump_version();

        let mut blocks_tree = self.blocks.write();
        let mut blocks_by_doc = self.blocks_by_doc.write();

        for block in blocks {
            let data = serialize(block)?;
            blocks_tree.insert(block.block_id.as_bytes(), &data)?;

            let comp = composite_key(&block.doc_id, &block.block_id);
            blocks_by_doc.insert(&comp, &[])?;
        }

        drop(blocks_tree);
        drop(blocks_by_doc);
        self.persist()
    }

    fn get_blocks_by_doc(&self, doc_id: &Uuid) -> Result<Vec<BlockRecord>> {
        let id_prefix = *doc_id.as_bytes();
        let blocks_by_doc = self.blocks_by_doc.read();
        let entries = blocks_by_doc.prefix_scan(&id_prefix)?;
        drop(blocks_by_doc);

        let blocks = self.blocks.read();
        let mut results = Vec::with_capacity(entries.len());
        for (comp_key, _) in &entries {
            if comp_key.len() >= 32 {
                let block_id_bytes = &comp_key[16..32];
                if let Some(data) = blocks.get(block_id_bytes)? {
                    results.push(deserialize(&data)?);
                }
            }
        }
        Ok(results)
    }

    fn query_blocks(
        &self,
        doc_id: Option<&Uuid>,
        block_type: Option<&str>,
        heading_level: Option<u8>,
        language: Option<&str>,
    ) -> Result<Vec<BlockRecord>> {
        let candidates: Vec<BlockRecord> = if let Some(id) = doc_id {
            self.get_blocks_by_doc(id)?
        } else {
            // Scan all blocks.
            let blocks = self.blocks.read();
            let entries = blocks.range_scan(None, None)?;
            let mut all = Vec::with_capacity(entries.len());
            for (_key, value) in entries {
                all.push(deserialize(&value)?);
            }
            all
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
            .collect();

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Links
    // -----------------------------------------------------------------------

    fn insert_links(&self, links: &[LinkRecord]) -> Result<()> {
        self.bump_version();

        let mut links_tree = self.links.write();
        let mut links_by_source = self.links_by_source.write();
        let mut links_by_target = self.links_by_target.write();

        for link in links {
            let data = serialize(link)?;
            links_tree.insert(link.link_id.as_bytes(), &data)?;

            let source_comp = composite_key(&link.source_doc_id, &link.link_id);
            links_by_source.insert(&source_comp, &[])?;

            let tkey = target_index_key(&link.target, &link.link_id);
            links_by_target.insert(&tkey, &[])?;
        }

        drop(links_tree);
        drop(links_by_source);
        drop(links_by_target);
        self.persist()
    }

    fn get_links_from(&self, doc_id: &Uuid) -> Result<Vec<LinkRecord>> {
        let id_prefix = *doc_id.as_bytes();
        let links_by_source = self.links_by_source.read();
        let entries = links_by_source.prefix_scan(&id_prefix)?;
        drop(links_by_source);

        let links = self.links.read();
        let mut results = Vec::with_capacity(entries.len());
        for (comp_key, _) in &entries {
            if comp_key.len() >= 32 {
                let link_id_bytes = &comp_key[16..32];
                if let Some(data) = links.get(link_id_bytes)? {
                    results.push(deserialize(&data)?);
                }
            }
        }
        Ok(results)
    }

    fn get_links_to(&self, target_path: &str) -> Result<Vec<LinkRecord>> {
        let prefix = target_index_prefix(target_path);
        let links_by_target = self.links_by_target.read();
        let entries = links_by_target.prefix_scan(&prefix)?;
        drop(links_by_target);

        let links = self.links.read();
        let mut results = Vec::with_capacity(entries.len());
        for (tkey, _) in &entries {
            // The key is `target_bytes ++ \0 ++ link_id(16)`.
            // Extract the last 16 bytes as the link_id.
            if tkey.len() >= 17 {
                let link_id_bytes = &tkey[tkey.len() - 16..];
                if let Some(data) = links.get(link_id_bytes)? {
                    results.push(deserialize(&data)?);
                }
            }
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Tags
    // -----------------------------------------------------------------------

    fn insert_tags(&self, tags: &[TagRecord]) -> Result<()> {
        self.bump_version();

        let mut tags_tree = self.tags.write();
        let mut tags_by_doc = self.tags_by_doc.write();

        for tag in tags {
            let data = serialize(tag)?;
            tags_tree.insert(tag.tag_id.as_bytes(), &data)?;

            let comp = composite_key(&tag.doc_id, &tag.tag_id);
            tags_by_doc.insert(&comp, &[])?;
        }

        drop(tags_tree);
        drop(tags_by_doc);
        self.persist()
    }

    fn get_tags_by_doc(&self, doc_id: &Uuid) -> Result<Vec<TagRecord>> {
        let id_prefix = *doc_id.as_bytes();
        let tags_by_doc = self.tags_by_doc.read();
        let entries = tags_by_doc.prefix_scan(&id_prefix)?;
        drop(tags_by_doc);

        let tags = self.tags.read();
        let mut results = Vec::with_capacity(entries.len());
        for (comp_key, _) in &entries {
            if comp_key.len() >= 32 {
                let tag_id_bytes = &comp_key[16..32];
                if let Some(data) = tags.get(tag_id_bytes)? {
                    results.push(deserialize(&data)?);
                }
            }
        }
        Ok(results)
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

    // 1. Create engine and insert/get a document.
    #[test]
    fn document_insert_and_get() {
        let (engine, _dir) = temp_engine();
        let doc = make_document("notes/test.md");
        let doc_id = doc.doc_id;

        engine.insert_document(&doc).unwrap();

        let fetched = engine.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(fetched.doc_id, doc_id);
        assert_eq!(fetched.path, "notes/test.md");
        assert_eq!(fetched.title, Some("Title for notes/test.md".to_string()));
    }

    // 2. Insert document, then get_document_by_path.
    #[test]
    fn document_lookup_by_path() {
        let (engine, _dir) = temp_engine();
        let doc = make_document("notes/lookup.md");
        let doc_id = doc.doc_id;

        engine.insert_document(&doc).unwrap();

        let fetched = engine
            .get_document_by_path("notes/lookup.md")
            .unwrap()
            .unwrap();
        assert_eq!(fetched.doc_id, doc_id);

        assert!(engine
            .get_document_by_path("nonexistent.md")
            .unwrap()
            .is_none());
    }

    // 3. list_documents.
    #[test]
    fn list_documents() {
        let (engine, _dir) = temp_engine();

        engine.insert_document(&make_document("a.md")).unwrap();
        engine.insert_document(&make_document("b.md")).unwrap();
        engine.insert_document(&make_document("c.md")).unwrap();

        let docs = engine.list_documents().unwrap();
        assert_eq!(docs.len(), 3);
    }

    // 4. delete_document cascades blocks, links, tags.
    #[test]
    fn delete_document_cascades() {
        let (engine, _dir) = temp_engine();
        let doc = make_document("cascade.md");
        let doc_id = doc.doc_id;

        engine.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::Paragraph),
        ];
        engine.insert_blocks(&blocks).unwrap();

        let links = vec![make_link(doc_id, "other.md")];
        engine.insert_links(&links).unwrap();

        let tags = vec![make_tag(doc_id, "status", serde_json::json!("draft"))];
        engine.insert_tags(&tags).unwrap();

        // Verify data exists before deletion.
        assert!(engine.get_document(&doc_id).unwrap().is_some());
        assert_eq!(engine.get_blocks_by_doc(&doc_id).unwrap().len(), 2);
        assert_eq!(engine.get_links_from(&doc_id).unwrap().len(), 1);
        assert_eq!(engine.get_links_to("other.md").unwrap().len(), 1);
        assert_eq!(engine.get_tags_by_doc(&doc_id).unwrap().len(), 1);

        // Delete should remove everything.
        engine.delete_document(&doc_id).unwrap();

        assert!(engine.get_document(&doc_id).unwrap().is_none());
        assert!(engine.get_document_by_path("cascade.md").unwrap().is_none());
        assert!(engine.get_blocks_by_doc(&doc_id).unwrap().is_empty());
        assert!(engine.get_links_from(&doc_id).unwrap().is_empty());
        assert!(engine.get_links_to("other.md").unwrap().is_empty());
        assert!(engine.get_tags_by_doc(&doc_id).unwrap().is_empty());
    }

    // 5. insert_blocks and get_blocks_by_doc.
    #[test]
    fn insert_and_get_blocks() {
        let (engine, _dir) = temp_engine();
        let doc = make_document("blocks.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::Paragraph),
            make_block(doc_id, 2, BlockType::CodeBlock {
                language: Some("rust".to_string()),
            }),
        ];
        let block_ids: Vec<Uuid> = blocks.iter().map(|b| b.block_id).collect();
        engine.insert_blocks(&blocks).unwrap();

        let fetched = engine.get_blocks_by_doc(&doc_id).unwrap();
        assert_eq!(fetched.len(), 3);

        let fetched_ids: Vec<Uuid> = fetched.iter().map(|b| b.block_id).collect();
        for id in &block_ids {
            assert!(fetched_ids.contains(id), "missing block_id {id}");
        }
    }

    // 6. query_blocks with filters.
    #[test]
    fn query_blocks_with_filters() {
        let (engine, _dir) = temp_engine();
        let doc = make_document("query.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();

        let blocks = vec![
            make_block(doc_id, 0, BlockType::Heading { level: 1 }),
            make_block(doc_id, 1, BlockType::Heading { level: 2 }),
            make_block(doc_id, 2, BlockType::Paragraph),
            make_block(doc_id, 3, BlockType::CodeBlock {
                language: Some("rust".to_string()),
            }),
            make_block(doc_id, 4, BlockType::CodeBlock {
                language: Some("python".to_string()),
            }),
        ];
        engine.insert_blocks(&blocks).unwrap();

        // Filter by block_type "heading".
        let headings = engine
            .query_blocks(Some(&doc_id), Some("heading"), None, None)
            .unwrap();
        assert_eq!(headings.len(), 2);

        // Filter by heading_level = 2.
        let h2 = engine
            .query_blocks(Some(&doc_id), Some("heading"), Some(2), None)
            .unwrap();
        assert_eq!(h2.len(), 1);

        // Filter by language "rust".
        let rust_blocks = engine
            .query_blocks(Some(&doc_id), Some("code_block"), None, Some("rust"))
            .unwrap();
        assert_eq!(rust_blocks.len(), 1);

        // Filter by "paragraph".
        let paragraphs = engine
            .query_blocks(Some(&doc_id), Some("paragraph"), None, None)
            .unwrap();
        assert_eq!(paragraphs.len(), 1);

        // Query all blocks across all documents (no doc_id filter).
        let all_headings = engine
            .query_blocks(None, Some("heading"), None, None)
            .unwrap();
        assert_eq!(all_headings.len(), 2);

        // No filters at all.
        let everything = engine.query_blocks(None, None, None, None).unwrap();
        assert_eq!(everything.len(), 5);
    }

    // 7. insert_links, get_links_from, get_links_to.
    #[test]
    fn link_storage_and_lookup() {
        let (engine, _dir) = temp_engine();
        let doc_a = make_document("a.md");
        let doc_b = make_document("b.md");
        engine.insert_document(&doc_a).unwrap();
        engine.insert_document(&doc_b).unwrap();

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
        engine.insert_links(&links).unwrap();

        // All links from doc_a.
        let from_a = engine.get_links_from(&doc_a.doc_id).unwrap();
        assert_eq!(from_a.len(), 2);

        // Backlinks to "b.md".
        let to_b = engine.get_links_to("b.md").unwrap();
        assert_eq!(to_b.len(), 1);
        assert_eq!(to_b[0].source_doc_id, doc_a.doc_id);

        // No links point to "a.md".
        let to_a = engine.get_links_to("a.md").unwrap();
        assert!(to_a.is_empty());
    }

    // 8. insert_tags and get_tags_by_doc.
    #[test]
    fn tag_storage_and_lookup() {
        let (engine, _dir) = temp_engine();
        let doc = make_document("tags.md");
        let doc_id = doc.doc_id;
        engine.insert_document(&doc).unwrap();

        let tags = vec![
            make_tag(doc_id, "category", serde_json::json!("notes")),
            make_tag(doc_id, "priority", serde_json::json!(1)),
        ];
        engine.insert_tags(&tags).unwrap();

        let fetched = engine.get_tags_by_doc(&doc_id).unwrap();
        assert_eq!(fetched.len(), 2);

        let keys: Vec<&str> = fetched.iter().map(|t| t.key.as_str()).collect();
        assert!(keys.contains(&"category"));
        assert!(keys.contains(&"priority"));
    }

    // 9. Reopen engine preserves data.
    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let doc_id;
        let block_id;

        // Write data and close.
        {
            let engine = CustomStorageEngine::new(dir.path()).unwrap();
            let doc = make_document("persist.md");
            doc_id = doc.doc_id;
            engine.insert_document(&doc).unwrap();

            let block = make_block(doc_id, 0, BlockType::Heading { level: 1 });
            block_id = block.block_id;
            engine.insert_blocks(&[block]).unwrap();
        }

        // Reopen and verify data survived.
        {
            let engine = CustomStorageEngine::new(dir.path()).unwrap();

            let doc = engine.get_document(&doc_id).unwrap().unwrap();
            assert_eq!(doc.path, "persist.md");

            let doc_by_path = engine.get_document_by_path("persist.md").unwrap().unwrap();
            assert_eq!(doc_by_path.doc_id, doc_id);

            let blocks = engine.get_blocks_by_doc(&doc_id).unwrap();
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].block_id, block_id);
        }
    }

    // 10. Replace existing document at same path.
    #[test]
    fn replace_document_at_same_path() {
        let (engine, _dir) = temp_engine();

        let mut doc = make_document("replace.md");
        let doc_id = doc.doc_id;

        engine.insert_document(&doc).unwrap();

        // Update the document (same doc_id, same path, new title).
        doc.title = Some("Updated Title".to_string());
        doc.version = 2;
        engine.insert_document(&doc).unwrap();

        let fetched = engine.get_document(&doc_id).unwrap().unwrap();
        assert_eq!(fetched.title, Some("Updated Title".to_string()));
        assert_eq!(fetched.version, 2);

        // Path index should still resolve.
        let by_path = engine
            .get_document_by_path("replace.md")
            .unwrap()
            .unwrap();
        assert_eq!(by_path.doc_id, doc_id);

        // list_documents should have exactly one entry.
        let all = engine.list_documents().unwrap();
        assert_eq!(all.len(), 1);
    }

    // Extra: document path update cleans old path index entry.
    #[test]
    fn document_path_update() {
        let (engine, _dir) = temp_engine();
        let mut doc = make_document("old/path.md");

        engine.insert_document(&doc).unwrap();

        // Update the path and re-insert.
        doc.path = "new/path.md".to_string();
        engine.insert_document(&doc).unwrap();

        // Old path should no longer resolve.
        assert!(engine
            .get_document_by_path("old/path.md")
            .unwrap()
            .is_none());

        // New path should work.
        assert!(engine
            .get_document_by_path("new/path.md")
            .unwrap()
            .is_some());
    }

    // Extra: get nonexistent document returns None.
    #[test]
    fn get_nonexistent_document_returns_none() {
        let (engine, _dir) = temp_engine();
        let result = engine.get_document(&Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    // Extra: delete nonexistent document is a no-op.
    #[test]
    fn delete_nonexistent_document_is_noop() {
        let (engine, _dir) = temp_engine();
        engine.delete_document(&Uuid::new_v4()).unwrap();
    }

    // Extra: blocks from different documents are isolated.
    #[test]
    fn blocks_isolated_across_documents() {
        let (engine, _dir) = temp_engine();
        let doc_a = make_document("a.md");
        let doc_b = make_document("b.md");
        engine.insert_document(&doc_a).unwrap();
        engine.insert_document(&doc_b).unwrap();

        engine
            .insert_blocks(&[
                make_block(doc_a.doc_id, 0, BlockType::Heading { level: 1 }),
                make_block(doc_a.doc_id, 1, BlockType::Paragraph),
            ])
            .unwrap();
        engine
            .insert_blocks(&[
                make_block(doc_b.doc_id, 0, BlockType::CodeBlock {
                    language: Some("go".to_string()),
                }),
            ])
            .unwrap();

        assert_eq!(engine.get_blocks_by_doc(&doc_a.doc_id).unwrap().len(), 2);
        assert_eq!(engine.get_blocks_by_doc(&doc_b.doc_id).unwrap().len(), 1);
    }

    // Extra: links_to with multiple sources.
    #[test]
    fn links_to_with_multiple_sources() {
        let (engine, _dir) = temp_engine();
        let doc_a = make_document("a.md");
        let doc_b = make_document("b.md");
        engine.insert_document(&doc_a).unwrap();
        engine.insert_document(&doc_b).unwrap();

        let links = vec![
            make_link(doc_a.doc_id, "target.md"),
            make_link(doc_b.doc_id, "target.md"),
        ];
        engine.insert_links(&links).unwrap();

        let to_target = engine.get_links_to("target.md").unwrap();
        assert_eq!(to_target.len(), 2);
    }

    // Extra: persistence of links and tags across reopen.
    #[test]
    fn persistence_of_links_and_tags() {
        let dir = tempfile::tempdir().unwrap();
        let doc_id;

        {
            let engine = CustomStorageEngine::new(dir.path()).unwrap();
            let doc = make_document("full.md");
            doc_id = doc.doc_id;
            engine.insert_document(&doc).unwrap();

            engine
                .insert_links(&[make_link(doc_id, "linked.md")])
                .unwrap();
            engine
                .insert_tags(&[make_tag(doc_id, "key", serde_json::json!("val"))])
                .unwrap();
        }

        {
            let engine = CustomStorageEngine::new(dir.path()).unwrap();

            let links = engine.get_links_from(&doc_id).unwrap();
            assert_eq!(links.len(), 1);
            assert_eq!(links[0].target, "linked.md");

            let backlinks = engine.get_links_to("linked.md").unwrap();
            assert_eq!(backlinks.len(), 1);

            let tags = engine.get_tags_by_doc(&doc_id).unwrap();
            assert_eq!(tags.len(), 1);
            assert_eq!(tags[0].key, "key");
        }
    }
}
