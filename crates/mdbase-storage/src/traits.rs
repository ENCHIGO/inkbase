use mdbase_core::types::{BlockRecord, DocumentRecord, LinkRecord, TagRecord};
use uuid::Uuid;

/// Trait abstracting all storage operations for Markdotabase.
///
/// Implementations must be thread-safe (`Send + Sync`) so they can be shared
/// across async tasks behind an `Arc`.
pub trait Storage: Send + Sync {
    // -- Document operations -------------------------------------------------

    /// Persist a document record. Overwrites any existing record with the same
    /// `doc_id` and updates the path index accordingly.
    fn insert_document(&self, doc: &DocumentRecord) -> mdbase_core::Result<()>;

    /// Look up a document by its unique id.
    fn get_document(&self, doc_id: &Uuid) -> mdbase_core::Result<Option<DocumentRecord>>;

    /// Look up a document by its file path (e.g. `"notes/rust.md"`).
    fn get_document_by_path(&self, path: &str) -> mdbase_core::Result<Option<DocumentRecord>>;

    /// Return all stored documents. Order is not guaranteed.
    fn list_documents(&self) -> mdbase_core::Result<Vec<DocumentRecord>>;

    /// Remove a document and **cascade-delete** its blocks, links, and tags.
    fn delete_document(&self, doc_id: &Uuid) -> mdbase_core::Result<()>;

    // -- Block operations ----------------------------------------------------

    /// Bulk-insert block records. Existing blocks with matching ids are
    /// overwritten.
    fn insert_blocks(&self, blocks: &[BlockRecord]) -> mdbase_core::Result<()>;

    /// Return all blocks belonging to a document, in insertion order.
    fn get_blocks_by_doc(&self, doc_id: &Uuid) -> mdbase_core::Result<Vec<BlockRecord>>;

    /// Query blocks with optional filters. All filter parameters are ANDed
    /// together — a block must satisfy every non-`None` filter to be included.
    ///
    /// * `doc_id` — restrict to a single document.
    /// * `block_type` — match the variant name, e.g. `"heading"`, `"code_block"`.
    /// * `heading_level` — for heading blocks, match this exact level.
    /// * `language` — for code blocks, match this language string.
    fn query_blocks(
        &self,
        doc_id: Option<&Uuid>,
        block_type: Option<&str>,
        heading_level: Option<u8>,
        language: Option<&str>,
    ) -> mdbase_core::Result<Vec<BlockRecord>>;

    // -- Link operations -----------------------------------------------------

    /// Bulk-insert link records.
    fn insert_links(&self, links: &[LinkRecord]) -> mdbase_core::Result<()>;

    /// Return all outgoing links from a document.
    fn get_links_from(&self, doc_id: &Uuid) -> mdbase_core::Result<Vec<LinkRecord>>;

    /// Return all links whose `target` matches the given path (backlinks).
    fn get_links_to(&self, target_path: &str) -> mdbase_core::Result<Vec<LinkRecord>>;

    // -- Tag operations ------------------------------------------------------

    /// Bulk-insert tag records.
    fn insert_tags(&self, tags: &[TagRecord]) -> mdbase_core::Result<()>;

    /// Return all tags associated with a document.
    fn get_tags_by_doc(&self, doc_id: &Uuid) -> mdbase_core::Result<Vec<TagRecord>>;
}
