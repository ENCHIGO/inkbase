use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// BlockType — the kind of structural element a block represents
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockType {
    Heading { level: u8 },
    Paragraph,
    CodeBlock { language: Option<String> },
    List { ordered: bool },
    ListItem,
    Table,
    BlockQuote,
    ThematicBreak,
    Image { url: String, alt: String },
    Html,
    FootnoteDefinition,
}

// ---------------------------------------------------------------------------
// LinkType — how two documents are connected
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
    WikiLink,
    MarkdownLink,
    AutoLink,
    ImageLink,
}

// ---------------------------------------------------------------------------
// DocumentRecord — represents a single Markdown file in the store
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub doc_id: Uuid,
    /// Relative path within the data directory (e.g. "notes/rust.md").
    pub path: String,
    /// Document title extracted from frontmatter or the first heading.
    pub title: Option<String>,
    /// Parsed frontmatter as an opaque JSON value.
    pub frontmatter: serde_json::Value,
    /// BLAKE3 hash of the raw file content, hex-encoded.
    pub raw_content_hash: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Monotonically increasing version counter for optimistic concurrency.
    pub version: u64,
}

// ---------------------------------------------------------------------------
// BlockRecord — a structural element within a document
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRecord {
    pub block_id: Uuid,
    pub doc_id: Uuid,
    pub block_type: BlockType,
    /// Nesting depth in the document tree (0 = top-level).
    pub depth: u32,
    /// Position among siblings at the same depth.
    pub ordinal: u32,
    /// Parent block, if this block is nested inside another.
    pub parent_block_id: Option<Uuid>,
    /// Plain-text content with markup stripped.
    pub text_content: String,
    /// Original Markdown source for this block.
    pub raw_markdown: String,
}

// ---------------------------------------------------------------------------
// LinkRecord — a directed link between documents (or to an external target)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkRecord {
    pub link_id: Uuid,
    pub source_doc_id: Uuid,
    /// The block where the link appears, if known.
    pub source_block_id: Option<Uuid>,
    /// Raw link target as written in the Markdown source.
    pub target: String,
    /// Resolved target document id, if the target is an internal document.
    pub target_doc_id: Option<Uuid>,
    pub link_type: LinkType,
    /// Visible anchor text of the link.
    pub anchor_text: String,
}

// ---------------------------------------------------------------------------
// EmbeddingRecord — a vector embedding for semantic search
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub embedding_id: Uuid,
    pub doc_id: Uuid,
    /// The specific block this embedding covers, or `None` for whole-document.
    pub block_id: Option<Uuid>,
    /// Dense float vector produced by the embedding model.
    pub vector: Vec<f32>,
    /// Identifier for the model that produced this vector (e.g. "all-MiniLM-L6-v2").
    pub model_id: String,
}

// ---------------------------------------------------------------------------
// TagRecord — a key/value tag extracted from frontmatter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    pub tag_id: Uuid,
    pub doc_id: Uuid,
    /// Tag key (e.g. "category", "status").
    pub key: String,
    /// Tag value — kept as opaque JSON to support strings, arrays, etc.
    pub value: serde_json::Value,
}
