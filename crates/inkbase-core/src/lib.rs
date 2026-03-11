pub mod config;
pub mod error;
pub mod types;

// Re-export the most commonly used items at crate root for ergonomic imports.
pub use config::InkbaseConfig;
pub use error::{InkbaseError, Result};
pub use types::{
    BlockRecord, BlockType, DocumentRecord, EmbeddingRecord, LinkRecord, LinkType, TagRecord,
};
