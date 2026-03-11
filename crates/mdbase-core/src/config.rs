use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level configuration for a Markdotabase instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdbaseConfig {
    /// Root directory for document and metadata storage.
    pub data_dir: PathBuf,
    /// Directory for search and vector index files.
    pub index_dir: PathBuf,
    /// Default embedding model identifier.
    pub embedding_model: String,
    /// Dimensionality of the embedding vectors.
    pub embedding_dim: usize,
    /// Logging verbosity level (e.g. "info", "debug", "warn").
    pub log_level: String,
}

impl Default for MdbaseConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            index_dir: PathBuf::from("./data/index"),
            embedding_model: String::from("all-MiniLM-L6-v2"),
            embedding_dim: 384,
            log_level: String::from("info"),
        }
    }
}
