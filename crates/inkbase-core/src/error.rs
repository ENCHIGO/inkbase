use thiserror::Error;

/// Central error type for all Inkbase operations.
#[derive(Debug, Error)]
pub enum InkbaseError {
    #[error("document not found: {0}")]
    DocumentNotFound(String),

    #[error("block not found: {0}")]
    BlockNotFound(String),

    #[error("parse error: {0}")]
    ParseError(String),

    #[error("storage error: {0}")]
    StorageError(String),

    #[error("index error: {0}")]
    IndexError(String),

    #[error("graph error: {0}")]
    GraphError(String),

    #[error("embedding error: {0}")]
    EmbeddingError(String),

    #[error("config error: {0}")]
    ConfigError(String),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    SerializationError(String),
}

impl From<serde_json::Error> for InkbaseError {
    fn from(err: serde_json::Error) -> Self {
        InkbaseError::SerializationError(err.to_string())
    }
}

/// Convenience alias used throughout the codebase.
pub type Result<T> = std::result::Result<T, InkbaseError>;
