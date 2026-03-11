#[derive(Debug, thiserror::Error)]
pub enum MqlError {
    #[error("parse error: {0}")]
    ParseError(String),
    #[error("invalid UUID: {0}")]
    InvalidUuid(String),
    #[error("invalid number: {0}")]
    InvalidNumber(String),
}
