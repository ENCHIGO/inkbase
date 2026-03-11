pub mod frontmatter;
pub mod parser;

// Re-export the primary public API at crate root for ergonomic access.
pub use parser::{parse_markdown, ParseResult};
