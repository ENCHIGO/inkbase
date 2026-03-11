use mdbase_core::{BlockRecord, BlockType, MdbaseError};

/// Extracts YAML frontmatter from the beginning of a Markdown document.
///
/// Frontmatter must start with `---` on the very first line and end with a
/// matching `---` delimiter. Returns `(Some(yaml_string), remaining_content)`
/// when frontmatter is found, or `(None, full_content)` otherwise.
pub fn extract_frontmatter(content: &str) -> (Option<String>, &str) {
    // Frontmatter must begin at byte 0 with exactly "---" followed by a newline.
    let trimmed_start = content.strip_prefix("---");
    let after_opener = match trimmed_start {
        Some(rest) if rest.starts_with('\n') || rest.starts_with("\r\n") => rest,
        // Allow "---" as the very first line with nothing else on it.
        _ => return (None, content),
    };

    // Find the closing "---" delimiter. It must appear on its own line.
    // We search for "\n---\n" or "\n---\r\n" or "\n---" at end-of-string.
    let search_start = 0;
    let bytes = after_opener.as_bytes();
    let mut pos = search_start;

    while pos < bytes.len() {
        // Find the next newline.
        let nl = match memchr_newline(bytes, pos) {
            Some(idx) => idx,
            None => break,
        };

        // Skip past the newline character(s) to the start of the next line.
        let line_start = if bytes.get(nl) == Some(&b'\r') && bytes.get(nl + 1) == Some(&b'\n') {
            nl + 2
        } else {
            nl + 1
        };

        // Check if this line starts with "---".
        if after_opener[line_start..].starts_with("---") {
            let after_dashes = line_start + 3;
            // The delimiter line must be *only* "---" (optionally followed by whitespace/newline).
            let rest_of_line = &after_opener[after_dashes..];
            let is_valid_close = rest_of_line.is_empty()
                || rest_of_line.starts_with('\n')
                || rest_of_line.starts_with("\r\n");

            if is_valid_close {
                let yaml = &after_opener[..line_start];
                // Remaining content starts after the closing "---\n".
                let remaining_start = after_dashes;
                let remaining = if rest_of_line.starts_with("\r\n") {
                    &after_opener[remaining_start + 2..]
                } else if rest_of_line.starts_with('\n') {
                    &after_opener[remaining_start + 1..]
                } else {
                    // EOF right after "---"
                    &after_opener[remaining_start..]
                };

                let yaml_trimmed = yaml.trim();
                if yaml_trimmed.is_empty() {
                    return (None, remaining);
                }
                return (Some(yaml_trimmed.to_string()), remaining);
            }
        }

        pos = line_start;
    }

    // No closing delimiter found — treat entire content as body (no frontmatter).
    (None, content)
}

/// Parses a YAML string into a `serde_json::Value`.
///
/// Returns `MdbaseError::ParseError` if the YAML is malformed.
pub fn parse_frontmatter(yaml: &str) -> mdbase_core::Result<serde_json::Value> {
    let yaml_value: serde_yaml::Value =
        serde_yaml::from_str(yaml).map_err(|e| MdbaseError::ParseError(format!("invalid frontmatter YAML: {e}")))?;

    serde_json::to_value(yaml_value)
        .map_err(|e| MdbaseError::ParseError(format!("failed to convert YAML to JSON: {e}")))
}

/// Extracts the document title.
///
/// Prefers the `title` key in frontmatter (if it is a string). Falls back to
/// the `text_content` of the first heading-level-1 block.
pub fn extract_title(frontmatter: &serde_json::Value, blocks: &[BlockRecord]) -> Option<String> {
    // Try frontmatter "title" key.
    if let Some(title) = frontmatter.get("title").and_then(|v| v.as_str()) {
        let trimmed = title.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    // Fallback: first H1 block.
    blocks
        .iter()
        .find(|b| matches!(b.block_type, BlockType::Heading { level: 1 }))
        .map(|b| b.text_content.clone())
}

// ---- helpers ----------------------------------------------------------------

/// Finds the index of the first `\n` or `\r` byte starting at `from`.
fn memchr_newline(bytes: &[u8], from: usize) -> Option<usize> {
    bytes[from..]
        .iter()
        .position(|&b| b == b'\n' || b == b'\r')
        .map(|p| p + from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_simple_frontmatter() {
        let input = "---\ntitle: Hello\ntags: [a, b]\n---\n# Body\n";
        let (fm, body) = extract_frontmatter(input);
        assert!(fm.is_some());
        assert_eq!(body, "# Body\n");

        let parsed = parse_frontmatter(&fm.unwrap()).unwrap();
        assert_eq!(parsed["title"], json!("Hello"));
    }

    #[test]
    fn no_frontmatter() {
        let input = "# Just a heading\nSome text.";
        let (fm, body) = extract_frontmatter(input);
        assert!(fm.is_none());
        assert_eq!(body, input);
    }

    #[test]
    fn frontmatter_at_eof() {
        let input = "---\nkey: val\n---";
        let (fm, body) = extract_frontmatter(input);
        assert!(fm.is_some());
        assert_eq!(body, "");
    }

    #[test]
    fn empty_frontmatter_block() {
        let input = "---\n---\nContent here";
        let (fm, body) = extract_frontmatter(input);
        assert!(fm.is_none());
        assert_eq!(body, "Content here");
    }

    #[test]
    fn extract_title_from_frontmatter() {
        let fm = json!({"title": "My Doc"});
        assert_eq!(extract_title(&fm, &[]), Some("My Doc".to_string()));
    }

    #[test]
    fn extract_title_fallback_h1() {
        let fm = json!({});
        let blocks = vec![BlockRecord {
            block_id: uuid::Uuid::new_v4(),
            doc_id: uuid::Uuid::new_v4(),
            block_type: BlockType::Heading { level: 1 },
            depth: 0,
            ordinal: 0,
            parent_block_id: None,
            text_content: "First Heading".to_string(),
            raw_markdown: "# First Heading".to_string(),
        }];
        assert_eq!(extract_title(&fm, &blocks), Some("First Heading".to_string()));
    }
}
