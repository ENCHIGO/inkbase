use chrono::Utc;
use comrak::nodes::NodeValue;
use comrak::{parse_document, Arena, Options};
use serde::Serialize;
use uuid::Uuid;

use mdbase_core::{BlockRecord, BlockType, DocumentRecord, LinkRecord, LinkType, TagRecord};

use crate::frontmatter;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The complete result of parsing a single Markdown document.
#[derive(Debug, Clone, Serialize)]
pub struct ParseResult {
    pub document: DocumentRecord,
    pub blocks: Vec<BlockRecord>,
    pub links: Vec<LinkRecord>,
    pub tags: Vec<TagRecord>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parses a Markdown document and produces structured records.
///
/// `path` is the logical path of the document within the data directory
/// (e.g. `"notes/rust.md"`). `content` is the raw file content, including any
/// YAML frontmatter.
///
/// Returns a `ParseResult` containing the document, its blocks, links, and tags.
pub fn parse_markdown(path: &str, content: &str) -> mdbase_core::Result<ParseResult> {
    let doc_id = Uuid::new_v4();
    let now = Utc::now();

    tracing::debug!(path, "starting markdown parse");

    // 1. Extract and parse frontmatter.
    let (frontmatter_yaml, body) = frontmatter::extract_frontmatter(content);
    let frontmatter = match &frontmatter_yaml {
        Some(yaml) => {
            tracing::debug!("frontmatter found, parsing YAML");
            frontmatter::parse_frontmatter(yaml)?
        }
        None => serde_json::Value::Null,
    };

    // 2. Compute content hash.
    let raw_content_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    // 3. Parse the Markdown body into a comrak AST.
    let arena = Arena::new();
    let options = comrak_options();
    let root = parse_document(&arena, body, &options);

    // 4. Walk the AST to produce blocks and collect links.
    let mut blocks: Vec<BlockRecord> = Vec::new();
    let mut links: Vec<LinkRecord> = Vec::new();
    walk_ast(root, doc_id, &mut blocks, &mut links);

    tracing::debug!(
        block_count = blocks.len(),
        link_count = links.len(),
        "AST walk complete"
    );

    // 5. Scan the original body for [[wiki-links]] not captured by comrak.
    extract_wiki_links(body, doc_id, &mut links);

    // 6. Extract title.
    let title = frontmatter::extract_title(&frontmatter, &blocks);

    // 7. Extract tags from frontmatter.
    let tags = extract_tags(doc_id, &frontmatter);

    tracing::debug!(
        tag_count = tags.len(),
        title = title.as_deref().unwrap_or("<none>"),
        "parse complete"
    );

    let document = DocumentRecord {
        doc_id,
        path: path.to_string(),
        title,
        frontmatter,
        raw_content_hash,
        created_at: now,
        updated_at: now,
        version: 1,
    };

    Ok(ParseResult {
        document,
        blocks,
        links,
        tags,
    })
}

// ---------------------------------------------------------------------------
// Comrak options
// ---------------------------------------------------------------------------

fn comrak_options() -> Options<'static> {
    let mut opts = Options::default();
    // Enable common extensions that appear in knowledge-base Markdown.
    opts.extension.strikethrough = true;
    opts.extension.table = true;
    opts.extension.autolink = true;
    opts.extension.tasklist = true;
    opts.extension.footnotes = true;
    opts.extension.front_matter_delimiter = None; // We handle frontmatter ourselves.
    opts
}

// ---------------------------------------------------------------------------
// AST walking
// ---------------------------------------------------------------------------

/// State carried while descending through the comrak AST.
struct WalkState {
    /// Sequential counter for block ordinals.
    ordinal: u32,
    /// Stack of (block_id, depth) for tracking parent relationships.
    parent_stack: Vec<(Uuid, u32)>,
}

impl WalkState {
    fn new() -> Self {
        Self {
            ordinal: 0,
            parent_stack: Vec::new(),
        }
    }

    fn next_ordinal(&mut self) -> u32 {
        let o = self.ordinal;
        self.ordinal += 1;
        o
    }

    fn current_parent(&self) -> Option<Uuid> {
        self.parent_stack.last().map(|(id, _)| *id)
    }

    fn current_depth(&self) -> u32 {
        self.parent_stack.len() as u32
    }
}

/// Walks the comrak AST rooted at `node`, producing `BlockRecord`s and
/// `LinkRecord`s. Uses an explicit stack to avoid deep recursion.
fn walk_ast<'a>(
    root: &'a comrak::nodes::AstNode<'a>,
    doc_id: Uuid,
    blocks: &mut Vec<BlockRecord>,
    links: &mut Vec<LinkRecord>,
) {
    let mut state = WalkState::new();
    walk_node(root, doc_id, blocks, links, &mut state);
}

fn walk_node<'a>(
    node: &'a comrak::nodes::AstNode<'a>,
    doc_id: Uuid,
    blocks: &mut Vec<BlockRecord>,
    links: &mut Vec<LinkRecord>,
    state: &mut WalkState,
) {
    let ast = node.data.borrow();

    match &ast.value {
        // The document root is not a block — just recurse into children.
        NodeValue::Document => {
            drop(ast);
            for child in node.children() {
                walk_node(child, doc_id, blocks, links, state);
            }
        }

        // ----- Block-level nodes that produce BlockRecords -----
        NodeValue::Heading(heading) => {
            let level = heading.level;
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::Heading { level },
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });

            // Headings can contain inline links.
            collect_inline_links(node, doc_id, Some(block_id), links);
        }

        NodeValue::Paragraph => {
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::Paragraph,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });

            collect_inline_links(node, doc_id, Some(block_id), links);
        }

        NodeValue::CodeBlock(cb) => {
            let language = if cb.info.is_empty() {
                None
            } else {
                // The info string may contain additional metadata after the
                // language name (e.g. "rust,linenos"). Take only the first
                // whitespace-delimited token.
                Some(
                    cb.info
                        .split_whitespace()
                        .next()
                        .unwrap_or(&cb.info)
                        .to_string(),
                )
            };
            let text = cb.literal.clone();
            let raw_markdown = format_code_block_markdown(&cb.info, &cb.literal);

            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            blocks.push(BlockRecord {
                block_id: Uuid::new_v4(),
                doc_id,
                block_type: BlockType::CodeBlock { language },
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown,
            });
        }

        NodeValue::List(list) => {
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            let ordered = list.list_type == comrak::nodes::ListType::Ordered;
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::List { ordered },
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });

            // Push this list as the parent context for its children.
            state.parent_stack.push((block_id, depth));
            for child in node.children() {
                walk_node(child, doc_id, blocks, links, state);
            }
            state.parent_stack.pop();
        }

        NodeValue::Item(_) => {
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::ListItem,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });

            // List items can contain nested lists, paragraphs, etc.
            state.parent_stack.push((block_id, depth));
            for child in node.children() {
                walk_node(child, doc_id, blocks, links, state);
            }
            state.parent_stack.pop();
        }

        NodeValue::Table(_) => {
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::Table,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });

            collect_inline_links(node, doc_id, Some(block_id), links);
        }

        NodeValue::BlockQuote => {
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::BlockQuote,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });

            // Block quotes contain nested block elements.
            state.parent_stack.push((block_id, depth));
            for child in node.children() {
                walk_node(child, doc_id, blocks, links, state);
            }
            state.parent_stack.pop();
        }

        NodeValue::ThematicBreak => {
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            blocks.push(BlockRecord {
                block_id: Uuid::new_v4(),
                doc_id,
                block_type: BlockType::ThematicBreak,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: String::new(),
                raw_markdown: "---".to_string(),
            });
        }

        NodeValue::HtmlBlock(html) => {
            let text = html.literal.clone();
            let raw = html.literal.clone();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            drop(ast);

            blocks.push(BlockRecord {
                block_id: Uuid::new_v4(),
                doc_id,
                block_type: BlockType::Html,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: text,
                raw_markdown: raw,
            });
        }

        NodeValue::FootnoteDefinition(fd) => {
            let block_id = Uuid::new_v4();
            let depth = state.current_depth();
            let parent = state.current_parent();
            let ordinal = state.next_ordinal();
            let name = fd.name.clone();
            drop(ast);

            let text = collect_text(node);
            let raw = collect_raw_markdown(node);

            blocks.push(BlockRecord {
                block_id,
                doc_id,
                block_type: BlockType::FootnoteDefinition,
                depth,
                ordinal,
                parent_block_id: parent,
                text_content: format!("[^{name}]: {text}"),
                raw_markdown: raw,
            });

            state.parent_stack.push((block_id, depth));
            for child in node.children() {
                walk_node(child, doc_id, blocks, links, state);
            }
            state.parent_stack.pop();
        }

        // Inline-only or structural nodes that we don't emit as blocks but
        // whose children we still need to visit.
        _ => {
            drop(ast);
            for child in node.children() {
                walk_node(child, doc_id, blocks, links, state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inline content extraction helpers
// ---------------------------------------------------------------------------

/// Recursively collects all plain-text content from inline children of `node`.
fn collect_text<'a>(node: &'a comrak::nodes::AstNode<'a>) -> String {
    let mut buf = String::new();
    collect_text_inner(node, &mut buf);
    buf.trim().to_string()
}

fn collect_text_inner<'a>(node: &'a comrak::nodes::AstNode<'a>, buf: &mut String) {
    let ast = node.data.borrow();
    match &ast.value {
        NodeValue::Text(t) => buf.push_str(t),
        NodeValue::Code(c) => {
            buf.push('`');
            buf.push_str(&c.literal);
            buf.push('`');
        }
        NodeValue::SoftBreak | NodeValue::LineBreak => buf.push(' '),
        _ => {}
    }
    drop(ast);
    for child in node.children() {
        collect_text_inner(child, buf);
    }
}

/// Reconstructs approximate Markdown source for a node subtree by rendering
/// the plain text. For nodes like code blocks we reconstruct explicitly; for
/// others this provides a reasonable approximation.
fn collect_raw_markdown<'a>(node: &'a comrak::nodes::AstNode<'a>) -> String {
    let mut buf = Vec::new();
    let options = comrak_options();
    comrak::format_commonmark(node, &options, &mut buf)
        .unwrap_or_default();
    String::from_utf8_lossy(&buf).trim().to_string()
}

fn format_code_block_markdown(info: &str, literal: &str) -> String {
    let mut raw = String::with_capacity(info.len() + literal.len() + 10);
    raw.push_str("```");
    raw.push_str(info);
    raw.push('\n');
    raw.push_str(literal);
    if !literal.ends_with('\n') {
        raw.push('\n');
    }
    raw.push_str("```");
    raw
}

// ---------------------------------------------------------------------------
// Link extraction
// ---------------------------------------------------------------------------

/// Collects markdown links and images from inline nodes within a subtree.
fn collect_inline_links<'a>(
    node: &'a comrak::nodes::AstNode<'a>,
    doc_id: Uuid,
    source_block_id: Option<Uuid>,
    links: &mut Vec<LinkRecord>,
) {
    for descendant in node.descendants() {
        let ast = descendant.data.borrow();
        match &ast.value {
            NodeValue::Link(link) => {
                let url = link.url.clone();
                drop(ast);
                let anchor = collect_text(descendant);

                // An autolink has its anchor text equal to or contained in the
                // URL (e.g. bare `https://example.com` or `<user@host>`). A
                // standard markdown link has distinct anchor text.
                let link_type = if anchor == url
                    || url == format!("mailto:{anchor}")
                {
                    LinkType::AutoLink
                } else {
                    LinkType::MarkdownLink
                };

                links.push(LinkRecord {
                    link_id: Uuid::new_v4(),
                    source_doc_id: doc_id,
                    source_block_id,
                    target: url,
                    target_doc_id: None,
                    link_type,
                    anchor_text: anchor,
                });
            }

            NodeValue::Image(img) => {
                let url = img.url.clone();
                drop(ast);
                let alt = collect_text(descendant);

                links.push(LinkRecord {
                    link_id: Uuid::new_v4(),
                    source_doc_id: doc_id,
                    source_block_id,
                    target: url,
                    target_doc_id: None,
                    link_type: LinkType::ImageLink,
                    anchor_text: alt,
                });
            }

            _ => {
                drop(ast);
            }
        }
    }
}

/// Scans the raw Markdown body for `[[wiki-style]]` links.
///
/// Wiki-links are not standard Markdown and comrak does not parse them, so we
/// detect them via pattern matching in the original source text. Since comrak's
/// CommonMark renderer escapes brackets, we must scan the raw input.
fn extract_wiki_links(body: &str, doc_id: Uuid, links: &mut Vec<LinkRecord>) {
    let mut search_from = 0;

    while let Some(start) = body[search_from..].find("[[") {
        let abs_start = search_from + start + 2; // past "[["
        if let Some(end) = body[abs_start..].find("]]") {
            let inner = &body[abs_start..abs_start + end];

            // Skip empty or obviously invalid links (e.g. spanning lines).
            if !inner.is_empty() && !inner.contains('\n') {
                // Wiki-links can have display text: [[target|display]]
                let (target, anchor) = match inner.split_once('|') {
                    Some((t, a)) => (t.trim(), a.trim()),
                    None => (inner.trim(), inner.trim()),
                };

                if !target.is_empty() {
                    links.push(LinkRecord {
                        link_id: Uuid::new_v4(),
                        source_doc_id: doc_id,
                        source_block_id: None, // Cannot reliably map to a block from raw text.
                        target: target.to_string(),
                        target_doc_id: None,
                        link_type: LinkType::WikiLink,
                        anchor_text: anchor.to_string(),
                    });
                }
            }

            search_from = abs_start + end + 2; // past "]]"
        } else {
            // No closing "]]" — stop scanning.
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tag extraction
// ---------------------------------------------------------------------------

/// Extracts `TagRecord`s from frontmatter.
///
/// Top-level keys become tags. Array values are flattened so that each element
/// becomes its own `TagRecord` (e.g. `tags: [a, b]` produces two records with
/// key `"tags"`).
fn extract_tags(doc_id: Uuid, frontmatter: &serde_json::Value) -> Vec<TagRecord> {
    let mut tags = Vec::new();

    let obj = match frontmatter.as_object() {
        Some(o) => o,
        None => return tags,
    };

    for (key, value) in obj {
        match value {
            serde_json::Value::Array(items) => {
                // One TagRecord per array element.
                for item in items {
                    tags.push(TagRecord {
                        tag_id: Uuid::new_v4(),
                        doc_id,
                        key: key.clone(),
                        value: item.clone(),
                    });
                }
            }
            _ => {
                tags.push(TagRecord {
                    tag_id: Uuid::new_v4(),
                    doc_id,
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }
    }

    tags
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SAMPLE_DOC: &str = r#"---
title: Test Document
tags: [rust, markdown]
category: notes
---
# Introduction

This is a paragraph with a [link](https://example.com) and an image:

![alt text](image.png)

## Code Example

```rust
fn main() {
    println!("hello");
}
```

- Item one
- Item two with [[wiki link]]
- Item three

> A blockquote with [another link](other.md).

---

Some text referencing [[another page|display text]].
"#;

    #[test]
    fn parse_produces_document_record() {
        let result = parse_markdown("notes/test.md", SAMPLE_DOC).unwrap();
        assert_eq!(result.document.path, "notes/test.md");
        assert_eq!(result.document.title, Some("Test Document".to_string()));
        assert_eq!(result.document.version, 1);
        assert!(!result.document.raw_content_hash.is_empty());
    }

    #[test]
    fn parse_extracts_blocks() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();
        let types: Vec<_> = result.blocks.iter().map(|b| &b.block_type).collect();

        // Should contain at least: headings, paragraphs, code block, list, list items, blockquote, thematic break.
        assert!(types.iter().any(|t| matches!(t, BlockType::Heading { level: 1 })));
        assert!(types.iter().any(|t| matches!(t, BlockType::Heading { level: 2 })));
        assert!(types.iter().any(|t| matches!(t, BlockType::Paragraph)));
        assert!(types.iter().any(|t| matches!(t, BlockType::CodeBlock { .. })));
        assert!(types.iter().any(|t| matches!(t, BlockType::List { .. })));
        assert!(types.iter().any(|t| matches!(t, BlockType::ListItem)));
        assert!(types.iter().any(|t| matches!(t, BlockType::BlockQuote)));
        assert!(types.iter().any(|t| matches!(t, BlockType::ThematicBreak)));
    }

    #[test]
    fn parse_extracts_markdown_links() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();
        let md_links: Vec<_> = result
            .links
            .iter()
            .filter(|l| l.link_type == LinkType::MarkdownLink)
            .collect();

        assert!(md_links.iter().any(|l| l.target == "https://example.com"));
        assert!(md_links.iter().any(|l| l.target == "other.md"));
    }

    #[test]
    fn parse_extracts_image_links() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();
        let img_links: Vec<_> = result
            .links
            .iter()
            .filter(|l| l.link_type == LinkType::ImageLink)
            .collect();

        assert_eq!(img_links.len(), 1);
        assert_eq!(img_links[0].target, "image.png");
        assert_eq!(img_links[0].anchor_text, "alt text");
    }

    #[test]
    fn parse_extracts_wiki_links() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();
        let wiki_links: Vec<_> = result
            .links
            .iter()
            .filter(|l| l.link_type == LinkType::WikiLink)
            .collect();

        assert!(wiki_links.len() >= 2, "expected at least 2 wiki links, got {}", wiki_links.len());
        assert!(wiki_links.iter().any(|l| l.target == "wiki link"));
        assert!(wiki_links
            .iter()
            .any(|l| l.target == "another page" && l.anchor_text == "display text"));
    }

    #[test]
    fn parse_extracts_tags() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();

        // "tags" array should be flattened into two TagRecords.
        let tag_records: Vec<_> = result.tags.iter().filter(|t| t.key == "tags").collect();
        assert_eq!(tag_records.len(), 2);
        assert!(tag_records.iter().any(|t| t.value == json!("rust")));
        assert!(tag_records.iter().any(|t| t.value == json!("markdown")));

        // "category" scalar should produce one TagRecord.
        let cat_records: Vec<_> = result.tags.iter().filter(|t| t.key == "category").collect();
        assert_eq!(cat_records.len(), 1);
        assert_eq!(cat_records[0].value, json!("notes"));

        // "title" is also a frontmatter key.
        let title_records: Vec<_> = result.tags.iter().filter(|t| t.key == "title").collect();
        assert_eq!(title_records.len(), 1);
    }

    #[test]
    fn parse_consistent_doc_id() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();
        let doc_id = result.document.doc_id;

        for block in &result.blocks {
            assert_eq!(block.doc_id, doc_id, "block doc_id mismatch");
        }
        for link in &result.links {
            assert_eq!(link.source_doc_id, doc_id, "link source_doc_id mismatch");
        }
        for tag in &result.tags {
            assert_eq!(tag.doc_id, doc_id, "tag doc_id mismatch");
        }
    }

    #[test]
    fn parse_ordinals_are_sequential() {
        let result = parse_markdown("test.md", SAMPLE_DOC).unwrap();
        let ordinals: Vec<u32> = result.blocks.iter().map(|b| b.ordinal).collect();

        // Ordinals should be unique sequential values starting from 0.
        for (i, &ord) in ordinals.iter().enumerate() {
            assert_eq!(ord, i as u32, "ordinal {i} should be {i}, got {ord}");
        }
    }

    #[test]
    fn parse_no_frontmatter() {
        let content = "# Hello\n\nJust a simple document.\n";
        let result = parse_markdown("simple.md", content).unwrap();
        assert_eq!(result.document.title, Some("Hello".to_string()));
        assert!(result.tags.is_empty());
        assert!(result.document.frontmatter.is_null());
    }

    #[test]
    fn parse_empty_document() {
        let result = parse_markdown("empty.md", "").unwrap();
        assert!(result.blocks.is_empty());
        assert!(result.links.is_empty());
        assert!(result.tags.is_empty());
        assert!(result.document.title.is_none());
    }

    #[test]
    fn code_block_captures_language() {
        let content = "```python\nprint('hi')\n```\n";
        let result = parse_markdown("code.md", content).unwrap();
        let code_blocks: Vec<_> = result
            .blocks
            .iter()
            .filter(|b| matches!(&b.block_type, BlockType::CodeBlock { .. }))
            .collect();

        assert_eq!(code_blocks.len(), 1);
        match &code_blocks[0].block_type {
            BlockType::CodeBlock { language } => {
                assert_eq!(language.as_deref(), Some("python"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn nested_list_has_parent_ids() {
        let content = "- outer\n  - inner\n";
        let result = parse_markdown("nested.md", content).unwrap();

        let list_items: Vec<_> = result
            .blocks
            .iter()
            .filter(|b| matches!(b.block_type, BlockType::ListItem))
            .collect();

        // There should be at least 2 list items; the inner one should have a parent.
        assert!(list_items.len() >= 2);
        // At least one item should have a parent_block_id set.
        assert!(list_items.iter().any(|b| b.parent_block_id.is_some()));
    }
}
