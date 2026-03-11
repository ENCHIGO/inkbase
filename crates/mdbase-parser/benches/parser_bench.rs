use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mdbase_parser::parse_markdown;

// ---------------------------------------------------------------------------
// Test documents of varying complexity
// ---------------------------------------------------------------------------

const SMALL_DOC: &str = "# Hello\n\nA paragraph.\n";

const MEDIUM_DOC: &str = r#"---
title: Benchmark Document
tags: [rust, performance, benchmarks]
category: engineering
status: published
---
# Introduction

This is the **introduction** section of a reasonably complex Markdown document.
It contains [links](https://example.com), *emphasis*, and `inline code`.

## Background

The background section provides context for the reader. Here is a list:

- First item with a [[wiki link]]
- Second item with [an external link](https://docs.rs)
- Third item with `code` inline
- Fourth item, plain text

## Code Example

```rust
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
```

## Data Table

| Column A | Column B | Column C |
|----------|----------|----------|
| 1        | alpha    | true     |
| 2        | beta     | false    |
| 3        | gamma    | true     |

## Nested Lists

1. Ordered item one
   - Nested unordered A
   - Nested unordered B
2. Ordered item two
   1. Sub-ordered 2.1
   2. Sub-ordered 2.2
3. Ordered item three

> A blockquote with **bold** and a [link](other.md).
>
> Second paragraph in the blockquote.

---

## Footnotes

This sentence has a footnote[^1]. And another[^2].

[^1]: This is the first footnote.
[^2]: This is the second footnote with [a link](https://example.org).

## Conclusion

Final paragraph referencing [[another page|display text]] and
![an image](assets/photo.png).
"#;

/// Generate a synthetic Markdown document with the given number of sections.
///
/// Each section contains a heading, a paragraph, a short code block, and
/// a bullet list, producing roughly 8-10 AST nodes per section.
fn generate_large_doc(sections: usize) -> String {
    let mut doc = String::with_capacity(sections * 300);
    doc.push_str("---\ntitle: Large Document\ntags: [bench, generated]\n---\n\n");
    for i in 0..sections {
        doc.push_str(&format!("## Section {}\n\n", i));
        doc.push_str(&format!(
            "This is paragraph {} with a [link](https://example.com/{}) \
             and a [[wiki link {}]]. It has *emphasis* and `code`.\n\n",
            i, i, i
        ));
        doc.push_str(&format!(
            "```python\ndef func_{}():\n    return {}\n```\n\n",
            i, i
        ));
        doc.push_str(&format!(
            "- Item {}.1 with **bold**\n- Item {}.2 with [link](page_{}.md)\n- Item {}.3\n\n",
            i, i, i, i
        ));
    }
    doc
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_markdown");

    group.bench_function("small", |b| {
        b.iter(|| parse_markdown("test.md", SMALL_DOC))
    });

    group.bench_function("medium", |b| {
        b.iter(|| parse_markdown("test.md", MEDIUM_DOC))
    });

    for size in [10, 50, 100, 500] {
        let doc = generate_large_doc(size);
        group.bench_with_input(BenchmarkId::new("sections", size), &doc, |b, doc| {
            b.iter(|| parse_markdown("test.md", doc))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
