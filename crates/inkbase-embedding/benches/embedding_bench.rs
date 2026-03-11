use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use uuid::Uuid;

use inkbase_embedding::{EmbeddingPipeline, EmbeddingProvider, TfIdfEmbedder};

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const PARAGRAPH: &str = "\
The Rust programming language helps developers write reliable and efficient \
software. Its ownership model guarantees memory safety without a garbage \
collector, while zero-cost abstractions ensure high performance. Rust is \
increasingly used for systems programming, web services, and embedded devices.";

/// Generate a batch of distinct paragraphs for embedding.
fn generate_paragraphs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "Section {} discusses topic {} in detail. \
                 Key concepts include abstraction, concurrency, and safety. \
                 The implementation leverages pattern matching and trait-based generics \
                 to achieve both correctness and performance for use case {}.",
                i, i * 7 + 3, i
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// TF-IDF embedding benchmarks
// ---------------------------------------------------------------------------

fn bench_tfidf_embed_single(c: &mut Criterion) {
    let embedder = TfIdfEmbedder::default();

    c.bench_function("tfidf_embed_single", |b| {
        b.iter(|| embedder.embed(PARAGRAPH).unwrap())
    });
}

fn bench_tfidf_embed_batch(c: &mut Criterion) {
    let embedder = TfIdfEmbedder::default();
    let mut group = c.benchmark_group("tfidf_embed_batch");

    for size in [10, 100] {
        let paragraphs = generate_paragraphs(size);
        let refs: Vec<&str> = paragraphs.iter().map(|s| s.as_str()).collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &refs, |b, refs| {
            b.iter(|| embedder.embed_batch(refs).unwrap())
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Pipeline benchmarks (embed_text and embed_blocks)
// ---------------------------------------------------------------------------

fn bench_pipeline_embed_text(c: &mut Criterion) {
    let pipeline = EmbeddingPipeline::default_tfidf();

    c.bench_function("pipeline_embed_text", |b| {
        b.iter(|| pipeline.embed_text(PARAGRAPH).unwrap())
    });
}

fn bench_pipeline_embed_blocks(c: &mut Criterion) {
    use inkbase_core::types::{BlockRecord, BlockType};

    let pipeline = EmbeddingPipeline::default_tfidf();
    let mut group = c.benchmark_group("pipeline_embed_blocks");

    for size in [10, 100] {
        let paragraphs = generate_paragraphs(size);
        let blocks: Vec<BlockRecord> = paragraphs
            .iter()
            .enumerate()
            .map(|(i, text)| BlockRecord {
                block_id: Uuid::new_v4(),
                doc_id: Uuid::new_v4(),
                block_type: BlockType::Paragraph,
                depth: 0,
                ordinal: i as u32,
                parent_block_id: None,
                text_content: text.clone(),
                raw_markdown: text.clone(),
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &blocks,
            |b, blocks| b.iter(|| pipeline.embed_blocks(blocks).unwrap()),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Vector search benchmark
// ---------------------------------------------------------------------------

fn bench_vector_search(c: &mut Criterion) {
    use inkbase_index::VectorIndex;

    let dim = 256;
    let n = 1000;
    let embedder = TfIdfEmbedder::default();
    let index = VectorIndex::new(dim);

    // Pre-populate the index with 1000 vectors.
    let paragraphs = generate_paragraphs(n);
    for (i, text) in paragraphs.iter().enumerate() {
        let vec = embedder.embed(text).unwrap();
        index
            .insert(
                &format!("doc_{}", i),
                &format!("block_{}", i),
                &format!("bench/doc_{}.md", i),
                text,
                vec,
            )
            .unwrap();
    }

    let query_vec = embedder.embed(PARAGRAPH).unwrap();

    c.bench_function("vector_search_1000", |b| {
        b.iter(|| index.search(&query_vec, 10).unwrap())
    });
}

criterion_group!(
    benches,
    bench_tfidf_embed_single,
    bench_tfidf_embed_batch,
    bench_pipeline_embed_text,
    bench_pipeline_embed_blocks,
    bench_vector_search,
);
criterion_main!(benches);
