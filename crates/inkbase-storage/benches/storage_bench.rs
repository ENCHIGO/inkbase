use std::sync::Arc;

use chrono::Utc;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use uuid::Uuid;

use inkbase_core::types::DocumentRecord;
use inkbase_storage::engine::btree::BTree;
use inkbase_storage::engine::buffer_pool::BufferPool;
use inkbase_storage::{CustomStorageEngine, SledStorage, Storage};

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

fn make_doc(i: usize) -> DocumentRecord {
    DocumentRecord {
        doc_id: Uuid::new_v4(),
        path: format!("bench/doc_{:04}.md", i),
        title: Some(format!("Document {}", i)),
        frontmatter: serde_json::json!({}),
        raw_content_hash: format!("{:064x}", i),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        version: 1,
    }
}

fn make_docs(n: usize) -> Vec<DocumentRecord> {
    (0..n).map(make_doc).collect()
}

// ---------------------------------------------------------------------------
// Sled benchmarks
// ---------------------------------------------------------------------------

fn bench_sled_insert_document(c: &mut Criterion) {
    let mut group = c.benchmark_group("sled_insert_document");
    let n = 100;
    group.throughput(Throughput::Elements(n));

    group.bench_function("100_docs", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let storage = SledStorage::new(dir.path()).unwrap();
                let docs = make_docs(n as usize);
                (storage, docs, dir)
            },
            |(storage, docs, _dir)| {
                for doc in &docs {
                    storage.insert_document(doc).unwrap();
                }
            },
        );
    });

    group.finish();
}

fn bench_sled_get_document(c: &mut Criterion) {
    let mut group = c.benchmark_group("sled_get_document");
    let n = 100;
    group.throughput(Throughput::Elements(n));

    group.bench_function("100_docs", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let storage = SledStorage::new(dir.path()).unwrap();
                let docs = make_docs(n as usize);
                for doc in &docs {
                    storage.insert_document(doc).unwrap();
                }
                let ids: Vec<Uuid> = docs.iter().map(|d| d.doc_id).collect();
                (storage, ids, dir)
            },
            |(storage, ids, _dir)| {
                for id in &ids {
                    storage.get_document(id).unwrap();
                }
            },
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// CustomStorageEngine benchmarks
// ---------------------------------------------------------------------------

fn bench_custom_insert_document(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_insert_document");
    let n = 100;
    group.throughput(Throughput::Elements(n));

    group.bench_function("100_docs", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let engine = CustomStorageEngine::new(dir.path()).unwrap();
                let docs = make_docs(n as usize);
                (engine, docs, dir)
            },
            |(engine, docs, _dir)| {
                for doc in &docs {
                    engine.insert_document(doc).unwrap();
                }
            },
        );
    });

    group.finish();
}

fn bench_custom_get_document(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_get_document");
    let n = 100;
    group.throughput(Throughput::Elements(n));

    group.bench_function("100_docs", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let engine = CustomStorageEngine::new(dir.path()).unwrap();
                let docs = make_docs(n as usize);
                for doc in &docs {
                    engine.insert_document(doc).unwrap();
                }
                let ids: Vec<Uuid> = docs.iter().map(|d| d.doc_id).collect();
                (engine, ids, dir)
            },
            |(engine, ids, _dir)| {
                for id in &ids {
                    engine.get_document(id).unwrap();
                }
            },
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Raw B+Tree benchmarks
// ---------------------------------------------------------------------------

fn bench_btree_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_insert");
    let n: u64 = 1000;
    group.throughput(Throughput::Elements(n));

    group.bench_function("1000_keys", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("btree_bench.db");
                let pool = Arc::new(BufferPool::open(&path, 2048).unwrap());
                let btree = BTree::new(pool).unwrap();
                let keys: Vec<Vec<u8>> = (0..n as usize)
                    .map(|i| format!("key_{:06}", i).into_bytes())
                    .collect();
                let value = vec![b'v'; 64];
                (btree, keys, value, dir)
            },
            |(mut btree, keys, value, _dir)| {
                for key in &keys {
                    btree.insert(key, &value).unwrap();
                }
            },
        );
    });

    group.finish();
}

fn bench_btree_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_get");
    let n: u64 = 1000;
    group.throughput(Throughput::Elements(n));

    group.bench_function("1000_keys", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("btree_bench.db");
                let pool = Arc::new(BufferPool::open(&path, 2048).unwrap());
                let mut btree = BTree::new(pool).unwrap();
                let keys: Vec<Vec<u8>> = (0..n as usize)
                    .map(|i| format!("key_{:06}", i).into_bytes())
                    .collect();
                let value = vec![b'v'; 64];
                for key in &keys {
                    btree.insert(key, &value).unwrap();
                }
                (btree, keys, dir)
            },
            |(btree, keys, _dir)| {
                for key in &keys {
                    btree.get(key).unwrap();
                }
            },
        );
    });

    group.finish();
}

fn bench_btree_range_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_range_scan");
    // We scan 100 keys out of 1000.
    group.throughput(Throughput::Elements(100));

    group.bench_function("100_of_1000", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("btree_bench.db");
                let pool = Arc::new(BufferPool::open(&path, 2048).unwrap());
                let mut btree = BTree::new(pool).unwrap();
                let value = vec![b'v'; 64];
                for i in 0..1000usize {
                    let key = format!("key_{:06}", i).into_bytes();
                    btree.insert(&key, &value).unwrap();
                }
                // Scan keys 450..550 (100 keys in the middle).
                let start = format!("key_{:06}", 450).into_bytes();
                let end = format!("key_{:06}", 550).into_bytes();
                (btree, start, end, dir)
            },
            |(btree, start, end, _dir)| {
                btree.range_scan(Some(&start), Some(&end)).unwrap();
            },
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sled_insert_document,
    bench_sled_get_document,
    bench_custom_insert_document,
    bench_custom_get_document,
    bench_btree_insert,
    bench_btree_get,
    bench_btree_range_scan,
);
criterion_main!(benches);
