//! Custom page-based storage engine.
//!
//! This module implements the low-level storage primitives: fixed-size
//! slotted pages, an LRU buffer pool, and the on-disk file format.
//! Higher layers build B-tree indexes and record management on top of
//! these primitives.

pub mod btree;
pub mod buffer_pool;
pub mod header;
pub mod mvcc;
pub mod page;
pub mod storage_engine;
pub mod wal;
