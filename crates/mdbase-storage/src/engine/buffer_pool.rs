//! LRU buffer pool managing pages in memory.
//!
//! The buffer pool sits between higher-level code and the data file on
//! disk. Pages are loaded lazily and evicted in LRU order when the pool
//! reaches capacity. Dirty pages are flushed to disk either explicitly
//! or when they need to be evicted.
//!
//! Thread safety is provided by a single `parking_lot::RwLock` around
//! the mutable inner state. This is adequate for moderate concurrency;
//! a more granular locking scheme can be introduced later if needed.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use parking_lot::RwLock;

use super::header::{FileHeader, HEADER_SIZE};
use super::page::{Page, PageId, PageType, PAGE_SIZE};

/// Default number of pages the buffer pool holds in memory.
pub const DEFAULT_CAPACITY: usize = 1024; // 8 MB at 8 KB per page

/// An LRU buffer pool that caches pages read from a data file.
pub struct BufferPool {
    inner: RwLock<BufferPoolInner>,
}

struct BufferPoolInner {
    file: File,
    header: FileHeader,
    pages: HashMap<PageId, CachedPage>,
    /// LRU tracking: the *front* of the Vec is the least recently used,
    /// the *back* is the most recently used.
    lru_order: Vec<PageId>,
    capacity: usize,
}

struct CachedPage {
    page: Page,
    dirty: bool,
}

impl BufferPool {
    /// Open or create a data file at `path` with the given buffer pool
    /// capacity (in number of pages).
    ///
    /// If the file already exists, its header is read and validated.
    /// Otherwise a new file is created with a fresh header.
    pub fn open(path: &Path, capacity: usize) -> mdbase_core::Result<Self> {
        let file_exists = path.exists() && path.metadata().map(|m| m.len() > 0).unwrap_or(false);

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let header = if file_exists {
            let mut buf = [0u8; HEADER_SIZE];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut buf)?;
            FileHeader::from_bytes(&buf)?
        } else {
            let header = FileHeader::new();
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;
            file.flush()?;
            header
        };

        Ok(Self {
            inner: RwLock::new(BufferPoolInner {
                file,
                header,
                pages: HashMap::new(),
                lru_order: Vec::new(),
                capacity,
            }),
        })
    }

    /// Read a page from the pool, loading it from disk if necessary.
    ///
    /// The returned `Page` is a clone — mutations must be written back
    /// via [`write_page`](Self::write_page).
    pub fn get_page(&self, page_id: PageId) -> mdbase_core::Result<Page> {
        let mut inner = self.inner.write();

        // Check cache first.
        if inner.pages.contains_key(&page_id) {
            inner.touch_lru(page_id);
            return Ok(inner.pages[&page_id].page.clone());
        }

        // Validate that the page_id is in range.
        if page_id >= inner.header.page_count {
            return Err(mdbase_core::MdbaseError::StorageError(format!(
                "page {page_id} out of range (page_count = {})",
                inner.header.page_count,
            )));
        }

        // Load from disk.
        let page = inner.read_page_from_disk(page_id)?;

        // Make room if needed.
        if inner.pages.len() >= inner.capacity {
            inner.evict_lru()?;
        }

        inner.pages.insert(
            page_id,
            CachedPage {
                page: page.clone(),
                dirty: false,
            },
        );
        inner.push_lru(page_id);

        Ok(page)
    }

    /// Write a page into the buffer pool, marking it as dirty.
    ///
    /// The page is not flushed to disk immediately — call [`flush`] or
    /// [`flush_page`] to persist.
    pub fn write_page(&self, page: Page) -> mdbase_core::Result<()> {
        let mut inner = self.inner.write();
        let page_id = page.page_id();

        if page_id >= inner.header.page_count {
            return Err(mdbase_core::MdbaseError::StorageError(format!(
                "cannot write page {page_id}: out of range (page_count = {})",
                inner.header.page_count,
            )));
        }

        if inner.pages.contains_key(&page_id) {
            inner.touch_lru(page_id);
            let cached = inner.pages.get_mut(&page_id).unwrap();
            cached.page = page;
            cached.dirty = true;
        } else {
            // Page not in cache — insert it.
            if inner.pages.len() >= inner.capacity {
                inner.evict_lru()?;
            }
            inner.pages.insert(page_id, CachedPage { page, dirty: true });
            inner.push_lru(page_id);
        }

        Ok(())
    }

    /// Allocate a new page of the given type.
    ///
    /// The page is added to the buffer pool (marked dirty) and the
    /// file header's page count is incremented.
    pub fn allocate_page(&self, page_type: PageType) -> mdbase_core::Result<Page> {
        let mut inner = self.inner.write();

        let page_id = inner.header.page_count;
        inner.header.page_count += 1;

        // Persist the updated header so the page count survives crashes.
        inner.write_header()?;

        let page = Page::new(page_id, page_type);

        // Make room if needed.
        if inner.pages.len() >= inner.capacity {
            inner.evict_lru()?;
        }

        inner.pages.insert(
            page_id,
            CachedPage {
                page: page.clone(),
                dirty: true,
            },
        );
        inner.push_lru(page_id);

        Ok(page)
    }

    /// Flush all dirty pages and the file header to disk.
    pub fn flush(&self) -> mdbase_core::Result<()> {
        let mut inner = self.inner.write();
        inner.flush_all()
    }

    /// Flush a single page to disk. No-op if the page is not cached
    /// or not dirty.
    pub fn flush_page(&self, page_id: PageId) -> mdbase_core::Result<()> {
        let mut inner = self.inner.write();
        inner.flush_single(page_id)
    }

    /// Return the total number of pages allocated in the file.
    pub fn page_count(&self) -> u32 {
        self.inner.read().header.page_count
    }
}

// ---------------------------------------------------------------------------
// BufferPoolInner — all methods assume the caller holds the lock.
// ---------------------------------------------------------------------------

impl BufferPoolInner {
    /// Compute the byte offset of a page within the data file.
    /// Pages start after the file header.
    fn page_offset(page_id: PageId) -> u64 {
        HEADER_SIZE as u64 + (page_id as u64) * (PAGE_SIZE as u64)
    }

    fn read_page_from_disk(&mut self, page_id: PageId) -> mdbase_core::Result<Page> {
        let offset = Self::page_offset(page_id);
        self.file.seek(SeekFrom::Start(offset))?;

        let mut buf = [0u8; PAGE_SIZE];
        self.file.read_exact(&mut buf)?;
        Ok(Page::from_bytes(buf))
    }

    fn write_header(&mut self) -> mdbase_core::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;
        Ok(())
    }

    fn flush_all(&mut self) -> mdbase_core::Result<()> {
        let dirty_ids: Vec<PageId> = self
            .pages
            .iter()
            .filter(|(_, cp)| cp.dirty)
            .map(|(&id, _)| id)
            .collect();

        for page_id in dirty_ids {
            // Borrow disjoint fields: &self.pages and &mut self.file.
            let cached = self.pages.get(&page_id).unwrap();
            let offset = Self::page_offset(cached.page.page_id());
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.write_all(cached.page.as_bytes())?;
            // Clear the dirty flag after successful write.
            self.pages.get_mut(&page_id).unwrap().dirty = false;
        }

        self.write_header()?;
        self.file.flush()?;
        Ok(())
    }

    fn flush_single(&mut self, page_id: PageId) -> mdbase_core::Result<()> {
        if let Some(cached) = self.pages.get(&page_id) {
            if cached.dirty {
                let offset = Self::page_offset(cached.page.page_id());
                self.file.seek(SeekFrom::Start(offset))?;
                self.file.write_all(cached.page.as_bytes())?;
                self.pages.get_mut(&page_id).unwrap().dirty = false;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // LRU management
    // ------------------------------------------------------------------

    /// Move `page_id` to the most-recently-used position (back of vec).
    fn touch_lru(&mut self, page_id: PageId) {
        if let Some(pos) = self.lru_order.iter().position(|&id| id == page_id) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push(page_id);
    }

    /// Add a page_id to the MRU end without checking for duplicates.
    fn push_lru(&mut self, page_id: PageId) {
        self.lru_order.push(page_id);
    }

    /// Evict the least recently used page. If the LRU candidate is
    /// dirty, flush it to disk first.
    fn evict_lru(&mut self) -> mdbase_core::Result<()> {
        // Try to find a clean page to evict first (starting from LRU end).
        let clean_pos = self
            .lru_order
            .iter()
            .position(|id| {
                self.pages
                    .get(id)
                    .map(|cp| !cp.dirty)
                    .unwrap_or(true)
            });

        let evict_idx = if let Some(pos) = clean_pos {
            pos
        } else if !self.lru_order.is_empty() {
            // All pages are dirty — evict the LRU one after flushing.
            0
        } else {
            return Ok(()); // nothing to evict
        };

        let page_id = self.lru_order.remove(evict_idx);

        if let Some(cached) = self.pages.get(&page_id) {
            if cached.dirty {
                let offset = Self::page_offset(cached.page.page_id());
                self.file.seek(SeekFrom::Start(offset))?;
                self.file.write_all(cached.page.as_bytes())?;
            }
        }
        self.pages.remove(&page_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_pool(capacity: usize) -> (BufferPool, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let pool = BufferPool::open(&path, capacity).unwrap();
        (pool, dir)
    }

    #[test]
    fn allocate_and_get_page() {
        let (pool, _dir) = temp_pool(DEFAULT_CAPACITY);

        let page = pool.allocate_page(PageType::Data).unwrap();
        assert_eq!(page.page_id(), 0);
        assert_eq!(page.page_type(), PageType::Data);
        assert_eq!(pool.page_count(), 1);

        let fetched = pool.get_page(0).unwrap();
        assert_eq!(fetched.page_id(), 0);
        assert_eq!(fetched.page_type(), PageType::Data);
    }

    #[test]
    fn write_and_read_back() {
        let (pool, _dir) = temp_pool(DEFAULT_CAPACITY);

        let mut page = pool.allocate_page(PageType::Data).unwrap();
        page.insert_record(b"hello engine").unwrap();

        pool.write_page(page).unwrap();

        let fetched = pool.get_page(0).unwrap();
        assert_eq!(
            fetched.get_record(0).unwrap(),
            b"hello engine".as_slice()
        );
    }

    #[test]
    fn flush_persists_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist.db");

        // Write data, flush, then reopen.
        {
            let pool = BufferPool::open(&path, DEFAULT_CAPACITY).unwrap();
            let mut page = pool.allocate_page(PageType::Data).unwrap();
            page.insert_record(b"persistent data").unwrap();
            pool.write_page(page).unwrap();
            pool.flush().unwrap();
        }

        // Reopen and verify.
        {
            let pool = BufferPool::open(&path, DEFAULT_CAPACITY).unwrap();
            assert_eq!(pool.page_count(), 1);
            let page = pool.get_page(0).unwrap();
            assert_eq!(
                page.get_record(0).unwrap(),
                b"persistent data".as_slice()
            );
        }
    }

    #[test]
    fn get_nonexistent_page_is_error() {
        let (pool, _dir) = temp_pool(DEFAULT_CAPACITY);
        let err = pool.get_page(0).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn write_out_of_range_page_is_error() {
        let (pool, _dir) = temp_pool(DEFAULT_CAPACITY);
        let page = Page::new(99, PageType::Data);
        let err = pool.write_page(page).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn eviction_under_pressure() {
        // Very small capacity: only 2 pages.
        let (pool, _dir) = temp_pool(2);

        // Allocate 3 pages — the third should evict the LRU.
        let p0 = pool.allocate_page(PageType::Data).unwrap();
        let p1 = pool.allocate_page(PageType::Data).unwrap();

        // Flush so p0 and p1 are clean (easier to evict).
        pool.flush().unwrap();

        let p2 = pool.allocate_page(PageType::Data).unwrap();

        assert_eq!(p0.page_id(), 0);
        assert_eq!(p1.page_id(), 1);
        assert_eq!(p2.page_id(), 2);
        assert_eq!(pool.page_count(), 3);

        // All three pages should still be readable (evicted pages
        // are reloaded from disk).
        assert_eq!(pool.get_page(0).unwrap().page_id(), 0);
        assert_eq!(pool.get_page(1).unwrap().page_id(), 1);
        assert_eq!(pool.get_page(2).unwrap().page_id(), 2);
    }

    #[test]
    fn multiple_pages_different_types() {
        let (pool, _dir) = temp_pool(DEFAULT_CAPACITY);

        let data = pool.allocate_page(PageType::Data).unwrap();
        let leaf = pool.allocate_page(PageType::BTreeLeaf).unwrap();
        let internal = pool.allocate_page(PageType::BTreeInternal).unwrap();
        let overflow = pool.allocate_page(PageType::Overflow).unwrap();

        assert_eq!(data.page_type(), PageType::Data);
        assert_eq!(leaf.page_type(), PageType::BTreeLeaf);
        assert_eq!(internal.page_type(), PageType::BTreeInternal);
        assert_eq!(overflow.page_type(), PageType::Overflow);
        assert_eq!(pool.page_count(), 4);
    }

    #[test]
    fn flush_page_selective() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("selective.db");

        {
            let pool = BufferPool::open(&path, DEFAULT_CAPACITY).unwrap();

            let mut p0 = pool.allocate_page(PageType::Data).unwrap();
            p0.insert_record(b"page zero").unwrap();
            pool.write_page(p0).unwrap();

            let mut p1 = pool.allocate_page(PageType::Data).unwrap();
            p1.insert_record(b"page one").unwrap();
            pool.write_page(p1).unwrap();

            // Only flush page 0.
            pool.flush_page(0).unwrap();
            // Also need to flush the header for page_count to survive.
            pool.flush().unwrap();
        }

        // Reopen and verify both pages survived (flush_all was called).
        {
            let pool = BufferPool::open(&path, DEFAULT_CAPACITY).unwrap();
            assert_eq!(pool.page_count(), 2);
            assert_eq!(
                pool.get_page(0).unwrap().get_record(0).unwrap(),
                b"page zero".as_slice()
            );
            assert_eq!(
                pool.get_page(1).unwrap().get_record(0).unwrap(),
                b"page one".as_slice()
            );
        }
    }
}
