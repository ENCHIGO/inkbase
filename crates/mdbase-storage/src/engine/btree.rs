//! On-disk B+Tree index built on top of the buffer pool.
//!
//! Stores `(key: Vec<u8>, value: Vec<u8>)` pairs in sorted order, supporting
//! point lookups, range scans, prefix scans, insertions, and deletions.
//!
//! # Page Layout
//!
//! **Leaf nodes** (`PageType::BTreeLeaf`):
//! - Each slotted-page record: `[key_len: u16][key bytes][value bytes]`
//! - Records are kept in sorted order by key within the page
//! - `next_page` links to the next leaf for range scans
//!
//! **Internal nodes** (`PageType::BTreeInternal`):
//! - Each record: `[key_len: u16][key bytes][child_page_id: u32]`
//! - `next_page` stores the rightmost child pointer
//! - Separator semantics: all keys in child\[i\] < separator\[i\] <= all keys in child\[i+1\]

use std::sync::Arc;

use super::buffer_pool::BufferPool;
use super::page::{Page, PageId, PageType, SLOT_SIZE};

// ---------------------------------------------------------------------------
// Record encoding helpers
// ---------------------------------------------------------------------------

/// Encode a leaf record: `[key_len: u16][key][value]`
fn encode_leaf_record(key: &[u8], value: &[u8]) -> Vec<u8> {
    let key_len = key.len() as u16;
    let mut buf = Vec::with_capacity(2 + key.len() + value.len());
    buf.extend_from_slice(&key_len.to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);
    buf
}

/// Decode a leaf record into `(key, value)`.
fn decode_leaf_record(record: &[u8]) -> (&[u8], &[u8]) {
    let key_len = u16::from_le_bytes([record[0], record[1]]) as usize;
    let key = &record[2..2 + key_len];
    let value = &record[2 + key_len..];
    (key, value)
}

/// Encode an internal record: `[key_len: u16][key][child_page_id: u32]`
fn encode_internal_record(key: &[u8], child_page_id: PageId) -> Vec<u8> {
    let key_len = key.len() as u16;
    let mut buf = Vec::with_capacity(2 + key.len() + 4);
    buf.extend_from_slice(&key_len.to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&child_page_id.to_le_bytes());
    buf
}

/// Decode an internal record into `(key, child_page_id)`.
fn decode_internal_record(record: &[u8]) -> (&[u8], PageId) {
    let key_len = u16::from_le_bytes([record[0], record[1]]) as usize;
    let key = &record[2..2 + key_len];
    let child_page_id = u32::from_le_bytes(
        record[2 + key_len..2 + key_len + 4]
            .try_into()
            .unwrap(),
    );
    (key, child_page_id)
}

// ---------------------------------------------------------------------------
// Page-level helpers
// ---------------------------------------------------------------------------

/// Read all live (non-deleted) records from a page, returning `(slot_index, record_bytes)`.
fn read_all_records(page: &Page) -> Vec<Vec<u8>> {
    let mut records = Vec::new();
    for slot in 0..page.num_slots() {
        if let Some(data) = page.get_record(slot) {
            records.push(data.to_vec());
        }
    }
    records
}

/// Rebuild a page from scratch with the given records, preserving its page_id,
/// page_type, and next_page. Returns the rebuilt page.
fn rebuild_page_with_records(
    page_id: PageId,
    page_type: PageType,
    next_page: PageId,
    records: &[Vec<u8>],
) -> Page {
    let mut page = Page::new(page_id, page_type);
    page.set_next_page(next_page);
    for record in records {
        page.insert_record(record)
            .expect("rebuild_page_with_records: records must fit in a single page");
    }
    page
}

/// Find the position where `key` should be inserted into a sorted leaf page.
/// Returns the slot index of the first record whose key >= `key`, or
/// `num_live_records` if `key` is larger than all existing keys.
fn leaf_search_position(records: &[Vec<u8>], key: &[u8]) -> usize {
    records
        .iter()
        .position(|rec| {
            let (k, _) = decode_leaf_record(rec);
            k >= key
        })
        .unwrap_or(records.len())
}

/// Find the child index for a key in an internal node.
/// Returns the index `i` such that the key belongs in the subtree rooted
/// at child `i`. Children are numbered 0..=num_separators, where child
/// `num_separators` is the rightmost child stored in `next_page`.
fn internal_search_child(page: &Page, key: &[u8]) -> (usize, PageId) {
    let n = page.num_slots();
    for slot in 0..n {
        if let Some(data) = page.get_record(slot) {
            let (sep_key, child_id) = decode_internal_record(data);
            if key < sep_key {
                return (slot as usize, child_id);
            }
        }
    }
    // Key is >= all separators, go to the rightmost child.
    (n as usize, page.next_page())
}

// ---------------------------------------------------------------------------
// BTree
// ---------------------------------------------------------------------------

pub struct BTree {
    root_page_id: PageId,
    pool: Arc<BufferPool>,
}

/// Result of attempting to insert into a subtree. If the child split,
/// the caller must insert the separator into the parent.
enum InsertResult {
    /// Insertion completed without splitting this node.
    Done,
    /// This node split. The caller must insert `(separator_key, new_page_id)`
    /// into the parent. `new_page_id` contains all keys >= separator_key.
    Split {
        separator: Vec<u8>,
        new_page_id: PageId,
    },
}

impl BTree {
    /// Create a new empty B+Tree, allocating a root leaf page.
    pub fn new(pool: Arc<BufferPool>) -> mdbase_core::Result<Self> {
        let root = pool.allocate_page(PageType::BTreeLeaf)?;
        let root_page_id = root.page_id();
        pool.write_page(root)?;
        Ok(Self {
            root_page_id,
            pool,
        })
    }

    /// Open an existing B+Tree with a known root page.
    pub fn open(pool: Arc<BufferPool>, root_page_id: PageId) -> Self {
        Self {
            root_page_id,
            pool,
        }
    }

    /// Get the root page ID (for persisting in the file header).
    pub fn root_page_id(&self) -> PageId {
        self.root_page_id
    }

    /// Insert a key-value pair. If the key already exists, update the value.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> mdbase_core::Result<()> {
        let result = self.insert_recursive(self.root_page_id, key, value)?;

        if let InsertResult::Split {
            separator,
            new_page_id,
        } = result
        {
            // The root split. Create a new root internal node.
            let old_root_id = self.root_page_id;
            let mut new_root = self.pool.allocate_page(PageType::BTreeInternal)?;

            // The single separator points to old_root (left child).
            // The rightmost child (next_page) points to new_page.
            let record = encode_internal_record(&separator, old_root_id);
            new_root
                .insert_record(&record)
                .expect("fresh internal page must fit one record");
            new_root.set_next_page(new_page_id);

            self.root_page_id = new_root.page_id();
            self.pool.write_page(new_root)?;
        }

        Ok(())
    }

    /// Look up a value by exact key.
    pub fn get(&self, key: &[u8]) -> mdbase_core::Result<Option<Vec<u8>>> {
        let leaf_id = self.find_leaf(key)?;
        let page = self.pool.get_page(leaf_id)?;

        for slot in 0..page.num_slots() {
            if let Some(data) = page.get_record(slot) {
                let (k, v) = decode_leaf_record(data);
                match k.cmp(key) {
                    std::cmp::Ordering::Equal => return Ok(Some(v.to_vec())),
                    std::cmp::Ordering::Greater => return Ok(None),
                    std::cmp::Ordering::Less => {}
                }
            }
        }

        Ok(None)
    }

    /// Delete a key. Returns true if the key existed.
    pub fn delete(&mut self, key: &[u8]) -> mdbase_core::Result<bool> {
        let leaf_id = self.find_leaf(key)?;
        let mut page = self.pool.get_page(leaf_id)?;

        for slot in 0..page.num_slots() {
            if let Some(data) = page.get_record(slot) {
                let (k, _) = decode_leaf_record(data);
                match k.cmp(key) {
                    std::cmp::Ordering::Equal => {
                        page.delete_record(slot);
                        self.pool.write_page(page)?;
                        return Ok(true);
                    }
                    std::cmp::Ordering::Greater => return Ok(false),
                    std::cmp::Ordering::Less => {}
                }
            }
        }

        Ok(false)
    }

    /// Scan all key-value pairs with keys in `[start, end)`.
    /// If `start` is `None`, scan from the beginning.
    /// If `end` is `None`, scan to the end.
    pub fn range_scan(
        &self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> mdbase_core::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Find the starting leaf.
        let leaf_id = match start {
            Some(start_key) => self.find_leaf(start_key)?,
            None => self.find_leftmost_leaf()?,
        };

        let mut results = Vec::new();
        let mut current_page_id = leaf_id;

        loop {
            let page = self.pool.get_page(current_page_id)?;

            for slot in 0..page.num_slots() {
                if let Some(data) = page.get_record(slot) {
                    let (k, v) = decode_leaf_record(data);

                    // Skip keys before the start bound.
                    if let Some(s) = start {
                        if k < s {
                            continue;
                        }
                    }

                    // Stop at the end bound.
                    if let Some(e) = end {
                        if k >= e {
                            return Ok(results);
                        }
                    }

                    results.push((k.to_vec(), v.to_vec()));
                }
            }

            let next = page.next_page();
            if next == 0 {
                break;
            }
            current_page_id = next;
        }

        Ok(results)
    }

    /// Scan all keys with a given prefix.
    pub fn prefix_scan(&self, prefix: &[u8]) -> mdbase_core::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        if prefix.is_empty() {
            // Empty prefix matches everything.
            return self.range_scan(None, None);
        }

        // Compute the exclusive upper bound: the smallest key that is greater
        // than all keys sharing this prefix. This is the prefix with its last
        // byte incremented, handling carry (e.g., [0xFF] rolls to [0x01, 0x00]).
        let end = prefix_successor(prefix);

        self.range_scan(Some(prefix), end.as_deref())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Walk from root to find the leaf page that should contain `key`.
    fn find_leaf(&self, key: &[u8]) -> mdbase_core::Result<PageId> {
        let mut page_id = self.root_page_id;

        loop {
            let page = self.pool.get_page(page_id)?;
            match page.page_type() {
                PageType::BTreeLeaf => return Ok(page_id),
                PageType::BTreeInternal => {
                    let (_idx, child_id) = internal_search_child(&page, key);
                    page_id = child_id;
                }
                other => {
                    return Err(mdbase_core::MdbaseError::StorageError(format!(
                        "unexpected page type {:?} during B+Tree traversal",
                        other
                    )));
                }
            }
        }
    }

    /// Find the leftmost leaf by always following the first child pointer.
    fn find_leftmost_leaf(&self) -> mdbase_core::Result<PageId> {
        let mut page_id = self.root_page_id;

        loop {
            let page = self.pool.get_page(page_id)?;
            match page.page_type() {
                PageType::BTreeLeaf => return Ok(page_id),
                PageType::BTreeInternal => {
                    // The first child is stored in record slot 0.
                    if let Some(data) = page.get_record(0) {
                        let (_key, child_id) = decode_internal_record(data);
                        page_id = child_id;
                    } else {
                        // Internal node with no separator records — only has
                        // the rightmost child in next_page (degenerate case).
                        page_id = page.next_page();
                    }
                }
                other => {
                    return Err(mdbase_core::MdbaseError::StorageError(format!(
                        "unexpected page type {:?} during B+Tree traversal",
                        other
                    )));
                }
            }
        }
    }

    /// Recursively insert into the subtree rooted at `page_id`.
    fn insert_recursive(
        &mut self,
        page_id: PageId,
        key: &[u8],
        value: &[u8],
    ) -> mdbase_core::Result<InsertResult> {
        let page = self.pool.get_page(page_id)?;

        match page.page_type() {
            PageType::BTreeLeaf => self.insert_into_leaf(page, key, value),
            PageType::BTreeInternal => self.insert_into_internal(page, key, value),
            other => Err(mdbase_core::MdbaseError::StorageError(format!(
                "unexpected page type {:?} during B+Tree insert",
                other
            ))),
        }
    }

    /// Insert a key-value pair into a leaf page. Handles updates, sorted
    /// insertion, and splits.
    fn insert_into_leaf(
        &mut self,
        page: Page,
        key: &[u8],
        value: &[u8],
    ) -> mdbase_core::Result<InsertResult> {
        let page_id = page.page_id();
        let next_page = page.next_page();

        let mut records = read_all_records(&page);
        let pos = leaf_search_position(&records, key);

        // Check if this is an update (key already exists at position `pos`).
        let is_update = pos < records.len() && {
            let (k, _) = decode_leaf_record(&records[pos]);
            k == key
        };

        let new_record = encode_leaf_record(key, value);

        if is_update {
            records[pos] = new_record;
        } else {
            records.insert(pos, new_record);
        }

        // Try to fit all records into the existing page.
        if records_fit_in_page(&records) {
            let rebuilt = rebuild_page_with_records(
                page_id,
                PageType::BTreeLeaf,
                next_page,
                &records,
            );
            self.pool.write_page(rebuilt)?;
            return Ok(InsertResult::Done);
        }

        // Need to split. Divide records roughly in half.
        let mid = records.len() / 2;
        let left_records = &records[..mid];
        let right_records = &records[mid..];

        // The separator is the first key in the right page.
        let (separator_key, _) = decode_leaf_record(&right_records[0]);
        let separator = separator_key.to_vec();

        // Allocate a new page for the right half.
        let new_leaf = self.pool.allocate_page(PageType::BTreeLeaf)?;
        let new_leaf_id = new_leaf.page_id();

        // Left page keeps its page_id, points next_page to new leaf.
        let left_page = rebuild_page_with_records(
            page_id,
            PageType::BTreeLeaf,
            new_leaf_id,
            left_records,
        );

        // Right page links to whatever the old page's next_page was.
        let right_page = rebuild_page_with_records(
            new_leaf_id,
            PageType::BTreeLeaf,
            next_page,
            right_records,
        );

        self.pool.write_page(left_page)?;
        self.pool.write_page(right_page)?;

        Ok(InsertResult::Split {
            separator,
            new_page_id: new_leaf_id,
        })
    }

    /// Insert into an internal node: descend to the correct child, then
    /// handle any split that propagates up.
    fn insert_into_internal(
        &mut self,
        page: Page,
        key: &[u8],
        value: &[u8],
    ) -> mdbase_core::Result<InsertResult> {
        let (_idx, child_id) = internal_search_child(&page, key);

        let child_result = self.insert_recursive(child_id, key, value)?;

        match child_result {
            InsertResult::Done => Ok(InsertResult::Done),
            InsertResult::Split {
                separator,
                new_page_id,
            } => {
                // We need to insert a new separator into this internal page.
                // Re-read the page in case it was evicted from the buffer pool
                // during the recursive insert.
                let page = self.pool.get_page(page.page_id())?;
                self.insert_into_internal_node(page, &separator, new_page_id)
            }
        }
    }

    /// Insert a separator key and new-child pointer into an internal node.
    ///
    /// A child at some pointer position `j` has split into `(old_child, new_child_id)`
    /// with `separator` between them. We model the internal node as a flat pointer
    /// array interleaved with keys:
    ///
    /// ```text
    /// ptr_0, key_0, ptr_1, key_1, ..., key_{n-1}, ptr_n
    /// ```
    ///
    /// In our on-disk encoding: `record[i] = (key_i, ptr_i)`, `next_page = ptr_n`.
    ///
    /// The split of `ptr_j` transforms the sequence from:
    ///   `..., ptr_j, key_j, ...`
    /// to:
    ///   `..., ptr_j (old_child), separator, new_child_id, key_j, ...`
    ///
    /// So we insert `(separator, old_child)` at position `j` and update
    /// `entries[j+1].ptr` (or `next_page`) to point to `new_child_id`.
    fn insert_into_internal_node(
        &mut self,
        page: Page,
        separator: &[u8],
        new_child_id: PageId,
    ) -> mdbase_core::Result<InsertResult> {
        let page_id = page.page_id();
        let rightmost_child = page.next_page();

        // Collect existing entries as (separator_key, left_child_id).
        let raw_records = read_all_records(&page);
        let mut entries: Vec<(Vec<u8>, PageId)> = raw_records
            .iter()
            .map(|rec| {
                let (k, child) = decode_internal_record(rec);
                (k.to_vec(), child)
            })
            .collect();

        // Find insertion position: first entry whose key > separator.
        let pos = entries
            .iter()
            .position(|(k, _)| separator.as_ref() < k.as_slice())
            .unwrap_or(entries.len());

        // The old child at position `pos` was split. In the pointer array
        // interpretation:
        //   If pos < entries.len(): the left-child of entries[pos] was split.
        //     We insert (separator, old_child) at pos, and entries[pos] now has
        //     new_child_id as its left-child.
        //   If pos == entries.len(): the rightmost child (next_page) was split.
        //     We append (separator, old_child) and set next_page = new_child_id.
        if pos < entries.len() {
            let old_child = entries[pos].1;
            entries.insert(pos, (separator.to_vec(), old_child));
            entries[pos + 1].1 = new_child_id;
        } else {
            // Split of the rightmost child.
            entries.push((separator.to_vec(), rightmost_child));
        }

        // Determine the new rightmost child.
        let new_rightmost = if pos < entries.len() - 1 {
            // pos was not at the end — rightmost child is unchanged
            // (unless it was the original rightmost that split, handled above).
            rightmost_child
        } else {
            // The new entry was appended at the end.
            new_child_id
        };

        // Re-encode all entries.
        let encoded: Vec<Vec<u8>> = entries
            .iter()
            .map(|(k, c)| encode_internal_record(k, *c))
            .collect();

        // Try to fit into the existing page.
        if records_fit_in_page(&encoded) {
            let rebuilt = rebuild_page_with_records(
                page_id,
                PageType::BTreeInternal,
                new_rightmost,
                &encoded,
            );
            self.pool.write_page(rebuilt)?;
            return Ok(InsertResult::Done);
        }

        // Split the internal node.
        let mid = entries.len() / 2;

        // The separator that goes up to the parent is entries[mid].key.
        // Left page: entries[0..mid], rightmost = entries[mid].child
        // Right page: entries[mid+1..], rightmost = new_rightmost
        let promote_key = entries[mid].0.clone();
        let left_rightmost = entries[mid].1;

        let left_encoded: Vec<Vec<u8>> = entries[..mid]
            .iter()
            .map(|(k, c)| encode_internal_record(k, *c))
            .collect();
        let right_encoded: Vec<Vec<u8>> = entries[mid + 1..]
            .iter()
            .map(|(k, c)| encode_internal_record(k, *c))
            .collect();

        let new_internal = self.pool.allocate_page(PageType::BTreeInternal)?;
        let new_internal_id = new_internal.page_id();

        let left_page = rebuild_page_with_records(
            page_id,
            PageType::BTreeInternal,
            left_rightmost,
            &left_encoded,
        );
        let right_page = rebuild_page_with_records(
            new_internal_id,
            PageType::BTreeInternal,
            new_rightmost,
            &right_encoded,
        );

        self.pool.write_page(left_page)?;
        self.pool.write_page(right_page)?;

        Ok(InsertResult::Split {
            separator: promote_key,
            new_page_id: new_internal_id,
        })
    }
}

/// Check whether a set of records can fit into a single fresh page.
/// Accounts for the slot array overhead per record.
fn records_fit_in_page(records: &[Vec<u8>]) -> bool {
    use super::page::{PAGE_HEADER_SIZE, PAGE_SIZE};
    let total: usize = records.iter().map(|r| SLOT_SIZE + r.len()).sum();
    total <= PAGE_SIZE - PAGE_HEADER_SIZE
}

/// Compute the smallest byte string that is strictly greater than all strings
/// sharing the given prefix. Returns `None` if the prefix is all `0xFF` bytes
/// (meaning every possible key starts with this prefix — scan to the end).
fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut succ = prefix.to_vec();
    // Increment the last byte, with carry.
    while let Some(last) = succ.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(succ);
        }
        succ.pop();
    }
    // All bytes were 0xFF — no upper bound.
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn temp_btree() -> (BTree, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("btree_test.db");
        let pool = Arc::new(
            BufferPool::open(&path, 1024).unwrap(),
        );
        let btree = BTree::new(pool).unwrap();
        (btree, dir)
    }

    #[test]
    fn insert_and_get_single_key() {
        let (mut btree, _dir) = temp_btree();
        btree.insert(b"hello", b"world").unwrap();

        let val = btree.get(b"hello").unwrap();
        assert_eq!(val, Some(b"world".to_vec()));
    }

    #[test]
    fn insert_multiple_keys_verify_all_found() {
        let (mut btree, _dir) = temp_btree();

        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..50)
            .map(|i| (format!("key_{:04}", i).into_bytes(), format!("val_{}", i).into_bytes()))
            .collect();

        for (k, v) in &pairs {
            btree.insert(k, v).unwrap();
        }

        for (k, v) in &pairs {
            let found = btree.get(k).unwrap();
            assert_eq!(found.as_ref(), Some(v), "missing key {:?}", String::from_utf8_lossy(k));
        }
    }

    #[test]
    fn insert_enough_keys_to_trigger_leaf_split() {
        let (mut btree, _dir) = temp_btree();

        // Each record is ~100 bytes of key + value + 2 byte key_len + 4 byte slot.
        // A page is 8192 bytes with 24-byte header => ~8168 usable.
        // At ~106 bytes/record, a page fits ~77 records.
        // Insert 200 keys to ensure at least one split.
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..200)
            .map(|i| {
                (
                    format!("key_{:04}", i).into_bytes(),
                    vec![b'v'; 90],
                )
            })
            .collect();

        for (k, v) in &pairs {
            btree.insert(k, v).unwrap();
        }

        // Verify all keys are still retrievable.
        for (k, v) in &pairs {
            let found = btree.get(k).unwrap();
            assert_eq!(
                found.as_ref(),
                Some(v),
                "key {:?} missing after split",
                String::from_utf8_lossy(k)
            );
        }

        // The root should no longer be a leaf (it must have split).
        let root = btree.pool.get_page(btree.root_page_id()).unwrap();
        assert_eq!(root.page_type(), PageType::BTreeInternal);
    }

    #[test]
    fn delete_key() {
        let (mut btree, _dir) = temp_btree();

        btree.insert(b"alpha", b"1").unwrap();
        btree.insert(b"beta", b"2").unwrap();
        btree.insert(b"gamma", b"3").unwrap();

        // Delete an existing key.
        assert!(btree.delete(b"beta").unwrap());
        assert_eq!(btree.get(b"beta").unwrap(), None);

        // Other keys still present.
        assert_eq!(btree.get(b"alpha").unwrap(), Some(b"1".to_vec()));
        assert_eq!(btree.get(b"gamma").unwrap(), Some(b"3".to_vec()));

        // Delete a non-existent key.
        assert!(!btree.delete(b"delta").unwrap());
    }

    #[test]
    fn range_scan() {
        let (mut btree, _dir) = temp_btree();

        for i in 0..20u32 {
            let key = format!("k_{:03}", i).into_bytes();
            let val = format!("v_{}", i).into_bytes();
            btree.insert(&key, &val).unwrap();
        }

        // Scan [k_005, k_010)
        let results = btree.range_scan(Some(b"k_005"), Some(b"k_010")).unwrap();
        let keys: Vec<String> = results
            .iter()
            .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
            .collect();
        assert_eq!(keys, vec!["k_005", "k_006", "k_007", "k_008", "k_009"]);

        // Scan from beginning to k_003
        let results = btree.range_scan(None, Some(b"k_003")).unwrap();
        let keys: Vec<String> = results
            .iter()
            .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
            .collect();
        assert_eq!(keys, vec!["k_000", "k_001", "k_002"]);

        // Scan from k_018 to end
        let results = btree.range_scan(Some(b"k_018"), None).unwrap();
        let keys: Vec<String> = results
            .iter()
            .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
            .collect();
        assert_eq!(keys, vec!["k_018", "k_019"]);

        // Scan everything
        let results = btree.range_scan(None, None).unwrap();
        assert_eq!(results.len(), 20);
    }

    #[test]
    fn prefix_scan() {
        let (mut btree, _dir) = temp_btree();

        btree.insert(b"doc:001:title", b"Hello").unwrap();
        btree.insert(b"doc:001:body", b"World").unwrap();
        btree.insert(b"doc:002:title", b"Foo").unwrap();
        btree.insert(b"doc:002:body", b"Bar").unwrap();
        btree.insert(b"tag:rust", b"1").unwrap();

        let results = btree.prefix_scan(b"doc:001:").unwrap();
        let keys: Vec<String> = results
            .iter()
            .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
            .collect();
        // Sorted order: "body" < "title"
        assert_eq!(keys, vec!["doc:001:body", "doc:001:title"]);

        let results = btree.prefix_scan(b"doc:").unwrap();
        assert_eq!(results.len(), 4);

        let results = btree.prefix_scan(b"tag:").unwrap();
        assert_eq!(results.len(), 1);

        let results = btree.prefix_scan(b"missing:").unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn update_existing_key() {
        let (mut btree, _dir) = temp_btree();

        btree.insert(b"key", b"original").unwrap();
        assert_eq!(btree.get(b"key").unwrap(), Some(b"original".to_vec()));

        btree.insert(b"key", b"updated").unwrap();
        assert_eq!(btree.get(b"key").unwrap(), Some(b"updated".to_vec()));

        // Should still be one entry, not two.
        let results = btree.range_scan(None, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn get_nonexistent_key_returns_none() {
        let (btree, _dir) = temp_btree();
        assert_eq!(btree.get(b"does_not_exist").unwrap(), None);
    }

    #[test]
    fn large_insert_and_scan_preserves_order() {
        let (mut btree, _dir) = temp_btree();

        // Insert 1000 keys in reverse order to stress the sorted insertion logic.
        let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..1000)
            .map(|i| {
                (
                    format!("k_{:06}", i).into_bytes(),
                    format!("v_{:06}", i).into_bytes(),
                )
            })
            .collect();

        // Insert in reverse order.
        for (k, v) in pairs.iter().rev() {
            btree.insert(k, v).unwrap();
        }

        // Range scan should return keys in sorted order.
        let results = btree.range_scan(None, None).unwrap();
        assert_eq!(results.len(), 1000);

        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for (i, (k, v)) in results.iter().enumerate() {
            assert_eq!(k, &pairs[i].0, "key mismatch at position {}", i);
            assert_eq!(v, &pairs[i].1, "value mismatch at position {}", i);
        }
    }

    #[test]
    fn delete_after_split() {
        let (mut btree, _dir) = temp_btree();

        // Insert enough to cause splits, then delete from various leaves.
        for i in 0..300u32 {
            let key = format!("k_{:04}", i).into_bytes();
            btree.insert(&key, b"val").unwrap();
        }

        // Delete every other key.
        for i in (0..300u32).step_by(2) {
            let key = format!("k_{:04}", i).into_bytes();
            assert!(btree.delete(&key).unwrap());
        }

        // Verify remaining keys.
        for i in 0..300u32 {
            let key = format!("k_{:04}", i).into_bytes();
            if i % 2 == 0 {
                assert_eq!(btree.get(&key).unwrap(), None);
            } else {
                assert_eq!(btree.get(&key).unwrap(), Some(b"val".to_vec()));
            }
        }
    }

    #[test]
    fn prefix_successor_edge_cases() {
        assert_eq!(prefix_successor(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(prefix_successor(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_successor(b"\xff\xff"), None);
        assert_eq!(prefix_successor(b"\x00"), Some(b"\x01".to_vec()));
    }

    #[test]
    fn empty_prefix_scan_returns_all() {
        let (mut btree, _dir) = temp_btree();
        btree.insert(b"a", b"1").unwrap();
        btree.insert(b"b", b"2").unwrap();
        btree.insert(b"c", b"3").unwrap();

        let results = btree.prefix_scan(b"").unwrap();
        assert_eq!(results.len(), 3);
    }
}
