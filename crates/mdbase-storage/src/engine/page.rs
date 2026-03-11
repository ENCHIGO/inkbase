//! 8KB slotted page format for the custom storage engine.
//!
//! # On-disk layout
//!
//! ```text
//! [Header (24 bytes)][Slot Array ->  ...  <- Record Data][Free Space]
//! ```
//!
//! - The **header** occupies bytes 0..24 and stores page metadata.
//! - The **slot array** grows forward from byte 24. Each slot is 4 bytes
//!   (offset: u16, length: u16) pointing to a record's location within
//!   the page.
//! - **Record data** grows backward from the end of the page.
//! - **Free space** is the gap between the last slot entry and the first
//!   (lowest-offset) record.
//!
//! All multi-byte integers are stored in **little-endian** byte order.

pub const PAGE_SIZE: usize = 8192; // 8 KB
pub const PAGE_HEADER_SIZE: usize = 24;
pub const SLOT_SIZE: usize = 4; // offset (u16) + length (u16)

/// Sentinel value indicating a deleted slot.
const DELETED_SLOT_OFFSET: u16 = 0;

/// Page types stored in the header byte at offset 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    Free = 0,
    Data = 1,
    BTreeInternal = 2,
    BTreeLeaf = 3,
    Overflow = 4,
}

impl PageType {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => PageType::Free,
            1 => PageType::Data,
            2 => PageType::BTreeInternal,
            3 => PageType::BTreeLeaf,
            4 => PageType::Overflow,
            _ => PageType::Free, // treat unknown as free
        }
    }
}

pub type PageId = u32;

/// Header field offsets (all sizes in bytes):
///
/// | Offset | Size | Field              |
/// |--------|------|--------------------|
/// |   0    |  4   | page_id (u32)      |
/// |   4    |  1   | page_type (u8)     |
/// |   5    |  3   | _reserved          |
/// |   8    |  2   | num_slots (u16)    |
/// |  10    |  2   | free_space_end (u16) — end of free region (= start of record data) |
/// |  12    |  4   | next_page (u32)    |
/// |  16    |  8   | lsn (u64)          |
/// Total: 24 bytes
mod offsets {
    pub const PAGE_ID: usize = 0;
    pub const PAGE_TYPE: usize = 4;
    // 5..8 reserved
    pub const NUM_SLOTS: usize = 8;
    pub const FREE_SPACE_END: usize = 10;
    pub const NEXT_PAGE: usize = 12;
    pub const LSN: usize = 16;
}

/// A fixed-size 8 KB page that lives in the buffer pool.
///
/// All reads and writes go through accessor methods that interpret the
/// underlying byte array using little-endian encoding.
pub struct Page {
    pub data: [u8; PAGE_SIZE],
}

impl Page {
    /// Create a new, empty page with the given id and type.
    ///
    /// The slot array is empty and `free_space_end` starts at the end of
    /// the page, meaning the entire body (after the header) is available.
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        let mut data = [0u8; PAGE_SIZE];

        // page_id
        data[offsets::PAGE_ID..offsets::PAGE_ID + 4]
            .copy_from_slice(&page_id.to_le_bytes());

        // page_type
        data[offsets::PAGE_TYPE] = page_type as u8;

        // num_slots = 0
        data[offsets::NUM_SLOTS..offsets::NUM_SLOTS + 2]
            .copy_from_slice(&0u16.to_le_bytes());

        // free_space_end starts at the very end of the page — no records yet
        let end = PAGE_SIZE as u16;
        data[offsets::FREE_SPACE_END..offsets::FREE_SPACE_END + 2]
            .copy_from_slice(&end.to_le_bytes());

        // next_page = 0 (null sentinel)
        data[offsets::NEXT_PAGE..offsets::NEXT_PAGE + 4]
            .copy_from_slice(&0u32.to_le_bytes());

        // lsn = 0
        data[offsets::LSN..offsets::LSN + 8]
            .copy_from_slice(&0u64.to_le_bytes());

        Self { data }
    }

    // ------------------------------------------------------------------
    // Header field accessors
    // ------------------------------------------------------------------

    pub fn page_id(&self) -> PageId {
        u32::from_le_bytes(
            self.data[offsets::PAGE_ID..offsets::PAGE_ID + 4]
                .try_into()
                .unwrap(),
        )
    }

    pub fn page_type(&self) -> PageType {
        PageType::from_u8(self.data[offsets::PAGE_TYPE])
    }

    pub fn num_slots(&self) -> u16 {
        u16::from_le_bytes(
            self.data[offsets::NUM_SLOTS..offsets::NUM_SLOTS + 2]
                .try_into()
                .unwrap(),
        )
    }

    fn set_num_slots(&mut self, n: u16) {
        self.data[offsets::NUM_SLOTS..offsets::NUM_SLOTS + 2]
            .copy_from_slice(&n.to_le_bytes());
    }

    /// The byte offset where the lowest record begins (i.e. the end of
    /// free space). Records are packed downward from `PAGE_SIZE`.
    fn free_space_end(&self) -> u16 {
        u16::from_le_bytes(
            self.data[offsets::FREE_SPACE_END..offsets::FREE_SPACE_END + 2]
                .try_into()
                .unwrap(),
        )
    }

    fn set_free_space_end(&mut self, offset: u16) {
        self.data[offsets::FREE_SPACE_END..offsets::FREE_SPACE_END + 2]
            .copy_from_slice(&offset.to_le_bytes());
    }

    pub fn next_page(&self) -> PageId {
        u32::from_le_bytes(
            self.data[offsets::NEXT_PAGE..offsets::NEXT_PAGE + 4]
                .try_into()
                .unwrap(),
        )
    }

    pub fn set_next_page(&mut self, next: PageId) {
        self.data[offsets::NEXT_PAGE..offsets::NEXT_PAGE + 4]
            .copy_from_slice(&next.to_le_bytes());
    }

    pub fn lsn(&self) -> u64 {
        u64::from_le_bytes(
            self.data[offsets::LSN..offsets::LSN + 8]
                .try_into()
                .unwrap(),
        )
    }

    pub fn set_lsn(&mut self, lsn: u64) {
        self.data[offsets::LSN..offsets::LSN + 8]
            .copy_from_slice(&lsn.to_le_bytes());
    }

    // ------------------------------------------------------------------
    // Slot array helpers
    // ------------------------------------------------------------------

    /// Byte offset of the start of the slot array (immediately after
    /// the header).
    fn slot_array_start(&self) -> usize {
        PAGE_HEADER_SIZE
    }

    /// Byte offset of the slot entry at the given index.
    fn slot_offset_for(&self, slot: u16) -> usize {
        self.slot_array_start() + (slot as usize) * SLOT_SIZE
    }

    /// Byte offset immediately after the last slot entry — this is
    /// the start of free space.
    fn free_space_start(&self) -> usize {
        self.slot_array_start() + (self.num_slots() as usize) * SLOT_SIZE
    }

    /// Read the (offset, length) pair stored in a slot entry.
    fn read_slot(&self, slot: u16) -> (u16, u16) {
        let base = self.slot_offset_for(slot);
        let off = u16::from_le_bytes(
            self.data[base..base + 2].try_into().unwrap(),
        );
        let len = u16::from_le_bytes(
            self.data[base + 2..base + 4].try_into().unwrap(),
        );
        (off, len)
    }

    /// Write the (offset, length) pair into a slot entry.
    fn write_slot(&mut self, slot: u16, offset: u16, length: u16) {
        let base = self.slot_offset_for(slot);
        self.data[base..base + 2].copy_from_slice(&offset.to_le_bytes());
        self.data[base + 2..base + 4].copy_from_slice(&length.to_le_bytes());
    }

    // ------------------------------------------------------------------
    // Public record operations
    // ------------------------------------------------------------------

    /// Amount of free space available for a new record (including its
    /// slot entry overhead).
    pub fn free_space(&self) -> usize {
        let start = self.free_space_start();
        let end = self.free_space_end() as usize;
        if end > start {
            end - start
        } else {
            0
        }
    }

    /// Insert a record into the page.
    ///
    /// Returns the slot index on success, or `None` if there is not
    /// enough space for both the new slot entry and the record data.
    ///
    /// Records are placed at the end of the page and grow downward;
    /// the slot array grows upward from just after the header.
    pub fn insert_record(&mut self, record: &[u8]) -> Option<u16> {
        let record_len = record.len();
        let needed = SLOT_SIZE + record_len;

        if self.free_space() < needed {
            return None;
        }

        // Place the record just below the current free_space_end.
        let new_free_end = self.free_space_end() as usize - record_len;
        self.data[new_free_end..new_free_end + record_len]
            .copy_from_slice(record);

        // Append a new slot entry.
        let slot_idx = self.num_slots();
        self.write_slot(slot_idx, new_free_end as u16, record_len as u16);
        self.set_num_slots(slot_idx + 1);
        self.set_free_space_end(new_free_end as u16);

        Some(slot_idx)
    }

    /// Retrieve the record data for a given slot index.
    ///
    /// Returns `None` if the slot index is out of range or the slot has
    /// been deleted.
    pub fn get_record(&self, slot: u16) -> Option<&[u8]> {
        if slot >= self.num_slots() {
            return None;
        }
        let (offset, length) = self.read_slot(slot);
        if offset == DELETED_SLOT_OFFSET && length == 0 {
            return None;
        }
        let start = offset as usize;
        let end = start + length as usize;
        if end > PAGE_SIZE {
            return None;
        }
        Some(&self.data[start..end])
    }

    /// Mark a slot as deleted by zeroing its offset and length.
    ///
    /// The record data is not physically removed — space is reclaimed
    /// only during page compaction (not yet implemented).
    pub fn delete_record(&mut self, slot: u16) {
        if slot >= self.num_slots() {
            return;
        }
        self.write_slot(slot, DELETED_SLOT_OFFSET, 0);
    }

    // ------------------------------------------------------------------
    // Byte-level conversion
    // ------------------------------------------------------------------

    /// Wrap an existing byte array as a `Page`.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self { data }
    }

    /// Borrow the underlying byte array (e.g. for writing to disk).
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }
}

impl Clone for Page {
    fn clone(&self) -> Self {
        Self { data: self.data }
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("page_id", &self.page_id())
            .field("page_type", &self.page_type())
            .field("num_slots", &self.num_slots())
            .field("free_space", &self.free_space())
            .field("next_page", &self.next_page())
            .field("lsn", &self.lsn())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_page_has_correct_header() {
        let page = Page::new(42, PageType::Data);
        assert_eq!(page.page_id(), 42);
        assert_eq!(page.page_type(), PageType::Data);
        assert_eq!(page.num_slots(), 0);
        assert_eq!(page.next_page(), 0);
        assert_eq!(page.lsn(), 0);
        assert_eq!(page.free_space_end(), PAGE_SIZE as u16);
    }

    #[test]
    fn free_space_of_empty_page() {
        let page = Page::new(0, PageType::Data);
        // All space between end-of-header and end-of-page is free.
        assert_eq!(page.free_space(), PAGE_SIZE - PAGE_HEADER_SIZE);
    }

    #[test]
    fn insert_and_get_single_record() {
        let mut page = Page::new(1, PageType::Data);
        let payload = b"hello, slotted page!";

        let slot = page.insert_record(payload).expect("should fit");
        assert_eq!(slot, 0);
        assert_eq!(page.num_slots(), 1);

        let record = page.get_record(0).expect("should exist");
        assert_eq!(record, payload);
    }

    #[test]
    fn insert_multiple_records() {
        let mut page = Page::new(1, PageType::Data);

        let a = b"record A";
        let b = b"record B is a bit longer";
        let c = b"C";

        let s0 = page.insert_record(a).unwrap();
        let s1 = page.insert_record(b).unwrap();
        let s2 = page.insert_record(c).unwrap();

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(page.num_slots(), 3);

        assert_eq!(page.get_record(0).unwrap(), a.as_slice());
        assert_eq!(page.get_record(1).unwrap(), b.as_slice());
        assert_eq!(page.get_record(2).unwrap(), c.as_slice());
    }

    #[test]
    fn delete_record_returns_none() {
        let mut page = Page::new(1, PageType::Data);
        page.insert_record(b"will be deleted").unwrap();
        page.insert_record(b"should survive").unwrap();

        page.delete_record(0);

        assert!(page.get_record(0).is_none());
        assert_eq!(
            page.get_record(1).unwrap(),
            b"should survive".as_slice()
        );
    }

    #[test]
    fn insert_fails_when_page_full() {
        let mut page = Page::new(1, PageType::Data);
        // Fill with large records until insertion fails.
        let big = vec![0xABu8; 1000];
        let mut count = 0u16;
        while page.insert_record(&big).is_some() {
            count += 1;
        }
        // We should have inserted some records before running out.
        assert!(count > 0);
        assert_eq!(page.num_slots(), count);
    }

    #[test]
    fn free_space_accounts_for_slot_and_record() {
        let mut page = Page::new(1, PageType::Data);
        let initial = page.free_space();

        let payload = b"12345678"; // 8 bytes
        page.insert_record(payload).unwrap();

        // Should have consumed SLOT_SIZE (4) + 8 = 12 bytes.
        assert_eq!(page.free_space(), initial - SLOT_SIZE - 8);
    }

    #[test]
    fn roundtrip_through_bytes() {
        let mut page = Page::new(7, PageType::BTreeLeaf);
        page.insert_record(b"persisted data").unwrap();
        page.set_next_page(99);
        page.set_lsn(12345);

        let bytes = *page.as_bytes();
        let restored = Page::from_bytes(bytes);

        assert_eq!(restored.page_id(), 7);
        assert_eq!(restored.page_type(), PageType::BTreeLeaf);
        assert_eq!(restored.num_slots(), 1);
        assert_eq!(restored.next_page(), 99);
        assert_eq!(restored.lsn(), 12345);
        assert_eq!(
            restored.get_record(0).unwrap(),
            b"persisted data".as_slice()
        );
    }

    #[test]
    fn get_record_out_of_range() {
        let page = Page::new(1, PageType::Data);
        assert!(page.get_record(0).is_none());
        assert!(page.get_record(100).is_none());
    }

    #[test]
    fn delete_out_of_range_is_noop() {
        let mut page = Page::new(1, PageType::Data);
        // Should not panic.
        page.delete_record(0);
        page.delete_record(999);
    }

    #[test]
    fn page_type_roundtrip() {
        for &pt in &[
            PageType::Free,
            PageType::Data,
            PageType::BTreeInternal,
            PageType::BTreeLeaf,
            PageType::Overflow,
        ] {
            let page = Page::new(0, pt);
            assert_eq!(page.page_type(), pt);
        }
    }

    #[test]
    fn insert_exact_remaining_space() {
        let mut page = Page::new(1, PageType::Data);

        // Calculate how large a record we can fit given current free space
        // minus one slot entry.
        let available = page.free_space();
        let max_record = available - SLOT_SIZE;
        let payload = vec![0x42u8; max_record];

        assert!(page.insert_record(&payload).is_some());
        // Now the page should be completely full.
        assert_eq!(page.free_space(), 0);
        // One more byte should not fit.
        assert!(page.insert_record(&[0]).is_none());
    }
}
