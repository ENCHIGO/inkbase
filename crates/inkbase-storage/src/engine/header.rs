//! File header for the custom storage engine data file.
//!
//! The header occupies the first 64 bytes of the file and stores
//! global metadata needed to bootstrap the engine on open.
//!
//! # Layout (64 bytes, little-endian)
//!
//! | Offset | Size | Field            |
//! |--------|------|------------------|
//! |   0    |  8   | magic            |
//! |   8    |  4   | version (u32)    |
//! |  12    |  4   | page_count (u32) |
//! |  16    |  4   | free_list_head   |
//! |  20    |  4   | root_btree_page  |
//! |  24    | 40   | _reserved        |

use super::page::PageId;

pub const MAGIC: &[u8; 8] = b"MDOTADB\0";
pub const HEADER_SIZE: usize = 64;
pub const CURRENT_VERSION: u32 = 1;

/// Offsets within the 64-byte header.
mod offsets {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 8;
    pub const PAGE_COUNT: usize = 12;
    pub const FREE_LIST_HEAD: usize = 16;
    pub const ROOT_BTREE_PAGE: usize = 20;
}

/// Metadata header stored at the beginning of the data file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    pub version: u32,
    pub page_count: u32,
    pub free_list_head: PageId,
    pub root_btree_page: PageId,
}

impl FileHeader {
    /// Create a new header with default values for a freshly created file.
    pub fn new() -> Self {
        Self {
            version: CURRENT_VERSION,
            page_count: 0,
            free_list_head: 0,
            root_btree_page: 0,
        }
    }

    /// Serialize the header into a 64-byte array.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        buf[offsets::MAGIC..offsets::MAGIC + 8].copy_from_slice(MAGIC);
        buf[offsets::VERSION..offsets::VERSION + 4]
            .copy_from_slice(&self.version.to_le_bytes());
        buf[offsets::PAGE_COUNT..offsets::PAGE_COUNT + 4]
            .copy_from_slice(&self.page_count.to_le_bytes());
        buf[offsets::FREE_LIST_HEAD..offsets::FREE_LIST_HEAD + 4]
            .copy_from_slice(&self.free_list_head.to_le_bytes());
        buf[offsets::ROOT_BTREE_PAGE..offsets::ROOT_BTREE_PAGE + 4]
            .copy_from_slice(&self.root_btree_page.to_le_bytes());

        buf
    }

    /// Deserialize a header from a 64-byte slice.
    ///
    /// Returns an error if the magic bytes are incorrect or the version
    /// is unsupported.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> inkbase_core::Result<Self> {
        // Validate magic.
        if &buf[offsets::MAGIC..offsets::MAGIC + 8] != MAGIC.as_slice() {
            return Err(inkbase_core::InkbaseError::StorageError(
                "invalid file header: bad magic bytes".to_string(),
            ));
        }

        let version = u32::from_le_bytes(
            buf[offsets::VERSION..offsets::VERSION + 4]
                .try_into()
                .unwrap(),
        );

        if version > CURRENT_VERSION {
            return Err(inkbase_core::InkbaseError::StorageError(format!(
                "unsupported file version {version} (max supported: {CURRENT_VERSION})"
            )));
        }

        let page_count = u32::from_le_bytes(
            buf[offsets::PAGE_COUNT..offsets::PAGE_COUNT + 4]
                .try_into()
                .unwrap(),
        );

        let free_list_head = u32::from_le_bytes(
            buf[offsets::FREE_LIST_HEAD..offsets::FREE_LIST_HEAD + 4]
                .try_into()
                .unwrap(),
        );

        let root_btree_page = u32::from_le_bytes(
            buf[offsets::ROOT_BTREE_PAGE..offsets::ROOT_BTREE_PAGE + 4]
                .try_into()
                .unwrap(),
        );

        Ok(Self {
            version,
            page_count,
            free_list_head,
            root_btree_page,
        })
    }
}

impl Default for FileHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = FileHeader {
            version: CURRENT_VERSION,
            page_count: 42,
            free_list_head: 7,
            root_btree_page: 3,
        };

        let bytes = header.to_bytes();
        let restored = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, restored);
    }

    #[test]
    fn header_default_roundtrip() {
        let header = FileHeader::new();
        let bytes = header.to_bytes();
        let restored = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(restored.page_count, 0);
        assert_eq!(restored.version, CURRENT_VERSION);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = FileHeader::new().to_bytes();
        bytes[0] = b'X'; // corrupt the magic
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("bad magic"));
    }

    #[test]
    fn future_version_is_rejected() {
        let mut header = FileHeader::new();
        header.version = CURRENT_VERSION + 1;
        let bytes = header.to_bytes();
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("unsupported file version"));
    }
}
