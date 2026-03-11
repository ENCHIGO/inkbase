//! Write-Ahead Log (WAL) for crash-safe durability.
//!
//! Every modification is first written to the WAL before actual data pages are
//! modified. On crash recovery the WAL is replayed to reconstruct the latest
//! consistent state.
//!
//! ## Record format (on disk)
//!
//! ```text
//! [lsn: u64][record_type: u8][page_id: u32][data_len: u32][data: [u8; data_len]][checksum: u32]
//! ```
//!
//! The checksum is a CRC32 computed over all preceding fields of the record.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use mdbase_core::{MdbaseError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Log Sequence Number -- monotonically increasing identifier for each WAL record.
pub type Lsn = u64;

/// Maximum segment file size before rotation.
const WAL_SEGMENT_MAX_SIZE: u64 = 64 * 1024 * 1024; // 64 MB

/// Fixed overhead per record: lsn(8) + record_type(1) + page_id(4) + data_len(4) + checksum(4)
const RECORD_HEADER_SIZE: usize = 8 + 1 + 4 + 4 + 4;

/// Discriminant for the kind of WAL record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalRecordType {
    PageWrite = 1,
    Commit = 2,
    Checkpoint = 3,
}

impl WalRecordType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::PageWrite),
            2 => Some(Self::Commit),
            3 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

/// An in-memory representation of a single WAL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub record_type: WalRecordType,
    pub page_id: u32,
    pub data: Vec<u8>,
}

/// The write-ahead log.
///
/// Records are appended to *segment files* inside a directory.  When a segment
/// exceeds [`WAL_SEGMENT_MAX_SIZE`] a new segment is created.
pub struct Wal {
    dir: PathBuf,
    current_segment: BufWriter<File>,
    current_segment_id: u64,
    current_segment_size: u64,
    next_lsn: Lsn,
    /// Configurable segment size limit (exposed for testing).
    segment_max_size: u64,
}

// ---------------------------------------------------------------------------
// CRC32
// ---------------------------------------------------------------------------

/// CRC32 using the standard IEEE polynomial (0xEDB88320 reflected).
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// Segment file helpers
// ---------------------------------------------------------------------------

/// Build the filename for a given segment id: `wal_000001.log`, etc.
fn segment_filename(segment_id: u64) -> String {
    format!("wal_{:06}.log", segment_id)
}

/// Parse a segment id from a filename like `wal_000001.log`. Returns `None` if
/// the filename does not match the expected pattern.
fn parse_segment_id(name: &str) -> Option<u64> {
    let name = name.strip_prefix("wal_")?;
    let name = name.strip_suffix(".log")?;
    name.parse::<u64>().ok()
}

/// Return sorted list of `(segment_id, path)` for all segment files in `dir`.
fn list_segments(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut segments = Vec::new();
    if !dir.exists() {
        return Ok(segments);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if let Some(id) = parse_segment_id(&name) {
            segments.push((id, entry.path()));
        }
    }
    segments.sort_by_key(|(id, _)| *id);
    Ok(segments)
}

/// Open (or create) a segment file for appending and return a `BufWriter` plus
/// the current file size.
fn open_segment_for_append(path: &Path) -> Result<(BufWriter<File>, u64)> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    let size = file.metadata()?.len();
    Ok((BufWriter::new(file), size))
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a WAL record into bytes (including the trailing checksum).
fn serialize_record(record: &WalRecord) -> Vec<u8> {
    let total_len = RECORD_HEADER_SIZE + record.data.len();
    let mut buf = Vec::with_capacity(total_len);

    buf.extend_from_slice(&record.lsn.to_le_bytes());
    buf.push(record.record_type as u8);
    buf.extend_from_slice(&record.page_id.to_le_bytes());
    buf.extend_from_slice(&(record.data.len() as u32).to_le_bytes());
    buf.extend_from_slice(&record.data);

    // Checksum covers everything written so far (excludes the checksum itself).
    let checksum = crc32(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());

    buf
}

/// Attempt to read a single WAL record from `reader`. Returns:
///  - `Ok(Some(record))` on success
///  - `Ok(None)` on clean EOF (no more data)
///  - `Err(...)` on corrupt/partial records
fn read_record<R: Read>(reader: &mut R) -> Result<Option<WalRecord>> {
    // Read the fixed-size header portion: lsn + record_type + page_id + data_len
    let mut header = [0u8; 8 + 1 + 4 + 4]; // 17 bytes
    match reader.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }

    let lsn = u64::from_le_bytes(header[0..8].try_into().unwrap());
    let record_type_raw = header[8];
    let page_id = u32::from_le_bytes(header[9..13].try_into().unwrap());
    let data_len = u32::from_le_bytes(header[13..17].try_into().unwrap());

    let record_type = WalRecordType::from_u8(record_type_raw).ok_or_else(|| {
        MdbaseError::StorageError(format!(
            "invalid WAL record type {} at LSN {}",
            record_type_raw, lsn
        ))
    })?;

    // Guard against absurdly large data_len to avoid OOM on corrupt files.
    // A single page payload should never exceed the segment size.
    if data_len as u64 > WAL_SEGMENT_MAX_SIZE {
        return Err(MdbaseError::StorageError(format!(
            "WAL record at LSN {} claims data_len={}, exceeding segment max size",
            lsn, data_len
        )));
    }

    // Read data payload.
    let mut data = vec![0u8; data_len as usize];
    if let Err(e) = reader.read_exact(&mut data) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            // Partial write -- treat as tail corruption.
            return Err(MdbaseError::StorageError(format!(
                "truncated WAL record data at LSN {}",
                lsn
            )));
        }
        return Err(e.into());
    }

    // Read checksum.
    let mut checksum_bytes = [0u8; 4];
    if let Err(e) = reader.read_exact(&mut checksum_bytes) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Err(MdbaseError::StorageError(format!(
                "truncated WAL record checksum at LSN {}",
                lsn
            )));
        }
        return Err(e.into());
    }
    let stored_checksum = u32::from_le_bytes(checksum_bytes);

    // Recompute expected checksum over header + data.
    let mut check_buf = Vec::with_capacity(header.len() + data.len());
    check_buf.extend_from_slice(&header);
    check_buf.extend_from_slice(&data);
    let computed_checksum = crc32(&check_buf);

    if stored_checksum != computed_checksum {
        return Err(MdbaseError::StorageError(format!(
            "WAL checksum mismatch at LSN {}: stored={:#010x}, computed={:#010x}",
            lsn, stored_checksum, computed_checksum
        )));
    }

    Ok(Some(WalRecord {
        lsn,
        record_type,
        page_id,
        data,
    }))
}

// ---------------------------------------------------------------------------
// Wal implementation
// ---------------------------------------------------------------------------

impl Wal {
    /// Open (or initialize) the WAL in the given directory.
    ///
    /// If segment files already exist the latest one is opened for appending and
    /// the next LSN is determined by scanning forward from the last segment.  If
    /// the directory is empty a fresh segment is created.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_max_size(dir, WAL_SEGMENT_MAX_SIZE)
    }

    /// Like [`open`](Self::open) but with a configurable segment size limit
    /// (useful for tests that want to trigger rotation without writing 64 MB).
    pub fn open_with_max_size(dir: &Path, segment_max_size: u64) -> Result<Self> {
        fs::create_dir_all(dir)?;

        let segments = list_segments(dir)?;

        if segments.is_empty() {
            // Fresh WAL -- create the first segment.
            let segment_id = 1;
            let path = dir.join(segment_filename(segment_id));
            let (writer, size) = open_segment_for_append(&path)?;
            return Ok(Self {
                dir: dir.to_path_buf(),
                current_segment: writer,
                current_segment_id: segment_id,
                current_segment_size: size,
                next_lsn: 1,
                segment_max_size,
            });
        }

        // Scan all segments to discover the highest LSN.
        let mut max_lsn: Lsn = 0;
        for (_seg_id, path) in &segments {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            loop {
                match read_record(&mut reader) {
                    Ok(Some(rec)) => {
                        if rec.lsn > max_lsn {
                            max_lsn = rec.lsn;
                        }
                    }
                    Ok(None) => break,
                    // Tail corruption in the last segment is expected after a crash.
                    Err(_) => break,
                }
            }
        }

        // Open the last segment for further appending.
        let (last_id, last_path) = segments.last().unwrap();
        let (writer, size) = open_segment_for_append(last_path)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            current_segment: writer,
            current_segment_id: *last_id,
            current_segment_size: size,
            next_lsn: max_lsn + 1,
            segment_max_size,
        })
    }

    /// Append a WAL record and return the assigned LSN.
    ///
    /// The record is flushed and `fsync`-ed before returning to guarantee
    /// durability.  If the current segment would exceed the size limit a new
    /// segment is created first.
    pub fn append(
        &mut self,
        record_type: WalRecordType,
        page_id: u32,
        data: &[u8],
    ) -> Result<Lsn> {
        let lsn = self.next_lsn;

        let record = WalRecord {
            lsn,
            record_type,
            page_id,
            data: data.to_vec(),
        };

        let bytes = serialize_record(&record);

        // Rotate segment if necessary *before* writing.
        if self.current_segment_size + bytes.len() as u64 > self.segment_max_size {
            self.rotate_segment()?;
        }

        self.current_segment.write_all(&bytes)?;
        self.current_segment.flush()?;

        // fsync for durability.
        self.current_segment.get_ref().sync_all()?;

        self.current_segment_size += bytes.len() as u64;
        self.next_lsn += 1;

        Ok(lsn)
    }

    /// Explicitly flush and fsync the current segment.
    pub fn sync(&mut self) -> Result<()> {
        self.current_segment.flush()?;
        self.current_segment.get_ref().sync_all()?;
        Ok(())
    }

    /// Write a checkpoint record and return its LSN.
    pub fn checkpoint(&mut self) -> Result<Lsn> {
        self.append(WalRecordType::Checkpoint, 0, &[])
    }

    /// The next LSN that will be assigned.
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn
    }

    /// Read **all** valid WAL records from every segment in the directory.
    ///
    /// This is the primary entry point for crash recovery.  Records with invalid
    /// checksums at the tail of the last segment are silently skipped (they
    /// represent partial writes from a crash).  Corrupt records in the middle of
    /// a segment are treated as an error.
    pub fn recover(dir: &Path) -> Result<Vec<WalRecord>> {
        let segments = list_segments(dir)?;
        let num_segments = segments.len();
        let mut records = Vec::new();

        for (idx, (_seg_id, path)) in segments.iter().enumerate() {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            let is_last_segment = idx == num_segments - 1;

            loop {
                // Remember position so we can detect whether we're at EOF.
                match read_record(&mut reader) {
                    Ok(Some(rec)) => records.push(rec),
                    Ok(None) => break, // clean EOF
                    Err(e) => {
                        if is_last_segment {
                            // Tail corruption in the last segment -- stop reading
                            // but don't error out (crash recovery tolerates this).
                            break;
                        }
                        // Corruption in a non-last segment is unexpected.
                        return Err(e);
                    }
                }
            }
        }

        Ok(records)
    }

    // -- private helpers ----------------------------------------------------

    /// Create a new segment file and switch the writer to it.
    fn rotate_segment(&mut self) -> Result<()> {
        // Flush and sync the old segment before switching.
        self.sync()?;

        let new_id = self.current_segment_id + 1;
        let path = self.dir.join(segment_filename(new_id));
        let (writer, size) = open_segment_for_append(&path)?;

        self.current_segment = writer;
        self.current_segment_id = new_id;
        self.current_segment_size = size;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: open a WAL with a small segment limit for fast rotation testing.
    fn open_small(dir: &Path, max_size: u64) -> Wal {
        Wal::open_with_max_size(dir, max_size).expect("failed to open WAL")
    }

    // 1. Write and read back a single record.
    #[test]
    fn write_and_read_single_record() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        let data = b"hello page";
        let lsn = {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append(WalRecordType::PageWrite, 42, data).unwrap()
        };
        assert_eq!(lsn, 1);

        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].lsn, 1);
        assert_eq!(records[0].record_type, WalRecordType::PageWrite);
        assert_eq!(records[0].page_id, 42);
        assert_eq!(records[0].data, data);
    }

    // 2. Multiple records with monotonically increasing LSNs.
    #[test]
    fn multiple_records_lsn_ordering() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        {
            let mut wal = Wal::open(&dir).unwrap();
            for i in 0..10u32 {
                let lsn = wal
                    .append(WalRecordType::PageWrite, i, &i.to_le_bytes())
                    .unwrap();
                assert_eq!(lsn, (i + 1) as u64);
            }
            assert_eq!(wal.next_lsn(), 11);
        }

        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 10);
        for (i, rec) in records.iter().enumerate() {
            assert_eq!(rec.lsn, (i + 1) as u64, "LSN mismatch at index {i}");
            assert_eq!(rec.page_id, i as u32);
            assert_eq!(rec.data, (i as u32).to_le_bytes());
        }
    }

    // 3. Segment rotation.
    #[test]
    fn segment_rotation() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        // Use a tiny segment limit so rotation triggers quickly.
        // Each record with 100 bytes of data is 100 + RECORD_HEADER_SIZE = 121 bytes.
        let record_data = vec![0xABu8; 100];
        let record_on_disk = RECORD_HEADER_SIZE + 100;
        // Set limit so that two records fit but the third triggers rotation.
        let max_size = (record_on_disk * 2 + 1) as u64;

        {
            let mut wal = open_small(&dir, max_size);
            for _ in 0..5 {
                wal.append(WalRecordType::PageWrite, 1, &record_data)
                    .unwrap();
            }
        }

        // Verify we have multiple segment files.
        let segments = list_segments(&dir).unwrap();
        assert!(
            segments.len() > 1,
            "expected multiple segments, got {}",
            segments.len()
        );

        // All records should still be recoverable.
        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 5);
        for (i, rec) in records.iter().enumerate() {
            assert_eq!(rec.lsn, (i + 1) as u64);
        }
    }

    // 4. Full recovery across re-opens.
    #[test]
    fn recovery_reads_all_records() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        // Write some records, close, re-open, write more.
        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append(WalRecordType::PageWrite, 1, b"first").unwrap();
            wal.append(WalRecordType::Commit, 0, &[]).unwrap();
        }
        {
            let mut wal = Wal::open(&dir).unwrap();
            assert_eq!(wal.next_lsn(), 3, "should resume after previous LSNs");
            wal.append(WalRecordType::PageWrite, 2, b"second").unwrap();
        }

        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].data, b"first");
        assert_eq!(records[1].record_type, WalRecordType::Commit);
        assert_eq!(records[2].data, b"second");
        assert_eq!(records[2].lsn, 3);
    }

    // 5. Checksum validation -- corrupt a byte and verify detection.
    #[test]
    fn checksum_detects_corruption() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append(WalRecordType::PageWrite, 1, b"important data")
                .unwrap();
        }

        // Corrupt a byte in the middle of the segment file.
        let segments = list_segments(&dir).unwrap();
        assert_eq!(segments.len(), 1);
        let seg_path = &segments[0].1;

        let mut contents = fs::read(seg_path).unwrap();
        assert!(!contents.is_empty());
        // Flip a bit in the data portion (offset 17 is where data starts:
        // 8 lsn + 1 type + 4 page_id + 4 data_len = 17).
        let corrupt_offset = 17;
        contents[corrupt_offset] ^= 0xFF;
        fs::write(seg_path, &contents).unwrap();

        // Recovery should detect the corrupt record in the last (only) segment
        // and skip it (tail corruption tolerance).
        let records = Wal::recover(&dir).unwrap();
        assert!(
            records.is_empty(),
            "corrupt record should be skipped, got {} records",
            records.len()
        );
    }

    // 6. Checkpoint record.
    #[test]
    fn checkpoint_record() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        let checkpoint_lsn = {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append(WalRecordType::PageWrite, 1, b"data").unwrap();
            wal.checkpoint().unwrap()
        };
        assert_eq!(checkpoint_lsn, 2);

        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].record_type, WalRecordType::Checkpoint);
        assert_eq!(records[1].page_id, 0);
        assert!(records[1].data.is_empty());
    }

    // 7. CRC32 known-value sanity check.
    #[test]
    fn crc32_known_values() {
        // "123456789" should produce 0xCBF43926 with standard CRC-32/ISO-HDLC.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // Empty input.
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    // 8. Recovery tolerates partial write (truncated record at tail).
    #[test]
    fn recovery_tolerates_partial_write() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");

        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append(WalRecordType::PageWrite, 1, b"good record")
                .unwrap();
            wal.append(WalRecordType::PageWrite, 2, b"also good")
                .unwrap();
        }

        // Append some garbage bytes to simulate a partial write.
        let segments = list_segments(&dir).unwrap();
        let seg_path = &segments[0].1;
        let mut file = OpenOptions::new().append(true).open(seg_path).unwrap();
        // Write a partial header (fewer bytes than a full record).
        file.write_all(&[0x01, 0x00, 0x00, 0x00, 0x00]).unwrap();

        // Recovery should return the two good records and skip the partial tail.
        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data, b"good record");
        assert_eq!(records[1].data, b"also good");
    }

    // 9. Re-open WAL after segment rotation and continue writing.
    #[test]
    fn reopen_after_rotation() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wal");
        let max_size = 128u64; // very small segments

        {
            let mut wal = open_small(&dir, max_size);
            for _ in 0..10 {
                wal.append(WalRecordType::PageWrite, 1, &[0u8; 50])
                    .unwrap();
            }
        }

        let segments_before = list_segments(&dir).unwrap();
        assert!(segments_before.len() > 1);

        // Re-open and write more.
        {
            let mut wal = Wal::open_with_max_size(&dir, max_size).unwrap();
            wal.append(WalRecordType::Commit, 0, &[]).unwrap();
        }

        let records = Wal::recover(&dir).unwrap();
        assert_eq!(records.len(), 11);
        assert_eq!(records.last().unwrap().record_type, WalRecordType::Commit);
    }
}
