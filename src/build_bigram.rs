//! Production bigram builder for large corpora (OpenSubtitles, Wikipedia)
//!
//! Features:
//! - Sharded processing (RAM-safe for 100M+ bigrams)
//! - Canonical lowercase mapping for better coverage  
//! - Correct binary layout: header + index + edges
//! - Weight quantization preserved
//!
//! Usage:
//!   cargo run --release --bin build_bigram_v2 -- <corpus.txt.gz> [--top N] [--shards S]

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use fst::Map;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// Binary format constants
const MAGIC: u32 = 0x4247524D; // "BGRM"
const VERSION: u32 = 1;

/// Header layout (32 bytes)
#[repr(C, packed)]
struct Header {
    magic: u32,         // 0x4247524D "BGRM"
    version: u32,       // 1
    vocab_size: u32,    // total entries in index
    edges_count: u32,   // total edges
    top_n: u32,         // max edges per prev
    reserved: [u32; 3], // padding to 32 bytes
}

/// Index entry (8 bytes per prev_id)
#[repr(C, packed)]
struct IndexEntry {
    offset: u32,   // byte offset into edges section
    len: u16,      // number of edges for this prev
    reserved: u16, // padding
}

/// Edge entry (8 bytes)
#[repr(C, packed)]
struct Edge {
    next_id: u32, // word_id of next word
    weight: u16,  // quantized weight (0-65535)
    flags: u16,   // reserved
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.txt.gz> [--top N] [--shards S]", args[0]);
        eprintln!("  --top N      : Keep top N next words per prev (default: 10)");
        eprintln!("  --shards S   : Number of shards for RAM control (default: 256)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let top_n: usize = parse_arg(&args, "--top").unwrap_or(10);
    let num_shards: usize = parse_arg(&args, "--shards").unwrap_or(256);

    println!("=== Production Bigram Builder ===");
    println!("Input: {}", input_path);
    println!("Top-N: {}", top_n);
    println!("Shards: {}", num_shards);

    // Step 1: Build canonical lowercase map
    println!("\n[1/4] Building canonical lowercase map...");
    let (vocab_size, canonical_map) = build_canonical_map("en.lex.fst", "en.vocab.txt")?;
    println!("  Vocab size: {}", vocab_size);
    println!("  Canonical entries: {}", canonical_map.len());

    // Step 2: Shard bigrams to disk
    println!("\n[2/4] Extracting bigrams to shards...");
    let shard_dir = Path::new("bigram_shards");
    std::fs::create_dir_all(shard_dir)?;
    let total_bigrams = shard_bigrams(input_path, &canonical_map, shard_dir, num_shards)?;
    println!("  Total bigrams emitted: {}", total_bigrams);

    // Step 3: Reduce shards to top-N per prev
    println!("\n[3/4] Reducing shards to top-{} per prev...", top_n);
    let (index, edges) = reduce_shards(shard_dir, num_shards, vocab_size, top_n)?;
    println!(
        "  Unique prev_ids with edges: {}",
        index.iter().filter(|e| e.len > 0).count()
    );
    println!("  Total edges: {}", edges.len());

    // Step 4: Write binary file
    println!("\n[4/4] Writing en.bigram.bin...");
    write_bigram_bin("en.bigram.bin", vocab_size, top_n as u32, &index, &edges)?;

    // Cleanup shards
    std::fs::remove_dir_all(shard_dir)?;

    let file_size = std::fs::metadata("en.bigram.bin")?.len();
    println!(
        "\n✓ en.bigram.bin created ({:.2} MB)",
        file_size as f64 / 1_000_000.0
    );
    println!("  Header: 32 bytes");
    println!(
        "  Index: {} entries × 8 bytes = {} bytes",
        vocab_size,
        vocab_size * 8
    );
    println!(
        "  Edges: {} entries × 8 bytes = {} bytes",
        edges.len(),
        edges.len() * 8
    );

    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

/// Build canonical lowercase -> best word_id map
/// "best" = highest prob_q among all case variants
fn build_canonical_map(fst_path: &str, vocab_path: &str) -> Result<(u32, HashMap<String, u32>)> {
    let file = File::open(fst_path).context("Failed to open FST")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let fst = Map::new(mmap)?;

    let vocab_file = BufReader::new(File::open(vocab_path)?);
    let mut canonical: HashMap<String, (u32, u8)> = HashMap::new(); // lower -> (best_id, best_prob)
    let mut vocab_size: u32 = 0;

    for line in vocab_file.lines() {
        let word = line?;
        vocab_size += 1;

        if let Some(v) = fst.get(&word) {
            let word_id = ((v >> 16) & 0xFFFF_FFFF) as u32;
            let prob = (v & 0xFF) as u8;
            let lower = word.to_lowercase();

            canonical
                .entry(lower)
                .and_modify(|(best_id, best_prob)| {
                    if prob > *best_prob {
                        *best_id = word_id;
                        *best_prob = prob;
                    }
                })
                .or_insert((word_id, prob));
        }
    }

    // Convert to simple id map
    let map: HashMap<String, u32> = canonical.into_iter().map(|(k, (id, _))| (k, id)).collect();
    Ok((vocab_size, map))
}

/// Emit bigrams to shard files: shard[prev_id % S] gets (prev_id, next_id)
fn shard_bigrams(
    input_path: &str,
    canonical: &HashMap<String, u32>,
    shard_dir: &Path,
    num_shards: usize,
) -> Result<u64> {
    // Open shard files
    let mut shards: Vec<BufWriter<File>> = (0..num_shards)
        .map(|i| {
            let path = shard_dir.join(format!("shard_{:03}.bin", i));
            BufWriter::new(File::create(path).unwrap())
        })
        .collect();

    let file = File::open(input_path)?;
    let reader: Box<dyn BufRead> = if input_path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, GzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    };

    let mut lines_processed = 0u64;
    let mut bigrams_emitted = 0u64;
    let mut prev_id: Option<u32> = None;

    for line in reader.lines() {
        let line = line?;
        lines_processed += 1;

        if lines_processed % 5_000_000 == 0 {
            println!(
                "  {} M lines, {} M bigrams",
                lines_processed / 1_000_000,
                bigrams_emitted / 1_000_000
            );
        }

        for word in line.split_whitespace() {
            let normalized = normalize_token(word);
            if normalized.is_empty() {
                prev_id = None;
                continue;
            }

            if let Some(&word_id) = canonical.get(&normalized) {
                if let Some(prev) = prev_id {
                    // Emit to shard
                    let shard_idx = (prev as usize) % num_shards;
                    shards[shard_idx].write_all(&prev.to_le_bytes())?;
                    shards[shard_idx].write_all(&word_id.to_le_bytes())?;
                    bigrams_emitted += 1;
                }
                prev_id = Some(word_id);
            } else {
                prev_id = None;
            }
        }
        prev_id = None; // End of line breaks chain
    }

    // Flush all shards
    for mut shard in shards {
        shard.flush()?;
    }

    Ok(bigrams_emitted)
}

fn normalize_token(word: &str) -> String {
    word.to_lowercase()
        .chars()
        .filter(|c| c.is_alphabetic() || *c == '\'')
        .collect()
}

/// Reduce shards: sort, count, top-N per prev
fn reduce_shards(
    shard_dir: &Path,
    num_shards: usize,
    vocab_size: u32,
    top_n: usize,
) -> Result<(Vec<IndexEntry>, Vec<Edge>)> {
    // Per-prev aggregation using external sort approach per shard
    let mut all_edges: Vec<Vec<(u32, u64)>> = vec![Vec::new(); vocab_size as usize];

    for shard_idx in 0..num_shards {
        let path = shard_dir.join(format!("shard_{:03}.bin", shard_idx));
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        if file_len == 0 {
            continue;
        }

        // Read entire shard into memory (each shard is ~1/256 of data)
        let mut buf = vec![0u8; file_len as usize];
        file.read_exact(&mut buf)?;

        // Parse pairs and count
        let mut shard_counts: HashMap<(u32, u32), u64> = HashMap::new();
        for chunk in buf.chunks_exact(8) {
            let prev = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let next = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            *shard_counts.entry((prev, next)).or_insert(0) += 1;
        }

        // Merge into global per-prev lists
        for ((prev, next), count) in shard_counts {
            if (prev as usize) < all_edges.len() {
                all_edges[prev as usize].push((next, count));
            }
        }

        if (shard_idx + 1) % 32 == 0 {
            println!("  Processed {}/{} shards", shard_idx + 1, num_shards);
        }
    }

    // Build index and edges arrays
    let mut index: Vec<IndexEntry> = Vec::with_capacity(vocab_size as usize);
    let mut edges: Vec<Edge> = Vec::new();

    for edges_for_prev in all_edges {
        let offset = (edges.len() * 8) as u32;

        if edges_for_prev.is_empty() {
            index.push(IndexEntry {
                offset,
                len: 0,
                reserved: 0,
            });
            continue;
        }

        // Sort by count descending, take top-N
        let mut sorted = edges_for_prev;
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(top_n);

        // Quantize weights: log-scale to 0-65535
        let max_count = sorted.first().map(|(_, c)| *c).unwrap_or(1);

        for (next_id, count) in &sorted {
            let weight = quantize_weight(*count, max_count);
            edges.push(Edge {
                next_id: *next_id,
                weight,
                flags: 0,
            });
        }

        index.push(IndexEntry {
            offset,
            len: sorted.len() as u16,
            reserved: 0,
        });
    }

    Ok((index, edges))
}

/// Quantize count to 16-bit weight using log scale
fn quantize_weight(count: u64, max_count: u64) -> u16 {
    if count == 0 || max_count == 0 {
        return 0;
    }
    // Relative weight: (count / max_count) scaled to 0-65535
    // Use log scale for better distribution
    let ratio = (count as f64).ln() / (max_count as f64).ln().max(1.0);
    (ratio.clamp(0.0, 1.0) * 65535.0) as u16
}

/// Write binary file with header + index + edges
fn write_bigram_bin(
    path: &str,
    vocab_size: u32,
    top_n: u32,
    index: &[IndexEntry],
    edges: &[Edge],
) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    // Write header
    let header = Header {
        magic: MAGIC,
        version: VERSION,
        vocab_size,
        edges_count: edges.len() as u32,
        top_n,
        reserved: [0; 3],
    };

    unsafe {
        let header_bytes = std::slice::from_raw_parts(
            &header as *const Header as *const u8,
            std::mem::size_of::<Header>(),
        );
        file.write_all(header_bytes)?;
    }

    // Write index
    for entry in index {
        file.write_all(&entry.offset.to_le_bytes())?;
        file.write_all(&entry.len.to_le_bytes())?;
        file.write_all(&entry.reserved.to_le_bytes())?;
    }

    // Write edges
    for edge in edges {
        file.write_all(&edge.next_id.to_le_bytes())?;
        file.write_all(&edge.weight.to_le_bytes())?;
        file.write_all(&edge.flags.to_le_bytes())?;
    }

    file.flush()?;
    Ok(())
}
