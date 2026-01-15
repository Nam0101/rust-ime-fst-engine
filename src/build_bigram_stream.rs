//! Streaming bigram builder - single pass, RAM-constrained
//!
//! Keeps only top-N candidates per prev_id in RAM using count-min sketch
//! for approximate counting + min-heap for top-N tracking.
//!
//! Trade-off: Less accurate than full count, but fits in memory.
//!
//! Usage:
//!   cargo run --release --bin build_bigram_stream -- <corpus.txt.gz> [--top N]

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use fst::Map;
use memmap2::Mmap;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

// Binary format constants
const MAGIC: u32 = 0x4247524D; // "BGRM"
const VERSION: u32 = 1;

/// TopN tracker using exact counting with pruning
/// Prunes when entry count exceeds threshold
struct TopNTracker {
    counts: HashMap<u32, u64>, // next_id -> count
    top_n: usize,
    prune_threshold: usize, // prune when len > this
}

impl TopNTracker {
    fn new(top_n: usize) -> Self {
        Self {
            counts: HashMap::new(),
            top_n,
            prune_threshold: top_n * 100, // keep 100x candidates before pruning
        }
    }

    fn add(&mut self, next_id: u32) {
        *self.counts.entry(next_id).or_insert(0) += 1;

        if self.counts.len() > self.prune_threshold {
            self.prune();
        }
    }

    fn prune(&mut self) {
        if self.counts.len() <= self.top_n * 2 {
            return;
        }

        // Keep top 2*N by count
        let mut items: Vec<_> = self.counts.drain().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(self.top_n * 2);

        self.counts = items.into_iter().collect();
    }

    fn finalize(mut self) -> Vec<(u32, u64)> {
        let mut items: Vec<_> = self.counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(self.top_n);
        items
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.txt.gz> [--top N] [--limit M]", args[0]);
        eprintln!("  --top N    : Keep top N next words per prev (default: 10)");
        eprintln!("  --limit M  : Process only first M million lines (default: all)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let top_n: usize = parse_arg(&args, "--top").unwrap_or(10);
    let limit_m: Option<usize> = parse_arg(&args, "--limit");

    println!("=== Streaming Bigram Builder ===");
    println!("Input: {}", input_path);
    println!("Top-N: {}", top_n);
    if let Some(m) = limit_m {
        println!("Limit: {} million lines", m);
    }

    // Step 1: Build canonical lowercase map
    println!("\n[1/3] Building canonical lowercase map...");
    let (vocab_size, canonical_map) = build_canonical_map("en.lex.fst", "en.vocab.txt")?;
    println!("  Vocab size: {}", vocab_size);
    println!("  Canonical entries: {}", canonical_map.len());

    // Step 2: Stream through corpus, maintain per-prev TopN trackers
    println!("\n[2/3] Streaming bigrams (single pass)...");

    // Per-prev tracking - only allocate when seen
    let mut trackers: HashMap<u32, TopNTracker> = HashMap::new();

    let file = File::open(input_path)?;
    let reader: Box<dyn BufRead> = if input_path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, GzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    };

    let mut lines_processed = 0u64;
    let mut bigrams_seen = 0u64;
    let mut prev_id: Option<u32> = None;
    let line_limit = limit_m.map(|m| m * 1_000_000);

    for line in reader.lines() {
        let line = line?;
        lines_processed += 1;

        if let Some(limit) = line_limit {
            if lines_processed as usize > limit {
                break;
            }
        }

        if lines_processed % 5_000_000 == 0 {
            println!(
                "  {} M lines, {} M bigrams, {} active prevs",
                lines_processed / 1_000_000,
                bigrams_seen / 1_000_000,
                trackers.len()
            );
        }

        for word in line.split_whitespace() {
            let normalized = normalize_token(word);
            if normalized.is_empty() {
                prev_id = None;
                continue;
            }

            if let Some(&word_id) = canonical_map.get(&normalized) {
                if let Some(prev) = prev_id {
                    trackers
                        .entry(prev)
                        .or_insert_with(|| TopNTracker::new(top_n))
                        .add(word_id);
                    bigrams_seen += 1;
                }
                prev_id = Some(word_id);
            } else {
                prev_id = None;
            }
        }
        prev_id = None;
    }

    println!(
        "\n  Total: {} lines, {} bigrams",
        lines_processed, bigrams_seen
    );
    println!("  Unique prev_ids tracked: {}", trackers.len());

    // Step 3: Finalize and write binary file
    println!("\n[3/3] Finalizing and writing en.bigram.bin...");

    // Build index and edges
    let mut index: Vec<(u32, u16)> = vec![(0, 0); vocab_size as usize]; // (offset, len)
    let mut edges: Vec<(u32, u16)> = Vec::new(); // (next_id, weight)

    for (prev_id, tracker) in trackers {
        let top_items = tracker.finalize();
        if top_items.is_empty() {
            continue;
        }

        let offset = edges.len() as u32;
        let max_count = top_items.first().map(|(_, c)| *c).unwrap_or(1);

        for (next_id, count) in top_items {
            let weight = quantize_weight(count, max_count);
            edges.push((next_id, weight));
        }

        if (prev_id as usize) < index.len() {
            let len = (edges.len() as u32 - offset) as u16;
            index[prev_id as usize] = (offset * 8, len); // offset in bytes
        }
    }

    // Write file
    let mut file = BufWriter::new(File::create("en.bigram.bin")?);

    // Header (32 bytes)
    file.write_all(&MAGIC.to_le_bytes())?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&vocab_size.to_le_bytes())?;
    file.write_all(&(edges.len() as u32).to_le_bytes())?;
    file.write_all(&(top_n as u32).to_le_bytes())?;
    file.write_all(&[0u8; 12])?; // reserved

    // Index (8 bytes per entry)
    for (offset, len) in &index {
        file.write_all(&offset.to_le_bytes())?;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&[0u8; 2])?; // reserved
    }

    // Edges (8 bytes per entry)
    for (next_id, weight) in &edges {
        file.write_all(&next_id.to_le_bytes())?;
        file.write_all(&weight.to_le_bytes())?;
        file.write_all(&[0u8; 2])?; // flags
    }

    file.flush()?;

    let file_size = std::fs::metadata("en.bigram.bin")?.len();
    println!(
        "\n✓ en.bigram.bin created ({:.2} MB)",
        file_size as f64 / 1_000_000.0
    );
    println!(
        "  Vocab entries with bigrams: {}",
        index.iter().filter(|(_, len)| *len > 0).count()
    );
    println!("  Total edges: {}", edges.len());

    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn build_canonical_map(fst_path: &str, vocab_path: &str) -> Result<(u32, HashMap<String, u32>)> {
    let file = File::open(fst_path).context("Failed to open FST")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let fst = Map::new(mmap)?;

    let vocab_file = BufReader::new(File::open(vocab_path)?);
    let mut canonical: HashMap<String, (u32, u8)> = HashMap::new();
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

    let map: HashMap<String, u32> = canonical.into_iter().map(|(k, (id, _))| (k, id)).collect();
    Ok((vocab_size, map))
}

fn normalize_token(word: &str) -> String {
    word.to_lowercase()
        .chars()
        .filter(|c| c.is_alphabetic() || *c == '\'')
        .collect()
}

fn quantize_weight(count: u64, max_count: u64) -> u16 {
    if count == 0 || max_count == 0 {
        return 0;
    }
    let ratio = (count as f64).ln() / (max_count as f64).ln().max(1.0);
    (ratio.clamp(0.0, 1.0) * 65535.0) as u16
}
