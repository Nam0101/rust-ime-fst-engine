//! Vietnamese bigram builder using syllable-based approach
//!
//! Usage: cargo run --release --bin build_vi_bigram -- <corpus.txt.gz> [--top N]

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use fst::Map;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

const MAGIC: u32 = 0x4247524D; // "BGRM"
const VERSION: u32 = 1;

/// TopN tracker with pruning
struct TopNTracker {
    counts: HashMap<u32, u64>,
    top_n: usize,
    prune_threshold: usize,
}

impl TopNTracker {
    fn new(top_n: usize) -> Self {
        Self {
            counts: HashMap::new(),
            top_n,
            prune_threshold: top_n * 100,
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
        let mut items: Vec<_> = self.counts.drain().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(self.top_n * 2);
        self.counts = items.into_iter().collect();
    }

    fn finalize(self) -> Vec<(u32, u64)> {
        let mut items: Vec<_> = self.counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(self.top_n);
        items
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <corpus.txt.gz> [--top N]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let top_n: usize = args
        .iter()
        .position(|a| a == "--top")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("=== Vietnamese Bigram Builder ===");
    println!("Input: {}", input_path);
    println!("Top-N: {}", top_n);

    // Load Vietnamese syllable FST
    println!("\n[1/3] Loading vi.syllable.fst...");
    let (vocab_size, syllable_map) = load_syllable_map("vi.syllable.fst", "vi.syllable.vocab.txt")?;
    println!("  Vocab size: {}", vocab_size);
    println!("  Syllables loaded: {}", syllable_map.len());

    // Stream through corpus
    println!("\n[2/3] Streaming bigrams...");

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

    for line in reader.lines() {
        let line = line?;
        lines_processed += 1;

        if lines_processed % 500_000 == 0 {
            println!(
                "  {} K lines, {} K bigrams, {} prevs",
                lines_processed / 1000,
                bigrams_seen / 1000,
                trackers.len()
            );
        }

        // Vietnamese: split by whitespace, each token is a syllable
        for word in line.split_whitespace() {
            let normalized = word.to_lowercase();

            if let Some(&syllable_id) = syllable_map.get(&normalized) {
                if let Some(prev) = prev_id {
                    trackers
                        .entry(prev)
                        .or_insert_with(|| TopNTracker::new(top_n))
                        .add(syllable_id);
                    bigrams_seen += 1;
                }
                prev_id = Some(syllable_id);
            } else {
                prev_id = None;
            }
        }
        prev_id = None; // End of line
    }

    println!(
        "\n  Total: {} lines, {} bigrams",
        lines_processed, bigrams_seen
    );
    println!("  Unique prev_ids: {}", trackers.len());

    // Write binary file
    println!("\n[3/3] Writing vi.bigram.bin...");

    let mut index: Vec<(u32, u16)> = vec![(0, 0); vocab_size as usize];
    let mut edges: Vec<(u32, u16)> = Vec::new();

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
            index[prev_id as usize] = (offset * 8, len);
        }
    }

    // Write file
    let mut file = BufWriter::new(File::create("vi.bigram.bin")?);

    // Header
    file.write_all(&MAGIC.to_le_bytes())?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(vocab_size as u32).to_le_bytes())?;
    file.write_all(&(edges.len() as u32).to_le_bytes())?;
    file.write_all(&(top_n as u32).to_le_bytes())?;
    file.write_all(&[0u8; 12])?;

    // Index
    for (offset, len) in &index {
        file.write_all(&offset.to_le_bytes())?;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&[0u8; 2])?;
    }

    // Edges
    for (next_id, weight) in &edges {
        file.write_all(&next_id.to_le_bytes())?;
        file.write_all(&weight.to_le_bytes())?;
        file.write_all(&[0u8; 2])?;
    }

    file.flush()?;

    let file_size = std::fs::metadata("vi.bigram.bin")?.len();
    println!(
        "\n✓ vi.bigram.bin created ({:.2} KB)",
        file_size as f64 / 1000.0
    );
    println!(
        "  Vocab entries with bigrams: {}",
        index.iter().filter(|(_, len)| *len > 0).count()
    );
    println!("  Total edges: {}", edges.len());

    Ok(())
}

fn load_syllable_map(fst_path: &str, vocab_path: &str) -> Result<(usize, HashMap<String, u32>)> {
    let file = File::open(fst_path).context("Failed to open vi.syllable.fst")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let fst = Map::new(mmap)?;

    let vocab: Vec<String> = BufReader::new(File::open(vocab_path)?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    let vocab_size = vocab.len();

    let mut map: HashMap<String, u32> = HashMap::new();
    for (id, word) in vocab.iter().enumerate() {
        let lower = word.to_lowercase();
        map.insert(lower, id as u32);
    }

    Ok((vocab_size, map))
}

fn quantize_weight(count: u64, max_count: u64) -> u16 {
    if count == 0 || max_count == 0 {
        return 0;
    }
    let ratio = (count as f64).ln() / (max_count as f64).ln().max(1.0);
    (ratio.clamp(0.0, 1.0) * 65535.0) as u16
}
