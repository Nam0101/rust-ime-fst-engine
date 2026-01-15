//! Build Vietnamese trigram cache for hybrid suggestion
//!
//! Only caches trigrams for top K most frequent bigram pairs
//!
//! Usage: cargo run --release --bin build_vi_trigram -- <corpus.txt.gz> [--pairs K] [--top N]

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

const MAGIC: u32 = 0x54524743; // "TRGC" = Trigram Cache
const VERSION: u32 = 1;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <corpus.txt.gz> [--pairs K] [--top N]", args[0]);
        eprintln!("  --pairs K : Keep top K bigram pairs (default: 5000)");
        eprintln!("  --top N   : Keep top N next syllables per pair (default: 10)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let max_pairs: usize = parse_arg(&args, "--pairs").unwrap_or(5000);
    let top_n: usize = parse_arg(&args, "--top").unwrap_or(10);

    println!("=== Vietnamese Trigram Cache Builder ===");
    println!("Input: {}", input_path);
    println!("Max pairs: {}", max_pairs);
    println!("Top-N per pair: {}", top_n);

    // Load syllable vocab
    println!("\n[1/4] Loading vocabulary...");
    let vocab: Vec<String> = BufReader::new(File::open("vi.syllable.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    let mut syllable_to_id: HashMap<String, u32> = HashMap::new();
    for (id, w) in vocab.iter().enumerate() {
        syllable_to_id.insert(w.to_lowercase(), id as u32);
    }
    println!("  Loaded {} syllables", vocab.len());

    // Pass 1: Count bigram pairs frequency
    println!("\n[2/4] Counting bigram pair frequencies...");
    let mut pair_freq: HashMap<(u32, u32), u64> = HashMap::new();

    let file = File::open(input_path)?;
    let reader: Box<dyn BufRead> = if input_path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, GzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    };

    let mut lines = 0u64;
    let mut prev_id: Option<u32> = None;
    let mut prev_prev_id: Option<u32> = None;

    for line in reader.lines() {
        let line = line?;
        lines += 1;
        if lines % 500_000 == 0 {
            println!(
                "  {} K lines, {} unique pairs",
                lines / 1000,
                pair_freq.len()
            );
        }

        for word in line.split_whitespace() {
            let norm = word.to_lowercase();
            if let Some(&id) = syllable_to_id.get(&norm) {
                if let (Some(pp), Some(p)) = (prev_prev_id, prev_id) {
                    *pair_freq.entry((pp, p)).or_insert(0) += 1;
                }
                prev_prev_id = prev_id;
                prev_id = Some(id);
            } else {
                prev_prev_id = None;
                prev_id = None;
            }
        }
        prev_prev_id = None;
        prev_id = None;
    }

    println!("  Total: {} unique pairs", pair_freq.len());

    // Select top K pairs
    let mut pairs: Vec<_> = pair_freq.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs.truncate(max_pairs);

    let top_pairs: HashMap<(u32, u32), usize> = pairs
        .iter()
        .enumerate()
        .map(|(idx, ((a, b), _))| ((*a, *b), idx))
        .collect();

    println!("  Selected top {} pairs", top_pairs.len());

    // Pass 2: Collect trigrams for selected pairs
    println!("\n[3/4] Collecting trigrams for top pairs...");

    // trigram_counts[pair_idx] = HashMap<next_id, count>
    let mut trigram_counts: Vec<HashMap<u32, u64>> = vec![HashMap::new(); top_pairs.len()];

    let file = File::open(input_path)?;
    let reader: Box<dyn BufRead> = if input_path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, GzDecoder::new(file)))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    };

    lines = 0;
    prev_id = None;
    prev_prev_id = None;

    for line in reader.lines() {
        let line = line?;
        lines += 1;
        if lines % 500_000 == 0 {
            println!("  {} K lines processed", lines / 1000);
        }

        for word in line.split_whitespace() {
            let norm = word.to_lowercase();
            if let Some(&id) = syllable_to_id.get(&norm) {
                if let (Some(pp), Some(p)) = (prev_prev_id, prev_id) {
                    if let Some(&pair_idx) = top_pairs.get(&(pp, p)) {
                        *trigram_counts[pair_idx].entry(id).or_insert(0) += 1;
                    }
                }
                prev_prev_id = prev_id;
                prev_id = Some(id);
            } else {
                prev_prev_id = None;
                prev_id = None;
            }
        }
        prev_prev_id = None;
        prev_id = None;
    }

    // Build output
    println!("\n[4/4] Writing vi.trigram.cache.bin...");

    // Prepare data: sort pairs by (w1, w2), finalize top-N
    let mut pair_data: Vec<((u32, u32), Vec<(u32, u16)>)> = Vec::new();

    for ((w1, w2), pair_idx) in &top_pairs {
        let counts = &trigram_counts[*pair_idx];
        if counts.is_empty() {
            continue;
        }

        let mut nexts: Vec<_> = counts.iter().map(|(&k, &v)| (k, v)).collect();
        nexts.sort_by(|a, b| b.1.cmp(&a.1));
        nexts.truncate(top_n);

        let max_count = nexts.first().map(|(_, c)| *c).unwrap_or(1);
        let weighted: Vec<(u32, u16)> = nexts
            .into_iter()
            .map(|(id, count)| {
                let w = quantize_weight(count, max_count);
                (id, w)
            })
            .collect();

        pair_data.push(((*w1, *w2), weighted));
    }

    pair_data.sort_by_key(|((a, b), _)| (*a, *b));

    // Binary format:
    // Header: magic(4) version(4) num_pairs(4) top_n(4) reserved(16) = 32 bytes
    // Index: [w1(4) w2(4) offset(4) len(2) reserved(2)] × num_pairs = 16 bytes each
    // Edges: [next_id(4) weight(2) reserved(2)] × total_edges = 8 bytes each

    let mut file = BufWriter::new(File::create("vi.trigram.cache.bin")?);

    // Count total edges
    let total_edges: usize = pair_data.iter().map(|(_, v)| v.len()).sum();

    // Header
    file.write_all(&MAGIC.to_le_bytes())?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(pair_data.len() as u32).to_le_bytes())?;
    file.write_all(&(top_n as u32).to_le_bytes())?;
    file.write_all(&[0u8; 16])?; // reserved

    // Index
    let mut edge_offset: u32 = 0;
    for ((w1, w2), edges) in &pair_data {
        file.write_all(&w1.to_le_bytes())?;
        file.write_all(&w2.to_le_bytes())?;
        file.write_all(&edge_offset.to_le_bytes())?;
        file.write_all(&(edges.len() as u16).to_le_bytes())?;
        file.write_all(&[0u8; 2])?;
        edge_offset += edges.len() as u32;
    }

    // Edges
    for (_, edges) in &pair_data {
        for (next_id, weight) in edges {
            file.write_all(&next_id.to_le_bytes())?;
            file.write_all(&weight.to_le_bytes())?;
            file.write_all(&[0u8; 2])?;
        }
    }

    file.flush()?;

    let file_size = std::fs::metadata("vi.trigram.cache.bin")?.len();
    println!(
        "\n✓ vi.trigram.cache.bin created ({:.2} KB)",
        file_size as f64 / 1000.0
    );
    println!("  Pairs with trigrams: {}", pair_data.len());
    println!("  Total edges: {}", total_edges);

    // Print some examples
    println!("\nSample entries:");
    for ((w1, w2), edges) in pair_data.iter().take(10) {
        let s1 = vocab.get(*w1 as usize).map(|s| s.as_str()).unwrap_or("?");
        let s2 = vocab.get(*w2 as usize).map(|s| s.as_str()).unwrap_or("?");
        let nexts: Vec<_> = edges
            .iter()
            .take(3)
            .filter_map(|(id, _)| vocab.get(*id as usize))
            .map(|s| s.as_str())
            .collect();
        println!("  ({}, {}) → {}", s1, s2, nexts.join(", "));
    }

    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn quantize_weight(count: u64, max_count: u64) -> u16 {
    if count == 0 || max_count == 0 {
        return 0;
    }
    let ratio = (count as f64).ln() / (max_count as f64).ln().max(1.0);
    (ratio.clamp(0.0, 1.0) * 65535.0) as u16
}
