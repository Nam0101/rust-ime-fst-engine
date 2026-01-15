//! Comprehensive bigram validation tests
//!
//! Usage: cargo run --release --bin validate_bigram

use anyhow::Result;
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

const MAGIC: u32 = 0x4247524D;

fn main() -> Result<()> {
    let file = File::open("en.bigram.bin")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data = mmap.as_ref();

    // Parse header
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let edges_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let top_n = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;

    println!("═══════════════════════════════════════════════════════════════");
    println!("                    BIGRAM VALIDATION TESTS                     ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========== 3.1 FORMAT TESTS ==========
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3.1 FORMAT INVARIANTS                                       │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Check magic
    let magic_ok = magic == MAGIC;
    println!("  Magic: 0x{:08X} {}", magic, status(magic_ok));

    // Check version
    let version_ok = version == 1;
    println!("  Version: {} {}", version, status(version_ok));

    // Invariant A: Size formula
    let header_size = 32;
    let index_size = vocab_size * 8;
    let edges_size = edges_count * 8;
    let expected_size = header_size + index_size + edges_size;
    let actual_size = data.len();
    let size_ok = actual_size == expected_size;
    println!(
        "  Size formula: expected={}, actual={} {}",
        expected_size,
        actual_size,
        status(size_ok)
    );

    // Invariant B: Check all offsets are within bounds
    let edges_base = header_size + index_size;
    let mut offset_errors = 0;
    let mut sorted_errors = 0;
    let mut duplicate_errors = 0;
    let mut lens: Vec<usize> = Vec::with_capacity(vocab_size);

    for prev_id in 0..vocab_size {
        let idx_offset = header_size + prev_id * 8;
        let offset = u32::from_le_bytes([
            data[idx_offset],
            data[idx_offset + 1],
            data[idx_offset + 2],
            data[idx_offset + 3],
        ]) as usize;
        let len = u16::from_le_bytes([data[idx_offset + 4], data[idx_offset + 5]]) as usize;

        lens.push(len);

        if len == 0 {
            continue;
        }

        // Check offset bounds
        let edge_start = edges_base + offset;
        let edge_end = edge_start + len * 8;
        if edge_end > actual_size {
            offset_errors += 1;
            continue;
        }

        // Read edges and check invariants
        let mut prev_weight = u16::MAX;
        let mut seen_ids: HashSet<u32> = HashSet::new();

        for i in 0..len {
            let e_off = edge_start + i * 8;
            let next_id = u32::from_le_bytes([
                data[e_off],
                data[e_off + 1],
                data[e_off + 2],
                data[e_off + 3],
            ]);
            let weight = u16::from_le_bytes([data[e_off + 4], data[e_off + 5]]);

            // Check sorted by weight (non-increasing)
            if weight > prev_weight {
                sorted_errors += 1;
            }
            prev_weight = weight;

            // Check no duplicates
            if !seen_ids.insert(next_id) {
                duplicate_errors += 1;
            }
        }
    }

    println!(
        "  Offset bounds: {} errors {}",
        offset_errors,
        status(offset_errors == 0)
    );
    println!(
        "  Weight sorted: {} errors {}",
        sorted_errors,
        status(sorted_errors == 0)
    );
    println!(
        "  No duplicates: {} errors {}",
        duplicate_errors,
        status(duplicate_errors == 0)
    );

    // ========== 3.2 COVERAGE/SPARSITY ==========
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3.2 COVERAGE / SPARSITY STATS                               │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let non_empty = lens.iter().filter(|&&l| l > 0).count();
    let coverage = (non_empty as f64 / vocab_size as f64) * 100.0;
    println!(
        "  prev_has_edges_ratio: {}/{} ({:.1}%)",
        non_empty, vocab_size, coverage
    );

    // Histogram
    let mut histogram = vec![0usize; top_n + 1];
    for &len in &lens {
        let bucket = len.min(top_n);
        histogram[bucket] += 1;
    }

    println!("\n  Length histogram:");
    for (len, count) in histogram.iter().enumerate() {
        if *count > 0 {
            let bar_len = (*count as f64 / vocab_size as f64 * 50.0) as usize;
            let bar: String = "█".repeat(bar_len.max(1));
            println!(
                "    len={:2}: {:6} ({:5.1}%) {}",
                len,
                count,
                (*count as f64 / vocab_size as f64) * 100.0,
                bar
            );
        }
    }

    // Median and P90 (only for non-empty)
    let mut non_empty_lens: Vec<usize> = lens.iter().filter(|&&l| l > 0).copied().collect();
    non_empty_lens.sort();

    let median = if non_empty_lens.is_empty() {
        0
    } else {
        non_empty_lens[non_empty_lens.len() / 2]
    };
    let p90_idx = (non_empty_lens.len() as f64 * 0.9) as usize;
    let p90 = non_empty_lens.get(p90_idx).copied().unwrap_or(0);
    let p10_idx = (non_empty_lens.len() as f64 * 0.1) as usize;
    let p10 = non_empty_lens.get(p10_idx).copied().unwrap_or(0);

    println!("\n  Stats (among entries with edges):");
    println!("    P10: {} edges", p10);
    println!("    Median: {} edges", median);
    println!("    P90: {} edges", p90);

    // ========== 3.3 PROBE LIST SANITY ==========
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3.3 PROBE LIST SANITY CHECK                                 │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Load vocab
    let vocab: Vec<String> = BufReader::new(File::open("en.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    let probes = [
        // Determiners
        "the", "a", "an", "this", "that", "these", "those", // Pronouns
        "i", "you", "we", "they", "he", "she", "it", // Auxiliaries
        "is", "are", "was", "were", "have", "has", "had", "will", "would", "can", "could",
        // Common verbs
        "go", "make", "know", "think", "want", "need", "see", "take", "get", "come",
        // Question words
        "what", "where", "when", "why", "how", "who", // Conjunctions/prepositions
        "and", "but", "or", "if", "because", "to", "for", "with", "in", "on",
    ];

    println!("  Testing {} probe words:\n", probes.len());

    for probe in &probes {
        let lower = probe.to_lowercase();
        if let Some(word_id) = vocab.iter().position(|w| w.to_lowercase() == lower) {
            let idx_offset = header_size + word_id * 8;
            let offset = u32::from_le_bytes([
                data[idx_offset],
                data[idx_offset + 1],
                data[idx_offset + 2],
                data[idx_offset + 3],
            ]) as usize;
            let len = u16::from_le_bytes([data[idx_offset + 4], data[idx_offset + 5]]) as usize;

            if len == 0 {
                println!("  {:12} → (no edges)", probe);
                continue;
            }

            // Get top 5
            let mut top5 = Vec::new();
            let edge_start = edges_base + offset;
            for i in 0..len.min(5) {
                let e_off = edge_start + i * 8;
                let next_id = u32::from_le_bytes([
                    data[e_off],
                    data[e_off + 1],
                    data[e_off + 2],
                    data[e_off + 3],
                ]) as usize;
                if let Some(word) = vocab.get(next_id) {
                    top5.push(word.as_str());
                }
            }
            println!("  {:12} → {}", probe, top5.join(", "));
        } else {
            println!("  {:12} → (not in vocab)", probe);
        }
    }

    // ========== SUMMARY ==========
    println!("\n═══════════════════════════════════════════════════════════════");
    let all_pass = magic_ok
        && version_ok
        && size_ok
        && offset_errors == 0
        && sorted_errors == 0
        && duplicate_errors == 0;
    if all_pass {
        println!("  ✅ ALL FORMAT TESTS PASSED");
    } else {
        println!("  ❌ SOME TESTS FAILED");
    }
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

fn status(ok: bool) -> &'static str {
    if ok {
        "✓"
    } else {
        "✗"
    }
}
