//! Vietnamese bigram validation tests
//!
//! Usage: cargo run --release --bin validate_vi_bigram

use anyhow::Result;
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

const MAGIC: u32 = 0x4247524D;

fn main() -> Result<()> {
    let file = File::open("vi.bigram.bin")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data = mmap.as_ref();

    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let edges_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let top_n = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;

    println!("═══════════════════════════════════════════════════════════════");
    println!("             VIETNAMESE BIGRAM VALIDATION TESTS                 ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // 3.1 FORMAT TESTS
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3.1 FORMAT INVARIANTS                                       │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let magic_ok = magic == MAGIC;
    println!(
        "  Magic: 0x{:08X} {}",
        magic,
        if magic_ok { "✓" } else { "✗" }
    );

    let version_ok = version == 1;
    println!(
        "  Version: {} {}",
        version,
        if version_ok { "✓" } else { "✗" }
    );

    let header_size = 32;
    let index_size = vocab_size * 8;
    let edges_size = edges_count * 8;
    let expected_size = header_size + index_size + edges_size;
    let actual_size = data.len();
    let size_ok = actual_size == expected_size;
    println!(
        "  Size: expected={}, actual={} {}",
        expected_size,
        actual_size,
        if size_ok { "✓" } else { "✗" }
    );

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

        let edge_start = edges_base + offset;
        let edge_end = edge_start + len * 8;
        if edge_end > actual_size {
            offset_errors += 1;
            continue;
        }

        let mut prev_weight = u16::MAX;
        let mut seen: HashSet<u32> = HashSet::new();

        for i in 0..len {
            let e_off = edge_start + i * 8;
            let next_id = u32::from_le_bytes([
                data[e_off],
                data[e_off + 1],
                data[e_off + 2],
                data[e_off + 3],
            ]);
            let weight = u16::from_le_bytes([data[e_off + 4], data[e_off + 5]]);
            if weight > prev_weight {
                sorted_errors += 1;
            }
            prev_weight = weight;
            if !seen.insert(next_id) {
                duplicate_errors += 1;
            }
        }
    }

    println!(
        "  Offset bounds: {} errors {}",
        offset_errors,
        if offset_errors == 0 { "✓" } else { "✗" }
    );
    println!(
        "  Weight sorted: {} errors {}",
        sorted_errors,
        if sorted_errors == 0 { "✓" } else { "✗" }
    );
    println!(
        "  No duplicates: {} errors {}",
        duplicate_errors,
        if duplicate_errors == 0 { "✓" } else { "✗" }
    );

    // 3.2 COVERAGE
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3.2 COVERAGE / SPARSITY                                     │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let non_empty = lens.iter().filter(|&&l| l > 0).count();
    let coverage = (non_empty as f64 / vocab_size as f64) * 100.0;
    println!(
        "  Coverage: {}/{} ({:.1}%)",
        non_empty, vocab_size, coverage
    );

    let mut histogram = vec![0usize; top_n + 1];
    for &len in &lens {
        histogram[len.min(top_n)] += 1;
    }
    println!("\n  Histogram:");
    for (len, count) in histogram.iter().enumerate() {
        if *count > 0 {
            let pct = (*count as f64 / vocab_size as f64) * 100.0;
            println!("    len={:2}: {:5} ({:5.1}%)", len, count, pct);
        }
    }

    // 3.3 PROBE TEST
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ 3.3 PROBE SANITY CHECK                                      │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let vocab: Vec<String> = BufReader::new(File::open("vi.syllable.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    let probes = [
        "tôi", "anh", "em", "chị", "ông", "bà", "là", "có", "được", "không", "đã", "sẽ", "đang",
        "muốn", "cần", "biết", "thấy", "nói", "làm", "rất", "quá", "lắm", "hơn", "nhất", "và",
        "nhưng", "vì", "nếu", "thì", "đây", "đó", "này", "kia", "gì", "sao", "đâu", "nào", "ai",
    ];

    println!("  Testing {} probes:\n", probes.len());

    for probe in &probes {
        if let Some(id) = vocab.iter().position(|w| w == *probe) {
            let idx_offset = header_size + id * 8;
            let offset = u32::from_le_bytes([
                data[idx_offset],
                data[idx_offset + 1],
                data[idx_offset + 2],
                data[idx_offset + 3],
            ]) as usize;
            let len = u16::from_le_bytes([data[idx_offset + 4], data[idx_offset + 5]]) as usize;

            if len == 0 {
                println!("  {:10} → (no edges)", probe);
            } else {
                let mut top5 = Vec::new();
                for i in 0..len.min(5) {
                    let e_off = edges_base + offset + i * 8;
                    let next_id = u32::from_le_bytes([
                        data[e_off],
                        data[e_off + 1],
                        data[e_off + 2],
                        data[e_off + 3],
                    ]) as usize;
                    if let Some(w) = vocab.get(next_id) {
                        top5.push(w.as_str());
                    }
                }
                println!("  {:10} → {}", probe, top5.join(", "));
            }
        } else {
            println!("  {:10} → (not in vocab)", probe);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    let all_pass = magic_ok
        && version_ok
        && size_ok
        && offset_errors == 0
        && sorted_errors == 0
        && duplicate_errors == 0;
    println!(
        "  {} ALL FORMAT TESTS {}",
        if all_pass { "✅" } else { "❌" },
        if all_pass { "PASSED" } else { "FAILED" }
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
