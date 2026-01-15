//! Read and verify en.bigram.bin format
//!
//! Usage: cargo run --release --bin read_bigram [word]

use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader};

const MAGIC: u32 = 0x4247524D;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Memory-map the bigram file
    let file = File::open("en.bigram.bin")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data = mmap.as_ref();

    // Parse header
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let edges_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let top_n = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);

    println!("=== en.bigram.bin ===");
    println!(
        "Magic: 0x{:08X} ({})",
        magic,
        if magic == MAGIC { "OK" } else { "BAD" }
    );
    println!("Version: {}", version);
    println!("Vocab size: {}", vocab_size);
    println!("Edges count: {}", edges_count);
    println!("Top-N: {}", top_n);

    let header_size = 32;
    let index_size = (vocab_size as usize) * 8;
    let edges_size = (edges_count as usize) * 8;

    println!("\nLayout:");
    println!("  Header: {} bytes", header_size);
    println!("  Index:  {} bytes ({} entries)", index_size, vocab_size);
    println!("  Edges:  {} bytes ({} entries)", edges_size, edges_count);
    println!("  Total:  {} bytes", header_size + index_size + edges_size);

    // Load vocab for reverse lookup
    let vocab: Vec<String> = BufReader::new(File::open("en.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    // If word argument given, look it up
    if let Some(word) = args.get(1) {
        println!("\n=== Lookup: '{}' ===", word);

        // Find word_id
        let lower = word.to_lowercase();
        let word_id = vocab.iter().position(|w| w.to_lowercase() == lower);

        match word_id {
            Some(id) => {
                println!("word_id: {}", id);

                // Read index entry
                let index_offset = header_size + id * 8;
                let offset = u32::from_le_bytes([
                    data[index_offset],
                    data[index_offset + 1],
                    data[index_offset + 2],
                    data[index_offset + 3],
                ]);
                let len = u16::from_le_bytes([data[index_offset + 4], data[index_offset + 5]]);

                println!("Index: offset={}, len={}", offset, len);

                if len > 0 {
                    println!("\nNext words:");
                    let edges_base = header_size + index_size;
                    for i in 0..len as usize {
                        let edge_offset = edges_base + offset as usize + i * 8;
                        let next_id = u32::from_le_bytes([
                            data[edge_offset],
                            data[edge_offset + 1],
                            data[edge_offset + 2],
                            data[edge_offset + 3],
                        ]);
                        let weight =
                            u16::from_le_bytes([data[edge_offset + 4], data[edge_offset + 5]]);

                        let next_word = vocab
                            .get(next_id as usize)
                            .map(|s| s.as_str())
                            .unwrap_or("<unknown>");

                        println!(
                            "  {:2}. {} (id={}, weight={})",
                            i + 1,
                            next_word,
                            next_id,
                            weight
                        );
                    }
                } else {
                    println!("No bigram data for this word");
                }
            }
            None => {
                println!("Word not found in vocabulary");
            }
        }
    } else {
        // Show some sample entries
        println!("\n=== Sample entries ===");
        let sample_words = ["the", "i", "you", "hello", "want", "need", "going"];

        for word in sample_words {
            if let Some(id) = vocab.iter().position(|w| w.to_lowercase() == word) {
                let index_offset = header_size + id * 8;
                let len = u16::from_le_bytes([data[index_offset + 4], data[index_offset + 5]]);
                println!("  '{}' (id={}): {} next words", word, id, len);
            }
        }
    }

    Ok(())
}
