//! Interactive sentence suggestion demo
//!
//! Usage: cargo run --release --bin suggest -- "i love"

use anyhow::Result;
use combined2fst::build_canonical_map;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} \"sentence prefix\"", args[0]);
        eprintln!("Example: {} \"i love\"", args[0]);
        std::process::exit(1);
    }

    let sentence = args[1..].join(" ");

    // Load resources
    let vocab: Vec<String> = BufReader::new(File::open("en.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;
    let (_, canonical_map) = build_canonical_map("en.lex.fst", "en.vocab.txt")?;

    // Load bigram
    let bigram_file = File::open("en.bigram.bin")?;
    let bigram_mmap = unsafe { Mmap::map(&bigram_file)? };
    let bigram_data = bigram_mmap.as_ref();

    // Parse bigram header
    let vocab_size = u32::from_le_bytes([
        bigram_data[8],
        bigram_data[9],
        bigram_data[10],
        bigram_data[11],
    ]) as usize;
    let header_size = 32;
    let index_size = vocab_size * 8;
    let edges_base = header_size + index_size;

    // Get last word of sentence
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.is_empty() {
        println!("Please enter a sentence prefix");
        return Ok(());
    }

    let last_word = normalize_token(words.last().unwrap());
    if last_word.is_empty() {
        println!("Please enter a sentence prefix");
        return Ok(());
    }

    println!("Input: \"{}\"", sentence);
    println!("Last word: \"{}\"", last_word);
    println!();

    // Look up bigram suggestions
    if let Some(&word_id) = canonical_map.get(&last_word) {
        let idx_offset = header_size + (word_id as usize) * 8;
        let offset = u32::from_le_bytes([
            bigram_data[idx_offset],
            bigram_data[idx_offset + 1],
            bigram_data[idx_offset + 2],
            bigram_data[idx_offset + 3],
        ]) as usize;
        let len =
            u16::from_le_bytes([bigram_data[idx_offset + 4], bigram_data[idx_offset + 5]]) as usize;

        if len == 0 {
            println!("No suggestions for \"{}\"", last_word);
            return Ok(());
        }

        println!("Suggestions after \"{}\":", sentence);
        println!("─────────────────────────────");

        for i in 0..len {
            let e_off = edges_base + offset + i * 8;
            let next_id = u32::from_le_bytes([
                bigram_data[e_off],
                bigram_data[e_off + 1],
                bigram_data[e_off + 2],
                bigram_data[e_off + 3],
            ]) as usize;
            let weight = u16::from_le_bytes([bigram_data[e_off + 4], bigram_data[e_off + 5]]);

            if let Some(next_word) = vocab.get(next_id) {
                let confidence = (weight as f64 / 65535.0 * 100.0) as u32;
                println!(
                    "  {}. {} ({}%)",
                    i + 1,
                    next_word.to_lowercase(),
                    confidence
                );
            }
        }

        // Show completed sentences
        println!();
        println!("Complete sentences:");
        for i in 0..len.min(5) {
            let e_off = edges_base + offset + i * 8;
            let next_id = u32::from_le_bytes([
                bigram_data[e_off],
                bigram_data[e_off + 1],
                bigram_data[e_off + 2],
                bigram_data[e_off + 3],
            ]) as usize;

            if let Some(next_word) = vocab.get(next_id) {
                println!("  → {} {}", sentence, next_word.to_lowercase());
            }
        }
    } else {
        println!("Word \"{}\" not found in vocabulary", last_word);
    }

    Ok(())
}

fn normalize_token(word: &str) -> String {
    word.to_lowercase()
        .chars()
        .filter(|c| c.is_alphabetic() || *c == '\'')
        .collect()
}
