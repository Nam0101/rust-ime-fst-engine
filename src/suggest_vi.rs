//! Vietnamese sentence suggestion demo
//!
//! Usage: cargo run --release --bin suggest_vi -- "tôi yêu"

use anyhow::Result;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} \"câu tiếng Việt\"", args[0]);
        eprintln!("Example: {} \"tôi yêu\"", args[0]);
        std::process::exit(1);
    }

    let sentence = args[1..].join(" ");

    // Load vocab
    let vocab: Vec<String> = BufReader::new(File::open("vi.syllable.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    let mut word_to_id: HashMap<String, usize> = HashMap::new();
    for (id, word) in vocab.iter().enumerate() {
        word_to_id.insert(word.to_lowercase(), id);
    }

    // Load bigram
    let bigram_file = File::open("vi.bigram.bin")?;
    let bigram_mmap = unsafe { Mmap::map(&bigram_file)? };
    let data = bigram_mmap.as_ref();

    let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let header_size = 32;
    let index_size = vocab_size * 8;
    let edges_base = header_size + index_size;

    // Get last syllable
    let syllables: Vec<&str> = sentence.split_whitespace().collect();
    if syllables.is_empty() {
        println!("Vui lòng nhập câu tiếng Việt");
        return Ok(());
    }

    let last_syllable = syllables.last().unwrap().to_lowercase();

    println!("Input: \"{}\"", sentence);
    println!("Âm tiết cuối: \"{}\"", last_syllable);
    println!();

    if let Some(&syllable_id) = word_to_id.get(&last_syllable) {
        let idx_offset = header_size + syllable_id * 8;
        let offset = u32::from_le_bytes([
            data[idx_offset],
            data[idx_offset + 1],
            data[idx_offset + 2],
            data[idx_offset + 3],
        ]) as usize;
        let len = u16::from_le_bytes([data[idx_offset + 4], data[idx_offset + 5]]) as usize;

        if len == 0 {
            println!("Không có gợi ý cho \"{}\"", last_syllable);
            return Ok(());
        }

        println!("Gợi ý sau \"{}\":", sentence);
        println!("─────────────────────────────");

        for i in 0..len {
            let e_off = edges_base + offset + i * 8;
            let next_id = u32::from_le_bytes([
                data[e_off],
                data[e_off + 1],
                data[e_off + 2],
                data[e_off + 3],
            ]) as usize;
            let weight = u16::from_le_bytes([data[e_off + 4], data[e_off + 5]]);

            if let Some(next_word) = vocab.get(next_id) {
                let confidence = (weight as f64 / 65535.0 * 100.0) as u32;
                println!("  {}. {} ({}%)", i + 1, next_word, confidence);
            }
        }

        println!();
        println!("Câu hoàn chỉnh:");
        for i in 0..len.min(5) {
            let e_off = edges_base + offset + i * 8;
            let next_id = u32::from_le_bytes([
                data[e_off],
                data[e_off + 1],
                data[e_off + 2],
                data[e_off + 3],
            ]) as usize;

            if let Some(next_word) = vocab.get(next_id) {
                println!("  → {} {}", sentence, next_word);
            }
        }
    } else {
        println!("Âm tiết \"{}\" không có trong từ điển", last_syllable);
    }

    Ok(())
}
