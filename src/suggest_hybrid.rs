use anyhow::{Context, Result};
use combined2fst::build_canonical_map;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} \"sentence...\"", args[0]);
        std::process::exit(1);
    }

    let sentence = args[1..].join(" ");

    // 1. Build Canonical Map (consistent with build_trigram)
    println!("Loading vocabulary and building canonical map...");
    let (_, canonical_map) = build_canonical_map("en.lex.fst", "en.vocab.txt")?;

    // Also load vocab list for printing results string
    let vocab: Vec<String> = BufReader::new(File::open("en.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    // 2. Load Bigram
    println!("Loading bigram model...");
    let bigram_file = File::open("en.bigram.bin")?;
    let bigram_mmap = unsafe { Mmap::map(&bigram_file)? };
    let bigram_data = bigram_mmap.as_ref();

    // 3. Load Trigram (Optional, if exists)
    let trigram_data = match File::open("en.trigram.cache.bin") {
        Ok(f) => {
            println!("Loading trigram cache...");
            Some(unsafe { Mmap::map(&f)? })
        }
        Err(_) => {
            println!("No trigram cache found (en.trigram.cache.bin). Using bigram only.");
            None
        }
    };

    // Parse input
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.is_empty() {
        return Ok(());
    }

    let normalized_words: Vec<String> = words
        .iter()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic() || *c == '\'')
                .collect()
        })
        .collect();

    println!("\nQuery: \"{}\"", sentence);

    let mut final_suggestions: Vec<(String, u16)> = Vec::new();
    let mut source = "None";

    // Try Trigram first if we have at least 2 words
    let mut found = false;
    if let Some(tri_mmap) = &trigram_data {
        if normalized_words.len() >= 2 {
            let w1_ids = canonical_map.get(&normalized_words[normalized_words.len() - 2]);
            let w2_ids = canonical_map.get(&normalized_words[normalized_words.len() - 1]);

            if let (Some(&id1), Some(&id2)) = (w1_ids, w2_ids) {
                if let Some(results) = lookup_trigram(tri_mmap, id1, id2, &vocab) {
                    if !results.is_empty() {
                        final_suggestions = results;
                        source = "Trigram";
                        found = true;
                    }
                }
            }
        }
    }

    // Fallback to Bigram
    if !found {
        let last_word = normalized_words.last().unwrap();
        if let Some(&id) = canonical_map.get(last_word) {
            if let Some(results) = lookup_bigram(bigram_data, id, &vocab) {
                final_suggestions = results;
                source = "Bigram";
                found = true;
            }
        }
    }

    // Apply Gating / Boosting
    if found {
        apply_gating(&mut final_suggestions);

        println!("\n[{}] Suggestions:", source);
        for (i, (word, score)) in final_suggestions.iter().enumerate() {
            println!("  {}. {} (prob: {})", i + 1, word, score);
        }

        if source == "Trigram" {
            println!(
                "(High confidence context: ... {} {})",
                normalized_words[normalized_words.len() - 2],
                normalized_words[normalized_words.len() - 1]
            );
        }
    } else {
        println!("No suggestions found.");
    }

    Ok(())
}

fn apply_gating(suggestions: &mut Vec<(String, u16)>) {
    let boost_words = [
        "to", "for", "are", "is", "of", "the", "a", "in", "on", "that",
    ];

    // Find matching indices
    let mut indices: Vec<usize> = Vec::new();
    for (i, (w, _)) in suggestions.iter().enumerate() {
        if boost_words.contains(&w.as_str()) {
            indices.push(i);
        }
    }

    // Move them to top, maintaining relative order among themselves?
    // Or just move them to front.
    // Let's move them to front in the order they appear (so highest prob boosts first).
    // Actually, usually we want common words to appear if they are REASONABLY probable.
    // If they are deep in the list (low prob), strictly boosting them might be wrong contextually.
    // But user asked to "Add gating... để top-3 nhìn 'đúng IME' hơn".
    // I will simply move them to the top if present.

    // Extract boosted items
    let mut boosted = Vec::new();
    let mut others = Vec::new();

    for (w, s) in suggestions.drain(..) {
        if boost_words.contains(&w.as_str()) {
            boosted.push((w, s));
        } else {
            others.push((w, s));
        }
    }

    // Put boosted first
    suggestions.extend(boosted);
    suggestions.extend(others);

    // Keep top results only? No, display all.
}

fn lookup_trigram(data: &[u8], w1: u32, w2: u32, vocab: &[String]) -> Option<Vec<(String, u16)>> {
    let num_pairs = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let header_size = 32;

    let mut low = 0;
    let mut high = num_pairs;

    while low < high {
        let mid = low + (high - low) / 2;
        let entry_offset = header_size + mid * 16;

        let mw1 = u32::from_le_bytes([
            data[entry_offset],
            data[entry_offset + 1],
            data[entry_offset + 2],
            data[entry_offset + 3],
        ]);
        let mw2 = u32::from_le_bytes([
            data[entry_offset + 4],
            data[entry_offset + 5],
            data[entry_offset + 6],
            data[entry_offset + 7],
        ]);

        match (mw1, mw2).cmp(&(w1, w2)) {
            std::cmp::Ordering::Equal => {
                let edges_start_offset = u32::from_le_bytes([
                    data[entry_offset + 8],
                    data[entry_offset + 9],
                    data[entry_offset + 10],
                    data[entry_offset + 11],
                ]) as usize;
                let len =
                    u16::from_le_bytes([data[entry_offset + 12], data[entry_offset + 13]]) as usize;

                let edges_base = header_size + num_pairs * 16;
                let mut results = Vec::new();

                for i in 0..len {
                    let off = edges_base + edges_start_offset + i * 8;
                    let next_id = u32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]);
                    let weight = u16::from_le_bytes([data[off + 4], data[off + 5]]);

                    if let Some(w) = vocab.get(next_id as usize) {
                        results.push((w.clone(), weight));
                    }
                }
                return Some(results);
            }
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
        }
    }

    None
}

fn lookup_bigram(data: &[u8], w_id: u32, vocab: &[String]) -> Option<Vec<(String, u16)>> {
    let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let header_size = 32;
    let index_offset = header_size + (w_id as usize) * 8;

    if index_offset
        .checked_add(8)
        .map_or(true, |end| end > header_size + vocab_size * 8)
    {
        return None;
    }
    if index_offset + 6 > data.len() {
        return None;
    }

    let edges_offset = u32::from_le_bytes([
        data[index_offset],
        data[index_offset + 1],
        data[index_offset + 2],
        data[index_offset + 3],
    ]) as usize;

    let len = u16::from_le_bytes([data[index_offset + 4], data[index_offset + 5]]) as usize;

    if len == 0 {
        return None;
    }

    let edges_base = header_size + vocab_size * 8;
    let mut results = Vec::new();

    for i in 0..len {
        let off = edges_base + edges_offset + i * 8;
        if off + 6 > data.len() {
            break;
        }

        let next_id = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let weight = u16::from_le_bytes([data[off + 4], data[off + 5]]);

        if let Some(w) = vocab.get(next_id as usize) {
            results.push((w.clone(), weight));
        }
    }

    Some(results)
}
