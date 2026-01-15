use anyhow::{Context, Result};
use combined2fst::build_canonical_map;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<()> {
    // 1. Setup Models
    println!("Loading models...");
    let (_, canonical_map) = build_canonical_map("en.lex.fst", "en.vocab.txt")?;

    let vocab: Vec<String> = BufReader::new(File::open("en.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    // DEBUG: Verify mapping
    println!("Verifying ID mapping for common words:");
    let sample_words = [
        "the", "of", "and", "a", "to", "him", "her", "she", "looked", "want", "go",
    ];
    for w in sample_words {
        if let Some(&id) = canonical_map.get(w) {
            if let Some(vw) = vocab.get(id as usize) {
                println!("  '{}' -> ID {} -> Vocab '{}'", w, id, vw);
            } else {
                println!("  '{}' -> ID {} -> OOB", w, id);
            }
        } else {
            println!("  '{}' -> Not in map", w);
        }
    }
    if let Some(w) = vocab.get(0) {
        println!("ID 0 = '{}'", w);
    }
    if let Some(w) = vocab.get(1) {
        println!("ID 1 = '{}'", w);
    }

    let bigram_file = File::open("en.bigram.bin")?;
    let bigram_mmap = unsafe { Mmap::map(&bigram_file)? };

    let trigram_mmap = match File::open("en.trigram.cache.bin") {
        Ok(f) => Some(unsafe { Mmap::map(&f)? }),
        Err(_) => None,
    };

    // 2. Define Test Sentences
    let sentences = vec![
        "I want to go to the store",
        "What are you doing here",
        "I don't know what to say",
        "She looked at him and smiled",
        "This is the best day of my life",
        "Can you tell me the time",
        "I am looking for a job",
        "Do you have a question",
        "I think that it is good",
        "We need to talk about it",
    ];

    // 3. Open Output CSV
    let mut file = File::create("hybrid_test_results.csv")?;
    writeln!(
        file,
        "Sentence_ID,Step,Input_Context,Last_Word,Model_Used,Top_Suggestions"
    )?;

    println!("Running tests...");

    for (s_idx, sent) in sentences.iter().enumerate() {
        let words: Vec<&str> = sent.split_whitespace().collect();
        let normalized: Vec<String> = words
            .iter()
            .map(|w| {
                w.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphabetic() || *c == '\'')
                    .collect()
            })
            .collect();

        for i in 1..words.len() {
            let context = &normalized[0..i];
            let context_str = words[0..i].join(" ");
            let w2 = context.last().unwrap();

            let mut model_used = "None";
            let mut suggestions = Vec::new();

            // Try Trigram
            let mut found_trigram = false;
            if let Some(tri_mmap) = &trigram_mmap {
                if context.len() >= 2 {
                    let w1 = &context[context.len() - 2];

                    if let (Some(&id1), Some(&id2)) = (canonical_map.get(w1), canonical_map.get(w2))
                    {
                        if let Some(results) = lookup_trigram(tri_mmap, id1, id2, &vocab) {
                            if !results.is_empty() {
                                model_used = "Trigram";
                                suggestions = results;
                                found_trigram = true;
                            }
                        }
                    }
                }
            }

            // Fallback Bigram
            if !found_trigram {
                if let Some(&id) = canonical_map.get(w2) {
                    if let Some(results) = lookup_bigram(bigram_mmap.as_ref(), id, &vocab) {
                        model_used = "Bigram";
                        suggestions = results;
                    }
                }
            }

            // Apply Gating
            apply_gating(&mut suggestions);

            // Format suggestions
            let sugg_str = suggestions
                .iter()
                .take(5)
                .map(|(w, p)| format!("{}({})", w, p))
                .collect::<Vec<_>>()
                .join(", ");

            writeln!(
                file,
                "{},{},\"{}\",\"{}\",{},\"{}\"",
                s_idx + 1,
                i,
                context_str,
                w2,
                model_used,
                sugg_str
            )?;
        }
    }

    println!("Done! Results exported to hybrid_test_results.csv");
    Ok(())
}

fn apply_gating(suggestions: &mut Vec<(String, u16)>) {
    let boost_words = [
        "to", "for", "are", "is", "of", "the", "a", "in", "on", "that",
    ];
    let mut boosted = Vec::new();
    let mut others = Vec::new();
    for (w, s) in suggestions.drain(..) {
        if boost_words.contains(&w.as_str()) {
            boosted.push((w, s));
        } else {
            others.push((w, s));
        }
    }
    suggestions.extend(boosted);
    suggestions.extend(others);
}

// Helpers
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
