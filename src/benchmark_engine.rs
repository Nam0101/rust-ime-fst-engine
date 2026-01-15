use anyhow::{Context, Result};
use combined2fst::build_canonical_map;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Benchmark: Vietnamese Suggestion Engine ===");

    // 1. Load Models
    let start_load = Instant::now();

    println!("Loading Vocabulary & Canonical Map...");
    // Note: Adjust filenames if needed based on what's available
    let fst_path = "vi.phrase.fst";
    let vocab_path = "vi.phrase.vocab.txt";
    let bigram_path = "vi.bigram.bin";
    let trigram_path = "vi.trigram.cache.bin";

    let (_, canonical_map) =
        build_canonical_map(fst_path, vocab_path).context("Failed to load FST/Vocab map")?;

    let vocab: Vec<String> = BufReader::new(File::open(vocab_path)?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    println!("Loading Bigram Model...");
    let bigram_file = File::open(bigram_path).context("Failed to open bigram")?;
    let bigram_mmap = unsafe { Mmap::map(&bigram_file)? };

    println!("Loading Trigram Cache...");
    let trigram_mmap = match File::open(trigram_path) {
        Ok(f) => Some(unsafe { Mmap::map(&f)? }),
        Err(_) => {
            println!("Warning: No trigram cache found.");
            None
        }
    };

    println!("Models loaded in {:.2?}", start_load.elapsed());

    // 2. Test Cases
    let test_phrases = vec![
        "tôi muốn",
        "hôm nay trời",
        "bạn có",
        "chúng ta sẽ",
        "làm việc",
        "đi học",
        "phát triển",
        "công nghệ",
        "người việt",
        "thành phố hồ",
    ];

    println!(
        "\n=== Running Benchmarks ({} phrases) ===\n",
        test_phrases.len()
    );

    let mut latencies = Vec::new();

    for phrase in &test_phrases {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        // Simulate typing the last word? Or next word prediction?
        // Let's assume prediction given the context `phrase`

        let normalized: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();

        let start_predict = Instant::now();

        // Predict logic
        let mut found_suggestions = Vec::new();
        let mut source = "None";

        // Try Trigram (Last 2 words)
        if let Some(tri_mmap) = &trigram_mmap {
            if normalized.len() >= 2 {
                let w1 = &normalized[normalized.len() - 2];
                let w2 = &normalized[normalized.len() - 1]; // Last word is context

                if let (Some(&id1), Some(&id2)) = (canonical_map.get(w1), canonical_map.get(w2)) {
                    if let Some(results) = lookup_trigram(tri_mmap, id1, id2, &vocab) {
                        if !results.is_empty() {
                            found_suggestions = results;
                            source = "Trigram";
                        }
                    }
                }
            }
        }

        // Fallback Bigram (Last 1 word)
        if found_suggestions.is_empty() {
            if let Some(last_word) = normalized.last() {
                if let Some(&id) = canonical_map.get(last_word) {
                    if let Some(results) = lookup_bigram(bigram_mmap.as_ref(), id, &vocab) {
                        found_suggestions = results;
                        source = "Bigram";
                    }
                }
            }
        }

        if !found_suggestions.is_empty() {
            apply_gating(&mut found_suggestions);
        }

        let duration = start_predict.elapsed();
        latencies.push(duration);

        let top_3: Vec<String> = found_suggestions
            .iter()
            .take(3)
            .map(|(w, _)| w.clone())
            .collect();
        println!(
            "Input: {:20} | Time: {:<10?} | Source: {:<7} | Top 3: {:?}",
            phrase, duration, source, top_3
        );
    }

    // 3. Stats
    let total_duration: std::time::Duration = latencies.iter().sum();
    let avg_latency = total_duration / latencies.len() as u32;
    let max_latency = latencies.iter().max().unwrap();
    let min_latency = latencies.iter().min().unwrap();

    println!("\n=== Summary ===");
    println!("Total Predictions: {}", latencies.len());
    println!("Avg Latency:       {:.2?}", avg_latency);
    println!("Min Latency:       {:.2?}", min_latency);
    println!("Max Latency:       {:.2?}", max_latency);

    Ok(())
}

fn apply_gating(suggestions: &mut Vec<(String, u16)>) {
    let boost_words = [
        "là", "của", "và", "có", "những", "trong", "được", "một", "cho", "với",
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
