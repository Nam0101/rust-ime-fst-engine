use anyhow::{Context, Result};
use fst::Map;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Build canonical lowercase -> best word_id map
///
/// Logic:
/// 1. If exact lowercase match exists in FST, use it.
/// 2. Else, use the case variant with highest probability.
pub fn build_canonical_map(
    fst_path: &str,
    vocab_path: &str,
) -> Result<(u32, HashMap<String, u32>)> {
    let file = File::open(fst_path).context("Failed to open FST")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let fst = Map::new(mmap)?;

    let vocab_file = BufReader::new(File::open(vocab_path).context("Failed to open vocab")?);
    let mut canonical: HashMap<String, (u32, u8, bool)> = HashMap::new(); // lower -> (best_id, best_prob, is_exact)
    let mut vocab_size: u32 = 0;

    for line in vocab_file.lines() {
        let word = line?;
        vocab_size += 1;

        if let Some(v) = fst.get(&word) {
            let word_id = ((v >> 16) & 0xFFFF_FFFF) as u32;
            let prob = (v & 0xFF) as u8;
            let lower = word.to_lowercase();
            let is_exact = word == lower;

            canonical
                .entry(lower)
                .and_modify(|(best_id, best_prob, best_exact)| {
                    // If we already have an exact match, don't change unless this is also exact (unlikely for duplicate keys)
                    if *best_exact {
                        return;
                    }

                    // If this is exact, take it immediately
                    if is_exact {
                        *best_id = word_id;
                        *best_prob = prob;
                        *best_exact = true;
                        return;
                    }

                    // Otherwise, follow probability
                    if prob > *best_prob {
                        *best_id = word_id;
                        *best_prob = prob;
                    }
                })
                .or_insert((word_id, prob, is_exact));
        }
    }

    // Convert to simple id map
    let map: HashMap<String, u32> = canonical
        .into_iter()
        .map(|(k, (id, _, _))| (k, id))
        .collect();
    Ok((vocab_size, map))
}
