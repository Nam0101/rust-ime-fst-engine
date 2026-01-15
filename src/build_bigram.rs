use anyhow::Result;
use flate2::read::GzDecoder;
use fst::{Map, MapBuilder};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.txt.gz> [--top N]", args[0]);
        eprintln!("  Extracts bigrams from OpenSubtitles and filters by en.vocab.txt");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let top_n: usize = args
        .iter()
        .position(|a| a == "--top")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10); // top 10 next words per word

    // Load vocabulary FST
    println!("Loading en.lex.fst...");
    let fst_file = File::open("en.lex.fst")?;
    let mmap = unsafe { Mmap::map(&fst_file)? };
    let vocab_fst = Map::new(mmap)?;
    println!("  {} words in vocabulary", vocab_fst.len());

    // Count bigrams: (word_id1, word_id2) -> count
    let mut bigram_counts: HashMap<(u32, u32), u64> = HashMap::new();

    println!("\nReading {}...", input_path);
    let file = File::open(input_path)?;
    let reader: Box<dyn BufRead> = if input_path.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut lines_processed = 0u64;
    let mut prev_word_id: Option<u32> = None;

    for line in reader.lines() {
        let line = line?;
        lines_processed += 1;

        if lines_processed % 1_000_000 == 0 {
            println!(
                "  {} million lines, {} unique bigrams",
                lines_processed / 1_000_000,
                bigram_counts.len()
            );
        }

        // Tokenize line
        for word in line.split_whitespace() {
            // Normalize: lowercase, strip punctuation
            let normalized: String = word
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic() || *c == '\'')
                .collect();

            if normalized.is_empty() {
                prev_word_id = None;
                continue;
            }

            // Look up in vocabulary
            if let Some(v) = vocab_fst.get(&normalized) {
                let word_id = ((v >> 16) & 0xFFFF_FFFF) as u32;

                if let Some(prev_id) = prev_word_id {
                    *bigram_counts.entry((prev_id, word_id)).or_insert(0) += 1;
                }

                prev_word_id = Some(word_id);
            } else {
                // Word not in vocab, break bigram chain
                prev_word_id = None;
            }
        }

        // End of line breaks the chain
        prev_word_id = None;
    }

    println!("\nProcessed {} lines", lines_processed);
    println!("Found {} unique bigrams", bigram_counts.len());

    // Build per-word top-N lists
    println!("\nBuilding top-{} next-word lists...", top_n);

    // Group by first word
    let mut by_first: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
    for ((w1, w2), count) in bigram_counts {
        by_first.entry(w1).or_default().push((w2, count));
    }

    // Sort and truncate to top-N
    let mut bigram_data: Vec<(u32, Vec<u32>)> = Vec::new();
    for (w1, mut nexts) in by_first {
        nexts.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending
        nexts.truncate(top_n);
        let next_ids: Vec<u32> = nexts.into_iter().map(|(id, _)| id).collect();
        bigram_data.push((w1, next_ids));
    }
    bigram_data.sort_by_key(|(w1, _)| *w1);

    println!("  {} words have bigram data", bigram_data.len());

    // Write binary format
    // Format: [num_entries: u32] [entries...]
    // Each entry: [word_id: u32] [num_next: u8] [next_ids: u32...]
    let mut out = BufWriter::new(File::create("en.bigram.bin")?);

    out.write_all(&(bigram_data.len() as u32).to_le_bytes())?;

    for (w1, nexts) in &bigram_data {
        out.write_all(&w1.to_le_bytes())?;
        out.write_all(&[nexts.len() as u8])?;
        for next_id in nexts {
            out.write_all(&next_id.to_le_bytes())?;
        }
    }

    drop(out);

    let file_size = std::fs::metadata("en.bigram.bin")?.len();
    println!(
        "\n✓ en.bigram.bin created ({:.1} MB)",
        file_size as f64 / 1_000_000.0
    );

    // Also build FST index for fast lookup by word_id
    println!("\nBuilding en.bigram.fst index...");
    {
        let file = BufWriter::new(File::create("en.bigram.fst")?);
        let mut builder = MapBuilder::new(file)?;

        // Key: word as string, Value: offset in .bin file
        // This is a simplified approach - we use word_id as a 4-byte key
        let mut offset: u64 = 4; // Skip header
        for (w1, nexts) in &bigram_data {
            // Store as string key for FST compatibility
            let key = format!("{:08x}", w1);
            builder.insert(key.as_bytes(), offset)?;
            offset += 4 + 1 + (nexts.len() as u64 * 4); // word_id + count + next_ids
        }
        builder.finish()?;
    }
    println!("✓ en.bigram.fst index created");

    Ok(())
}
