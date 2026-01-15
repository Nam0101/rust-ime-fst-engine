use anyhow::Result;
use fst::MapBuilder;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};

#[derive(serde::Deserialize)]
struct WordEntry {
    text: String,
    #[allow(dead_code)]
    source: String,
}

fn main() -> Result<()> {
    let input = BufReader::new(File::open("words.txt")?);

    let mut phrases: BTreeMap<String, u64> = BTreeMap::new();
    let mut syllables: HashSet<String> = HashSet::new();

    println!("Reading words.txt...");

    for (idx, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let entry: WordEntry = serde_json::from_str(&line)?;
        let text = entry.text.to_lowercase();

        // Add to phrases (use index as word_id, no frequency available)
        // Value format: word_id (32 bits) | flags (8 bits) | prob (8 bits)
        // prob = 128 (default), flags = 0
        let word_id = idx as u64;
        let prob: u64 = 128;
        let flags: u64 = 0;
        let value = (word_id << 16) | (flags << 8) | prob;

        phrases.entry(text.clone()).or_insert(value);

        // Extract syllables (split by space)
        for syllable in text.split_whitespace() {
            // Clean syllable (remove hyphens at edges)
            let clean = syllable.trim_matches('-');
            if !clean.is_empty() && clean.chars().all(|c| c.is_alphabetic() || c == '-') {
                syllables.insert(clean.to_string());
            }
        }
    }

    println!("Found {} unique phrases", phrases.len());
    println!("Found {} unique syllables", syllables.len());

    // Build phrase FST
    println!("\nBuilding vi.phrase.fst...");
    {
        let file = BufWriter::new(File::create("vi.phrase.fst")?);
        let mut builder = MapBuilder::new(file)?;

        for (key, value) in &phrases {
            builder.insert(key.as_bytes(), *value)?;
        }
        builder.finish()?;
    }
    println!("✓ vi.phrase.fst created");

    // Build syllable FST (sorted)
    println!("\nBuilding vi.syllable.fst...");
    {
        let mut sorted_syllables: Vec<_> = syllables.into_iter().collect();
        sorted_syllables.sort();

        let file = BufWriter::new(File::create("vi.syllable.fst")?);
        let mut builder = MapBuilder::new(file)?;

        for (idx, syllable) in sorted_syllables.iter().enumerate() {
            // Value: syllable_id (32 bits) | flags (8 bits) | prob (8 bits)
            let value = ((idx as u64) << 16) | 128;
            builder.insert(syllable.as_bytes(), value)?;
        }
        builder.finish()?;

        // Also write vocab file
        let mut vocab = BufWriter::new(File::create("vi.syllable.vocab.txt")?);
        use std::io::Write;
        for s in &sorted_syllables {
            writeln!(vocab, "{}", s)?;
        }
        println!(
            "✓ vi.syllable.fst created ({} syllables)",
            sorted_syllables.len()
        );
    }

    // Write phrase vocab
    {
        let mut vocab = BufWriter::new(File::create("vi.phrase.vocab.txt")?);
        use std::io::Write;
        for (key, _) in &phrases {
            writeln!(vocab, "{}", key)?;
        }
        println!("✓ vi.phrase.vocab.txt created");
    }

    println!("\nDone! Files created:");
    println!("  - vi.phrase.fst");
    println!("  - vi.phrase.vocab.txt");
    println!("  - vi.syllable.fst");
    println!("  - vi.syllable.vocab.txt");

    Ok(())
}
