use anyhow::Result;
use fst::automaton::{Automaton, Str};
use fst::{IntoStreamer, Map, Streamer};
use memmap2::Mmap;
use std::fs::File;

fn main() -> Result<()> {
    println!("=== Testing vi.syllable.fst ===\n");
    test_fst(
        "vi.syllable.fst",
        &["xin", "chào", "việt", "nam", "tôi", "yêu"],
    )?;

    println!("\n=== Testing vi.phrase.fst ===\n");
    test_fst(
        "vi.phrase.fst",
        &["xin chào", "việt nam", "an toàn", "anh hùng"],
    )?;

    // Prefix search test
    println!("\n=== Prefix search in vi.phrase.fst ===\n");
    {
        let file = File::open("vi.phrase.fst")?;
        let mmap = unsafe { Mmap::map(&file)? };
        let map = Map::new(mmap)?;

        let prefix = Str::new("an ").starts_with();
        let mut stream = map.search(prefix).into_stream();

        println!("Phrases starting with 'an ':");
        let mut count = 0;
        while let Some((k, v)) = stream.next() {
            if count < 15 {
                let word = std::str::from_utf8(k)?;
                let id = (v >> 16) & 0xFFFF_FFFF;
                println!("  {} (id={})", word, id);
            }
            count += 1;
        }
        println!("  ... total {} matches", count);
    }

    println!("\n=== Prefix search in vi.syllable.fst ===\n");
    {
        let file = File::open("vi.syllable.fst")?;
        let mmap = unsafe { Mmap::map(&file)? };
        let map = Map::new(mmap)?;

        let prefix = Str::new("ngu").starts_with();
        let mut stream = map.search(prefix).into_stream();

        println!("Syllables starting with 'ngu':");
        while let Some((k, _)) = stream.next() {
            println!("  {}", std::str::from_utf8(k)?);
        }
    }

    Ok(())
}

fn test_fst(path: &str, keys: &[&str]) -> Result<()> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let map = Map::new(mmap)?;

    println!("FST: {} ({} keys)", path, map.len());

    for key in keys {
        match map.get(*key) {
            Some(v) => {
                let id = (v >> 16) & 0xFFFF_FFFF;
                let prob = v & 0xFF;
                println!("  '{}' → id={}, prob={}", key, id, prob);
            }
            None => {
                println!("  '{}' → NOT FOUND", key);
            }
        }
    }

    Ok(())
}
