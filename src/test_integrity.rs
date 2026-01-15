use anyhow::Result;
use fst::Map;
use memmap2::Mmap;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn main() -> Result<()> {
    let file = File::open("en.lex.fst")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let map = Map::new(mmap)?;

    let vocab: Vec<String> = BufReader::new(File::open("en.vocab.txt")?)
        .lines()
        .collect::<std::io::Result<_>>()?;

    println!("Loaded {} words from vocab", vocab.len());
    println!("Testing word_id ↔ vocab integrity...\n");

    let mut rng = StdRng::seed_from_u64(1);
    let mut passed = 0;
    let mut failed = 0;

    for _ in 0..1000 {
        let i = rng.gen_range(0..vocab.len());
        let key = &vocab[i];
        match map.get(key) {
            Some(v) => {
                let id = ((v >> 16) & 0xFFFF_FFFF) as usize;
                if id >= vocab.len() {
                    println!(
                        "FAIL: key={key} id={id} out of bounds (vocab.len={})",
                        vocab.len()
                    );
                    failed += 1;
                } else if vocab[id] != *key {
                    println!("FAIL: key={key} id={id} vocab[id]={}", vocab[id]);
                    failed += 1;
                } else {
                    passed += 1;
                }
            }
            None => {
                println!("FAIL: key={key} not found in FST");
                failed += 1;
            }
        }
    }

    println!("\nResults: {passed} passed, {failed} failed");
    if failed > 0 {
        anyhow::bail!("Integrity check failed with {failed} errors");
    }
    println!("OK: 1000 random id<->vocab checks passed.");
    Ok(())
}
