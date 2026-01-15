use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use fst::MapBuilder;
use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::{BufRead, BufReader, Write},
};

fn parse_kv_csvish(s: &str) -> Vec<(&str, &str)> {
    // "word=the,f=222,flags=,originalFreq=222" -> [("word","the"), ("f","222"), ...]
    s.split(',')
        .filter_map(|p| p.split_once('='))
        .collect()
}

fn pack_value(prob_q: u8, flags: u8, word_id: u32) -> u64 {
    (prob_q as u64) | ((flags as u64) << 8) | ((word_id as u64) << 16)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.combined.gz> <out.lex.fst> [out.vocab.txt]", args[0]);
        std::process::exit(2);
    }
    let input_gz = &args[1];
    let out_fst = &args[2];
    let out_vocab = args.get(3);

    // Read gz line-by-line
    let f = File::open(input_gz).with_context(|| format!("open {}", input_gz))?;
    let gz = GzDecoder::new(f);
    let rd = BufReader::new(gz);

    // Use BTreeMap to keep keys sorted (fst::MapBuilder requires sorted inserts)
    let mut unigram: BTreeMap<String, u8> = BTreeMap::new();

    let mut saw_header = false;
    for line in rd.lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }

        // header line: dictionary=...,locale=...
        if !saw_header && t.starts_with("dictionary=") {
            saw_header = true;
            continue;
        }

        // The file has leading spaces before "word="
        if t.starts_with("word=") {
            let kv = parse_kv_csvish(t);
            let mut word: Option<&str> = None;
            let mut fval: Option<u16> = None;

            for (k, v) in kv {
                if k == "word" {
                    word = Some(v);
                } else if k == "f" {
                    fval = Some(v.parse::<u16>().unwrap_or(0));
                }
            }

            if let (Some(w), Some(fu16)) = (word, fval) {
                if w.is_empty() {
                    continue;
                }
                let prob_q = fu16.min(255) as u8;
                // keep max if duplicated
                unigram
                    .entry(w.to_string())
                    .and_modify(|old| *old = (*old).max(prob_q))
                    .or_insert(prob_q);
            }
        }
    }

    // Build FST
    let mut out = File::create(out_fst).with_context(|| format!("create {}", out_fst))?;
    let mut builder = MapBuilder::new(&mut out).context("fst MapBuilder")?;

    let mut vocab_writer: Option<File> = match out_vocab {
        Some(p) => Some(File::create(p).with_context(|| format!("create {}", p))?),
        None => None,
    };

    for (i, (w, prob_q)) in unigram.iter().enumerate() {
        let word_id = i as u32;

        let mut flags: u8 = 0;
        if *prob_q == 0 {
            flags |= 1 << 0; // IS_PROFANITY / nosuggest-like marker (your engine decides)
        }
        let v = pack_value(*prob_q, flags, word_id);
        builder.insert(w, v).with_context(|| format!("insert {}", w))?;

        if let Some(vw) = vocab_writer.as_mut() {
            writeln!(vw, "{w}")?;
        }
    }
    builder.finish().context("finish fst")?;
    Ok(())
}
