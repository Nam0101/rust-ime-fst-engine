use fst::automaton::{Automaton, Str};
use fst::{IntoStreamer, Map, Streamer};
use memmap2::Mmap;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let file = File::open("en.lex.fst")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let map = Map::new(mmap)?;

    // 1) get exact key
    for k in ["the", "and", "hello", "Android"] {
        if let Some(v) = map.get(k) {
            let prob_q = (v & 0xFF) as u8;
            let flags = ((v >> 8) & 0xFF) as u8;
            let word_id = ((v >> 16) & 0xFFFF_FFFF) as u32;
            println!("{k}: prob={prob_q} flags={flags} id={word_id} v=0x{v:016x}");
        } else {
            println!("{k}: (not found)");
        }
    }

    // 2) prefix search
    let prefix = Str::new("pre").starts_with();
    let mut s = map.search(prefix).into_stream();
    for _ in 0..10 {
        if let Some((k, v)) = s.next() {
            println!("pre*: {} -> {}", std::str::from_utf8(k)?, v);
        } else {
            break;
        }
    }
    Ok(())
}
