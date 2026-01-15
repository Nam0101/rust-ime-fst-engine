# Rust IME FST Engine

A high-performance Finite State Transducer (FST) based engine for keyboard input method suggestions.

## Features

- **FST-based vocabulary lookup** - O(k) lookup time with minimal memory footprint
- **Prefix search** - Fast autocomplete suggestions
- **Bigram support** - Next-word prediction from corpus data
- **Multi-language support** - English and Vietnamese

## Building

```bash
cargo build --release
```

## Tools

### Build English Lexicon FST
```bash
cargo run --release --bin combined2fst -- en_US_wordlist.combined.gz en.lex.fst en.vocab.txt
```

### Build Vietnamese FST (phrases + syllables)
```bash
cargo run --release --bin build_vi_fst
```
Creates: `vi.phrase.fst`, `vi.syllable.fst`, and vocab files.

### Build English Bigram from OpenSubtitles
```bash
# Download corpus first
curl -C - -L -o opensubtitles-en.txt.gz \
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz"

# Build bigram database
cargo run --release --bin build_bigram -- opensubtitles-en.txt.gz --top 10
```

### Test FST Files
```bash
cargo run --release --bin fst              # Test English FST
cargo run --release --bin test_vi_fst      # Test Vietnamese FST
cargo run --release --bin test_integrity   # Verify word_id <-> vocab mapping
```

## FST Value Format

Each FST entry stores a 64-bit value:
```
| word_id (32 bits) | flags (8 bits) | prob (8 bits) |
|     bits 16-47    |    bits 8-15   |    bits 0-7   |
```

## Data Files (not in repo)

- `en.lex.fst` - English lexicon FST
- `en.vocab.txt` - English vocabulary (sorted)
- `en.bigram.bin` - English bigram data
- `vi.phrase.fst` - Vietnamese phrase FST
- `vi.syllable.fst` - Vietnamese syllable FST

## License

MIT
