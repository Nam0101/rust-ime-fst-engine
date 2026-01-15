use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use unicode_normalization::UnicodeNormalization;

// --- Constants & Config ---
const USER_ID_START: u32 = 0x80000000;
const USER_ID_MAX: u32 = 0xFFFFFFF0; // Safety buffer
const HL_LEXICON_SEC: f64 = 14.0 * 24.0 * 3600.0; // 14 days
const HL_BIGRAM_SEC: f64 = 7.0 * 24.0 * 3600.0; // 7 days
const SCORE_SCALE: f64 = 10000.0;
const BONUS_ACCEPT: f64 = 3000.0;
const MAX_SCORE: f64 = 65535.0;

fn now_sec() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32
}

fn exp2_decay(age_sec: u32, half_life_sec: f64) -> f64 {
    if half_life_sec <= 0.0 {
        return 0.0;
    }
    2f64.powf(-(age_sec as f64) / half_life_sec)
}

// --- Data Structures ---

#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
pub struct WordStat {
    pub freq: u32,
    pub accept: u32,
    pub last_used: u32,
}

impl WordStat {
    pub fn touch_commit(&mut self, now: u32) {
        self.freq = self.freq.saturating_add(1);
        self.last_used = now;
    }
    pub fn touch_accept(&mut self, now: u32) {
        self.freq = self.freq.saturating_add(3);
        self.accept = self.accept.saturating_add(1);
        self.last_used = now;
    }

    pub fn score(&self, now: u32) -> u16 {
        let age = now.saturating_sub(self.last_used);
        let decay = exp2_decay(age, HL_LEXICON_SEC);
        let eff = (self.freq as f64) * decay;
        let base = (1.0 + eff).ln() * SCORE_SCALE;
        let accept = (self.accept as f64) * BONUS_ACCEPT;
        (base + accept).clamp(0.0, MAX_SCORE) as u16
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
pub struct EdgeStat {
    pub count: u32,
    pub last_used: u32,
}

impl EdgeStat {
    pub fn touch(&mut self, now: u32, delta: u32) {
        self.count = self.count.saturating_add(delta);
        self.last_used = now;
    }

    pub fn score(&self, now: u32) -> u16 {
        let age = now.saturating_sub(self.last_used);
        let decay = exp2_decay(age, HL_BIGRAM_SEC);
        let eff = (self.count as f64) * decay;
        let val = (1.0 + eff).ln() * SCORE_SCALE;
        val.clamp(0.0, MAX_SCORE) as u16
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct UserLexicon {
    word_to_id: HashMap<String, u32>,
    id_to_meta: HashMap<u32, (String, WordStat)>, // Store String here to easy reverse
    next_id: u32,
}

impl UserLexicon {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_meta: HashMap::new(),
            next_id: USER_ID_START,
        }
    }

    pub fn get_or_create(&mut self, word: &str, now: u32) -> Option<u32> {
        if let Some(&id) = self.word_to_id.get(word) {
            // Touch existing
            if let Some((_, stat)) = self.id_to_meta.get_mut(&id) {
                stat.touch_commit(now);
            }
            Some(id)
        } else {
            // Create new
            if self.next_id >= USER_ID_MAX {
                // Overflow protection: Refuse to add new words.
                // In real app, we should prune the lexicon here.
                return None;
            }
            let id = self.next_id;
            self.next_id += 1;

            let mut stat = WordStat::default();
            stat.touch_commit(now);

            self.word_to_id.insert(word.to_string(), id);
            self.id_to_meta.insert(id, (word.to_string(), stat));
            Some(id)
        }
    }

    pub fn get_word(&self, id: u32) -> Option<&str> {
        self.id_to_meta.get(&id).map(|(s, _)| s.as_str())
    }

    pub fn score(&self, id: u32, now: u32) -> u16 {
        self.id_to_meta
            .get(&id)
            .map(|(_, s)| s.score(now))
            .unwrap_or(0)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TopNTracker {
    // next_id -> Stat
    counts: HashMap<u32, EdgeStat>,
    #[serde(skip)]
    top_n: usize,
    #[serde(skip)]
    prune_threshold: usize,
}

impl Default for TopNTracker {
    fn default() -> Self {
        Self::new(20)
    }
}

impl TopNTracker {
    pub fn new(top_n: usize) -> Self {
        Self {
            counts: HashMap::new(),
            top_n: if top_n == 0 { 20 } else { top_n },
            prune_threshold: if top_n == 0 { 2000 } else { top_n * 100 },
        }
    }

    pub fn increment(&mut self, next_id: u32, delta: u32, now: u32) {
        self.counts
            .entry(next_id)
            .and_modify(|s| s.touch(now, delta))
            .or_insert_with(|| {
                let mut s = EdgeStat::default();
                s.touch(now, delta);
                s
            });

        if self.counts.len() > self.prune_threshold {
            self.prune(now);
        }
    }

    fn prune(&mut self, now: u32) {
        let keep = self.top_n * 2;
        if self.counts.len() <= keep {
            return;
        }

        let mut entries: Vec<(u32, EdgeStat)> = self.counts.drain().collect();
        // Sort by effective score
        entries.sort_by(|a, b| b.1.score(now).cmp(&a.1.score(now)));

        entries.truncate(keep);
        self.counts = entries.into_iter().collect();
    }

    pub fn get_top(&self, now: u32) -> Vec<(u32, u32)> {
        // returns (id, score) like original requirement or (id, raw_count)?
        // Requirement was "predict" returning suggestions.
        // Let's return (id, score_u16)
        let mut entries: Vec<(u32, u16)> = self
            .counts
            .iter()
            .map(|(&k, &v)| (k, v.score(now)))
            .collect();

        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(self.top_n);
        entries
            .into_iter()
            .map(|(id, score)| (id, score as u32))
            .collect() // User wanted u32 score API?
    }
}

#[derive(Serialize, Deserialize)]
pub struct UserHistory {
    lexicon: UserLexicon,
    // prev_id -> Tracker
    bigrams: HashMap<u32, TopNTracker>,
}

impl UserHistory {
    pub fn new() -> Self {
        Self {
            lexicon: UserLexicon::new(),
            bigrams: HashMap::new(),
        }
    }

    /// Learn from input text.
    /// `lookup_global`: Closure to resolve global IDs.
    pub fn learn<F>(&mut self, text: &str, lookup_global: F)
    where
        F: Fn(&str) -> Option<u32>,
    {
        let now = now_sec();
        let tokens = tokenize(text);
        let mut prev_id: Option<u32> = None;

        for token in tokens {
            let id = if let Some(gid) = lookup_global(&token) {
                gid
            } else {
                if let Some(uid) = self.lexicon.get_or_create(&token, now) {
                    uid
                } else {
                    // Lexicon full
                    prev_id = None;
                    continue;
                }
            };

            if let Some(pid) = prev_id {
                let tracker = self
                    .bigrams
                    .entry(pid)
                    .or_insert_with(|| TopNTracker::new(20));
                tracker.increment(id, 1, now);
            }
            prev_id = Some(id);
        }
    }

    pub fn predict(&self, prev_id: u32) -> Vec<(u32, u32)> {
        // (id, score)
        let now = now_sec();
        if let Some(tracker) = self.bigrams.get(&prev_id) {
            tracker.get_top(now)
        } else {
            Vec::new()
        }
    }

    /// Find user words starting with `prefix`
    pub fn lookup_prefix(&self, prefix: &str, limit: usize) -> Vec<(u32, u32)> {
        let now = now_sec();
        // Since UserLexicon is relatively small (thousands), linear scan is acceptable for now.
        // For larger lexicons, a Trie or FST should be used.
        let norm_prefix = normalize_token(prefix);
        if norm_prefix.is_empty() {
            return Vec::new();
        }

        let mut matches: Vec<(u32, u16)> = self
            .lexicon
            .id_to_meta
            .iter()
            .filter(|(_, (word, _))| word.starts_with(&norm_prefix))
            .map(|(&id, (_, stat))| (id, stat.score(now)))
            .collect();

        matches.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        matches.truncate(limit);

        matches.into_iter().map(|(id, s)| (id, s as u32)).collect()
    }

    pub fn get_user_word(&self, id: u32) -> Option<&str> {
        self.lexicon.get_word(id)
    }

    pub fn get_user_word_id(&self, word: &str) -> Option<u32> {
        self.lexicon.word_to_id.get(word).copied()
    }

    /// Save UserHistory to a JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let file = std::fs::File::create(path).context("Failed to create history file")?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self).context("Failed to serialize history")?;
        Ok(())
    }

    /// Load UserHistory from a JSON file. Returns empty if file doesn't exist or error.
    pub fn load(path: &str) -> Result<Self> {
        if !std::path::Path::new(path).exists() {
            return Ok(Self::new());
        }
        let file = std::fs::File::open(path).context("Failed to open history file")?;
        let reader = std::io::BufReader::new(file);
        let history = serde_json::from_reader(reader).context("Failed to deserialize history")?;
        Ok(history)
    }
}

/// Robust normalization and tokenization
fn normalize_token(raw: &str) -> String {
    let s = raw.nfc().collect::<String>();
    s.chars()
        .map(|c| if c == '’' || c == '‘' { '\'' } else { c })
        .flat_map(|c| c.to_lowercase())
        .filter(|c| c.is_alphabetic() || *c == '\'')
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalize_token)
        .filter(|s| !s.is_empty())
        .collect()
}
