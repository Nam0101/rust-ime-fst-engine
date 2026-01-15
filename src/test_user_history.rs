mod user_history;
use std::collections::HashMap;
use user_history::UserHistory;

fn main() {
    println!("=== Test User History (Robust) ===");

    // 1. Setup Mock Global Map
    let mut global_map = HashMap::new();
    global_map.insert("hello".to_string(), 100);
    global_map.insert("world".to_string(), 101);
    global_map.insert("my".to_string(), 102);
    global_map.insert("name".to_string(), 103);
    global_map.insert("is".to_string(), 104);

    let mut history = UserHistory::new();
    let global_lookup = |s: &str| global_map.get(s).copied();

    // 2. Learn known words
    println!("\nLearning 'Hello World'...");
    history.learn("Hello World", global_lookup);

    // Predict after "Hello" (ID 100)
    let suggestions = history.predict(100);
    println!("Suggestions after 'Hello' (100): {:?}", suggestions);
    // suggestions is Vec<(id, score)>
    assert!(
        suggestions.iter().any(|&(id, _)| id == 101),
        "Should suggest 'World' (101)"
    );

    // 3. Learn NEW word "Gox"
    println!("\nLearning 'My name is Gox'...");
    history.learn("My name is Gox", global_lookup);

    // "Gox" should have a User ID (> 0x80000000)
    // "is" -> 104.
    let suggestions = history.predict(104);
    println!("Suggestions after 'is' (104): {:?}", suggestions);

    let (gox_id, score1) = suggestions.get(0).expect("Should have suggestion");
    assert!(*gox_id >= 0x80000000, "Gox should have User ID");

    let word = history
        .get_user_word(*gox_id)
        .expect("Should resolve User ID");
    println!("Resolved ID {} -> '{}'", gox_id, word);
    assert_eq!(word, "gox");

    // 4. Learn repeated (Score should increase)
    println!("\nLearning 'My name is Gox' again...");
    history.learn("My name is Gox", global_lookup);

    let suggestions = history.predict(104);
    println!("Suggestions after 'is' (104): {:?}", suggestions);
    let (_, score2) = suggestions.iter().find(|(id, _)| *id == *gox_id).unwrap();

    println!("Score1: {}, Score2: {}", score1, score2);
    assert!(*score2 > *score1, "Score should increase on repetition");

    // 5. Test Tokenization (apostrophe)
    println!("\nLearning 'I don’t know'...");
    history.learn("I don’t know", global_lookup);
    // "don’t" -> "don't" (normalized)

    // "I" (new user word?)
    if let Some(i_id) = history.get_user_word_id("i") {
        let after_i = history.predict(i_id);
        println!("Suggestions after 'i': {:?}", after_i);
        // Expect "don't" (normalized)
        if let Some((dont_id, _)) = after_i.get(0) {
            let w = history.get_user_word(*dont_id).unwrap();
            println!("After 'i' -> '{}'", w);
            assert_eq!(w, "don't");
        }
    } else {
        println!("Word 'i' not found in user lexicon?");
    }

    // 6. Test Prefix Lookup ("nhỉi" -> "nhỉiii")
    println!("\nLearning 'nhỉiii' (teencode)...");
    history.learn("nhỉiii", global_lookup);

    let candidates = history.lookup_prefix("nhỉi", 5); // prefix "nhỉi"
    println!("Prefix search 'nhỉi': {:?}", candidates);

    // Validate
    assert!(!candidates.is_empty(), "Should find matches for 'nhỉi'");
    let (match_id, _) = candidates[0];
    let w = history.get_user_word(match_id).unwrap();
    println!("Found: '{}'", w);
    assert_eq!(w, "nhỉiii");

    println!("\nPASSED all tests!");
}
