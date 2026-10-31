// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! On-device grammar correction for voice typing.
//!
//! Backed by the `harper-core` rule engine (Apache-2.0): a curated, data-driven
//! set of English grammar, spelling, punctuation and word-choice linters. It
//! runs in a few milliseconds on a transcript and needs no model download or
//! network call, so dictation stays fast and private.
//!
//! The linter is built off the UI thread and reused for the life of the app
//! (one instance per English dialect). User dictionary words are filtered after
//! linting instead of rebuilding the linter, so a custom-spelling edit never
//! adds latency. Only conservative lint categories are applied: suggestions
//! that merely restyle prose (enhancement, readability, style, regionalism) are
//! ignored.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

use harper_core::language_detection::is_doc_likely_english;
use harper_core::linting::{Lint, LintGroup, LintKind, Linter};
use harper_core::parsers::PlainEnglish;
use harper_core::spell::FstDictionary;
use harper_core::{Dialect, Document};

/// Transcripts longer than this are left untouched: dictation is short-form
/// text, and an unbounded lint pass is not worth the latency.
const MAX_TEXT_CHARS: usize = 20_000;
/// Hard cap on applied suggestions, mirroring the short-form use case.
const MAX_APPLIED_LINTS: usize = 200;

static LINTERS: LazyLock<Mutex<HashMap<u8, LintGroup>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Build the linter ahead of the first dictation so the ~0.7 s dictionary
/// construction never lands in a press-to-insert path.
pub fn init() {
    std::thread::spawn(|| {
        let mut guard = LINTERS.lock().unwrap_or_else(|err| err.into_inner());
        guard
            .entry(dialect_key(Dialect::American))
            .or_insert_with(|| build_linter(Dialect::American));
        eprintln!("[grammar] on-device grammar linter ready");
    });
}

fn build_linter(dialect: Dialect) -> LintGroup {
    LintGroup::new_curated(FstDictionary::curated(), dialect)
}

/// Lint categories that fix genuine mistakes without rewriting the speaker's
/// voice. Ordering and style-only categories are deliberately excluded.
fn is_corrective(lint: &Lint) -> bool {
    match lint.lint_kind {
        LintKind::Agreement
        | LintKind::BoundaryError
        | LintKind::Capitalization
        | LintKind::Eggcorn
        | LintKind::Grammar
        | LintKind::Malapropism
        | LintKind::Punctuation
        | LintKind::Repetition
        | LintKind::Spelling
        | LintKind::Typo
        | LintKind::Usage
        | LintKind::WordChoice
        | LintKind::WordOrder => true,
        // Harper files the a/an rule under Miscellaneous. Match its message so
        // other miscellaneous rules stay untouched.
        LintKind::Miscellaneous => lint.message.contains("indefinite article"),
        // Only the sentence-spacing fix is safe; list/heading formatting is not.
        LintKind::Formatting => lint.message.contains("Unnecessary space"),
        _ => false,
    }
}

/// True when the transcript is plausibly English, matching the TypeScript
/// fixer's guard so a non-English dictation is never edited.
fn is_english_like(text: &str, language: &str) -> bool {
    let base = language
        .split(['-', '_'])
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !base.is_empty() && base != "en" && base != "auto" {
        return false;
    }
    let mut letters = 0usize;
    let mut ascii = 0usize;
    for ch in text.chars().filter(|ch| ch.is_alphabetic()) {
        letters += 1;
        if ch.is_ascii() {
            ascii += 1;
        }
    }
    letters > 0 && (ascii as f32 / letters as f32) >= 0.9
}

fn dialect_for(language: &str) -> Dialect {
    let tag = language.trim().to_ascii_lowercase();
    let tag = tag.split('_').next().unwrap_or(&tag);
    match tag {
        "en-gb" => Dialect::British,
        "en-au" => Dialect::Australian,
        "en-ca" => Dialect::Canadian,
        "en-in" => Dialect::Indian,
        _ => Dialect::American,
    }
}

fn dialect_key(dialect: Dialect) -> u8 {
    match dialect {
        Dialect::American => 1,
        Dialect::Canadian => 2,
        Dialect::Australian => 4,
        Dialect::British => 8,
        Dialect::Indian => 16,
    }
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// The whitespace-delimited token around a lint, trimmed of surrounding
/// punctuation. Lints that touch code, paths, versions or URLs are skipped so
/// `llama.cpp` and `RTX4090` are never "corrected" into prose words.
fn safe_token(source: &[char], start: usize, end: usize) -> Option<Vec<char>> {
    let mut left = start.min(source.len());
    let mut right = end.min(source.len());
    while left > 0 && !source[left - 1].is_whitespace() {
        left -= 1;
    }
    while right < source.len() && !source[right].is_whitespace() {
        right += 1;
    }
    let mut core = &source[left..right];
    while core
        .first()
        .is_some_and(|ch| !is_word_char(*ch))
    {
        core = &core[1..];
    }
    while core
        .last()
        .is_some_and(|ch| !is_word_char(*ch))
    {
        core = &core[..core.len() - 1];
    }
    if core.is_empty() {
        return None;
    }
    let mut all_upper = true;
    let mut has_letter = false;
    let mut has_space = false;
    for ch in core {
        if *ch == ' ' {
            has_space = true;
            continue;
        }
        if ch.is_ascii_digit()
            || matches!(
                *ch,
                '.' | '_' | '/' | '\\' | '@' | ':' | '=' | '#' | '&' | '+' | '*' | '<' | '>'
            )
        {
            return None;
        }
        if ch.is_alphabetic() {
            has_letter = true;
            if !ch.is_uppercase() {
                all_upper = false;
            }
        }
    }
    if !has_letter {
        return None;
    }
    // Acronyms such as "GGUF" or "CUDA" are intentional even when Harper's
    // dictionary does not know them. A multi-word span (a repeated phrase) is
    // never an acronym.
    if all_upper && !has_space && core.len() >= 2 {
        return None;
    }
    Some(core.to_vec())
}

/// Dictation engines sometimes omit the capital letter at the very start.
/// Capitalise it unless the first token looks like code (so `llama.cpp` and
/// `RTX4090` keep their casing).
fn capitalize_start(source: &mut [char]) {
    let Some(start) = source.iter().position(|ch| ch.is_alphabetic()) else {
        return;
    };
    if source[..start]
        .iter()
        .any(|ch| !ch.is_whitespace() && !matches!(*ch, '"' | '\'' | '(' | '[' | '{'))
    {
        return;
    }
    let end = source[start..]
        .iter()
        .position(|ch| ch.is_whitespace())
        .map(|offset| start + offset)
        .unwrap_or(source.len());
    let token = &source[start..end];
    if token.iter().any(|ch| {
        ch.is_ascii_digit()
            || matches!(
                *ch,
                '.' | '_' | '/' | '\\' | '@' | ':' | '=' | '#' | '&' | '+' | '*' | '<' | '>'
            )
    }) {
        return;
    }
    if source[start].is_lowercase() {
        if let Some(upper) = source[start].to_uppercase().next() {
            source[start] = upper;
        }
    }
}

fn custom_word_set(custom_words: &[String]) -> HashSet<String> {
    let mut set = HashSet::new();
    for word in custom_words {
        let trimmed = word.trim();
        if trimmed.is_empty() {
            continue;
        }
        set.insert(trimmed.to_lowercase());
        for part in trimmed.split_whitespace() {
            let part = part.trim_matches(|ch: char| !ch.is_alphanumeric());
            if !part.is_empty() {
                set.insert(part.to_lowercase());
            }
        }
    }
    set
}

fn correct(text: &str, language: &str, custom_words: &[String]) -> String {
    if text.trim().is_empty() || text.chars().count() > MAX_TEXT_CHARS {
        return text.to_owned();
    }
    if !is_english_like(text, language) {
        return text.to_owned();
    }
    let custom = custom_word_set(custom_words);
    let dialect = dialect_for(language);

    let mut guard = LINTERS.lock().unwrap_or_else(|err| err.into_inner());
    let group = guard
        .entry(dialect_key(dialect))
        .or_insert_with(|| build_linter(dialect));
    let dictionary = FstDictionary::curated();
    let document = Document::new(text, &PlainEnglish, &dictionary);
    // Longer prose gets a dictionary-based language check so an English
    // grammar rule never rewrites a different language dictated with
    // language detection set to automatic. Short phrases skip it: one unknown
    // name would make them look foreign.
    if text.split_whitespace().count() > 7 && !is_doc_likely_english(&document, &dictionary) {
        return text.to_owned();
    }
    let mut lints = group.lint(&document);
    drop(guard);

    // Apply from the end so earlier offsets stay valid. Overlapping
    // suggestions are skipped: the rightmost one already covers that region.
    lints.retain(|lint| is_corrective(lint));
    lints.sort_by_key(|lint| std::cmp::Reverse(lint.span.start));

    let mut source: Vec<char> = text.chars().collect();
    let mut applied_start = usize::MAX;
    let mut applied = 0usize;
    for lint in lints {
        if applied >= MAX_APPLIED_LINTS {
            break;
        }
        if lint.span.end > source.len() || lint.span.start >= lint.span.end {
            continue;
        }
        // Overlap with a suggestion already applied further right.
        if lint.span.end > applied_start {
            continue;
        }
        let Some(token) = safe_token(&source, lint.span.start, lint.span.end) else {
            continue;
        };
        if custom.contains(&token.iter().collect::<String>().to_lowercase()) {
            continue;
        }
        let Some(suggestion) = lint.suggestions.first() else {
            continue;
        };
        suggestion.apply(lint.span, &mut source);
        applied_start = lint.span.start;
        applied += 1;
    }

    let mut fixed: Vec<char> = source.into_iter().collect();
    capitalize_start(&mut fixed);
    let fixed: String = fixed.into_iter().collect();
    if fixed.trim().is_empty() {
        text.to_owned()
    } else {
        fixed
    }
}

/// Correct grammar, spelling and punctuation in an English dictation.
///
/// Returns the original text when it is not English, is empty, or linting
/// fails, so a correction can never lose a transcript.
#[tauri::command]
pub async fn fix_grammar(
    text: String,
    language: Option<String>,
    custom_words: Option<Vec<String>>,
) -> String {
    // Linting is CPU work and the first call may build the dictionary; keep it
    // off the webview thread.
    let fallback = text.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let language = language.unwrap_or_default();
        let custom_words = custom_words.unwrap_or_default();
        std::panic::catch_unwind(|| correct(&text, &language, &custom_words)).unwrap_or(text)
    })
    .await
    .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(text: &str) -> String {
        correct(text, "en", &[])
    }

    #[test]
    fn fixes_articles_and_capitalization() {
        let fixed = fix("this is a apple .");
        assert!(fixed.starts_with("This"), "{fixed}");
        assert!(fixed.contains("an apple"), "{fixed}");
        assert!(!fixed.contains(" ."), "{fixed}");
    }

    #[test]
    fn fixes_repeated_words() {
        let fixed = fix("I I went to the the store.");
        assert!(!fixed.contains("I I"), "{fixed}");
        assert!(!fixed.contains("the the"), "{fixed}");
        assert!(fixed.starts_with("I "), "{fixed}");
    }

    #[test]
    fn fixes_word_choice() {
        let fixed = fix("i went too the store");
        assert!(fixed.contains("to the store"), "{fixed}");
    }

    #[test]
    fn leaves_code_tokens_alone() {
        let fixed = fix("we run llama.cpp and RTX4090 daily .");
        assert!(fixed.contains("llama.cpp"), "{fixed}");
        assert!(fixed.contains("RTX4090"), "{fixed}");
    }

    #[test]
    fn respects_custom_words() {
        let fixed = correct("unsloth is great", "en", &["Unsloth".to_string()]);
        assert!(fixed.contains("Unsloth"), "{fixed}");
    }

    #[test]
    fn skips_non_english() {
        assert!(!is_english_like("こんにちは世界", "ja"));
        assert!(!is_english_like("bonjour tout le monde", "fr"));
        assert!(is_english_like("hello there", "en-US"));
        assert!(is_english_like("hello there", "auto"));

        let spanish = "esto es una prueba de que el corrector no deberia cambiar nada";
        assert_eq!(correct(spanish, "auto", &[]), spanish);
    }
}
