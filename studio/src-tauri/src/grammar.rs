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

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

use harper_core::language_detection::is_doc_likely_english;
use harper_core::linting::{Lint, LintGroup, LintKind, Linter};
use harper_core::parsers::PlainEnglish;
use harper_core::spell::FstDictionary;
use harper_core::{Dialect, Document};
use regex::Regex;

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

const NUMBER_WORDS: &str = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and|a)";
const ORDINAL_WORDS: &str = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty[- ]first|twenty[- ]second|twenty[- ]third|twenty[- ]fourth|twenty[- ]fifth|twenty[- ]sixth|twenty[- ]seventh|twenty[- ]eighth|twenty[- ]ninth|thirtieth|thirty[- ]first)";
const MONTH_WORDS: &str = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)";
const YEAR_WORDS: &str = r"(?:(?:nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)){1,2})";
const CURRENCY_WORDS: &str = r"(?:(?:pakistani|indian|us|american|australian|canadian|british|european)\s+)?(?:dollars?|euros?|pounds?|rupees?|dirhams?|riyals?|yen|yuan|rmb|won|taka|afghanis?|usd|eur|gbp|pkr|inr|aed|sar|cny|jpy|krw|bdt|afn)";

static CURRENCY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b((?:{NUMBER_WORDS})(?:\s+{NUMBER_WORDS}){{0,6}})\s+({CURRENCY_WORDS})\b"
    ))
    .expect("currency regex")
});
static PERCENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b((?:{NUMBER_WORDS})(?:\s+{NUMBER_WORDS}){{0,6}})\s+(?:percent|per\s+cent)\b"
    ))
    .expect("percent regex")
});
static TIME_WORDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s+(?:oh\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty))?)\s*(a\.?m\.?|p\.?m\.?)\b"
    ))
    .expect("time words regex")
});
static TIME_DIGITS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b")
        .expect("time digits regex")
});
static DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b({MONTH_WORDS})\s+({ORDINAL_WORDS}|\d{{1,2}}(?:st|nd|rd|th)?)(?:\s*,?\s*({YEAR_WORDS}|\d{{4}}))?\b"
    ))
    .expect("date regex")
});
static NUMBER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b((?:{NUMBER_WORDS})(?:\s+{NUMBER_WORDS}){{0,6}})\b"
    ))
    .expect("number regex")
});

fn format_number(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn number_word_value(word: &str) -> Option<u64> {
    Some(match word {
        "zero" => 0,
        "one" | "a" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

fn parse_number_words(phrase: &str) -> Option<u64> {
    let mut total = 0u64;
    let mut current = 0u64;
    let mut any = false;
    for raw in phrase.split_whitespace() {
        let word = raw.to_ascii_lowercase();
        match word.as_str() {
            "and" => continue,
            "hundred" => {
                current = current.max(1) * 100;
                any = true;
            }
            "thousand" | "million" | "billion" => {
                let scale = match word.as_str() {
                    "thousand" => 1_000,
                    "million" => 1_000_000,
                    _ => 1_000_000_000,
                };
                total = total.saturating_add(current.max(1).saturating_mul(scale));
                current = 0;
                any = true;
            }
            _ => {
                let value = number_word_value(&word)?;
                current = current.saturating_add(value);
                any = true;
            }
        }
    }
    any.then_some(total.saturating_add(current))
}

fn parse_year_like(phrase: &str) -> Option<u64> {
    let words: Vec<String> = phrase
        .split_whitespace()
        .filter(|word| !word.eq_ignore_ascii_case("and"))
        .map(|word| word.to_ascii_lowercase())
        .collect();
    if words.len() < 3 {
        return None;
    }
    let century = number_word_value(&words[0])?;
    if !(19..=29).contains(&century) {
        return None;
    }
    let rest: Vec<&str> = words[1..].iter().map(String::as_str).collect();
    let year = parse_number_words(&rest.join(" "))?;
    if year > 99 {
        return None;
    }
    let value = century * 100 + year;
    (1000..=2999).contains(&value).then_some(value)
}

fn parse_ordinal(phrase: &str) -> Option<u32> {
    let lower = phrase.to_ascii_lowercase().replace('-', " ");
    if let Ok(value) = lower
        .trim_end_matches(|ch: char| ch.is_ascii_alphabetic())
        .parse::<u32>()
    {
        return Some(value);
    }
    let cardinal = match lower.as_str() {
        "first" => "one",
        "second" => "two",
        "third" => "three",
        "fourth" => "four",
        "fifth" => "five",
        "sixth" => "six",
        "seventh" => "seven",
        "eighth" => "eight",
        "ninth" => "nine",
        "tenth" => "ten",
        "eleventh" => "eleven",
        "twelfth" => "twelve",
        "thirteenth" => "thirteen",
        "fourteenth" => "fourteen",
        "fifteenth" => "fifteen",
        "sixteenth" => "sixteen",
        "seventeenth" => "seventeen",
        "eighteenth" => "eighteen",
        "nineteenth" => "nineteen",
        "twentieth" => "twenty",
        "thirtieth" => "thirty",
        "thirty first" => "thirty one",
        other => other,
    };
    parse_number_words(cardinal).map(|value| value as u32)
}

fn currency_symbol(phrase: &str) -> (String, bool) {
    let lower = phrase.to_ascii_lowercase();
    let code = |value: &str| (value.to_string(), true);
    if lower.contains("pakistani") && lower.contains("rupee") {
        return code("PKR");
    }
    if lower.contains("indian") && lower.contains("rupee") {
        return code("INR");
    }
    if lower.contains("dirham") || lower == "aed" {
        return code("AED");
    }
    if lower.contains("riyal") || lower == "sar" {
        return code("SAR");
    }
    if lower.contains("rupee") {
        return code("Rs");
    }
    if lower.contains("dollar") || lower == "usd" {
        return ("$".to_string(), false);
    }
    if lower.contains("euro") || lower == "eur" {
        return ("€".to_string(), false);
    }
    if lower.contains("pound") || lower == "gbp" {
        return ("£".to_string(), false);
    }
    if lower.contains("yen") || lower == "jpy" {
        return ("¥".to_string(), false);
    }
    if lower.contains("yuan") || lower == "cny" || lower == "rmb" {
        return code("CNY");
    }
    if lower.contains("won") || lower == "krw" {
        return ("₩".to_string(), false);
    }
    if lower.contains("taka") || lower == "bdt" {
        return code("BDT");
    }
    if lower.contains("afghani") || lower == "afn" {
        return code("AFN");
    }
    (lower.to_ascii_uppercase(), true)
}

fn format_clock(hour: u32, minutes: u32, meridiem: &str) -> String {
    let meridiem = meridiem
        .replace('.', "")
        .to_ascii_uppercase();
    if minutes == 0 {
        format!("{hour} {meridiem}")
    } else {
        format!("{hour}:{minutes:02} {meridiem}")
    }
}

fn parse_word_clock(phrase: &str) -> Option<(u32, u32)> {
    let mut words = phrase.split_whitespace().map(|word| word.to_ascii_lowercase());
    let hour = number_word_value(&words.next()?)? as u32;
    if !(1..=12).contains(&hour) {
        return None;
    }
    let rest: Vec<String> = words.collect();
    if rest.is_empty() {
        return Some((hour, 0));
    }
    let mut minutes_text = rest.join(" ");
    let had_oh = minutes_text.starts_with("oh ");
    if had_oh {
        minutes_text = minutes_text.trim_start_matches("oh ").to_string();
    }
    let minutes = parse_number_words(&minutes_text)? as u32;
    if minutes > 59 {
        return None;
    }
    let minutes = if had_oh && minutes < 10 {
        minutes
    } else {
        minutes
    };
    Some((hour, minutes))
}

/// Normalise dictated numbers, currency, percentages, times and dates into
/// their conventional written forms. Returns the text and how many spans
/// changed, so the history can report a fix count.
fn normalize_numbers(text: &str) -> (String, usize) {
    let changes = Cell::new(0usize);

    let mut out = CURRENCY_RE
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let Some(value) = parse_number_words(&caps[1]) else {
                return caps[0].to_string();
            };
            let (symbol, spaced) = currency_symbol(&caps[2]);
            changes.set(changes.get() + 1);
            if spaced {
                format!("{symbol} {}", format_number(value))
            } else {
                format!("{symbol}{}", format_number(value))
            }
        })
        .into_owned();

    out = PERCENT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let Some(value) = parse_number_words(&caps[1]) else {
                return caps[0].to_string();
            };
            changes.set(changes.get() + 1);
            format!("{}%", format_number(value))
        })
        .into_owned();

    out = TIME_WORDS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let Some((hour, minutes)) = parse_word_clock(&caps[1]) else {
                return caps[0].to_string();
            };
            changes.set(changes.get() + 1);
            format_clock(hour, minutes, &caps[2])
        })
        .into_owned();

    out = TIME_DIGITS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let hour: u32 = caps[1].parse().unwrap_or(0);
            let minutes: u32 = caps.get(2).map(|m| m.as_str().parse().unwrap_or(0)).unwrap_or(0);
            if hour > 23 || minutes > 59 {
                return caps[0].to_string();
            }
            changes.set(changes.get() + 1);
            format_clock(hour, minutes, &caps[3])
        })
        .into_owned();

    out = DATE_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let month = {
                let mut chars = caps[1].chars();                match chars.next() {
                    Some(first) => {
                        format!("{}{}", first.to_uppercase(), chars.as_str().to_ascii_lowercase())
                    }
                    None => return caps[0].to_string(),
                }
            };
            let Some(day) = parse_ordinal(&caps[2]) else {
                return caps[0].to_string();
            };
            changes.set(changes.get() + 1);
            match caps.get(3) {
                Some(year_match) => {
                    let raw = year_match.as_str();
                    let year = if raw.chars().all(|ch| ch.is_ascii_digit()) {
                        raw.parse::<u64>().ok()
                    } else {
                        parse_year_like(raw).or_else(|| parse_number_words(raw))
                    };
                    match year {
                        Some(year) => format!("{month} {day}, {year}"),
                        None => format!("{month} {day}"),
                    }
                }
                None => format!("{month} {day}"),
            }
        })
        .into_owned();

    out = NUMBER_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let phrase = caps[1].to_ascii_lowercase();
            if phrase == "a" || phrase == "one" || phrase == "and" {
                return caps[0].to_string();
            }
            if let Some(year) = parse_year_like(&phrase) {
                changes.set(changes.get() + 1);
                return format!("{year}");
            }
            let Some(value) = parse_number_words(&phrase) else {
                return caps[0].to_string();
            };
            // Leave small standalone words alone ("one of the best"), but
            // convert quantities and any phrase with a scale word.
            let has_scale = phrase.contains("hundred")
                || phrase.contains("thousand")
                || phrase.contains("million")
                || phrase.contains("billion");
            let multi_word = phrase.split_whitespace().count() > 1;
            if !has_scale && !multi_word && value < 10 {
                return caps[0].to_string();
            }
            changes.set(changes.get() + 1);
            format_number(value)
        })
        .into_owned();

    (out, changes.get())
}

fn correct(text: &str, language: &str, custom_words: &[String]) -> (String, usize) {
    if text.trim().is_empty() || text.chars().count() > MAX_TEXT_CHARS {
        return (text.to_owned(), 0);
    }
    if !is_english_like(text, language) {
        return (text.to_owned(), 0);
    }
    let custom = custom_word_set(custom_words);
    let dialect = dialect_for(language);
    // Normalise numbers before linting: "twenty twenty six" must become 2026
    // before the repetition rule mistakes the year for a repeated word.
    let (normalized_text, number_fixes) = normalize_numbers(text);

    let mut guard = LINTERS.lock().unwrap_or_else(|err| err.into_inner());
    let group = guard
        .entry(dialect_key(dialect))
        .or_insert_with(|| build_linter(dialect));
    let dictionary = FstDictionary::curated();
    let document = Document::new(&normalized_text, &PlainEnglish, &dictionary);
    // Longer prose gets a dictionary-based language check so an English
    // grammar rule never rewrites a different language dictated with
    // language detection set to automatic. Short phrases skip it: one unknown
    // name would make them look foreign.
    if normalized_text.split_whitespace().count() > 7
        && !is_doc_likely_english(&document, &dictionary)
    {
        return (text.to_owned(), 0);
    }
    let mut lints = group.lint(&document);
    drop(guard);

    // Apply from the end so earlier offsets stay valid. Overlapping
    // suggestions are skipped: the rightmost one already covers that region.
    lints.retain(|lint| is_corrective(lint));
    lints.sort_by_key(|lint| std::cmp::Reverse(lint.span.start));

    let mut source: Vec<char> = normalized_text.chars().collect();
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

    let raw_fixed: String = source.into_iter().collect();
    let mut fixed: Vec<char> = raw_fixed.chars().collect();
    capitalize_start(&mut fixed);
    let fixed: String = fixed.into_iter().collect();
    if fixed.trim().is_empty() {
        (text.to_owned(), 0)
    } else {
        (fixed, applied + number_fixes)
    }
}

/// Result of one grammar pass. `fixes` is the number of applied suggestions.
#[derive(serde::Serialize)]
pub struct GrammarResult {
    text: String,
    fixes: usize,
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
) -> GrammarResult {
    // Linting is CPU work and the first call may build the dictionary; keep it
    // off the webview thread.
    let fallback = text.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let language = language.unwrap_or_default();
        let custom_words = custom_words.unwrap_or_default();
        std::panic::catch_unwind(|| correct(&text, &language, &custom_words))
            .map(|(text, fixes)| GrammarResult { text, fixes })
            .unwrap_or(GrammarResult {
                text,
                fixes: 0,
            })
    })
    .await
    .unwrap_or(GrammarResult {
        text: fallback,
        fixes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(text: &str) -> String {
        correct(text, "en", &[]).0
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
    fn converts_currency_and_numbers() {
        let rupees = fix("i have thousand pakistani rupee");
        assert!(rupees.contains("PKR 1,000"), "{rupees}");
        let dollars = fix("it will cost me fifty five dollars");
        assert!(dollars.contains("$55"), "{dollars}");
        let percent = fix("it is fifty percent done");
        assert!(percent.contains("50%"), "{percent}");
        let apples = fix("i bought twenty five apples");
        assert!(apples.contains("25 apples"), "{apples}");
    }

    #[test]
    fn converts_time_and_dates() {
        let time = fix("meet me at five thirty pm");
        assert!(time.contains("5:30 PM"), "{time}");
        let date = fix("today is january fifth twenty twenty six");
        assert!(date.contains("January 5, 2026"), "{date}");
    }

    #[test]
    fn leaves_code_tokens_alone() {
        let fixed = fix("we run llama.cpp and RTX4090 daily .");
        assert!(fixed.contains("llama.cpp"), "{fixed}");
        assert!(fixed.contains("RTX4090"), "{fixed}");
    }

    #[test]
    fn respects_custom_words() {
        let fixed = correct("unsloth is great", "en", &["Unsloth".to_string()]).0;
        assert!(fixed.contains("Unsloth"), "{fixed}");
    }

    #[test]
    fn skips_non_english() {
        assert!(!is_english_like("こんにちは世界", "ja"));
        assert!(!is_english_like("bonjour tout le monde", "fr"));
        assert!(is_english_like("hello there", "en-US"));
        assert!(is_english_like("hello there", "auto"));

        let spanish = "esto es una prueba de que el corrector no deberia cambiar nada";
        assert_eq!(correct(spanish, "auto", &[]), (spanish.to_owned(), 0));
    }
}
