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
/// "10 by 12" is a size next to these words, not a ratio.
const DIMENSION_CONTEXT_WORDS: &[&str] = &[
    "room",
    "screen",
    "table",
    "desk",
    "board",
    "paper",
    "sheet",
    "photo",
    "image",
    "canvas",
    "window",
    "frame",
    "bed",
    "garden",
    "pool",
    "plot",
    "card",
    "poster",
    "tile",
    "panel",
    "display",
    "monitor",
    "wall",
    "size",
    "dimension",
    "wide",
    "length",
    "height",
];

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
static HYPHENATED_NUMBER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([a-z]+)-([a-z]+)\b").expect("hyphenated number regex")
});
static DIGIT_HYPHEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(\d{{2}})-(one|two|three|four|five|six|seven|eight|nine|\d)\b"
    ))
    .expect("digit hyphen regex")
});
static DECIMAL_WORDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b((?:{NUMBER_WORDS})(?:\s+{NUMBER_WORDS}){{0,4}})\s+point\s+((?:{NUMBER_WORDS})(?:\s+{NUMBER_WORDS}){{0,4}})\b"
    ))
    .expect("decimal words regex")
});
static DECIMAL_DIGITS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d+)\s+point\s+(\d+)\b").expect("decimal digits regex")
});
static NUMERIC_CURRENCY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(\d[\d,]*(?:\.\d+)?)\s+({CURRENCY_WORDS})\b"
    ))
    .expect("numeric currency regex")
});
static CURRENCY_HYPHEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d{2})-\$(\d{1,3}(?:,\d{3})+)\b")
        .expect("currency hyphen regex")
});
const CURRENCY_CODES: &str = "USD|EUR|GBP|INR|PKR|JPY|CNY|KRW|RUB|TRY|NGN|PHP|THB|VND|ILS|UAH|KZT|BDT|LKR|NPR|AED|SAR|QAR|KWD|BHD|OMR|CAD|AUD|NZD|CHF|SEK|NOK|DKK|PLN|CZK|HUF|RON|BGN|ZAR|MXN|BRL|ARS|CLP|COP|PEN|SGD|HKD|TWD";
static SPACED_CURRENCY_CODE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([A-Z])\s+([A-Z])\s+([A-Z])\b")
        .expect("spaced currency code regex")
});
static DOLLARS_CENTS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:\$\s*)?(\d[\d,]*)\s+(?:dollars?\s+)?(?:and\s+)?(\d{1,2})\s+cents?\b",
    )
    .expect("dollars cents regex")
});
static CENTS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d{1,3})\s+cents?\b").expect("cents regex")
});
static CURRENCY_CODE_SPLIT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b({CURRENCY_CODES})\s+(\d{{1,3}})\s+(\d{{1,3}}(?:,\d{{3}})+|\d{{3}})\b"
    ))
    .expect("currency code split regex")
});
static CURRENCY_CODE_AMOUNT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b({CURRENCY_CODES})\s+(\d[\d,]*(?:\.\d+)?)\b"
    ))
    .expect("currency code amount regex")
});
static SPLIT_RS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d[\d,]*)\s*\.\s*R'?s\.?\s*(\d[\d,]*(?:\.\d+)?)\b")
        .expect("split rupees regex")
});
static RS_AMOUNT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bR'?s\.?\s*(\d[\d,]*(?:\.\d+)?)\b")
        .expect("rupees amount regex")
});
static RUPEES_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\brupees?\s+(\d[\d,]*(?:\.\d+)?)\b")
        .expect("rupees prefix regex")
});
static NUMERIC_PERCENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s+(?:percent|per\s+cent)\b")
        .expect("numeric percent regex")
});
static ORDINAL_CONTEXT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b({ORDINAL_WORDS})\s+(position|place|rank|prize|grade|class|division|attempt|try|round|lap|spot|choice)\b"
    ))
    .expect("ordinal context regex")
});
static DOMAIN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b([a-z0-9][a-z0-9-]{1,})\s+dot\s+(com|org|net|io|ai|dev|co|app|cpp|js|py|rs)\b",
    )
    .expect("domain regex")
});
static KILO_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(\d+)\s+k\b").expect("kilo regex"));
static APPROX_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Only a quantity is shorthand-ed; ordinary prose keeps the word.
    Regex::new(r"(?i)\bapproximately\s+([$€£₹¥₩₽₺₦₱฿₫₪₴₸৳₨]?\d[\d,.]*)")
        .expect("approximately regex")
});
static EQUAL_TO_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:is\s+)?equal\s+to\b").expect("equal-to regex")
});
static TIMES_RE: LazyLock<Regex> = LazyLock::new(|| {
    // "3 times 2 years ago" is a cadence, not multiplication.
    Regex::new(
        r"(?i)\b(\d+)\s+times\s+(\d+)(?:\s+(years?|months?|weeks?|days?|hours?|minutes?|seconds?))?\b",
    )
    .expect("times regex")
});
static CURRENCY_RANGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    // "10 to 20 dollars" / "between 10 and 20 dollars": the unit applies to both.
    Regex::new(&format!(
        r"(?i)(?:[$€£₹¥₩₽₺₦₱฿₫₪₴₸৳₨]\s*)?(\d[\d,]*(?:\.\d+)?)\s+(to|and)\s+(?:[$€£₹¥₩₽₺₦₱฿₫₪₴₸৳₨]\s*)?(\d[\d,]*(?:\.\d+)?)\s+({CURRENCY_WORDS})\b"
    ))
    .expect("currency range regex")
});
static RATIO_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d+)\s+by\s+(\d+)\b").expect("ratio regex")
});
static DIMENSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(\d+)\s+by\s+(\d+)\s+(feet|foot|ft|inches|inch|in|meters?|metres?|m|cm|mm|yards?|miles?|km)\b",
    )
    .expect("dimension regex")
});
static MULTIPLIER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d+)\s*[xX]\s+(smaller|larger|bigger|faster|slower|more|less)\b")
        .expect("multiplier regex")
});
static VALUE_OF_PI_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bvalue\s+of\s+pi\b").expect("value-of-pi regex")
});
static PI_R_SQUARED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d+)?\s*pi\s+r\s+squared\b").expect("pi r squared regex")
});
static PI_R_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d+)?\s*pi\s+r\b").expect("pi r regex")
});
static JOIN_DAY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(was|were|is|are)\s*\.\s*({MONTH_WORDS}|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    ))
    .expect("join day regex")
});
static PI_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bpi\b").expect("pi word regex"));
static MODAL_CAP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i:(?:cannot|can|will|would|should|could|may|might|must|do|does|did))\s+([A-Z][a-z]+)\b",
    )
    .expect("modal capitalization regex")
});
static AUX_SENTENCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\.\s+(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|may|might|must)\b",
    )
    .expect("auxiliary sentence regex")
});
static PRONOUN_CONTRACTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(I|you|he|she|it|we|they)\s+(am|have|has|will|would|are|is)\b")
        .expect("pronoun contraction regex")
});
static NEGATION_CONTRACTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(do|does|did|is|are|was|were|have|has|had|will|would|could|should|must|can)\s+not\b",
    )
    .expect("negation contraction regex")
});
static CANNOT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bcannot\b").expect("cannot regex"));

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
        return ("Rs.".to_string(), true);
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
fn ordinal_suffix(value: u32) -> &'static str {
    if (11..=13).contains(&(value % 100)) {
        return "th";
    }
    match value % 10 {
        1 => "st",
        2 => "nd",
        3 => "rd",
        _ => "th",
    }
}

fn format_numeric_amount(raw: &str) -> String {
    let cleaned = raw.replace(',', "");
    match cleaned.split_once('.') {
        Some((integer, fraction)) => {
            let integer = integer.parse::<u64>().unwrap_or(0);
            format!("{}.{}", format_number(integer), fraction)
        }
        None => cleaned
            .parse::<u64>()
            .map(format_number)
            .unwrap_or_else(|_| cleaned),
    }
}

fn is_number_word(word: &str) -> bool {
    number_word_value(word).is_some()
        || matches!(word, "hundred" | "thousand" | "million" | "billion" | "and")
}

fn format_amount(value: f64) -> String {
    let rounded = (value * 100.0).round() / 100.0;
    let integer = rounded.trunc() as u64;
    let cents = ((rounded - integer as f64) * 100.0).round() as u64;
    if cents == 0 {
        format_number(integer)
    } else {
        format!("{}.{:02}", format_number(integer), cents)
    }
}

fn combine_amounts(first: &str, second: &str) -> Option<String> {
    let left: f64 = first.replace(',', "").parse().ok()?;
    let right: f64 = second.replace(',', "").parse().ok()?;
    // A split "fifty two thousand eight hundred..." arrives as "50 2,873":
    // the first part is the thousands bucket. A whole amount followed by a
    // decimal remainder ("58,000. R's 733.27") is a straightforward sum.
    let value = if !first.contains(',') && left < 1000.0 && !second.contains('.') {
        left * 1000.0 + right
    } else {
        left + right
    };
    Some(format_amount(value))
}

fn currency_code_symbol(code: &str) -> (String, bool) {
    match code.to_ascii_uppercase().as_str() {
        "USD" | "CAD" | "AUD" | "NZD" | "SGD" | "HKD" | "MXN" | "BRL"
        | "ARS" | "CLP" | "COP" | "PEN" => {
            let symbol = match code.to_ascii_uppercase().as_str() {
                "USD" => "$".to_string(),
                "CAD" => "C$".to_string(),
                "AUD" => "A$".to_string(),
                "NZD" => "NZ$".to_string(),
                "SGD" => "S$".to_string(),
                "HKD" => "HK$".to_string(),
                "MXN" => "MX$".to_string(),
                "BRL" => "R$".to_string(),
                "ARS" => "AR$".to_string(),
                "CLP" => "CLP".to_string(),
                "COP" => "COL$".to_string(),
                "PEN" => "S/".to_string(),
                _ => "$".to_string(),
            };
            (symbol, false)
        }
        "EUR" => ("€".to_string(), false),
        "GBP" => ("£".to_string(), false),
        "INR" => ("₹".to_string(), false),
        "JPY" | "CNY" => ("¥".to_string(), false),
        "KRW" => ("₩".to_string(), false),
        "RUB" => ("₽".to_string(), false),
        "TRY" => ("₺".to_string(), false),
        "NGN" => ("₦".to_string(), false),
        "PHP" => ("₱".to_string(), false),
        "THB" => ("฿".to_string(), false),
        "VND" => ("₫".to_string(), false),
        "ILS" => ("₪".to_string(), false),
        "UAH" => ("₴".to_string(), false),
        "KZT" => ("₸".to_string(), false),
        "BDT" => ("৳".to_string(), false),
        "LKR" | "NPR" => ("₨".to_string(), false),
        "ZAR" => ("R".to_string(), true),
        // PKR and the Gulf codes are clearer as ISO codes.
        _ => (code.to_ascii_uppercase(), true),
    }
}

fn normalize_currency_expressions(text: &str) -> (String, usize) {
    let changes = Cell::new(0usize);

    // The model sometimes spells a code as "P K R".
    let mut out = SPACED_CURRENCY_CODE_RE
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let code = format!(
                "{}{}{}",
                &caps[1].to_ascii_uppercase(),
                &caps[2].to_ascii_uppercase(),
                &caps[3].to_ascii_uppercase()
            );
            if CURRENCY_CODES.split('|').any(|known| known == code) {
                changes.set(changes.get() + 1);
                code
            } else {
                caps[0].to_string()
            }
        })
        .into_owned();

    // "58,000. R's 733.27" is one amount split by the speech model.
    out = SPLIT_RS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            match combine_amounts(&caps[1], &caps[2]) {
                Some(value) => {
                    changes.set(changes.get() + 1);
                    format!("Rs. {value}")
                }
                None => caps[0].to_string(),
            }
        })
        .into_owned();

    // "$87 14 cents" -> $87.14
    out = DOLLARS_CENTS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let dollars = caps[1].replace(',', "");
            let cents: u32 = caps[2].parse().unwrap_or(0);
            if dollars.parse::<u64>().is_err() || cents > 99 {
                return caps[0].to_string();
            }
            changes.set(changes.get() + 1);
            format!("${dollars}.{cents:02}")
        })
        .into_owned();

    // "13 cents" -> ¢13
    out = CENTS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("¢{}", &caps[1])
        })
        .into_owned();

    // "PKR 50 2,873" -> PKR 52,873
    out = CURRENCY_CODE_SPLIT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let code = caps[1].to_ascii_uppercase();
            match combine_amounts(&caps[2], &caps[3]) {
                Some(value) => {
                    changes.set(changes.get() + 1);
                    let (symbol, spaced) = currency_code_symbol(&code);
                    if spaced {
                        format!("{symbol} {value}")
                    } else {
                        format!("{symbol}{value}")
                    }
                }
                None => caps[0].to_string(),
            }
        })
        .into_owned();

    // "INR 27" -> ₹27, "PKR 47" -> PKR 47.
    out = CURRENCY_CODE_AMOUNT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let code = caps[1].to_ascii_uppercase();
            let amount = format_numeric_amount(&caps[2]);
            changes.set(changes.get() + 1);
            let (symbol, spaced) = currency_code_symbol(&code);
            if spaced {
                format!("{symbol} {amount}")
            } else {
                format!("{symbol}{amount}")
            }
        })
        .into_owned();

    // "R's 8,000" -> Rs. 8,000
    out = RS_AMOUNT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("Rs. {}", format_numeric_amount(&caps[1]))
        })
        .into_owned();

    out = RUPEES_PREFIX_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("Rs. {}", format_numeric_amount(&caps[1]))
        })
        .into_owned();

    (out, changes.get())
}

fn normalize_numbers(text: &str) -> (String, usize) {
    let (currency_text, currency_changes) = normalize_currency_expressions(text);
    let changes = Cell::new(currency_changes);

    // Repair the speech model's hyphenated or split number words before any
    // currency/number rule sees them: "twenty-five" -> "twenty five".
    let mut out = HYPHENATED_NUMBER_RE
        .replace_all(&currency_text, |caps: &regex::Captures<'_>| {
            let first = caps[1].to_ascii_lowercase();
            let second = caps[2].to_ascii_lowercase();
            if is_number_word(&first) && is_number_word(&second) {
                changes.set(changes.get() + 1);
                format!("{} {}", caps[1].trim(), caps[2].trim())
            } else {
                caps[0].to_string()
            }
        })
        .into_owned();

    // "80-nine" / "90-9%" are a tens digit plus a unit: 89 / 99.
    out = DIGIT_HYPHEN_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let first: u32 = caps[1].parse().unwrap_or(0);
            if first % 10 != 0 {
                return caps[0].to_string();
            }
            let second = caps[2]
                .parse::<u32>()
                .ok()
                .or_else(|| number_word_value(&caps[2].to_ascii_lowercase()).map(|v| v as u32));
            match second {
                Some(second) if second <= 9 => {
                    changes.set(changes.get() + 1);
                    format!("{}", first + second)
                }
                _ => caps[0].to_string(),
            }
        })
        .into_owned();

    // "five point two six" -> 5.26.
    out = DECIMAL_WORDS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let Some(integer) = parse_number_words(&caps[1]) else {
                return caps[0].to_string();
            };
            let fraction_words: Vec<String> = caps[2]
                .split_whitespace()
                .map(|word| word.to_ascii_lowercase())
                .collect();
            let fraction = fraction_words
                .iter()
                .map(|word| match number_word_value(word) {
                    Some(value) if value <= 9 => Some(value.to_string()),
                    _ => None,
                })
                .collect::<Option<String>>()
                .or_else(|| parse_number_words(&caps[2]).map(|value| value.to_string()));
            let Some(fraction) = fraction else {
                return caps[0].to_string();
            };
            changes.set(changes.get() + 1);
            format!("{}.{}", format_number(integer), fraction)
        })
        .into_owned();

    out = DECIMAL_DIGITS_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("{}.{}", &caps[1], &caps[2])
        })
        .into_owned();

    out = CURRENCY_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
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
            if !has_scale && !multi_word && value < 10 && phrase == "one" {
                return caps[0].to_string();
            }
            changes.set(changes.get() + 1);
            format_number(value)
        })
        .into_owned();

    // Digit forms the word passes cannot see.
    out = CURRENCY_HYPHEN_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let first: u64 = caps[1].parse().unwrap_or(0);
            let second: u64 = caps[2].replace(',', "").parse().unwrap_or(0);
            if first % 10 == 0 && second >= 1000 && second % 1000 == 0 {
                changes.set(changes.get() + 1);
                format!("${}", format_number(first * 1000 + second))
            } else {
                caps[0].to_string()
            }
        })
        .into_owned();

    out = CURRENCY_RANGE_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let (symbol, spaced) = currency_symbol(&caps[4]);
            let first = format_numeric_amount(&caps[1]);
            let second = format_numeric_amount(&caps[3]);
            changes.set(changes.get() + 1);
            if spaced {
                format!("{symbol} {first} {} {symbol} {second}", &caps[2])
            } else {
                format!("{symbol}{first} {} {symbol}{second}", &caps[2])
            }
        })
        .into_owned();

    out = NUMERIC_CURRENCY_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let (symbol, spaced) = currency_symbol(&caps[2]);
            changes.set(changes.get() + 1);
            if spaced {
                format!("{symbol} {}", format_numeric_amount(&caps[1]))
            } else {
                format!("{symbol}{}", format_numeric_amount(&caps[1]))
            }
        })
        .into_owned();

    out = NUMERIC_PERCENT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("{}%", caps[1].trim())
        })
        .into_owned();

    out = ORDINAL_CONTEXT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let Some(value) = parse_ordinal(&caps[1]) else {
                return caps[0].to_string();
            };
            changes.set(changes.get() + 1);
            format!("{value}{} {}", ordinal_suffix(value), &caps[2])
        })
        .into_owned();

    out = DOMAIN_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let word = caps[1].to_ascii_lowercase();
            if matches!(
                word.as_str(),
                "the"
                    | "and"
                    | "but"
                    | "for"
                    | "this"
                    | "that"
                    | "with"
                    | "from"
                    | "your"
                    | "our"
                    | "their"
                    | "his"
                    | "her"
                    | "my"
            ) {
                return caps[0].to_string();
            }
            changes.set(changes.get() + 1);
            format!("{}.{}", &caps[1], &caps[2].to_ascii_lowercase())
        })
        .into_owned();

    out = KILO_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("{}k", &caps[1])
        })
        .into_owned();

    out = APPROX_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            format!("~{}", &caps[1])
        })
        .into_owned();

    (out, changes.get())
}

fn modal_verb(word: &str) -> bool {
    matches!(
        word,
        "go" | "come"
            | "see"
            | "eat"
            | "do"
            | "make"
            | "take"
            | "get"
            | "give"
            | "know"
            | "think"
            | "say"
            | "tell"
            | "find"
            | "work"
            | "call"
            | "try"
            | "ask"
            | "need"
            | "feel"
            | "become"
            | "leave"
            | "put"
            | "mean"
            | "keep"
            | "let"
            | "begin"
            | "seem"
            | "help"
            | "talk"
            | "turn"
            | "start"
            | "show"
            | "hear"
            | "play"
            | "run"
            | "move"
            | "like"
            | "live"
            | "believe"
            | "hold"
            | "bring"
            | "happen"
            | "write"
            | "provide"
            | "sit"
            | "stand"
            | "lose"
            | "pay"
            | "meet"
            | "include"
            | "continue"
            | "set"
            | "learn"
            | "change"
            | "lead"
            | "understand"
            | "watch"
            | "follow"
            | "stop"
            | "create"
            | "speak"
            | "read"
            | "spend"
            | "grow"
            | "open"
            | "walk"
            | "win"
            | "offer"
            | "remember"
            | "love"
            | "consider"
            | "appear"
            | "buy"
            | "wait"
            | "serve"
            | "die"
            | "send"
            | "expect"
            | "build"
            | "stay"
            | "fall"
            | "cut"
            | "reach"
            | "kill"
            | "remain"
    )
}

fn join_auxiliary_sentences(text: &str) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut changes = 0usize;
    for caps in AUX_SENTENCE_RE.captures_iter(text) {
        let Some(full) = caps.get(0) else { continue };
        let aux = caps[1].to_ascii_lowercase();
        if !matches!(aux.as_str(), "is" | "are" | "was" | "were") {
            continue;
        }
        let prefix = &text[..full.start()];
        let sentence_start = prefix
            .rfind(['.', '!', '?', '\n'])
            .map(|index| index + 1)
            .unwrap_or(0);
        let sentence = &prefix[sentence_start..];
        let lower = sentence.to_ascii_lowercase();
        let joinable = lower.contains(", and ")
            || lower.contains(", but ")
            || lower.contains(", or ")
            || lower.starts_with("and ")
            || lower.starts_with("but ")
            || lower.starts_with("or ");
        if !joinable {
            continue;
        }
        out.push_str(&text[last..full.start()]);
        out.push(' ');
        out.push_str(&aux);
        last = full.end();
        changes += 1;
    }
    out.push_str(&text[last..]);
    (out, changes)
}

fn normalize_math(text: &str) -> (String, usize) {
    // Harper can mistake "Rs." for "R's."; re-run the currency rules after
    // linting so the amount keeps its conventional form.
    let (currency_fixed, currency_fixes) = normalize_currency_expressions(text);
    let changes = Cell::new(currency_fixes);
    let count = || changes.set(changes.get() + 1);

    let mut out = EQUAL_TO_RE
        .replace_all(&currency_fixed, |caps: &regex::Captures<'_>| {
            let start = caps.get(0).map(|m| m.start()).unwrap_or(0);
            let window_start = start.saturating_sub(40);
            let window_end = (start + 80).min(currency_fixed.len());
            let window = currency_fixed
                .get(window_start..window_end)
                .unwrap_or(&currency_fixed)
                .to_ascii_lowercase();
            let maths = window.chars().any(|ch| ch.is_ascii_digit())
                || [
                    "area",
                    "circumference",
                    "radius",
                    "value",
                    "sum",
                    "total",
                    "equation",
                    "formula",
                    "squared",
                ]
                .iter()
                .any(|word| window.contains(word));
            if !maths {
                return caps[0].to_string();
            }
            count();
            "=".to_string()
        })
        .into_owned();

    out = TIMES_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            if caps.get(3).is_some() {
                return caps[0].to_string();
            }
            count();
            format!("{}*{}", &caps[1], &caps[2])
        })
        .into_owned();

    out = DIMENSION_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            count();
            format!("{}x{} {}", &caps[1], &caps[2], &caps[3])
        })
        .into_owned();

    let ratio_source = out.clone();
    out = RATIO_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let start = caps.get(0).map(|m| m.start()).unwrap_or(0);
            let end = caps.get(0).map(|m| m.end()).unwrap_or(0);
            let before = ratio_source
                .get(start.saturating_sub(48)..start)
                .unwrap_or("")
                .to_ascii_lowercase();
            let after = ratio_source
                .get(end..(end + 48).min(ratio_source.len()))
                .unwrap_or("")
                .to_ascii_lowercase();
            let dimension = DIMENSION_CONTEXT_WORDS
                .iter()
                .any(|word| before.contains(word) || after.contains(word));
            count();
            if dimension {
                format!("{}x{}", &caps[1], &caps[2])
            } else {
                format!("{}/{}", &caps[1], &caps[2])
            }
        })
        .into_owned();

    out = MULTIPLIER_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            count();
            format!("{}x {}", &caps[1], &caps[2].to_ascii_lowercase())
        })
        .into_owned();

    out = VALUE_OF_PI_RE
        .replace_all(&out, |_caps: &regex::Captures<'_>| {
            count();
            "value of π".to_string()
        })
        .into_owned();

    out = PI_R_SQUARED_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            count();
            match caps.get(1) {
                Some(coefficient) => format!("{}πr^2", coefficient.as_str()),
                None => "πr^2".to_string(),
            }
        })
        .into_owned();

    out = PI_R_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            count();
            match caps.get(1) {
                Some(coefficient) => format!("{}πr", coefficient.as_str()),
                None => "πr".to_string(),
            }
        })
        .into_owned();

    let maths_context = out.contains('=')
        || out.to_ascii_lowercase().contains("circle")
        || out.to_ascii_lowercase().contains("area")
        || out.to_ascii_lowercase().contains("radius")
        || out.to_ascii_lowercase().contains("circumference")
        || out.to_ascii_lowercase().contains("squared")
        || out.contains('π');
    if maths_context {
        out = PI_WORD_RE
            .replace_all(&out, |_caps: &regex::Captures<'_>| {
                count();
                "π".to_string()
            })
            .into_owned();
    }

    out = JOIN_DAY_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            count();
            format!("{} {}", &caps[1], &caps[2])
        })
        .into_owned();

    out = MODAL_CAP_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let lower = caps[1].to_ascii_lowercase();
            if modal_verb(&lower) {
                count();
                let prefix = &caps[0][..caps[0].len() - caps[1].len()];
                format!("{prefix}{lower}")
            } else {
                caps[0].to_string()
            }
        })
        .into_owned();

    let (joined, joined_count) = join_auxiliary_sentences(&out);
    changes.set(changes.get() + joined_count);
    (joined, changes.get())
}

fn apply_case(original: &str, value: &str) -> String {
    if original.chars().next().is_some_and(char::is_uppercase) {
        let mut chars = value.chars();
        match chars.next() {
            Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
            None => value.to_string(),
        }
    } else {
        value.to_string()
    }
}

fn pronoun_contraction(pronoun: &str, verb: &str) -> Option<&'static str> {
    Some(match (pronoun, verb) {
        ("i", "am") => "i'm",
        ("i", "have") => "i've",
        ("i", "will") => "i'll",
        ("i", "would") => "i'd",
        ("you", "are") => "you're",
        ("you", "have") => "you've",
        ("you", "will") => "you'll",
        ("you", "would") => "you'd",
        ("he", "is") | ("he", "has") => "he's",
        ("she", "is") | ("she", "has") => "she's",
        ("it", "is") | ("it", "has") => "it's",
        ("we", "are") => "we're",
        ("we", "have") => "we've",
        ("we", "will") => "we'll",
        ("we", "would") => "we'd",
        ("they", "are") => "they're",
        ("they", "have") => "they've",
        ("they", "will") => "they'll",
        ("they", "would") => "they'd",
        _ => return None,
    })
}

fn negation_contraction(verb: &str) -> Option<&'static str> {
    Some(match verb {
        "do" => "don't",
        "does" => "doesn't",
        "did" => "didn't",
        "is" => "isn't",
        "are" => "aren't",
        "was" => "wasn't",
        "were" => "weren't",
        "have" => "haven't",
        "has" => "hasn't",
        "had" => "hadn't",
        "will" => "won't",
        "would" => "wouldn't",
        "could" => "couldn't",
        "should" => "shouldn't",
        "must" => "mustn't",
        "can" => "can't",
        _ => return None,
    })
}

fn normalize_contractions(text: &str) -> (String, usize) {
    let changes = Cell::new(0usize);

    let mut out = PRONOUN_CONTRACTION_RE
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let pronoun = caps[1].to_ascii_lowercase();
            let verb = caps[2].to_ascii_lowercase();
            let Some(contraction) = pronoun_contraction(&pronoun, &verb) else {
                return caps[0].to_string();
            };
            // "I have 5 apples" / "I have $87" read better uncontracted; only
            // contract before a normal word, matching the requested style.
            if pronoun == "i" && verb == "have" {
                let end = caps.get(0).map(|m| m.end()).unwrap_or(text.len());
                let rest = text.get(end..).unwrap_or("").trim_start();
                let next_char = rest.chars().next();
                let next_word = rest
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_matches(|ch: char| !ch.is_alphanumeric())
                    .to_ascii_uppercase();
                let currency_symbol = matches!(
                    next_char,
                    Some(
                        '$' | '€'
                            | '£'
                            | '₹'
                            | '¥'
                            | '₩'
                            | '₽'
                            | '₺'
                            | '₦'
                            | '₱'
                            | '฿'
                            | '₫'
                            | '₪'
                            | '₴'
                            | '₸'
                            | '৳'
                            | '₨'
                            | '¢'
                    )
                );
                let currency_code = CURRENCY_CODES
                    .split('|')
                    .any(|code| code == next_word)
                    || next_word == "RS";
                if next_char.is_some_and(|ch| ch.is_ascii_digit())
                    || currency_symbol
                    || currency_code
                {
                    return caps[0].to_string();
                }
            }
            changes.set(changes.get() + 1);
            apply_case(&caps[1], contraction)
        })
        .into_owned();

    out = NEGATION_CONTRACTION_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let verb = caps[1].to_ascii_lowercase();
            let Some(contraction) = negation_contraction(&verb) else {
                return caps[0].to_string();
            };
            changes.set(changes.get() + 1);
            apply_case(&caps[1], contraction)
        })
        .into_owned();

    out = CANNOT_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            changes.set(changes.get() + 1);
            apply_case(&caps[0], "can't")
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
    let (math_fixed, math_fixes) = normalize_math(&raw_fixed);
    let (contraction_fixed, contraction_fixes) = normalize_contractions(&math_fixed);
    let mut fixed: Vec<char> = contraction_fixed.chars().collect();
    capitalize_start(&mut fixed);
    let fixed: String = fixed.into_iter().collect();
    if fixed.trim().is_empty() {
        (text.to_owned(), 0)
    } else {
        (
            fixed,
            applied + number_fixes + math_fixes + contraction_fixes,
        )
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
    fn keeps_prose_and_cadence_that_look_like_math() {
        let approx = fix("i will approximately arrive at noon");
        assert!(approx.contains("approximately arrive"), "{approx}");
        let cadence = fix("i visited 3 times 2 years ago");
        assert!(cadence.contains("3 times 2 years ago"), "{cadence}");
        assert!(!cadence.contains("3*2"), "{cadence}");
        let range = fix("the price is 10 to 20 dollars");
        assert!(range.contains("$10 to $20"), "{range}");
        let between = fix("the price is between 10 and 20 dollars");
        assert!(between.contains("between $10 and $20"), "{between}");
        let room = fix("the room is 10 by 12");
        assert!(room.contains("10x12"), "{room}");
        let ratio = fix("the ratio is 10 by 20");
        assert!(ratio.contains("10/20"), "{ratio}");
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
        let small = fix("i have five apples");
        assert!(small.contains("5 apples"), "{small}");
        let hyphenated = fix("i am earning twenty-five thousand dollars per month");
        assert!(hyphenated.contains("$25,000"), "{hyphenated}");
        let marks = fix("i got 80-nine marks out of 100 marks");
        assert!(marks.contains("89 marks"), "{marks}");
        let exam = fix("i achieved 90-9% in my final examination");
        assert!(exam.contains("99%"), "{exam}");
        let ordinal = fix("i got first position in this race");
        assert!(ordinal.contains("1st position"), "{ordinal}");
        let height = fix("his height is five feet 6 inches");
        assert!(height.contains("5 feet 6 inches"), "{height}");
        let model_split = fix("i am earning 20-$5,000 per month");
        assert!(model_split.contains("$25,000"), "{model_split}");
        let cents = fix("i have $87 14 cents");
        assert!(cents.contains("$87.14"), "{cents}");
        let cents_only = fix("i have 13 cents");
        assert!(cents_only.contains("¢13"), "{cents_only}");
        let inr = fix("i have INR 27");
        assert!(inr.contains("₹27"), "{inr}");
        let spaced_code = fix("i have 47 P K R");
        assert!(spaced_code.contains("PKR 47"), "{spaced_code}");
        let rs = fix("i have R's 8,000 to spend");
        assert!(rs.contains("Rs. 8,000"), "{rs}");
        let split_pkr = fix("i have PKR 50 2,873 to spend");
        assert!(split_pkr.contains("PKR 52,873"), "{split_pkr}");
        let split_rs = fix("i have 58,000. R's 733.27");
        assert!(split_rs.contains("Rs. 58,733.27"), "{split_rs}");
    }

    #[test]
    fn converts_math_and_units() {
        let ratio = fix("10 by 20 marble pieces are two X smaller");
        assert!(ratio.contains("10/20"), "{ratio}");
        assert!(ratio.contains("2x smaller"), "{ratio}");
        let pi_value = fix("what is the value of pi?");
        assert!(pi_value.contains("value of π"), "{pi_value}");
        let circumference = fix("circumference of a circle equal to two pi r");
        assert!(circumference.contains("circle = 2πr"), "{circumference}");
        let area = fix("area is equal to pi r squared");
        assert!(area.contains("Area = πr^2"), "{area}");
        let times = fix("two times five is equal to 10");
        assert!(times.contains("2*5 = 10"), "{times}");
        let approx = fix(
            "we improved by approximately ninety eight point three seven percent",
        );
        assert!(approx.contains("~98.37%"), "{approx}");
        let prose = fix("she is equal to the task");
        assert!(prose.contains("equal to the task"), "{prose}");
        assert!(!prose.contains(" = "), "{prose}");
        let domain = fix("we use llama dot cpp daily");
        assert!(domain.contains("llama.cpp"), "{domain}");
    }

    #[test]
    fn joins_paused_sentences_and_fixes_modal_case() {
        let day = fix("today is thursday, and yesterday were. wednesday");
        assert!(day.contains("were Wednesday"), "{day}");
        let modal = fix("i cannot Go to college today");
        assert!(modal.contains("can't go"), "{modal}");
        let clause = fix(
            "people building crazy stuff with ai, and someone on youtube. is earning 40 k a month",
        );
        assert!(clause.contains("YouTube is earning 40k"), "{clause}");
    }

    #[test]
    fn applies_common_contractions() {
        let bike = fix("I have a bike.");
        assert!(bike.contains("I've a bike."), "{bike}");
        let apples = fix("I have 5 apples.");
        assert!(apples.contains("I have 5 apples."), "{apples}");
        let money = fix("I have $87.14.");
        assert!(money.contains("I have $87.14."), "{money}");
        let worry = fix("do not worry");
        assert!(worry.contains("Don't worry"), "{worry}");
        let she = fix("she is here");
        assert!(she.contains("She's here"), "{she}");
        let they = fix("they are ready");
        assert!(they.contains("They're ready"), "{they}");
        let will = fix("will not go");
        assert!(will.contains("Won't go"), "{will}");
        let cannot = fix("cannot go");
        assert!(cannot.contains("Can't go"), "{cannot}");
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
