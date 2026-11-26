// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { escapeRegExp } from "../../../lib/escape-regexp.ts";
import { estimateTokenCount } from "../../../lib/estimate-tokens.ts";
import type { Lorebook, LorebookEntry } from "../types";
import { DEFAULT_INSERTION_ORDER, DEFAULT_LORE_BUDGET, DEFAULT_NOTE_DEPTH, DEFAULT_SCAN_DEPTH } from "./defaults.ts";
import { linearRegex, type StepBudget } from "../../../lib/linear-regex.ts";

export interface ActivatedLore {
  before: string[];
  after: string[];
  atDepth: { depth: number; content: string }[];
  /** Entries that fired, for the prompt inspector. */
  fired: { id: string | number; label: string; reason: "constant" | "keyword" }[];
}

const REGEX_KEY = /^\/(.+)\/([gimsuy]*)$/;
const MAX_REGEX_KEY = 160;
const MAX_LITERAL_KEY = 500;
// Only the tail of recent messages is scanned.
const MAX_SCAN_CHARS = 16_000;
// NFA steps (live threads x characters) all regex keys of one prompt may take, about 200 ms worst case.
const MAX_REGEX_STEPS = 6_000_000;
// Characters all literal keys of one prompt may scan, about 100 ms; past it the rest of a huge card stays quiet.
const MAX_LITERAL_CHARS = 400_000_000;
const MAX_RECURSION = 3;
const MAX_CACHED_MATCHERS = 4096;
const WORDY_KEY = /^[\p{L}\p{N}_][\p{L}\p{N}_\s'-]*[\p{L}\p{N}_]$|^[\p{L}\p{N}_]$/u;
const WORD_CHAR = /[\p{L}\p{N}_]/u;
// Scripts written without spaces between words, where a whole-word key could never match.
const NO_SPACE_SCRIPT = /[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\p{sc=Hangul}\p{sc=Thai}\p{sc=Lao}\p{sc=Khmer}\p{sc=Myanmar}]/u;

/** A RegExp source for a name or keyword: whole words, unless the key is in a script without spaces. */
export function wordSource(key: string, wholeWords = true): string {
  const escaped = escapeRegExp(key);
  const wordy = WORDY_KEY.test(key);
  if (!wholeWords || !wordy || NO_SPACE_SCRIPT.test(key)) return escaped;
  return `(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])`;
}

/** What one prompt's lore may spend, shared by every book in the cast. */
export interface LoreBudget {
  tokens: number;
  regexSteps: StepBudget;
  literalChars: StepBudget;
}

function bookBudget(book: Lorebook | null | undefined): number {
  return book && book.token_budget > 0 ? book.token_budget : DEFAULT_LORE_BUDGET;
}

export function loreBudget(books: readonly (Lorebook | null | undefined)[]): LoreBudget {
  const withEntries = books.filter((book) => book?.entries?.length);
  return {
    tokens: withEntries.length ? Math.max(...withEntries.map(bookBudget)) : DEFAULT_LORE_BUDGET,
    regexSteps: { left: MAX_REGEX_STEPS },
    literalChars: { left: MAX_LITERAL_CHARS },
  };
}

type KeyOptions = { caseSensitive: boolean; useRegex: boolean; wholeWords: boolean };
// null: the prompt's budget ran out before this key was checked.
type Matcher = (text: string, budget: LoreBudget) => boolean | null;
// Compiled once per key: keys are matched on every send and every token count.
const matchers = new Map<string, Matcher | null>();

function buildMatcher(trimmed: string, options: KeyOptions): Matcher | null {
  const slashed = REGEX_KEY.exec(trimmed);
  const regex = slashed ? [slashed[1], slashed[2]] : options.useRegex ? [trimmed, options.caseSensitive ? "" : "i"] : null;
  // Card keys run on the UI thread, so they never reach the backtracking RegExp engine.
  if (regex) {
    if (trimmed.length > MAX_REGEX_KEY) return null;
    const match = linearRegex(regex[0], regex[1]);
    return match && ((text, budget) => match(text, budget.regexSteps) || (budget.regexSteps.left >= 0 ? false : null));
  }
  if (trimmed.length > MAX_LITERAL_KEY) return null;
  return literalMatcher(trimmed, options);
}

// The simple folds (CaseFolding.txt S) whose characters only uppercase to several, so no case mapping reaches them.
const folds = new Map<string, string>([
  ["\u1fd3", "\u0390"],
  ["\u1fe3", "\u03b0"],
  ["\ufb05", "\ufb06"],
]);
// Simple case folding checked against the /iu engine itself: the lowercase of the uppercase when /iu agrees,
// so Σ σ ς and µ μ fold together while ı, İ and ß stay apart, as the RegExp keys matched.
function foldChar(ch: string): string {
  let out = folds.get(ch);
  if (out === undefined) {
    const same = new RegExp(`^${escapeRegExp(ch)}$`, "iu");
    const upper = ch.toUpperCase();
    const candidates = [[...upper].length === 1 ? upper.toLowerCase() : "", ch.toLowerCase()];
    out = candidates.find((c) => [...c].length === 1 && same.test(c)) ?? ch;
    folds.set(ch, out);
  }
  return out;
}

export function foldCase(text: string): string {
  let out = "";
  for (const ch of text) out += foldChar(ch);
  return out;
}

let foldedFrom = "";
let folded = "";
// The scan text folded once per prompt pass, not once per key.
function foldedOf(text: string): string {
  if (text !== foldedFrom) {
    foldedFrom = text;
    folded = foldCase(text);
  }
  return folded;
}

const charAt = (text: string, at: number) => String.fromCodePoint(text.codePointAt(at) ?? 32);
function charBefore(text: string, at: number): string {
  if (at <= 0) return " ";
  const low = text.charCodeAt(at - 1);
  return low >= 0xdc00 && low <= 0xdfff && at >= 2 ? charAt(text, at - 2) : text[at - 1];
}

/** A plain key as a substring search: a RegExp per key costs milliseconds to compile, and a card can hold thousands. */
function literalMatcher(key: string, options: KeyOptions): Matcher {
  const needle = options.caseSensitive ? key : foldCase(key);
  const whole = options.wholeWords && WORDY_KEY.test(key) && !NO_SPACE_SCRIPT.test(key);
  return (text, budget) => {
    if (budget.literalChars.left <= 0) return null;
    const hay = options.caseSensitive ? text : foldedOf(text);
    budget.literalChars.left -= hay.length;
    for (let at = hay.indexOf(needle); at >= 0; at = hay.indexOf(needle, at + 1)) {
      if (!whole || (!WORD_CHAR.test(charBefore(hay, at)) && !WORD_CHAR.test(charAt(hay, at + needle.length)))) {
        return true;
      }
    }
    return false;
  };
}

function keyMatcher(key: string, options: KeyOptions): Matcher | null {
  const trimmed = key.trim();
  if (!trimmed) return null;
  const cacheKey = `${Number(options.caseSensitive)}${Number(options.useRegex)}${Number(options.wholeWords)}${trimmed}`;
  let matcher = matchers.get(cacheKey);
  if (matcher === undefined) {
    if (matchers.size >= MAX_CACHED_MATCHERS) matchers.clear();
    matcher = buildMatcher(trimmed, options);
    matchers.set(cacheKey, matcher);
  }
  return matcher;
}

/** Exports with these fields set use_regex on every entry but mean regex only for /pattern/ keys,
 *  and substring matches unless whole words is on. */
function fromSillyTavern(entry: LorebookEntry): boolean {
  const ext = entry.extensions;
  return Boolean(ext) && ("match_whole_words" in ext! || "display_index" in ext!);
}

function keyOptions(entry: LorebookEntry): KeyOptions {
  const sillyTavern = fromSillyTavern(entry);
  const wholeWords = entry.extensions?.match_whole_words;
  return {
    caseSensitive: Boolean(entry.case_sensitive),
    useRegex: Boolean(entry.use_regex) && !sillyTavern,
    wholeWords: sillyTavern ? wholeWords === true : wholeWords !== false,
  };
}

function keyHits(key: string, text: string, budget: LoreBudget, options: KeyOptions): boolean | null {
  const matcher = keyMatcher(key, options);
  return matcher ? matcher(text, budget) : false;
}

// A key the budget never reached is unknown: it neither fires an entry nor clears an exclusion.
function entryHits(entry: LorebookEntry, text: string, budget: LoreBudget): boolean {
  const options = keyOptions(entry);
  if (!(entry.keys ?? []).some((key) => keyHits(key, text, budget, options) === true)) return false;
  if (!entry.selective || entry.secondary_keys.length === 0) return true;
  const hits = entry.secondary_keys.map((key) => keyHits(key, text, budget, options));
  switch (entry.selective_logic) {
    case "and_all":
      return hits.every((hit) => hit === true);
    case "not_any":
      return hits.every((hit) => hit === false);
    case "not_all":
      return hits.some((hit) => hit === false);
    default:
      return hits.some((hit) => hit === true);
  }
}

export function emptyLorebook(): Lorebook {
  return {
    name: "",
    scan_depth: DEFAULT_SCAN_DEPTH,
    token_budget: DEFAULT_LORE_BUDGET,
    recursive_scanning: false,
    extensions: {},
    entries: [],
  };
}

/** A blank entry with the next free numeric id. */
export function newLorebookEntry(entries: LorebookEntry[]): LorebookEntry {
  const numeric = entries.map((entry) => Number(entry.id)).filter(Number.isFinite);
  return {
    id: numeric.length ? Math.max(...numeric) + 1 : 0,
    keys: [],
    secondary_keys: [],
    content: "",
    comment: "",
    enabled: true,
    constant: false,
    selective: false,
    case_sensitive: false,
    insertion_order: DEFAULT_INSERTION_ORDER,
    position: "before_char",
    depth: DEFAULT_NOTE_DEPTH,
    selective_logic: "and_any",
    extensions: {},
  };
}

/** Keys typed as one comma-separated line; a /regex/ key keeps its own commas, as in /a{2,4}/. */
export function parseKeyList(text: string): string[] {
  const keys: string[] = [];
  let at = 0;
  while (at < text.length) {
    while (at < text.length && /\s/.test(text[at])) at += 1;
    let from = at;
    if (text[at] === "/") {
      let inClass = false;
      for (from = at + 1; from < text.length; from += 1) {
        const ch = text[from];
        if (ch === "\\") from += 1;
        else if (ch === "[") inClass = true;
        else if (ch === "]") inClass = false;
        else if (ch === "/" && !inClass) break;
      }
    }
    const comma = text.indexOf(",", from);
    const end = comma < 0 ? text.length : comma;
    const key = text.slice(at, end).trim();
    if (key) keys.push(key);
    at = end + 1;
  }
  return keys;
}

const priorityOf = (entry: LorebookEntry) => (typeof entry.priority === "number" ? entry.priority : 0);
const orderOf = (entry: LorebookEntry) => entry.insertion_order ?? DEFAULT_INSERTION_ORDER;

/** Which lorebook entries the recent chat calls for, split by where they go in the prompt. */
export function activateLorebook(
  book: Lorebook | null | undefined,
  recentNewestFirst: string[],
  budget: LoreBudget = loreBudget([book]),
): ActivatedLore {
  const result: ActivatedLore = { before: [], after: [], atDepth: [], fired: [] };
  if (!book || !Array.isArray(book.entries) || book.entries.length === 0) return result;
  const depth = Math.max(1, book.scan_depth || DEFAULT_SCAN_DEPTH);
  // Oldest to newest, keeping the tail: an over-long window drops the oldest text, not a reply's ending.
  const chat = recentNewestFirst.slice(0, depth).reverse().join("\n");
  const live = book.entries.filter((entry) => entry.enabled !== false && entry.content?.trim());
  const candidates = new Map<LorebookEntry, "constant" | "keyword">();
  for (const entry of live) if (entry.constant) candidates.set(entry, "constant");
  // Recursion reads always-on lore too; the first pass reads only the chat, as exclude_recursion expects.
  let lore = book.recursive_scanning
    ? live
        .filter((entry) => entry.constant && !entry.extensions?.prevent_recursion)
        .map((entry) => `\n${entry.content}`)
        .join("")
    : "";
  for (let pass = 0; pass <= (book.recursive_scanning ? MAX_RECURSION : 0); pass += 1) {
    const text = `${chat}${pass > 0 ? lore : ""}`.slice(-MAX_SCAN_CHARS);
    const found = live.filter(
      (entry) =>
        !candidates.has(entry) &&
        !(pass > 0 && entry.extensions?.exclude_recursion) &&
        entryHits(entry, text, budget),
    );
    if (found.length === 0 && (pass > 0 || !lore)) break;
    for (const entry of found) candidates.set(entry, "keyword");
    // A recursive pass also reads the lore that just fired, except entries that opt out.
    lore += found
      .filter((entry) => !entry.extensions?.prevent_recursion)
      .map((entry) => `\n${entry.content}`)
      .join("");
  }
  // Over budget, constants stay, then higher priority, then lower insertion order (CCv2).
  const ranked = [...candidates].sort(
    ([a, ra], [b, rb]) =>
      Number(rb === "constant") - Number(ra === "constant") || priorityOf(b) - priorityOf(a) || orderOf(a) - orderOf(b),
  );
  const own = bookBudget(book);
  let used = 0;
  const kept: [LorebookEntry, "constant" | "keyword"][] = [];
  for (const [entry, reason] of ranked) {
    const cost = estimateTokenCount(entry.content) ?? 0;
    // Even the first: an oversized system-block entry would fail every turn.
    if (used + cost > own || cost > budget.tokens) continue;
    used += cost;
    budget.tokens -= cost;
    kept.push([entry, reason]);
  }
  // Placed by insertion order: lower goes higher in the prompt.
  kept.sort(([a], [b]) => orderOf(a) - orderOf(b));
  for (const [entry, reason] of kept) {
    const content = entry.content.trim();
    if (entry.position === "at_depth") result.atDepth.push({ depth: Math.max(0, entry.depth ?? DEFAULT_NOTE_DEPTH), content });
    else if (entry.position === "after_char") result.after.push(content);
    else result.before.push(content);
    result.fired.push({ id: entry.id, label: entry.comment || entry.keys?.[0] || String(entry.id), reason });
  }
  return result;
}
