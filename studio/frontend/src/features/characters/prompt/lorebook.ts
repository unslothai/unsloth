// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { escapeRegExp } from "../../../lib/escape-regexp.ts";
import type { Lorebook, LorebookEntry } from "../types";
import { linearRegex, type StepBudget } from "./linear-regex.ts";

export interface ActivatedLore {
  before: string[];
  after: string[];
  atDepth: { depth: number; content: string }[];
  /** Entries that fired, for the prompt inspector. */
  fired: { id: string | number; label: string; reason: "constant" | "keyword" }[];
}

/** Roughly four characters per token, enough to keep a budget without a tokenizer. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

const REGEX_KEY = /^\/(.+)\/([gimsuy]*)$/;
const MAX_REGEX_KEY = 160;
const MAX_LITERAL_KEY = 500;
// Only the tail of recent messages is scanned.
const MAX_SCAN_CHARS = 16_000;
// NFA steps (live threads x characters) all regex keys of one prompt may take, about 200 ms worst case.
const MAX_REGEX_STEPS = 6_000_000;
const MAX_RECURSION = 3;
const MAX_CACHED_MATCHERS = 4096;
// Scripts written without spaces between words, where a whole-word key could never match.
const NO_SPACE_SCRIPT = /[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\p{sc=Hangul}\p{sc=Thai}\p{sc=Lao}\p{sc=Khmer}\p{sc=Myanmar}]/u;

/** A RegExp source for a name or keyword: whole words, unless the key is in a script without spaces. */
export function wordSource(key: string, wholeWords = true): string {
  const escaped = escapeRegExp(key);
  const wordy = /^[\p{L}\p{N}_][\p{L}\p{N}_\s'-]*[\p{L}\p{N}_]$|^[\p{L}\p{N}_]$/u.test(key);
  if (!wholeWords || !wordy || NO_SPACE_SCRIPT.test(key)) return escaped;
  return `(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])`;
}

/** What one prompt's lore may spend, shared by every book in the cast. */
export interface LoreBudget {
  tokens: number;
  regexSteps: StepBudget;
}

function bookBudget(book: Lorebook | null | undefined): number {
  return book && book.token_budget > 0 ? book.token_budget : 1024;
}

export function loreBudget(books: readonly (Lorebook | null | undefined)[]): LoreBudget {
  const withEntries = books.filter((book) => book?.entries?.length);
  return {
    tokens: withEntries.length ? Math.max(...withEntries.map(bookBudget)) : 1024,
    regexSteps: { left: MAX_REGEX_STEPS },
  };
}

type KeyOptions = { caseSensitive: boolean; useRegex: boolean; wholeWords: boolean };
type Matcher = (text: string, steps: StepBudget) => boolean;
// Compiled once per key: keys are matched on every send and every token count.
const matchers = new Map<string, Matcher | null>();

function buildMatcher(trimmed: string, options: KeyOptions): Matcher | null {
  const slashed = REGEX_KEY.exec(trimmed);
  const regex = slashed ? [slashed[1], slashed[2]] : options.useRegex ? [trimmed, options.caseSensitive ? "" : "i"] : null;
  // Card keys run on the UI thread, so they never reach the backtracking RegExp engine.
  if (regex) return trimmed.length > MAX_REGEX_KEY ? null : linearRegex(regex[0], regex[1]);
  if (trimmed.length > MAX_LITERAL_KEY) return null;
  let pattern: RegExp;
  try {
    pattern = new RegExp(wordSource(trimmed, options.wholeWords), options.caseSensitive ? "u" : "iu");
  } catch {
    return null;
  }
  return (text) => pattern.test(text);
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

function keyOptions(entry: LorebookEntry): KeyOptions {
  return {
    caseSensitive: Boolean(entry.case_sensitive),
    useRegex: Boolean(entry.use_regex),
    wholeWords: entry.extensions?.match_whole_words !== false,
  };
}

function keyHits(key: string, text: string, steps: StepBudget, options: KeyOptions): boolean {
  return keyMatcher(key, options)?.(text, steps) ?? false;
}

function entryHits(entry: LorebookEntry, text: string, steps: StepBudget): boolean {
  const options = keyOptions(entry);
  if (!(entry.keys ?? []).some((key) => keyHits(key, text, steps, options))) return false;
  if (!entry.selective || entry.secondary_keys.length === 0) return true;
  const hits = entry.secondary_keys.map((key) => keyHits(key, text, steps, options));
  switch (entry.selective_logic) {
    case "and_all":
      return hits.every(Boolean);
    case "not_any":
      return !hits.some(Boolean);
    case "not_all":
      return !hits.every(Boolean);
    default:
      return hits.some(Boolean);
  }
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
const orderOf = (entry: LorebookEntry) => entry.insertion_order ?? 100;

/** Which lorebook entries the recent chat calls for, split by where they go in the prompt. */
export function activateLorebook(
  book: Lorebook | null | undefined,
  recentNewestFirst: string[],
  budget: LoreBudget = loreBudget([book]),
): ActivatedLore {
  const result: ActivatedLore = { before: [], after: [], atDepth: [], fired: [] };
  if (!book || !Array.isArray(book.entries) || book.entries.length === 0) return result;
  const depth = Math.max(1, book.scan_depth || 4);
  // Oldest to newest, keeping the tail: an over-long window drops the oldest text, not a reply's ending.
  const chat = recentNewestFirst.slice(0, depth).reverse().join("\n");
  const live = book.entries.filter((entry) => entry.enabled !== false && entry.content?.trim());
  const candidates = new Map<LorebookEntry, "constant" | "keyword">();
  for (const entry of live) if (entry.constant) candidates.set(entry, "constant");
  let lore = "";
  for (let pass = 0; pass <= (book.recursive_scanning ? MAX_RECURSION : 0); pass += 1) {
    const text = `${chat}${lore}`.slice(-MAX_SCAN_CHARS);
    const found = live.filter(
      (entry) =>
        !candidates.has(entry) &&
        !(pass > 0 && entry.extensions?.exclude_recursion) &&
        entryHits(entry, text, budget.regexSteps),
    );
    if (found.length === 0) break;
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
    const cost = estimateTokens(entry.content);
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
    if (entry.position === "at_depth") result.atDepth.push({ depth: Math.max(0, entry.depth ?? 4), content });
    else if (entry.position === "after_char") result.after.push(content);
    else result.before.push(content);
    result.fired.push({ id: entry.id, label: entry.comment || entry.keys?.[0] || String(entry.id), reason });
  }
  return result;
}
