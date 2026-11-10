// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Lorebook, LorebookEntry } from "../types";
import { linearRegex } from "./linear-regex.ts";

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
// Scan only the tail of recent messages: keys are matched on every send and token count.
const MAX_SCAN_CHARS = 16_000;

// Regex keys matched per character per prompt: each is linear in the text, but a card can carry thousands.
const MAX_REGEX_KEYS_PER_SCAN = 128;
const MAX_CACHED_MATCHERS = 4096;

type Matcher = { regex: boolean; test: (text: string) => boolean };
// Compiled once per key: keys are matched on every send and every token count.
const matchers = new Map<string, Matcher | null>();

function buildMatcher(trimmed: string, caseSensitive: boolean, useRegex: boolean): Matcher | null {
  const slashed = REGEX_KEY.exec(trimmed);
  const regex = slashed ? [slashed[1], slashed[2]] : useRegex ? [trimmed, caseSensitive ? "" : "i"] : null;
  // Card keys run on every send on the UI thread, so they never reach the backtracking RegExp engine.
  if (regex) {
    const test = trimmed.length > MAX_REGEX_KEY ? null : linearRegex(regex[0], regex[1].replace("g", ""));
    return test ? { regex: true, test } : null;
  }
  // A keyword is a word or phrase; a card's oversized one is skipped rather than abort the prompt.
  if (trimmed.length > MAX_LITERAL_KEY) return null;
  const escaped = trimmed.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Whole words when the key is a word, so "cat" does not fire on "category".
  const wordy = /^[\p{L}\p{N}_][\p{L}\p{N}_\s'-]*[\p{L}\p{N}_]$|^[\p{L}\p{N}_]$/u.test(trimmed);
  const source = wordy ? `(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])` : escaped;
  let pattern: RegExp;
  try {
    pattern = new RegExp(source, caseSensitive ? "u" : "iu");
  } catch {
    return null;
  }
  return { regex: false, test: (text) => pattern.test(text) };
}

function keyMatcher(key: string, caseSensitive: boolean, useRegex: boolean): Matcher | null {
  const trimmed = key.trim();
  if (!trimmed) return null;
  const cacheKey = `${caseSensitive ? 1 : 0}${useRegex ? 1 : 0}${trimmed}`;
  let matcher = matchers.get(cacheKey);
  if (matcher === undefined) {
    if (matchers.size >= MAX_CACHED_MATCHERS) matchers.clear();
    matcher = buildMatcher(trimmed, caseSensitive, useRegex);
    matchers.set(cacheKey, matcher);
  }
  return matcher;
}

type Scan = { text: string; regexLeft: number };

function keyHits(key: string, scan: Scan, caseSensitive: boolean, useRegex = false): boolean {
  const matcher = keyMatcher(key, caseSensitive, useRegex);
  if (!matcher) return false;
  if (matcher.regex) {
    if (scan.regexLeft <= 0) return false;
    scan.regexLeft -= 1;
  }
  return matcher.test(scan.text);
}

function anyKey(keys: string[], scan: Scan, caseSensitive: boolean, useRegex = false): boolean {
  return keys.some((key) => keyHits(key, scan, caseSensitive, useRegex));
}

function secondaryPasses(entry: LorebookEntry, scan: Scan): boolean {
  if (!entry.selective || entry.secondary_keys.length === 0) return true;
  const hits = entry.secondary_keys.map((key) => keyHits(key, scan, entry.case_sensitive, entry.use_regex));
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

/** Which lorebook entries the recent chat calls for, split by where they go in the prompt. */
export function activateLorebook(
  book: Lorebook | null | undefined,
  recentNewestFirst: string[],
  options: { maxTokens?: number } = {},
): ActivatedLore {
  const result: ActivatedLore = { before: [], after: [], atDepth: [], fired: [] };
  if (!book || !Array.isArray(book.entries) || book.entries.length === 0) return result;
  const depth = Math.max(1, book.scan_depth || 4);
  // Oldest to newest, keeping the tail: an over-long window drops the oldest text, not a reply's ending.
  const scan: Scan = {
    text: recentNewestFirst.slice(0, depth).reverse().join("\n").slice(-MAX_SCAN_CHARS),
    regexLeft: MAX_REGEX_KEYS_PER_SCAN,
  };
  const candidates: { entry: LorebookEntry; reason: "constant" | "keyword" }[] = [];
  for (const entry of book.entries) {
    if (entry.enabled === false || !entry.content?.trim()) continue;
    if (entry.constant) {
      candidates.push({ entry, reason: "constant" });
      continue;
    }
    if (anyKey(entry.keys ?? [], scan, entry.case_sensitive, entry.use_regex) && secondaryPasses(entry, scan)) {
      candidates.push({ entry, reason: "keyword" });
    }
  }
  // Constant entries first, then by insertion order (lower goes higher in the prompt).
  candidates.sort(
    (a, b) =>
      Number(b.reason === "constant") - Number(a.reason === "constant") ||
      (a.entry.insertion_order ?? 100) - (b.entry.insertion_order ?? 100),
  );
  const budget = options.maxTokens ?? (book.token_budget > 0 ? book.token_budget : 1024);
  let used = 0;
  for (const { entry, reason } of candidates) {
    const cost = estimateTokens(entry.content);
    // The first entry too: a system-block entry is never compacted away, so an oversized one would fail every turn.
    if (used + cost > budget) continue;
    used += cost;
    const content = entry.content.trim();
    if (entry.position === "at_depth") result.atDepth.push({ depth: Math.max(0, entry.depth ?? 4), content });
    else if (entry.position === "after_char") result.after.push(content);
    else result.before.push(content);
    result.fired.push({
      id: entry.id,
      label: entry.comment || entry.keys?.[0] || String(entry.id),
      reason,
    });
  }
  return result;
}
