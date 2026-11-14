// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { findCodeBlockRegions, isInRegion } from "../../../lib/latex.ts";

export const SEARCH_IMAGES_MARKER = "\n__WEB_IMAGES__:";
export const SEARCH_IMAGE_TAG = "search-image";
// The only tool whose result may carry the envelope; elsewhere the marker is content.
export const SEARCH_IMAGE_TOOL = "web_search";
const IMAGE_ID_RE = /^[0-9a-f]{12}$/;
const TOKEN_RE = /\[\[img:([0-9a-f]{12})\]\]/g;
// A token cut off mid-stream: any prefix of "[[img:xxxxxxxxxxxx]]".
const PARTIAL_TOKEN_RE = /\[(?:\[(?:i(?:m(?:g(?::[0-9a-f]{0,12}\]?)?)?)?)?)?$/;

export interface SearchImageEntry {
  id: string;
  title: string;
  domain: string;
  source: string;
  subject?: string;
}

export interface SearchImagesToolResult {
  text: string;
  webImages: SearchImageEntry[];
}

export function isSearchImageEntry(value: unknown): value is SearchImageEntry {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    IMAGE_ID_RE.test(v.id) &&
    typeof v.title === "string" &&
    typeof v.domain === "string" &&
    typeof v.source === "string" &&
    // Re-checked so a spoofed envelope cannot put another scheme in an href.
    /^https?:\/\//i.test(v.source) &&
    (v.subject === undefined || typeof v.subject === "string")
  );
}

export function isSearchImagesToolResult(
  value: unknown,
): value is SearchImagesToolResult {
  if (typeof value !== "object" || value === null) return false;
  const v = value as { text?: unknown; webImages?: unknown };
  return (
    typeof v.text === "string" &&
    Array.isArray(v.webImages) &&
    v.webImages.length > 0 &&
    v.webImages.every(isSearchImageEntry)
  );
}

export function extractSearchImages(raw: string): {
  text: string;
  images: SearchImageEntry[];
} {
  const start = raw.lastIndexOf(SEARCH_IMAGES_MARKER);
  if (start === -1) return { text: raw, images: [] };
  const payloadStart = start + SEARCH_IMAGES_MARKER.length;
  let end = raw.indexOf("\n__", payloadStart);
  if (end === -1) end = raw.length;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw.slice(payloadStart, end));
  } catch {
    return { text: raw, images: [] };
  }
  if (
    !Array.isArray(parsed) ||
    parsed.length === 0 ||
    !parsed.every(isSearchImageEntry)
  ) {
    return { text: raw, images: [] };
  }
  return {
    text: (raw.slice(0, start) + raw.slice(end)).trimEnd(),
    images: parsed,
  };
}

export function searchResultText(result: unknown): string {
  if (typeof result === "string") return result;
  if (isSearchImagesToolResult(result)) return result.text;
  return "";
}

export function searchImagePath(id: string): string {
  return `/api/inference/search-images/${encodeURIComponent(id)}`;
}

/** `token`, not `id`: rehype-sanitize prefixes `id` values with `user-content-`. */
export function rewriteSearchImageTokens(
  text: string,
  known: { has(id: string): boolean },
): string {
  if (!text.includes("[[img:")) return text;
  const codeRegions = findCodeBlockRegions(text);
  return text.replace(TOKEN_RE, (match, id: string, offset: number) => {
    if (isInRegion(offset, codeRegions)) return match;
    if (!known.has(id)) return "";
    return `<${SEARCH_IMAGE_TAG} token="${id}"></${SEARCH_IMAGE_TAG}>`;
  });
}

/** Strip model-written image tokens from plain-text output (clipboard, export, read-aloud). */
export function stripSearchImageTokens(text: string): string {
  if (!text.includes("[[img:")) return text;
  const codeRegions = findCodeBlockRegions(text);
  // One pass keeps code-region offsets valid; also drops the blank line before a token.
  return text.replace(
    /\n\n[ \t]*\[\[img:[0-9a-f]{12}\]\][ \t]*(?=\n\n|\n?$)|\[\[img:[0-9a-f]{12}\]\]/g,
    (match, offset: number) => (isInRegion(offset, codeRegions) ? match : ""),
  );
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function tokenPlacedOutsideCode(
  text: string,
  id: string,
  codeRegions: Array<[number, number]>,
): boolean {
  const needle = `[[img:${id}]]`;
  for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + 1)) {
    if (!isInRegion(at, codeRegions)) return true;
  }
  return false;
}

function firstMatchOutsideCode(
  pattern: RegExp,
  text: string,
  regions: Array<[number, number]>,
): number | null {
  if (!text) return null;
  pattern.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    const at = match.index + match[1].length;
    if (!isInRegion(at, regions)) return at;
  }
  return null;
}

const LIST_MARKER_RE = /^(\s*(?:[-*+•]|\d{1,2}[.)])\s+)/;
const HEADING_LINE_RE = /^\s*#{1,6}\s/;
const BLOCK_BREAK_RE = /^(?:\s*$|\s*(?:[-*+•]|\d{1,2}[.)])\s|\s*#{1,6}\s|\s*(?:```|~~~))/;
// Display math, invisible to BLOCK_BREAK_RE; bounded so an unclosed `$$` cannot scan everything.
const DISPLAY_MATH_RE = /\$\$[\s\S]{0,4096}?\$\$|\\\[[\s\S]{0,4096}?\\\]/g;

function findDisplayMathRegions(text: string): Array<[number, number]> {
  const regions: Array<[number, number]> = [];
  if (!text.includes("$$") && !text.includes("\\[")) return regions;
  DISPLAY_MATH_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = DISPLAY_MATH_RE.exec(text)) !== null) {
    regions.push([match.index, match.index + match[0].length]);
  }
  return regions;
}

/** Models wrap list items across lines, so insert at the block end, not the first line. */
function blockEndFrom(
  text: string,
  index: number,
  mathRegions: Array<[number, number]> = [],
): number {
  let at = text.indexOf("\n", index);
  while (at !== -1) {
    const nextBreak = text.indexOf("\n", at + 1);
    const line = text.slice(at + 1, nextBreak === -1 ? text.length : nextBreak);
    if (BLOCK_BREAK_RE.test(line) && !isInRegion(at + 1, mathRegions)) return at;
    at = nextBreak;
  }
  return text.length;
}

function namesSameThing(a: string, b: string): boolean {
  if (a === b) return true;
  const [shorter, longer] = a.length <= b.length ? [a, b] : [b, a];
  return new RegExp(
    `(^|[^\\p{L}\\p{N}])${escapeRegExp(shorter)}([^\\p{L}\\p{N}]|$)`,
    "u",
  ).test(longer);
}

export function placeSubjectImages(
  text: string,
  images: ReadonlyMap<string, SearchImageEntry>,
  isStreaming: boolean,
  alreadyNamed = "",
  messageTexts: readonly string[] = [text],
): string {
  if (isStreaming || images.size === 0) return text;
  const bySubject = new Map<string, SearchImageEntry>();
  for (const entry of images.values()) {
    const key = entry.subject?.trim().toLowerCase();
    if (!key || bySubject.has(key)) continue;
    bySubject.set(key, entry);
  }
  if (bySubject.size === 0) return text;

  const codeRegions = findCodeBlockRegions(text);
  const mathRegions = findDisplayMathRegions(text);
  const insertions: Array<{ at: number; chunk: string }> = [];
  const namedRegions = findCodeBlockRegions(alreadyNamed);
  const messageParts = messageTexts.map((part) => ({
    text: part,
    codeRegions: findCodeBlockRegions(part),
  }));
  for (const [key, entry] of bySubject) {
    // Outside code only: a token inside a fence renders as literal text, not a picture.
    if (
      messageParts.some(({ text: part, codeRegions: regions }) =>
        tokenPlacedOutsideCode(part, entry.id, regions),
      )
    ) {
      continue;
    }
    // Search the original text: lowercasing can change length and shift offsets.
    const pattern = new RegExp(
      `(^|[^\\p{L}\\p{N}])${escapeRegExp(key)}(?![\\p{L}\\p{N}])`,
      "giu",
    );
    // A mention only inside code shows no card, so it must not suppress this one.
    if (firstMatchOutsideCode(pattern, alreadyNamed, namedRegions) !== null) continue;
    const at = firstMatchOutsideCode(pattern, text, codeRegions);
    if (at === null) continue;
    const lineStart = text.lastIndexOf("\n", at) + 1;
    const newline = text.indexOf("\n", at);
    const lineEnd = newline === -1 ? text.length : newline;
    const line = text.slice(lineStart, lineEnd);
    const marker = LIST_MARKER_RE.exec(line);
    if (HEADING_LINE_RE.test(line)) {
      insertions.push({ at: lineEnd, chunk: `\n\n[[img:${entry.id}]]` });
    } else if (marker) {
      // Indented to the item's content column so the card stays inside the list item.
      insertions.push({
        at: blockEndFrom(text, at, mathRegions),
        chunk: `\n\n${" ".repeat(marker[1].length)}[[img:${entry.id}]]`,
      });
    } else {
      insertions.push({
        at: blockEndFrom(text, at, mathRegions),
        chunk: `\n\n[[img:${entry.id}]]`,
      });
    }
  }
  if (insertions.length === 0) return text;

  insertions.sort((a, b) => b.at - a.at);
  let out = text;
  for (const { at, chunk } of insertions) {
    out = `${out.slice(0, at)}${chunk}${out.slice(at)}`;
  }
  return out;
}

// Marker and body are split so whitespace quantifiers never overlap (catastrophic backtracking).
const LIST_ITEM_MARKER_RE = /^[ \t]*(?:\d{1,2}[.)]|[-*+•])[ \t]+/;
const LIST_ITEM_RE =
  /^(?:\*\*|__)?[ \t\r]*([^\s*_\n][^*_\n]{0,58}?[^\s*_\n])(?:[ \t\r]*(?:\*\*|__))?[ \t\r]*(?::|[-–—][ \t\r]|\(|$)/;
const HEADING_RE =
  /^[ \t]*#{2,4}[ \t]+(?:\d{1,2}[.)][ \t]+)?([^\s\n#][^\n#]{0,58}?[^\s\n#])(?:[ \t\r]*#*)?[ \t\r]*$/;
const MAX_AUTO_SUBJECTS = 5;
const STEP_VERBS = new Set([
  "add",
  "apply",
  "avoid",
  "build",
  "call",
  "change",
  "check",
  "choose",
  "click",
  "close",
  "configure",
  "confirm",
  "connect",
  "copy",
  "create",
  "define",
  "delete",
  "disable",
  "download",
  "enable",
  "ensure",
  "enter",
  "find",
  "fix",
  "follow",
  "get",
  "go",
  "import",
  "install",
  "keep",
  "launch",
  "let",
  "load",
  "log",
  "make",
  "mix",
  "move",
  "navigate",
  "open",
  "paste",
  "pick",
  "place",
  "plan",
  "preheat",
  "prepare",
  "press",
  "pull",
  "push",
  "put",
  "read",
  "remove",
  "rename",
  "repeat",
  "replace",
  "restart",
  "review",
  "run",
  "save",
  "select",
  "set",
  "sign",
  "start",
  "stop",
  "take",
  "test",
  "try",
  "turn",
  "type",
  "update",
  "upgrade",
  "use",
  "verify",
  "visit",
  "wait",
  "write",
]);

function looksLikeStep(name: string): boolean {
  const first = name
    .split(/\s+/)[0]
    ?.toLowerCase()
    .replace(/[^a-z]/g, "");
  return first !== undefined && STEP_VERBS.has(first);
}

const ABSTRACT_HEADS = new Set([
  "advantages",
  "benefits",
  "bottom line",
  "caveats",
  "con",
  "conclusion",
  "cons",
  "cost",
  "disadvantages",
  "drawbacks",
  "example",
  "examples",
  "features",
  "key takeaways",
  "limitations",
  "note",
  "notes",
  "option",
  "options",
  "overview",
  "performance",
  "price",
  "pro",
  "pros",
  "risks",
  "summary",
  "takeaways",
  "tip",
  "tips",
  "tldr",
  "tl;dr",
  "verdict",
  "warning",
  "why it matters",
]);

function isAbstractHead(name: string): boolean {
  return ABSTRACT_HEADS.has(name.toLowerCase().replace(/\s+/g, " ").trim());
}

/** Text parts only: a model's reasoning must never be illustrated. */
export function answerTextFromParts(
  parts: ReadonlyArray<{ type: string; text?: unknown }>,
): string {
  return parts
    .filter(
      (part): part is { type: "text"; text: string } =>
        part.type === "text" && typeof part.text === "string",
    )
    .map((part) => part.text)
    .join("\n\n");
}

export function precedingTextForMessagePart(
  parts: ReadonlyArray<{ type: string; text?: unknown }>,
  partIndex: number,
): string {
  return answerTextFromParts(parts.slice(0, partIndex));
}

export function extractListSubjects(text: string): string[] {
  if (text.includes("```") || text.includes("~~~")) return [];
  const codeRegions = findCodeBlockRegions(text);
  const named: string[] = [];
  const seen = new Set<string>();
  let steps = 0;
  let offset = 0;
  for (const line of text.split("\n")) {
    const at = offset;
    offset += line.length + 1;
    if (isInRegion(at, codeRegions)) continue;
    const marker = LIST_ITEM_MARKER_RE.exec(line);
    const match = marker
      ? LIST_ITEM_RE.exec(line.slice(marker[0].length))
      : HEADING_RE.exec(line);
    if (!match) continue;
    const name = match[1].replace(/[\s:.,;!?]+$/g, "").trim();
    const words = name.split(/\s+/);
    if (
      name.length < 2 ||
      words.length > 6 ||
      /https?:|www\.|\d{3,}/i.test(name) ||
      !/\p{L}/u.test(name)
    ) {
      continue;
    }
    if (looksLikeStep(name)) {
      steps += 1;
      continue;
    }
    if (isAbstractHead(name)) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    named.push(name);
  }
  // Proportional, so a trailing "Choose X if:" in a comparison is not a how-to.
  if (steps >= named.length) return [];
  return named.length >= 2 ? named.slice(0, MAX_AUTO_SUBJECTS) : [];
}

export function missingListSubjects(
  text: string,
  parts: ReadonlyArray<{ type: string; toolName?: string; result?: unknown }>,
): string[] {
  const listed = extractListSubjects(text);
  if (listed.length === 0) return [];
  const covered: string[] = [];
  for (const entry of collectSearchImages(parts).values()) {
    const subject = entry.subject?.trim().toLowerCase();
    if (subject) covered.push(subject);
  }
  return listed.filter((name) => {
    const key = name.toLowerCase();
    return !covered.some((c) => namesSameThing(c, key));
  });
}

export function holdBackPartialSearchImageToken(
  text: string,
  isStreaming: boolean,
): string {
  if (!isStreaming) return text;
  const match = PARTIAL_TOKEN_RE.exec(text);
  if (!match) return text;
  const codeRegions = findCodeBlockRegions(text);
  if (isInRegion(match.index, codeRegions)) return text;
  return text.slice(0, match.index);
}

export function collectSearchImages(
  parts: ReadonlyArray<{ type: string; toolName?: string; result?: unknown }>,
): Map<string, SearchImageEntry> {
  const images = new Map<string, SearchImageEntry>();
  for (const part of parts) {
    if (
      part.type !== "tool-call" ||
      part.toolName !== SEARCH_IMAGE_TOOL
    )
      continue;
    if (!isSearchImagesToolResult(part.result)) continue;
    for (const entry of part.result.webImages) {
      if (!images.has(entry.id)) images.set(entry.id, entry);
    }
  }
  return images;
}

export function searchImagesSignature(
  parts: ReadonlyArray<{ type: string; toolName?: string; result?: unknown }>,
): string {
  const entries = Array.from(collectSearchImages(parts).values());
  return entries.length === 0 ? "" : JSON.stringify(entries);
}

export function parseSearchImagesSignature(
  signature: string,
): Map<string, SearchImageEntry> {
  if (!signature) return new Map();
  try {
    const parsed = JSON.parse(signature) as unknown;
    if (!Array.isArray(parsed)) return new Map();
    return new Map(
      parsed.filter(isSearchImageEntry).map((entry) => [entry.id, entry]),
    );
  } catch {
    return new Map();
  }
}
