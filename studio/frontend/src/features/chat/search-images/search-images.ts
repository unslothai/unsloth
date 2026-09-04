// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// web_search image results: the backend keeps the URLs and hands out opaque `[[img:<id>]]`
// tokens, resolved here against that tool call's envelope.

import { findCodeBlockRegions, isInRegion } from "../../../lib/latex.ts";

export const SEARCH_IMAGES_MARKER = "\n__WEB_IMAGES__:";
export const SEARCH_IMAGE_TAG = "search-image";
// The only tool whose result may carry the envelope; elsewhere the marker is content.
export const SEARCH_IMAGE_TOOL = "web_search";
const IMAGE_ID_RE = /^[0-9a-f]{12}$/;
const TOKEN_RE = /\[\[img:([0-9a-f]{12})\]\]/g;
// A token cut off mid-stream: any suffix of "[[img:xxxxxxxxxxxx]]" short of the close.
const PARTIAL_TOKEN_RE = /\[(?:\[(?:i(?:m(?:g(?::[0-9a-f]{0,12}\]?)?)?)?)?)?$/;

export interface SearchImageEntry {
  id: string;
  title: string;
  domain: string;
  source: string;
  /** What was searched for, so the renderer can place the card itself. */
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
    // Re-checked so a spoofed envelope cannot put another scheme in an <a href>.
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

/** Split a trailing image envelope off a web_search result; unchanged when absent or malformed. */
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

/** Result text in either shape; citations come from this, and an image-bearing result is an object. */
export function searchResultText(result: unknown): string {
  if (typeof result === "string") return result;
  if (isSearchImagesToolResult(result)) return result.text;
  return "";
}

export function searchImagePath(id: string): string {
  return `/api/inference/search-images/${encodeURIComponent(id)}`;
}

/** `[[img:<id>]]` -> `<search-image token="...">`, skipping code; unknown ids are dropped.
 *  `token`, not `id`: rehype-sanitize prefixes `id` values with `user-content-`. */
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

/** Drop every `[[img:<id>]]` the model wrote. The tokens are markup for the renderer, so
 *  anything handing the answer to the user as plain text -- clipboard, export, read-aloud --
 *  has to strip them or they show up verbatim. */
export function stripSearchImageTokens(text: string): string {
  if (!text.includes("[[img:")) return text;
  const codeRegions = findCodeBlockRegions(text);
  // One pass over the original, so the code-region offsets stay valid. A token sits in its own
  // block, so the first branch takes the blank line introducing it too; dropping the token
  // alone would leave a widening gap. Code is left alone, where the token is prose.
  return text.replace(
    /\n\n[ \t]*\[\[img:[0-9a-f]{12}\]\][ \t]*(?=\n\n|\n?$)|\[\[img:[0-9a-f]{12}\]\]/g,
    (match, offset: number) => (isInRegion(offset, codeRegions) ? match : ""),
  );
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Whether a `[[img:<id>]]` the model wrote sits somewhere it will actually render. */
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

/** Offset of the first match outside `regions`, or null. Resets `pattern` for reuse. */
function firstMatchOutsideCode(
  pattern: RegExp,
  text: string,
  regions: Array<[number, number]>,
): number | null {
  if (!text) return null;
  pattern.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    // The leading group is a boundary char, not part of the name.
    const at = match.index + match[1].length;
    if (!isInRegion(at, regions)) return at;
    // Zero-width matches cannot happen (the name is non-empty), so no manual bump.
  }
  return null;
}

const LIST_MARKER_RE = /^(\s*(?:[-*+•]|\d{1,2}[.)])\s+)/;
const HEADING_LINE_RE = /^\s*#{1,6}\s/;
const BLOCK_BREAK_RE = /^(?:\s*$|\s*(?:[-*+•]|\d{1,2}[.)])\s|\s*#{1,6}\s|\s*(?:```|~~~))/;
// Display math, which BLOCK_BREAK_RE cannot see: a blank line inside `$$ ... $$` read as the
// end of the block and the card was spliced into the equation. Bounded like latex.ts, so an
// unclosed `$$` cannot run the scan over the whole answer.
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

/** End of the block containing `index`. Models wrap a list item across several lines, and
 *  inserting at the end of the first one splits the sentence in half. */
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

/** Whether two lowercased subject names refer to the same thing, on word boundaries. */
function namesSameThing(a: string, b: string): boolean {
  if (a === b) return true;
  const [shorter, longer] = a.length <= b.length ? [a, b] : [b, a];
  return new RegExp(
    `(^|[^\\p{L}\\p{N}])${escapeRegExp(shorter)}([^\\p{L}\\p{N}]|$)`,
    "u",
  ).test(longer);
}

/** Put each subject's image under the first line naming it. Subjects the answer never names
 *  are left out, tokens the model placed win, and streaming text is untouched. */
export function placeSubjectImages(
  text: string,
  images: ReadonlyMap<string, SearchImageEntry>,
  isStreaming: boolean,
  // Earlier text parts, so a subject named twice is illustrated once.
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
    // Outside code only, the same rule the subject match follows: rewriteSearchImageTokens leaves
    // a token inside a fence as literal text, so counting that as "already placed" showed no
    // picture at all.
    if (
      messageParts.some(({ text: part, codeRegions: regions }) =>
        tokenPlacedOutsideCode(part, entry.id, regions),
      )
    ) {
      continue;
    }
    // On the original text: lowercasing can change length and shift every offset. Global, so a
    // mention inside code can be stepped over rather than ending the search: a name in a snippet
    // used to abandon the subject a later prose item names.
    const pattern = new RegExp(
      `(^|[^\\p{L}\\p{N}])${escapeRegExp(key)}(?![\\p{L}\\p{N}])`,
      "giu",
    );
    // An earlier text part already carries this subject's card -- unless it only said so in code,
    // which shows no card and so must not suppress this one.
    if (firstMatchOutsideCode(pattern, alreadyNamed, namedRegions) !== null) continue;
    // Where the NAME starts, not the boundary char before it: same line either way, and the
    // code-region test is about the name.
    const at = firstMatchOutsideCode(pattern, text, codeRegions);
    if (at === null) continue;
    const lineStart = text.lastIndexOf("\n", at) + 1;
    const newline = text.indexOf("\n", at);
    const lineEnd = newline === -1 ? text.length : newline;
    const line = text.slice(lineStart, lineEnd);
    const marker = LIST_MARKER_RE.exec(line);
    if (HEADING_LINE_RE.test(line)) {
      // A heading is a one-line block, so its own end is the insertion point.
      insertions.push({ at: lineEnd, chunk: `\n\n[[img:${entry.id}]]` });
    } else if (marker) {
      // Indented to the item's content column, so the card is a block inside the item rather than a
      // lazy continuation of its sentence, and the list keeps its numbering.
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

// Split marker from subject, and never let two whitespace quantifiers span the same run: the
// single-regex form backtracked at ~O(n^3.5), so 250 spaces in one bullet froze the thread
// for 13s. The body is anchored on non-space edges so the surrounding runs cannot overlap.
const LIST_ITEM_MARKER_RE = /^[ \t]*(?:\d{1,2}[.)]|[-*+•])[ \t]+/;
const LIST_ITEM_RE =
  /^(?:\*\*|__)?[ \t\r]*([^\s*_\n][^*_\n]{0,58}?[^\s*_\n])(?:[ \t\r]*(?:\*\*|__))?[ \t\r]*(?::|[-–—][ \t\r]|\(|$)/;
const HEADING_RE =
  /^[ \t]*#{2,4}[ \t]+(?:\d{1,2}[.)][ \t]+)?([^\s\n#][^\n#]{0,58}?[^\s\n#])(?:[ \t\r]*#*)?[ \t\r]*$/;
const MAX_AUTO_SUBJECTS = 5;
// Items that start with an instruction: "Install Python", "Preheat the oven".
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

// Section labels that lead list items in comparisons and reviews, not things.
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

/** The lead of each listed item or small heading. Short names only; a procedure, a code answer
 *  or a single item yields nothing. */
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
  // Proportional, so "Choose the Civic if:" at the end of a comparison is not a how-to.
  if (steps >= named.length) return [];
  return named.length >= 2 ? named.slice(0, MAX_AUTO_SUBJECTS) : [];
}

/** List subjects the model has not already fetched a picture for. */
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
  // Word boundaries: "cat" must not swallow "Caterpillar".
  return listed.filter((name) => {
    const key = name.toLowerCase();
    return !covered.some((c) => namesSameThing(c, key));
  });
}

/** While streaming, hold back a half-arrived token so it never flashes as literal text. */
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

/** The `[[img:<id>]]` ids a message can resolve, read from its web_search tool parts. */
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

/** A stable string for the same set of entries, so a store subscription can select it. */
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
