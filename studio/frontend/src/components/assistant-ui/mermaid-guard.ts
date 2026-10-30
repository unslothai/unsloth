// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { JSON_SCHEMA, load } from "js-yaml";

/*
 * Markdown never loads remote images (safeMarkdownUrl), but mermaid fetches what a diagram names
 * while measuring, so the source is vetted first: an image must be a data: URI; a source naming
 * none is unchanged, and a second pass gives the same text.
 */

const INLINE_IMAGE = /^\s*data:/i;

// Elements an HTML label could use to fetch something. `<` before one of them becomes mermaid's
// own `#lt;` entity, so the label shows the text instead of building the element.
const FETCHING_TAG =
  /<(?=\/?(?:img|image|picture|source|video|audio|track|input|iframe|frame|embed|object|link|style|svg|use|meta|base|body|table|thead|tbody|tr|td|th)\b)/gi;

// A style rule that could name a resource: CSS functions that load one, or an escape hiding one.
// A statement may follow `;`; a stateDiagram classDef runs to end of line, so cut from the keyword on.
const STYLE_START = /(^|[;\s])[^\S\n]*(?:classDef|style|linkStyle)\b/gi;
const STYLE_FETCH = /\\|url\s*\(|image(?:-set)?\s*\(|cross-fade\s*\(|element\s*\(|@import/i;
const STYLE_FETCHES = new RegExp(STYLE_FETCH.source, "gi");
// These lexers take a style statement mid-line after a space and allow brackets in ids: every keyword counts.
const MIDLINE_STYLE = /^[^\S\n]*(?:stateDiagram|erDiagram|block(?:-beta)?|requirementDiagram)\b/im;

// An HTML label tag with a style attribute; its CSS can fetch (background-image and the like).
const STYLED_TAG = /<([a-z][^<>]*\bstyle\s*=[^<>]*)/gi;
// Numeric character references, which mermaid (`#117;`) and the browser (`&#117;`) both decode.
const CHAR_REF = /(?:&#|#)(x[0-9a-f]+|\d+);?/gi;
// `&` left after the numeric decode is a named reference (`&bsol;` is a backslash), so it counts too.
const CSS_FETCH = /url|image|cross-fade|element\s*\(|@import|\\|&/i;

// Mermaid drops a whole line starting with `%%` before parsing (an `%%{init}%%` directive stays).
const COMMENT_LINE = /^[^\S\n]*%%(?!\{).*$/gm;

// `details Actor: id` in a sequence diagram merges the page element `id`'s JSON (an `icon` too),
// which this cannot see; the line is dropped.
const DETAILS_LINE = /^[^\S\n]*details\b[^:\n[({]*:.*$/gim;
const SEQUENCE = /^[^\S\n]*sequenceDiagram\b/m;

// A `%%{init}%%` directive or front-matter config whose values could fetch (an XY chart copies
// theme colours into SVG fills, so url() there loads). Ordinary theme colours stay.
const DIRECTIVE = /%%\{[\s\S]*?\}%%/g;
const FRONT_MATTER = /^\s*---[^\S\n]*\n[\s\S]*?\n[^\S\n]*---[^\S\n]*(?=\n|$)/;

// `properties Actor: {json}` in a sequence diagram, where `icon` is drawn as an <image>.
const ACTOR_PROPERTIES = /(^|;)([^\S\n]*properties\b[^:\n;#]*:)([^#\n;]*)/gim;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Every `@{` with its block up to the first `}` outside a "string", read on its own: lexer state decides
 * whether a `@{` is shape data or text, so blocks may overlap. Blocks without `img` or `\` are skipped.
 */
function shapeDataBlocks(source: string): { start: number; end: number; body: string }[] {
  const blocks: { start: number; end: number; body: string }[] = [];
  if (!source.includes("@{")) return blocks;
  // First `}` outside a string from i, starting outside (closeFrom) or inside (quotedFrom) one: linear.
  const closeFrom = new Int32Array(source.length + 1).fill(-1);
  const quotedFrom = new Int32Array(source.length + 1).fill(-1);
  for (let index = source.length - 1; index >= 0; index--) {
    const char = source[index];
    if (char === '"') {
      closeFrom[index] = quotedFrom[index + 1];
      quotedFrom[index] = closeFrom[index + 1];
    } else {
      closeFrom[index] = char === "}" ? index : closeFrom[index + 1];
      quotedFrom[index] = quotedFrom[index + 1];
    }
  }
  const hintEnd = new Int32Array(source.length + 1).fill(-1);
  for (let index = source.length - 1; index >= 0; index--) {
    if (source[index] === "\\") hintEnd[index] = index + 1;
    else if (source.slice(index, index + 3).toLowerCase() === "img") hintEnd[index] = index + 3;
    else hintEnd[index] = hintEnd[index + 1];
  }
  for (let open = source.indexOf("@{"); open >= 0; open = source.indexOf("@{", open + 2)) {
    const close = closeFrom[open + 2];
    const hint = hintEnd[open + 2];
    if (close >= 0 && hint >= 0 && hint <= close) {
      blocks.push({ start: open, end: close + 1, body: source.slice(open + 2, close) });
    }
  }
  return blocks;
}

// Mermaid's own entity for `@`: it shows as `@` in text and never opens shape data.
const INERT_OPENER = "#64;{";

// Overlapping blocks can sum to the source's square; past this every `@{` is made inert unparsed.
const PARSE_BUDGET = 1 << 20;

/**
 * Remote images taken out of every block: a lone block is rewritten, a run of overlapping ones has all
 * its openers made inert (which reading wins depends on lexer state). Repeats until stable.
 */
function withoutRemoteShapeImages(source: string): string {
  let current = source;
  let budget = PARSE_BUDGET;
  for (let pass = 0; pass < 8; pass++) {
    // Comments are blanked to the same length, so a `@{` or `"` in one cannot hide a later block.
    const scanned = current.replace(COMMENT_LINE, (line) => " ".repeat(line.length));
    const fetching: { start: number; end: number; replaced: string }[] = [];
    for (const block of shapeDataBlocks(scanned)) {
      budget -= block.body.length;
      if (budget < 0) return current.replaceAll("@{", INERT_OPENER);
      const replaced = withoutRemoteImage(block.body);
      if (replaced !== null) fetching.push({ start: block.start, end: block.end, replaced });
    }
    if (fetching.length === 0) return current;
    let out = "";
    let at = 0;
    for (let first = 0; first < fetching.length; ) {
      let last = first;
      let runEnd = fetching[first].end;
      while (last + 1 < fetching.length && fetching[last + 1].start < runEnd) {
        last++;
        runEnd = Math.max(runEnd, fetching[last].end);
      }
      if (last === first) {
        out += current.slice(at, fetching[first].start) + fetching[first].replaced;
        at = fetching[first].end;
      } else {
        for (let index = first; index <= last; index++) {
          out += current.slice(at, fetching[index].start) + INERT_OPENER;
          at = fetching[index].start + 2;
        }
      }
      first = last + 1;
    }
    current = out + current.slice(at);
  }
  return current.replaceAll("@{", INERT_OPENER);
}

/** A shape data block without a remote `img`, or null when it carries none and stays as written. */
function withoutRemoteImage(body: string): string | null {
  let doc: unknown;
  try {
    // Parsed the way the flowchart parser parses it, so escaped keys and values are seen decoded.
    doc = load(body.includes("\n") ? `${body}\n` : `{\n${body}\n}`, { schema: JSON_SCHEMA });
  } catch {
    // Unparseable here is unparseable there; only drop it when it might name an image.
    return /img|\\/i.test(body) ? "" : null;
  }
  if (!isRecord(doc) || !("img" in doc)) return null;
  if (typeof doc.img === "string" && INLINE_IMAGE.test(doc.img)) return null;
  const rest = Object.fromEntries(Object.entries(doc).filter(([key]) => key !== "img"));
  return Object.keys(rest).length > 0 ? `@{ ${JSON.stringify(rest).slice(1, -1)} }` : "";
}

function withoutRemoteIcon(json: string): string | null {
  let properties: unknown;
  try {
    properties = JSON.parse(json);
  } catch {
    return /icon|\\|&/i.test(json) ? " {}" : null;
  }
  if (!isRecord(properties) || typeof properties.icon !== "string") return null;
  const icon = properties.icon.trim();
  // `@id` points at an element already in the page; `data:` is inline.
  if (icon.startsWith("@") || INLINE_IMAGE.test(icon)) return null;
  const rest = Object.fromEntries(Object.entries(properties).filter(([key]) => key !== "icon"));
  return ` ${JSON.stringify(rest)}`;
}

/** Outside `midLine` diagrams a statement starts a line or follows `;`; one inside a quoted or bracketed label is text. */
function withoutFetchingStyle(line: string, midLine: boolean): string {
  let lastFetch = -1;
  for (const found of line.matchAll(STYLE_FETCHES)) lastFetch = found.index;
  if (lastFetch < 0) return line;
  let depth = 0;
  let quoted = false;
  let at = 0;
  for (const match of line.matchAll(STYLE_START)) {
    const start = match.index + match[1].length;
    if (start > lastFetch) break;
    if (midLine) return line.slice(0, start);
    for (; at < start; at++) {
      const ch = line[at];
      if (ch === '"') quoted = !quoted;
      else if (!quoted && "[({".includes(ch)) depth++;
      else if (!quoted && ")]}".includes(ch)) depth = Math.max(0, depth - 1);
    }
    if (start === 0 || (match[1] === ";" && !quoted && depth === 0)) return line.slice(0, start);
  }
  return line;
}

/** The diagram source with every image or style that would load from elsewhere taken out. */
export function stripRemoteDiagramImages(source: string): string {
  let out = withoutRemoteShapeImages(source);
  const scanned = source.replace(COMMENT_LINE, (line) => " ".repeat(line.length));
  if (SEQUENCE.test(scanned)) out = out.replace(DETAILS_LINE, "");
  // Mermaid reads the diagram type after dropping directives: `%%{init}%%stateDiagram` counts.
  const midLine = MIDLINE_STYLE.test(scanned.replace(DIRECTIVE, ""));
  return out
    .replace(DIRECTIVE, (block) => (STYLE_FETCH.test(block) ? "" : block))
    .replace(FRONT_MATTER, (block) => (STYLE_FETCH.test(block) ? "" : block))
    .replace(FETCHING_TAG, "#lt;")
    .replace(STYLED_TAG, (whole, tag: string) => {
      const decoded = tag.replace(CHAR_REF, (_ref, code: string) => {
        const point = code[0].toLowerCase() === "x" ? parseInt(code.slice(1), 16) : Number(code);
        return point <= 0x10ffff ? String.fromCodePoint(point) : "";
      });
      return CSS_FETCH.test(decoded) ? `#lt;${tag}` : whole;
    })
    .replace(/^.*$/gm, (line) => withoutFetchingStyle(line, midLine))
    .replace(ACTOR_PROPERTIES, (whole, lead: string, head: string, json: string) => {
      const replaced = withoutRemoteIcon(json);
      return replaced === null ? whole : `${lead}${head}${replaced}`;
    });
}
