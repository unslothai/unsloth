// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { JSON_SCHEMA, load } from "js-yaml";

/*
 * Chat and document markdown never loads a remote image (`safeMarkdownUrl`), but mermaid draws
 * into the live page and fetches what a diagram names while it measures it, before any SVG comes
 * back. So the diagram source is vetted before mermaid sees it, the same way: an image a diagram
 * names must be a `data:` URI. A source naming none comes back unchanged, so ordinary diagrams
 * render exactly as before; running this twice gives the same text.
 */

const INLINE_IMAGE = /^\s*data:/i;

// Elements an HTML label could use to fetch something. `<` before one of them becomes mermaid's
// own `#lt;` entity, so the label shows the text instead of building the element.
const FETCHING_TAG =
  /<(?=\/?(?:img|image|picture|source|video|audio|track|input|iframe|frame|embed|object|link|style|svg|use|meta|base|body|table|thead|tbody|tr|td|th)\b)/gi;

// A style rule that could name a resource: CSS functions that load one, or an escape hiding one.
const STYLE_LINE = /^[^\S\n]*(?:classDef|style|linkStyle)\b.*$/gim;
const STYLE_FETCH = /\\|url\s*\(|image(?:-set)?\s*\(|cross-fade\s*\(|element\s*\(|@import/i;

// An HTML label tag with a style attribute; its CSS can fetch (background-image and the like).
const STYLED_TAG = /<([a-z][^<>]*\bstyle\s*=[^<>]*)/gi;
// Numeric character references, which mermaid (`#117;`) and the browser (`&#117;`) both decode.
const CHAR_REF = /(?:&#|#)(x[0-9a-f]+|\d+);?/gi;
const CSS_FETCH = /url|image|cross-fade|element\s*\(|@import|\\/i;

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

/** The `id@{ ... }` shape data blocks, as mermaid's lexer ends them: at the first `}` outside a "string". */
function shapeDataBlocks(source: string): { start: number; end: number; body: string }[] {
  const blocks: { start: number; end: number; body: string }[] = [];
  let from = 0;
  for (;;) {
    const open = source.indexOf("@{", from);
    if (open < 0) return blocks;
    let inString = false;
    let close = -1;
    for (let index = open + 2; index < source.length; index++) {
      const char = source[index];
      if (char === '"') inString = !inString;
      else if (char === "}" && !inString) {
        close = index;
        break;
      }
    }
    if (close < 0) return blocks;
    blocks.push({ start: open, end: close + 1, body: source.slice(open + 2, close) });
    from = close + 1;
  }
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

/** The diagram source with every image or style that would load from elsewhere taken out. */
export function stripRemoteDiagramImages(source: string): string {
  let out = "";
  let at = 0;
  // Comments are blanked to the same length, so a `@{` or `"` in one cannot hide a later block.
  const scanned = source.replace(COMMENT_LINE, (line) => " ".repeat(line.length));
  for (const block of shapeDataBlocks(scanned)) {
    const replaced = withoutRemoteImage(block.body);
    if (replaced === null) continue;
    out += source.slice(at, block.start) + replaced;
    at = block.end;
  }
  out += source.slice(at);
  if (SEQUENCE.test(scanned)) out = out.replace(DETAILS_LINE, "");
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
    .replace(STYLE_LINE, (line) => (STYLE_FETCH.test(line) ? "" : line))
    .replace(ACTOR_PROPERTIES, (whole, lead: string, head: string, json: string) => {
      const replaced = withoutRemoteIcon(json);
      return replaced === null ? whole : `${lead}${head}${replaced}`;
    });
}
