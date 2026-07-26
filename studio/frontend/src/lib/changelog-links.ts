// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Release notes are authored in CHANGELOG.md, where a relative link means
 * "somewhere in the Unsloth repository". Rendered inside Studio those links
 * would resolve against Studio's own origin, so the renderer blocks them or
 * points them at a Studio route. Rewriting them to absolute repository URLs
 * first makes them behave the way GitHub renders the same file.
 */

import {
  type CodeSpan,
  codeSpans,
  insideSpan,
} from "@/lib/markdown-code-spans";

const LINK_BASE = "https://github.com/unslothai/unsloth/blob/main/";
const IMAGE_BASE = "https://raw.githubusercontent.com/unslothai/unsloth/main/";

// Inline `](dest)` plus the `[label]: dest` reference form. The destination is
// either <bracketed> or runs to whitespace or the closing paren.
const NESTED_LABEL = String.raw`((?:[^[\]\\]|\\.|\[(?:[^[\]\\]|\\.)*\])*)`;
const INLINE_TARGET = new RegExp(
  String.raw`(!?)\[${NESTED_LABEL}\]\(\s*(<[^<>\n]*>|(?:\\.|[^\s()])*)`,
  "g",
);
const REFERENCE_TARGET = /^( {0,3}\[((?:[^[\]\\]|\\.)*)\]:\s*)(<[^<>\n]*>|\S+)/;
// `![alt][label]`, `![label][]` and `![label]`: a definition they point at
// has to resolve to the raw file, not to its page on GitHub.
const IMAGE_REFERENCE =
  /!\[((?:[^[\]\\]|\\.)*)\](?:\[((?:[^[\]\\]|\\.)*)\]|(?!\())/g;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
// Four spaces starts an indented code block, unless a paragraph is open.
const INDENTED_CODE = /^(?: {4}|\t)/;
// CommonMark type 1 HTML blocks show their contents verbatim.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
// Lines that are blocks in their own right, so no paragraph is open after.
const BLOCK_LINE =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$|>|=+[ \t]*$)/;
const LINE_ENDINGS = /\r\n?/g;
// A scheme, a protocol-relative host, or a fragment: already absolute enough.
// `//` needs a host after it, so `///docs` stays a repository path.
const ABSOLUTE = /^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/\/[^/]|#)/;

/** A reference label as CommonMark compares them. */
function label(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

const NEEDS_BRACKETS = /[()\s]/;
// `\(` in a destination is a literal paren, not part of the path.
const ESCAPE = /\\(.)/g;
// Only spaces and tabs may follow a closing fence.
const NON_SPACE = /[^ \t]/;
const LEADING_SLASHES = /^\/+/;

function absolute(target: string, image: boolean): string {
  const base = image ? IMAGE_BASE : LINK_BASE;
  const trimmed = target.trim().replace(ESCAPE, "$1");
  if (!trimmed || ABSOLUTE.test(trimmed)) {
    return target;
  }
  try {
    // GitHub resolves a leading slash against the repository root, not the
    // site root, so it is appended to the base rather than replacing its path.
    const resolved = new URL(
      trimmed.replace(LEADING_SLASHES, ""),
      base,
    ).toString();
    // `../` can climb out of the repository. Leave those alone rather than
    // inventing a target outside it.
    return resolved.startsWith(base) ? resolved : target;
  } catch {
    return target;
  }
}

/** True when the character at `index` is escaped by an odd run of slashes. */
function isEscaped(line: string, index: number): boolean {
  let slashes = 0;
  while (line[index - 1 - slashes] === "\\") {
    slashes += 1;
  }
  return slashes % 2 === 1;
}

/** The destination itself, without its angle brackets. */
function unwrap(target: string): string {
  return target.startsWith("<") && target.endsWith(">")
    ? target.slice(1, -1)
    : target;
}

/** The destination as it goes back into the line. */
function wrap(resolved: string, original: string): string {
  const bracketed = original.startsWith("<") && original.endsWith(">");
  return bracketed || (resolved !== original && NEEDS_BRACKETS.test(resolved))
    ? `<${resolved}>`
    : resolved;
}

/** Rewrites one line's link and image targets, leaving code spans alone. */
function rewriteLine(
  line: string,
  imageLabels: Set<string>,
  spans: CodeSpan[],
  base: number,
  isDefinition: boolean,
): string {
  const reference = isDefinition ? REFERENCE_TARGET.exec(line) : null;
  if (reference) {
    const target = reference[3] ?? "";
    const resolved = absolute(
      unwrap(target),
      imageLabels.has(label(reference[2] ?? "")),
    );
    const rest = line.slice(reference[0].length);
    return `${reference[1]}${wrap(resolved, target)}${rest}`;
  }

  INLINE_TARGET.lastIndex = 0;
  return line.replace(INLINE_TARGET, (match, bang, text, target, offset) => {
    // `\\[` is a literal bracket, so the expression is not a link.
    const opener = offset + (bang ? 1 : 0);
    if (insideSpan(spans, base + offset) || isEscaped(line, opener)) {
      return match;
    }
    const resolved = absolute(unwrap(target), bang === "!");
    // A badge nests an image inside a link, so the label is rewritten too.
    const inner = text.includes("](")
      ? rewriteLine(text, imageLabels, codeSpans(text), 0, false)
      : text;
    return `${bang}[${inner}](${wrap(resolved, target)}`;
  });
}

interface Classified {
  // Lines the renderer shows as Markdown, by index.
  text: number[];
  // Same lines, blanked where the renderer shows code, for span scanning.
  masked: string;
  // Lines where a `[label]: dest` definition can start.
  definition: Set<number>;
}

/**
 * Sorts lines into Markdown and code, and masks the code so a code span cannot
 * be paired across it. Offsets are preserved, so a span found in the mask is at
 * the same place in the document.
 */
function classify(lines: string[]): Classified {
  const text: number[] = [];
  const definition = new Set<number>();
  const masked: string[] = [];
  let openFence: string | null = null;
  let inRawHtml = false;
  let inCode = false;
  let afterParagraph = false;

  lines.forEach((line, index) => {
    if (inRawHtml) {
      inRawHtml = !RAW_HTML_CLOSE.test(line);
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    const fence = FENCE.exec(line);
    if (fence) {
      const marker = fence[1] ?? "";
      if (openFence === null) {
        // A backtick fence's info string may not contain a backtick.
        openFence =
          marker[0] !== "`" || !(fence[2] ?? "").includes("`") ? marker : null;
        if (openFence === null) {
          text.push(index);
          masked.push(line);
          afterParagraph = true;
          return;
        }
      } else if (
        // A closer matches the opening character and carries nothing after it.
        marker[0] === openFence[0] &&
        marker.length >= openFence.length &&
        !NON_SPACE.test(fence[2] ?? "")
      ) {
        openFence = null;
      }
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    if (openFence !== null) {
      masked.push(" ".repeat(line.length));
      return;
    }
    if (RAW_HTML_OPEN.test(line)) {
      inRawHtml = !RAW_HTML_CLOSE.test(line.replace(RAW_HTML_OPEN, ""));
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    const blank = !line.trim();
    // Indented code starts only outside a paragraph and runs to a dedent.
    if (inCode) {
      inCode = blank || INDENTED_CODE.test(line);
    } else {
      inCode = !afterParagraph && !blank && INDENTED_CODE.test(line);
    }
    if (inCode) {
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    // A definition cannot interrupt a paragraph.
    if (!afterParagraph) {
      definition.add(index);
    }
    text.push(index);
    masked.push(line);
    afterParagraph = !blank && !BLOCK_LINE.test(line);
  });

  return { text, masked: masked.join("\n"), definition };
}

/** Absolute repository URLs for every relative link and image in `markdown`. */
export function resolveChangelogLinks(markdown: string): string {
  // The desktop updater body arrives with CRLF, which would hide fences.
  const lines = markdown.replace(LINE_ENDINGS, "\n").split("\n");
  const { text, masked, definition } = classify(lines);
  // Scanned over the whole document, so a span may cross a line break.
  const spans = codeSpans(masked);

  // Offset of each line in the document, to place matches inside it.
  const offsets: number[] = [];
  let cursor = 0;
  for (const line of lines) {
    offsets.push(cursor);
    cursor += line.length + 1;
  }

  // Which definitions are used as images has to be known before rewriting
  // them, because only images resolve against the raw host.
  const imageLabels = new Set<string>();
  for (const index of text) {
    const line = lines[index] ?? "";
    IMAGE_REFERENCE.lastIndex = 0;
    for (
      let match = IMAGE_REFERENCE.exec(line);
      match !== null;
      match = IMAGE_REFERENCE.exec(line)
    ) {
      if (insideSpan(spans, (offsets[index] ?? 0) + match.index)) {
        continue;
      }
      const explicit = match[2] ?? "";
      imageLabels.add(label(explicit.trim() ? explicit : (match[1] ?? "")));
    }
  }

  const rewritten = [...lines];
  for (const index of text) {
    rewritten[index] = rewriteLine(
      lines[index] ?? "",
      imageLabels,
      spans,
      offsets[index] ?? 0,
      definition.has(index),
    );
  }
  return rewritten.join("\n");
}
