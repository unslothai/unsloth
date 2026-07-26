// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Release notes are authored in CHANGELOG.md, where a relative link means
 * "somewhere in the Unsloth repository". Rendered inside Studio those links
 * would resolve against Studio's own origin, so the renderer blocks them or
 * points them at a Studio route. Rewriting them to absolute repository URLs
 * first makes them behave the way GitHub renders the same file.
 */

import { codeSpans, insideSpan } from "@/lib/markdown-code-spans";

const LINK_BASE = "https://github.com/unslothai/unsloth/blob/main/";
const IMAGE_BASE = "https://raw.githubusercontent.com/unslothai/unsloth/main/";

// Inline `](dest)` plus the `[label]: dest` reference form. The destination is
// either <bracketed> or runs to whitespace or the closing paren.
const INLINE_TARGET =
  /(!?)\[((?:[^[\]\\]|\\.)*)\]\(\s*(<[^<>\n]*>|(?:\\.|[^\s()])*)/g;
const REFERENCE_TARGET = /^( {0,3}\[((?:[^[\]\\]|\\.)*)\]:\s*)(<[^<>\n]*>|\S+)/;
// `![alt][label]`, `![label][]` and `![label]`: a definition they point at
// has to resolve to the raw file, not to its page on GitHub.
const IMAGE_REFERENCE = /!\[((?:[^[\]\\]|\\.)*)\](?:\[((?:[^[\]\\]|\\.)*)\])?/g;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
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
function rewriteLine(line: string, imageLabels: Set<string>): string {
  const reference = REFERENCE_TARGET.exec(line);
  if (reference) {
    const target = reference[3] ?? "";
    const resolved = absolute(
      unwrap(target),
      imageLabels.has(label(reference[2] ?? "")),
    );
    const rest = line.slice(reference[0].length);
    return `${reference[1]}${wrap(resolved, target)}${rest}`;
  }

  // Code spans are literal, so their contents keep whatever they say.
  const spans = codeSpans(line);

  INLINE_TARGET.lastIndex = 0;
  return line.replace(INLINE_TARGET, (match, bang, text, target, offset) => {
    if (insideSpan(spans, offset)) {
      return match;
    }
    const resolved = absolute(unwrap(target), bang === "!");
    return `${bang}[${text}](${wrap(resolved, target)}`;
  });
}

/** Runs `visit` over the lines outside fenced code blocks. */
function outsideFences(
  lines: string[],
  visit: (line: string, index: number) => void,
): void {
  let openFence: string | null = null;
  lines.forEach((line, index) => {
    const fence = FENCE.exec(line);
    if (fence) {
      const marker = fence[1] ?? "";
      if (openFence === null) {
        openFence = marker;
      } else if (
        // A closer matches the opening character and carries nothing after it.
        marker[0] === openFence[0] &&
        marker.length >= openFence.length &&
        !NON_SPACE.test(fence[2] ?? "")
      ) {
        openFence = null;
      }
      return;
    }
    if (openFence === null) {
      visit(line, index);
    }
  });
}

/** Absolute repository URLs for every relative link and image in `markdown`. */
export function resolveChangelogLinks(markdown: string): string {
  const lines = markdown.split("\n");

  // Which definitions are used as images has to be known before rewriting
  // them, because only images resolve against the raw host.
  const imageLabels = new Set<string>();
  outsideFences(lines, (line) => {
    IMAGE_REFERENCE.lastIndex = 0;
    for (
      let match = IMAGE_REFERENCE.exec(line);
      match !== null;
      match = IMAGE_REFERENCE.exec(line)
    ) {
      const explicit = match[2] ?? "";
      imageLabels.add(label(explicit.trim() ? explicit : (match[1] ?? "")));
    }
  });

  const rewritten = [...lines];
  outsideFences(lines, (line, index) => {
    rewritten[index] = rewriteLine(line, imageLabels);
  });
  return rewritten.join("\n");
}
