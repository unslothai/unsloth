// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Release notes are authored in CHANGELOG.md, where a relative link means
 * "somewhere in the Unsloth repository". Rendered inside Studio those links
 * would resolve against Studio's own origin, so the renderer blocks them or
 * points them at a Studio route. Rewriting them to absolute repository URLs
 * first makes them behave the way GitHub renders the same file.
 */

const LINK_BASE = "https://github.com/unslothai/unsloth/blob/main/";
const IMAGE_BASE = "https://raw.githubusercontent.com/unslothai/unsloth/main/";

// Inline `](dest)` plus the `[label]: dest` reference form. The destination is
// either <bracketed> or runs to whitespace or the closing paren.
const INLINE_TARGET = /(!?)\[((?:[^[\]\\]|\\.)*)\]\(\s*(<[^<>\n]*>|[^\s()]*)/g;
const REFERENCE_TARGET = /^( {0,3}\[((?:[^[\]\\]|\\.)*)\]:\s*)(<[^<>\n]*>|\S+)/;
// `![alt][label]`, `![label][]` and `![label]`: a definition they point at
// has to resolve to the raw file, not to its page on GitHub.
const IMAGE_REFERENCE = /!\[((?:[^[\]\\]|\\.)*)\](?:\[((?:[^[\]\\]|\\.)*)\])?/g;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const CODE_SPAN = /(`+)[\s\S]*?\1(?!`)/g;
// A scheme, a protocol-relative host, or a fragment: already absolute enough.
// `//` needs a host after it, so `///docs` stays a repository path.
const ABSOLUTE = /^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/\/[^/]|#)/;

/** A reference label as CommonMark compares them. */
function label(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

function absolute(target: string, image: boolean): string {
  const base = image ? IMAGE_BASE : LINK_BASE;
  const trimmed = target.trim();
  if (!trimmed || ABSOLUTE.test(trimmed)) {
    return target;
  }
  try {
    // GitHub resolves a leading slash against the repository root, not the
    // site root, so it is appended to the base rather than replacing its path.
    const resolved = new URL(trimmed.replace(/^\/+/, ""), base).toString();
    // `../` can climb out of the repository. Leave those alone rather than
    // inventing a target outside it.
    return resolved.startsWith(base) ? resolved : target;
  } catch {
    return target;
  }
}

/** Rewrites one line's link and image targets, leaving code spans alone. */
function rewriteLine(line: string, imageLabels: Set<string>): string {
  const reference = REFERENCE_TARGET.exec(line);
  if (reference) {
    const target = reference[3] ?? "";
    const bracketed = target.startsWith("<") && target.endsWith(">");
    const inner = bracketed ? target.slice(1, -1) : target;
    const resolved = absolute(
      inner,
      imageLabels.has(label(reference[2] ?? "")),
    );
    const rest = line.slice(reference[0].length);
    return `${reference[1]}${bracketed ? `<${resolved}>` : resolved}${rest}`;
  }

  // Code spans are literal, so their contents keep whatever they say.
  const spans: [number, number][] = [];
  CODE_SPAN.lastIndex = 0;
  for (
    let span = CODE_SPAN.exec(line);
    span !== null;
    span = CODE_SPAN.exec(line)
  ) {
    spans.push([span.index, span.index + span[0].length]);
  }
  const inSpan = (index: number): boolean =>
    spans.some(([start, end]) => index >= start && index < end);

  INLINE_TARGET.lastIndex = 0;
  return line.replace(INLINE_TARGET, (match, bang, text, target, offset) => {
    if (inSpan(offset)) {
      return match;
    }
    const bracketed = target.startsWith("<") && target.endsWith(">");
    const inner = bracketed ? target.slice(1, -1) : target;
    const resolved = absolute(inner, bang === "!");
    return `${bang}[${text}](${bracketed ? `<${resolved}>` : resolved}`;
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
        !(fence[2] ?? "").trim()
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
