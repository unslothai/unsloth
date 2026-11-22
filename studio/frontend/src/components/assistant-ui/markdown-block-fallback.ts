// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What a Markdown block shows when it cannot render: its own source, fence scaffolding removed. */

export type MarkdownBlockFallback = {
  text: string;
  language: string | null;
  fenced: boolean;
};

type OpeningFence = {
  /** The opening run, whose CHARACTER and LENGTH both constrain the close. */
  marker: string;
  info: string;
  body: string;
  indent: number;
};

/** Scanned, not regex-matched: the regex backtracked quadratically on an unterminated opening run. */
function openingFence(content: string): OpeningFence | null {
  let i = 0;
  while (i < 3 && content[i] === " ") i += 1;
  const char = content[i];
  if (char !== "`" && char !== "~") return null;
  let run = 0;
  while (content[i + run] === char) run += 1;
  if (run < 3) return null;
  const lineEnd = content.indexOf("\n", i + run);
  // CommonMark 0.31.2 closes an unclosed block at document end, so an opening line alone is a fence.
  const rest = lineEnd === -1 ? content.length : lineEnd;
  const info = content.slice(i + run, rest).replace(/\r$/, "");
  // A backtick fence's info string may not contain backticks (CommonMark 0.31.2); such a line is a paragraph.
  if (char === "`" && info.includes("`")) return null;
  const body = lineEnd === -1 ? "" : content.slice(lineEnd + 1);
  return { marker: char.repeat(run), info, body, indent: i };
}

/** Remove up to the opener's indent from each line, by columns with 4-column tab stops (CommonMark 0.31.2). */
const TAB_STOP = 4;

function stripIndent(body: string, indent: number): string {
  if (indent === 0 || body === "") return body;
  const lines = body.split("\n");
  for (let n = 0; n < lines.length; n += 1) {
    const line = lines[n];
    let column = 0;
    let at = 0;
    let carry = "";
    while (column < indent && at < line.length) {
      const ch = line[at];
      if (ch === " ") {
        column += 1;
        at += 1;
        continue;
      }
      if (ch !== "\t") break;
      column += TAB_STOP - (column % TAB_STOP);
      at += 1;
      if (column > indent) {
        // The tab straddles the boundary: consume it and return the extra columns as spaces.
        carry = " ".repeat(column - indent);
        break;
      }
    }
    lines[n] = carry + line.slice(at);
  }
  return lines.join("\n");
}

/** Close needs the same char and AT LEAST the opener's length (CommonMark); bounds avoid slicing lines. */
function closesFenceAt(
  text: string,
  from: number,
  to: number,
  marker: string,
): boolean {
  let i = from;
  while (i < from + 3 && text[i] === " ") i += 1;
  let run = 0;
  while (i + run < to && text[i + run] === marker[0]) run += 1;
  if (run < marker.length) return false;
  // Only trailing spaces/tabs plus one final CR of a CRLF line may follow the run.
  const last = text[to - 1] === "\r" ? to - 1 : to;
  for (let j = i + run; j < last; j += 1) {
    const c = text[j];
    if (c !== " " && c !== "\t") return false;
  }
  return true;
}

export function markdownBlockFallback(content: string): MarkdownBlockFallback {
  const open = openingFence(content);
  if (!open) {
    return { text: content, language: null, fenced: false };
  }
  const body = fenceBody(open.body, open.marker);
  // The fence closed before the block ended, so the block's own source is the readable form.
  if (body === null) {
    return { text: content, language: null, fenced: false };
  }
  const language = open.info.trim().split(/\s+/)[0] || null;
  return { text: stripIndent(body, open.indent), language, fenced: true };
}

/**
 * Fence content, or null when the fence does not own the whole block (streamdown returns a whole
 * reply as one block once it has a footnote). Searches by indexOf: walking every line is slow.
 */
function fenceBody(body: string, marker: string): string | null {
  const withoutTrailingBreak = body.replace(/\r?\n$/, "");
  // Three, not the marker: a huge opening run is a slow needle; closesFenceAt decides.
  const probe = marker.slice(0, 3);
  let from = 0;
  for (;;) {
    const hit = withoutTrailingBreak.indexOf(probe, from);
    if (hit === -1) return body;
    const start = withoutTrailingBreak.lastIndexOf("\n", hit) + 1;
    const nl = withoutTrailingBreak.indexOf("\n", hit);
    const end = nl === -1 ? withoutTrailingBreak.length : nl;
    if (closesFenceAt(withoutTrailingBreak, start, end, marker)) {
      if (nl !== -1) return null;
      // An empty fence closes on the next line, so there is no body.
      if (start === 0) return "";
      const cut =
        withoutTrailingBreak[start - 2] === "\r" ? start - 2 : start - 1;
      return withoutTrailingBreak.slice(0, cut);
    }
    if (nl === -1) return body;
    from = nl + 1;
  }
}
