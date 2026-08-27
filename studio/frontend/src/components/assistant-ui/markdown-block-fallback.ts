// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What one Markdown block should show when it cannot be RENDERED.
 *
 * Streamdown loads two parts of a reply through `React.lazy`: the syntax
 * highlighted code body and the Mermaid renderer. Both are fetched at the
 * moment a reply first contains that construct, and a lazy import that rejects
 * rethrows during render. Nothing between the thread and the router catches
 * that today, so one chunk that will not load replaces the whole application
 * with the router's error page and takes the reply that was already on screen
 * with it.
 *
 * The answer is to keep the CONTENT. A fence whose highlighter is missing is
 * still perfectly readable as text: it is the same characters in the same
 * order, which is what the model actually said. So the degraded form of a block
 * is its own source, with the Markdown fence scaffolding removed so a reader
 * sees code rather than backticks.
 *
 * Deliberately NOT an error card and deliberately not blank. A reader who asked
 * a model for a shell command needs the command, not an apology, and least of
 * all an empty box where their answer used to be.
 */

export type MarkdownBlockFallback = {
  /** The text to show. Empty only when the block itself carried no content. */
  text: string;
  /** The fence's language tag, when the block was a fence and named one. */
  language: string | null;
  /** Whether the source was a fenced block, so the caller can pick `pre` over prose. */
  fenced: boolean;
};

type OpeningFence = {
  /** The opening run, whose CHARACTER and LENGTH both constrain the close. */
  marker: string;
  /** Everything after the run on the opening line, the info string. */
  info: string;
  /** The block after the opening line. */
  body: string;
  /** The opener's own indentation, which every content line is measured against. */
  indent: number;
};

/**
 * Scanned rather than matched.
 *
 * The regex this replaces put `[^\r\n]*` straight after `` `{3,} ``, and the two
 * compete for the same backticks, so an opening run with no line break after it
 * backtracks quadratically. Scanning takes the run greedily once and never
 * reconsiders, which is linear by construction on any input.
 */
function openingFence(content: string): OpeningFence | null {
  let i = 0;
  while (i < 3 && content[i] === " ") i += 1;
  const char = content[i];
  if (char !== "`" && char !== "~") return null;
  let run = 0;
  while (content[i + run] === char) run += 1;
  if (run < 3) return null;
  const lineEnd = content.indexOf("\n", i + run);
  // A reply that ENDS on the opening line has still opened a fence. CommonMark
  // 0.31.2 closes an unclosed block at the end of the document, so the opening
  // line is the whole fence and the body is empty. Returning null here showed
  // the delimiter and the language tag as prose instead.
  const rest = lineEnd === -1 ? content.length : lineEnd;
  const info = content.slice(i + run, rest).replace(/\r$/, "");
  // "If the info string comes after a backtick fence, it may not contain any
  // backtick characters" (CommonMark 0.31.2). Such a line is a PARAGRAPH, so
  // the block is not a whole-block fence and belongs to the caller unchanged;
  // treating it as one drops the opening line and the reader loses that text.
  // Tilde fences carry no such restriction. `CODE_FENCE_RE` in
  // `features/chat/artifacts/html-fences` already spells the same rule.
  if (char === "`" && info.includes("`")) return null;
  const body = lineEnd === -1 ? "" : content.slice(lineEnd + 1);
  return { marker: char.repeat(run), info, body, indent: i };
}

/**
 * The body with the opener's indentation taken off each line.
 *
 * "If the leading code fence is indented N spaces, then up to N spaces of
 * indentation are removed from each line of the content" (CommonMark 0.31.2).
 * UP TO: a line indented less than the opener loses only what it has, and a line
 * indented more keeps the remainder, which is the code's own structure.
 *
 * Recognising the indent when opening the fence and then not removing it is the
 * half of the rule that shows. The rendered block reads `x = 1`, the degraded
 * one reads `   x = 1`, and a reader who copies that into a file gets an
 * IndentationError from text their model never wrote. `extractHtmlFences`
 * already applies the same rule to the same construct.
 */
function stripIndent(body: string, indent: number): string {
  if (indent === 0 || body === "") return body;
  const lines = body.split("\n");
  for (let n = 0; n < lines.length; n += 1) {
    let take = 0;
    while (take < indent && lines[n][take] === " ") take += 1;
    lines[n] = lines[n].slice(take);
  }
  return lines.join("\n");
}

/**
 * Whether the line `[from, to)` of `text` closes a fence opened with `marker`.
 *
 * CommonMark 0.31.2 requires the closing fence to use the same character and
 * "at least as many" of it, so the opening length is a MINIMUM and not a match:
 * a four-backtick close is how a model closes a fence whose body contains a
 * three-backtick one. The previous back-reference demanded the exact same run
 * and left the close on screen as if it were code.
 *
 * Bounds rather than a line STRING because the caller has thousands of lines and
 * the only thing most of them need is a look at their first character. Slicing
 * each one, and testing each one against a regex, cost more than everything else
 * in the scan put together.
 */
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
  // A closing fence carries no info string; only trailing spaces and tabs are
  // allowed, plus the carriage return of a CRLF line, which is the LAST
  // character of the line and only ever one. Accepting a CR anywhere in the tail
  // instead would quietly close fences the line-based code did not.
  const last = text[to - 1] === "\r" ? to - 1 : to;
  for (let j = i + run; j < last; j += 1) {
    const c = text[j];
    if (c !== " " && c !== "\t") return false;
  }
  return true;
}

/**
 * The readable form of a block that failed to render.
 *
 * Falls back to the block's own source unchanged for anything that is not a
 * whole-block fence: a paragraph, a list or a table is already readable as
 * Markdown, and inventing a renderer here would be a second thing to go wrong.
 *
 * This runs only once rendering has ALREADY failed, so it is the last thing
 * between the reader and a lost block. Getting the fence wrong here does not
 * cost a nicety, it puts stray backticks in the only view of the answer that
 * still exists.
 */
export function markdownBlockFallback(content: string): MarkdownBlockFallback {
  const open = openingFence(content);
  if (!open) {
    return { text: content, language: null, fenced: false };
  }
  const body = fenceBody(open.body, open.marker);
  // The fence closed with the block still running, so the block is a DOCUMENT
  // that merely starts with a fence. Its own source is the readable form.
  if (body === null) {
    return { text: content, language: null, fenced: false };
  }
  const language = open.info.trim().split(/\s+/)[0] || null;
  return { text: stripIndent(body, open.indent), language, fenced: true };
}

/**
 * The fence's content, or null when the fence does not own the whole block.
 *
 * One block is usually one construct, but not always: Streamdown 2.5's
 * `parseMarkdownIntoBlocks` returns an ENTIRE reply as a single block once it
 * contains a footnote (measured: the same reply splits into 5 blocks without one
 * and 1 with). A fence at the top of such a block is followed by its own close
 * and then by prose, and reading the whole thing as the fence's body put the
 * closing delimiter, the prose and the footnote on screen as if the model had
 * written them as Python.
 *
 * So the close is looked for from the TOP, and where it falls decides:
 *   last line   the fence is the block. Its content is everything above.
 *   earlier     the block continues past the fence, so it is not a whole-block
 *               fence and belongs to the caller unchanged.
 *   never       still streaming. All of it is content, which is the case the
 *               reader needs most: the failure happens mid-fence, long before it
 *               closes.
 *
 * Driven by `indexOf` on the delimiter rather than walked line by line. A close
 * has to carry at least three of the opening character, so the only lines worth
 * looking at are the ones holding such a run, and an ordinary code block has
 * exactly one: its own. Walking every line instead cost ~50ns per LINE, which a
 * long block pays in full every time it re-renders.
 */
function fenceBody(body: string, marker: string): string | null {
  // The line break before the closing fence belongs to the fence's line, and a
  // block may or may not carry a trailing one of its own.
  const withoutTrailingBreak = body.replace(/\r?\n$/, "");
  // Three, not the marker itself: the opening run can be enormous, and a needle
  // that long is both slow to build and slow to search for. This is a superset
  // of the candidates, and `closesFenceAt` is what decides.
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
      // An empty fence closes on the line after it opens, so there is no body at
      // all. Returning the closing run here is what showed ``` as its own code.
      if (start === 0) return "";
      const cut =
        withoutTrailingBreak[start - 2] === "\r" ? start - 2 : start - 1;
      return withoutTrailingBreak.slice(0, cut);
    }
    if (nl === -1) return body;
    // Past the whole LINE, not past the run: every delimiter left on a line the
    // scan has already rejected would be re-examined for nothing.
    from = nl + 1;
  }
}
