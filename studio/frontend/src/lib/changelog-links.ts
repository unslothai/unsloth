// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A relative link in CHANGELOG.md means "somewhere in the Unsloth repository",
 * but inside Studio it would resolve against Studio's own origin. Rewriting to
 * absolute repository URLs makes them behave the way GitHub renders the file.
 */

import {
  type CodeSpan,
  codeSpans,
  insideSpan,
} from "@/lib/markdown-code-spans";
import {
  EMPTY_LIST_STATE,
  type ListState,
  NO_QUOTE,
  type QuoteState,
  hiddenStructure,
  indentWidth,
  openLists,
  quoteState,
} from "@/lib/markdown-list-columns";

const LINK_BASE = "https://github.com/unslothai/unsloth/blob/main/";
const IMAGE_BASE = "https://raw.githubusercontent.com/unslothai/unsloth/main/";

// Inline `](dest)` plus the `[label]: dest` reference form. The destination is
// either <bracketed> or runs to whitespace or the closing paren. It may hold
// parentheses when they balance, so `[x]((draft).md)` points at `(draft).md`;
// one nesting level is all a file name needs, as in the preview's DESTINATION.
const NESTED_LABEL = String.raw`((?:[^[\]\\]|\\.|\[(?:[^[\]\\]|\\.)*\])*)`;
const BALANCED_DESTINATION = String.raw`(?:\\.|[^\s()]|\((?:\\.|[^\s()])*\))*`;
const PLAIN_DESTINATION = String.raw`(?:\\.|[^\s()])*`;
// A balanced pair counts only while a `)` or a title still closes the link
// after it. Otherwise the paren in `[x](a(b.md)` is the closer, which is how
// CommonMark reads it, and swallowing it would invent a link across lines.
const CLOSES_LINK = String.raw`(?=[ \t]*[)'"])`;
const INLINE_TARGET = new RegExp(
  String.raw`(!?)\[${NESTED_LABEL}\]\(\s*(<[^<>\n]*>|${BALANCED_DESTINATION}${CLOSES_LINK}|${PLAIN_DESTINATION})`,
  "g",
);
const REFERENCE_TARGET = /^( {0,3}\[((?:[^[\]\\]|\\.)*)\]:\s*)(<[^<>\n]*>|\S+)/;
// `![alt][label]`, `![label][]` and `![label]`: a definition they point at
// has to resolve to the raw file, not to its page on GitHub.
const IMAGE_REFERENCE =
  /!\[((?:[^[\]\\]|\\.)*)\](?:\[((?:[^[\]\\]|\\.)*)\]|(?!\())/g;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
// Four columns past the container start an indented code block, unless a
// paragraph is open. Inside a list item that is four past the item's content
// column, so a link indented under a bullet is prose and still resolves.
const INDENTED_CODE_INDENT = 4;
// CommonMark type 1 HTML blocks show their contents verbatim.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
// Type 6 and 7 blocks are literal too and run to the next blank line, not to a
// closing tag, so `<details>` holds Markdown only after a blank line. Type 7
// (any other complete tag alone on a line) cannot interrupt a paragraph.
const HTML_BLOCK_OPEN = /^ {0,3}<\/?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)/;
const HTML_ATTRIBUTE =
  "(?:\\s+[a-zA-Z_:][a-zA-Z0-9_.:-]*(?:\\s*=\\s*(?:[^\\s\"'=<>`]+|'[^']*'|\"[^\"]*\"))?)";
const HTML_TAG_ONLY_LINE = new RegExp(
  `^ {0,3}(?:<[a-zA-Z][a-zA-Z0-9-]*${HTML_ATTRIBUTE}*\\s*/?>|</[a-zA-Z][a-zA-Z0-9-]*\\s*>)\\s*$`,
);
const HTML_BLOCK_TAGS = new Set(
  `address article aside base basefont blockquote body caption center col colgroup
   dd details dialog dir div dl dt fieldset figcaption figure footer form frame
   frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu
   menuitem nav noframes ol optgroup option p param search section summary table
   tbody td tfoot th thead title tr track ul`.split(/\s+/),
);
// Lines that are blocks in their own right, so no paragraph is open after.
const BLOCK_LINE =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$|>|=+[ \t]*$)/;
const LINE_ENDINGS = /\r\n?/g;
// A scheme, a protocol-relative host, or a fragment: already absolute enough.
// `//` needs a host after it, so `///docs` stays a repository path.
const ABSOLUTE = /^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/\/[^/]|#)/;

const COMMENT_OPEN = "<!--";
const COMMENT_CLOSE = "-->";
const COMMENT_BLOCK_OPEN = /^ {0,3}<!--/;

/**
 * `line` with its commented spans blanked, and whether a comment block is still
 * open below it. Commented content renders as nothing, so it holds no fence,
 * block or code span. Lengths are preserved so offsets still line up with the
 * original.
 *
 * Only a comment that starts a line opens a block (CommonMark type 2), and only
 * that block runs on to the line holding `-->`. One written mid-sentence is
 * inline raw HTML: unclosed it is ordinary text, so a note that merely mentions
 * `<!--` may not hide the links below it, as in `_strip_comments` on the
 * backend and `stripCommentSpans` in the preview.
 */
function maskComments(line: string, inComment: boolean): [string, boolean] {
  if (inComment) {
    // The closing line belongs to the block, tail included.
    return [" ".repeat(line.length), !line.includes(COMMENT_CLOSE)];
  }
  if (COMMENT_BLOCK_OPEN.test(line)) {
    // `<!-->` and `<!--->` are complete comments, so the closer may overlap the
    // opener; searching past it would blank the rest of the file.
    return [" ".repeat(line.length), !line.includes(COMMENT_CLOSE)];
  }

  let out = "";
  let index = 0;
  // Scanned only once an opener turns up, so a line without one never pays for
  // it. Spans are ordered and disjoint and each opener sits at or past the one
  // before, so the search resumes where it stopped rather than restarting.
  let spans: CodeSpan[] | null = null;
  let cursor = 0;
  while (index < line.length) {
    const start = line.indexOf(COMMENT_OPEN, index);
    if (start < 0) {
      return [out + line.slice(index), false];
    }
    spans ??= codeSpans(line);
    while (cursor < spans.length && (spans[cursor]?.end ?? 0) <= start) {
      cursor += 1;
    }
    // A delimiter inside inline code is literal, not a comment opener.
    const span = spans[cursor];
    if (span !== undefined && span.start <= start) {
      out += line.slice(index, span.end);
      index = span.end;
      continue;
    }
    // `<!-->` and `<!--->` are complete comments, so the closer may overlap.
    const close = line.indexOf(COMMENT_CLOSE, start + 2);
    if (close < 0) {
      // Nothing closes it here, so the renderer shows it as ordinary text.
      return [out + line.slice(index), false];
    }
    out += line.slice(index, start);
    out += " ".repeat(close + COMMENT_CLOSE.length - start);
    index = close + COMMENT_CLOSE.length;
  }
  return [out, false];
}

/** True if `line` starts a CommonMark type 6 or type 7 HTML block. */
function opensHtmlBlock(line: string, afterParagraph: boolean): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  if (named && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase())) {
    return true;
  }
  return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);
}

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
    // A leading slash means the repository root, not the site root, so append
    // it to the base instead of replacing the base path.
    const resolved = new URL(
      trimmed.replace(LEADING_SLASHES, ""),
      base,
    ).toString();
    // `../` can climb out of the repository: leave those alone.
    return resolved.startsWith(base) ? resolved : target;
  } catch {
    return target;
  }
}

/** True when `index` is escaped by an odd run of backslashes. */
function isEscaped(line: string, index: number): boolean {
  let slashes = 0;
  while (line[index - 1 - slashes] === "\\") {
    slashes += 1;
  }
  return slashes % 2 === 1;
}

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
    // `\\!` is a literal mark, so what follows is a link, not an image.
    const image = bang === "!" && !isEscaped(line, offset);
    const resolved = absolute(unwrap(target), image);
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
  // Document ranges the renderer hides inside HTML comments.
  comments: CodeSpan[];
}

/**
 * Sorts lines into Markdown and code, masking the code so a code span cannot
 * pair across it. Offsets are preserved, so a span found in the mask sits at
 * the same place in the document.
 */
function classify(lines: string[]): Classified {
  const text: number[] = [];
  const definition = new Set<number>();
  const masked: string[] = [];
  let openFence: string | null = null;
  // Content column of the list item an open fence belongs to, 0 at document
  // level. A fence is scoped to its container, so the item's end closes it.
  let fenceColumn = 0;
  let inRawHtml = false;
  let inHtmlBlock = false;
  let inComment = false;
  let inCode = false;
  let afterParagraph = false;
  let quote: QuoteState = NO_QUOTE;
  let lists: ListState = EMPTY_LIST_STATE;
  const comments: CodeSpan[] = [];
  let offset = 0;

  // The line as list tracking sees it: blank wherever nothing renders. Taken
  // with the paragraph state from the line above, as the renderer would.
  const track = (structural: string, above: QuoteState): void => {
    lists = openLists(structural, lists, afterParagraph, above.quoted);
  };

  lines.forEach((original, index) => {
    const start = offset;
    offset += original.length + 1;
    // The quote state from the line above, which is the one list tracking
    // asks about. Only a plain line of text below rewrites it, so every block
    // that returns early leaves no quoted paragraph open behind it.
    const above = quote;
    quote = NO_QUOTE;
    // A fence inside a list item runs only to the end of that item, so a line
    // dedented out of the item closes both. Lazy continuation cannot reach
    // into a fence, so any content to the left of the item ends it.
    if (
      openFence !== null &&
      fenceColumn > 0 &&
      original.trim() &&
      indentWidth(original) < fenceColumn
    ) {
      openFence = null;
      fenceColumn = 0;
    }
    // A comment cannot open a fence and a fence hides a comment opener, so
    // resolve them in that order or a hidden delimiter opens a phantom fence.
    const fenceSource = inComment ? null : FENCE.exec(original);
    if (inRawHtml) {
      track("", above);
      inRawHtml = !RAW_HTML_CLOSE.test(original);
      masked.push(" ".repeat(original.length));
      afterParagraph = false;
      return;
    }
    if (inHtmlBlock) {
      track("", above);
      // Only a blank line ends a type 6 or 7 block, so nothing inside one is
      // a fence or a link.
      inHtmlBlock = !!original.trim();
      masked.push(" ".repeat(original.length));
      afterParagraph = false;
      return;
    }
    const fence = fenceSource;
    if (fence) {
      // A fence renders as nothing, but its indent still closes an item.
      track(original, above);
      const marker = fence[1] ?? "";
      if (openFence === null) {
        // A backtick fence's info string may not contain a backtick.
        openFence =
          marker[0] !== "`" || !(fence[2] ?? "").includes("`") ? marker : null;
        if (openFence === null) {
          text.push(index);
          masked.push(original);
          afterParagraph = true;
          return;
        }
        // Taken after the opener closed the items it is dedented out of, so
        // the fence belongs to the item it is really written inside.
        fenceColumn = lists.columns.at(-1) ?? 0;
      } else if (
        // A closer matches the opening character and carries nothing after it.
        marker[0] === openFence[0] &&
        marker.length >= openFence.length &&
        !NON_SPACE.test(fence[2] ?? "")
      ) {
        openFence = null;
        fenceColumn = 0;
      }
      masked.push(" ".repeat(original.length));
      afterParagraph = false;
      return;
    }
    if (openFence !== null) {
      track("", above);
      // Fenced content is literal, so a comment opener in it is not one.
      masked.push(" ".repeat(original.length));
      return;
    }
    // A block already open owns this line, so the line is its content rather
    // than a block written at the column it happens to start in.
    const hidden = inComment;
    // Only now, outside every fence, does a comment hide what follows.
    const [line, stillInComment] = maskComments(original, inComment);
    inComment = stillInComment;
    // Taken before an HTML opener is hidden: it renders as nothing, but its
    // indentation still closes a list item it sits to the left of. A comment
    // or a <pre> keeps only its column, since the text it hides is not
    // Markdown and must not open a list of its own.
    const opensRaw = RAW_HTML_OPEN.test(line);
    track(
      !hidden && (opensRaw || !line.trim()) ? hiddenStructure(original) : line,
      above,
    );
    for (let at = 0; at < line.length; at += 1) {
      if (line[at] === " " && original[at] !== " ") {
        const from = at;
        while (at < line.length && line[at] === " " && original[at] !== " ") {
          at += 1;
        }
        comments.push({ start: start + from, end: start + at, content: "" });
      }
    }
    if (opensRaw) {
      inRawHtml = !RAW_HTML_CLOSE.test(line.replace(RAW_HTML_OPEN, ""));
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    if (line.trim() && opensHtmlBlock(line, afterParagraph)) {
      inHtmlBlock = true;
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    const blank = !line.trim();
    // Measured from the innermost open item's content column, not the margin:
    // four spaces under "- Details:" is a paragraph, not a code block.
    const column = lists.columns.at(-1) ?? 0;
    const indented = indentWidth(line) - column >= INDENTED_CODE_INDENT;
    // Indented code starts only outside a paragraph and runs to a dedent.
    if (inCode) {
      inCode = blank || indented;
    } else {
      inCode = !afterParagraph && !blank && indented;
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
    quote = quoteState(line, above.inQuote);
  });

  return { text, masked: masked.join("\n"), definition, comments };
}

/** Absolute repository URLs for every relative link and image in `markdown`. */
export function resolveChangelogLinks(markdown: string): string {
  // The desktop updater body arrives with CRLF, which would hide fences.
  const lines = markdown.replace(LINE_ENDINGS, "\n").split("\n");
  const { text, masked, definition, comments } = classify(lines);
  // Scanned over the whole document, so a span may cross a line break.
  // Commented ranges join them: the renderer shows neither, so a link in one
  // is not followable and rewriting it would only mutate hidden text.
  const spans = [...codeSpans(masked), ...comments].sort(
    (a, b) => a.start - b.start,
  );

  // Offset of each line in the document, to place matches inside it.
  const offsets: number[] = [];
  let cursor = 0;
  for (const line of lines) {
    offsets.push(cursor);
    cursor += line.length + 1;
  }

  // Only images resolve against the raw host, so collect the image labels
  // before rewriting any definition.
  const imageLabels = new Set<string>();
  for (const index of text) {
    const line = lines[index] ?? "";
    IMAGE_REFERENCE.lastIndex = 0;
    for (
      let match = IMAGE_REFERENCE.exec(line);
      match !== null;
      match = IMAGE_REFERENCE.exec(line)
    ) {
      // An escaped mark makes it a link, so its definition stays a page URL.
      if (
        insideSpan(spans, (offsets[index] ?? 0) + match.index) ||
        isEscaped(line, match.index)
      ) {
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
