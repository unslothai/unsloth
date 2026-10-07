// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Relative links mean the Unsloth repository, so they are rewritten as GitHub would render them. */

import {
  type CodeSpan,
  codeSpans,
  insideSpan,
} from "@/lib/markdown-code-spans";
import { commentClosesBelow } from "@/lib/markdown-inline-comments";
import {
  EMPTY_LIST_STATE,
  type ListState,
  NO_QUOTE,
  type QuoteState,
  containerContent,
  hiddenStructure,
  indentWidth,
  itemContent,
  openLists,
  quoteDepth,
  quoteState,
} from "@/lib/markdown-list-columns";

const LINK_BASE = "https://github.com/unslothai/unsloth/blob/main/";
const IMAGE_BASE = "https://raw.githubusercontent.com/unslothai/unsloth/main/";

const NESTED_LABEL = String.raw`((?:[^[\]\\]|\\.|\[(?:[^[\]\\]|\\.)*\])*)`;
// Only ASCII punctuation is escapable, so `a\ b.md` keeps its backslash.
const ESCAPABLE = String.raw`[!-/:-@[-\`{-~]`;
const DESTINATION_CHAR = String.raw`\\${ESCAPABLE}|[^\s()]`;
// Balanced parens are unrolled to cmark's nesting limit, which is what GitHub renders.
const MAX_DESTINATION_NESTING = 32;

function nestedParens(depth: number): string {
  let group = String.raw`\((?:${DESTINATION_CHAR})*\)`;
  for (let left = depth - 1; left > 0; left -= 1) {
    group = String.raw`\((?:${DESTINATION_CHAR}|${group})*\)`;
  }
  return group;
}

const BALANCED_DESTINATION = String.raw`(?:${DESTINATION_CHAR}|${nestedParens(MAX_DESTINATION_NESTING)})*`;
const PLAIN_DESTINATION = String.raw`(?:${DESTINATION_CHAR})*`;
// A balanced pair counts only while a `)` or title still closes the link after it.
const CLOSES_LINK = String.raw`(?=[ \t]*[)'"])`;
const CLOSES_OR_ENDS_LINE = String.raw`(?=[ \t]*(?:[)'"]|$))`;
const INLINE_TARGET = new RegExp(
  String.raw`(!?)\[${NESTED_LABEL}\]\(\s*(<[^<>\n]*>|${BALANCED_DESTINATION}${CLOSES_LINK}|${PLAIN_DESTINATION}${CLOSES_OR_ENDS_LINE})`,
  "g",
);
const REFERENCE_TARGET = /^( {0,3}\[((?:[^[\]\\]|\\.)*)\]:\s*)(<[^<>\n]*>|\S+)/;
const IMAGE_REFERENCE =
  /!\[((?:[^[\]\\]|\\.)*)\](?:\[((?:[^[\]\\]|\\.)*)\]|(?!\())/g;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
// Inside a list item, measured from the item's content column.
const INDENTED_CODE_INDENT = 4;
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
// Type 6 and 7 blocks run to the next blank line; type 7 cannot interrupt a paragraph.
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
const BLOCK_LINE =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$|>|=+[ \t]*$)/;
// A definition cannot interrupt a paragraph and opens none (spec 0.31.2 4.7). Same rule as
// the backend's `_LINK_DEFINITION`.
const LINK_DEFINITION = /^ {0,3}\[(?:[^[\]\\]|\\.)+\]:/;
const LINE_ENDINGS = /\r\n?/g;
// `//` needs a host after it, so `///docs` stays a repository path.
const ABSOLUTE = /^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|\/\/[^/]|#)/;

const COMMENT_OPEN = "<!--";
const COMMENT_CLOSE = "-->";
const COMMENT_BLOCK_OPEN = /^ {0,3}<!--/;

/**
 * Blanks commented spans, preserving offsets. Only a line-start comment opens a block; a mid-line
 * one is inline HTML whose `-->` may arrive later in the paragraph (`closesBelow`).
 */
function maskComments(
  line: string,
  inComment: boolean,
  runOn: boolean,
  closesBelow: boolean,
  blockOpen: boolean,
): [string, boolean, boolean] {
  if (inComment) {
    return [" ".repeat(line.length), !line.includes(COMMENT_CLOSE), false];
  }
  if (runOn) {
    const closed = line.indexOf(COMMENT_CLOSE);
    if (closed < 0) {
      return [" ".repeat(line.length), false, true];
    }
    const resumed = closed + COMMENT_CLOSE.length;
    return maskInline(line, resumed, closesBelow);
  }
  if (blockOpen) {
    // `<!-->` and `<!--->` are complete comments; searching past the opener would blank the file.
    return [" ".repeat(line.length), !line.includes(COMMENT_CLOSE), false];
  }
  return maskInline(line, 0, closesBelow);
}

function maskInline(
  line: string,
  from: number,
  closesBelow: boolean,
): [string, boolean, boolean] {
  let out = " ".repeat(from);
  let index = from;
  // Spans are ordered and disjoint, so the search resumes rather than restarts.
  let spans: CodeSpan[] | null = null;
  let cursor = 0;
  while (index < line.length) {
    const start = line.indexOf(COMMENT_OPEN, index);
    if (start < 0) {
      return [out + line.slice(index), false, false];
    }
    spans ??= codeSpans(line);
    while (cursor < spans.length && (spans[cursor]?.end ?? 0) <= start) {
      cursor += 1;
    }
    const span = spans[cursor];
    if (span !== undefined && span.start <= start) {
      out += line.slice(index, span.end);
      index = span.end;
      continue;
    }
    // `<!-->` and `<!--->` are complete comments, so the closer may overlap.
    const close = line.indexOf(COMMENT_CLOSE, start + 2);
    if (close < 0) {
      if (closesBelow) {
        return [
          out + line.slice(index, start) + " ".repeat(line.length - start),
          false,
          true,
        ];
      }
      return [out + line.slice(index), false, false];
    }
    out += line.slice(index, start);
    out += " ".repeat(close + COMMENT_CLOSE.length - start);
    index = close + COMMENT_CLOSE.length;
  }
  return [out, false, false];
}

/** Fences and HTML blocks hold no lazy lines, so leaving the container ends them. */
function leavesContainer(
  line: string,
  quotes: number,
  column: number,
  blockQuotes: number,
  rawInItem: boolean,
): boolean {
  if (quotes < blockQuotes) {
    return true;
  }
  if (!line.trim()) {
    return rawInItem;
  }
  return column > 0 && indentWidth(line) < column;
}

function opensHtmlBlock(line: string, afterParagraph: boolean): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  if (named && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase())) {
    return true;
  }
  return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);
}

function label(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

const NEEDS_BRACKETS = /[()\s]/;
// Only ASCII punctuation is escapable, so `docs\alpha.md` keeps its backslash.
const ESCAPE = new RegExp(String.raw`\\(${ESCAPABLE})`, "g");
// A URL parser treats a backslash as a separator, so encode it first.
const BACKSLASH = /\\/g;
const NON_SPACE = /[^ \t]/;
const LEADING_SLASHES = /^\/+/;

function absolute(target: string, image: boolean): string {
  const base = image ? IMAGE_BASE : LINK_BASE;
  const trimmed = target.trim().replace(ESCAPE, "$1");
  if (!trimmed || ABSOLUTE.test(trimmed)) {
    return target;
  }
  try {
    // A leading slash means the repository root, so append to the base path.
    const resolved = new URL(
      trimmed.replace(LEADING_SLASHES, "").replace(BACKSLASH, "%5C"),
      base,
    ).toString();
    // `../` can climb out of the repository: leave those alone.
    return resolved.startsWith(base) ? resolved : target;
  } catch {
    return target;
  }
}

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

function wrap(resolved: string, original: string): string {
  const bracketed = original.startsWith("<") && original.endsWith(">");
  return bracketed || (resolved !== original && NEEDS_BRACKETS.test(resolved))
    ? `<${resolved}>`
    : resolved;
}

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
    const opener = offset + (bang ? 1 : 0);
    if (insideSpan(spans, base + offset) || isEscaped(line, opener)) {
      return match;
    }
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
  text: number[];
  // Code blanked out, for span scanning.
  masked: string;
  definition: Set<number>;
  comments: CodeSpan[];
}

/** Offsets are preserved, so a mask span sits where it does in the document. */
function classify(lines: string[]): Classified {
  const text: number[] = [];
  const definition = new Set<number>();
  const masked: string[] = [];
  let openFence: string | null = null;
  let inRawHtml = false;
  let inHtmlBlock = false;
  // Container of the open block (item column plus quotes); a line outside it ends the block.
  let blockColumn = 0;
  let blockQuotes = 0;
  let inComment = false;
  let runOn = false;
  const closesBelow = commentClosesBelow(lines);
  let inCode = false;
  let afterParagraph = false;
  let quote: QuoteState = NO_QUOTE;
  let lists: ListState = EMPTY_LIST_STATE;
  const comments: CodeSpan[] = [];
  let offset = 0;

  const track = (structural: string, above: QuoteState): void => {
    lists = openLists(structural, lists, afterParagraph, above.quoted);
  };
  const startBlock = (quotes: number): void => {
    blockColumn = lists.columns.at(-1) ?? 0;
    blockQuotes = quotes;
  };
  const endBlock = (): void => {
    blockColumn = 0;
    blockQuotes = 0;
  };

  lines.forEach((original, index) => {
    const start = offset;
    offset += original.length + 1;
    const above = quote;
    quote = NO_QUOTE;
    const quotes = quoteDepth(original);
    let inBlock = openFence !== null || inRawHtml || inHtmlBlock || inComment;
    if (
      inBlock &&
      leavesContainer(
        original,
        quotes,
        blockColumn,
        blockQuotes,
        (inRawHtml || inComment) && blockColumn > 0 && blockQuotes === 0,
      )
    ) {
      openFence = null;
      inRawHtml = false;
      inHtmlBlock = false;
      inComment = false;
      endBlock();
      inBlock = false;
    }
    // Read from the line's container, so a fence under a nested bullet or quote still opens.
    const container = containerContent(
      original,
      lists,
      inBlock ? blockQuotes : quotes,
    );
    // Resolve comments before fences, or a hidden delimiter opens a phantom fence.
    const fenceSource = inComment
      ? null
      : FENCE.exec(
          openFence === null
            ? itemContent(container, afterParagraph)
            : container,
        );
    if (inRawHtml) {
      track("", above);
      inRawHtml = !RAW_HTML_CLOSE.test(container);
      if (!inRawHtml) {
        endBlock();
      }
      masked.push(" ".repeat(original.length));
      afterParagraph = false;
      return;
    }
    if (inHtmlBlock) {
      track("", above);
      // Only a blank line ends a type 6 or 7 block; a bare quote marker counts as one.
      inHtmlBlock = !!container.trim();
      if (!inHtmlBlock) {
        endBlock();
      }
      masked.push(" ".repeat(original.length));
      afterParagraph = false;
      return;
    }
    const fence = fenceSource;
    if (fence) {
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
        startBlock(quotes);
      } else if (
        marker[0] === openFence[0] &&
        marker.length >= openFence.length &&
        !NON_SPACE.test(fence[2] ?? "")
      ) {
        openFence = null;
        endBlock();
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
    const hidden = inComment;
    const carried = runOn;
    // A comment written as an item's first content opens inside that item, like a fence.
    const opensComment =
      !(hidden || carried) &&
      COMMENT_BLOCK_OPEN.test(itemContent(container, afterParagraph));
    const [line, stillInComment, stillRunOn] = maskComments(
      original,
      inComment,
      runOn,
      closesBelow[index + 1] ?? false,
      opensComment,
    );
    inComment = stillInComment;
    runOn = stillRunOn;
    const structure = carried ? original : line;
    const source = opensComment ? original : line;
    const visible = containerContent(source, lists, quotes);
    const content = itemContent(visible, afterParagraph);
    const marker =
      content === visible
        ? ""
        : source.slice(0, source.length - content.length);
    // Taken before the opener is hidden: its indent still closes a list item it sits left of.
    const opensRaw = !carried && RAW_HTML_OPEN.test(content);
    track(
      !(hidden || carried) && (opensRaw || !line.trim())
        ? hiddenStructure(original, marker)
        : structure,
      above,
    );
    if (inComment !== hidden) {
      if (inComment) {
        startBlock(quotes);
      } else {
        endBlock();
      }
    }
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
      inRawHtml = !RAW_HTML_CLOSE.test(content.replace(RAW_HTML_OPEN, ""));
      if (inRawHtml) {
        startBlock(quotes);
      }
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    if (!carried && content.trim() && opensHtmlBlock(content, afterParagraph)) {
      inHtmlBlock = true;
      startBlock(quotes);
      masked.push(" ".repeat(line.length));
      afterParagraph = false;
      return;
    }
    const blank = !structure.trim();
    // Measured from the innermost item's content column: four spaces under a bullet is a paragraph.
    const column = lists.columns.at(-1) ?? 0;
    const indented = indentWidth(structure) - column >= INDENTED_CODE_INDENT;
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
    if (!afterParagraph) {
      definition.add(index);
    }
    text.push(index);
    masked.push(line);
    afterParagraph =
      !blank &&
      !BLOCK_LINE.test(structure) &&
      (afterParagraph || !LINK_DEFINITION.test(structure));
    quote = quoteState(structure, above.inQuote);
  });

  return { text, masked: masked.join("\n"), definition, comments };
}

export function resolveReleaseBodyLinks(markdown: string): string {
  // A release body edited on GitHub arrives with CRLF, which would hide fences.
  const lines = markdown.replace(LINE_ENDINGS, "\n").split("\n");
  const { text, masked, definition, comments } = classify(lines);
  // Comment ranges join code spans: hidden links are not followable, so they are not rewritten.
  const spans = [...codeSpans(masked), ...comments].sort(
    (a, b) => a.start - b.start,
  );

  const offsets: number[] = [];
  let cursor = 0;
  for (const line of lines) {
    offsets.push(cursor);
    cursor += line.length + 1;
  }

  // Collect image labels first: only images resolve against the raw host.
  const imageLabels = new Set<string>();
  for (const index of text) {
    const line = lines[index] ?? "";
    IMAGE_REFERENCE.lastIndex = 0;
    for (
      let match = IMAGE_REFERENCE.exec(line);
      match !== null;
      match = IMAGE_REFERENCE.exec(line)
    ) {
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
