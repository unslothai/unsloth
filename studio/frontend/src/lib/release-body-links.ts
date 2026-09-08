// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A relative link in a release body means "somewhere in the Unsloth
 * repository", but inside Unsloth it would resolve against Unsloth's own origin.
 * Rewriting to absolute repository URLs makes them behave the way GitHub
 * renders the release page.
 */

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

// Inline `](dest)` plus the `[label]: dest` reference form. The destination is
// either <bracketed> or runs to whitespace or the closing paren.
const NESTED_LABEL = String.raw`((?:[^[\]\\]|\\.|\[(?:[^[\]\\]|\\.)*\])*)`;
// Only ASCII punctuation is escapable, so the backslash in `a\ b.md` is an
// ordinary character of the destination and the space still ends it.
const ESCAPABLE = String.raw`[!-/:-@[-\`{-~]`;
const DESTINATION_CHAR = String.raw`\\${ESCAPABLE}|[^\s()]`;
// A destination may hold balanced parentheses, and a path may nest them, so
// `[x](((draft)).md)` points at `((draft)).md`. An expression cannot count, so
// pairs are unrolled to the depth cmark stops at, which is what GitHub renders.
const MAX_DESTINATION_NESTING = 32;

/** A balanced parenthesised run nested up to `depth` levels deep. */
function nestedParens(depth: number): string {
  let group = String.raw`\((?:${DESTINATION_CHAR})*\)`;
  for (let left = depth - 1; left > 0; left -= 1) {
    group = String.raw`\((?:${DESTINATION_CHAR}|${group})*\)`;
  }
  return group;
}

const BALANCED_DESTINATION = String.raw`(?:${DESTINATION_CHAR}|${nestedParens(MAX_DESTINATION_NESTING)})*`;
const PLAIN_DESTINATION = String.raw`(?:${DESTINATION_CHAR})*`;
// A balanced pair counts only while a `)` or a title still closes the link
// after it, or swallowing it would invent a link across lines.
const CLOSES_LINK = String.raw`(?=[ \t]*[)'"])`;
// A destination that runs out of line has its closer below it, the line being
// only part of the link. One stopping short of a closer is no destination at all,
// so `[x](a b.md)` and `[x](a(b.md)` stay plain text and keep the paths they name.
const CLOSES_OR_ENDS_LINE = String.raw`(?=[ \t]*(?:[)'"]|$))`;
const INLINE_TARGET = new RegExp(
  String.raw`(!?)\[${NESTED_LABEL}\]\(\s*(<[^<>\n]*>|${BALANCED_DESTINATION}${CLOSES_LINK}|${PLAIN_DESTINATION}${CLOSES_OR_ENDS_LINE})`,
  "g",
);
const REFERENCE_TARGET = /^( {0,3}\[((?:[^[\]\\]|\\.)*)\]:\s*)(<[^<>\n]*>|\S+)/;
// `![alt][label]`, `![label][]` and `![label]`: a definition they point at
// has to resolve to the raw file, not to its page on GitHub.
const IMAGE_REFERENCE =
  /!\[((?:[^[\]\\]|\\.)*)\](?:\[((?:[^[\]\\]|\\.)*)\]|(?!\())/g;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
// Four columns past the container start indented code, unless a paragraph is
// open. Inside a list item that is measured from the item's content column, so a
// link indented under a bullet is prose and still resolves.
const INDENTED_CODE_INDENT = 4;
// CommonMark type 1 HTML blocks show their contents verbatim.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
// Type 6 and 7 blocks are literal too and run to the next blank line, not to a
// closing tag, so `<details>` holds Markdown only after a blank line. Type 7 (any
// other complete tag alone on a line) cannot interrupt a paragraph.
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
// A definition is a block of its own but may not interrupt a paragraph, so it
// ends the one above only when there is none to continue. It opens none either,
// or consecutive definitions could never start (spec 0.31.2 section 4.7). Same
// rule as `_LINK_DEFINITION` in the backend's `after_paragraph`.
const LINK_DEFINITION = /^ {0,3}\[(?:[^[\]\\]|\\.)+\]:/;
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
 * block or code span. Lengths are preserved so offsets still line up.
 *
 * Only a comment that starts a line opens a block (CommonMark type 2), and only
 * that runs on to the line holding `-->`, tail included. One written mid-sentence
 * is inline raw HTML belonging to its paragraph, so its `-->` may arrive on a
 * later line and only the text up to it is hidden. `closesBelow` says one does;
 * without it the opener is ordinary text, so a note merely mentioning `<!--` must
 * not hide the links below it.
 *
 * "Starts a line" is read inside the container, so `blockOpen` comes from the
 * item's content rather than the raw line.
 */
function maskComments(
  line: string,
  inComment: boolean,
  runOn: boolean,
  closesBelow: boolean,
  blockOpen: boolean,
): [string, boolean, boolean] {
  if (inComment) {
    // The closing line belongs to the block, tail included.
    return [" ".repeat(line.length), !line.includes(COMMENT_CLOSE), false];
  }
  if (runOn) {
    const closed = line.indexOf(COMMENT_CLOSE);
    if (closed < 0) {
      return [" ".repeat(line.length), false, true];
    }
    // Only up to the closer: the tail is the paragraph's own text again.
    const resumed = closed + COMMENT_CLOSE.length;
    return maskInline(line, resumed, closesBelow);
  }
  if (blockOpen) {
    // `<!-->` and `<!--->` are complete comments, so the closer may overlap the
    // opener; searching past it would blank the rest of the file.
    return [" ".repeat(line.length), !line.includes(COMMENT_CLOSE), false];
  }
  return maskInline(line, 0, closesBelow);
}

/** `maskComments` from `from`, where no comment block is open. */
function maskInline(
  line: string,
  from: number,
  closesBelow: boolean,
): [string, boolean, boolean] {
  let out = " ".repeat(from);
  let index = from;
  // Scanned only once an opener turns up. Spans are ordered and disjoint and each
  // opener sits at or past the last, so the search resumes rather than restarts.
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
      if (closesBelow) {
        // The paragraph carries the comment on, so the line from the opener is
        // inside it, and so is the line below.
        return [
          out + line.slice(index, start) + " ".repeat(line.length - start),
          false,
          true,
        ];
      }
      // Nothing closes it at all, so the renderer shows it as ordinary text.
      return [out + line.slice(index), false, false];
    }
    out += line.slice(index, start);
    out += " ".repeat(close + COMMENT_CLOSE.length - start);
    index = close + COMMENT_CLOSE.length;
  }
  return [out, false, false];
}

/**
 * Whether `line` is written outside the container an open block belongs to. A
 * fence and an HTML block hold no lazy continuation line, so content left of the
 * item, or outside the quote, ends the block with its container. A raw block or
 * comment inside a list item ends on a blank line too: the item takes the break,
 * so what follows is a block of the item's own.
 */
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
// `\(` in a destination is a literal paren. Only ASCII punctuation is escapable,
// so the backslash in `docs\alpha.md` is part of the path and has to survive.
const ESCAPE = new RegExp(String.raw`\\(${ESCAPABLE})`, "g");
// A URL parser reads a backslash as a path separator, so `docs\a.md` would
// resolve to `docs/a.md`. Encode it first, the way a renderer normalises it.
const BACKSLASH = /\\/g;
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
      trimmed.replace(LEADING_SLASHES, "").replace(BACKSLASH, "%5C"),
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
 * Sorts lines into Markdown and code, masking the code so a span cannot pair
 * across it. Offsets are preserved, so a mask span sits where it does in the doc.
 */
function classify(lines: string[]): Classified {
  const text: number[] = [];
  const definition = new Set<number>();
  const masked: string[] = [];
  let openFence: string | null = null;
  let inRawHtml = false;
  let inHtmlBlock = false;
  // Where the open block was written: the content column of the item it belongs
  // to, 0 at document level, plus the blockquotes it sits inside. Only one is ever
  // open, and none holds a lazy continuation line, so a line left of the item or
  // outside the quote ends the block with its container.
  let blockColumn = 0;
  let blockQuotes = 0;
  let inComment = false;
  // True while an inline comment opened above runs on into this line, carried by
  // the paragraph holding it.
  let runOn = false;
  const closesBelow = commentClosesBelow(lines);
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
  // Where a block just opened sits, read after the opener closed the items it
  // is dedented out of, so it belongs to the container it is really in.
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
    // The quote state from the line above, which is what list tracking asks
    // about. Only plain text below rewrites it, so every block returning early
    // leaves no quoted paragraph open behind it.
    const above = quote;
    quote = NO_QUOTE;
    // A fence, comment or HTML block runs only to the end of the container it was
    // written in, so a line dedented out of that item or outside that quote
    // closes both.
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
    // Read from the container the line is written in, so a fence three columns
    // past a nested bullet or behind a quote marker still opens one. A block
    // already open keeps only its own quote stripped, or a deeper marker in it
    // would read as a closer.
    const container = containerContent(
      original,
      lists,
      inBlock ? blockQuotes : quotes,
    );
    // A comment cannot open a fence and a fence hides a comment opener, so resolve
    // them in that order or a hidden delimiter opens a phantom fence. An opener is
    // read past a marker on the same line too, since a fence written as an item's
    // first content opens inside it. Only an opener: fenced content is literal and
    // a closer carries no marker.
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
      // Only a blank line ends a type 6 or 7 block, so nothing inside one is a
      // fence or a link. A bare quote marker holds nothing, so it ends one too.
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
        startBlock(quotes);
      } else if (
        // A closer matches the opening character and carries nothing after it.
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
    // A block already open owns this line, so it is content rather than a block
    // written at the column it happens to start in.
    const hidden = inComment;
    const carried = runOn;
    // A comment is an HTML block too, so one written as a list item's first
    // content opens inside that item exactly as a fence does: read past a marker
    // on the same line and from its container's column, not the line's margin.
    const opensComment =
      !(hidden || carried) &&
      COMMENT_BLOCK_OPEN.test(itemContent(container, afterParagraph));
    // Only now, outside every fence, does a comment hide what follows.
    const [line, stillInComment, stillRunOn] = maskComments(
      original,
      inComment,
      runOn,
      closesBelow[index + 1] ?? false,
      opensComment,
    );
    inComment = stillInComment;
    runOn = stillRunOn;
    // A line an inline comment runs on into is still a line of the paragraph
    // that carries it: only its text is hidden, never its block structure.
    const structure = carried ? original : line;
    // The same container reading as above, now the comments are masked. A comment
    // blanks its own line, so that line is read as written: the block renders as
    // nothing, but the item it is the content of still opens.
    const source = opensComment ? original : line;
    const visible = containerContent(source, lists, quotes);
    // An HTML block written as a list item's first content opens inside that item,
    // as a fence does, so an opener is read past a marker on the same line. The
    // marker survives into the structural line, so its item is still tracked.
    const content = itemContent(visible, afterParagraph);
    const marker =
      content === visible
        ? ""
        : source.slice(0, source.length - content.length);
    // Taken before an HTML opener is hidden: it renders as nothing, but its indent
    // still closes a list item it sits left of. A comment or a <pre> keeps only its
    // column and marker, since the text it hides is not Markdown and opens no list.
    const opensRaw = !carried && RAW_HTML_OPEN.test(content);
    track(
      !(hidden || carried) && (opensRaw || !line.trim())
        ? hiddenStructure(original, marker)
        : structure,
      above,
    );
    // Read once the opener has closed the items it is dedented out of, so the
    // comment block belongs to the item it is really written inside.
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
    // Measured from the innermost open item's content column, not the margin:
    // four spaces under "- Details:" is a paragraph, not a code block.
    const column = lists.columns.at(-1) ?? 0;
    const indented = indentWidth(structure) - column >= INDENTED_CODE_INDENT;
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
    afterParagraph =
      !blank &&
      !BLOCK_LINE.test(structure) &&
      (afterParagraph || !LINK_DEFINITION.test(structure));
    quote = quoteState(structure, above.inQuote);
  });

  return { text, masked: masked.join("\n"), definition, comments };
}

/** Absolute repository URLs for every relative link and image in `markdown`. */
export function resolveReleaseBodyLinks(markdown: string): string {
  // A release body edited on GitHub arrives with CRLF, which would hide fences.
  const lines = markdown.replace(LINE_ENDINGS, "\n").split("\n");
  const { text, masked, definition, comments } = classify(lines);
  // Scanned over the whole document, so a span may cross a line break. Commented
  // ranges join them: the renderer shows neither, so a link in one is not
  // followable and rewriting it would only mutate hidden text.
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
