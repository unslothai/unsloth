// Adapted from LibreChat's latex.ts
// https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts
//
// Two jobs, in order:
//   1. Convert LaTeX bracket delimiters (`\[...\]`, `\(...\)`) into the dollar
//      forms remark-math understands (`$$...$$`, `$...$`). remark-math only
//      tokenizes dollar delimiters, so models that emit `\[...\]` / `\(...\)`
//      would otherwise render as literal text.
//   2. Escape currency dollar signs so they are not misinterpreted as LaTeX
//      math delimiters when singleDollarTextMath is enabled.

/**
 * Matches a single $ followed by a number pattern (currency), e.g.:
 *   $5, $1,000, $5.99, $100K, $3.5M
 *
 * Does NOT match:
 *   $$ (display math), \$ (already escaped), $\alpha (LaTeX command)
 */
const CURRENCY_REGEX =
  /(?<![\\$])\$(?!\$)(?=\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?(?:\s|$|[^a-zA-Z\d]))/g;

/**
 * Find code-block regions (``` ... ```, ~~~ ... ~~~, and ` ... `) to skip.
 * Returns a sorted array of [start, end] index pairs.
 */
function findCodeBlockRegions(content: string): Array<[number, number]> {
  const regions: Array<[number, number]> = [];

  // Fenced code blocks: ```...``` and ~~~...~~~ (both are code in GFM)
  const fencedRe = /```[\s\S]*?```|~~~[\s\S]*?~~~/g;
  let match: RegExpExecArray | null;
  while ((match = fencedRe.exec(content)) !== null) {
    regions.push([match.index, match.index + match[0].length]);
  }

  // Inline code: `...` (skip spans inside fenced blocks, filtered below)
  const inlineRe = /`[^`\n]+`/g;
  while ((match = inlineRe.exec(content)) !== null) {
    const start = match.index;
    const end = start + match[0].length;
    let inside = false;
    for (const [rs, re] of regions) {
      if (start >= rs && end <= re) {
        inside = true;
        break;
      }
    }
    if (!inside) {
      regions.push([start, end]);
    }
  }

  // Sort by start position for binary search.
  regions.sort((a, b) => a[0] - b[0]);
  return regions;
}

/**
 * Match an inline link/image `[text](DEST)`, capturing the destination as group 1
 * with the `d` flag so its span is read straight from `match.indices` (the text
 * can contain an escaped `\](`, so a string search for the separator is unsafe).
 * The text disallows unescaped `]`; the destination allows escapes and one level
 * of balanced parens.
 */
const LINK_DEST_RE =
  /!?\[(?:\\.|[^\]\\])*?\]\(((?:\\.|[^()\\]|\([^()]*\))*)\)/gd;

/**
 * Find the destination spans of inline links/images, so a `\(...\)` written with
 * escaped parens inside a URL isn't rewritten as math (which would break the
 * link). Only the destination is returned, not the link text, so math in the
 * visible text still converts. Sorted, non-overlapping (matches are disjoint).
 */
function findLinkDestinationRegions(content: string): Array<[number, number]> {
  if (!content.includes("](")) return [];
  const regions: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  LINK_DEST_RE.lastIndex = 0;
  while ((match = LINK_DEST_RE.exec(content)) !== null) {
    // `indices` is present (the `d` flag); group 1 spans the destination.
    regions.push(match.indices![1]);
  }
  return regions;
}

/**
 * Binary search to check if a position falls inside any region. Regions must be
 * sorted by start and non-overlapping.
 */
function isInRegion(
  position: number,
  regions: Array<[number, number]>,
): boolean {
  let lo = 0;
  let hi = regions.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const [start, end] = regions[mid];
    if (position < start) {
      hi = mid - 1;
    } else if (position >= end) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

/** A whitespace-free token that looks purely like currency, e.g. `5`, `1,000`, `5.99`, `100K`, `3.5M`. */
const CURRENCY_BODY_RE = /^\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?$/;

/** Body characters that almost always indicate real LaTeX. */
const LATEX_CHAR_RE = /[\\^_{}]/;

/**
 * Operators that strongly suggest math. Omits `^` and `_` since
 * `LATEX_CHAR_RE` short-circuits on those before this regex runs.
 */
const MATH_OP_RE = /[=+\-<>/*]/;

/**
 * Trailing chars stripped before the currency check: prose punctuation
 * plus `-` and `/` from compact ranges like `$5-$10`; without them the
 * body `5-` or `5/` would slip through the single-token math shortcut.
 */
const TRAIL_PUNCT_RE = /[.,;:!?\-/]+$/;

/**
 * A standalone single-letter variable, not part of a longer word, so
 * prose like "5 to attend" isn't misread as math with variable `t`.
 */
const LONE_LETTER_RE = /(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])/;

/**
 * Numeric or single-letter operands joined by math operators (optional
 * whitespace): `2 + 2`, `100 < 200`, `1,000 - 500`, `x + y`. Recognises
 * numeric-only expressions like `$2 + 2$` without a lone variable token.
 */
const SIMPLE_MATH_RE =
  /^(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z])(?:\s*[=+\-<>/*]\s*(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z]))+$/;

/**
 * True if the substring between two `$` delimiters looks like LaTeX
 * rather than prose between two currency tokens.
 *
 * Rule of thumb:
 *   - `$30^\circ$`  -> math (LaTeX chars)
 *   - `$x$`         -> math (single non-currency token)
 *   - `$90 - x$`    -> math (math op + lone variable)
 *   - `$5 to $10`   -> NOT math (multi-token prose, no math op)
 *   - `$5, $10`     -> NOT math (currency-like token + trailing punct)
 *   - `$1,000$`     -> NOT math (single currency-like token)
 */
function looksLikeMathBody(body: string): boolean {
  if (LATEX_CHAR_RE.test(body)) return true;
  const trimmed = body.trim().replace(TRAIL_PUNCT_RE, "");
  if (!trimmed) return false;
  if (CURRENCY_BODY_RE.test(trimmed)) return false;
  // Numeric-only operator forms: `2 + 2`, `100 < 200`, `1,000 - 500`.
  // Recognised without requiring a lone-variable letter.
  if (SIMPLE_MATH_RE.test(trimmed)) return true;
  if (!/\s/.test(trimmed)) return true;
  if (!MATH_OP_RE.test(trimmed)) return false;
  return LONE_LETTER_RE.test(trimmed);
}

/**
 * True if the `$` at `offset` opens a balanced inline math span (`$...$`)
 * on the same line. The closer must be unescaped, not part of `$$`, and
 * within 200 chars. The body must look like LaTeX so we don't pair two
 * currency tokens on a line (e.g. "$5 to $10"). Bold-wrapped spans
 * (`**$X$**`, `__$X$__`) are always math: LLMs use that for "bold math"
 * and the heuristic would otherwise reject prose-shaped bodies like "90 - x".
 */
function hasInlineMathCloser(
  content: string,
  offset: number,
  mathRegions: Array<[number, number]>,
): boolean {
  const MAX_SPAN = 200;
  const limit = Math.min(content.length, offset + 1 + MAX_SPAN);
  for (let i = offset + 1; i < limit; i++) {
    const c = content[i];
    if (c === "\n") return false;
    if (c !== "$") continue;
    if (content[i - 1] === "\\") continue;
    // A `$` opening a generated span (from `\(...\)`) is not a currency closer;
    // pairing with it would swallow the price into math (`$5 + x \(y\)`).
    if (isInRegion(i, mathRegions)) return false;
    if (content[i + 1] === "$") {
      i++;
      continue;
    }
    // A `$` followed by a digit is more likely another currency token than
    // the closer. Keep scanning so prose like `$5 + a $10 add-on` doesn't
    // pair the two currency markers as a math span.
    if (/\d/.test(content[i + 1] ?? "")) {
      continue;
    }
    if (offset >= 2) {
      const op = content[offset - 1];
      if (
        (op === "*" || op === "_") &&
        content[offset - 2] === op &&
        content[i + 1] === op &&
        content[i + 2] === op
      ) {
        return true;
      }
    }
    return looksLikeMathBody(content.slice(offset + 1, i));
  }
  return false;
}

/**
 * Matches a `\[...\]` (display) or `\(...\)` (inline) LaTeX span. Non-greedy so
 * the first closer wins; dotall so display spans can wrap lines. `(?<!\\)` on
 * each opener leaves an escaped literal `\\[` (a real backslash then bracket)
 * alone. The body is length-capped so an unclosed opener can't scan to
 * end-of-input: without the cap, many unclosed `\(`/`\[` make matching O(n^2),
 * and this runs per animation frame while streaming. Real spans are far shorter
 * than the cap; a longer one just stays literal.
 */
const LATEX_DELIM_RE =
  /(?<!\\)\\\[([\s\S]{0,4096}?)\\\]|(?<!\\)\\\(([\s\S]{0,4096}?)\\\)/g;

/**
 * Rewrite `\[...\]` -> block `$$...$$` and `\(...\)` -> inline `$...$` so
 * remark-math can tokenize them. Bodies are trimmed: remark-math won't open an
 * inline span on `$ ` (a `$` followed by whitespace), and display fences must
 * sit on their own line to render as a centered block (not inline math), so
 * `\[...\]` becomes `\n$$\n...\n$$\n`.
 *
 * Spans inside code blocks/spans are left intact (a code sample showing `\(x\)`
 * must not be rewritten).
 *
 * A space is inserted between a converted span and a following `$` so their
 * delimiters can't fuse (`\(a\)\(b\)` -> `$a$$b$` would mis-tokenize into one
 * broken span). A preceding currency (`$5\(x\)`) is instead broken later by the
 * currency escape pass.
 *
 * Returns the rewritten text and the `[start, end)` ranges (in the rewritten
 * string) of every span it produced, so the currency pass can skip them.
 */
function convertLatexDelimiters(content: string): {
  text: string;
  mathRegions: Array<[number, number]>;
} {
  if (!content.includes("\\[") && !content.includes("\\(")) {
    return { text: content, mathRegions: [] };
  }

  const codeRegions = findCodeBlockRegions(content);
  const linkRegions = findLinkDestinationRegions(content);
  const inSkipZone = (pos: number) =>
    isInRegion(pos, codeRegions) || isInRegion(pos, linkRegions);
  // Pushed in ascending, non-overlapping order (offset only grows), so this
  // stays valid for isInRegion's binary search without a sort.
  const mathRegions: Array<[number, number]> = [];
  // Accumulate into an array, not a string: reading the last char off a growing
  // `+=` accumulator flattens its rope every append (O(n^2) over many spans, on
  // the per-frame streaming path), so track the tail char and length instead.
  const parts: string[] = [];
  let offset = 0;
  let lastChar = "";
  let last = 0;
  // Append a chunk, separating a trailing `$` from a leading `$` so two spans
  // can't fuse. Returns where the chunk landed (after any inserted space).
  const append = (chunk: string): number => {
    if (!chunk) return offset;
    if (lastChar === "$" && chunk.startsWith("$")) {
      parts.push(" ");
      offset += 1;
    }
    const start = offset;
    parts.push(chunk);
    offset += chunk.length;
    lastChar = chunk[chunk.length - 1];
    return start;
  };
  let match: RegExpExecArray | null;
  LATEX_DELIM_RE.lastIndex = 0;
  while ((match = LATEX_DELIM_RE.exec(content)) !== null) {
    const matchEnd = match.index + match[0].length;
    // Skip if either delimiter is inside code or a link destination: an opener
    // outside such a zone must not consume a closer inside one and rewrite
    // across the boundary. Resume right after this opener (not past the whole
    // match) so a valid span that this match spanned across (a stray code `\(`
    // paired with a real closer) is still found on the next pass, not swallowed.
    if (inSkipZone(match.index) || inSkipZone(matchEnd - 1)) {
      LATEX_DELIM_RE.lastIndex = match.index + 1;
      continue;
    }
    const isDisplay = match[1] !== undefined;
    const body = (isDisplay ? match[1] : match[2]).trim();
    // Leave an empty span (`\(\)`) literal; a bare `$$` would open a stray
    // display block that swallows following text.
    if (!body) {
      continue;
    }
    append(content.slice(last, match.index));
    let wrapped: string;
    if (isDisplay) {
      // Keep the opener's leading indentation so a `$$` block inside a list item
      // stays in the container instead of breaking out at column 0. Only when the
      // opener is whitespace-prefixed, so inline `text \[x\]` keeps column 0.
      const lineStart =
        match.index > 0 ? content.lastIndexOf("\n", match.index - 1) + 1 : 0;
      const prefix = content.slice(lineStart, match.index);
      const indent = /^\s*$/.test(prefix) ? prefix : "";
      // Indent every body line, not just the first, so multi-line display math
      // (`\[a\nb\]`) stays wholly inside the container.
      const inner = indent ? body.replace(/\n/g, `\n${indent}`) : body;
      wrapped = `\n${indent}$$\n${indent}${inner}\n${indent}$$\n`;
    } else {
      wrapped = `$${body}$`;
    }
    const start = append(wrapped);
    mathRegions.push([start, offset]);
    last = matchEnd;
  }
  append(content.slice(last));
  return { text: parts.join(""), mathRegions };
}

/**
 * Preprocess a markdown string so LaTeX renders: convert bracket delimiters to
 * dollar forms, then escape currency dollar signs so they are not parsed as
 * math delimiters.
 *
 * - `\[E = mc^2\]` becomes a `$$` display block on its own lines (display math)
 * - `\(\alpha\)` becomes `$\alpha$` (inline math)
 * - `\(x\)` in a code span is untouched
 * - `$5` alone becomes `\$5` (currency, not math)
 * - `$\alpha$` is untouched (real LaTeX)
 * - `$30^\circ$` is untouched (LaTeX whose body starts with a digit)
 * - `**$30^\circ$**` is untouched (LaTeX wrapped in bold)
 * - `$$E = mc^2$$` is untouched (display math)
 * - Currency inside code blocks/spans is untouched
 */
export function preprocessLaTeX(content: string): string {
  const { text, mathRegions } = convertLatexDelimiters(content);

  if (!text.includes("$")) return text;

  const codeRegions = findCodeBlockRegions(text);

  return text.replace(CURRENCY_REGEX, (match, offset) => {
    if (isInRegion(offset, codeRegions)) {
      return match;
    }
    // Skip the spans we just created from `\(...\)` so a numeric body like
    // `$5$` isn't re-escaped back to literal `\$5$`.
    if (isInRegion(offset, mathRegions)) {
      return match;
    }
    if (hasInlineMathCloser(text, offset, mathRegions)) {
      return match;
    }
    return "\\" + match;
  });
}
