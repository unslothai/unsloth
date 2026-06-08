// Adapted from LibreChat's latex.ts
// https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts
//
// Escapes currency dollar signs so they are not misinterpreted as LaTeX math
// delimiters when singleDollarTextMath is enabled.

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
 * Find regions inside code blocks (``` ... ``` and ` ... `) so we can skip them.
 * Returns sorted array of [start, end] index pairs.
 */
function findCodeBlockRegions(content: string): Array<[number, number]> {
  const regions: Array<[number, number]> = [];

  // Fenced code blocks: ```...```
  const fencedRe = /```[\s\S]*?```/g;
  let match: RegExpExecArray | null;
  while ((match = fencedRe.exec(content)) !== null) {
    regions.push([match.index, match.index + match[0].length]);
  }

  // Inline code: `...` (but not inside fenced blocks -- we filter below)
  const inlineRe = /`[^`\n]+`/g;
  while ((match = inlineRe.exec(content)) !== null) {
    const start = match.index;
    const end = start + match[0].length;
    // Skip if this backtick span falls inside a fenced block
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

  // Sort by start position for binary search
  regions.sort((a, b) => a[0] - b[0]);
  return regions;
}

/**
 * Binary search to check if a position falls inside any code region.
 */
function isInCodeBlock(
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

/**
 * A token (no whitespace) that looks purely like currency, e.g. `5`,
 * `1,000`, `5.99`, `100K`, `3.5M`.
 */
const CURRENCY_BODY_RE = /^\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?$/;

/** Body characters that almost always indicate real LaTeX. */
const LATEX_CHAR_RE = /[\\^_{}]/;

/**
 * Operators that strongly suggest a math expression. We deliberately
 * leave out `^` and `_` because `LATEX_CHAR_RE` already short-circuits
 * on those before we ever consult this regex.
 */
const MATH_OP_RE = /[=+\-<>/*]/;

/**
 * Trailing chars stripped before the currency check. Includes prose
 * punctuation plus the connectors `-` and `/` that appear in compact
 * currency ranges like `$5-$10` or `$5/$10`; without them the body
 * `5-` or `5/` would slip through the single-token math shortcut.
 */
const TRAIL_PUNCT_RE = /[.,;:!?\-/]+$/;

/**
 * A standalone single letter (variable name) inside the body. We require
 * that the letter is not part of a longer word so that prose like
 * "5 to attend" doesn't get misread as a math expression with the
 * variable `t`.
 */
const LONE_LETTER_RE = /(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])/;

/**
 * Numeric or single-letter operands joined by math operators, with
 * optional whitespace. Matches `2 + 2`, `100 < 200`, `1,000 - 500`,
 * `2.5 * 3`, `x + y`. Lets us recognise numeric-only expressions like
 * `$2 + 2$` as math without forcing a lone variable token.
 */
const SIMPLE_MATH_RE =
  /^(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z])(?:\s*[=+\-<>/*]\s*(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z]))+$/;

/**
 * Return true if the substring between two `$` delimiters looks like a
 * LaTeX expression rather than a span of prose between two currency
 * tokens.
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
 * Return true if the `$` at `offset` opens a balanced inline math span
 * (`$...$`) on the same line. The closer must be unescaped, must not be
 * part of `$$`, and must lie inside a 200-character window. The body
 * must look like LaTeX so we don't pair two currency tokens that share
 * a line (e.g. "$5 to $10"). Bold-wrapped spans (`**$X$**` and the
 * underscore equivalent `__$X$__`) are always treated as math because
 * LLMs reach for that pattern when they want "bold math" and the
 * heuristic would otherwise reject prose-shaped bodies like "90 - x".
 */
function hasInlineMathCloser(content: string, offset: number): boolean {
  const MAX_SPAN = 200;
  const limit = Math.min(content.length, offset + 1 + MAX_SPAN);
  for (let i = offset + 1; i < limit; i++) {
    const c = content[i];
    if (c === "\n") return false;
    if (c !== "$") continue;
    if (content[i - 1] === "\\") continue;
    if (content[i + 1] === "$") {
      i++;
      continue;
    }
    // A `$` immediately followed by a digit is far more likely the start
    // of another currency token than the closer for the current span. Keep
    // scanning so prose like `Starts at $5 + a $10 add-on` doesn't pair
    // the two currency markers as a math span.
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
 * Preprocess a markdown string to escape currency dollar signs so they are not
 * parsed as LaTeX math delimiters.
 *
 * - `$5` alone becomes `\$5` (currency, not math)
 * - `$\alpha$` is untouched (real LaTeX)
 * - `$30^\circ$` is untouched (LaTeX whose body starts with a digit)
 * - `**$30^\circ$**` is untouched (LaTeX wrapped in bold)
 * - `$$E = mc^2$$` is untouched (display math)
 * - Currency inside code blocks/spans is untouched
 */
export function preprocessLaTeX(content: string): string {
  if (!content.includes("$")) return content;

  const codeRegions = findCodeBlockRegions(content);

  return content.replace(CURRENCY_REGEX, (match, offset) => {
    if (isInCodeBlock(offset, codeRegions)) {
      return match;
    }
    if (hasInlineMathCloser(content, offset)) {
      return match;
    }
    return "\\" + match;
  });
}
