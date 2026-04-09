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
 * Preprocess a markdown string to escape currency dollar signs so they are not
 * parsed as LaTeX math delimiters.
 *
 * - `$5` alone becomes `\$5` (currency, not math)
 * - `$\alpha$` is untouched (real LaTeX)
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
    return "\\" + match;
  });
}
