


/**
 * An HTML comment written mid-sentence is inline raw HTML, not a block, so it
 * belongs to its paragraph: the `-->` may arrive on a later line of that same
 * paragraph and everything between renders as nothing, while past the paragraph
 * the `<!--` is ordinary text. Both release-note scanners share that answer here.
 *
 * The backend needs none of it: a heading closes the paragraph it sits under, so
 * no heading can ever land inside one of these comments.
 */

import { interruptsParagraph } from "@/lib/markdown-list-columns";

const COMMENT_CLOSE = "-->";
// A line that cannot be more of the paragraph above it: blank, or a block that
// may interrupt one. Leading punctuation is not one: `-->` alone is the ordinary
// multiline close and a continuation may open with emphasis, so reading either as
// a break leaves the comment unclosed and its text on show. Indented code and link
// definitions are absent: neither may interrupt a paragraph (spec 0.31.2 4.4, 4.7).
const BLANK = /^[ \t]*$/;
const ATX_HEADING = /^ {0,3}#{1,6}([ \t]|$)/;
const FENCE = /^ {0,3}(?:`{3,}|~{3,})/;
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
// A row of `=` or `-` alone makes the paragraph above it a setext heading, ending it.
const SETEXT_UNDERLINE = /^ {0,3}(?:=+|-+)[ \t]*$/;
// A tag, comment or declaration at the start of a line. HTML block types 1 to 6
// interrupt a paragraph; type 7 does not, but reading one as a break only leaves
// the opener as plain text, which is what a leading `<` has always meant here.
const HTML_LINE = /^ {0,3}</;

/** Whether `line` starts a block of its own rather than continuing a paragraph. */
function startsBlock(line: string): boolean {
  return (
    BLANK.test(line) ||
    ATX_HEADING.test(line) ||
    FENCE.test(line) ||
    THEMATIC_BREAK.test(line) ||
    SETEXT_UNDERLINE.test(line) ||
    HTML_LINE.test(line) ||
    // Blockquote, or a list item with content: the rule the other scanners share.
    interruptsParagraph(line)
  );
}

/**
 * For each line, whether a `-->` is reachable without leaving the paragraph it
 * starts in. Read at `index + 1` it answers whether an inline comment opened on
 * `index` and left unclosed there is a comment at all.
 */
export function commentClosesBelow(lines: string[]): boolean[] {
  const closes: boolean[] = new Array(lines.length + 1).fill(false);
  for (let at = lines.length - 1; at >= 0; at -= 1) {
    const line = lines[at] ?? "";
    closes[at] =
      !startsBlock(line) &&
      (line.includes(COMMENT_CLOSE) || (closes[at + 1] ?? false));
  }
  return closes;
}
