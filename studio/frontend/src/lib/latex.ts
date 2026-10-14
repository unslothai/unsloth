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

import { parseMarkdownIntoBlocks } from "./parse-markdown-blocks.ts";
import {
  codeSpans,
  crossesCodeSpanBlockBoundary,
} from "./markdown-code-spans.ts";
import {
  EMPTY_LIST_STATE,
  NO_QUOTE,
  containerContent,
  indentWidth,
  itemContent,
  openLists,
  quoteDepth,
  quoteState,
} from "./markdown-list-columns.ts";

/** matches currency bodies after `$`, including separators, decimals, and K/M/B suffixes. */
const CURRENCY_REGEX = /\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?(?:\s|$|[^a-zA-Z\d])/y;

const HEADING_LINE_RE = / {0,3}#{1,6}(?=[ \t\r\n]|$)/y;
const TABLE_ROW_RE = /[ \t]*\|/y;
const BLOCK_BREAK_RE =
  /\n[ \t\r]*(?:\n|#{1,6}(?=[ \t\r\n])|[>|]|[-*+][ \t]|1[.)][ \t]|```|~~~|(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,}|=+[ \t]*)(?=\r?(?:\n|$)))/;

/** matches prose-like `$NAME ... $word` spans without math symbols. */
// non-ASCII letters and CJK punctuation are prose (KaTeX rejects them in math mode).
const VARIABLE_PROSE_RE =
  /^(?!\w+\s+$)(?:[A-Za-z]{2,}\w*|_\w+|\{[A-Za-z_]\w*\})[\w\s.,;:!?'"()/`|&<>=*\-\p{L}\p{M}\u3000-\u303f\uff00-\uff65]*(?:[\s/:,.;|<>=\-\u3000-\u303f\uff00-\uff65]|[^\P{L}\p{ASCII}]|[\s(]["'(`])$/u;
// VARIABLE_PROSE_RE backtracks ~cubically: longer spans, or any past the per-message budget, stay math.
const MAX_VARIABLE_PROSE_SPAN = 128;
const VARIABLE_PROSE_BUDGET = 2048;
const NEW_TOKEN_RE = /[\w{\\]/;
const isAsciiLetter = (c: number) => (c | 32) >= 97 && (c | 32) <= 122;
const isWordChar = (c: number) =>
  isAsciiLetter(c) || (c >= 48 && c <= 57) || c === 95;

/** necessary prefix of VARIABLE_PROSE_RE, checked before slicing the span. */
function startsLikeVariable(text: string, at: number): boolean {
  const a = text.charCodeAt(at);
  const b = text.charCodeAt(at + 1);
  if (isAsciiLetter(a)) return isAsciiLetter(b);
  if (a === 95) return isWordChar(b);
  return a === 123 && (isAsciiLetter(b) || b === 95);
}

const MAX_SHELL_EXPANSION_SPAN = 4096;
const MAX_MARKDOWN_EMPHASIS_SPAN = 256;
const SHELL_QUOTE_CONTEXT_CANDIDATE_RE =
  /\$(?:["'!(*\-?_]|\[)|\$\d+\$[A-Za-z0-9_{$]/;
const SHELL_COMMANDS = new Set([
  "awk",
  "bash",
  "builtin",
  "cat",
  "command",
  "dash",
  "echo",
  "env",
  "eval",
  "exec",
  "export",
  "find",
  "fish",
  "git",
  "grep",
  "ksh",
  "kubectl",
  "printf",
  "read",
  "sed",
  "set",
  "sh",
  "source",
  "sudo",
  "test",
  "unset",
  "xargs",
  "zsh",
]);
const SHELL_LEADIN_RE = /^(?:Run|Use|Try|Execute)$/i;
const PROSE_COMMAND_RE =
  /^(?:a|an|consider|imagine|note|suppose|the|this|that)$/i;
const SHELL_WORD_RE = /^(?:[a-z_][\w.-]*|\.{0,2}\/[^\s]+)$/;
const SHELL_POSITIONAL_WORD_RE = /^[A-Za-z0-9_][\w.-]*$/;
const SHELL_ARGUMENT_RE =
  /^(?!(?:\d+\.)+\d+$)(?:-{1,2}[\w-]+|[\w.-]+=\S*|\.{0,2}\/\S+|\S*[./*?[\]{}]\S*)$/;
const MARKDOWN_CONTAINER_PREFIX_RE =
  /^(?:(?: {0,3}>[ \t]?)|(?: {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+(?:\[[ xX]\][ \t]+)?))+/;
const SHELL_PROMPT_PREFIX_RE = /^\$[ \t]+/;
const MARKDOWN_TABLE_PREFIX_RE = /^[ \t]*\|/;
const SHELL_ASSIGNMENT_PREFIX_RE = /(?:^|[;&|]\s*)[A-Za-z_]\w*=$/;
const SHELL_ASSIGNMENT_WORD_RE = /^[A-Za-z_]\w*=\S*$/;
const SHELL_OPERATOR_PREFIX_RE = /\S\s*[;&|]{1,2}$/;
const SHELL_QUOTE_BLOCK_START_RE =
  /^(?: {0,3}(?:#{1,6}(?=[ \t])|[-+*][ \t]+|\d{1,9}[.)][ \t]+)|[ \t]*\|| {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,}|=+[ \t]*)$)/;

function hasCommandArguments(words: string[], at: number): boolean {
  if (at >= words.length) return false;
  let hasShellShapedArgument = false;
  for (let index = at; index < words.length; index++) {
    const word = words[index];
    if (SHELL_ARGUMENT_RE.test(word)) {
      hasShellShapedArgument = true;
    } else if (!SHELL_POSITIONAL_WORD_RE.test(word)) {
      return false;
    }
  }
  return hasShellShapedArgument;
}

function isShellWordBoundary(char: number): boolean {
  return (
    char <= 32 ||
    char === 34 ||
    char === 36 ||
    char === 38 ||
    char === 39 ||
    char === 40 ||
    char === 41 ||
    char === 59 ||
    char === 60 ||
    char === 61 ||
    char === 62 ||
    char === 92 ||
    char === 124
  );
}

function logicalShellLineStart(
  text: string,
  at: number,
  minimum: number,
): number {
  let start = at;
  while (start > minimum) {
    while (
      start > minimum &&
      text.charCodeAt(start - 1) !== 10 &&
      text.charCodeAt(start - 1) !== 13
    ) {
      start--;
    }
    if (start === minimum) return start;
    let newlineStart = start - 1;
    if (
      text.charCodeAt(newlineStart) === 10 &&
      text.charCodeAt(newlineStart - 1) === 13
    ) {
      newlineStart--;
    }
    let slash = newlineStart;
    while (slash > minimum && text.charCodeAt(slash - 1) === 92) slash--;
    if ((newlineStart - slash) % 2 === 0) return start;
    start = newlineStart;
  }
  return start;
}

function lastShellControlOperatorEnd(prefix: string): number {
  let quote = 0;
  let last = 0;
  for (let index = 0; index < prefix.length; index++) {
    const char = prefix.charCodeAt(index);
    if (quote !== 0) {
      if (char === 92 && quote !== 39) index++;
      else if (char === quote) quote = 0;
      continue;
    }
    if (char === 34 || char === 39) {
      quote = char;
    } else if (char === 92) {
      index++;
    } else if (char === 38 || char === 59 || char === 124) {
      if (prefix.charCodeAt(index + 1) === char) index++;
      last = index + 1;
    }
  }
  return last;
}

function isShellPromptMarker(text: string, at: number): boolean {
  const next = text.charCodeAt(at + 1);
  if (next !== 9 && next !== 32) return false;
  const lineStart =
    Math.max(text.lastIndexOf("\n", at - 1), text.lastIndexOf("\r", at - 1)) +
    1;
  return !text
    .slice(lineStart, at)
    .replace(MARKDOWN_CONTAINER_PREFIX_RE, "")
    .trim();
}

function hasShellCommandPrefix(
  text: string,
  at: number,
  allowBareCommand = false,
): boolean {
  const previousChar = text.charCodeAt(at - 1);
  if (
    !allowBareCommand &&
    at > 0 &&
    previousChar > 32 &&
    previousChar !== 38 &&
    previousChar !== 59 &&
    previousChar !== 61 &&
    previousChar !== 124
  ) {
    if (isShellWordBoundary(previousChar)) return false;
    const minimum = Math.max(0, at - 256);
    let wordStart = at;
    while (wordStart > minimum) {
      const char = text.charCodeAt(wordStart - 1);
      if (isShellWordBoundary(char)) break;
      wordStart--;
    }
    if (wordStart === at - 1 && text.charCodeAt(wordStart - 1) === 36) {
      return false;
    }
    return (
      wordStart > minimum &&
      wordStart < at &&
      hasShellCommandPrefix(text, wordStart, true)
    );
  }
  const minimum = Math.max(0, at - 256);
  const lineStart = logicalShellLineStart(text, at, minimum);
  const completeLinePrefix =
    lineStart === 0 ||
    text.charCodeAt(lineStart - 1) === 10 ||
    text.charCodeAt(lineStart - 1) === 13;
  const rawPrefix = text
    .slice(lineStart, at)
    .replace(/\\(?:\r\n|\n|\r)[ \t]*/g, " ")
    .trimEnd();
  const prefix = rawPrefix
    .replace(MARKDOWN_CONTAINER_PREFIX_RE, "")
    .replace(SHELL_PROMPT_PREFIX_RE, "");
  if (!prefix) return false;
  if (
    MARKDOWN_TABLE_PREFIX_RE.test(rawPrefix) ||
    MARKDOWN_TABLE_PREFIX_RE.test(prefix)
  )
    return false;
  if (
    SHELL_ASSIGNMENT_PREFIX_RE.test(prefix) ||
    (!MARKDOWN_TABLE_PREFIX_RE.test(rawPrefix) &&
      SHELL_OPERATOR_PREFIX_RE.test(prefix))
  ) {
    return true;
  }
  if (/\b(?:shell|bash|command) examples?\b[^\n.]*$/i.test(prefix)) return true;
  const commandStart = lastShellControlOperatorEnd(prefix);
  const commandPrefix = prefix.slice(commandStart).trimStart();
  const words = commandPrefix.match(/[^\s]+/g) ?? [];
  let commandIndex = 0;
  while (
    commandIndex < words.length - 1 &&
    SHELL_LEADIN_RE.test(words[commandIndex])
  ) {
    commandIndex++;
  }
  while (
    commandIndex < words.length - 1 &&
    SHELL_ASSIGNMENT_WORD_RE.test(words[commandIndex])
  ) {
    commandIndex++;
  }
  const command = words[commandIndex] ?? "";
  if (SHELL_COMMANDS.has(command) || /^\.{0,2}\//.test(command)) return true;
  if (
    allowBareCommand &&
    completeLinePrefix &&
    command.length >= 2 &&
    !PROSE_COMMAND_RE.test(command) &&
    SHELL_WORD_RE.test(command) &&
    (commandStart > 0 || hasCommandArguments(words, commandIndex + 1))
  ) {
    return true;
  }
  return false;
}

function hasShellParameterContext(
  text: string,
  at: number,
  numericOffset: number,
  quoteState: number,
): boolean {
  if (quoteState > 0 || hasShellCommandPrefix(text, numericOffset)) return true;
  if (at !== numericOffset + 3) return false;
  const lineStart =
    Math.max(
      text.lastIndexOf("\n", numericOffset - 1),
      text.lastIndexOf("\r", numericOffset - 1),
    ) + 1;
  const prefix = text
    .slice(lineStart, numericOffset)
    .trimEnd()
    .replace(MARKDOWN_CONTAINER_PREFIX_RE, "")
    .replace(SHELL_PROMPT_PREFIX_RE, "");
  return SHELL_LEADIN_RE.test(prefix);
}

function startsNewMarkdownBlock(text: string, at: number): boolean {
  const char = text.charCodeAt(at);
  if (at >= text.length || char === 10 || char === 13) return true;
  const nextLf = text.indexOf("\n", at);
  const nextCr = text.indexOf("\r", at);
  const nextEnd =
    nextLf === -1 ? nextCr : nextCr === -1 ? nextLf : Math.min(nextLf, nextCr);
  const nextLine = text.slice(at, nextEnd === -1 ? text.length : nextEnd);
  const nextQuotes = quoteDepth(nextLine);
  if (nextQuotes === 0) return SHELL_QUOTE_BLOCK_START_RE.test(nextLine);

  let previousEnd = at - 1;
  if (
    text.charCodeAt(previousEnd) === 10 &&
    text.charCodeAt(previousEnd - 1) === 13
  ) {
    previousEnd--;
  }
  const previousStart =
    Math.max(
      text.lastIndexOf("\n", previousEnd - 1),
      text.lastIndexOf("\r", previousEnd - 1),
    ) + 1;
  const previousLine = text.slice(previousStart, previousEnd);
  if (quoteDepth(previousLine) !== nextQuotes) return true;
  const content = containerContent(nextLine, EMPTY_LIST_STATE, nextQuotes);
  return !content.trim() || SHELL_QUOTE_BLOCK_START_RE.test(content);
}

function findShellQuoteStates(
  text: string,
  resetRegions: ReadonlyArray<readonly [number, number]>,
): Map<number, number> {
  const states = new Map<number, number>();
  let quote = 0;
  let comment = false;
  let resetIndex = 0;
  for (let index = 0; index < text.length; index++) {
    while (
      resetIndex < resetRegions.length &&
      resetRegions[resetIndex][1] <= index
    ) {
      resetIndex++;
    }
    const resetRegion = resetRegions[resetIndex];
    if (resetRegion && index >= resetRegion[0] && index < resetRegion[1]) {
      quote = 0;
      index = resetRegion[1] - 1;
      continue;
    }
    const char = text.charCodeAt(index);
    if (comment) {
      if (char === 10 || char === 13) {
        comment = false;
      } else {
        if (char === 36) states.set(index, 34);
        continue;
      }
    }
    if (
      (char === 10 || (char === 13 && text.charCodeAt(index + 1) !== 10)) &&
      quote !== 0 &&
      startsNewMarkdownBlock(text, index + 1)
    ) {
      quote = 0;
      continue;
    }
    const activeQuote = Math.abs(quote);
    if (char === 36) states.set(index, quote);
    if (
      quote === 0 &&
      char === 35 &&
      (index === 0 || text.charCodeAt(index - 1) <= 32) &&
      hasShellCommandPrefix(text, index)
    ) {
      comment = true;
      continue;
    }
    if (char === 92 && activeQuote !== 39) {
      index++;
    } else if (quote === 0 && (char === 34 || char === 39)) {
      const shellQuote = hasShellCommandPrefix(text, index, true);
      if (
        char === 39 &&
        isWordChar(text.charCodeAt(index - 1)) &&
        !shellQuote
      ) {
        continue;
      }
      quote = shellQuote ? char : -char;
    } else if (char === activeQuote) {
      quote = 0;
    }
  }
  return states;
}

function startsMarkdownEmphasis(
  text: string,
  at: number,
  marker: number,
): boolean {
  const width = text.charCodeAt(at + 1) === marker ? 2 : 1;
  const bodyStart = at + width;
  const first = text.charCodeAt(bodyStart);
  if (first <= 32) return false;
  const token = width === 2 ? text.slice(at, at + 2) : text[at];
  const nextLf = text.indexOf("\n", bodyStart);
  const nextCr = text.indexOf("\r", bodyStart);
  const newline =
    nextLf === -1 ? nextCr : nextCr === -1 ? nextLf : Math.min(nextLf, nextCr);
  const limit = Math.min(
    text.length,
    at + MAX_MARKDOWN_EMPHASIS_SPAN,
    newline === -1 ? text.length : newline,
  );
  const close = text.indexOf(token, bodyStart + 1);
  return close > bodyStart && close < limit && text.charCodeAt(close - 1) > 32;
}

function enclosingMarkdownEmphasisOpener(
  text: string,
  at: number,
  before: number,
  marker: number,
): number {
  const width = text.charCodeAt(at + 1) === marker ? 2 : 1;
  const token = width === 2 ? text.slice(at, at + 2) : text[at];
  const lineStart =
    Math.max(text.lastIndexOf("\n", at - 1), text.lastIndexOf("\r", at - 1)) +
    1;
  const minimum = Math.max(lineStart, at - MAX_MARKDOWN_EMPHASIS_SPAN);
  let opener = text.lastIndexOf(token, before - 1);
  while (opener >= minimum) {
    const bodyStart = opener + width;
    if (
      bodyStart <= before &&
      startsMarkdownEmphasis(text, opener, marker) &&
      text.indexOf(token, bodyStart + 1) === at
    ) {
      return opener;
    }
    opener = text.lastIndexOf(token, opener - 1);
  }
  return -1;
}

function findLegacyArithmeticCloses(text: string): Int32Array {
  const closes = new Int32Array(text.length);
  let nextClose = -1;
  for (let index = text.length - 1; index >= 0; index--) {
    const char = text.charCodeAt(index);
    if (char === 93) nextClose = index;
    else if (char === 91 && text.charCodeAt(index - 1) === 36) {
      closes[index] = nextClose + 1;
    }
  }
  return closes;
}

function startsLegacyArithmetic(
  text: string,
  at: number,
  numericOffset: number,
  quoteState: number,
  closes: Int32Array | undefined,
  isStreaming: boolean,
): boolean {
  const limit = Math.min(text.length, at + MAX_SHELL_EXPANSION_SPAN);
  const close = (closes?.[at] ?? 0) - 1;
  if (close === -1) {
    return (
      isStreaming &&
      quoteState >= 0 &&
      (quoteState > 0 || hasShellCommandPrefix(text, numericOffset))
    );
  }
  if (close >= limit || text[at + 1] === "^") return false;
  if (text[close + 1] === "(" || text[close + 1] === "[") return false;
  if (quoteState < 0) return false;
  return quoteState > 0 || hasShellCommandPrefix(text, numericOffset);
}

/** valid Bash expansion starts after `$`, disambiguated from prose punctuation and closing quotes. */
function startsShellExpansion(
  text: string,
  at: number,
  numericOffset: number,
  quoteState: number,
  legacyArithmeticCloses: Int32Array | undefined,
  isStreaming: boolean,
): boolean {
  if (at >= text.length) return false;
  const char = text.charCodeAt(at);
  const quote = Math.abs(quoteState);
  if (char === 34 || char === 39) {
    return (
      quote !== char &&
      (quoteState > 0 || hasShellCommandPrefix(text, numericOffset))
    );
  }
  if (char === 33 || char === 45 || char === 63) {
    return quoteState > 0 || hasShellCommandPrefix(text, numericOffset);
  }
  if (char === 91)
    return startsLegacyArithmetic(
      text,
      at,
      numericOffset,
      quoteState,
      legacyArithmeticCloses,
      isStreaming,
    );
  if (char === 42 || char === 95) {
    const emphasisOpener = enclosingMarkdownEmphasisOpener(
      text,
      at,
      numericOffset,
      char,
    );
    if (
      emphasisOpener !== -1 &&
      quoteState <= 0 &&
      !hasShellCommandPrefix(text, emphasisOpener)
    ) {
      return false;
    }
  }
  if (
    (char === 42 || char === 95) &&
    startsMarkdownEmphasis(text, at, char) &&
    quoteState <= 0 &&
    !hasShellCommandPrefix(text, numericOffset)
  ) {
    return false;
  }
  if (char === 42) return true;
  if (char === 64 || char === 35) return true;
  if (char === 40) {
    return quoteState > 0 || hasShellCommandPrefix(text, numericOffset);
  }
  return (
    (isWordChar(char) || char === 123 || char === 36) &&
    hasShellParameterContext(text, at, numericOffset, quoteState)
  );
}

function numericShellParameterStart(text: string, at: number): number {
  let start = at;
  while (start > 0) {
    const char = text.charCodeAt(start - 1);
    if (char < 48 || char > 57) break;
    start--;
  }
  return start < at &&
    text[start - 1] === "$" &&
    text[start - 2] !== "\\" &&
    text[start - 2] !== "$"
    ? start - 1
    : -1;
}

function shellExpansionFallbackEnd(
  text: string,
  at: number,
  limit: number,
): number {
  const nextLf = text.indexOf("\n", at + 2);
  const nextCr = text.indexOf("\r", at + 2);
  const lineEnd =
    nextLf === -1 ? nextCr : nextCr === -1 ? nextLf : Math.min(nextLf, nextCr);
  return lineEnd !== -1 && lineEnd < limit ? lineEnd : limit;
}

function balancedShellExpansionEnd(
  text: string,
  at: number,
  open: number,
  close: number,
): number {
  let depth = 1;
  let quote = 0;
  const limit = Math.min(text.length, at + MAX_SHELL_EXPANSION_SPAN);
  for (let index = at + 2; index < limit; index++) {
    const char = text.charCodeAt(index);
    if (quote !== 0) {
      if (char === 92 && quote !== 39) index++;
      else if (char === quote) quote = 0;
      continue;
    }
    if (char === 34 || char === 39 || char === 96) {
      quote = char;
      continue;
    }
    if (char === 92) {
      index++;
      continue;
    }
    if (char === open) depth++;
    else if (char === close && --depth === 0) return index + 1;
  }
  return shellExpansionFallbackEnd(text, at, limit);
}

function quotedShellExpansionEnd(
  text: string,
  at: number,
  quote: number,
): number {
  const limit = Math.min(text.length, at + MAX_SHELL_EXPANSION_SPAN);
  for (let index = at + 2; index < limit; index++) {
    const char = text.charCodeAt(index);
    if (char === 92) index++;
    else if (char === quote) return index + 1;
  }
  return shellExpansionFallbackEnd(text, at, limit);
}

function shellExpansionEnd(text: string, at: number): number {
  const first = text.charCodeAt(at + 1);
  if (first === 123) return balancedShellExpansionEnd(text, at, 123, 125);
  if (first === 40) return balancedShellExpansionEnd(text, at, 40, 41);
  if (first === 91) return balancedShellExpansionEnd(text, at, 91, 93);
  if (first === 34 || first === 39)
    return quotedShellExpansionEnd(text, at, first);
  if (!isAsciiLetter(first) && first !== 95) return at + 2;
  let end = at + 2;
  while (end < text.length && isWordChar(text.charCodeAt(end))) end++;
  return end;
}

function encodeMarkdownLiteral(text: string): string {
  let encoded = "";
  for (const char of text) encoded += `&#${char.codePointAt(0)};`;
  return encoded;
}
// an entity stays literal in Markdown without showing an escape slash in raw HTML.
const VARIABLE_DOLLAR = "&#36;";

/** merges ascending spans; overlaps merge and non-ascending spans drop. */
function mergeRegions(
  left: ReadonlyArray<readonly [number, number]>,
  right: ReadonlyArray<readonly [number, number]>,
): Array<[number, number]> {
  const merged: Array<[number, number]> = [];
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < left.length || rightIndex < right.length) {
    const takeLeft =
      rightIndex >= right.length ||
      (leftIndex < left.length && left[leftIndex][0] <= right[rightIndex][0]);
    const next = takeLeft ? left[leftIndex++] : right[rightIndex++];
    const last = merged[merged.length - 1];
    if (last && next[0] < last[1]) {
      if (next[1] > last[1]) {
        last[1] = next[1];
      }
      continue;
    }
    merged.push([next[0], next[1]]);
  }
  return merged;
}

const FENCE_LINE_RE = /^ {0,3}(`{3,}|~{3,})([^\r\n]*)$/;
const FENCE_CANDIDATE_RE =
  /(^|\r\n|\n|\r)((?:(?: {0,3}>[ \t]?)|(?:[ \t]*(?:[-+*]|\d{1,9}[.)])[ \t]+))*[ \t]*)(`{3,}|~{3,})([^\r\n]*)/g;
const INDENTED_CODE_CANDIDATE_RE =
  /(^|\r\n|\n|\r)(?: {0,3}>[ \t]?)*(?:(?: {4}| {0,3}\t)| {0,3}(?:[-+*]|\d{1,9}[.)])(?: {5,}|[ \t]*\t[ \t]*))/g;
const BLOCK_LINE_RE =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$|>|=+[ \t]*$)/;
const LINK_DEFINITION_RE = /^ {0,3}\[(?:[^\[\]\\]|\\.)+\]:/;
const NON_LINE_ENDING_RE = /[^\r\n]/g;

/** `line` with up to `columns` columns of leading whitespace removed. */
function stripIndent(line: string, columns: number): string {
  let width = 0;
  let index = 0;
  while (index < line.length && width < columns) {
    const char = line[index];
    if (char !== " " && char !== "\t") break;
    width += char === " " ? 1 : 4 - (width % 4);
    index += 1;
  }
  return line.slice(index);
}

/** Display columns occupied by a Markdown container prefix. */
function columnWidth(prefix: string): number {
  let width = 0;
  for (const char of prefix) {
    width += char === "\t" ? 4 - (width % 4) : 1;
  }
  return width;
}

/**
 * Find code regions (fenced, indented, and inline) to skip.
 * Returns a sorted, non-overlapping array of [start, end] index pairs.
 */
function normalizedMarkdownOffsets(content: string): {
  text: string;
  offsets: number[];
} {
  let text = "";
  const offsets = [0];
  for (let index = 0; index < content.length; index += 1) {
    if (content[index] === "\r") {
      if (content[index + 1] === "\n") {
        index += 1;
      }
      text += "\n";
    } else {
      text += content[index];
    }
    offsets.push(index + 1);
  }
  return { text, offsets };
}

function findInlineCodeRegions(
  content: string,
  blockRegions: Array<[number, number]>,
): Array<[number, number]> {
  // Every backtick inside a block region becomes a space in the mask below, so unless one sits
  // outside a block region the mask cannot contain a backtick, `codeSpans` returns nothing and the
  // whole rebuild is spent to learn that. Worth checking first because the streaming path calls
  // this once per frame over a growing prefix, twice per preprocessLaTeX, so the rebuild is O(n)
  // per frame and O(n^2) over a reply; and a reply carrying one fenced example and no inline code,
  // where every backtick is inside the fence, is the common shape rather than a corner case.
  let spanTickOutsideBlock = false;
  for (
    let index = content.indexOf("`");
    index !== -1;
    index = content.indexOf("`", index + 1)
  ) {
    if (!isInRegion(index, blockRegions)) {
      spanTickOutsideBlock = true;
      break;
    }
  }
  if (!spanTickOutsideBlock) return [];

  let masked = content;
  if (blockRegions.length > 0) {
    const parts: string[] = [];
    let cursor = 0;
    for (const [start, end] of blockRegions) {
      parts.push(content.slice(cursor, start));
      parts.push(content.slice(start, end).replace(NON_LINE_ENDING_RE, " "));
      cursor = end;
    }
    parts.push(content.slice(cursor));
    masked = parts.join("");
  }

  const spans = codeSpans(masked);
  const crossesBlock = spans.some(({ start, end }) =>
    crossesCodeSpanBlockBoundary(masked.slice(start, end)),
  );
  if (!crossesBlock) {
    return spans.map(({ start, end }) => [start, end]);
  }

  const normalized = normalizedMarkdownOffsets(masked);
  const blocks = parseMarkdownIntoBlocks(normalized.text);
  const regions: Array<[number, number]> = [];
  let blockStart = 0;
  for (const block of blocks) {
    for (const span of codeSpans(block)) {
      regions.push([
        normalized.offsets[blockStart + span.start] ?? content.length,
        normalized.offsets[blockStart + span.end] ?? content.length,
      ]);
    }
    blockStart += block.length;
  }
  return regions;
}

export function findCodeBlockRegions(content: string): Array<[number, number]> {
  const fenced: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  const indented: Array<[number, number]> = [];

  FENCE_CANDIDATE_RE.lastIndex = 0;
  INDENTED_CODE_CANDIDATE_RE.lastIndex = 0;
  let simpleFenceStart = -1;
  let simpleFenceMarker = "";
  let needsBlockScan = false;
  while ((match = FENCE_CANDIDATE_RE.exec(content)) !== null) {
    const prefix = match[2] ?? "";
    if (prefix.trim() || indentWidth(prefix) > 3) {
      needsBlockScan = true;
      break;
    }
    const marker = match[3] ?? "";
    const tail = match[4] ?? "";
    const lineStart = match.index + (match[1]?.length ?? 0);
    if (simpleFenceStart < 0) {
      if (marker[0] !== "`" || !tail.includes("`")) {
        simpleFenceStart = lineStart;
        simpleFenceMarker = marker;
      }
    } else if (
      marker[0] === simpleFenceMarker[0] &&
      marker.length >= simpleFenceMarker.length &&
      !tail.trim()
    ) {
      fenced.push([
        simpleFenceStart,
        lineStart + prefix.length + marker.length,
      ]);
      simpleFenceStart = -1;
      simpleFenceMarker = "";
    }
  }
  if (!needsBlockScan && simpleFenceStart >= 0) {
    fenced.push([simpleFenceStart, content.length]);
  }
  if (!needsBlockScan) {
    while ((match = INDENTED_CODE_CANDIDATE_RE.exec(content)) !== null) {
      if (!isInRegion(match.index + match[0].length - 1, fenced)) {
        needsBlockScan = true;
        break;
      }
    }
  }
  if (needsBlockScan) {
    fenced.length = 0;
    const lines = content.matchAll(/[^\r\n]*(?:\r\n|\n|\r|$)/g);
    let openFence: {
      start: number;
      marker: string;
      column: number;
      quotes: number;
    } | null = null;
    let indentedStart = -1;
    let indentedEnd = -1;
    let afterParagraph = false;
    let lists = EMPTY_LIST_STATE;
    let quote = NO_QUOTE;

    for (const lineMatch of lines) {
      const line = lineMatch[0];
      if (!line) break;
      const start = lineMatch.index;
      const text = line.replace(/(?:\r\n|\n|\r)$/, "");
      const above = quote;
      quote = NO_QUOTE;
      const quotes = quoteDepth(text);

      if (openFence !== null) {
        const quoted = containerContent(
          text,
          EMPTY_LIST_STATE,
          openFence.quotes,
        );
        const leftContainer =
          quotes < openFence.quotes ||
          (quoted.trim() !== "" &&
            openFence.column > 0 &&
            indentWidth(quoted) < openFence.column);
        if (leftContainer) {
          fenced.push([openFence.start, start]);
          openFence = null;
        }
      }

      const activeFence = openFence;
      const container = containerContent(
        text,
        lists,
        activeFence?.quotes ?? quotes,
      );
      let fenceSource: string;
      if (activeFence) {
        fenceSource = stripIndent(container, activeFence.column);
      } else {
        fenceSource = container;
        let previous: string;
        let itemAfterParagraph = afterParagraph;
        do {
          previous = fenceSource;
          fenceSource = itemContent(fenceSource, itemAfterParagraph);
          itemAfterParagraph = false;
        } while (fenceSource !== previous);
      }
      const fence = FENCE_LINE_RE.exec(fenceSource);
      if (fence !== null) {
        lists = openLists(text, lists, afterParagraph, above.quoted);
        const marker = fence[1] ?? "";
        const tail = fence[2] ?? "";
        if (openFence === null) {
          if (marker[0] !== "`" || !tail.includes("`")) {
            const quoted = containerContent(text, EMPTY_LIST_STATE, quotes);
            openFence = {
              start,
              marker,
              column: columnWidth(
                quoted.slice(0, quoted.length - fenceSource.length),
              ),
              quotes,
            };
          }
        } else if (
          marker[0] === openFence.marker[0] &&
          marker.length >= openFence.marker.length &&
          !tail.trim()
        ) {
          fenced.push([openFence.start, start + line.length]);
          openFence = null;
        }
        afterParagraph = false;
        continue;
      }

      if (openFence !== null) {
        lists = openLists("", lists, afterParagraph, above.quoted);
        afterParagraph = false;
        continue;
      }

      lists = openLists(text, lists, afterParagraph, above.quoted);
      const inner = itemContent(
        containerContent(text, lists, quoteDepth(text)),
        afterParagraph,
      );
      const blank = /^\s*$/.test(inner);
      const code = indentWidth(inner) >= 4;
      if (indentedStart >= 0) {
        if (code || blank) {
          indentedEnd = start + line.length;
          continue;
        }
        indented.push([indentedStart, indentedEnd]);
        indentedStart = -1;
      }
      if (code && !afterParagraph && !blank) {
        indentedStart = start;
        indentedEnd = start + line.length;
        afterParagraph = false;
        continue;
      }
      afterParagraph =
        !blank &&
        !BLOCK_LINE_RE.test(text) &&
        (afterParagraph || !LINK_DEFINITION_RE.test(text));
      quote = quoteState(text, above.inQuote);
    }
    if (openFence !== null) fenced.push([openFence.start, content.length]);
    if (indentedStart >= 0) indented.push([indentedStart, indentedEnd]);
  }

  const blocks = mergeRegions(fenced, indented);
  const inline = findInlineCodeRegions(content, blocks);

  // merge overlaps so binary search does not miss a containing inline span.
  return mergeRegions(blocks, inline);
}

/** destinations allow escapes and one level of balanced parentheses. */
const LINK_DEST_RE =
  /!?\[(?:\\.|[^\]\\])*?\]\(((?:\\.|[^()\\]|\([^()]*\))*)\)/dg;
const AUTOLINK_RE =
  /<(?:[a-z][a-z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>|\b(?:https?:\/\/|www\.)[^\s<]+/gi;

/** shields link destinations from math rewrites; link text stays convertible. */
function findLinkDestinationRegions(content: string): Array<[number, number]> {
  if (!content.includes("](")) return [];
  const regions: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  LINK_DEST_RE.lastIndex = 0;
  while ((match = LINK_DEST_RE.exec(content)) !== null) {
    // escaped separators make searches unsafe; use the `d` flag's bounds.
    regions.push(match.indices![1]);
  }
  return regions;
}

// raw-text element bodies do not decode character references.
const RAW_TEXT_ELEMENT_RE =
  /<(script|style|xmp|iframe|noembed|noframes)\b[^>]*>[\s\S]*?(?:<\/\1\s*>|$)/gi;

function findRawTextRegions(content: string): Array<[number, number]> {
  if (!content.includes("<")) return [];
  const regions: Array<[number, number]> = [];
  for (const match of content.matchAll(RAW_TEXT_ELEMENT_RE)) {
    regions.push([match.index, match.index + match[0].length]);
  }
  return regions;
}

function findAutolinkRegions(content: string): Array<[number, number]> {
  const regions: Array<[number, number]> = [];
  AUTOLINK_RE.lastIndex = 0;
  for (const match of content.matchAll(AUTOLINK_RE)) {
    regions.push([match.index, match.index + match[0].length]);
  }
  return regions;
}

/** `isInRegion` for ascending positions: amortised O(1) per query. */
function regionCursor(
  regions: Array<[number, number]>,
): (position: number) => boolean {
  let k = 0;
  return (position) => {
    while (k < regions.length && regions[k][1] <= position) k++;
    return k < regions.length && regions[k][0] <= position;
  };
}

/** regions must be sorted by start and non-overlapping. */
export function isInRegion(
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

const CURRENCY_BODY_RE = /^\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?$/;

const BARE_NUMBER_RE = /^\d+(?:,\d{3})*(?:\.\d+)?$/;

/** these characters almost always indicate real LaTeX. */
const LATEX_CHAR_RE = /[\\^_{}]/;

/** math operators excluding `^` and `_`, which LATEX_CHAR_RE checks first. */
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

/** accepts bold-wrapped spans to preserve prose-shaped math and bare numbers unless the closer starts a shell parameter. */
function hasInlineMathCloser(
  content: string,
  offset: number,
  mathRegions: Array<[number, number]>,
  shellQuoteStates: Map<number, number>,
  legacyArithmeticCloses: Int32Array | undefined,
  isStreaming: boolean,
): boolean {
  const maxSpan = 200;
  const limit = Math.min(content.length, offset + 1 + maxSpan);
  for (let i = offset + 1; i < limit; i++) {
    const c = content[i];
    if (c === "\n") return false;
    if (c !== "$") continue;
    if (content[i - 1] === "\\") continue;
    // generated `\(...\)` openers cannot close currency spans because they would consume the price as math.
    if (isInRegion(i, mathRegions)) return false;
    if (content[i + 1] === "$") {
      i++;
      continue;
    }
    // a digit after `$` likely starts another currency token, so keep scanning.
    if (/\d/.test(content[i + 1] ?? "")) {
      continue;
    }
    const body = content.slice(offset + 1, i);
    if (BARE_NUMBER_RE.test(body)) {
      if (isStreaming && i + 1 === content.length) return false;
      if (
        !startsShellExpansion(
          content,
          i + 1,
          offset,
          shellQuoteStates.get(offset) ?? 0,
          legacyArithmeticCloses,
          isStreaming,
        )
      )
        return true;
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
    return looksLikeMathBody(body);
  }
  return false;
}

// remark-math closes an open span at the next single `$`, escaped or not, within the block.
function findInlineMathCloser(
  content: string,
  offset: number,
  lineStart: number,
  lineEnd: number,
  tableRow: boolean,
): number {
  let i = content.indexOf("$", offset + 1);
  while (i !== -1 && content[i + 1] === "$") {
    while (content[i + 1] === "$") i++;
    i = content.indexOf("$", i + 1);
  }
  if (i === -1) return -1;
  // hot path: only multi-line spans can cross a block break or leave a heading.
  const multiline = lineEnd !== -1 && lineEnd < i;
  if (multiline) {
    if (BLOCK_BREAK_RE.test(content.slice(offset + 1, i))) return -1;
    HEADING_LINE_RE.lastIndex = lineStart;
    if (HEADING_LINE_RE.test(content)) return -1;
  }
  if (tableRow) {
    if (multiline) return -1;
    for (
      let p = content.indexOf("|", offset + 1);
      p !== -1 && p < i;
      p = content.indexOf("|", p + 1)
    ) {
      if (content[p - 1] !== "\\") return -1;
    }
  }
  return i;
}

/** caps LaTeX delimiter bodies at 4,096 characters so repeated incomplete openers stay linear while streaming. */
const CONVERT_LATEX_DELIM_RE =
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
  CONVERT_LATEX_DELIM_RE.lastIndex = 0;
  while ((match = CONVERT_LATEX_DELIM_RE.exec(content)) !== null) {
    const matchEnd = match.index + match[0].length;
    // Skip if either delimiter is inside code or a link destination: an opener
    // outside such a zone must not consume a closer inside one and rewrite
    // across the boundary. Resume right after this opener (not past the whole
    // match) so a valid span that this match spanned across (a stray code `\(`
    // paired with a real closer) is still found on the next pass, not swallowed.
    if (inSkipZone(match.index) || inSkipZone(matchEnd - 1)) {
      CONVERT_LATEX_DELIM_RE.lastIndex = match.index + 1;
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

/** converts bracketed LaTeX and protects currency or shell variables from single-dollar math. */
export function preprocessLaTeX(content: string, isStreaming = false): string {
  const { text, mathRegions } = convertLatexDelimiters(content);

  if (!text.includes("$")) return text;

  let proseBudget = VARIABLE_PROSE_BUDGET;
  const codeRegions = findCodeBlockRegions(text);
  const linkRegions = mergeRegions(
    findLinkDestinationRegions(text),
    text.includes("<") || text.includes("://") || text.includes("www.")
      ? findAutolinkRegions(text)
      : [],
  );
  const rawTextRegions = findRawTextRegions(text);
  const shellQuoteStates = SHELL_QUOTE_CONTEXT_CANDIDATE_RE.test(text)
    ? findShellQuoteStates(text, mergeRegions(codeRegions, rawTextRegions))
    : new Map<number, number>();
  const legacyArithmeticCloses = text.includes("$[")
    ? findLegacyArithmeticCloses(text)
    : undefined;
  const inCode = regionCursor(codeRegions);
  const inLink = regionCursor(linkRegions);
  const inMath = regionCursor(mathRegions);
  const inRawText = regionCursor(rawTextRegions);
  let closer = -1;
  let lineStart = 0;
  let nextNewline = text.indexOf("\n");
  let tableRow: boolean | null = null;

  const rewrite = (offset: number): string => {
    // evaluate all cursors so each advances past `offset`.
    const code = inCode(offset);
    const link = inLink(offset);
    const math = inMath(offset);
    const rawText = inRawText(offset);
    if (code || link || rawText) {
      return "$";
    }
    // preserve converted numeric math to avoid currency re-escaping
    if (math) {
      return "$";
    }
    if (isShellPromptMarker(text, offset)) {
      return "\\$";
    }
    const digit = text.charCodeAt(offset + 1) - 48;
    CURRENCY_REGEX.lastIndex = offset + 1;
    const currency = digit >= 0 && digit <= 9 && CURRENCY_REGEX.test(text);
    if (
      currency &&
      !hasInlineMathCloser(
        text,
        offset,
        mathRegions,
        shellQuoteStates,
        legacyArithmeticCloses,
        isStreaming,
      )
    ) {
      return "\\$";
    }
    if (offset === closer) {
      return "$";
    }
    while (nextNewline !== -1 && nextNewline < offset) {
      lineStart = nextNewline + 1;
      nextNewline = text.indexOf("\n", lineStart);
      tableRow = null;
    }
    if (tableRow === null) {
      TABLE_ROW_RE.lastIndex = lineStart;
      tableRow = TABLE_ROW_RE.test(text);
    }
    const next = findInlineMathCloser(
      text,
      offset,
      lineStart,
      nextNewline,
      tableRow,
    );
    if (
      !currency &&
      next !== -1 &&
      startsLikeVariable(text, offset + 1) &&
      // a closer at the end of a streaming reply may still be followed by a name.
      (next + 1 === text.length
        ? isStreaming
        : NEW_TOKEN_RE.test(text[next + 1])) &&
      next - offset - 1 <= MAX_VARIABLE_PROSE_SPAN &&
      (proseBudget -= next - offset - 1) >= 0 &&
      VARIABLE_PROSE_RE.test(text.slice(offset + 1, next))
    ) {
      return VARIABLE_DOLLAR;
    }
    closer = next;
    return "$";
  };

  // a single `$`: not escaped, not part of `$$`. A plain scan, since a regex replace
  // callback per dollar dominated the streaming re-render cost.
  const parts: string[] = [];
  let last = 0;
  for (
    let offset = text.indexOf("$");
    offset !== -1;
    offset = text.indexOf("$", offset + 1)
  ) {
    const prev = text[offset - 1];
    if (offset === closer) continue;
    const numericOffset = numericShellParameterStart(text, offset);
    if (
      numericOffset !== -1 &&
      startsShellExpansion(
        text,
        offset + 1,
        numericOffset,
        shellQuoteStates.get(numericOffset) ?? 0,
        legacyArithmeticCloses,
        isStreaming,
      ) &&
      !inCode(offset) &&
      !inLink(offset) &&
      !inMath(offset) &&
      !inRawText(offset)
    ) {
      const end = shellExpansionEnd(text, offset);
      parts.push(
        text.slice(last, offset),
        encodeMarkdownLiteral(text.slice(offset, end)),
      );
      last = end;
      offset = end - 1;
      continue;
    }
    if (prev === "\\" || prev === "$" || text[offset + 1] === "$") continue;
    const out = rewrite(offset);
    if (out !== "$") {
      const marker = text.charCodeAt(offset - 1);
      if (
        out === "\\$" &&
        (marker === 42 || marker === 95) &&
        hasShellCommandPrefix(text, offset - 1)
      ) {
        parts.push(
          text.slice(last, offset - 1),
          encodeMarkdownLiteral(text[offset - 1]),
          out,
        );
      } else {
        parts.push(text.slice(last, offset), out);
      }
      last = offset + 1;
    }
  }
  if (last === 0) return text;
  parts.push(text.slice(last));
  return parts.join("");
}
