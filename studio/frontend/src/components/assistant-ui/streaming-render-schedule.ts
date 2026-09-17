// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import remend from "remend";
import { type BlockProps, parseMarkdownIntoBlocks } from "streamdown";

// How far behind the live edge a block has to be before it can be retained.
// The block list interleaves "\n\n" separators, so this is about four
// paragraphs of slack for a construct that a later line can still reinterpret.
const ROLLBACK_BLOCKS = 8;
// A marker the reply never closes leaves the tail growing with nothing to retain, and the boundary
// scan is then paid on top of the full repair it was meant to replace. Give up at a character
// budget, since characters are what that scan costs, and the spending grows with the square of the
// tail. Measured on a marker that closes 40,000 characters later: median cost against the
// full-document path is +73% at 32,768 and +1.6% at 8,192, which is still ~1,300 words of slack.
const STALLED_TAIL_CHARACTERS = 8_192;
// remend repairs the trailing incomplete construct, but reaches that decision with whole-string
// passes: one walks the text once per `[` looking for an unterminated link, another escapes
// comparison operators line by line. Inside an unterminated fence every such character is literal
// and every pass declines, so those walks cost the length of the fence and change nothing. The
// fence also lexes into a single block, so nothing is ever retained and a reply costs the square of
// it. Repair the head plus a window of the fence body instead and splice the untouched middle back
// in; the window carries the last line and the one before it, which is all the trailing repairs read.
const OPEN_FENCE_REPAIR_WINDOW = 4_096;
// A plain two-line probe. When remend leaves the head alone with ordinary text after it, no marker
// is pending for it to close, so it cannot append anything beyond the window either; without it a
// marker left open before the fence gets its closer at the wrong offset. Its verdict is a property
// of the HEAD while remend decides from the whole string, so on its own it is not enough: text
// later in the body can flip a global parity the probe assumed. The body marker refusal below is
// what closes that gap.
const OPEN_FENCE_PROBE = "\nq\n";
// An index can only be resolved once the two characters after it are known, so
// the scan keeps that many back from the live edge and re-reads them.
const FENCE_SCAN_MARGIN = 2;
// Balanced marker prefixes preserve the whole-document facts that remend uses
// to decide how an incomplete tail should close, without changing parity.
const MULTILINE_KATEX_CONTEXT = "$$\n$$\n\n";
const BOLD_CONTEXT = "**x**\n\n";
const SINGLE_ASTERISK_CONTEXT = "*x*\n\n";
const SINGLE_UNDERSCORE_CONTEXT = "_x_\n\n";
const INLINE_CODE_ASTERISK_CONTEXT = "`a *b* c`\n\n";
const INLINE_CODE_UNDERSCORE_CONTEXT = "`a _b_ c`\n\n";
// The one context that is deliberately UNBALANCED, because the fact it carries is an open region
// rather than a marker behind the boundary. remend has no other way to enter the state: `\(` is the
// only transition into inline LaTeX. The blank line after it is load bearing twice over: it keeps
// the opener off the tail's first line, which every line-oriented repair would otherwise read as
// part of that line, and it puts a newline between the tail and the `(`, which is where remend's
// backwards scan for a link destination stops. The region survives it, since remend's math scan has
// no newline rule. There is deliberately no `\[` twin; see `hasUncarriableMath`.
const INLINE_LATEX_CONTEXT = "\\(\n\n";
const FOOTNOTE_REFERENCE_RE = /\[\^[\w-]{1,200}\](?!:)/;
const FOOTNOTE_DEFINITION_RE = /\[\^[\w-]{1,200}\]:/;
const LINK_DEFINITION_RE = /\[(?:\\.|[^\]\n\\]){1,200}\]:/;
// Inside a block marked did not lex as code, the container markers and their indentation
// have already been accounted for, so the label may sit behind any mix of them.
// A block quote marker may be followed by nothing, but a list marker needs whitespace after
// it or no list opens -- `-[label]:` is ordinary prose, not a bullet holding a definition.
const LINK_DEFINITION_LINE_RE =
  /^[ \t]*(?:(?:>[ \t]*)|(?:(?:[-*+]|\d{1,9}[.)])[ \t]+))*\[(?:\\.|[^\]\n\\]){1,200}\]:/m;
// The two block shapes whose body is literal code: an opening fence, and an indent that
// reaches column four -- four spaces, or a tab, which advances to the same column.
const CODE_BLOCK_RE = /^(?: {0,3}(?:`{3,}|~{3,})|(?: {4,}| {0,3}\t)[ \t]*[^ \t\r\n])/;
// A backtick opener may not carry a backtick in its info string, or it is not a fence at all
// and the line is ordinary prose -- which is where a reference can still be waiting. Tilde
// openers have no such rule, so their info string is left alone.
const BACKTICK_OPENER_RE = /^ {0,3}`{3,}([^\n]*)/;

function isCodeBlock(block: string): boolean {
  if (!CODE_BLOCK_RE.test(block)) {
    return false;
  }
  const backtick = BACKTICK_OPENER_RE.exec(block);
  return backtick === null || !backtick[1].includes("`");
}
const LINK_REFERENCE_RE =
  /!?\[(?:\\.|[^\]\n\\]){1,200}\]\[(?:\\.|[^\]\n\\]){0,200}\]/;
// Still the first line of a single block, for `updateLinkDefinitionParity` below.
const FENCED_CODE_BLOCK_RE = /^ {0,3}(?:```|~~~)/;
const WORD_CHARACTER_RE = /[\p{L}\p{N}_]/u;
const HTML_TAG_START_RE = /[a-zA-Z/]/;

// One split per reply, shared by all three exported entry points. markdown-text.tsx asks for
// the key and then hands `parseMarkdownIntoRenderableBlocks` to Streamdown, which calls it with
// the same string, so a single slot is all the reuse this needs -- and it keeps the blocks path
// paying for exactly the one split it already paid for before any of this existed.
let splitMarkdown: string | null = null;
let splitBlocks: readonly string[] = [];

function blocksOf(markdown: string): readonly string[] {
  if (splitMarkdown !== markdown) {
    splitMarkdown = markdown;
    splitBlocks = parseMarkdownIntoBlocks(markdown);
  }
  return splitBlocks;
}

// Which replies have to be lexed in one piece.
//
// marked keeps link reference definitions in one document-wide map and emits no token for a
// label it has already seen, so a `[label][ref]` and its `[ref]: url` must reach the lexer
// together or the reference survives as literal text. The question is therefore whether a real
// definition exists outside code -- and the earlier answer, a hand-rolled scan for fences,
// containers and raw HTML, kept disagreeing with marked at the seams: nested fences, the seven
// HTML block shapes, list continuation indentation, lone-CR line endings.
//
// marked has already resolved every one of those by the time it hands back blocks, so the split
// is the answer rather than something to re-derive. A fenced or indented block is code; anything
// else is prose, and a definition line anywhere in the prose counts.
//
// Being wrong is not symmetric, which is why the residual imprecision sits where it does. Saying
// `blocks` when the reply needed one document splits the pair apart and loses content. Saying
// `document` when blocks would have done only costs that reply its per-code-block Copy and
// Download controls -- which is what this path did for EVERY reply containing a `]:` substring
// before. See tests/link-definition-oracle.test.ts, which pins the first case exhaustively.
function documentProse(markdown: string): string | null {
  if (!LINK_REFERENCE_RE.test(markdown) || !LINK_DEFINITION_RE.test(markdown)) {
    return null;
  }
  const prose = blocksOf(markdown)
    .filter((block) => !isCodeBlock(block))
    .join("\n");
  return LINK_DEFINITION_LINE_RE.test(prose) && LINK_REFERENCE_RE.test(prose)
    ? prose
    : null;
}

export function markdownRenderScope(markdown: string): "blocks" | "document" {
  return documentProse(markdown) === null ? "blocks" : "document";
}

export function markdownRenderKey(markdown: string): string {
  const prose = documentProse(markdown);
  if (prose === null) {
    return "blocks";
  }
  return `document:${prose
    .split("\n")
    .filter((line) => LINK_DEFINITION_LINE_RE.test(line))
    .join("\n")}`;
}

export function parseMarkdownIntoRenderableBlocks(markdown: string): string[] {
  return markdownRenderScope(markdown) === "document"
    ? [markdown]
    : [...blocksOf(markdown)];
}

// Where remend believes the emphasis scan sits with respect to math.
//
// remend 1.3.0 kept two booleans here, one for `$...$` and one for `$$...$$`, and knew nothing
// about the LaTeX bracket delimiters. That is the bug the 1.3.1 bump fixes: on
// `where \( \delta_{r} = 1 \) holds.` the subscript `_` was counted as an unmatched emphasis
// marker and "completed" with a second `_` appended to a finished document. 1.3.1 replaced the pair
// with the five-state machine below and skips every marker inside any of the four open states. The
// dollar half is unchanged, so the two booleans map onto `inlineDollar` and `blockDollar` exactly.
// Unsloth mirrors it because `RepairParity` is a hand-written copy of remend's marker rules.
type EmphasisMathState =
  | "none"
  | "inlineLatex"
  | "blockLatex"
  | "inlineDollar"
  | "blockDollar";

type RepairParity = {
  bold: boolean;
  boldCandidate: boolean;
  boldFence: boolean;
  bracketDepth: number;
  linkDefinition: boolean;
  doubleUnderscore: boolean;
  emphasisInlineCode: boolean;
  emphasisMath: EmphasisMathState;
  // Scan state, not a parity: whether the last asterisk that counted had a word
  // character on both sides. See `countsAsSingleAsterisk`. It is cleared by any
  // non word character outside a fence, and a block separator is two of them,
  // so it is always false at a commit boundary and never needs carrying.
  inWordAsteriskChain: boolean;
  firstBoldOrSingleUnderscore: "bold" | "singleUnderscore" | null;
  singleAsterisk: boolean;
  singleAsteriskCandidate: boolean;
  firstSingleAsteriskCandidate: "inlineCode" | "normal" | null;
  singleUnderscore: boolean;
  singleUnderscoreCandidate: boolean;
  firstSingleUnderscoreCandidate: "inlineCode" | "normal" | null;
  tripleAsterisk: boolean;
  displayMathInlineCode: boolean;
  inlineCode: boolean;
  inlineMathInlineCode: boolean;
  strikethrough: boolean;
  displayMath: boolean;
  inlineMath: boolean;
};

// The subset of the math states a retained prefix can end in. Three of the four
// open states are excluded on purpose, because `hasNeutralRepairParity` refuses
// to commit a block while any of them is open; see `hasUncarriableMath`.
type RetainedLatexState = "none" | "inlineLatex";

// `latex` is where the retained prefix left remend's math scan. Everything else
// starts neutral because a boundary is only taken when it is neutral.
const createRepairParity = (
  latex: RetainedLatexState = "none",
): RepairParity => ({
  bold: false,
  boldCandidate: false,
  boldFence: false,
  bracketDepth: 0,
  linkDefinition: false,
  doubleUnderscore: false,
  emphasisInlineCode: false,
  emphasisMath: latex,
  inWordAsteriskChain: false,
  firstBoldOrSingleUnderscore: null,
  singleAsterisk: false,
  singleAsteriskCandidate: false,
  firstSingleAsteriskCandidate: null,
  singleUnderscore: false,
  singleUnderscoreCandidate: false,
  firstSingleUnderscoreCandidate: null,
  tripleAsterisk: false,
  displayMathInlineCode: false,
  inlineCode: false,
  inlineMathInlineCode: false,
  strikethrough: false,
  displayMath: false,
  inlineMath: false,
});

// Marked keeps link reference definitions in one document-wide map and emits no token for a label
// it has already seen, so a definition retained while its twin is still live would be lexed apart
// and shown as a literal line. Keeping every definition in the live tail makes the two lexes agree.
// Marked reads a fenced block as code, so those do not count.
function updateLinkDefinitionParity(parity: RepairParity, text: string): void {
  if (!FENCED_CODE_BLOCK_RE.test(text) && LINK_DEFINITION_RE.test(text)) {
    parity.linkDefinition = true;
  }
}

const isTripleBacktick = (text: string, index: number): boolean =>
  (index >= 2 && text.slice(index - 2, index + 1) === "```") ||
  (index >= 1 && text.slice(index - 1, index + 2) === "```") ||
  text.slice(index, index + 3) === "```";

const isWordCharacter = (character: string | undefined): boolean =>
  character !== undefined && WORD_CHARACTER_RE.test(character);

function findLinkDestinationStart(text: string, index: number): number {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    const character = text[cursor];
    if (character === ")" || character === "\n") {
      return -1;
    }
    if (character === "(") {
      return cursor > 0 && text[cursor - 1] === "]" ? cursor : -1;
    }
  }
  return -1;
}

function hasLinkDestinationEnd(text: string, index: number): boolean {
  for (let cursor = index; cursor < text.length; cursor += 1) {
    if (text[cursor] === ")") {
      return true;
    }
    if (text[cursor] === "\n") {
      return false;
    }
  }
  return false;
}

const isWithinLinkDestination = (text: string, index: number): boolean =>
  findLinkDestinationStart(text, index) >= 0 &&
  hasLinkDestinationEnd(text, index);

function isWithinHtmlTag(text: string, index: number): boolean {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (text[cursor] === ">") {
      return false;
    }
    if (text[cursor] === "<") {
      return HTML_TAG_START_RE.test(text[cursor + 1] ?? "");
    }
    if (text[cursor] === "\n") {
      return false;
    }
  }
  return false;
}

// A `\` opens or closes a LaTeX region when the character after it is the matching bracket, and
// does nothing otherwise. Both regions only open from `none` and only close from their own state,
// so a `\]` inside a `\(...\)` span is inert. Returns null when this backslash is not a delimiter,
// which is remend's signal to read the escaped character normally.
function latexMathTransition(
  state: EmphasisMathState,
  next: string | undefined,
): EmphasisMathState | null {
  if (next === "[" && state === "none") {
    return "blockLatex";
  }
  if (next === "]" && state === "blockLatex") {
    return "none";
  }
  if (next === "(" && state === "none") {
    return "inlineLatex";
  }
  if (next === ")" && state === "inlineLatex") {
    return "none";
  }
  return null;
}

// `$$` toggles the block state from wherever it was, which is how an unclosed `$...$` is swallowed
// by a following display block. A lone `$` cannot close a block one and cannot open anything inside
// one, so `blockDollar` absorbs it. This half is byte for byte remend 1.3.0's rule.
function dollarMathTransition(
  state: EmphasisMathState,
  isDouble: boolean,
): EmphasisMathState {
  if (isDouble) {
    return state === "blockDollar" ? "none" : "blockDollar";
  }
  if (state === "blockDollar") {
    return state;
  }
  return state === "inlineDollar" ? "none" : "inlineDollar";
}

const isLatexMathState = (state: EmphasisMathState): boolean =>
  state === "inlineLatex" || state === "blockLatex";

// remend recomputes this from the start of the document for every marker it looks at; this runs
// once, in step with the emphasis scan, and reaches the same state at every offset. It runs before
// the fence handling below because remend's math scan is a separate pass that knows nothing about
// fenced code. Returns the index to resume from, one past a two-character delimiter.
function updateEmphasisMathParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (text[index] === "\\") {
    // The escape is checked first and wins: `\$` is a literal dollar and never
    // a delimiter, whatever state the scan is in.
    if (text[index + 1] === "$") {
      return index + 1;
    }
    const transitioned = latexMathTransition(
      parity.emphasisMath,
      text[index + 1],
    );
    if (transitioned === null) {
      return index;
    }
    parity.emphasisMath = transitioned;
    return index + 1;
  }
  // Inside `\(...\)` or `\[...\]` a dollar is ordinary text, so it neither
  // opens a dollar region nor consumes the character after it. remend 1.3.1
  // added this guard along with the LaTeX states; 1.3.0 had no state for it to
  // guard against.
  if (text[index] !== "$" || isLatexMathState(parity.emphasisMath)) {
    return index;
  }
  const isDouble = text[index + 1] === "$";
  parity.emphasisMath = dollarMathTransition(parity.emphasisMath, isDouble);
  return isDouble ? index + 1 : index;
}

// Absent, or one of the three characters remend treats as a boundary next to a
// marker. remend spells this as `!character || isWhitespace(character)`, where
// an out of range read yields the empty string; here it yields undefined.
const isBoundaryCharacter = (character: string | undefined): boolean =>
  character === undefined ||
  character === " " ||
  character === "\t" ||
  character === "\n";

// remend's skip list for the single asterisk counter. 1.3.1 dropped ONE clause, the one skipping an
// asterisk with a word character on each side, and moved that case into `countsAsSingleAsterisk`
// below. Everything else here is byte for byte what 1.3.0 had.
function shouldSkipAsterisk(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  if (previous === "\\" || parity.emphasisMath !== "none") {
    return true;
  }
  if (previous !== "*" && next === "*") {
    // An out of range third character reads as the empty string in remend,
    // which is not an asterisk, so the marker is skipped either way.
    return text[index + 2] !== "*";
  }
  if (previous === "*") {
    return true;
  }
  return isBoundaryCharacter(previous) && isBoundaryCharacter(next);
}

// Does this asterisk move the single asterisk parity? This is remend 1.3.1's rule and NOT 1.3.0's.
// 1.3.0 dropped every asterisk with a word character on both sides, so `*foo*bar` counted one
// marker, came out odd, and the repair closed it. 1.3.1 counts the in-word one as soon as the count
// is already odd or an in-word chain is running, so the same text counts two and nothing is
// appended: remend of `*foo*bar` followed by `~~s` appends a stray `*` under 1.3.0 and nothing
// under 1.3.1. `inWordAsteriskChain` is what lets `*foo*bar*baz` keep counting after the second
// marker; any non word character outside a fence clears it.
function countsAsSingleAsterisk(
  parity: RepairParity,
  text: string,
  index: number,
): { counts: boolean; inWordChain: boolean } {
  const previous = text[index - 1];
  const next = text[index + 1];
  const inWord = isWordCharacter(previous) && isWordCharacter(next);
  // "Text" here means present and not whitespace, which is remend's test for whether a marker has
  // something on that side to attach to. It is weaker than "word character": punctuation counts.
  const previousIsText = !isBoundaryCharacter(previous);
  const nextIsText = !isBoundaryCharacter(next);
  if (inWord && !parity.singleAsterisk && !parity.inWordAsteriskChain) {
    return { counts: false, inWordChain: false };
  }
  if ((previousIsText && parity.singleAsterisk) || nextIsText) {
    return { counts: true, inWordChain: inWord };
  }
  return { counts: false, inWordChain: false };
}

function isSingleAsteriskCandidate(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  if (
    previous === "\\" ||
    previous === "*" ||
    next === "*" ||
    parity.emphasisMath !== "none"
  ) {
    return false;
  }
  // 1.3.1 added a third skip to remend's search for the first unmatched
  // asterisk: a marker with nothing but whitespace or the end of the document
  // after it cannot OPEN emphasis, so it is passed over whatever sits before
  // it. That subsumes the old "boundary on both sides" clause, which is why
  // only the next character is read here now.
  return !(
    isBoundaryCharacter(next) ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

function countsAsSingleUnderscore(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  return !(
    previous === "\\" ||
    parity.emphasisMath !== "none" ||
    isWithinLinkDestination(text, index) ||
    isWithinHtmlTag(text, index) ||
    previous === "_" ||
    next === "_" ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

function isSingleUnderscoreCandidate(
  parity: RepairParity,
  text: string,
  index: number,
): boolean {
  const previous = text[index - 1];
  const next = text[index + 1];
  return !(
    previous === "\\" ||
    previous === "_" ||
    next === "_" ||
    parity.emphasisMath !== "none" ||
    isWithinLinkDestination(text, index) ||
    (isWordCharacter(previous) && isWordCharacter(next))
  );
}

// Remend decides these closers from marker parity over the whole document, so a
// retained prefix must end with neutral parity: repairing the tail alone could
// otherwise add or omit a closer that a full repair would place differently.
function updateAsteriskParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (isSingleAsteriskCandidate(parity, text, index)) {
    parity.singleAsteriskCandidate = true;
    parity.firstSingleAsteriskCandidate ??= parity.emphasisInlineCode
      ? "inlineCode"
      : "normal";
  }
  if (!shouldSkipAsterisk(parity, text, index)) {
    const decision = countsAsSingleAsterisk(parity, text, index);
    if (decision.counts) {
      parity.singleAsterisk = !parity.singleAsterisk;
      parity.inWordAsteriskChain = decision.inWordChain;
    }
  }
  if (text[index + 1] === "*") {
    parity.boldCandidate = true;
    parity.firstBoldOrSingleUnderscore ??= "bold";
    parity.bold = !parity.bold;
    return index + 1;
  }
  return index;
}

function updateUnderscoreParity(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (isSingleUnderscoreCandidate(parity, text, index)) {
    parity.singleUnderscoreCandidate = true;
    parity.firstSingleUnderscoreCandidate ??= parity.emphasisInlineCode
      ? "inlineCode"
      : "normal";
    parity.firstBoldOrSingleUnderscore ??= "singleUnderscore";
  }
  if (text[index + 1] === "_") {
    parity.doubleUnderscore = !parity.doubleUnderscore;
    return index + 1;
  }
  if (countsAsSingleUnderscore(parity, text, index)) {
    parity.singleUnderscore = !parity.singleUnderscore;
  }
  return index;
}

// Remend finds the marker that orders its closers with a raw indexOf("**") but
// counts pairs only outside fenced code, so a `**` in a fence has to seed the
// bold context without reaching the fence-aware counter in updateAsteriskParity.
function recordBoldMarker(
  parity: RepairParity,
  text: string,
  index: number,
): void {
  if (text[index] === "*" && text[index + 1] === "*") {
    parity.boldCandidate = true;
    parity.firstBoldOrSingleUnderscore ??= "bold";
  }
}

// Remend completes a dangling link by appending to the end of the whole
// document, so a retained block still holding an unmatched bracket would move
// that completion out of the tail. Brackets in code do not count, as in remend.
function updateBracketDepth(parity: RepairParity, character: string): void {
  if (character === "[") {
    parity.bracketDepth += 1;
  } else if (character === "]" && parity.bracketDepth > 0) {
    parity.bracketDepth -= 1;
  }
}

// Remend consumes an escaped backtick before testing for a fence, so a fence
// that starts one character after a backslash is not a fence. Returns the index
// to resume from, or the same index when neither applies.
function skipEscapeOrFence(
  parity: RepairParity,
  text: string,
  index: number,
): number {
  if (text[index] === "\\" && text[index + 1] === "`") {
    return index + 1;
  }
  if (text.slice(index, index + 3) === "```") {
    parity.boldFence = !parity.boldFence;
    return index + 2;
  }
  return index;
}

// remend's asterisk counter clears the in-word chain on any character that is
// neither an asterisk nor a word character, and only while it believes it is
// outside a fence, where it stops reading characters at all.
function clearInWordAsteriskChain(
  parity: RepairParity,
  character: string | undefined,
): void {
  if (!parity.boldFence && character !== "*" && !isWordCharacter(character)) {
    parity.inWordAsteriskChain = false;
  }
}

function updateEmphasisParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    // The chain is cleared at the top of the loop, before any of the multi character skips below,
    // because every one of those skips starts on a backslash, a dollar or a backtick, and all three
    // are non word characters that clear the chain in remend as well. Reading only the first
    // character of a skipped pair therefore reaches the same state remend does.
    clearInWordAsteriskChain(parity, text[index]);
    index = updateEmphasisMathParity(parity, text, index);
    const skipped = skipEscapeOrFence(parity, text, index);
    if (skipped !== index) {
      index = skipped;
      continue;
    }
    recordBoldMarker(parity, text, index);
    if (parity.boldFence) {
      continue;
    }
    if (text[index] === "`") {
      parity.emphasisInlineCode = !parity.emphasisInlineCode;
      continue;
    }
    if (!parity.emphasisInlineCode) {
      updateBracketDepth(parity, text[index]);
    }
    if (text[index] === "*") {
      index = updateAsteriskParity(parity, text, index);
      continue;
    }
    if (text[index] === "_") {
      index = updateUnderscoreParity(parity, text, index);
    }
  }
}

function updateTripleAsteriskParity(parity: RepairParity, text: string): void {
  let inFence = false;
  let runLength = 0;
  const finishRun = () => {
    if (Math.floor(runLength / 3) % 2 === 1) {
      parity.tripleAsterisk = !parity.tripleAsterisk;
    }
    runLength = 0;
  };
  for (let index = 0; index < text.length; index += 1) {
    if (text.slice(index, index + 3) === "```") {
      finishRun();
      inFence = !inFence;
      index += 2;
    } else if (!inFence && text[index] === "*") {
      runLength += 1;
    } else {
      finishRun();
    }
  }
  finishRun();
}

function updateInlineCodeParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "\\" && text[index + 1] === "`") {
      index += 1;
      continue;
    }
    if (text[index] === "`" && !isTripleBacktick(text, index)) {
      parity.inlineCode = !parity.inlineCode;
    }
  }
}

function updateStrikethroughParity(parity: RepairParity, text: string): void {
  const strikeMarkers = text.match(/~~/g)?.length ?? 0;
  if (strikeMarkers % 2 === 1) {
    parity.strikethrough = !parity.strikethrough;
  }
}

// Remend's display-math scan stops one character early: it only reads whole
// documents, where the last character cannot open a pair. This one runs per
// retained block, whose boundaries are interior, so it counts to text.length.
function updateDisplayMathParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "`" && !isTripleBacktick(text, index)) {
      parity.displayMathInlineCode = !parity.displayMathInlineCode;
      continue;
    }
    if (
      !parity.displayMathInlineCode &&
      text[index] === "$" &&
      text[index + 1] === "$"
    ) {
      parity.displayMath = !parity.displayMath;
      index += 1;
    }
  }
}

function updateInlineMathParity(parity: RepairParity, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "\\") {
      index += 1;
      continue;
    }
    if (text[index] === "`" && !isTripleBacktick(text, index)) {
      parity.inlineMathInlineCode = !parity.inlineMathInlineCode;
      continue;
    }
    if (parity.inlineMathInlineCode || text[index] !== "$") {
      continue;
    }
    if (text[index + 1] === "$") {
      index += 1;
    } else {
      parity.inlineMath = !parity.inlineMath;
    }
  }
}

function updateRepairParity(parity: RepairParity, text: string): void {
  updateLinkDefinitionParity(parity, text);
  updateEmphasisParity(parity, text);
  updateTripleAsteriskParity(parity, text);
  updateInlineCodeParity(parity, text);
  updateStrikethroughParity(parity, text);
  updateDisplayMathParity(parity, text);
  updateInlineMathParity(parity, text);
}

// An open math region a boundary may NOT sit inside, because the tail repair cannot be told about
// it. Only `inlineLatex` is missing, and the asymmetry is a property of the openers:
//
//   `$` and `$$` are counted by the katex and inlineKatex repairs as well as by the emphasis scan,
//   so a bare dollar in the context prefix would make those two close a delimiter the retained
//   prefix already holds. This is the refusal 1.3.0's two booleans always had.
//
//   `\[` is the only way into `blockLatex` and carries a `[` that remend's link repair reads: that
//   repair walks backwards over every `[` outside code and completes the first one whose `]` is
//   missing, so an opener standing in for a `\[` several blocks back turns a tail with no bracket
//   of its own into one ending `](streamdown:incomplete-link)`, while the whole document is left
//   alone. Found by differential fuzzing against a full remend() of the same text.
//
// `inlineLatex` has neither problem: `(` is read only by the backwards scan for a link destination,
// which stops at the first newline, and the context prefix ends with a blank line.
const hasUncarriableMath = (parity: RepairParity): boolean =>
  parity.emphasisMath !== "none" && parity.emphasisMath !== "inlineLatex";

const hasNeutralRepairParity = (parity: RepairParity): boolean =>
  ![
    parity.bracketDepth > 0,
    parity.linkDefinition,
    parity.bold,
    parity.boldFence,
    parity.emphasisInlineCode,
    parity.doubleUnderscore,
    hasUncarriableMath(parity),
    parity.singleAsterisk,
    parity.singleUnderscore,
    parity.tripleAsterisk,
    parity.displayMathInlineCode,
    parity.inlineCode,
    parity.inlineMathInlineCode,
    parity.strikethrough,
    parity.displayMath,
    parity.inlineMath,
  ].includes(true);

export type IncrementalMarkdownRender = {
  markdown: string;
  parseMarkdownIntoBlocks: (markdown: string) => string[];
};

// Marker facts the retained prefix carries into the tail repair.
type RetainedContext = {
  multilineKatex: boolean;
  bold: boolean;
  singleAsterisk: boolean;
  singleUnderscore: boolean;
  latex: RetainedLatexState;
  firstSingleAsterisk: "inlineCode" | "normal" | null;
  firstSingleUnderscore: "inlineCode" | "normal" | null;
  firstBoldOrSingleUnderscore: "bold" | "singleUnderscore" | null;
};

const createRetainedContext = (): RetainedContext => ({
  multilineKatex: false,
  bold: false,
  singleAsterisk: false,
  singleUnderscore: false,
  latex: "none",
  firstSingleAsterisk: null,
  firstSingleUnderscore: null,
  firstBoldOrSingleUnderscore: null,
});

const singleUnderscoreContext = (context: RetainedContext): string => {
  if (!context.singleUnderscore) {
    return "";
  }
  return context.firstSingleUnderscore === "inlineCode"
    ? INLINE_CODE_UNDERSCORE_CONTEXT
    : SINGLE_UNDERSCORE_CONTEXT;
};

const singleAsteriskContext = (context: RetainedContext): string => {
  if (!context.singleAsterisk) {
    return "";
  }
  return context.firstSingleAsterisk === "inlineCode"
    ? INLINE_CODE_ASTERISK_CONTEXT
    : SINGLE_ASTERISK_CONTEXT;
};

const emphasisContext = (context: RetainedContext): string => {
  const bold = context.bold ? BOLD_CONTEXT : "";
  const underscore = singleUnderscoreContext(context);
  return context.firstBoldOrSingleUnderscore === "singleUnderscore"
    ? underscore + bold
    : bold + underscore;
};

const latexContext = (context: RetainedContext): string =>
  context.latex === "inlineLatex" ? INLINE_LATEX_CONTEXT : "";

// Taking the context as a value lets a candidate commit be priced before it is applied, which the
// repeated-Markdown check in update() needs. The LaTeX opener goes LAST, immediately before the
// tail, and the ordering is not cosmetic: everything after it is inside the region it opens, so a
// `_x_` written after it would be a pair of underscores remend declines to count and the emphasis
// context would carry nothing. It is also what the document looks like, since a marker that reached
// the boundary as a candidate was counted, which means it sat outside the region.
function repairContextPrefix(context: RetainedContext): string {
  return (
    emphasisContext(context) +
    singleAsteriskContext(context) +
    (context.multilineKatex ? MULTILINE_KATEX_CONTEXT : "") +
    latexContext(context)
  );
}

function repairTail(tail: string, context: RetainedContext): string {
  const prefix = repairContextPrefix(context);
  if (!prefix) {
    return remend(tail);
  }
  return remend(prefix + tail).slice(prefix.length);
}

// Where remend believes a fence is open. It toggles on any ``` run, wherever on the line that run
// sits, and a backslash escapes the backtick after it, so this deliberately mirrors remend rather
// than CommonMark: a mid-line ``` closes the fence for remend and must close it here too.
type OpenFenceState = {
  index: number;
  fenceOpen: boolean;
  bodyStart: number;
  // The FIRST ` $ or ~ in the current fence body, or -1. Its mere presence disqualifies the splice,
  // so only whether one exists ever matters; the index reads better in a test than a bare boolean.
  // remend does not have one notion of "inside a fence"; it has at least three, and they disagree.
  // `isWithinCodeBlock` honours a backslash before a backtick, the emphasis counters toggle on ```
  // without honouring it, and the inline-code repair just counts /```/g. The opener standing in for
  // the elided text can only reproduce all three when the elided part holds none of the characters
  // any of them counts. The first such character, not the last: a marker in the window would
  // otherwise mask an earlier one in the elided middle, and first is monotone.
  firstBodyMarker: number;
};

const initialOpenFenceState = (): OpenFenceState => ({
  index: 0,
  fenceOpen: false,
  bodyStart: -1,
  firstBodyMarker: -1,
});

function advanceOpenFence(
  state: OpenFenceState,
  text: string,
  limit: number,
): OpenFenceState {
  let { index, fenceOpen, bodyStart, firstBodyMarker } = state;
  while (index < limit) {
    const character = text[index];
    if (character === "\\" && text[index + 1] === "`") {
      // Escaped for `isWithinCodeBlock`, not for the passes that only count
      // ``` runs, so the backtick still counts as a marker.
      if (fenceOpen && firstBodyMarker < 0) {
        firstBodyMarker = index + 1;
      }
      index += 2;
      continue;
    }
    if (
      character === "`" &&
      text[index + 1] === "`" &&
      text[index + 2] === "`"
    ) {
      fenceOpen = !fenceOpen;
      bodyStart = fenceOpen ? index + 3 : -1;
      firstBodyMarker = -1;
      index += 3;
      continue;
    }
    if (
      fenceOpen &&
      firstBodyMarker < 0 &&
      (character === "`" || character === "$" || character === "~")
    ) {
      firstBodyMarker = index;
    }
    index += 1;
  }
  return { index, fenceOpen, bodyStart, firstBodyMarker };
}

// Carries the scan across updates so a growing tail costs the characters that
// arrived, not the characters it holds. Any text that is not an extension of the
// last one rescans from the start, which is what a commit or a rewrite hands it.
class OpenFenceTracker {
  private text = "";
  private resolved = initialOpenFenceState();

  // Where the open fence's body starts and where its first marker sits, or null
  // when the whole-tail repair applies.
  spliceBounds(
    text: string,
  ): { bodyStart: number; firstBodyMarker: number } | null {
    if (!hasPrefix(text, this.text)) {
      this.resolved = initialOpenFenceState();
    }
    const limit = Math.max(0, text.length - FENCE_SCAN_MARGIN);
    if (limit > this.resolved.index) {
      this.resolved = advanceOpenFence(this.resolved, text, limit);
    }
    this.text = text;
    const live = advanceOpenFence(this.resolved, text, text.length);
    if (!live.fenceOpen) {
      return null;
    }
    return {
      bodyStart: live.bodyStart,
      firstBodyMarker: live.firstBodyMarker,
    };
  }
}

// A head the probe found inert contributes nothing but "a fence is open", which one opener
// reproduces. Standing in for it keeps the repair off the head as well as off the elided body.
const OPEN_FENCE_SYNTHETIC_HEAD = "```\n";

// The spliced repair, or null when it cannot be shown to match repairTail. All four refusals fall
// back to repairing the whole tail:
//   a body still shorter than the window has nothing to elide;
//   a final line longer than the window leaves no line boundary to cut on, the only place the
//     splice can start without changing what the trailing repairs read;
//   a PRECEDING line longer than the window puts the cut on the newline that ends it, so the window
//     holds the final line alone. remend's setext repair reads exactly one line back, and the
//     synthetic opener is never blank, so a whitespace-only line elided that way turns a tail
//     ending in `-`, `--`, `=` or `==` into one carrying a zero-width space that repairing the
//     whole tail does not add, and that the copy button would put on the clipboard;
//   a ` $ or ~ ANYWHERE in the body, elided or retained, is a character one of remend's fence
//     notions counts, and the opener cannot stand in for it.
// The last one covers the retained window and not just the elided middle, because the probe that
// licenses the opener reads the head ALONE while remend decides from the whole string: an escaped
// backtick fence in the window flips the global triple-run parity, and a `$$` there moves the math
// parity. Keeping the body free of all three characters is what makes the probe's verdict a
// property of the whole tail rather than of the head it was handed.
function repairOpenFenceTail(
  tail: string,
  bodyStart: number,
  firstBodyMarker: number,
): string | null {
  const cut = tail.indexOf("\n", tail.length - OPEN_FENCE_REPAIR_WINDOW);
  if (
    cut < 0 ||
    cut + 1 <= bodyStart ||
    tail.indexOf("\n", cut + 1) < 0 ||
    firstBodyMarker >= 0
  ) {
    return null;
  }
  const repaired = remend(OPEN_FENCE_SYNTHETIC_HEAD + tail.slice(cut + 1));
  if (!hasPrefix(repaired, OPEN_FENCE_SYNTHETIC_HEAD)) {
    return null;
  }
  return (
    tail.slice(0, cut + 1) + repaired.slice(OPEN_FENCE_SYNTHETIC_HEAD.length)
  );
}

// Every field but `latex` records that something EXISTS behind the boundary and so can only be
// turned on. `latex` is a position rather than a fact: the region the prefix opened can be closed by
// a later commit, so it is taken from the parity outright. That is why each commit point stores its
// own context, which a rewind then restores.
const retainedLatexState = (parity: RepairParity): RetainedLatexState =>
  parity.emphasisMath === "inlineLatex" ? "inlineLatex" : "none";

const advanceContext = (
  context: RetainedContext,
  parity: RepairParity,
  committedText: string,
): RetainedContext => ({
  multilineKatex: context.multilineKatex || committedText.includes("$$"),
  bold: context.bold || parity.boldCandidate,
  singleAsterisk: context.singleAsterisk || parity.singleAsteriskCandidate,
  singleUnderscore:
    context.singleUnderscore || parity.singleUnderscoreCandidate,
  latex: retainedLatexState(parity),
  firstSingleAsterisk:
    context.firstSingleAsterisk ?? parity.firstSingleAsteriskCandidate,
  firstSingleUnderscore:
    context.firstSingleUnderscore ?? parity.firstSingleUnderscoreCandidate,
  firstBoldOrSingleUnderscore:
    context.firstBoldOrSingleUnderscore ?? parity.firstBoldOrSingleUnderscore,
});

type CommitBoundary = {
  count: number;
  length: number;
  // The parity at the boundary, or null when no block can be retained.
  parity: RepairParity | null;
  repairBroke: boolean;
};

// Where one commit left the retained prefix. `advanceContext` only ever adds facts and cannot be
// undone, so each commit's context is stored to allow a rewind to an earlier boundary.
type CommitPoint = {
  blockCount: number;
  length: number;
  context: RetainedContext;
};

// CommonMark counts a line feed, a lone carriage return, and CRLF as the same line ending, and
// reference parsers normalise to LF before parsing, so this cannot change what is rendered. The
// scan runs per frame and costs nothing next to the repair and lex it protects (0.4 us on an 88,000
// character reply); the replace only runs for a reply that carries a carriage return.
function normalizeLineEndings(text: string): string {
  return text.includes("\r") ? text.replace(/\r\n?/g, "\n") : text;
}

/**
 * `a` begins with `b`.
 *
 * `String.prototype.startsWith` is the obvious spelling and is far slower here, because it scans
 * where slicing to the prefix length and comparing lets the engine reject on length and then
 * compare natively. The win is in the PREFIX length, not in how the strings are represented: swept
 * on V8, at an 8 character prefix `startsWith` is FASTER (0.4x), at 1,024 it is 105x slower and at
 * 60,000 it is 250x slower. So do not reach for this helper for short prefixes.
 *
 * Semantically identical: `slice` clamps to the string length.
 */
export const hasPrefix = (a: string, b: string): boolean =>
  a.length >= b.length && a.slice(0, b.length) === b;

function sharedPrefixLength(left: string, right: string): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left.charCodeAt(index) === right.charCodeAt(index)) {
    index += 1;
  }
  return index;
}

// Does `text` end at a blank line, counting the start of the document as one? Unchanged characters
// alone cannot keep a block across a rewrite: Marked reads `paragraph` plus new text as a lazy
// continuation, so an edit that closes up a blank line re-segments the paragraph before it even
// though that paragraph's characters never moved. `\r` counts as line-ending whitespace.
function endsAtBlankLine(text: string, end: number): boolean {
  if (end === 0) {
    return true;
  }
  if (text.charCodeAt(end - 1) !== 10) {
    return false;
  }
  let index = end - 2;
  while (index >= 0) {
    const code = text.charCodeAt(index);
    if (code === 32 || code === 9 || code === 13) {
      index -= 1;
      continue;
    }
    return code === 10;
  }
  return true;
}

// The last blank line at or before `limit`, or 0 for none. An untouched blank line between the
// boundary and the rewrite carries most of the insulation, so the boundary need not land on the
// blank line itself. Not all of it: see `rewindToRewrite`, which also keeps a rollback window.
function lastBlankLineEnd(text: string, limit: number): number {
  for (let end = limit; end > 0; end -= 1) {
    if (endsAtBlankLine(text, end)) {
      return end;
    }
  }
  return 0;
}

// Remend may synthesize closing syntax at the end of an incomplete tail.
// Never retain synthetic or mid-string repaired text. Scan forward once,
// recording the latest exact boundary whose global repair parity is neutral.
function findCommitBoundary(
  tail: string,
  blocks: string[],
  candidateCount: number,
  latex: RetainedLatexState,
): CommitBoundary {
  // The tail does not start at the top of the document, so the scan starts where the retained
  // prefix left remend's math scan. Only the LaTeX state can be anything but neutral there, and it
  // is the one state whose markers the tail repair would otherwise start counting again.
  const parity = createRepairParity(latex);
  const commit: CommitBoundary = {
    count: 0,
    length: 0,
    parity: null,
    repairBroke: false,
  };
  let exactLength = 0;

  for (let index = 0; index < candidateCount; index += 1) {
    const block = blocks[index];
    if (!tail.startsWith(block, exactLength)) {
      commit.repairBroke = true;
      break;
    }
    exactLength += block.length;
    updateRepairParity(parity, block);
    if (hasNeutralRepairParity(parity)) {
      commit.count = index + 1;
      commit.length = exactLength;
      commit.parity = { ...parity };
    }
  }

  return commit;
}

// Streamdown normally repairs and lexes the entire growing reply on every
// update. Retain blocks that are safely behind a rollback window and give
// Streamdown only the active tail. The parser callback puts the retained blocks
// back into its block list, so output and React keys stay identical.
export class IncrementalMarkdownCache {
  private source = "";
  private tail = "";
  private committedBlocks: string[] = [];
  private committedLength = 0;
  private commitPoints: CommitPoint[] = [];
  private context = createRetainedContext();
  private fenceTracker = new OpenFenceTracker();
  private openFenceHead: string | null = null;
  private openFenceHeadInert = false;
  private fullDocumentMode = false;
  private lastMarkdown: string | null = null;
  private droppedRetainedBlocks = false;
  // How often a rewrite discarded the whole retained prefix, and how many characters a rewind
  // handed back to the live tail. Both redo work with an identical result, so time is the only
  // other evidence they happened. Tests read these to hold the rewind path in place.
  private retainedPrefixRebuilds = 0;
  private rewoundCharacters = 0;
  // Bumped only when the Markdown string alone cannot signal a changed render.
  renderGeneration = 0;

  readonly parseMarkdownIntoBlocks = (markdown: string): string[] => [
    ...this.committedBlocks,
    ...parseMarkdownIntoRenderableBlocks(markdown),
  ];

  // Streamdown memoises the whole component on the Markdown string and ignores
  // the parser callback, so the string is the only thing that can schedule a
  // render. Retaining a block can wait for a string that differs, but dropping
  // retained blocks cannot, so that case moves the render identity instead.
  private render(markdown: string): IncrementalMarkdownRender {
    if (this.droppedRetainedBlocks && markdown === this.lastMarkdown) {
      this.renderGeneration += 1;
    }
    this.droppedRetainedBlocks = false;
    this.lastMarkdown = markdown;
    return { markdown, parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks };
  }

  // The repair of a tail sitting inside an open fence, or null to repair the whole tail as before.
  // The head is everything the repair would have to keep: once the probe shows remend leaves it
  // alone, it holds nothing that could change the repair of the body, and it stays fixed until the
  // fence closes, so the probe is paid once per fence rather than once per chunk.
  private repairOpenFence(): string | null {
    const bounds = this.fenceTracker.spliceBounds(this.tail);
    if (bounds === null) {
      return null;
    }
    const head =
      repairContextPrefix(this.context) + this.tail.slice(0, bounds.bodyStart);
    if (head !== this.openFenceHead) {
      this.openFenceHead = head;
      const probed = head + OPEN_FENCE_PROBE;
      this.openFenceHeadInert = remend(probed) === probed;
    }
    return this.openFenceHeadInert
      ? repairOpenFenceTail(this.tail, bounds.bodyStart, bounds.firstBodyMarker)
      : null;
  }

  private resetIncrementalState(markdown: string): void {
    this.droppedRetainedBlocks ||= this.committedBlocks.length > 0;
    this.source = markdown;
    this.tail = markdown;
    this.committedBlocks = [];
    this.committedLength = 0;
    this.commitPoints = [];
    this.context = createRetainedContext();
  }

  private renderFullDocument(markdown: string): IncrementalMarkdownRender {
    this.resetIncrementalState(markdown);
    this.fullDocumentMode = true;
    return this.render(remend(markdown));
  }

  // The text handed to the cache is not always an extension of the last one: `preprocessLaTeX`
  // rewrites an already emitted span when a `\(...\)` closes, a `\[...\]` becomes a `$$` block or a
  // currency `$` turns out not to open math, and a closing fence rewrites its own body. Each edits
  // one span, so rewind to the last commit the rewrite can neither reach nor re-segment rather than
  // discarding the whole prefix. Returns false when nothing survives; mutates nothing before then, so
  // the reset fallback never sees a half-rewound cache.
  private rewindToRewrite(markdown: string): boolean {
    if (this.commitPoints.length === 0) {
      return false;
    }

    // Characters the rewrite left alone. Usually the rewrite is inside the live tail, and `slice`
    // shares the original's characters, so that case costs one native comparison plus a scan of the
    // tail (about to be re-lexed anyway) rather than a scan of the whole reply.
    const committedPrefix = this.source.slice(0, this.committedLength);
    const shared = hasPrefix(markdown, committedPrefix)
      ? this.committedLength +
        sharedPrefixLength(markdown.slice(this.committedLength), this.tail)
      : sharedPrefixLength(markdown, committedPrefix);

    // Unchanged characters alone do not make a boundary safe to keep, so stop
    // at the last blank line the rewrite left intact.
    const safeLimit = lastBlankLineEnd(this.source, shared);

    // A blank line is not a wall either: Marked merges a run of them into one separator block, so
    // inserting a newline re-segments the separator in front of it, and a list or indented code block
    // reopens across one. So demand the same margin the append path requires, ROLLBACK_BLOCKS blocks
    // behind the boundary, measured from the first changed character.
    let blocksBeforeLimit = this.committedBlocks.length;
    let scanned = this.committedLength;
    while (blocksBeforeLimit > 0 && scanned > safeLimit) {
      blocksBeforeLimit -= 1;
      scanned -= this.committedBlocks[blocksBeforeLimit].length;
    }
    const blockLimit = blocksBeforeLimit - ROLLBACK_BLOCKS;
    if (blockLimit <= 0) {
      return false;
    }

    let index = this.commitPoints.length - 1;
    while (
      index >= 0 &&
      (this.commitPoints[index].length > safeLimit ||
        this.commitPoints[index].blockCount > blockLimit)
    ) {
      index -= 1;
    }
    if (index < 0) {
      return false;
    }

    const point = this.commitPoints[index];
    if (point.length < this.committedLength) {
      this.rewoundCharacters += this.committedLength - point.length;
      this.committedBlocks.length = point.blockCount;
      this.committedLength = point.length;
      this.commitPoints.length = index + 1;
      this.droppedRetainedBlocks = true;
    }

    this.context = point.context;
    this.tail = markdown.slice(this.committedLength);
    return true;
  }

  private updateTail(markdown: string): void {
    if (hasPrefix(markdown, this.source)) {
      this.tail += markdown.slice(this.source.length);
    } else if (!this.rewindToRewrite(markdown)) {
      if (this.committedBlocks.length > 0) {
        this.retainedPrefixRebuilds += 1;
      }
      this.resetIncrementalState(markdown);
      this.fullDocumentMode = false;
    }
    this.source = markdown;
  }

  update(rawMarkdown: string): IncrementalMarkdownRender {
    // Every boundary this class finds is a byte offset into the text it was handed, but the blocks it
    // compares against come back from Streamdown with their line endings already normalised. On a CRLF
    // reply the two disagree one block in, nothing is ever committed, and the whole reply re-repairs
    // and re-lexes on every frame. Normalise first so both sides speak LF.
    const markdown = normalizeLineEndings(rawMarkdown);

    // Tokens arrive faster than frames, so the coalescer hands the same text to
    // several renders. Nothing about the result can differ, and repeating the
    // work would be the whole reply again once the document path is in use.
    if (markdown === this.source && this.lastMarkdown !== null) {
      return {
        markdown: this.lastMarkdown,
        parseMarkdownIntoBlocks: this.parseMarkdownIntoBlocks,
      };
    }

    if (this.fullDocumentMode && hasPrefix(markdown, this.source)) {
      this.source = markdown;
      return this.renderFullDocument(markdown);
    }

    this.updateTail(markdown);

    const repaired =
      this.repairOpenFence() ?? repairTail(this.tail, this.context);

    // globally scoped definitions must stay in the same rendered document as
    // their uses, so neither construct can retain an independently parsed prefix.
    // Computed here rather than at the top of the method on purpose: both early returns above
    // -- the coalescer handing the same text to several renders, and a reply already in
    // full-document mode -- answer without it, and the precise scope costs a lex of everything
    // received so far. Reaching this point means the reply is still a retention candidate,
    // which is the only case where the answer is used.
    if (
      FOOTNOTE_REFERENCE_RE.test(repaired) ||
      FOOTNOTE_DEFINITION_RE.test(repaired) ||
      markdownRenderScope(markdown) === "document"
    ) {
      return this.renderFullDocument(markdown);
    }

    const blocks = parseMarkdownIntoBlocks(repaired);

    const candidateCount = Math.max(0, blocks.length - ROLLBACK_BLOCKS);
    if (candidateCount === 0) {
      return this.render(repaired);
    }

    const commit = findCommitBoundary(
      this.tail,
      blocks,
      candidateCount,
      this.context.latex,
    );

    // A mid-string repair can never become a raw prefix on a later append, so
    // make that fallback sticky, and do the same once the tail grows past the
    // budget with nothing to show for it. A temporarily unbalanced marker can
    // close in a later block, so below the budget keep the tail live and retry.
    if (!commit.parity) {
      if (commit.repairBroke || this.tail.length > STALLED_TAIL_CHARACTERS) {
        return this.renderFullDocument(markdown);
      }
      return this.render(repaired);
    }

    const committedText = this.tail.slice(0, commit.length);
    const nextContext = advanceContext(
      this.context,
      commit.parity,
      committedText,
    );
    const nextTail = this.tail.slice(commit.length);
    const nextMarkdown = repairTail(nextTail, nextContext);

    // A repeating reply can leave the tail unchanged once a block is retained.
    // Streamdown would then see the Markdown it already holds and skip the
    // render, so the retained blocks would never be displayed. Keep them in the
    // live tail instead; the next update commits them with a longer string.
    if (nextMarkdown === this.lastMarkdown) {
      return this.render(repaired);
    }

    this.committedBlocks.push(...blocks.slice(0, commit.count));
    this.committedLength += commit.length;
    this.context = nextContext;
    this.tail = nextTail;
    this.commitPoints.push({
      blockCount: this.committedBlocks.length,
      length: this.committedLength,
      context: nextContext,
    });

    return this.render(nextMarkdown);
  }
}

export function withoutStreamdownAnimationPlugin(
  rehypePlugins: BlockProps["rehypePlugins"],
  animatePlugin: BlockProps["animatePlugin"],
): BlockProps["rehypePlugins"] {
  const animationPlugin = animatePlugin?.rehypePlugin;
  if (!animationPlugin) {
    return rehypePlugins;
  }

  return rehypePlugins?.filter((plugin) => {
    const pluginFunction = Array.isArray(plugin) ? plugin[0] : plugin;
    return pluginFunction !== animationPlugin;
  });
}
