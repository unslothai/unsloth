// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type CompileContext,
  type Extension as FromMarkdownExtension,
  fromMarkdown,
} from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { type InlineMath, mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import type {
  Construct,
  Extension as MicromarkExtension,
  State,
  Token,
  TokenizeContext,
  Tokenizer,
} from "micromark-util-types";
import { openCodeSpanTail } from "./markdown-code-spans.ts";

declare module "micromark-util-types" {
  interface TokenTypeMap {
    escapedMathText: "escapedMathText";
    escapedMathTextData: "escapedMathTextData";
    escapedMathTextSequence: "escapedMathTextSequence";
  }
}

const BACKSLASH = 92;
const DOLLAR = 36;
const MAX_BODY_LENGTH = 200;
const LINE_ENDINGS = new Set([-5, -4, -3]);

const TOKEN = "escapedMathText";
const SEQUENCE_TOKEN = "escapedMathTextSequence";
const DATA_TOKEN = "escapedMathTextData";

const STRONG_MATH_RE = /[\\^_{}]/;
const SINGLE_VARIABLE_RE = /^[a-zA-Z]$/;
const SIMPLE_OPERAND = String.raw`(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z])`;
const SIMPLE_EXPRESSION_RE = new RegExp(
  String.raw`^${SIMPLE_OPERAND}(?:\s*[=+\-<>/*]\s*${SIMPLE_OPERAND})+$`,
);
const SIMPLE_FUNCTION_RE = /^[a-zA-Z]\([a-zA-Z0-9]\)$/;
const COMMAND_RE = /^[a-zA-Z]+/;
const MARKDOWN_INLINE_BOUNDARY_RE =
  /`|!?\[[^\]\n]*\](?:\(|\[[^\]\n]*\])|<\/?[a-zA-Z][^>\n]*>/;
const HTML_LITERAL_TAG_START_RE = /^<\/?(code|pre|textarea)(?=[\s/>])/i;
const CURRENCY_OPENER_RE =
  /(?<![\\$])\$(?!\$)(?=\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?(?:\s|$|[^a-zA-Z\d]))/g;
const DIGIT_RE = /\d/;
const MASKED_DOLLAR = "＄";
const TEXT_COMMANDS = new Set([
  "emph",
  "hbox",
  "mbox",
  "operatorname",
  "text",
  "textbf",
  "textit",
  "textmd",
  "textnormal",
  "textrm",
  "textsc",
  "textsf",
  "textsl",
  "texttt",
  "textup",
]);
const htmlLiteralStates = new WeakMap<
  TokenizeContext,
  { eventIndex: number; openTags: string[] }
>();
type LatexBracketState = { eventIndex: number; openDelimiters: string[] };
const latexBracketStates = new WeakMap<TokenizeContext, LatexBracketState>();
const escapedMathNodes = new WeakSet<object>();

function isInHtmlLiteral(context: TokenizeContext): boolean {
  let state = htmlLiteralStates.get(context);
  if (!state || state.eventIndex > context.events.length) {
    state = { eventIndex: 0, openTags: [] };
  }
  for (; state.eventIndex < context.events.length; state.eventIndex += 1) {
    const [phase, token] = context.events[state.eventIndex];
    if (phase !== "exit" || token.type !== "htmlText") {
      continue;
    }
    const source = context.sliceSerialize(token);
    const match = HTML_LITERAL_TAG_START_RE.exec(source);
    if (!match) {
      continue;
    }
    const name = match[1].toLowerCase();
    if (source.startsWith("</")) {
      const opening = state.openTags.lastIndexOf(name);
      if (opening >= 0) {
        state.openTags.length = opening;
      }
    } else if (!source.endsWith("/>")) {
      state.openTags.push(name);
    }
  }
  htmlLiteralStates.set(context, state);
  return state.openTags.length > 0;
}

function updateLatexBracketState(
  state: LatexBracketState,
  source: string,
): void {
  if (source === "\\(" || source === "\\[") {
    state.openDelimiters.push(source[1]);
    return;
  }
  if (source === "\\)" || source === "\\]") {
    const opening = state.openDelimiters.lastIndexOf(
      source === "\\)" ? "(" : "[",
    );
    if (opening >= 0) {
      state.openDelimiters.length = opening;
    }
  }
}

function isInLatexBracketMath(context: TokenizeContext): boolean {
  let state = latexBracketStates.get(context);
  if (!state || state.eventIndex > context.events.length) {
    state = { eventIndex: 0, openDelimiters: [] };
  }
  for (; state.eventIndex < context.events.length; state.eventIndex += 1) {
    const [phase, token] = context.events[state.eventIndex];
    if (phase === "exit" && token.type === "characterEscape") {
      updateLatexBracketState(state, context.sliceSerialize(token));
    }
  }
  latexBracketStates.set(context, state);
  return state.openDelimiters.length > 0;
}

/** deliberately conservative: ambiguous prose remains literal. */
export function looksLikeEscapedInlineMath(body: string): boolean {
  const value = body.trim();
  if (!value || value.length > MAX_BODY_LENGTH) {
    return false;
  }
  return (
    STRONG_MATH_RE.test(value) ||
    SINGLE_VARIABLE_RE.test(value) ||
    SIMPLE_EXPRESSION_RE.test(value) ||
    SIMPLE_FUNCTION_RE.test(value)
  );
}

function appendCode(value: string, code: number): string {
  if (code === -2) {
    return `${value}\t`;
  }
  if (code === -1) {
    return value;
  }
  return `${value}${String.fromCodePoint(code)}`;
}

function braceBalance(value: string): number {
  let depth = 0;
  let escaped = false;
  for (const character of value) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (character === "\\") {
      escaped = true;
    } else if (character === "{") {
      depth += 1;
    } else if (character === "}") {
      if (depth === 0) {
        return -1;
      }
      depth -= 1;
    }
  }
  return depth;
}

type MathBodyProtection = {
  readonly parts: string[];
  readonly textGroups: boolean[];
  textCommandPending: boolean;
};

function consumeMathCommand(
  value: string,
  index: number,
  state: MathBodyProtection,
): number {
  if (value[index + 1] === "$") {
    state.parts.push(String.raw`{\char"24}`);
    state.textCommandPending = false;
    return index + 1;
  }
  const command = COMMAND_RE.exec(value.slice(index + 1))?.[0];
  if (command) {
    state.parts.push(`\\${command}`);
    state.textCommandPending = TEXT_COMMANDS.has(command);
    return index + command.length;
  }
  state.parts.push(value.slice(index, index + 2));
  state.textCommandPending = false;
  return index + 1;
}

function consumeMathCharacter(
  value: string,
  index: number,
  state: MathBodyProtection,
): number {
  const character = value[index];
  if (character === "\\") {
    return consumeMathCommand(value, index, state);
  }
  if (character === "{") {
    state.textGroups.push(
      state.textCommandPending || state.textGroups.at(-1) === true,
    );
    state.textCommandPending = false;
  } else if (character === "}") {
    state.textGroups.pop();
    state.textCommandPending = false;
  } else if (character === "<") {
    state.parts.push(
      state.textGroups.at(-1) === true
        ? String.raw`{\char"3C}`
        : String.raw`\lt `,
    );
    state.textCommandPending = false;
    return index;
  } else if (character.trim() !== "") {
    state.textCommandPending = false;
  }
  state.parts.push(character);
  return index;
}

function protectMathBody(value: string): string {
  const state: MathBodyProtection = {
    parts: [],
    textGroups: [],
    textCommandPending: false,
  };
  for (let index = 0; index < value.length; index += 1) {
    index = consumeMathCharacter(value, index, state);
  }
  return state.parts.join("");
}

const tokenizeEscapedMath: Tokenizer = function (effects, ok, nok) {
  let body = "";
  let pending: Token;

  const start: State = (code) => {
    const previousToken = this.events.at(-1)?.[1];
    if (
      code !== BACKSLASH ||
      isInHtmlLiteral(this) ||
      isInLatexBracketMath(this) ||
      (this.previous === DOLLAR && previousToken?.type !== TOKEN)
    ) {
      return nok(code);
    }
    effects.enter(TOKEN);
    effects.enter(SEQUENCE_TOKEN);
    effects.consume(code);
    return openingDollar;
  };

  const openingDollar: State = (code) => {
    if (code !== DOLLAR) {
      return nok(code);
    }
    effects.consume(code);
    return afterOpening;
  };

  const afterOpening: State = (code) => {
    effects.exit(SEQUENCE_TOKEN);
    effects.enter(DATA_TOKEN);
    return data(code);
  };

  const data: State = (code) => {
    if (code === null || LINE_ENDINGS.has(code)) {
      return nok(code);
    }
    if (code === BACKSLASH) {
      effects.exit(DATA_TOKEN);
      pending = effects.enter(SEQUENCE_TOKEN);
      effects.consume(code);
      return closingDollar;
    }
    body = appendCode(body, code);
    if (body.length > MAX_BODY_LENGTH) {
      return nok(code);
    }
    effects.consume(code);
    return data;
  };

  const closingDollar: State = (code) => {
    if (code === DOLLAR) {
      effects.consume(code);
      if (braceBalance(body) > 0) {
        pending.type = DATA_TOKEN;
        body += "\\$";
        if (body.length > MAX_BODY_LENGTH) {
          return nok(code);
        }
        return data;
      }
      return finish;
    }
    pending.type = DATA_TOKEN;
    body += "\\";
    if (body.length > MAX_BODY_LENGTH) {
      return nok(code);
    }
    return data(code);
  };

  const finish: State = (code) => {
    if (
      braceBalance(body) !== 0 ||
      MARKDOWN_INLINE_BOUNDARY_RE.test(body) ||
      !looksLikeEscapedInlineMath(body)
    ) {
      return nok(code);
    }
    effects.exit(SEQUENCE_TOKEN);
    effects.exit(TOKEN);
    return ok(code);
  };

  return start;
};

const escapedMathConstruct: Construct = {
  name: TOKEN,
  tokenize: tokenizeEscapedMath,
};

const escapedMathSyntax: MicromarkExtension = {
  text: { [BACKSLASH]: escapedMathConstruct },
};

const escapedMathFromMarkdown: FromMarkdownExtension = {
  enter: {
    [TOKEN](this: CompileContext, token: Token): void {
      const node = {
        type: "inlineMath",
        value: "",
        data: {
          hName: "code",
          hProperties: { className: ["language-math", "math-inline"] },
          hChildren: [],
        },
      } satisfies InlineMath;
      escapedMathNodes.add(node);
      this.enter(node, token);
      this.buffer();
    },
  },
  exit: {
    [TOKEN](this: CompileContext, token: Token): void {
      const value = this.resume().trim();
      const node = this.stack.at(-1);
      if (node?.type !== "inlineMath") {
        throw new Error(
          "escaped inline math did not produce an inlineMath node",
        );
      }
      this.exit(token);
      node.value = value;
      node.data = {
        ...node.data,
        hName: "code",
        hProperties: { className: ["language-math", "math-inline"] },
        hChildren: [{ type: "text", value }],
      };
    },
    [DATA_TOKEN](this: CompileContext, token: Token): void {
      this.config.enter.data.call(this, token);
      this.config.exit.data.call(this, token);
    },
  },
};

type MarkdownNode = {
  readonly type: string;
  readonly position?: {
    readonly start: { readonly offset?: number };
    readonly end: { readonly offset?: number };
  };
  readonly children?: readonly MarkdownNode[];
};

type EscapedMathRange = {
  readonly start: number;
  readonly end: number;
  readonly value: string;
};

function hasLikelyMathCloser(markdown: string, offset: number): boolean {
  const limit = Math.min(markdown.length, offset + MAX_BODY_LENGTH + 1);
  for (let index = offset + 1; index < limit; index += 1) {
    const character = markdown[index];
    if (character === "\n" || character === "\r") {
      return false;
    }
    if (character !== "$" || markdown[index - 1] === "\\") {
      continue;
    }
    if (markdown[index + 1] === "$") {
      index += 1;
      continue;
    }
    if (DIGIT_RE.test(markdown[index + 1] ?? "")) {
      continue;
    }
    return looksLikeEscapedInlineMath(markdown.slice(offset + 1, index));
  }
  return false;
}

function escapedMathRanges(markdown: string): EscapedMathRange[] {
  const masked = markdown.replace(CURRENCY_OPENER_RE, (opener, offset) =>
    hasLikelyMathCloser(markdown, offset) ? opener : MASKED_DOLLAR,
  );
  const openCodeTail = openCodeSpanTail(masked);
  const tree = fromMarkdown(masked, {
    extensions: [
      gfm(),
      math({ singleDollarTextMath: true }),
      escapedMathSyntax,
    ],
    mdastExtensions: [
      gfmFromMarkdown(),
      mathFromMarkdown(),
      escapedMathFromMarkdown,
    ],
  }) as MarkdownNode;
  const ranges: EscapedMathRange[] = [];
  const pending = [tree];
  while (pending.length > 0) {
    const node = pending.pop();
    if (!node) {
      continue;
    }
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    if (
      node.type === "inlineMath" &&
      escapedMathNodes.has(node) &&
      start !== undefined &&
      end !== undefined &&
      !(openCodeTail && start >= openCodeTail.start && start < openCodeTail.end)
    ) {
      ranges.push({
        start,
        end,
        value: markdown.slice(start + 2, end - 2).trim(),
      });
    }
    if (node.children) {
      pending.push(...node.children);
    }
  }
  return ranges.sort((left, right) => left.start - right.start);
}

/** normalize model-emitted `\$…\$` before currency and streaming repair. */
export function normalizeEscapedInlineMath(markdown: string): string {
  if (!markdown.includes("\\$")) {
    return markdown;
  }
  const ranges = escapedMathRanges(markdown);
  if (ranges.length === 0) {
    return markdown;
  }

  const parts: string[] = [];
  let offset = 0;
  let previousCharacter = "";
  const append = (value: string): void => {
    if (previousCharacter === "$" && value.startsWith("$")) {
      parts.push(" ");
    }
    parts.push(value);
    if (value.length > 0) {
      previousCharacter = value.at(-1) ?? "";
    }
  };
  for (const range of ranges) {
    append(markdown.slice(offset, range.start));
    append(`$${protectMathBody(range.value)}$`);
    offset = range.end;
  }
  append(markdown.slice(offset));
  return parts.join("");
}
