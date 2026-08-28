// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CompileContext,
  Extension as FromMarkdownExtension,
} from "mdast-util-from-markdown";
import type { InlineMath } from "mdast-util-math";
import type {
  Construct,
  Extension as MicromarkExtension,
  State,
  Token,
  Tokenizer,
} from "micromark-util-types";
import type { Plugin } from "unified";

declare module "micromark-util-types" {
  interface TokenTypeMap {
    escapedMathText: "escapedMathText";
    escapedMathTextData: "escapedMathTextData";
    escapedMathTextSequence: "escapedMathTextSequence";
  }
}

declare module "unified" {
  interface Data {
    micromarkExtensions?: MicromarkExtension[];
    fromMarkdownExtensions?: Array<
      FromMarkdownExtension[] | FromMarkdownExtension
    >;
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

/** Deliberately conservative: ambiguous prose remains literal. */
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

const tokenizeEscapedMath: Tokenizer = (effects, ok, nok) => {
  let body = "";
  let pending: Token;

  const start: State = (code) => {
    if (code !== BACKSLASH) {
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
    if (!looksLikeEscapedInlineMath(body)) {
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

/** Parse model-emitted `\$…\$` as math only in Markdown text contexts. */
export const remarkEscapedInlineMath: Plugin<[]> = function () {
  const data = this.data();
  const micromarkExtensions = data.micromarkExtensions ?? [];
  const fromMarkdownExtensions = data.fromMarkdownExtensions ?? [];
  data.micromarkExtensions = micromarkExtensions;
  data.fromMarkdownExtensions = fromMarkdownExtensions;
  micromarkExtensions.push(escapedMathSyntax);
  fromMarkdownExtensions.push(escapedMathFromMarkdown);
};
