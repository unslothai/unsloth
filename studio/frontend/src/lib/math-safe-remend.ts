// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RemendHandler, RemendOptions } from "remend";

// remend's htmlTags pass, except a `<` inside math (`x_{<i}`) is passed over instead of cut.
const TRAILING_TAG_RE = /<[a-zA-Z/][^>]*$/;
const TAG_START_RE = /<[a-zA-Z/]/g;

type MathState =
  | "none"
  | "inlineDollar"
  | "blockDollar"
  | "inlineLatex"
  | "blockLatex";

// Mirrors remend's internal code lookup: lookup[i] is whether index i sits in code.
function codeLookup(text: string): Uint8Array {
  const lookup = new Uint8Array(text.length + 1);
  let inline = false;
  let fence = false;
  let i = 0;
  while (i < text.length) {
    if (text[i] === "\\" && text[i + 1] === "`") {
      const state = inline || fence ? 1 : 0;
      lookup[i + 1] = state;
      lookup[i + 2] = state;
      i += 2;
      continue;
    }
    if (text.startsWith("```", i)) {
      fence = !fence;
      const state = inline || fence ? 1 : 0;
      const end = Math.min(i + 3, text.length);
      for (let j = i + 1; j <= end; j++) {
        lookup[j] = state;
      }
      i = end;
      continue;
    }
    if (!fence && text[i] === "`") {
      inline = !inline;
    }
    lookup[i + 1] = inline || fence ? 1 : 0;
    i += 1;
  }
  return lookup;
}

function nextLatexState(state: MathState, char: string): MathState | null {
  if (char === "[" && state === "none") return "blockLatex";
  if (char === "]" && state === "blockLatex") return "none";
  if (char === "(" && state === "none") return "inlineLatex";
  if (char === ")" && state === "inlineLatex") return "none";
  return null;
}

function nextDollarState(state: MathState, block: boolean): MathState {
  if (block) return state === "blockDollar" ? "none" : "blockDollar";
  if (state === "blockDollar") return state;
  return state === "inlineDollar" ? "none" : "inlineDollar";
}

// Mirrors remend's isWithinMathBlock in one pass, ignoring characters that are code. Only `$$`
// counts while still open (remend closes it); other math counts once it closes, so an unclosed
// `$` stays prose as remark-math reads it, like `echo $PATH <done`.
function mathAt(
  text: string,
  code: Uint8Array,
  positions: number[],
): boolean[] {
  const openAt: Array<number | null> = [];
  const closedSpans = new Set<number>();
  let state: MathState = "none";
  let spanId = 0;
  let next = 0;
  for (let i = 0; i <= text.length; i++) {
    while (next < positions.length && positions[next] <= i) {
      openAt.push(
        state === "none"
          ? null
          : state === "blockDollar"
            ? -1
            : spanId,
      );
      next += 1;
    }
    if (i >= text.length || code[i] || code[i + 1]) continue;
    const char = text[i];
    const after = text[i + 1];
    if (char === "\\" && after === "$") {
      i += 1;
      continue;
    }
    let nextState: MathState | null = null;
    let width = 0;
    if (char === "\\") {
      nextState = nextLatexState(state, after);
      width = 1;
    }
    if (
      nextState === null &&
      char === "$" &&
      state !== "inlineLatex" &&
      state !== "blockLatex"
    ) {
      const block = after === "$";
      nextState = nextDollarState(state, block);
      width = block ? 1 : 0;
    }
    if (nextState === null) continue;
    if (state === "none" && nextState !== "blockDollar") {
      spanId += 1;
    }
    if (state !== "none" && state !== "blockDollar" && nextState === "none") {
      closedSpans.add(spanId);
    }
    state = nextState;
    i += width;
  }
  return openAt.map((id) => id === -1 || (id !== null && closedSpans.has(id)));
}

export function stripIncompleteHtmlTag(text: string): string {
  const match = TRAILING_TAG_RE.exec(text);
  if (!match) return text;
  const code = codeLookup(text);
  if (code[match.index]) return text;
  const head = text.slice(0, match.index);
  if (!/[$\\]/.test(head)) return head.trimEnd();
  const candidates: number[] = [];
  TAG_START_RE.lastIndex = match.index;
  for (
    let found = TAG_START_RE.exec(text);
    found;
    found = TAG_START_RE.exec(text)
  ) {
    candidates.push(found.index);
  }
  const inMath = mathAt(text, code, candidates);
  for (let k = 0; k < candidates.length; k++) {
    if (code[candidates[k]]) return text;
    if (!inMath[k]) return text.slice(0, candidates[k]).trimEnd();
  }
  return text;
}

const MATH_SAFE_HTML_TAGS: RemendHandler = {
  name: "mathSafeHtmlTags",
  handle: stripIncompleteHtmlTag,
  priority: 10,
};

const DEFAULT_OPTIONS: RemendOptions = {
  htmlTags: false,
  handlers: [MATH_SAFE_HTML_TAGS],
};
const wrapped = new WeakMap<RemendOptions, RemendOptions>();

export function mathSafeRemendOptions(options?: RemendOptions): RemendOptions {
  if (!options) return DEFAULT_OPTIONS;
  if (options.htmlTags === false) return options;
  let safe = wrapped.get(options);
  if (!safe) {
    safe = {
      ...options,
      htmlTags: false,
      handlers: [...(options.handlers ?? []), MATH_SAFE_HTML_TAGS],
    };
    wrapped.set(options, safe);
  }
  return safe;
}
