// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Plugin } from "vite";

const IS = ":is(";
const HAS = ":has(";
const COMBINATORS = new Set([" ", "\n", "\t", ">", "+", "~"]);

function escaped(text: string, at: number): boolean {
  let slashes = 0;
  for (let i = at - 1; i >= 0 && text[i] === "\\"; i--) slashes++;
  return slashes % 2 === 1;
}

function closingParen(text: string, from: number): number {
  let depth = 1;
  for (let i = from; i < text.length; i++) {
    const char = text[i];
    if (char === "\\") i++;
    else if (char === "(" || char === "[") depth++;
    else if ((char === ")" || char === "]") && --depth === 0) return i;
  }
  return -1;
}

// -1 inside another pseudo-class.
function selectorStart(text: string, at: number): number {
  let depth = 0;
  for (let i = at - 1; i >= 0; i--) {
    const char = text[i];
    if (escaped(text, i)) continue;
    if (char === ")" || char === "]") depth++;
    else if ((char === "(" || char === "[") && --depth < 0) return -1;
    else if (depth === 0 && (char === "{" || char === "}" || char === ";" || char === ",")) {
      return i + 1;
    }
  }
  return 0;
}

function selectorEnd(text: string, from: number): number {
  let depth = 0;
  for (let i = from; i < text.length; i++) {
    const char = text[i];
    if (char === "\\") i++;
    else if (char === "(" || char === "[") depth++;
    else if (char === ")" || char === "]") depth--;
    else if (depth === 0 && (char === "{" || char === ",")) return i;
  }
  return text.length;
}

function hasTopLevelCombinator(selector: string): boolean {
  let depth = 0;
  for (let i = 0; i < selector.length; i++) {
    const char = selector[i];
    if (char === "\\") i++;
    else if (char === "(" || char === "[") depth++;
    else if (char === ")" || char === "]") depth--;
    else if (depth === 0 && COMBINATORS.has(char)) return true;
  }
  return false;
}

function hasTopLevelComma(selector: string): boolean {
  let depth = 0;
  for (let i = 0; i < selector.length; i++) {
    const char = selector[i];
    if (char === "\\") i++;
    else if (char === "(" || char === "[") depth++;
    else if (char === ")" || char === "]") depth--;
    else if (depth === 0 && char === ",") return true;
  }
  return false;
}

// `:is(ANCHOR *)` -> descendant of ANCHOR, `:is(ANCHOR ~ *)` -> later sibling, and so on.
function splitAnchor(argument: string): { anchor: string; combinator: string } | null {
  const trimmed = argument.trim();
  if (!trimmed.endsWith("*") || escaped(trimmed, trimmed.length - 1) || hasTopLevelComma(trimmed)) {
    return null;
  }
  const rest = trimmed.slice(0, -1);
  const anchor = rest.trimEnd();
  const last = anchor.at(-1) ?? "";
  if (">+~".includes(last) && last !== "" && !escaped(anchor, anchor.length - 1)) {
    return { anchor: anchor.slice(0, -1).trim(), combinator: ` ${last} ` };
  }
  return anchor !== rest && anchor ? { anchor, combinator: " " } : null;
}

// Tailwind's group-has-*/peer-has-* emit `X:is(:where(.group):has(A) *)`; Chromium restyles a whole anchor
// subtree on any insertion below it. `:where(.group):has(A) X` matches the same with the same specificity.
export function hoistHasAnchors(css: string): string {
  let out = "";
  let cursor = 0;
  let at = css.indexOf(IS);
  while (at !== -1) {
    const start = at + IS.length;
    const close = closingParen(css, start);
    if (close === -1) break;
    const argument = css.slice(start, close);
    const split = argument.includes(HAS) ? splitAnchor(argument) : null;
    const from = selectorStart(css, at);
    if (split && from !== -1 && from >= cursor) {
      const end = selectorEnd(css, close + 1);
      const leading = css.slice(from, at).match(/^\s*/)?.[0] ?? "";
      // Only from the first compound: an ancestor written before it may sit inside or outside ANCHOR.
      const head = css.slice(from, at).trim();
      const compound = (head + css.slice(close + 1, end)).trim();
      if (!hasTopLevelCombinator(head)) {
        const trailing = css.slice(from, end).match(/\s*$/)?.[0] ?? "";
        out += `${css.slice(cursor, from)}${leading}${split.anchor}${split.combinator}${compound || "*"}${trailing}`;
        cursor = end;
        at = css.indexOf(IS, cursor);
        continue;
      }
    }
    at = css.indexOf(IS, start);
  }
  return out + css.slice(cursor);
}

export function hasAnchorOrder(): Plugin {
  return {
    name: "has-anchor-order",
    transform: {
      filter: { id: /\.css(?:$|\?)/, code: HAS },
      handler(code) {
        const next = hoistHasAnchors(code);
        return next === code ? null : { code: next, map: null };
      },
    },
  };
}
