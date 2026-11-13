// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Plugin } from "vite";

const WHERE = ":where(";
const NOT_LAST_CHILD = ":not(:last-child)";

function closingParen(text: string, from: number): number {
  let depth = 1;
  for (let i = from; i < text.length; i++) {
    const char = text[i];
    if (char === "\\") i++;
    else if (char === "(") depth++;
    else if (char === ")" && --depth === 0) return i;
  }
  return -1;
}

function lastChildCombinator(selector: string): number {
  let depth = 0;
  let found = -1;
  for (let i = 0; i < selector.length; i++) {
    const char = selector[i];
    if (char === "\\") i++;
    else if (char === "(" || char === "[") depth++;
    else if (char === ")" || char === "]") depth--;
    else if (depth === 0 && char === ",") return -1;
    else if (depth === 0 && char === ">") found = i;
  }
  return found;
}

export function parentFirstChildSelectors(css: string): string {
  let out = "";
  let cursor = 0;
  for (
    let at = css.indexOf(WHERE);
    at !== -1;
    at = css.indexOf(WHERE, cursor)
  ) {
    const start = at + WHERE.length;
    const end = closingParen(css, start);
    if (end === -1) break;
    const argument = css.slice(start, end);
    const split = lastChildCombinator(argument);
    if (split > 0 && argument.slice(split + 1).trim() === NOT_LAST_CHILD) {
      const parent = argument.slice(0, split).trim();
      out += `${css.slice(cursor, at)}:where(${parent} > *):where(${NOT_LAST_CHILD})`;
      cursor = end + 1;
    } else {
      out += css.slice(cursor, start);
      cursor = start;
    }
  }
  return out + css.slice(cursor);
}

export function childSelectorOrder(): Plugin {
  return {
    name: "child-selector-order",
    transform: {
      filter: { id: /\.css(?:$|\?)/, code: NOT_LAST_CHILD },
      handler(code) {
        const next = parentFirstChildSelectors(code);
        return next === code ? null : { code: next, map: null };
      },
    },
  };
}
