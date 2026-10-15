// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Strips a trailing `${...}` leaked by providers, scanning a bounded suffix to avoid O(n^2). */

/** Max distance back to the opening `${`. Never removes more than the unbounded pattern. */
export const TRAILING_PLACEHOLDER_WINDOW = 4096;

const TRAILING_TEMPLATE_PLACEHOLDER = /\s*\$\{[^}]*\}\s*$/;
const WHITESPACE = /\s/;
const CLOSE_BRACE = "}";

/** Same as `text.replace(/\s*\$\{[^}]*\}\s*$/, "")` for fragments up to `window` long. */
export function stripTrailingTemplatePlaceholder(
  text: string,
  window: number = TRAILING_PLACEHOLDER_WINDOW,
): string {
  const floor = Math.max(0, text.length - window);
  let end = text.length;
  while (end > floor && WHITESPACE.test(text[end - 1])) {
    end -= 1;
  }
  if (end === 0 || text[end - 1] !== CLOSE_BRACE) {
    return text;
  }

  // `[^}]*` cannot span `}`, so the opener must follow the previous `}`.
  const from = Math.max(0, end - 1 - window);
  const tail = text.slice(from);
  const previousBrace = tail.lastIndexOf(CLOSE_BRACE, end - 2 - from);
  const scanFrom = previousBrace === -1 ? 0 : previousBrace + 1;
  const match = TRAILING_TEMPLATE_PLACEHOLDER.exec(tail.slice(scanFrom));
  if (!match) {
    return text;
  }
  return text.slice(0, from + scanFrom + match.index);
}

/** Everything the scan can reach lies in the last 2 * window chars, plus two for `${`. */
const RESEED_WINDOW = 2 * TRAILING_PLACEHOLDER_WINDOW + 2;

const DOLLAR_BRACE = "${";

export type TrailingPlaceholderWatch = {
  append(delta: string): void;
  retract(text: string): void;
  /** Never false when the strip would cut something; may be a false positive. */
  isCandidate(): boolean;
};

/** Decides from deltas alone, since reading the buffer flattens the cons string. */
export function createTrailingPlaceholderWatch(): TrailingPlaceholderWatch {
  let length = 0;
  let lastNonWhitespace = "";
  let lastDollarBrace = -1;
  let lastCloseBrace = -1;
  let previousCloseBrace = -1;
  // One character, so a `${` split across two arrivals is still seen.
  let overlap = "";

  const scan = (window: string, from: number): void => {
    for (let index = 0; index < window.length; index += 1) {
      const character = window[index];
      if (character === CLOSE_BRACE) {
        previousCloseBrace = lastCloseBrace;
        lastCloseBrace = from + index;
      } else if (
        character === "{" &&
        index > 0 &&
        window[index - 1] === DOLLAR_BRACE[0]
      ) {
        lastDollarBrace = from + index - 1;
      }
      if (!WHITESPACE.test(character)) {
        lastNonWhitespace = character;
      }
    }
  };

  return {
    append(delta: string): void {
      if (!delta) {
        return;
      }
      if (overlap === DOLLAR_BRACE[0] && delta[0] === "{") {
        lastDollarBrace = length - 1;
      }
      scan(delta, length);
      length += delta.length;
      overlap = delta[delta.length - 1];
    },
    retract(text: string): void {
      length = text.length;
      lastNonWhitespace = "";
      lastDollarBrace = -1;
      lastCloseBrace = -1;
      previousCloseBrace = -1;
      overlap = "";
      const from = Math.max(0, text.length - RESEED_WINDOW);
      scan(text.slice(from), from);
      if (text.length > 0) {
        overlap = text[text.length - 1];
      }
    },
    isCandidate(): boolean {
      return (
        lastNonWhitespace === CLOSE_BRACE &&
        lastDollarBrace > previousCloseBrace &&
        lastDollarBrace < lastCloseBrace
      );
    },
  };
}
