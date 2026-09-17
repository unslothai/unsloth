// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Some external providers leak a trailing template-literal fragment such as `${answer}` into
 *  the stream, so the adapter strips one off the end of the accumulated reply on every
 *  arrival. The pattern is anchored at the end, but a regex engine still tries it at every
 *  start offset, so running it over the whole buffer costs O(reply) per arrival and
 *  O(reply^2) over a reply. This module runs the same pattern over a bounded suffix. */

/** How far back the opening `${` may sit. Providers emit short placeholders, so an opener
 *  further back is not treated as one. The guarantee is about what is REMOVED: this removes
 *  no more than the unbounded pattern would and always returns a prefix of the input, so it
 *  can never delete text the unbounded pattern kept. Nested placeholders show the difference:
 *  with the outer opener out of window only the inner `${nested}` is stripped, 9 characters
 *  where the unbounded pattern deletes 4,208. */
export const TRAILING_PLACEHOLDER_WINDOW = 4096;

const TRAILING_TEMPLATE_PLACEHOLDER = /\s*\$\{[^}]*\}\s*$/;
const WHITESPACE = /\s/;
const CLOSE_BRACE = "}";

/** Remove one trailing `${...}` fragment, with the whitespace around it. Equivalent to
 *  `text.replace(/\s*\$\{[^}]*\}\s*$/, "")` for fragments up to `window` characters long, at
 *  a cost independent of the reply preceding it. */
export function stripTrailingTemplatePlaceholder(
  text: string,
  window: number = TRAILING_PLACEHOLDER_WINDOW,
): string {
  // `\s*$` can only match the whitespace run at the very end, and the character in front of it
  // has to be the closing brace. Almost every arrival fails here for the cost of that run,
  // which is where the quadratic term goes. The run is walked no further than the window.
  const floor = Math.max(0, text.length - window);
  let end = text.length;
  while (end > floor && WHITESPACE.test(text[end - 1])) {
    end -= 1;
  }
  if (end === 0 || text[end - 1] !== CLOSE_BRACE) {
    return text;
  }

  // `[^}]*` cannot span a `}`, so the opening `${` has to sit after the previous `}`. The window
  // keeps the search off the whole buffer; the previous `}` usually keeps it shorter still.
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

/** How far back a re-seed looks after a strip. The scan above never reads further back than
 *  `end - 1 - window`, and `end` is no further back than `length - window`, so everything it
 *  can reach lives in the last `2 * window` characters, plus two for a straddling `${`. */
const RESEED_WINDOW = 2 * TRAILING_PLACEHOLDER_WINDOW + 2;

const DOLLAR_BRACE = "${";

export type TrailingPlaceholderWatch = {
  /** Take the characters an arrival added. Reads `delta`, never the buffer. */
  append(delta: string): void;
  /** Take back the suffix a strip removed. `text` is the buffer after it. */
  retract(text: string): void;
  /** Whether `stripTrailingTemplatePlaceholder` could still cut something. Never false when it
   *  would, so the strip can be skipped whenever this is false. It may be true when the strip
   *  then cuts nothing, which costs one wasted scan. */
  isCandidate(): boolean;
};

/** Decide whether the trailing `${...}` strip has anything to do, from the deltas alone. The
 *  strip is bounded, but it still touches the end of the accumulated reply, and that flattens
 *  the cons string `text += delta` built, costing O(reply) per arrival. The pattern needs the
 *  last non-whitespace character to be `}` with a `${` in front and no `}` between, and both
 *  facts follow from the deltas, so a reply not ending in a brace is rejected unread. */
export function createTrailingPlaceholderWatch(): TrailingPlaceholderWatch {
  let length = 0;
  // The last non-whitespace character, or "" when there is none.
  let lastNonWhitespace = "";
  // Index of the last `${`, of the last `}`, and of the `}` before that one.
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
      // The `${` straddling the boundary is the one thing `delta` alone cannot show, because its `$`
      // may have arrived last time.
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
      // Only what the strip itself could reach. A `${` or `}` further back cannot take part in a
      // match, so forgetting it cannot hide one.
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
