// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Some external providers (mistral magistral, occasionally) leak a trailing
 * template-literal fragment such as `${answer}` into the stream, so the
 * adapter strips one off the end of the accumulated reply on every arrival.
 *
 * The pattern below is anchored at the end, but a regex engine still tries it
 * at every start offset, so running it over the whole buffer costs O(reply)
 * per arrival and O(reply^2) over a reply. This module runs the same pattern
 * over a bounded suffix instead.
 */

/**
 * How far back the opening `${` may sit. Providers emit short placeholders, so
 * an opener further back than this is not treated as one.
 *
 * The guarantee is stated in terms of what is REMOVED, not what is left whole:
 * whatever this returns, it removed no more than the unbounded pattern would,
 * and the result is always a prefix of the input. So it can never delete text
 * the unbounded pattern would have kept. That is asserted directly across
 * windows of 1, 2, 4, 8 and 16 characters.
 *
 * "The oversized fragment is left whole" is the usual consequence but not the
 * rule, and the difference shows up when placeholders nest. For
 * `"answer ${" + "a".repeat(4196) + "${nested}"` the outer opener is out of
 * window, so the inner `${nested}` is what gets stripped and the outer text
 * stays. The unbounded pattern deletes 4,208 characters of that input; this
 * deletes 9. Less is removed, which is the direction that matters, but the
 * outer fragment is not returned untouched.
 */
export const TRAILING_PLACEHOLDER_WINDOW = 4096;

const TRAILING_TEMPLATE_PLACEHOLDER = /\s*\$\{[^}]*\}\s*$/;
const WHITESPACE = /\s/;
const CLOSE_BRACE = "}";

/**
 * Remove one trailing `${...}` fragment, with the whitespace around it.
 *
 * Equivalent to `text.replace(/\s*\$\{[^}]*\}\s*$/, "")` for every fragment
 * up to `window` characters long, at a cost that does not grow with the part
 * of the reply that precedes it.
 */
export function stripTrailingTemplatePlaceholder(
  text: string,
  window: number = TRAILING_PLACEHOLDER_WINDOW,
): string {
  // `\s*$` can only be satisfied by the whitespace run at the very end, and
  // the character in front of that run has to be the closing brace. Almost
  // every arrival in a reply fails here, for the cost of the trailing
  // whitespace run, which is where the quadratic term goes. The run is walked
  // no further than the window, so a reply ending in a growing field of
  // whitespace cannot reintroduce it.
  const floor = Math.max(0, text.length - window);
  let end = text.length;
  while (end > floor && WHITESPACE.test(text[end - 1])) {
    end -= 1;
  }
  if (end === 0 || text[end - 1] !== CLOSE_BRACE) {
    return text;
  }

  // `[^}]*` cannot span a `}`, so the opening `${` has to sit after the
  // previous `}`. Bounding the slice first keeps that search off the whole
  // buffer; bounding it at the previous `}` then usually keeps it far shorter
  // than the window, since brace-heavy replies are the ones that reach here.
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
