// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A tool-call argument as text, whatever the model actually sent.
 *
 * Arguments reach a card as parsed JSON, so a prop declared `string` is a
 * request to the model, not a guarantee from it. A model that answers
 * `{"code": 42}` used to reach `code.split("\n")` during render and throw, and
 * a throw in a card has no boundary above it: the nearest catcher is the
 * router's, which replaces all of Studio with "Something went wrong!" and
 * unmounts the assistant-ui runtime with it (see markdown-block-boundary.tsx
 * for the same failure measured on a Markdown block). The message is persisted,
 * so reopening the thread reproduces it.
 *
 * Coercing keeps the card readable -- 42 shows as "42" -- rather than losing
 * the session over an argument that is only cosmetic to begin with.
 */
/**
 * How much of a serialised object a card will show.
 *
 * Only the JSON branch is capped, so a string argument -- which is every
 * well-formed call -- is returned untouched and nothing a user already sees
 * changes. It exists because the engines disagree about how big a string may
 * get: V8 stops at 2^29-24 (~512 Mi) and throws RangeError, SpiderMonkey at
 * 2^30-2 (~1 Gi), JavaScriptCore at 2^31-1 (~2 Gi). Without a cap the same
 * nested object renders as "" in Chrome, because the throw is caught below, and
 * as a several-hundred-megabyte string in Safari, which then has to be laid out.
 * Measured in all three at 100k nesting levels: Chrome produced 600,002
 * characters and handed them to React, where Firefox and Safari threw. The cap
 * does not make the engines identical -- they still disagree about when
 * serialising fails at all -- but it bounds what any of them can put in the DOM.
 */
const MAX_SERIALISED_LENGTH = 100_000;

export const toolArgText = (value: unknown): string => {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    // Objects are SERIALISED, not coerced. `String()` calls the value's own
    // `toString`, and `{"code":{"toString":null}}` is valid JSON a model can
    // send: that property shadows the one on Object.prototype and is not
    // callable, so coercing throws "Cannot convert object to primitive value"
    // -- the exact crash this helper exists to stop. JSON also shows the reader
    // what arrived, which "[object Object]" never did.
    if (typeof value === "object") {
      // `?? ""` because a `toJSON` can return undefined, a function or a symbol,
      // for which JSON.stringify returns undefined rather than a string.
      const json = JSON.stringify(value) ?? "";
      return json.length > MAX_SERIALISED_LENGTH
        ? `${json.slice(0, MAX_SERIALISED_LENGTH)}…`
        : json;
    }
    // NOT a template literal: `String(sym)` is spec'd to return the descriptive
    // string (ECMA-262 22.1.1.1), while `${sym}` throws.
    return String(value);
  } catch {
    // LOAD-BEARING, not defensive. Measured on the three engines Studio runs in,
    // with the deepest JSON each one will parse: all three parse 4,000,000 levels
    // of nesting, and then Firefox throws "InternalError: too much recursion" and
    // Safari "RangeError: Maximum call stack size exceeded" out of
    // JSON.stringify, where Chrome returns the string. So on two of three engines
    // this catch is reached by ordinary wire data, and without it the card would
    // throw exactly where it used to.
    return "";
  }
};
