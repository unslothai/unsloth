// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How much of a serialised object a card shows. Only the JSON branch is capped,
 * so every string argument is returned untouched.
 *
 * Engines disagree about how long a string may be (V8 ~512 Mi then RangeError,
 * SpiderMonkey ~1 Gi, JavaScriptCore ~2 Gi), so an uncapped serialisation is ""
 * in Chrome and hundreds of megabytes of DOM in Safari. Measured at 100k nesting
 * levels: Chrome produced 600,002 characters, Firefox and Safari threw.
 */
const MAX_SERIALISED_LENGTH = 100_000;

/**
 * A tool-call argument as text, whatever the model actually sent.
 *
 * Arguments reach a card as parsed JSON, so a prop declared `string` is a
 * request to the model, not a guarantee from it. `{"code": 42}` used to reach
 * `code.split("\n")` during render, and a throw in a card has no boundary above
 * it: the router catches it and replaces all of Studio with "Something went
 * wrong!" (markdown-block-boundary.tsx measures the same failure). The message
 * is persisted, so reopening the thread reproduces it.
 */
export const toolArgText = (value: unknown): string => {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    if (typeof value === "object") {
      // SERIALISED, not coerced: `String({"toString":null})` throws "Cannot
      // convert object to primitive value", the exact crash this stops. JSON
      // also shows what arrived, which "[object Object]" never did.
      // `?? ""` because a `toJSON` returning undefined makes stringify do so.
      const json = JSON.stringify(value) ?? "";
      return json.length > MAX_SERIALISED_LENGTH
        ? `${json.slice(0, MAX_SERIALISED_LENGTH)}…`
        : json;
    }
    // NOT a template literal: `String(sym)` returns the descriptive string
    // (ECMA-262 22.1.1.1), `${sym}` throws.
    return String(value);
  } catch {
    // Load-bearing, not defensive. All three engines parse 4,000,000 levels of
    // nesting; Firefox then throws "InternalError: too much recursion" and
    // Safari a RangeError out of JSON.stringify where Chrome succeeds. Keep the
    // binding bare: Firefox's is not a RangeError.
    return "";
  }
};
