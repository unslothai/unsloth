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
    if (typeof value === "object") return JSON.stringify(value) ?? "";
    return String(value);
  } catch {
    // Unreachable for anything JSON.parse can produce, and kept anyway: the
    // whole contract of this helper is that it does not throw.
    return "";
  }
};
