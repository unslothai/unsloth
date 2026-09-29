// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// code-plugin.ts builds its own highlighter from a static `shiki` import, so
// nothing it exposes can be wrapped from a test: an ES module namespace is
// read-only. Redirect that one import to a counting re-export instead, and
// leave every other importer of `shiki` (the tests' own reference highlighter)
// on the real module.
const COUNTER = new URL("./shiki-tokenization-counter.mts", import.meta.url)
  .href;

export function resolve(specifier, context, next) {
  if (specifier === "shiki" && context.parentURL?.endsWith("/code-plugin.ts")) {
    return next(COUNTER, context);
  }
  return next(specifier, context);
}
