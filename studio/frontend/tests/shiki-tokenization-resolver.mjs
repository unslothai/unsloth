// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// code-plugin.ts imports shiki statically and ES namespaces are read-only, so redirect it.
const COUNTER = new URL("./shiki-tokenization-counter.mts", import.meta.url)
  .href;

export function resolve(specifier, context, next) {
  if (specifier === "shiki" && context.parentURL?.endsWith("/code-plugin.ts")) {
    return next(COUNTER, context);
  }
  return next(specifier, context);
}
