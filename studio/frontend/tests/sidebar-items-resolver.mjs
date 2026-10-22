// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// These modules do not load in bare node; one stub answers for all so the real body runs.
import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const DEPS = "./helpers/store-stubs/sidebar-items-deps.ts";
const STUBS = new Map([
  ["../api/chat-api", DEPS],
  ["../artifacts/store", DEPS],
  ["../stores/chat-runtime-store", DEPS],
  ["../utils/chat-history-storage", DEPS],
  ["../utils/composer-draft", DEPS],
  ["../utils/offer-kept-sandbox-files", DEPS],
  ["../utils/stop-chat-thread", DEPS],
  ["../utils/chat-thread-tombstones", DEPS],
  ["../utils/prompt-queue-boundary", DEPS],
  ["../utils/repair-legacy-chat-titles", DEPS],
]);

export function resolve(specifier, context, next) {
  const stub = STUBS.get(specifier);
  if (stub) return next(new URL(stub, import.meta.url).href, context);
  return resolveBundler(specifier, context, next);
}
