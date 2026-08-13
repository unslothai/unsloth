// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// use-chat-search-index pulls Dexie and the chat API transitively, neither of which loads in
// bare node. Stub just those, so the cache and hint bookkeeping can be driven for real.
import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const STUBS = new Map([
  ["@/features/auth", "./helpers/store-stubs/chat-search-auth.ts"],
  ["../api/chat-api", "./helpers/store-stubs/chat-search-history.ts"],
  ["../utils/chat-history-storage", "./helpers/store-stubs/chat-search-history.ts"],
]);

export function resolve(specifier, context, next) {
  const stub = STUBS.get(specifier);
  if (stub) return next(new URL(stub, import.meta.url).href, context);
  return resolveBundler(specifier, context, next);
}
