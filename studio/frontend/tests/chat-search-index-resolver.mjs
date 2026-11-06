// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only Dexie and the chat API are stubbed, so the real cache and hint bookkeeping runs.
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
