// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// use-chat-sidebar-items pulls the chat API, Dexie-backed storage and the zustand stores,
// none of which load in bare node. One stub module answers for all of them so the real
// archiveAllChatItems body runs.
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
