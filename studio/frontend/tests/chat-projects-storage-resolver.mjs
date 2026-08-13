// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The store stubs plus a deferred chat-history-storage, so a test drives the projects
// requests by hand. Scoped to use-chat-projects so no other module sees the stub.
import { resolve as resolveStoreStubs } from "./store-stub-resolver.mjs";

export function resolve(specifier, context, next) {
  if (
    specifier.endsWith("utils/chat-history-storage") &&
    context.parentURL?.includes("use-chat-projects")
  ) {
    const stub = new URL(
      "./helpers/store-stubs/chat-history-storage.ts",
      import.meta.url,
    );
    return next(stub.href, context);
  }
  return resolveStoreStubs(specifier, context, next);
}
