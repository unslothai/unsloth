// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const dependencies = new Set([
  "@/features/auth",
  "../api/chat-api",
  "../api/providers-api",
  "../stores/chat-runtime-store",
  "../stores/external-providers-store",
  "./chat-history-storage",
]);

export function resolve(specifier, context, next) {
  if (dependencies.has(specifier)) {
    return next(
      new URL("./helpers/store-stubs/refresh-chat-title.ts", import.meta.url)
        .href,
      context,
    );
  }
  return resolveBundler(specifier, context, next);
}
