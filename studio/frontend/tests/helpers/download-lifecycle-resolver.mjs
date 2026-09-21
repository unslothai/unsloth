// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Node cannot load the TSX feature barrels reached by the download manager. Keep
// this integration test on the real download modules while replacing only those
// browser-only boundaries and Sonner.
import { resolve as resolveBundler } from "../bundler-resolver.mjs";

const STUBS = new Map([
  ["@/features/auth", "./download-lifecycle-auth-stub.mjs"],
  ["@/features/auth/api", "./download-lifecycle-auth-stub.mjs"],
  ["@/features/settings", "./download-lifecycle-settings-stub.mjs"],
  ["@/lib/toast", "./toast-stub.mjs"],
]);

export function resolve(specifier, context, next) {
  const stub = STUBS.get(specifier);
  if (stub) return next(new URL(stub, import.meta.url).href, context);
  return resolveBundler(specifier, context, next);
}
