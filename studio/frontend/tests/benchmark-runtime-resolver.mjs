// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const STUBS = new Map([
  ["../api/benchmark-api", "./helpers/store-stubs/benchmark-api.ts"],
  ["@/features/benchmarks/stores/benchmarks-store", "./helpers/store-stubs/benchmarks-store.ts"],
]);

export function resolve(specifier, context, next) {
  const stub = STUBS.get(specifier);
  if (stub) return next(new URL(stub, import.meta.url).href, context);
  return resolveBundler(specifier, context, next);
}
