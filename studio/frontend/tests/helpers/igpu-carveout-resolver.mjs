// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// bundler-resolver's two rules, plus the two redirects this notice needs at once:
// "@/lib/toast" -> toast-stub.mjs, which records the whole options bag (id, duration,
// action -- none of which the store-stubs toast keeps), and "@/features/auth" -> the
// store stub, so the dismissal POST never reaches the network.
//
// Its own resolver rather than registering the toast and store-stub ones together:
// whichever ran first would resolve "@/..." to a real path, hiding the bare specifier
// from the other.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const SRC = fileURLToPath(new URL("../../src/", import.meta.url));
const REDIRECTS = new Map([
  ["@/lib/toast", new URL("./toast-stub.mjs", import.meta.url).href],
  ["@/features/auth", new URL("./store-stubs/auth.ts", import.meta.url).href],
]);

function firstExisting(base) {
  for (const candidate of [`${base}.ts`, `${base}/index.ts`, base]) {
    if (existsSync(candidate)) return pathToFileURL(candidate).href;
  }
  return null;
}

export function resolve(specifier, context, next) {
  const redirect = REDIRECTS.get(specifier);
  if (redirect) return next(redirect, context);
  if (specifier.startsWith("@/")) {
    const resolved = firstExisting(SRC + specifier.slice(2));
    return next(resolved ?? specifier, context);
  }
  if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
    const resolved = firstExisting(
      fileURLToPath(new URL(specifier, context.parentURL)),
    );
    if (resolved) return next(resolved, context);
  }
  return next(specifier, context);
}
