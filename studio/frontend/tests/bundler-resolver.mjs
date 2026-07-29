// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The two resolution rules vite and tsconfig's "bundler" mode give the app that bare node
// does not: the "@/*" path alias, and a relative import written without its extension.
// Register this from a test that imports a src module using either.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

function firstExisting(base) {
  for (const candidate of [`${base}.ts`, `${base}/index.ts`, base]) {
    if (existsSync(candidate)) {
      return pathToFileURL(candidate).href;
    }
  }
  return null;
}

export function resolve(specifier, context, next) {
  if (specifier.startsWith("@/")) {
    const resolved = firstExisting(SRC + specifier.slice(2));
    return next(resolved ?? specifier, context);
  }
  if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
    const resolved = firstExisting(
      fileURLToPath(new URL(specifier, context.parentURL)),
    );
    if (resolved) {
      return next(resolved, context);
    }
  }
  return next(specifier, context);
}
