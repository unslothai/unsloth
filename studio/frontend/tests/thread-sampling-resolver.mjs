// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// store-settings-resolver, plus the two things a per-chat sampling simulation needs.
//
// 1. The thread row: the store reaches it through a dynamic relative import the "@/" stub
//    map does not cover, so it has to be intercepted here.
// 2. Scenario scoping: the store keeps the whole feature in module state, so a simulation
//    running hundreds of orderings needs a fresh copy per ordering. A "?scenario=N" query
//    gives one, and the modules listed below are re-instantiated with it so they talk to
//    THAT copy. Everything else stays shared, which is what makes the stubs observable.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

import { resolve as resolveSettings } from "./store-settings-resolver.mjs";

const ROW_STUB = new URL(
  "./helpers/store-stubs/chat-history-storage.ts",
  import.meta.url,
).href;

/** Modules that must follow the scenario's store instance, not the shared one. */
const SCENARIO_SCOPED = new Set([
  new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url)
    .href,
  new URL("../src/features/chat/utils/qwen-params.ts", import.meta.url).href,
]);

function firstExisting(base) {
  for (const candidate of [`${base}.ts`, `${base}/index.ts`, base]) {
    if (existsSync(candidate)) {
      return pathToFileURL(candidate).href;
    }
  }
  return null;
}

export function resolve(specifier, context, next) {
  if (specifier.endsWith("utils/chat-history-storage")) {
    return next(ROW_STUB, context);
  }
  const scenario = context.parentURL?.startsWith("file:")
    ? new URL(context.parentURL).searchParams.get("scenario")
    : null;
  if (scenario !== null && specifier.startsWith(".")) {
    // Relative resolution drops the parent's query, so this is the plain path.
    const target = firstExisting(
      fileURLToPath(new URL(specifier, context.parentURL)),
    );
    if (target !== null && SCENARIO_SCOPED.has(target)) {
      return next(`${target}?scenario=${scenario}`, context);
    }
  }
  return resolveSettings(specifier, context, next);
}
