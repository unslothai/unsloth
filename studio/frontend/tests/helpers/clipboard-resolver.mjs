// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Adds two rules to bundler-resolver: a stub for the Tauri clipboard plugin, and copying a
// "?bust=N" query onto every src import so module-level `isTauri` can be re-evaluated.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const SRC = fileURLToPath(new URL("../../src/", import.meta.url));
const CLIPBOARD_STUB = new URL("./tauri-clipboard-stub.mjs", import.meta.url).href;

function firstExisting(base) {
  for (const candidate of [`${base}.ts`, `${base}/index.ts`, base]) {
    if (existsSync(candidate)) return pathToFileURL(candidate).href;
  }
  return null;
}

function bustOf(parentURL) {
  if (!parentURL) return "";
  const bust = new URL(parentURL).searchParams.get("bust");
  return bust ? `?bust=${bust}` : "";
}

export function resolve(specifier, context, next) {
  const suffix = bustOf(context.parentURL);

  if (specifier === "@tauri-apps/plugin-clipboard-manager") {
    return next(CLIPBOARD_STUB + suffix, context);
  }
  if (specifier.startsWith("@/")) {
    const resolved = firstExisting(SRC + specifier.slice(2));
    return next(resolved ? resolved + suffix : specifier, context);
  }
  if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
    const parent = new URL(context.parentURL);
    parent.search = "";
    const resolved = firstExisting(fileURLToPath(new URL(specifier, parent)));
    if (resolved) return next(resolved + suffix, context);
  }
  return next(specifier, context);
}
