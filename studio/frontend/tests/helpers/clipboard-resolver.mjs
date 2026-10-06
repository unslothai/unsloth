// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// bundler-resolver's two rules, plus the two this suite needs:
//
//   1. "@tauri-apps/plugin-clipboard-manager" -> a local stub, because the real
//      package only resolves inside a Tauri webview.
//   2. A "?bust=N" query on the importer is copied onto every src module it pulls
//      in. lib/api-base computes `isTauri` once, at module evaluation, so the only
//      way to exercise both branches in one process is to evaluate the whole
//      subgraph again under a fresh key.
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
