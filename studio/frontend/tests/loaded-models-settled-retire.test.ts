// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A finished load is retired from `pending` once a status read has replaced its
// optimistic row with a real one. If that read could not see the source, there
// is no real row to replace it with: `readLoadedModels` is handed the polled
// rows only, and an optimistic row lives in `pending`, so it has nothing to
// preserve. Retiring anyway takes the row for a model that has just finished
// loading off the card, on the strength of a request that failed.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

const HOOK = read("../src/features/loaded-models/use-loaded-models.ts");
const API = read("../src/features/loaded-models/loaded-models-api.ts");

test("the read reports which sources it could not see", () => {
  assert.match(API, /export type LoadedModelsRead = \{/);
  assert.match(API, /unreadable: LoadedModelSource\[\];/);
  // Every source that fell back to the previous rows is named, so the caller
  // can tell "still there" from "confirmed gone".
  assert.match(API, /unreadable\.push\(source\);/);
  assert.match(API, /return \{ entries, unreadable \};/);
});

test("an unreadable source keeps its settled row", () => {
  const retire = HOOK.slice(
    HOOK.indexOf("const retireSettled = useCallback("),
    HOOK.indexOf("const refreshRef ="),
  );
  assert.match(retire, /unreadable: LoadedModelSource\[\] = \[\]/);
  assert.match(retire, /\.filter\(\s*\(source\) => !unreadable\.includes\(source\)/);
  // And it stays settled, so the next readable poll is the one that retires it.
  assert.match(retire, /settledRef\.current = new Set\(/);
});

test("a read that failed outright is evidence about nothing", () => {
  assert.match(HOOK, /const ALL_SOURCES: LoadedModelSource\[\] = \["chat", "image", "video", "stt"\]/);
  const refresh = HOOK.slice(
    HOOK.indexOf("void readLoadedModels(polledRef.current)"),
    HOOK.indexOf("}, [track, retireSettled]);"),
  );
  assert.match(refresh, /unreadable = ALL_SOURCES;/);
  assert.match(refresh, /retireSettled\(unreadable\)/);
});

test("a readable source is still retired, or the row would never go", () => {
  const retire = HOOK.slice(
    HOOK.indexOf("const retireSettled = useCallback("),
    HOOK.indexOf("const refreshRef ="),
  );
  // The early return is on nothing left to retire, not on anything unreadable.
  assert.match(retire, /if \(done\.length === 0\) return;/);
  assert.match(retire, /for \(const source of done\) next\.delete\(source\);/);
});
