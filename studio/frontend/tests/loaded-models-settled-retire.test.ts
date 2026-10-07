// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An unreadable status read cannot retire an optimistic row: there is nothing to replace it.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const HOOK = readText("../src/features/loaded-models/use-loaded-models.ts");
const API = readText("../src/features/loaded-models/loaded-models-api.ts");

test("the read reports which sources it could not see", () => {
  assert.match(API, /export type LoadedModelsRead = \{/);
  assert.match(API, /unreadable: LoadedModelSource\[\];/);
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
  assert.match(retire, /if \(done\.length === 0\) return;/);
  assert.match(retire, /for \(const source of done\) next\.delete\(source\);/);
});
