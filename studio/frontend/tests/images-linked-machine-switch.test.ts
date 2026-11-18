// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Picking a model on another machine moves progress, Stop and the gallery to that machine, so
// the move is refused while work is running, and the old machine's shelf is dropped and refetched.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const src = readSrc("features/images/images-page.tsx");
const start = src.indexOf("if (machine !== getImagesMachine()) {");
const block = src.slice(start, src.indexOf("const meta: ModelSelectorChangeMeta", start));

test("a machine switch is refused while a load or generation is running", () => {
  assert.ok(start > 0);
  const guard = block.indexOf("if (busy !== null) return;");
  assert.ok(guard > 0, "no busy guard in the machine switch");
  assert.ok(guard < block.indexOf("switchImagesMachine(machine);"));
});

test("a machine switch drops the old shelf, fences its listings and reloads", () => {
  const at = src.indexOf("const switchImagesMachine = useCallback(");
  assert.ok(at > 0);
  const helper = src.slice(at, src.indexOf("[loadGallery, refreshStatus]", at));
  for (const line of [
    "setImagesMachine(machine);",
    "stripEpoch.current += 1;",
    "galleryCache.images = [];",
    "setImages([]);",
    "void loadGallery();",
  ]) {
    assert.ok(helper.includes(line), line);
  }
});

test("a stored machine whose instance was removed falls back to this one", () => {
  assert.match(
    src,
    /state\.loadedAt > 0 &&\s+!state\.instances\.some\(\(i\) => i\.id === imagesMachine\)/,
  );
  assert.match(src, /if \(imagesMachineGone\) switchImagesMachine\(null\);/);
  assert.match(src, /if \(imagesMachine\) void refreshLinkedMachines\(true\);/);
});
