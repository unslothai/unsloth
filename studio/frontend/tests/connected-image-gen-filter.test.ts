// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

test("an image picker's connected rows all pass the Image gen filter", async () => {
  const src = await readText(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
  );
  const at = src.indexOf("const connectedMatches = useMemo");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("const marks = connectedModelMarks", at));
  assert.match(
    body,
    /if \(connectedModality === "imageGen" && imageGenPicker\) return true;/,
  );
  assert.match(
    src,
    /const imageGenPicker = useMemo\(\(\) => \{[\s\S]*?IMAGE_GEN_TASKS\.some\(\(t\) => tasks\.includes\(t\)\)/,
  );
});
