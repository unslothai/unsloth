// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("typing a Hub search clears an opened model but not the split view's auto-picked preview", () => {
  const src = readSrc("features/hub/hub-page.tsx");
  const at = src.indexOf("const handleQueryChange");
  assert.ok(at > 0);
  const handler = src.slice(at, src.indexOf("setQuery(next);", at));
  assert.match(
    handler,
    /const openedModel =\s*!!urlModel && urlModel !== autoPreviewIdRef\.current;/,
  );
  assert.match(handler, /if \(openedModel \|\| urlSection\) \{/);
  assert.match(src, /autoPreviewIdRef\.current = firstId;/);
});
