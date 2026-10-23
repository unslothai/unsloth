// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("a drag that loses pointer capture without a pointerup still ends, so its overlay cannot swallow the UI", () => {
  const source = readSrc("components/ui/panel-resize-handle.tsx");
  assert.match(source, /onPointerCancel=\{endDrag\}/);
  assert.match(
    source,
    /onLostPointerCapture=\{\(\) => \{\s*if \(dragRef\.current\) endDrag\(\)\s*\}\}/,
    "only a drag still in progress: a normal pointerup has already ended it, and the capture it releases is lost right after",
  );
  const up = source.slice(
    source.indexOf("const handlePointerUp"),
    source.indexOf("const handleKeyDown"),
  );
  assert.ok(
    up.indexOf("releasePointerCapture") < up.indexOf("endDrag()"),
    "pointerup releases capture before ending the drag, so the commit below reads the drag it captured first",
  );
  assert.match(up, /const drag = dragRef\.current/);
});
