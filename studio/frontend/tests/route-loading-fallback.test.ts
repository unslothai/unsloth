// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("route loading preserves layout and accessible status without flashing text", () => {
  const source = readSrc("app/routes/__root.tsx");
  const fallback = source.match(/function RouteFallback\(\) \{[\s\S]*?\n\}/)?.[0];
  assert.ok(fallback);
  assert.match(fallback, /h-full min-h-0 flex-1/);
  assert.match(fallback, /role="status"/);
  assert.match(fallback, /aria-label=\{t\("common\.loading"\)\}/);
  assert.doesNotMatch(fallback, />\s*\{t\("common\.loading"\)\}/);
});
