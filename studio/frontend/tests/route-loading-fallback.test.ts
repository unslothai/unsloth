// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("route loading retains the original centered loading label", () => {
  const source = readSrc("app/routes/__root.tsx");
  const fallback = source.match(/function RouteFallback\(\) \{[\s\S]*?\n\}/)?.[0];
  assert.ok(fallback);
  assert.match(fallback, /flex h-full min-h-0 flex-1 items-center justify-center text-muted-foreground text-sm/);
  assert.match(fallback, />\s*\{t\("common\.loading"\)\}\s*<\/div>/);
});
