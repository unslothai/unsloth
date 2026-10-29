// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

test("the Hub opens on Discover unless the URL names a tab", async () => {
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  assert.match(page, /const tab = urlTab \?\? "discover";/);
  assert.doesNotMatch(page, /unsloth\.hub\.modelsTab/);
});
