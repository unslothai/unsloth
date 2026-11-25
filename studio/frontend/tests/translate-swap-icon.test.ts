// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const page = readFileSync(
  new URL("../src/features/translate/translate-page.tsx", import.meta.url),
  "utf8",
);

test("Translate's swap button draws the Keyline swap icon on both layouts", () => {
  assert.doesNotMatch(page, /ArrowLeftRightIcon/);
  assert.equal(page.match(/<SwapArrowsIcon /g)?.length, 2);
});
