// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A palette is chosen by comparing options, so a pick must not dismiss the list. Radix
// DropdownMenu closes on select unless the handler prevents it, and that includes tearing down the
// submenu the pick came from.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PALETTE = "features/settings/components/color-theme-select.tsx";

test("the palette menu keeps itself open on select", () => {
  const source = readSrc(PALETTE);

  assert.match(
    source,
    /onSelect=\{\(event\) => \{\s*event\.preventDefault\(\);\s*setPalette\(id\);/,
    "without preventDefault the menu closes on every pick",
  );
  assert.doesNotMatch(
    source,
    /onSelect=\{\(\) => setPalette\(/,
    "the plain handler is what regressed in the first place",
  );
});
