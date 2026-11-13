// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { selectableOptionStateClassName } from "../src/features/studio/sections/params-section-styles.ts";
import { readSrc } from "./helpers/kit.ts";

test("a selected target module or LoRA option is a lighter grey in dark mode, not green", () => {
  const selected = selectableOptionStateClassName(true);
  assert.doesNotMatch(selected, /emerald/);
  assert.match(
    selected,
    /dark:bg-\[color-mix\(in_srgb,var\(--foreground\)_16%,transparent\)\]/,
  );
  assert.match(
    selectableOptionStateClassName(false),
    /dark:bg-\[rgb\(255_255_255_\/_calc\(0\.05/,
  );
});

test("Train cards sink the toggle track and lift the selected pill in dark mode", () => {
  const css = readSrc("features/hub/hub.css");
  assert.match(
    css,
    /html\.dark \.elevated-card \{\s*--hub-tab-track: color-mix\(in srgb, var\(--background\) 55%, transparent\);/,
  );
  assert.match(
    css,
    /html\.dark \.elevated-card \.hub-tab-toggle-pill,\s*html\.dark \.elevated-card \.hub-tab-toggle-pill:hover \{\s*background-color: color-mix\(in srgb, var\(--accent\), white 7\.5%\);/,
  );
});
