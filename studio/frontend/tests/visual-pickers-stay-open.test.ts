// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A visual setting is chosen by comparing options, so its picker must not dismiss on every pick.
// The palette menu keeps itself open with preventDefault; the glass and background pickers use the
// RadioPicker popover, which never closes on selection. Select and DropdownMenu both close by
// design, so this guards against a picker drifting back to one of them.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PICKER = "components/ui/radio-picker.tsx";
const CONTROLS = "features/chat-background/controls.tsx";

test("the shared visual picker opens a popover, not a menu that closes on select", () => {
  const source = readSrc(PICKER);

  assert.match(source, /Popover/, "RadioPicker must be a Popover");
  assert.doesNotMatch(
    source,
    /from "@\/components\/ui\/(select|dropdown-menu)"/,
    "Select and DropdownMenu both close on selection",
  );
  assert.match(source, /role="radiogroup"/);
  assert.match(source, /role="radio"/);
  assert.match(source, /aria-checked/);
});

test("every glass and background option picker is the stay-open picker", () => {
  const source = readSrc(CONTROLS);

  assert.equal(
    (source.match(/<RadioPicker/g) ?? []).length,
    5,
    "glass style, glass scope, lens, background scope and background effect",
  );
  assert.doesNotMatch(
    source,
    /<Select[\s>]/,
    "a visual picker must not go back to Select, which dismisses on pick",
  );
});
