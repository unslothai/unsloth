// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { dialogDropdownGlow } from "../src/components/ui/dropdown-glow.ts";
import { readSrc } from "./helpers/kit.ts";

test("a card dialog surface sets the card glow, anything else keeps the one around it", () => {
  for (const className of [
    "dialog-soft-surface",
    "sm:max-w-lg settings-surface p-0",
    "bg-background dark:bg-card",
    "dark:bg-card/80 rounded-3xl",
  ]) {
    assert.equal(dialogDropdownGlow(className, null), "card", className);
    assert.equal(dialogDropdownGlow(className, "picker"), "card", className);
  }
  for (const className of [undefined, "", "sm:max-w-md", "settings-surface-muted", "bg-card"]) {
    assert.equal(dialogDropdownGlow(className, null), null, String(className));
    assert.equal(dialogDropdownGlow(className, "picker"), "picker", String(className));
  }
});

test("every menu surface carries the glow of the dialog or picker it opens from", () => {
  for (const [file, slots] of [
    ["components/ui/dropdown-menu.tsx", ["dropdown-menu-content", "dropdown-menu-sub-content"]],
    ["components/ui/context-menu.tsx", ["context-menu-content", "context-menu-sub-content"]],
    ["components/ui/select.tsx", ["select-content"]],
    ["components/ui/combobox.tsx", ["combobox-content"]],
  ] as const) {
    const source = readSrc(file);
    for (const slot of slots) {
      assert.match(
        source,
        new RegExp(`data-slot="${slot}"\\s*data-dropdown-glow=\\{useDropdownGlow\\(\\)\\}`),
        `${file}: ${slot}`,
      );
    }
  }
  const popover = readSrc("components/ui/popover.tsx");
  assert.match(popover, /data-slot="popover-content"\s*data-dropdown-glow=\{inheritedGlow \?\? undefined\}/);
  assert.match(
    popover,
    /<DropdownGlowContext\.Provider value=\{dropdownGlow \?\? inheritedGlow\}>\s*\{children\}/,
  );
  assert.match(
    readSrc("features/model-picker/components/model-selector.tsx"),
    /<PopoverContent\s+align="start"\s+alignOffset=\{10\}\s+dropdownGlow="picker"/,
  );
});

test("dialogs hand their surface to the menus opened inside them, without remounting their content", () => {
  const dialog = readSrc("components/ui/dialog.tsx");
  assert.match(
    dialog,
    /resolvedContainer === null\s*\? dialogDropdownGlow\(className, inheritedGlow\)\s*: inheritedGlow;/,
    "a dialog portaled into a container is not the body-level surface its menus sit beside",
  );
  assert.match(dialog, /<DropdownGlowContext\.Provider value=\{glow\}>\s*\{children\}/);
  const alert = readSrc("components/ui/alert-dialog.tsx");
  assert.match(alert, /dialogDropdownGlow\(className, useContext\(DropdownGlowContext\)\)/);
  assert.match(alert, /<DropdownGlowContext\.Provider value=\{glow\}>\s*\{children\}/);
});
