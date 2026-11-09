// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  assert.match(source, /RadioGroup as RadioGroupPrimitive.*from "radix-ui"/);
  assert.match(source, /<RadioGroupPrimitive.Root/);
  assert.match(source, /<RadioGroupPrimitive.Item/);
});

test("only style, lens and effect use the stay-open picker", () => {
  const source = readSrc(CONTROLS);

  const labels = [...source.matchAll(/<RadioPicker\s+ariaLabel=\{t\("([^"]+)"\)\}/g)]
    .map((match) => match[1]);
  assert.deepEqual(labels, [
    "chatBackground.materialStyle",
    "chatBackground.lens",
    "chatBackground.effect",
  ]);
  assert.equal((source.match(/<Select[\s>]/g) ?? []).length, 2);
});
