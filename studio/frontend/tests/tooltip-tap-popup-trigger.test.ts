// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { opensPopup } from "../src/components/ui/touch-click.ts";

const el = (attrs: Record<string, string>) => ({
  getAttribute: (name: string) => attrs[name] ?? null,
});

test("popover and menu triggers open a popup", () => {
  assert.equal(opensPopup(el({ "aria-haspopup": "dialog" })), true);
  assert.equal(opensPopup(el({ "aria-haspopup": "menu" })), true);
  assert.equal(opensPopup(el({ "aria-haspopup": "true" })), true);
});

test("plain buttons still get the tap-pinned tooltip", () => {
  assert.equal(opensPopup(el({})), false);
  assert.equal(opensPopup(el({ "aria-haspopup": "false" })), false);
  assert.equal(opensPopup(null), false);
});

// issue 18298: tap pin skips PopoverTrigger, since preventDefault cancels the popover open.
test("the tap pin skips popup triggers before preventDefault", () => {
  const src = readFileSync(
    new URL("../src/components/ui/tooltip.tsx", import.meta.url),
    "utf8",
  );
  const guard = src.indexOf("opensPopup(e.currentTarget)");
  const prevent = src.indexOf("e.preventDefault();", guard);
  assert.ok(guard > 0 && prevent > guard);
});
