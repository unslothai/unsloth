// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveTooltipOpen } from "../src/components/ui/tooltip-open-state.ts";

const base = {
  blocked: false,
  dismissedUntilOwnerResets: false,
  hoverOpen: false,
  clickOpen: false,
};

test("hover and a tap pin both open an uncontrolled tooltip", () => {
  assert.equal(resolveTooltipOpen({ ...base, hoverOpen: true }), true);
  assert.equal(resolveTooltipOpen({ ...base, clickOpen: true }), true);
  assert.equal(resolveTooltipOpen(base), false);
});

test("a modal shuts one whose trigger is outside it", () => {
  assert.equal(
    resolveTooltipOpen({ ...base, blocked: true, hoverOpen: true }),
    false,
  );
  assert.equal(
    resolveTooltipOpen({ ...base, blocked: true, controlledOpen: true }),
    false,
  );
});

test("a controlled tooltip follows its owner", () => {
  assert.equal(resolveTooltipOpen({ ...base, controlledOpen: true }), true);
  assert.equal(resolveTooltipOpen({ ...base, controlledOpen: false }), false);
});

test("a controlled tooltip stays shut after a modal until its owner resets", () => {
  // The resize handle keeps `hovered` true because no pointerleave arrived, so
  // honouring `open` again would put the tooltip back with the pointer gone.
  assert.equal(
    resolveTooltipOpen({
      ...base,
      controlledOpen: true,
      dismissedUntilOwnerResets: true,
    }),
    false,
  );
});

test("an uncontrolled tooltip is not held by that latch", () => {
  // Its own hover state was cleared when the modal opened, so a real hover
  // afterwards must show it immediately.
  assert.equal(
    resolveTooltipOpen({
      ...base,
      hoverOpen: true,
      dismissedUntilOwnerResets: true,
    }),
    true,
  );
});
