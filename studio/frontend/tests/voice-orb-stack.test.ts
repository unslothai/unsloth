// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ORB_BALL_EXTENT,
  ORB_CHIP_HEIGHT,
  ORB_RING_EXTENT,
  ORB_STACK_GAP,
  ORB_STATUS_HEIGHT,
  orbStackLayout,
} from "../src/features/chat/voice/orb-stack-layout.ts";

test("every combination of arcs and chip keeps the same gap down the whole stack", () => {
  for (const arcs of [true, false]) {
    for (const chip of [true, false]) {
      const l = orbStackLayout({ arcs, chip, card: true });
      assert.equal(
        l.status,
        (arcs ? ORB_RING_EXTENT : ORB_BALL_EXTENT) + ORB_STACK_GAP,
      );
      if (chip) {
        assert.equal(l.chip, l.status + ORB_STATUS_HEIGHT + ORB_STACK_GAP);
        assert.equal(
          l.card,
          (l.chip as number) + ORB_CHIP_HEIGHT + ORB_STACK_GAP,
        );
      } else {
        assert.equal(l.chip, null);
        assert.equal(l.card, l.status + ORB_STATUS_HEIGHT + ORB_STACK_GAP);
      }
    }
  }
});

test("arcs off pulls the stack up under the ball; the chip hidden pulls the card up", () => {
  const all = orbStackLayout({ arcs: true, chip: true, card: true });
  const noArcs = orbStackLayout({ arcs: false, chip: true, card: true });
  const noChip = orbStackLayout({ arcs: true, chip: false, card: true });
  assert.ok(
    noArcs.status < all.status &&
      (noArcs.card as number) < (all.card as number),
  );
  assert.equal(noChip.status, all.status);
  assert.equal(noChip.card, all.chip);
  assert.equal(
    orbStackLayout({ arcs: true, chip: true, card: false }).card,
    null,
  );
});
