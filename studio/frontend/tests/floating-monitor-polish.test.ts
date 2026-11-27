// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import {
  liftAboveObstacle,
  nudgeDelta,
  snapsToAnchor,
} from "../src/hooks/floating-panel-position.ts";
import {
  STALE_AFTER_FAILED_POLLS,
  isReadingStale,
  nextFailedPolls,
} from "../src/hooks/system-discovery.ts";

test("one failed poll is a blip, two in a row make the reading stale", () => {
  let failed = 0;
  failed = nextFailedPolls(failed, false);
  assert.equal(isReadingStale("ready", failed, true), false);
  failed = nextFailedPolls(failed, false);
  assert.equal(failed, STALE_AFTER_FAILED_POLLS);
  assert.equal(isReadingStale("ready", failed, true), true);
  failed = nextFailedPolls(failed, true);
  assert.equal(isReadingStale("ready", failed, true), false);
});

test("only a polled, real reading can go stale", () => {
  assert.equal(
    isReadingStale("ready", 5, false),
    false,
    "nothing would clear it",
  );
  assert.equal(isReadingStale("pending", 5, true), false);
  assert.equal(isReadingStale("unavailable", 5, true), false);
});

const PANEL = { left: 1168, top: 684, width: 256, height: 200 };
const CARD = { left: 1156, top: 764, right: 1424, bottom: 884 };

test("at the default corner the monitor rests just above the card", () => {
  assert.equal(liftAboveObstacle(PANEL, CARD, 8), 764 - 8 - 200);
});

test("a card the composer lifted still has the monitor resting above it, not below over the composer", () => {
  const lifted = { left: 1156, top: 460, right: 1424, bottom: 549 };
  assert.equal(liftAboveObstacle(PANEL, lifted, 8), 460 - 8 - 200);
});

test("no card, or a card elsewhere, leaves the corner alone", () => {
  assert.equal(liftAboveObstacle(PANEL, null, 8), null);
  const dragged = { left: 40, top: 40, right: 308, bottom: 160 };
  assert.equal(liftAboveObstacle(PANEL, dragged, 8), null);
  const leftOfPanel = { left: 800, top: 764, right: 1100, bottom: 884 };
  assert.equal(liftAboveObstacle(PANEL, leftOfPanel, 8), null);
});

test("a drop near the resting or far corner snaps back to the anchor", () => {
  const rest = { left: 1152, top: 540 };
  const max = { left: 1152, top: 668 };
  assert.equal(snapsToAnchor({ left: 1140, top: 530 }, rest, max), true);
  assert.equal(snapsToAnchor({ left: 1150, top: 660 }, rest, max), true);
  assert.equal(snapsToAnchor({ left: 1100, top: 540 }, rest, max), false);
  assert.equal(snapsToAnchor({ left: 200, top: 200 }, rest, max), false);
});

test("arrow keys nudge by 8 px, 32 with Shift, and nothing else moves it", () => {
  assert.deepEqual(nudgeDelta("ArrowLeft", false), { dx: -8, dy: 0 });
  assert.deepEqual(nudgeDelta("ArrowDown", true), { dx: 0, dy: 32 });
  assert.equal(nudgeDelta("Enter", false), null);
  assert.equal(nudgeDelta("Escape", false), null);
});

test("a nudge makes only the moved axis absolute, so the other stays anchored", async () => {
  const { readFileSync } = await import("node:fs");
  const layout = readFileSync(
    new URL("../src/hooks/use-floating-panel-layout.ts", import.meta.url),
    "utf8",
  );
  assert.match(layout, /left: delta\.dx !== 0 \? left : previous\.left/);
  assert.match(layout, /top: delta\.dy !== 0 \? top : previous\.top/);
});
