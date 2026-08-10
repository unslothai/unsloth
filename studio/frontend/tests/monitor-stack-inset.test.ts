// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  type MonitorFrame,
  stackBottomInset,
  stackGeometry,
} from "../src/features/settings/stores/monitor-frame-store.ts";

const W = 1440;
const H = 900;

/** The Live monitor where it opens by default: bottom-right, w-64, inset-4. */
function corner(height = 300): MonitorFrame {
  return {
    left: W - 16 - 256,
    top: H - 16 - height,
    right: W - 16,
    bottom: H - 16,
  };
}

test("with no monitor open the stack keeps its own inset", () => {
  assert.equal(stackBottomInset(null, W, H), 16);
});

// The reported overlap: both default to the same corner.
test("a monitor in its default corner lifts the stack clear of it", () => {
  const frame = corner(300);
  const inset = stackBottomInset(frame, W, H);
  // The stack's bottom edge now sits above the monitor's top edge.
  assert.ok(inset > 16, "the stack must move");
  assert.ok(H - inset <= frame.top, "no vertical overlap remains");
});

test("a monitor dragged to the left is not dodged", () => {
  const frame: MonitorFrame = {
    left: 16,
    top: H - 316,
    right: 272,
    bottom: H - 16,
  };
  assert.equal(stackBottomInset(frame, W, H), 16);
});

test("a monitor dragged to the top right is not dodged", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: 316,
  };
  assert.equal(stackBottomInset(frame, W, H), 16);
});

// The update banners are max-w-[448px], wider than the download panel, so the
// column has to be measured from them.
test("a monitor beside the download panel but under a banner is dodged", () => {
  const frame: MonitorFrame = {
    left: W - 16 - 430,
    top: H - 316,
    right: W - 16 - 410,
    bottom: H - 16,
  };
  assert.ok(stackBottomInset(frame, W, H) > 16);
});

// Lifting the stack without shortening it pushes its top off the screen.
test("lifting the stack shortens it by the same amount", () => {
  assert.equal(stackGeometry(null, W, H).maxHeight, H - 32);
  const lifted = stackGeometry(corner(300), W, H);
  assert.ok(lifted.bottom > 16);
  assert.equal(lifted.maxHeight, H - lifted.bottom - 16);
});

// A monitor parked high is not lifted over, since the room is beneath it, but
// the stack grows upwards and would still run into it.
test("a monitor high in the column caps the stack instead of lifting it", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: 316,
  };
  const geometry = stackGeometry(frame, W, H);
  // Nothing to lift over: the free space is below it.
  assert.equal(geometry.bottom, 16);
  // The stack's top edge stays clear of the monitor's bottom edge.
  assert.ok(H - geometry.bottom - geometry.maxHeight >= frame.bottom);
  assert.ok(geometry.maxHeight < H - 32, "it must actually be capped");
});

test("a monitor high but outside the column does not cap the stack", () => {
  const frame: MonitorFrame = { left: 16, top: 16, right: 272, bottom: 316 };
  assert.equal(stackGeometry(frame, W, H).maxHeight, H - 32);
});

// A monitor filling almost the whole column would otherwise leave no stack.
test("the cap never shrinks the stack below its floor", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 0,
    right: W - 16,
    bottom: H / 2 - 1,
  };
  assert.ok(stackGeometry(frame, W, H).maxHeight >= 120);
});

// Clamping the lift to the stack's floor was worse than not lifting: it put
// the stack across the monitor's own top edge, which is where its Close and
// collapse controls are. The chat UI suite maximises the monitor and then
// clicks Close, and the card swallowed the click.
//
// Nothing above such a monitor is free, so the stack goes inside it, and the
// two things it may not bury are the header at the top and the native resize
// grip in the bottom-right corner (Chromium's hit area reaches 15px in from
// that corner, Firefox's 12px). Both edges are checked, because the inset
// alone does not hold the top: an expanded download list plus the loaded
// models card grows from the bottom right back over the header.
/** Where the stack's own edges land, given everything it dodges. */
function stackEdges(frame: MonitorFrame) {
  const { bottom, maxHeight } = stackGeometry(frame, W, H);
  return { bottom, top: H - bottom - maxHeight, edge: H - bottom };
}

test("a monitor too tall to lift over keeps its header and grip clear", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: H - 16,
  };
  const { top, edge } = stackEdges(frame);
  assert.ok(
    edge <= frame.bottom - 16,
    "the stack sits clear of the resize grip",
  );
  assert.ok(top >= frame.top + 64, "and stops below the header controls");
});

test("a monitor resized to fill the viewport keeps its header and grip clear", () => {
  const frame: MonitorFrame = {
    left: 16,
    top: 16,
    right: W - 16,
    bottom: H - 16,
  };
  const { bottom, top, edge } = stackEdges(frame);
  assert.ok(
    edge <= frame.bottom - 16,
    "the stack sits clear of the resize grip",
  );
  assert.ok(top >= frame.top + 64, "and stops below the header controls");
  // Seated inside the monitor near its foot, not lifted to the stack's floor,
  // which is the whole of the difference and the only place both edges fit.
  assert.ok(bottom < H / 2, "the stack stays at the bottom of the screen");
});

// Tall enough that a lift over it would not fit, but it stops short of the
// bottom edge, so it never crowds the stack's corner to begin with. The stack
// keeps that corner and is capped to the room underneath rather than being
// pushed up inside the monitor.
test("a monitor too tall to lift over but clear of the bottom is sat under", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: H / 2 + 100,
  };
  const { bottom, top } = stackEdges(frame);
  assert.equal(bottom, 16, "the corner is free, so stay in it");
  assert.ok(top >= frame.bottom, "and the stack stays underneath the monitor");
});

// The lift is dropped only when it cannot clear; one that fits still applies.
test("a tall monitor that can still be cleared is lifted over", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 200,
    right: W - 16,
    bottom: H - 16,
  };
  const inset = stackBottomInset(frame, W, H);
  assert.ok(inset > 16, "the stack must move");
  assert.ok(H - inset <= frame.top, "no vertical overlap remains");
  assert.ok(H - inset - 16 >= 120, "and it keeps its floor");
});

// The union was the trap. A tall monitor and the wide docked composer share
// almost no area, so the rectangle around the pair covers most of the viewport;
// reading that as one obstacle lifted the stack to the top of the screen and
// dropped it back onto the monitor it was meant to dodge, which the chat UI
// suite caught as the card swallowing the monitor's Close button.
test("two obstacles are folded one at a time, not as their bounding box", () => {
  // The monitor dragged up the column, as the chat UI suite does before it
  // clicks Close: too high to be lifted over, so on its own it asks for
  // nothing. The composer, docked, asks for a modest lift.
  const monitor = { left: W - 16 - 256, top: 40, right: W - 16, bottom: 340 };
  const composer = { left: 300, top: H - 120, right: W - 340, bottom: H - 40 };
  const both = stackGeometry([monitor, composer], W, H);
  const monitorOnly = stackGeometry(monitor, W, H);
  const composerOnly = stackGeometry(composer, W, H);
  assert.equal(
    both.bottom,
    Math.max(monitorOnly.bottom, composerOnly.bottom),
    "the stack takes the largest lift either one asks for",
  );
  // The union's own answer, which is what went wrong.
  const unioned = stackGeometry(
    {
      left: Math.min(monitor.left, composer.left),
      top: Math.min(monitor.top, composer.top),
      right: Math.max(monitor.right, composer.right),
      bottom: Math.max(monitor.bottom, composer.bottom),
    },
    W,
    H,
  );
  assert.notEqual(
    both.bottom,
    unioned.bottom,
    "folding must not agree with the bounding box, or nothing was fixed",
  );
  // The union is wrong in whichever direction its own answer happens to go: it
  // used to lift to the cap and land on the monitor, and a box that tall is now
  // seated inside itself, which asks for less than the composer does on its own
  // and puts the card back over the Send button. Folding gives each box what it
  // asked for.
  assert.equal(
    both.bottom,
    composerOnly.bottom,
    "the composer still gets the lift it asked for",
  );
});

test("an empty list behaves exactly like nothing published", () => {
  assert.deepEqual(stackGeometry([], W, H), stackGeometry(null, W, H));
});

test("one box in a list matches passing it on its own", () => {
  const frame = corner(300);
  assert.deepEqual(stackGeometry([frame], W, H), stackGeometry(frame, W, H));
});

// The two composer layouts, which are what this gate exists for. Boxes taken
// from a 1280x830 window: the docked one has to be dodged, or the card covers
// Send, which below a 1584px viewport it does. The welcome one must not be,
// because it sits high on the page and lifting over it stranded the banners in
// the middle of the screen with the corner underneath them empty.
const CHAT_W = 1280;
const CHAT_H = 830;

test("a docked composer is dodged, so the card cannot cover Send", () => {
  const docked = { left: 412, top: 664, right: 1148, bottom: 814 };
  const inset = stackBottomInset(docked, CHAT_W, CHAT_H);
  assert.ok(inset > 16, "it reaches the stack's strip, so the stack lifts");
  assert.ok(
    inset >= CHAT_H - docked.top,
    "and lifts clear of it, not part way",
  );
  // Still the bottom of the screen, which is the point: above the composer,
  // not adrift in the middle.
  assert.ok(inset < CHAT_H / 2);
});

test("the welcome composer is left alone, and the stack stays in the corner", () => {
  const welcome = { left: 412, top: 435, right: 1148, bottom: 660 };
  assert.equal(stackBottomInset(welcome, CHAT_W, CHAT_H), 16);
});

// A short window, where the welcome composer really does sit close to the
// bottom: 921x534, the reported case. It leaves 83px under it, and the loaded
// models card is 80px tall, so the corner it is being lifted out of is the one
// place it fits. Asking for a fixed 120 lifted the card clear over the
// composer's top edge and parked it in the middle of the screen.
const SHORT_W = 921;
const SHORT_H = 534;
const SHORT_WELCOME = { left: 316, top: 308, right: 877, bottom: 427 };

test("a card that fits under the welcome composer keeps the corner", () => {
  const geometry = stackGeometry(SHORT_WELCOME, SHORT_W, SHORT_H, 80);
  assert.equal(geometry.bottom, 16, "the card belongs in the corner");
  const stackTop = SHORT_H - geometry.bottom - geometry.maxHeight;
  assert.ok(stackTop >= SHORT_WELCOME.bottom, "and still clears the composer");
  assert.ok(geometry.maxHeight >= 80, "with room for the card it measured");
});

test("a stack too tall for that gap is still lifted over", () => {
  const inset = stackBottomInset(SHORT_WELCOME, SHORT_W, SHORT_H, 200);
  assert.ok(inset > 16, "200px cannot fit in 83px, so it has to move");
  assert.ok(SHORT_H - inset <= SHORT_WELCOME.top, "and clears it fully");
});

test("a docked composer is dodged whatever the stack measures", () => {
  const docked = { left: 412, top: 664, right: 1148, bottom: 814 };
  for (const height of [40, 80, 120, 260]) {
    assert.ok(
      stackBottomInset(docked, CHAT_W, CHAT_H, height) > 16,
      `a ${height}px stack must still clear Send`,
    );
  }
});

// The measurement feeds the cap, and the cap is on the element being measured.
// The overlays are flex items with min-h-0, so under the cap they shrink to it
// and both clientHeight and scrollHeight come back as the cap: a stack taller
// than the gap would measure as exactly the gap, never ask for a lift, and sit
// there clipped. The read has to be taken with the cap off.
test("the height is measured with the hook's own cap lifted", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/stores/monitor-frame-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const measure = source.slice(
    source.indexOf("const measure = () => {"),
    source.indexOf("observer.observe(node)"),
  );
  assert.ok(measure, "the measure callback moved");
  assert.match(
    measure,
    /node\.style\.maxHeight = "none";[\s\S]*node\.scrollHeight/,
    "scrollHeight is read while the cap still applies",
  );
  assert.match(
    measure,
    /node\.scrollHeight;[\s\S]*node\.style\.maxHeight = capped;/,
    "the cap is not restored after the read",
  );
});

// The sweep above, re-run at the heights the stack actually takes, since the
// dodge test is now driven by them.
test("no measured height leaves the stack overlapping a box", () => {
  const composer = { left: 412, top: 664, right: 1148, bottom: 814 };
  for (const needed of [0, 40, 80, 120, 200, 320]) {
    for (let bottom = 60; bottom <= CHAT_H - 16; bottom += 4) {
      const box = { left: 996, top: Math.max(0, bottom - 180), right: 1264, bottom };
      for (const boxes of [[box], [box, composer]]) {
        const geometry = stackGeometry(boxes, CHAT_W, CHAT_H, needed);
        const label = `needed=${needed} bottom=${bottom} n=${boxes.length}`;
        assert.ok(geometry.maxHeight > 0, `non-positive cap for ${label}`);
        const stackTop = CHAT_H - geometry.bottom - geometry.maxHeight;
        for (const each of boxes) {
          const clearsAbove = CHAT_H - geometry.bottom <= each.top;
          const clearsBelow = stackTop >= each.bottom;
          assert.ok(clearsAbove || clearsBelow, `overlap for ${label}`);
        }
      }
    }
  }
});

// Same rule, applied to the other publisher: a monitor dragged up the screen
// leaves the corner free, so the stack belongs in it.
test("a monitor away from the corner no longer lifts the stack", () => {
  const middle = { left: 996, top: 300, right: 1264, bottom: 560 };
  const geometry = stackGeometry(middle, CHAT_W, CHAT_H);
  assert.equal(geometry.bottom, 16);
  // It is still in the column, so the stack is capped short of it instead.
  assert.ok(geometry.maxHeight < CHAT_H - 16 - 16);
  assert.ok(geometry.maxHeight <= CHAT_H - 16 - middle.bottom);
});

// The capped branch used to be reachable with a floor that did not fit: a box
// ending just above the old cutoff was left uncapped-in-practice, and the
// stack's guaranteed MIN_STACK_ROOM put its top back over the box by up to
// 24px. The cutoff is derived from the cap now, so the two cannot disagree.
test("a capped stack always fits under the box it is capped by", () => {
  for (let bottom = 100; bottom <= CHAT_H - 16; bottom += 1) {
    const frame = {
      left: 996,
      top: Math.max(0, bottom - 260),
      right: 1264,
      bottom,
    };
    const geometry = stackGeometry(frame, CHAT_W, CHAT_H);
    // Lifted boxes sit under the stack by design; only the capped ones apply.
    if (geometry.bottom !== 16) continue;
    const stackTop = CHAT_H - geometry.bottom - geometry.maxHeight;
    assert.ok(
      stackTop >= bottom,
      `a box ending at ${bottom} left the stack top at ${stackTop}`,
    );
  }
});

// Reachability has to be asked at the inset actually in force. Folding each box
// against the default one, a monitor with room at the corner kept the capped
// branch after the composer had already lifted the stack into it, and the cap
// came out negative. Browsers drop an invalid max-height, so the cap vanished
// exactly where it was needed.
test("a box the shared lift moves the stack into is lifted over too", () => {
  const composer = { left: 412, top: 664, right: 1148, bottom: 814 };
  const monitor = { left: 996, top: 400, right: 1264, bottom: 660 };
  assert.equal(
    stackGeometry(monitor, CHAT_W, CHAT_H).bottom,
    16,
    "on its own the monitor leaves the corner free",
  );
  const both = stackGeometry([monitor, composer], CHAT_W, CHAT_H);
  assert.ok(both.maxHeight > 0, "a negative cap is dropped by the browser");
  assert.ok(
    CHAT_H - both.bottom <= monitor.top,
    "the stack has to clear the monitor, not just the composer",
  );
});

// The same, swept: no arrangement may leave the stack overlapping a box, and no
// cap may come out at zero or below.
test("no pairing produces an overlap or an unusable cap", () => {
  const composer = { left: 412, top: 664, right: 1148, bottom: 814 };
  for (let bottom = 60; bottom <= CHAT_H - 16; bottom += 2) {
    for (const height of [80, 180, 280, 400]) {
      const box = {
        left: 996,
        top: Math.max(0, bottom - height),
        right: 1264,
        bottom,
      };
      for (const boxes of [[box], [box, composer]]) {
        const geometry = stackGeometry(boxes, CHAT_W, CHAT_H);
        const label = `bottom=${bottom} height=${height} n=${boxes.length}`;
        assert.ok(geometry.maxHeight > 0, `non-positive cap for ${label}`);
        const stackTop = CHAT_H - geometry.bottom - geometry.maxHeight;
        for (const each of boxes) {
          const clearsAbove = CHAT_H - geometry.bottom <= each.top;
          const clearsBelow = stackTop >= each.bottom;
          assert.ok(clearsAbove || clearsBelow, `overlap for ${label}`);
        }
      }
    }
  }
});
