// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  type MonitorFrame,
  stackBottomInset,
  stackGeometry,
  usableFloorRoom,
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

// The reported case: a 921x534 window, where the welcome composer leaves 83px
// under it and the 80px card fits in the corner it was being lifted out of.
// Asking for a fixed 120 parked it in the middle of the screen instead.
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

// Two active downloads on the welcome screen of a 1600x891 window. The rail is
// taller than the 163px band under the composer but its cards scroll, so the
// height it would PREFER is not worth the corner: dodging for it left the panel
// mid-window with the corner under it empty.
const WIDE_W = 1600;
const WIDE_H = 891;
const WIDE_WELCOME = {
  left: 532,
  top: 524,
  right: 1520,
  bottom: 704,
  coverable: true,
};

test("a rail that scrolls into the welcome composer's band keeps the corner", () => {
  const geometry = stackGeometry(WIDE_WELCOME, WIDE_W, WIDE_H, 328, 140);
  assert.equal(geometry.bottom, 16, "the panel belongs in the corner");
  assert.ok(geometry.maxHeight >= 140, "and holds the cards at their floor");
  const stackTop = WIDE_H - geometry.bottom - geometry.maxHeight;
  assert.ok(stackTop >= WIDE_WELCOME.bottom, "without reaching the composer");
});

test("a rail whose floor will not fit that band is still lifted over", () => {
  const geometry = stackGeometry(WIDE_WELCOME, WIDE_W, WIDE_H, 328, 300);
  assert.ok(geometry.bottom > 16, "300px cannot fit in 163px, so it moves");
  assert.ok(
    WIDE_H - geometry.bottom <= WIDE_WELCOME.top,
    "and clears the composer fully",
  );
});

test("the docked composer is dodged however little the rail insists on", () => {
  const docked = {
    left: 412,
    top: 664,
    right: 1148,
    bottom: 814,
    coverable: true,
  };
  for (const floor of [1, 40, 120]) {
    assert.ok(
      stackBottomInset(docked, CHAT_W, CHAT_H, 260, floor) > 16,
      `a rail with a ${floor}px floor must still clear Send`,
    );
  }
});

// The published form leaves a 5px strip above the footnote. A border-only floor
// used to accept that strip and clip the download panel into a hairline.
const NOTE_H = 800;
const NOTE_DOCKED: MonitorFrame = {
  left: 407,
  top: 658.53,
  right: 1143,
  bottom: 770.53,
  coverable: true,
};
// One download row under the panel's header, measured in the same window.
const ONE_CARD = 121;
// The download panel alone in the rail, which is what the report had.
const alone = (squeezed: number, natural: number) =>
  usableFloorRoom([{ squeezed, natural }], natural);

test("the strip the composer's footnote leaves is not room for the rail", () => {
  const geometry = stackGeometry(
    NOTE_DOCKED,
    CHAT_W,
    NOTE_H,
    ONE_CARD,
    alone(2, ONE_CARD),
  );
  assert.ok(geometry.bottom > 16, "5px under the note is not a placement");
  assert.ok(
    NOTE_H - geometry.bottom <= NOTE_DOCKED.top,
    "so the rail clears the composer",
  );
  assert.ok(geometry.maxHeight >= ONE_CARD, "and the card fits where it lands");
});

test("the floor is a card the reader can see, not what the rail can shrink to", () => {
  assert.equal(alone(2, ONE_CARD), ONE_CARD);
  assert.equal(alone(2, 40), 40);
  assert.equal(alone(300, 480), 300);
  // Long lists still scroll rather than asking for their full height.
  assert.ok(alone(2, 480) < 480);
});

// The same window with the app update card up. Its min-height is its own floor,
// so a rail floored as one number is already past the minimum while the panel
// beside it is still measuring its borders.
const BANNER_FLOOR = 189;
const BANNER_NATURAL = 196;
const PANEL_BORDERS = 2;
const RAIL_CARDS = [
  { squeezed: BANNER_FLOOR, natural: BANNER_NATURAL },
  { squeezed: PANEL_BORDERS, natural: ONE_CARD },
];
const RAIL_NATURAL = BANNER_NATURAL + 8 + ONE_CARD;

test("a banner's own floor does not answer for the panel beside it", () => {
  const floor = usableFloorRoom(RAIL_CARDS, RAIL_NATURAL);
  assert.ok(
    floor > BANNER_FLOOR + 8 + PANEL_BORDERS,
    "the panel is asked for as well as the banner",
  );
  // Shorter than the minimum, so it is asked for whole rather than for room it
  // could not fill.
  assert.equal(floor, BANNER_FLOOR + 8 + ONE_CARD);
});

// A 1366x768 laptop, where the browser leaves a 640px viewport: the welcome
// composer sits with exactly 200px under it, which holds the banner and left
// the download panel an 8px strip with no scrollbar to reach it by.
const NOTE_W = 1366;
const NOTE_WELCOME_H = 640;
const NOTE_WELCOME: MonitorFrame = {
  left: 450,
  top: 304,
  right: 1186,
  bottom: 416,
  coverable: true,
};

test("a band that only holds the banner is not room for the rail", () => {
  const geometry = stackGeometry(
    NOTE_WELCOME,
    NOTE_W,
    NOTE_WELCOME_H,
    RAIL_NATURAL,
    usableFloorRoom(RAIL_CARDS, RAIL_NATURAL),
    ONE_CARD + 8,
  );
  assert.ok(
    NOTE_WELCOME_H - geometry.bottom - geometry.maxHeight >= 0,
    "the rail stays on the page",
  );
  assert.ok(
    geometry.maxHeight - BANNER_FLOOR - 8 >= ONE_CARD,
    "and the panel is more than its borders",
  );
});

// Preserve #8462: the 163px welcome-screen band still keeps the corner.
test("the welcome band still keeps the corner under a real floor", () => {
  const geometry = stackGeometry(
    WIDE_WELCOME,
    WIDE_W,
    WIDE_H,
    328,
    alone(2, 328),
  );
  assert.equal(geometry.bottom, 16, "the panel belongs in the corner");
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

// The measurement feeds the cap, and the cap is on the element measured. The
// overlays are min-h-0 flex items, so under the cap they shrink to it and
// scrollHeight reports the cap: a stack taller than the gap would measure as
// the gap, never ask for a lift, and sit clipped. So read with the cap off.
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
    source.indexOf("const observer = new ResizeObserver"),
  );
  assert.ok(measure, "the measure callback moved");
  assert.match(
    measure,
    /node\.style\.maxHeight = "none";[\s\S]*node\.scrollHeight/,
    "scrollHeight is read while the cap still applies",
  );
  assert.match(
    measure,
    /node\.scrollHeight[\s\S]*node\.style\.maxHeight = capped;/,
    "the cap is not restored after the read",
  );
});

// A rect is read through the update cards' scale 0.96 enter transform, and a
// finished transform resizes nothing, so the short reading is the floor the
// placement keeps. See cardHeight.
test("the card floors are read off the layout box, not the painted one", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/stores/monitor-frame-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const measure = source.slice(
    source.indexOf("const measure = () => {"),
    source.indexOf("const observer = new ResizeObserver"),
  );
  assert.match(
    source,
    /function cardHeight\([\s\S]*?\)\.offsetHeight;/,
    "a card's height is not the untransformed layout box",
  );
  assert.match(measure, /const grown = cards\.map\(cardHeight\)/);
  assert.match(measure, /squeezed: cardHeight\(child\)/);
  assert.match(measure, /tail \+= cardHeight\(child\) \+ STACK_GAP/);
});

// Dragged, the loaded models card pins itself `position: fixed` and leaves the
// rail's flow. scrollHeight drops it, so a floor that still counted it asked
// for 225px where 128 was needed and lifted the rail off a corner it fitted.
test("a card dragged out of the rail is not floored for", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/stores/monitor-frame-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const measure = source.slice(
    source.indexOf("const measure = () => {"),
    source.indexOf("const observer = new ResizeObserver"),
  );
  assert.match(
    source,
    /function inRailFlow\([\s\S]*?position !== "fixed" && position !== "absolute";/,
    "an out-of-flow card is not recognised",
  );
  assert.match(
    measure,
    /const cards = \[\.\.\.node\.children\]\.filter\(inRailFlow\)/,
  );
});

// The sweep above, re-run at the heights the stack actually takes.
// The stack scrolls now, and an uncapped box does not, so the measurement has
// to put scrollTop back with the cap or a resize below throws a reader of the
// download list back to the first card.
test("the measurement restores the stack's scroll position", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/stores/monitor-frame-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const measure = source.slice(
    source.indexOf("const measure = () => {"),
    source.indexOf("const observer = new ResizeObserver"),
  );
  assert.match(
    measure,
    /const scrolled = node\.scrollTop;[\s\S]*node\.style\.maxHeight = "none";/,
    "scrollTop is read after the cap comes off, when it has already clamped",
  );
  assert.match(
    measure,
    /node\.style\.maxHeight = capped;[\s\S]*node\.scrollTop = scrolled;/,
    "the scroll position is never put back",
  );
});

test("no measured height leaves the stack overlapping a box", () => {
  const composer = { left: 412, top: 664, right: 1148, bottom: 814 };
  for (const needed of [0, 40, 80, 120, 200, 320]) {
    for (let bottom = 60; bottom <= CHAT_H - 16; bottom += 4) {
      const box = {
        left: 996,
        top: Math.max(0, bottom - 180),
        right: 1264,
        bottom,
      };
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

// A window too short to hold the cards above the composer has no arrangement
// that both dodges it and shows them whole. Clipping them is what the overlap
// report was about, and a card sliced off at the rail's edge reads as one that
// has slid behind the page, so the stack takes the corner and paints over the
// composer instead. Only for a box that said it may be covered.
test("a stack that cannot fit above the composer covers it rather than clipping", () => {
  // 534px tall, the size from the report, with the welcome composer centred.
  const H = 534;
  const composer = {
    left: 236,
    top: 300,
    right: 684,
    bottom: 420,
    coverable: true,
  };
  const needed = 339;
  const covered = stackGeometry(composer, 921, H, needed);
  assert.equal(covered.bottom, 16, "the stack left the corner");
  assert.ok(
    covered.maxHeight >= needed,
    `the cards are still clipped: ${covered.maxHeight} < ${needed}`,
  );
  // The same box that has to be dodged when dodging it costs nothing.
  const roomy = stackGeometry(composer, 921, 1200, needed);
  assert.ok(
    1200 - roomy.bottom <= composer.top ||
      roomy.maxHeight <= 1200 - 16 - composer.bottom,
    "a composer that could have been dodged was covered anyway",
  );
});

test("a box that never said it may be covered is still never covered", () => {
  // The Live monitor. Its Close button and resize grip are why this store
  // exists, so it keeps the old answer even when the stack does not fit.
  const H = 534;
  const monitor = { left: 236, top: 300, right: 684, bottom: 420 };
  const geometry = stackGeometry(monitor, 921, H, 339);
  const stackTop = H - geometry.bottom - geometry.maxHeight;
  const clearsAbove = H - geometry.bottom <= monitor.top;
  const clearsBelow = stackTop >= monitor.bottom;
  assert.ok(
    clearsAbove || clearsBelow,
    "the stack was allowed over the monitor",
  );
});

// The fallback drops the coverable boxes and places against what is left, so
// the monitor has to come out of it exactly as it would have on its own. Stated
// as a comparison rather than an absolute: a window this short cannot always
// clear a monitor either, and the claim here is that the composer's permission
// is not inherited, not that the monitor is always dodged.
test("a coverable composer does not licence covering the monitor beside it", () => {
  const composer = {
    left: 236,
    top: 300,
    right: 684,
    bottom: 420,
    coverable: true,
  };
  const monitor = { left: 640, top: 60, right: 905, bottom: 250 };
  for (const height of [534, 700, 900, 1200]) {
    for (const needed of [56, 200, 339, 600]) {
      const alone = stackGeometry(monitor, 921, height, needed);
      const withComposer = stackGeometry(
        [composer, monitor],
        921,
        height,
        needed,
      );
      const clearance = (g: { bottom: number; maxHeight: number }) => ({
        edge: height - g.bottom,
        top: height - g.bottom - g.maxHeight,
      });
      const a = clearance(alone);
      const b = clearance(withComposer);
      const label = `height=${height} needed=${needed}`;
      assert.ok(
        b.top >= a.top || b.edge <= monitor.top,
        `the composer pushed the stack further over the monitor: ${label}`,
      );
    }
  }
});

// The cards may give up their notes, so a placement short of the height they
// would PREFER still shows all of them. Covering the composer to win those few
// pixels is the worse answer, and testing the fallback against the natural
// height rather than the floor is what did it: at 1280x830 a 3px shortfall took
// the stack off a dodge that fitted and put it over the composer.
test("a placement that fits the cards at their floor is not given up on", () => {
  const H = 830;
  const composer = {
    left: 416,
    top: 415,
    right: 864,
    bottom: 530,
    coverable: true,
  };
  const natural = 394;
  const floor = 339;
  const geometry = stackGeometry(composer, 1280, H, natural, floor);
  assert.ok(
    H - geometry.bottom <= composer.top,
    "the stack covered a composer it could have dodged",
  );
  assert.ok(geometry.maxHeight >= floor, "and it still holds the cards");
});

// The loaded models indicator is the LAST child of the rail, so it is the one
// that lands on the corner, and it is persistent: over Send it is #8210 again,
// permanently rather than until a dismiss. #8346 ships it off by default, which
// makes this whoever turned it on rather than nobody.
test("a persistent card that would land on the composer stops the cover", () => {
  const H = 534;
  // Docked under a thread: it sits on the bottom edge, so the corner is where
  // the persistent tail would go and the tail would land on Send.
  const docked = {
    left: 236,
    top: 380,
    right: 684,
    bottom: 518,
    coverable: true,
  };
  const dodging = stackGeometry(docked, 921, H, 460, 420, 60);
  assert.ok(
    H - dodging.bottom <= docked.top,
    "the persistent tail took the docked composer's corner",
  );
});

test("a persistent card clear of the composer does not stop the cover", () => {
  const H = 534;
  // The welcome composer, centred. The corner underneath it is free, so the
  // cards that reach it are the dismissible ones and there is nothing to
  // protect: covering is still the right answer.
  const welcome = {
    left: 236,
    top: 275,
    right: 684,
    bottom: 387,
    coverable: true,
  };
  const covering = stackGeometry(welcome, 921, H, 460, 420, 60);
  assert.equal(covering.bottom, 16, "it gave up a cover that was safe");
  assert.ok(covering.maxHeight >= 460, "and the cards are whole");
});

test("covering is refused when it still cannot show the cards", () => {
  // A short window with a monitor across the top: dropping the composer buys a
  // few pixels and still leaves the rail under its floor. The rail scrolls
  // either way, so paying the composer for those pixels buys nothing.
  const H = 400;
  const monitor = { left: 473, top: 0, right: 921, bottom: 200 };
  const composer = {
    left: 236,
    top: 250,
    right: 684,
    bottom: 384,
    coverable: true,
  };
  const both = stackGeometry([monitor, composer], 921, H, 460, 192, 0);
  const dodging = stackGeometry([monitor, composer], 921, H, 460, 10_000, 0);
  assert.deepEqual(
    both,
    dodging,
    "it covered the composer for a placement that is still under the floor",
  );
});

test("covering is taken when it does reach the floor", () => {
  // The same shape with room above the composer once it is dropped: now the
  // cover earns its price and the cards come out whole.
  const H = 534;
  const composer = {
    left: 236,
    top: 275,
    right: 684,
    bottom: 500,
    coverable: true,
  };
  const covering = stackGeometry(composer, 921, H, 460, 300, 0);
  // 16, the corner inset: the stack took the corner and the composer with it.
  assert.equal(covering.bottom, 16, "it refused a cover that fits");
  assert.ok(covering.maxHeight >= 300, "and the cards are whole");
});

test("the persistent tail is judged where the fallback actually lands", () => {
  // Dropping the composer does not always leave the stack at the corner: an
  // uncoverable monitor at the bottom still lifts it, and the lift carries the
  // tail up onto the composer that the corner-based reading had just cleared.
  const H = 800;
  const monitor = { left: 473, top: 600, right: 921, bottom: 800 };
  // Well above the corner, so a corner-based check calls covering safe.
  const composer = {
    left: 236,
    top: 380,
    right: 684,
    bottom: 560,
    coverable: true,
  };
  const placement = stackGeometry([monitor, composer], 921, H, 700, 240, 80);
  const railBottom = H - placement.bottom;
  assert.ok(
    railBottom - 80 >= composer.bottom || railBottom <= composer.top,
    `the persistent tail landed on the composer (rail bottom ${railBottom})`,
  );
});

test("no persistent tail means the old answer", () => {
  const composer = {
    left: 236,
    top: 300,
    right: 684,
    bottom: 420,
    coverable: true,
  };
  const withZero = stackGeometry(composer, 921, 534, 420, 420, 0);
  const withoutArg = stackGeometry(composer, 921, 534, 420, 420);
  assert.deepEqual(withoutArg, withZero);
});

// One number means the strict reading of it, so nothing that knows only the
// natural height silently starts covering things.
test("a caller that passes one height gets the stricter answer", () => {
  const composer = {
    left: 416,
    top: 415,
    right: 864,
    bottom: 530,
    coverable: true,
  };
  const one = stackGeometry(composer, 1280, 830, 394);
  const two = stackGeometry(composer, 1280, 830, 394, 394);
  assert.deepEqual(one, two);
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

// At its cap the stack's border box does not move when its content grows, so neither a
// root-only ResizeObserver nor a childList watcher hears an image finish loading inside an
// expanded release note, and the stack keeps the pre-load height and stays clipped.
test("every box inside the stack is observed, not just the stack", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/stores/monitor-frame-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const wiring = source.slice(
    source.indexOf("const observer = new ResizeObserver"),
  );
  assert.ok(
    !/observer\.observe\(node\)/.test(wiring),
    "the root alone is observed, so an intrinsic descendant resize is missed",
  );
  assert.match(
    wiring,
    /querySelectorAll\("\*"\)/,
    "the descendants are never enumerated, so none of them is observed",
  );
  // A changed subtree has to be re-observed, or a banner arriving after mount is missed.
  const onMutation = wiring.slice(wiring.indexOf("new MutationObserver"));
  assert.match(
    onMutation,
    /syncObserved\(\)/,
    "the observed set is not resynced",
  );
  // And unobserved on the way out, or a detached node keeps the observer alive.
  assert.match(wiring, /observer\.unobserve\(/, "nothing is ever unobserved");
});

// The llama.cpp update banner animates its progress bar's width every frame, inside this
// same stack. Observing descendants without filtering therefore remeasures at ~60Hz for a
// stack whose height never moved, and each remeasure lifts the cap and reads scrollHeight,
// which forces a synchronous layout.
test("a width-only animation does not remeasure the stack", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/stores/monitor-frame-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const wiring = source.slice(
    source.indexOf("const observer = new ResizeObserver"),
    source.indexOf("const observed = new Set<Element>()"),
  );
  assert.ok(
    !/new ResizeObserver\(measure\)/.test(wiring),
    "every observed resize remeasures, width-only ones included",
  );
  assert.match(wiring, /blockSize/, "the entry's height is never read");
  assert.match(
    wiring,
    /if \(moved\) measure\(\)/,
    "measure runs whether or not a height moved",
  );
});
