// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  bakeScale,
  dragPosition,
  imageBox,
  zoomPosition,
} from "../src/features/chat-background/artwork-geometry.ts";
import { readSrc } from "./helpers/kit.ts";

const ARTWORK = readSrc("features/chat-background/artwork.tsx");
const BACKGROUND = readSrc("features/chat-background/background.tsx");

test("the baked box is where CSS paints a no-repeat background for every sizing", () => {
  assert.deepEqual(imageBox(1000, 1000, 400, 200, "cover", "50% 50%"), {
    x: -500,
    y: 0,
    w: 2000,
    h: 1000,
  });
  assert.deepEqual(imageBox(1000, 1000, 400, 200, "contain", "50% 50%"), {
    x: 0,
    y: 250,
    w: 1000,
    h: 500,
  });
  assert.deepEqual(imageBox(1000, 1000, 400, 200, "width", "0% 100%"), {
    x: 0,
    y: 500,
    w: 1000,
    h: 500,
  });
  assert.deepEqual(imageBox(1000, 1000, 400, 200, "height", "100% 0%"), {
    x: -1000,
    y: 0,
    w: 2000,
    h: 1000,
  });
  assert.deepEqual(imageBox(1000, 1000, 400, 200, "stretch", "30% 70%"), {
    x: 0,
    y: 0,
    w: 1000,
    h: 1000,
  });
  assert.deepEqual(imageBox(1000, 1000, 400, 200, "original", "50% 50%"), {
    x: 300,
    y: 400,
    w: 400,
    h: 200,
  });
  assert.deepEqual(
    imageBox(1000, 1000, 400, 200, "original", "bad"),
    { x: 300, y: 400, w: 400, h: 200 },
    "an unparsable position centres, as the model does",
  );
});

test("a wide blur bakes at a lower resolution, a narrow one at the device pixel ratio", () => {
  assert.equal(bakeScale(20, 2), 0.4);
  assert.equal(bakeScale(2, 2), 2);
  assert.equal(bakeScale(4, 1), 1);
  assert.equal(bakeScale(100, 3), 0.25, "never below a quarter of a CSS pixel");
});

test("the live blur filter shows only until the bitmap for the current inputs is baked", () => {
  assert.match(ARTWORK, /const ready = baked !== null && baked\.key === key;/);
  assert.match(
    ARTWORK,
    /backgroundSize: "100% 100%",[\s\S]{0,120}transform: `scale\(\$\{zoom \/ 100\}\)`/,
    "the baked bitmap covers the box and keeps the zoom transform, which CSS applies after the filter",
  );
  assert.match(
    ARTWORK,
    /filter: blur \? `blur\(\$\{blur\}px\)` : undefined,/,
    "the pending path is the old live filter",
  );
  assert.match(
    ARTWORK,
    /`\$\{url\}\|\$\{sizing\}\|\$\{position\}\|\$\{blur\}\|\$\{size\.w\}x\$\{size\.h\}\|/,
    "a new image, sizing, position, blur or box size bakes again",
  );
  assert.match(
    ARTWORK,
    /octx\.filter = `blur\(\$\{blur \* k\}px\)`;\s*octx\.drawImage\(plain, 0, 0\);/,
    "the box is drawn clipped first, then blurred, as filter: blur() blurs the painted box",
  );
  assert.match(
    ARTWORK,
    /"filter" in CanvasRenderingContext2D\.prototype/,
    "older WebKit without a canvas filter gets the downscale path",
  );
  assert.match(
    ARTWORK,
    /URL\.revokeObjectURL\(bakedUrl\.current\)/,
    "baked URLs are released",
  );
});

test("only the current frame bakes; a departing frame fades out with the live filter", () => {
  assert.match(
    BACKGROUND,
    /<WallpaperArtwork\s+url=\{frame\.url\}[\s\S]{0,200}bakeable=\{i === 1\}/,
  );
  assert.equal(BACKGROUND.includes("filter: adjustment.blur"), false);
});

test("adjust: a drag moves the image with the pointer, a zoom keeps the point under it", () => {
  const edge = (p: number, box: number, drawn: number, zoom: number) =>
    p * (box - drawn * zoom);
  const box = 400;
  const drawn = imageBox(400, 200, 800, 200, "cover", "50% 50%").w; // 800: wider than the box
  const z = 1.5;
  const p = dragPosition(0.5, 30, box, drawn, z);
  assert.ok(
    Math.abs(edge(p, box, drawn, z) - (edge(0.5, box, drawn, z) + 30)) < 1e-9,
  );
  assert.equal(dragPosition(0, 5000, box, drawn, z), 0);
  assert.equal(dragPosition(0.3, 50, 400, 400, 1), 0.3);
  const at = 120;
  const q = zoomPosition(0.5, at, box, drawn, 1, 2);
  const along = (at - edge(0.5, box, drawn, 1)) / (drawn * 1);
  const after = edge(q, box, drawn, 2) + along * drawn * 2;
  assert.ok(Math.abs(after - at) < 1e-9);
});

test("adjust: framing lives in a dialog, not the sizing menu or a zoom slider", () => {
  const controls = readSrc("features/chat-background/controls.tsx");
  assert.match(controls, /<WallpaperAdjustDialog/);
  assert.match(controls, /t\("chatBackground\.adjust"\)/);
  assert.doesNotMatch(controls, /PositionField|POSITIONS\.map/);
  assert.doesNotMatch(controls, /key: "zoom", label: "chatBackground\.zoom"/);
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(
    dialog,
    /position: `\$\{Math\.round\(draft\.x \* 100\)\}% \$\{Math\.round\(draft\.y \* 100\)\}%`/,
  );
  assert.match(
    dialog,
    /addEventListener\("wheel", onWheel, \{ passive: false \}\)/,
  );
  assert.match(dialog, /addEventListener\("gesturechange", onGestureChange\)/);
});

test("adjust: one-word buttons and no text selection", () => {
  const controls = readSrc("features/chat-background/controls.tsx");
  assert.doesNotMatch(controls, /CropIcon|Cancel01Icon/);
  assert.match(
    controls,
    /\{!value && \(\s*<HugeiconsIcon\s+icon=\{Upload01Icon\}/,
  );
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(
    dialog,
    /event\.preventDefault\(\);\s+window\.getSelection\(\)\?\.removeAllRanges\(\);/,
  );
  assert.match(
    dialog,
    /onMouseDown=\{\(event\) => event\.preventDefault\(\)\}/,
  );
  assert.match(dialog, /touch-none select-none/);
});

test("adjust previews the whole wallpaper area scaled down, so any sizing and window shape match", () => {
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.doesNotMatch(dialog, /Math\.max\(1, w \/ Math\.max\(1, h\)\)/);
  assert.match(dialog, /const aspect = area\.w \/ area\.h;/);
  assert.match(
    dialog,
    /width: area\.w,\s*height: area\.h,\s*transform: `scale\(\$\{scale\}\)`/,
  );
  assert.match(dialog, /dragPosition\(from\.x, dx \/ scale, area\.w, drawn\.w/);
  assert.match(dialog, /\{ x: at\.x \/ scale, y: at\.y \/ scale \}/);
});

test("adjust: arrow keys move the image the way they point, like a drag", () => {
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(
    dialog,
    /\["ArrowLeft", \[-step, 0\]\],\s*\["ArrowRight", \[step, 0\]\],\s*\["ArrowUp", \[0, -step\]\],\s*\["ArrowDown", \[0, step\]\],/,
  );
});

test("adjust: the page area follows a resize or rotation while the dialog is open", () => {
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(dialog, /window\.addEventListener\("resize", read\);/);
  assert.match(dialog, /const next = pageArea\(fade\);/);
});

test("adjust: a two-finger touch pinch zooms where WebKit gesture events do not exist", () => {
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(
    dialog,
    /if \(pointers\.current\.size === 2\) \{[\s\S]*?touchPinch\.current = \{\s*dist: Math\.hypot\(a\.x - b\.x, a\.y - b\.y\) \|\| 1,\s*zoom: draftRef\.current\.zoom,/,
  );
  assert.match(
    dialog,
    /\(spread\.zoom \* Math\.hypot\(a\.x - b\.x, a\.y - b\.y\)\) \/ spread\.dist/,
  );
  assert.match(dialog, /onPointerUp=\{release\}\s*onPointerCancel=\{release\}/);
});

test("adjust: the preview URL survives StrictMode's double setup", () => {
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(
    dialog,
    /const next = URL\.createObjectURL\(value\.blob\);\s*setUrl\(next\);\s*return \(\) => URL\.revokeObjectURL\(next\);/,
  );
  assert.doesNotMatch(dialog, /useMemo\(\(\) => URL\.createObjectURL/);
});

test("adjust: a zoom made mid-drag is kept, as moves apply to the current draft", () => {
  const dialog = readSrc("features/chat-background/adjust-dialog.tsx");
  assert.match(
    dialog,
    /start\.x = event\.clientX;\s*start\.y = event\.clientY;\s*setDraft\(\(d\) => moveBy\(d, dx, dy\)\);/,
  );
  assert.doesNotMatch(dialog, /start\.from/);
});
