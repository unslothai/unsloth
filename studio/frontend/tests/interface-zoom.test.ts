// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import {
  ZOOM_CHORDS,
  zoomDirectionForKey,
} from "../src/features/interface-zoom/lib/zoom-chords.ts";
import {
  WHEEL_ZOOM_IDLE_MS,
  WHEEL_ZOOM_STEP_PX,
  createWheelZoomAccumulator,
} from "../src/features/interface-zoom/lib/zoom-wheel.ts";
import { readSrc } from "./helpers/kit.ts";

const press = (
  key: string,
  code: string,
  mods: { meta?: boolean; ctrl?: boolean; alt?: boolean } = {},
) => ({
  key,
  code,
  metaKey: Boolean(mods.meta),
  ctrlKey: Boolean(mods.ctrl),
  altKey: Boolean(mods.alt),
});

test("Cmd on macOS and Ctrl elsewhere zoom in, out and back", () => {
  for (const [mac, mod] of [
    [true, { meta: true }],
    [false, { ctrl: true }],
  ] as const) {
    assert.equal(zoomDirectionForKey(press("=", "Equal", mod), mac), 1);
    assert.equal(zoomDirectionForKey(press("+", "Equal", mod), mac), 1);
    assert.equal(zoomDirectionForKey(press("-", "Minus", mod), mac), -1);
    assert.equal(zoomDirectionForKey(press("_", "Minus", mod), mac), -1);
    assert.equal(zoomDirectionForKey(press("0", "Digit0", mod), mac), 0);
  }
});

test("the keypad zooms too", () => {
  const ctrl = { ctrl: true };
  assert.equal(zoomDirectionForKey(press("+", "NumpadAdd", ctrl), false), 1);
  assert.equal(
    zoomDirectionForKey(press("-", "NumpadSubtract", ctrl), false),
    -1,
  );
  assert.equal(zoomDirectionForKey(press("0", "Numpad0", ctrl), false), 0);
});

test("a layout's own + key zooms, and the physical keys still do when they type something else", () => {
  // German: + has its own key (BracketRight); AZERTY: Digit0 types "à".
  assert.equal(
    zoomDirectionForKey(press("+", "BracketRight", { ctrl: true }), false),
    1,
  );
  assert.equal(
    zoomDirectionForKey(press("à", "Digit0", { ctrl: true }), false),
    0,
  );
  assert.equal(
    zoomDirectionForKey(press("ß", "Minus", { ctrl: true }), false),
    -1,
  );
});

test("the wrong modifier, Alt, or no modifier leaves the key alone", () => {
  assert.equal(
    zoomDirectionForKey(press("=", "Equal", { ctrl: true }), true),
    null,
  );
  assert.equal(
    zoomDirectionForKey(press("=", "Equal", { meta: true }), false),
    null,
  );
  assert.equal(
    zoomDirectionForKey(press("=", "Equal", { meta: true, alt: true }), true),
    null,
  );
  assert.equal(
    zoomDirectionForKey(press("=", "Equal", { meta: true, ctrl: true }), true),
    null,
  );
  assert.equal(zoomDirectionForKey(press("-", "Minus"), false), null);
  assert.equal(
    zoomDirectionForKey(press("a", "KeyA", { ctrl: true }), false),
    null,
  );
});

test("the chords match the ones the View menu shows", () => {
  const chords = readSrc("app/app-menu-chords.ts");
  assert.ok(chords.includes(`"zoom-in": { chord: "${ZOOM_CHORDS[1]}" }`));
  assert.ok(chords.includes(`"zoom-out": { chord: "${ZOOM_CHORDS[-1]}" }`));
  assert.ok(chords.includes(`"actual-size": { chord: "${ZOOM_CHORDS[0]}" }`));
});

test("the popup is mounted on every route and only takes chords in the desktop app", () => {
  const root = readSrc("app/routes/__root.tsx");
  const zoom = readSrc("features/interface-zoom/components/interface-zoom.tsx");
  assert.match(root, /<InterfaceZoom \/>/);
  assert.match(zoom, /if \(!isTauri\) return;/);
  assert.match(
    zoom,
    /shortcutOwningBinding\(overrides, ZOOM_CHORDS\[direction\]\)/,
  );
});

test("one Ctrl+wheel notch is one step, up zooms in, and a pause starts over", () => {
  const step = createWheelZoomAccumulator();
  const wheel = (
    deltaY: number,
    timeStamp: number,
    mods: { ctrl?: boolean } = { ctrl: true },
  ) => ({
    deltaY,
    deltaMode: 0,
    ctrlKey: Boolean(mods.ctrl),
    metaKey: false,
    altKey: false,
    timeStamp,
  });
  // A Windows notch is 100px: one step, not two.
  assert.equal(step(wheel(-100, 0), 1), 1);
  assert.equal(step(wheel(100, 400), 1), -1);
  // Without Ctrl, a wheel scrolls.
  assert.equal(step(wheel(-100, 800, {}), 1), null);
  // A touchpad pinch arrives in small pieces and steps once it has travelled far enough.
  assert.equal(step(wheel(-20, 1200), 1), null);
  assert.equal(step(wheel(-20, 1210), 1), null);
  assert.equal(step(wheel(-20, 1220), 1), 1);
  // Leftover travel does not carry over a pause.
  assert.equal(step(wheel(-40, 1300), 1), null);
  assert.equal(step(wheel(-40, 1300 + WHEEL_ZOOM_IDLE_MS + 1), 1), null);
  // Deltas come in CSS pixels, which shrink as the zoom grows: at 200% a notch reads 50.
  assert.equal(step(wheel(-50, 5000), 2), 1);
  assert.equal(step(wheel(-25, 6000), 1), null);
  assert.ok(WHEEL_ZOOM_STEP_PX <= 50);
});

test("the popup scales with the page and does not dismiss a modal", () => {
  const zoom = readSrc("features/interface-zoom/components/interface-zoom.tsx");
  // Scales with the page, as the find bar does: nothing divides the page zoom back out.
  assert.doesNotMatch(zoom, /zoom: 1 \//);
  assert.match(
    zoom,
    /onPointerDown=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
  assert.match(zoom, /pointer-events-auto/);
  // The live region outlives the popup, so the first zoom is announced too.
  assert.match(
    zoom,
    /\{isTauri && <ZoomAnnouncer open=\{open\} \/>\}\s*\{open && <ZoomPopup \/>\}/,
  );
  // Canvases that zoom on Ctrl+wheel themselves get the event first; macOS keeps its own.
  assert.match(
    zoom,
    /if \(event\.defaultPrevented \|\| !event\.ctrlKey\) return;/,
  );
  assert.match(
    zoom,
    /if \(!mac\) window\.addEventListener\("wheel", onWheel, \{ passive: false \}\)/,
  );
});

test("the zoom popup has two places: the find bar's corner, or under the bar while it shows", () => {
  const css = readSrc("index.css");
  const provider = readSrc("app/provider.tsx");
  const findBar = readSrc("features/find-in-page/components/find-bar.tsx");
  const findInPage = readSrc(
    "features/find-in-page/components/find-in-page.tsx",
  );
  // The wrapper's inset, mirrored onto <html> where both body-portaled bars read it.
  assert.match(provider, /"--studio-content-top-inset": "34px"/);
  assert.match(
    provider,
    /"--studio-portal-content-top-inset",\s*usesCustomTitlebar \? "34px" : null/,
  );
  const top =
    /top-\[calc\(var\(--studio-portal-content-top-inset,0px\)\+3\.5rem\)\] right-4 z-50 flex h-13/;
  assert.match(findBar, top);
  assert.match(findInPage, top);
  assert.match(
    css,
    /\.interface-zoom-position \{\s*top: calc\(var\(--studio-portal-content-top-inset, 0px\) \+ 3\.5rem\);/,
  );
  // Under the bar only while it is showing: a bar hidden behind a modal does not count.
  assert.match(
    css,
    /body:has\(> \[data-find-bar-layer\]:not\(\[hidden\]\) \[role="search"\]\) \.interface-zoom-position \{\s*top: calc\(var\(--studio-portal-content-top-inset, 0px\) \+ 3\.5rem \+ var\(--spacing, 0\.25rem\) \* 13 \+ 0\.5rem\);/,
  );
  // Those are the only rules that place it.
  assert.equal(css.match(/\.interface-zoom-position/g)?.length, 2);
  // A little smaller than the bar.
  assert.match(
    readSrc("features/interface-zoom/components/interface-zoom.tsx"),
    /find-bar-surface flex h-11 /,
  );
});

test("both bars paint over everything, on one named layer at the top of the scale", () => {
  const layers = readSrc("lib/z-layers.ts");
  const layer = (name: string) =>
    Number(new RegExp(`^  ${name}: (\\d+),$`, "m").exec(layers)?.[1]);
  const bars = layer("WINDOW_BARS");
  for (const [, name, value] of /^ {2}([A-Z_]+): (\d+),$/gm[Symbol.matchAll](
    layers,
  )) {
    if (name !== "WINDOW_BARS")
      assert.ok(bars > Number(value), `${name} is not under the bars`);
  }
  // Every literal number in the app (classes, the stylesheet, inline styles) and the toaster's,
  // bar the reload snapshot.
  const sonner = readSrc("../node_modules/sonner/dist/styles.css");
  const sources = readdirSync(new URL("../src/", import.meta.url), {
    recursive: true,
  })
    .map(String)
    .filter((file) => /\.(tsx?|css)$/.test(file))
    .map((file) => readSrc(file));
  const numbers = [
    ...sources.flatMap((text) => [
      ...text.matchAll(/z-index:\s*(\d+)/g),
      ...text.matchAll(/\bz-\[(\d+)\]/g),
      ...text.matchAll(/zIndex(?:\s*=|:)\s*"?(\d+)/g),
    ]),
    ...sonner.matchAll(/z-index:\s*(\d+)/g),
  ]
    .map((hit) => Number(hit[1]))
    .filter((value) => value !== 2147483647);
  assert.ok(numbers.includes(999999999), "the toaster's layer was not read");
  for (const value of numbers)
    assert.ok(bars > value, `z-index ${value} is over the bars`);
  assert.ok(bars < 2147483647);
  // Both are portaled to <body> onto that layer, out of every stacking context in the shell.
  const findInPage = readSrc(
    "features/find-in-page/components/find-in-page.tsx",
  );
  const zoom = readSrc("features/interface-zoom/components/interface-zoom.tsx");
  assert.match(
    findInPage,
    /createPortal\(\s*<div\s+data-find-bar-layer=""\s+hidden=\{!foreground\}\s+className="fixed top-0 right-0"\s+style=\{\{ zIndex: Z_LAYER\.WINDOW_BARS \}\}/,
  );
  assert.match(findInPage, /document\.body,\s*\);\s*\}\s*$/);
  assert.match(
    zoom,
    /interface-zoom-position pointer-events-auto fixed right-4"\s+style=\{\{ zIndex: Z_LAYER\.WINDOW_BARS \}\}/,
  );
  assert.match(zoom, /createPortal\([\s\S]*document\.body,\s*\);/);
});
