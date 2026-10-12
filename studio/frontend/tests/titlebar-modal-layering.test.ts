// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const Z_INDEX_PATTERN = /z-\[(\d+)\]|\bz-(\d+)\b/;
const DECORATION_PATTERN =
  /data-slot="window-titlebar-decoration"[\s\S]*?className="([^"]+)"/;
const TITLEBAR_PATTERN = /<header\s+className=\{cn\(\s*"([^"]+)"/;
const DIALOG_OVERLAY_PATTERN =
  /data-slot="dialog-overlay"[\s\S]*?"([^"]*\bz-50\b[^"]*)"/;
const ALERT_DIALOG_OVERLAY_PATTERN =
  /data-slot="alert-dialog-overlay"[\s\S]*?"([^"]*\bz-50\b[^"]*)"/;
const TOP_FULL_PATTERN = /top-full/;
const CLOSED_DECORATION_PATTERN = /<\/div>\s*\)\}\s*$/;
const DIALOG_SURFACE_CLASSES =
  /<(?:DialogContent|AlertDialogContent|CommandDialog)\b(?:[^>]|=>)*?\bclassName="([^"]*)"/g;
const WHOLE_WINDOW_CENTRE = /(?:^|\s)(?:max-sm:)?top-1\/2(?:\s|$)/;
const DOCUMENT_ROOT_HAS_PATTERN =
  /(?:^|[\s,{}(>~+])(?:html|body|:root)(?:\[[^\]]*\]|\.[\w-]+|#[\w-]+)*:has\(/m;
const TITLEBAR_BACKDROP_RULE_PATTERN =
  /\[data-slot="window-titlebar"\]\[data-viewport-backdrop-open\]::after\s*\{\s*opacity:\s*1;\s*\}/;
const TITLEBAR_BACKDROP_STORE_PATTERN =
  /useSyncExternalStore\(\s*enabled \? subscribeViewportBackdrop : subscribeNever,/;
const TITLEBAR_BACKDROP_ATTRIBUTE_PATTERN =
  /data-slot="window-titlebar"\s+data-viewport-backdrop-open=\{backdropOpen \? "" : undefined\}/;
const TOUR_VIEWPORT_BACKDROP_PATTERN =
  /data-slot="dialog-overlay"[\s\S]*?data-viewport-backdrop=\{true\}/;

/** Every component under src, so a new dialog cannot slip past. */
const COMPONENTS = readdirSync(join(import.meta.dirname, "../src"), {
  recursive: true,
  encoding: "utf8",
}).filter((path) => path.endsWith(".tsx"));

function zIndex(block: string): number {
  const match = block.match(Z_INDEX_PATTERN);
  if (!match) throw new Error(`no z-index class in ${block}`);
  return Number(match[1] ?? match[2]);
}

test("titlebar decoration stays below modal backdrops and window controls", async () => {
  const [titlebar, dialog, alertDialog] = await Promise.all([
    readSrc("components/tauri/window-titlebar.tsx"),
    readSrc("components/ui/dialog.tsx"),
    readSrc("components/ui/alert-dialog.tsx"),
  ]);

  const decoration = titlebar.match(DECORATION_PATTERN);
  const titlebarHeader = titlebar.match(TITLEBAR_PATTERN);
  const dialogOverlay = dialog.match(DIALOG_OVERLAY_PATTERN);
  const alertDialogOverlay = alertDialog.match(ALERT_DIALOG_OVERLAY_PATTERN);

  assert.ok(decoration);
  assert.ok(titlebarHeader);
  assert.ok(dialogOverlay);
  assert.ok(alertDialogOverlay);

  const decorationLayer = zIndex(decoration[1]);
  const titlebarLayer = zIndex(titlebarHeader[1]);
  for (const overlay of [dialogOverlay[1], alertDialogOverlay[1]]) {
    const overlayLayer = zIndex(overlay);
    assert.ok(decorationLayer < overlayLayer);
    assert.ok(overlayLayer < titlebarLayer);
  }
});

test("below-titlebar decoration is not trapped in the titlebar stacking context", async () => {
  const titlebar = await readSrc("components/tauri/window-titlebar.tsx");
  const decorationIndex = titlebar.indexOf(
    'data-slot="window-titlebar-decoration"',
  );
  const headerIndex = titlebar.indexOf("<header", decorationIndex);

  assert.notEqual(decorationIndex, -1);
  assert.notEqual(headerIndex, -1);
  assert.ok(decorationIndex < headerIndex);
  assert.match(
    titlebar.slice(decorationIndex, headerIndex),
    CLOSED_DECORATION_PATTERN,
  );

  const headerEnd = titlebar.indexOf("</header>", headerIndex);
  assert.notEqual(headerEnd, -1);
  const header = titlebar.slice(headerIndex, headerEnd);
  assert.doesNotMatch(header, TOP_FULL_PATTERN);
});

test("the pinned sidebar edge is drawn with borders, not a box over the chat", () => {
  const titlebar = readSrc("components/tauri/window-titlebar.tsx");
  const start = titlebar.indexOf('data-slot="window-titlebar-decoration"');
  const decoration = titlebar.slice(start, titlebar.indexOf("<header", start));
  assert.doesNotMatch(
    decoration,
    /right-0[^"]*h-\[calc\(100dvh|h-\[calc\(100dvh[^"]*right-0/,
    "a full-height box from the sidebar edge to the window edge covered the chat, so the compositor could not hit-test a wheel and every desktop scroll waited on the main thread",
  );
  assert.match(decoration, /pinned && "rounded-tl-\[12px\] border-l"/);
  assert.match(
    decoration,
    /className="absolute top-\[12px\] h-\[calc\(100dvh-var\(--studio-custom-titlebar-height\)-12px\)\] w-0 border-l border-sidebar-edge dark:border-transparent"\s*style=\{\{ left: cornerLeft \}\}/,
    "the edge below the corner is a border on a zero-width box, so it snaps to device pixels like the corner",
  );
});

// A top-* class at a call site makes twMerge drop the base's chrome-aware centre, so a plain
// top-1/2 centres on the whole window and the titlebar covers the dialog's top.
test("dialogs that set their own top still centre below the window chrome", () => {
  let checked = 0;
  for (const file of COMPONENTS) {
    for (const [, classes] of readSrc(file).matchAll(DIALOG_SURFACE_CLASSES)) {
      checked += 1;
      assert.doesNotMatch(classes, WHOLE_WINDOW_CENTRE, file);
    }
  }
  assert.ok(checked > 10, `only ${checked} dialog surfaces matched`);
});

test("the titlebar dims for viewport backdrops without a :has() on the document root", async () => {
  const [styles, titlebar, tour] = await Promise.all([
    readSrc("index.css"),
    readSrc("components/tauri/window-titlebar.tsx"),
    readSrc("features/tour/components/guided-tour.tsx"),
  ]);

  assert.doesNotMatch(styles.replace(/\/\*[\s\S]*?\*\//g, ""), DOCUMENT_ROOT_HAS_PATTERN);
  assert.match(styles, TITLEBAR_BACKDROP_RULE_PATTERN);
  assert.match(titlebar, TITLEBAR_BACKDROP_STORE_PATTERN);
  assert.match(titlebar, TITLEBAR_BACKDROP_ATTRIBUTE_PATTERN);
  assert.match(tour, TOUR_VIEWPORT_BACKDROP_PATTERN);
});
