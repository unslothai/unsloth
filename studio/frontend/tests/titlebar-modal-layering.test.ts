// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const Z_INDEX_PATTERN = /z-\[(\d+)\]|\bz-(\d+)\b/;
const TITLEBAR_PATTERN =
  /<header\s+data-slot="window-titlebar"\s+className="([^"]+)"/;
const OVERLAY_SLOTS = [
  "dialog-overlay",
  "alert-dialog-overlay",
  "sheet-overlay",
];
const DIALOG_SURFACE_CLASSES =
  /<(?:DialogContent|AlertDialogContent|CommandDialog)\b(?:[^>]|=>)*?\bclassName="([^"]*)"/g;
const WHOLE_WINDOW_CENTRE = /(?:^|\s)(?:max-sm:)?top-1\/2(?:\s|$)/;
const MODAL_BAND_RULE =
  /:root:has\(([\s\S]*?)\)\s*\[data-slot="window-titlebar"\]\s*\{\s*background-color:/;

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

test("window controls stay above every modal backdrop", () => {
  const titlebar = readSrc("components/tauri/window-titlebar.tsx").match(
    TITLEBAR_PATTERN,
  );
  assert.ok(titlebar);
  const titlebarLayer = zIndex(titlebar[1]);
  for (const [file, slot] of [
    ["components/ui/dialog.tsx", "dialog-overlay"],
    ["components/ui/alert-dialog.tsx", "alert-dialog-overlay"],
    ["components/ui/sheet.tsx", "sheet-overlay"],
  ]) {
    const overlay = readSrc(file).match(
      new RegExp(`data-slot="${slot}"[\\s\\S]*?"([^"]*\\bz-50\\b[^"]*)"`),
    );
    assert.ok(overlay, slot);
    assert.ok(zIndex(overlay[1]) < titlebarLayer, slot);
  }
});

// Page headers share the band, so a backdrop dims it; without the painted band the
// controls above the backdrop sit on grey and read as disabled. A clear backdrop (Ctrl+K)
// dims nothing, so painting there would only hide the header.
test("the titlebar band is painted while a dimming modal backdrop is up", () => {
  const rule = readSrc("index.css").match(MODAL_BAND_RULE);
  assert.ok(rule);
  for (const slot of OVERLAY_SLOTS) {
    assert.ok(
      rule[1].includes(`[data-slot="${slot}"]:not(.bg-transparent)`),
      slot,
    );
  }
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
