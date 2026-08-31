// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const VIEWPORT_CAP = /max-h-\[min\(\d+vh/;
const ROW_BOUNDS_IMAGE =
  /className="flex min-h-0 max-h-\[\d+px\] flex-1 flex-col[^"]*"[\s\S]{0,200}?className="[^"]*relative min-h-0"[\s\S]{0,300}?<img[\s\S]{0,300}?max-h-full/;
const PANEL_CLASSES = /className="[^"]*max-w-\[1100px\][^"]*"/;
const FRAME_OPTS_IN = /className="pointer-events-auto relative min-h-0"/;
const POINTER_EVENTS_AUTO = /pointer-events-auto/;

async function overlaySource(): Promise<string> {
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const start = thread.indexOf("const GeneratedImageViewportOverlay");
  const end = thread.indexOf("\n};", start);
  return start === -1 || end === -1 ? "" : thread.slice(start, end);
}

test("the preview is bounded by the row it sits in, not by the viewport", async () => {
  const overlay = await overlaySource();
  assert.notEqual(overlay, "", "GeneratedImageViewportOverlay not found");

  // A vh/px cap ignores the section's bottom offset and the caption, so under ~900px
  // tall the image covers it; max-h-full only resolves under a definite parent height.
  assert.doesNotMatch(overlay, VIEWPORT_CAP);
  assert.match(overlay, ROW_BOUNDS_IMAGE);
});

test("only the image frame swallows clicks, so the backdrop stays reachable", async () => {
  const overlay = await overlaySource();
  const panel = overlay.match(PANEL_CLASSES)?.[0];
  assert.ok(panel, "overlay panel class list not found");

  // The panel is transparent, so pointer-events-auto there kills backdrop dismissal.
  assert.doesNotMatch(panel, POINTER_EVENTS_AUTO);
  assert.match(overlay, FRAME_OPTS_IN);
});
