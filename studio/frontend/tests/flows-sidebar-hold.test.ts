// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opening a flow leaves the app sidebar as the user had it; on Mac the toolbar takes the titlebar row.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("the Flows editor never pins or unpins the app sidebar", () => {
  const page = read("features/flows/editor/flows-editor-page.tsx");
  assert.doesNotMatch(page, /holdSidebar|releaseSidebarPinned|setPinned|togglePinned/);
  assert.doesNotMatch(read("hooks/use-sidebar-pin.ts"), /holdSidebarUnpinned/);
});

test("on Mac the toolbar is the titlebar row, on the canvas colour, clear of the window buttons", () => {
  const page = read("features/flows/editor/flows-editor-page.tsx");
  assert.match(page, /const \[macChrome\] = useState\(shouldUseNativeMacWindowTitlebar\);/);
  assert.match(page, /\$\{macChrome \? "bg-background" : `bg-sidebar \$\{BELOW_TITLEBAR\}`\}/, "no titlebar strip on Mac");
  assert.match(page, /clearNav=\{macChrome && !\(sidebar\?\.pinned \?\? true\)\}/);
  assert.match(page, /sidebarBeside && !macChrome \? "rounded-tl-xl"/);
  const bar = read("features/flows/editor/toolbar.tsx");
  assert.match(bar, /data-tauri-drag-region=\{titlebar \|\| undefined\}/);
  assert.match(bar, /\$\{titlebar \? "bg-background" : "bg-sidebar"\}/);
  assert.match(bar, /clearNav \? "pl-\[var\(--studio-collapsed-chat-controls-inset,0\.5rem\)\]"/);
});
