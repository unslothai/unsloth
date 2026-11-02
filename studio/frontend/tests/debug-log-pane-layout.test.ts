// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const tab = readSrc("features/settings/tabs/debugging-tab.tsx");
const pane = tab.slice(
  tab.indexOf("ref={frameRef}"),
  tab.indexOf('data-testid="debug-log-resize"'),
);

test("the wrap toggle sits on the log pane, not in the toolbar", () => {
  assert.ok(pane.includes('data-testid="debug-log-wrap"'));
  assert.ok(pane.includes('t("settings.debugging.wrapLines")'));
  assert.ok(
    !tab
      .slice(0, tab.indexOf("ref={frameRef}"))
      .includes("settings.debugging.wrapLines"),
  );
});

test("the frame draws the border so the pane's scrollbar stays off its corners", () => {
  assert.match(
    pane,
    /"flex rounded-lg border border-border\/60 bg-muted\/20 py-1 pr-0 pl-1/,
  );
  assert.match(
    pane,
    /"h-full w-full overflow-auto \[overflow-anchor:none\] scroll-rounded rounded-md/,
  );
});

test("jump to latest is a plain down arrow with even padding", () => {
  assert.match(pane, /rounded-full pr-2\.5 pl-2 shadow-md"\s*>\s*<ArrowDownIcon/);
  assert.doesNotMatch(tab, /ChevronDownDouble/);
});

test("the pane resizes by drag and keyboard and remembers its height", () => {
  for (const needle of [
    'role="separator"',
    "onPointerDown={onResizeStart}",
    "onKeyDown={onResizeKey}",
    "onDoubleClick={() => resizePane(null)}",
    'const PANE_HEIGHT_STORAGE_KEY = "unsloth_debug_log_pane_height";',
    "}, [text, wrap, paneHeight]);",
  ]) {
    assert.ok(tab.includes(needle), needle);
  }
});
