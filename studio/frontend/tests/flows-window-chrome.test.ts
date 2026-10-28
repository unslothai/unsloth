// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Flows' editor keeps the window titlebar and its corners clear on macOS and Windows.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("the editor starts below the titlebar with the inset Train and Recipe Studio get", () => {
  const page = read("features/flows/editor/flows-editor-page.tsx");
  assert.match(
    page,
    /pt-\[var\(--studio-non-chat-content-top-inset,var\(--studio-content-top-inset,0px\)\)\]/,
  );
  assert.match(page, /flows-editor flex[^`]*\$\{BELOW_TITLEBAR\}/);
  // The shell treats the editor as chat-like, so it adds no inset of its own.
  assert.match(read("app/routes/__root.tsx"), /isFlowEditorRoute/);
});

test("the toolbar holds no run or share actions; they float at the bottom of the canvas", () => {
  const toolbar = read("features/flows/editor/toolbar.tsx");
  for (const id of ["flows-run", "flows-stop", "flows-play", "flows-open-app"])
    assert.ok(!toolbar.includes(id), `${id} is back in the toolbar`);
  assert.ok(!toolbar.includes("justify-self-end"));
  const bar = read("features/flows/editor/run-bar.tsx");
  for (const id of ["flows-run", "flows-stop", "flows-play", "flows-open-app", "flows-plan", "flows-share"])
    assert.ok(bar.includes(`"${id}"`), `${id} missing from the run bar`);
  assert.match(bar, /absolute inset-x-0 bottom-3 z-20 flex justify-center/);
  assert.match(bar, /title="Run \(Ctrl\+Enter\)"/);
  assert.match(read("features/flows/editor/flows-editor-page.tsx"), /<RunBar\b/);
});
