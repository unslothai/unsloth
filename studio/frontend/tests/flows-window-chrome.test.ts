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
  assert.match(bar, /absolute inset-x-0 z-20 flex justify-center/);
  assert.match(bar, /title="Run \(Ctrl\+Enter\)"/);
  assert.match(read("features/flows/editor/flows-editor-page.tsx"), /<RunBar\b/);
});

test("the run bar sits on the editor body, so opening the Nodes panel or the inspector never moves it", () => {
  const page = read("features/flows/editor/flows-editor-page.tsx");
  const body = page.slice(page.indexOf('<div className="relative flex min-h-0 flex-1 overflow-hidden rounded-tl-xl bg-background">'));
  const canvasColumn = body.slice(body.indexOf('<div className="relative min-h-0 flex-1">'), body.indexOf("<RunDrawer />"));
  assert.ok(!canvasColumn.includes("<RunBar"), "the bar is not inside the canvas column");
  assert.ok(body.indexOf("<RunBar") > body.indexOf("<Inspector />"), "the bar is a child of the body row");
  const bar = read("features/flows/editor/run-bar.tsx");
  assert.ok(!/paletteOpen|inspector/i.test(bar.replace(/\/\*[\s\S]*?\*\//g, "")), "its position ignores panel state");
  assert.match(bar, /bottom: drawerOpen \? DRAWER_CLEARANCE : "0\.75rem"/);
});

test("the game player starts below the titlebar too, so the window buttons never cover its HUD", () => {
  const page = read("features/flows/player/flow-player-page.tsx");
  const outer = page.slice(page.indexOf("return ("), page.indexOf('data-testid="flow-player-page"'));
  assert.match(outer, /pt-\[var\(--studio-non-chat-content-top-inset,var\(--studio-content-top-inset,0px\)\)\]/);
  assert.ok(outer.indexOf("pt-[var(") < outer.indexOf("ref={box}"), "the inset wraps the player box");
});

test("the titlebar strip and toolbar share the sidebar's colour with no rule under them; the pill inverts", () => {
  const page = read("features/flows/editor/flows-editor-page.tsx");
  assert.match(page, /flows-editor flex h-full min-h-0 w-full flex-col bg-sidebar/);
  assert.match(page, /<div className="relative flex min-h-0 flex-1 overflow-hidden rounded-tl-xl bg-background">/);
  const bar = read("features/flows/editor/toolbar.tsx");
  assert.match(bar, /items-center gap-3 bg-sidebar px-2 text-foreground">/);
  assert.doesNotMatch(bar, /border-b bg-background px-2/);
  assert.match(bar, /rounded-full bg-background p-\[3px\]/);
  assert.match(bar, /wireStyle === style \? "bg-sidebar-accent/);
});
