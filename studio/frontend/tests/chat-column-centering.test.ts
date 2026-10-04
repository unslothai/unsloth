// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { atDefaultUiScale, readSrcAsync } from "./helpers/kit.ts";

const DOCK_INSETS =
  /aui-thread-composer-dock[^"]*md:left-\[var\(--thread-scrollbar-gutter,10px\)\] md:right-\[var\(--thread-scrollbar-gutter,10px\)\]/;
const DOCK_ONE_SIDED =
  /aui-thread-composer-dock[^"]*right-0 md:right-\[var\(--thread-scrollbar-gutter,10px\)\]"/;
const SCALED_GUTTER_INSET =
  /(?:left|right)-\[calc\(10px\*var\(--ui-space-scale,1\)\)\]/;

// The chat column is centred by margin, so every inset around it has to be
// symmetric. A one-sided scrollbar gutter counts as an inset.

test("the thread viewport reserves its scrollbar gutter on both edges", async () => {
  const css = atDefaultUiScale(await readSrcAsync("index.css"));

  const at = css.indexOf(".aui-thread-viewport {");
  assert.notEqual(at, -1);
  const rule = css.slice(at, css.indexOf("}", at));
  assert.match(rule, /scrollbar-gutter:\s*stable both-edges;/);

  // Messages sit inside this viewport, the dock outside it. A one-sided
  // gutter offsets only the first, so they stop agreeing with each other.
  assert.doesNotMatch(rule, /scrollbar-gutter:\s*stable;/);
});

test("nothing around the composer re-adds a one-sided inset", async () => {
  const chatPage = atDefaultUiScale(await readSrcAsync("features/chat/chat-page.tsx"));
  const thread = atDefaultUiScale(await readSrcAsync("components/assistant-ui/thread.tsx"));

  // The compare-mode wrapper mirrored the old one-sided gutter.
  assert.doesNotMatch(chatPage, /pl-5 pr-5 md:pr-\[30px\]/);
  assert.match(chatPage, /pl-5 pr-5 md:px-\[30px\]/);

  // The dock's own padding stays even.
  assert.match(thread, /unsloth-composer-dock-inner relative px-5/);

  // The dock offset keeps the bottom fade off the scrollbar, so it stays,
  // but one-sided it also shifts the composer half its width off centre.
  assert.match(thread, DOCK_INSETS);
  assert.doesNotMatch(thread, DOCK_ONE_SIDED);
});

// The gutter is a scrollbar, which the UI scale does not touch, so a scaled inset drifted the
// composer off the message column. Windows lets the 8px ::-webkit-scrollbar through.
test("the overlays around the thread stop at its real scrollbar gutter", async () => {
  const css = await readSrcAsync("index.css");
  for (const rule of [
    ":root {\n\t--thread-scrollbar-gutter: 10px;",
    ":root.client-windows {\n\t\t--thread-scrollbar-gutter: 8px;",
  ]) {
    assert.ok(css.includes(rule), rule);
  }
  for (const file of [
    "features/chat/chat-page.tsx",
    "components/assistant-ui/thread.tsx",
    "features/chat/components/chat-model-notice.tsx",
  ]) {
    assert.doesNotMatch(await readSrcAsync(file), SCALED_GUTTER_INSET, file);
  }
});
