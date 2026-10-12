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

// The chat column is centred by margin, so every inset, including the scrollbar gutter, is symmetric.

test("the thread viewport reserves its scrollbar gutter on both edges", async () => {
  const css = atDefaultUiScale(await readSrcAsync("index.css"));

  const at = css.indexOf(".aui-thread-viewport {");
  assert.notEqual(at, -1);
  const rule = css.slice(at, css.indexOf("}", at));
  assert.match(rule, /scrollbar-gutter:\s*stable both-edges;/);

  assert.doesNotMatch(rule, /scrollbar-gutter:\s*stable;/);
});

test("nothing around the composer re-adds a one-sided inset", async () => {
  const chatPage = atDefaultUiScale(await readSrcAsync("features/chat/chat-page.tsx"));
  const thread = atDefaultUiScale(await readSrcAsync("components/assistant-ui/thread.tsx"));

  assert.doesNotMatch(chatPage, /pl-5 pr-5 md:pr-\[30px\]/);
  assert.match(chatPage, /pl-5 pr-5 md:px-\[30px\]/);

  assert.match(thread, /unsloth-composer-dock-inner relative px-5/);

  assert.match(thread, DOCK_INSETS);
  assert.doesNotMatch(thread, DOCK_ONE_SIDED);
});

// The gutter is a scrollbar, which the UI scale does not touch, so a scaled inset drifted the
// composer off the message column. The thread keeps Chromium's thin standard scrollbar on
// Windows too, so there is one gutter width everywhere.
test("the overlays around the thread stop at its real scrollbar gutter", async () => {
  const css = await readSrcAsync("index.css");
  assert.match(css, /@property --thread-scrollbar-gutter \{\s*syntax: "<length>";\s*inherits: false;\s*initial-value: 10px;/);
  assert.ok(!css.includes("--thread-scrollbar-gutter: 8px;"));
  assert.match(
    css,
    /:root\.client-windows \.aui-thread-viewport \{\s*scrollbar-width: thin;/,
  );
  for (const file of [
    "features/chat/chat-page.tsx",
    "components/assistant-ui/thread.tsx",
    "features/chat/components/chat-model-notice.tsx",
  ]) {
    assert.doesNotMatch(await readSrcAsync(file), SCALED_GUTTER_INSET, file);
  }
});
