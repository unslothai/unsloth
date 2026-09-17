// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// The chat column is centred by margin, so every inset around it has to be
// symmetric. A one-sided scrollbar gutter counts as an inset.

test("the thread viewport reserves its scrollbar gutter on both edges", async () => {
  const css = await readSrcAsync("index.css");

  const at = css.indexOf(".aui-thread-viewport {");
  assert.notEqual(at, -1);
  const rule = css.slice(at, css.indexOf("}", at));
  assert.match(rule, /scrollbar-gutter:\s*stable both-edges;/);

  // Messages sit inside this viewport, the dock outside it. A one-sided
  // gutter offsets only the first, so they stop agreeing with each other.
  assert.doesNotMatch(rule, /scrollbar-gutter:\s*stable;/);
});

test("nothing around the composer re-adds a one-sided inset", async () => {
  const chatPage = await readSrcAsync("features/chat/chat-page.tsx");
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");

  // The compare-mode wrapper mirrored the old one-sided gutter.
  assert.doesNotMatch(chatPage, /pl-5 pr-5 md:pr-\[30px\]/);
  assert.match(chatPage, /pl-5 pr-5 md:px-\[30px\]/);

  // The dock's own padding stays even.
  assert.match(thread, /unsloth-composer-dock-inner relative px-5/);

  // The dock offset keeps the bottom fade off the scrollbar, so it stays,
  // but one-sided it also shifts the composer half its width off centre.
  assert.match(thread, /aui-thread-composer-dock[^"]*md:left-\[10px\] md:right-\[10px\]/);
  assert.doesNotMatch(thread, /aui-thread-composer-dock[^"]*right-0 md:right-\[10px\]"/);
});
