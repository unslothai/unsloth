// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the turn rail sits in the left gutter and its dashes grow rightwards", () => {
  const source = readSrc("components/assistant-ui/turn-navigation.tsx");
  assert.match(source, /aui-turn-navigator-gutter @container\/turn-gutter absolute top-0 h-0/);
  // From the panel edge, so an OS switch to classic scroll bars does not push the rail right.
  assert.match(source, /left: "calc\(-1 \* var\(--rail-edge, 0px\)\)"/);
  assert.match(source, /w-6 origin-left scale-x-\[0\.333\]/);
  // 18px from the sidebar: -0.5rem in the gutter plus the button's pl-1.5, inside the thread's px-5.
  assert.match(source, /absolute top-0 left-\[-0\.5rem\] flex w-9/);
  assert.match(source, /left: `calc\(\$\{preview\.left\}px \+ 2px\)`/);
  assert.match(source, /opacity-0[^"]*hover:opacity-100 focus-within:opacity-100 @min-\[2\.25rem\]\/turn-gutter:opacity-100/);
});

test("scrolling restyles no rail markers: no colour transition, no inherited fade properties", () => {
  const source = readSrc("components/assistant-ui/turn-navigation.tsx");
  // Tailwind's scale-x-* sets `scale`, which a `transform` transition does not animate.
  assert.match(source, /rounded-full transition-\[scale\] duration-150/);
  assert.doesNotMatch(source, /transition-\[[^\]]*(?:background-color|transform)/);
  assert.doesNotMatch(source, /--rail-fade-/);
  assert.match(source, /setProperty\("mask-image", next\)/);
});

test("turn navigation is on by default and stays off once turned off", async () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /<TurnNavigator viewportRef=\{viewportElRef\} \/>\s*\{\/\* Drop-in for ThreadPrimitive\.Messages/);
  const prefs = readSrc("features/chat/stores/chat-preferences-store.ts");
  assert.match(prefs, /showTurnNavigation: true,/);
  assert.match(prefs, /showTurnNavigation: saved\?\.showTurnNavigation !== false,/);
});
