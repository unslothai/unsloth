// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the turn rail sits in the left gutter and its dashes grow rightwards", () => {
  const source = readSrc("components/assistant-ui/turn-navigation.tsx");
  assert.match(source, /aui-turn-navigator-gutter @container\/turn-gutter absolute top-0 left-0/);
  assert.match(source, /w-6 origin-left scale-x-\[0\.333\]/);
  assert.match(source, /left: `calc\(\$\{preview\.left\}px \+ 0\.5rem\)`/);
  assert.match(source, /opacity-0[^"]*hover:opacity-100 focus-within:opacity-100 @min-\[2\.25rem\]\/turn-gutter:opacity-100/);
});

test("turn navigation is on by default and stays off once turned off", async () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /<TurnNavigator viewportRef=\{viewportElRef\} \/>\s*\{\/\* Drop-in for ThreadPrimitive\.Messages/);
  const prefs = readSrc("features/chat/stores/chat-preferences-store.ts");
  assert.match(prefs, /showTurnNavigation: true,/);
  assert.match(prefs, /showTurnNavigation: saved\?\.showTurnNavigation !== false,/);
});
