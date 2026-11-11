// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { normaliseComposerPosition, useChatPreferencesStore } =
  await import("../src/features/chat/stores/chat-preferences-store.ts");

test("new installs keep the composer in the middle", () => {
  assert.equal(useChatPreferencesStore.getInitialState().newChatComposerPosition, 50);
});

test("the old on/off setting maps to bottom or middle", () => {
  assert.equal(normaliseComposerPosition(undefined, true), 100);
  assert.equal(normaliseComposerPosition(undefined, false), 50);
  assert.equal(normaliseComposerPosition(30, true), 30);
});

test("stored positions are clamped whole numbers", () => {
  assert.equal(normaliseComposerPosition(-5), 0);
  assert.equal(normaliseComposerPosition(140), 100);
  assert.equal(normaliseComposerPosition(42.6), 43);
  assert.equal(normaliseComposerPosition(Number.NaN), 50);
  assert.equal(normaliseComposerPosition("80"), 50);
});

test("only the top of the range docks the composer", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /s\.newChatComposerPosition >= COMPOSER_POSITION_BOTTOM/);
  assert.match(thread, /const composerAtBottom = composerPosition >= COMPOSER_POSITION_BOTTOM;/);
  assert.match(thread, /pt-\[calc\(27\.5dvh\*var\(--welcome-lift,1\)\)\]/);
});

test("an image greeting keeps room for its raised row at low composer positions", () => {
	const thread = readSrc("components/assistant-ui/thread.tsx");
	assert.match(thread, /-mt-\[calc\(76px\*var\(--ui-space-scale,1\)\)\]/);
	assert.match(thread, /greeting\.kind === "image" && !incognito\n\s*\? "pt-\[max\(calc\(76px\*var\(--ui-space-scale,1\)\),calc\(27\.5dvh\*var\(--welcome-lift,1\)\)\)\]"/);
});
