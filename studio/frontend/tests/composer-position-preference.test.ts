// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { normaliseComposerPosition, useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

test("new installs put the composer a little above the middle", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().newChatComposerPosition,
    40,
  );
});

test("the old on/off setting maps to bottom or the default", () => {
  assert.equal(normaliseComposerPosition(undefined, true), 100);
  assert.equal(normaliseComposerPosition(undefined, false), 40);
  assert.equal(normaliseComposerPosition(30, true), 30);
});

test("stored positions are clamped whole numbers", () => {
  assert.equal(normaliseComposerPosition(-5), 0);
  assert.equal(normaliseComposerPosition(140), 100);
  assert.equal(normaliseComposerPosition(42.6), 43);
  assert.equal(normaliseComposerPosition(Number.NaN), 40);
  assert.equal(normaliseComposerPosition("80"), 40);
});

test("only the top of the range docks the composer", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /s\.newChatComposerPosition >= COMPOSER_POSITION_BOTTOM/,
  );
  // A short window keeps it centred: docking lifts the corner rail over the composer.
  assert.doesNotMatch(thread, /max-height: 34rem/);
  assert.equal(thread.match(/= useNewChatComposerDocked\(\);/g)?.length, 2);
});

test("the composer box alone is centred, the greeting 16px above it", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  // Spacers split the free height by the setting, so 50 centres the composer itself.
  assert.ok(
    thread.includes("calc(var(--welcome-spacer-grow,1)*${composerPosition})"),
  );
  assert.ok(
    thread.includes(
      "calc(var(--welcome-spacer-grow,1)*${COMPOSER_POSITION_BOTTOM - composerPosition})",
    ),
  );
  // The greeting rides the top spacer's foot, so it never shifts the composer.
  assert.match(
    thread,
    /aui-thread-welcome-spacer flex w-full basis-0 flex-col justify-end"/,
  );
  assert.match(
    thread,
    /<div className="flex flex-col pb-\[calc\(16px\*var\(--ui-space-scale,1\)\)\]">\{greetingRow\}<\/div>/,
  );
});
