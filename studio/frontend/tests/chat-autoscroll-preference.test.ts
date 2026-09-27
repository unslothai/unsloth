// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Auto-scroll while generating" (issue #11665): off, a streaming response grows below the reader
// instead of dragging the thread to the bottom. The hook is a .tsx, which node's type stripping
// cannot import, so its shape is pinned from source as chat-autoscroll-frame-budget.test.ts does.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

const HOOK = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");
const STORE = readSrc("features/chat/stores/chat-preferences-store.ts");

function body(signature: string): string {
  const start = HOOK.indexOf(signature);
  assert.notEqual(start, -1, `${signature} is gone; this test needs rewriting`);
  return HOOK.slice(start, HOOK.indexOf("\n      };", start));
}

test("auto-scroll is off for installs that never saw the setting", () => {
  // Off by default: most readers start at the top of a response, so following the stream is
  // the opt in. A saved payload without the key rehydrates to the same.
  assert.equal(
    useChatPreferencesStore.getInitialState().autoScrollWhileGenerating,
    false,
  );
  assert.match(
    STORE,
    /autoScrollWhileGenerating: saved\?\.autoScrollWhileGenerating \?\? false/,
  );
});

test("the setting is read per call, so flipping it mid-run applies at once", () => {
  const holdStill = body("const holdStill = (): boolean =>");
  assert.match(
    holdStill,
    /useChatPreferencesStore\.getState\(\)\.autoScrollWhileGenerating/,
  );
  // Only a running thread holds still: opening or switching a chat still lands on the bottom.
  assert.match(holdStill, /aui\.thread\(\)\.getState\(\)\.isRunning/);
});

test("only a run started on screen holds still, not one already streaming when opened", () => {
  const holdStill = body("const holdStill = (): boolean =>");
  assert.match(holdStill, /runStartedHereRef\.current &&/);
  // An opened chat's messages arrive after its pin window, so holding still there would strand
  // the reader at the top of a chat they never saw.
  for (const event of ["thread.initialize", "threadListItem.switchedTo"]) {
    const at = HOOK.indexOf(`useAuiEvent("${event}"`);
    assert.notEqual(at, -1, event);
    assert.match(
      HOOK.slice(at, HOOK.indexOf("});", at)),
      /runStartedHereRef\.current = false;/,
    );
  }
  const runStart = HOOK.indexOf('useAuiEvent("thread.runStart"');
  assert.match(
    HOOK.slice(runStart, HOOK.indexOf("});", runStart)),
    /runStartedHereRef\.current = true;/,
  );
});

test("streamed content does not extend the follow window while holding still", () => {
  const onLayoutChange = body("const onLayoutChange = (): void => {");
  assert.match(onLayoutChange, /if \(!holdStill\(\)\) \{\s*extendFollow\(\);/);
  // Detaching once the send-time pin lapses is what stops the run's final layout change
  // re-opening the window and yanking the reader to the bottom as the response ends.
  assert.match(onLayoutChange, /detach\(\);/);
});

test("reaching the bottom by hand does not re-attach while holding still", () => {
  const onScroll = body("const onScroll = () => {");
  assert.match(
    onScroll,
    /distanceNow <= RE_ATTACH_THRESHOLD_PX &&\s*!holdStill\(\)/,
  );
});
