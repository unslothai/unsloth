// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hook is .tsx, which node's type stripping cannot import, so its shape is pinned from source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { useChatPreferencesStore } =
  await import("../src/features/chat/stores/chat-preferences-store.ts");

const HOOK = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");
const STORE = readSrc("features/chat/stores/chat-preferences-store.ts");

function body(signature: string): string {
  const start = HOOK.indexOf(signature);
  assert.notEqual(start, -1, `${signature} is gone; this test needs rewriting`);
  return HOOK.slice(start, HOOK.indexOf("\n      };", start));
}

test("auto-scroll follows by default, including installs that never saw the setting", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().autoScrollWhileGenerating,
    true,
  );
  assert.match(
    STORE,
    /autoScrollWhileGenerating: saved\?\.autoScrollWhileGenerating \?\? true/,
  );
});

test("the setting is an Auto / Manual pill selector, not a switch", () => {
  const TAB = readSrc("features/settings/tabs/chat-tab.tsx");
  const at = TAB.indexOf('label={t("settings.chat.autoScroll")}');
  assert.notEqual(at, -1);
  const row = TAB.slice(at, TAB.indexOf("</SettingsRow>", at));
  assert.match(row, /hub-tab-toggle/);
  assert.match(row, /settings\.chat\.autoScrollAuto/);
  assert.match(row, /settings\.chat\.autoScrollManual/);
  assert.ok(!row.includes("<Switch"));
});

test("the setting is read per call, so flipping it mid-run applies at once", () => {
  const holdStill = body("const holdStill = (): boolean =>");
  assert.match(
    holdStill,
    /useChatPreferencesStore\.getState\(\)\.autoScrollWhileGenerating/,
  );
  assert.match(holdStill, /aui\.thread\(\)\.getState\(\)\.isRunning/);
});

test("only a run started on screen holds still, not one already streaming when opened", () => {
  const holdStill = body("const holdStill = (): boolean =>");
  assert.match(holdStill, /runStartedHereRef\.current &&/);
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

test("a held run follows until its turn reaches the top, then parks", () => {
  const onLayoutChange = body("const onLayoutChange = (): void => {");
  assert.match(onLayoutChange, /if \(!parkIfHeld\(\)\) \{\s*extendFollow\(\);/);
  const park = body("const parkIfHeld = (): boolean => {");
  assert.match(park, /!holdStill\(\)/);
  assert.match(park, /holdCeiling\(\)/);
  assert.match(
    park,
    /el\.scrollTo\(\{ top: ceiling, behavior: "instant" \}\);\s*detach\(\);/,
  );
  const ceiling = body("const holdCeiling = (): number | null => {");
  assert.match(ceiling, /querySelectorAll<HTMLElement>\("\[data-role\]"\)/);
  const tick = body("const tick = (): void => {");
  assert.ok(tick.indexOf("parkIfHeld();") < tick.indexOf("const following"));
});

test("the scroll-to-bottom button ends the hold for the rest of the run", () => {
  const at = HOOK.indexOf("const scrollToBottom = useCallback<ScrollToBottom>");
  assert.notEqual(at, -1);
  assert.match(
    HOOK.slice(at, HOOK.indexOf("}, []);", at)),
    /runStartedHereRef\.current = false;/,
  );
});

test("reaching the bottom by hand does not re-attach while holding still", () => {
  const onScroll = body("const onScroll = () => {");
  assert.match(
    onScroll,
    /distanceNow <= RE_ATTACH_THRESHOLD_PX &&\s*!holdStill\(\)/,
  );
});

test("the hold outlives runEnd briefly, so the last chunk still parks", () => {
  const holdStill = body("const holdStill = (): boolean =>");
  assert.match(
    holdStill,
    /performance\.now\(\) - runEndAtRef\.current < FOLLOW_SETTLE_MS/,
  );
  const runEnd = HOOK.indexOf('useAuiEvent("thread.runEnd"');
  assert.notEqual(runEnd, -1);
  assert.match(
    HOOK.slice(runEnd, HOOK.indexOf("});", runEnd)),
    /runEndAtRef\.current = performance\.now\(\);/,
  );
});

test("turning auto-scroll on mid-run releases a park", () => {
  const park = body("const parkIfHeld = (): boolean => {");
  assert.match(park, /detach\(\);\s*parked = true;/);
  const at = HOOK.indexOf("useChatPreferencesStore.subscribe(");
  assert.notEqual(at, -1);
  const listener = HOOK.slice(at, HOOK.indexOf("\n      );", at));
  assert.match(listener, /!parked \|\|/);
  assert.match(listener, /prev\.autoScrollWhileGenerating/);
  assert.match(
    listener,
    /userDetachedRef\.current = false;\s*extendFollow\(\);/,
  );
  assert.match(HOOK, /unsubscribePreferences\(\);/);
});
