// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The memory bar is opt-in, and that is the whole backwards-compatibility
// story: an existing install upgrading into this build has no such key in
// localStorage, so it must read false and the app must behave exactly as it did
// before. The hook checks the flag before it builds a request, so an install
// that never opts in also generates no extra traffic.
//
// This asserts the source of both, since the default is one boolean and losing
// it would silently turn a new chart on for every existing user.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

const SRC = new URL("../src/", import.meta.url);

const STORE = readFileSync(
  new URL("features/chat/stores/chat-runtime-store.ts", SRC),
  "utf8",
);
const HOOK = readFileSync(new URL("hooks/use-model-memory.ts", SRC), "utf8");

test("the bar is off unless the user turns it on", () => {
  const key = STORE.match(
    /CHAT_SHOW_MEMORY_BAR_KEY\s*=\s*"([^"]+)"/,
  );
  assert.ok(key, "no CHAT_SHOW_MEMORY_BAR_KEY");

  const hydrate = STORE.match(
    /showMemoryBar:\s*loadBool\(\s*CHAT_SHOW_MEMORY_BAR_KEY\s*,\s*(true|false)\s*\)/,
  );
  assert.ok(hydrate, "showMemoryBar is not hydrated through loadBool");
  assert.equal(
    hydrate[1],
    "false",
    "an install with no such key would get the bar switched on",
  );
});

test("a disabled bar issues no estimate request", () => {
  // The enabled check has to come before the plan is built, or a user who never
  // opted in still pays one backend read per visible row.
  const plan = HOOK.indexOf("const plan = useMemo(");
  assert.ok(plan > 0, "no plan memo");
  // The first statement of the memo, so no request is even described when the
  // feature is off. Searched from the memo rather than from the top of the
  // file, since the budget effect above it carries its own enabled guard.
  // Widened from 200 characters: the guard is still the memo's first statement,
  // but it now carries a comment explaining the direct-.gguf case above it, and
  // a window sized to the code alone fails on the prose rather than on the
  // property. Still bounded, so a check buried after real work does not pass.
  assert.match(
    HOOK.slice(plan, plan + 600),
    /if \(!enabled\b/,
    "the plan memo does not stand down when the bar is disabled",
  );

  // And the fetch has to be gated on that plan rather than run unconditionally.
  assert.match(
    HOOK,
    /useEffect\(\(\) => \{\s*if \(!plan\) return;/,
    "the estimate effect does not stand down when there is no plan",
  );
});

test("the settings request is gated on a row that will draw a bar", () => {
  // The VRAM budget lookup is a second request, and its gate is `plan` rather
  // than `enabled`. That is strictly stronger: the plan memo stands down when the
  // feature is off, so a plan can only exist when enabled is true, and it also
  // excludes the remote and undownloaded rows that mount this hook but will never
  // draw a bar. loadVramBudgetSettings folds together only calls already in
  // flight -- it keeps no read-through cache -- so gating on the flag alone made
  // scrolling a long list issue one request per row that appeared.
  const effect = HOOK.match(
    /useEffect\(\(\) => \{\s*if \(!plan\) return;[\s\S]*?loadVramBudgetSettings/,
  );
  assert.ok(
    effect,
    "the vram budget request is not gated on the row having a plan",
  );
});

test("Reset All clears the memory-bar opt-in", () => {
  // The feature's default is off, so a reset that leaves the key behind leaves
  // the bar switched on while telling the user it restored defaults.
  const GENERAL_TAB = readFileSync(
    new URL("features/settings/tabs/general-tab.tsx", SRC),
    "utf8",
  );
  const key = STORE.match(/CHAT_SHOW_MEMORY_BAR_KEY\s*=\s*"([^"]+)"/);
  assert.ok(key, "no CHAT_SHOW_MEMORY_BAR_KEY");
  // The reset list spells the key out because importing it would hit an import
  // cycle and read the constant in its temporal dead zone. That makes drift
  // possible, which is what this pins.
  assert.match(
    GENERAL_TAB,
    new RegExp(`PREFS_KEYS[\\s\\S]*?"${key[1]}"`),
    `Reset All does not clear ${key[1]}`,
  );
});

test("only a row that could draw a bar watches the runtime store", () => {
  // The subscription watches the whole chat runtime store, which ticks on every
  // streamed token, and re-reads the epoch (localStorage included) each time.
  // Every mounted picker row calls this hook, most with no source at all, so
  // gating on the opt-in alone multiplied each token by the row count.
  assert.match(
    HOOK,
    /const watching = enabled && source != null;/,
    "the store subscription is not gated on the row having a source",
  );
  assert.match(
    HOOK,
    /useSyncExternalStore\(\s*watching \? subscribeToConfigChanges : subscribeNothing/,
    "the subscription does not stand down for a row that cannot draw",
  );
});

test("the session GPU pin moves the config epoch", () => {
  // budgetIsMeaningful reads selectedGpuIds, but the pin lives in the runtime
  // store rather than in a saved config, so a preset changes it with no config
  // write. Without it in the snapshot the store update fires and
  // useSyncExternalStore suppresses the rerender anyway, leaving a bar drawn
  // against aggregate VRAM after the launch was pinned to one card.
  assert.match(
    HOOK,
    /pinSignature[\s\S]{0,160}selectedGpuIds/,
    "the epoch signature ignores the session GPU pin",
  );
  // The namespace too: the same ordinals mean different cards under a different
  // index kind, and the field is selectedGpuIndexKind, not gpuIndexKind.
  assert.match(
    HOOK,
    /selectedGpuIndexKind/,
    "the epoch signature ignores the pin's index namespace",
  );
  assert.match(
    HOOK,
    /const prefSignature = [^;]*pinSignature/,
    "pinSignature is computed but not folded into the epoch signature",
  );
});
