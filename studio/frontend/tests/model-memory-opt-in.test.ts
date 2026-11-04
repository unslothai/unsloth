// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Off by default so upgrading installs behave as before and send no extra requests.

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
  const plan = HOOK.indexOf("const plan = useMemo(");
  assert.ok(plan > 0, "no plan memo");
  // Bounded window so a guard buried after real work does not pass.
  assert.match(
    HOOK.slice(plan, plan + 600),
    /if \(!enabled\b/,
    "the plan memo does not stand down when the bar is disabled",
  );

  assert.match(
    HOOK,
    /useEffect\(\(\) => \{\s*if \(!plan\) return;/,
    "the estimate effect does not stand down when there is no plan",
  );
});

test("the settings request is gated on a row that will draw a bar", () => {
  // loadVramBudgetSettings has no cache, so gating on plan avoids one request per row.
  const effect = HOOK.match(
    /useEffect\(\(\) => \{\s*if \(!plan\) return;[\s\S]*?loadVramBudgetSettings/,
  );
  assert.ok(
    effect,
    "the vram budget request is not gated on the row having a plan",
  );
});

test("Reset All clears the memory-bar opt-in", () => {
  const GENERAL_TAB = readFileSync(
    new URL("features/settings/tabs/general-tab.tsx", SRC),
    "utf8",
  );
  const key = STORE.match(/CHAT_SHOW_MEMORY_BAR_KEY\s*=\s*"([^"]+)"/);
  assert.ok(key, "no CHAT_SHOW_MEMORY_BAR_KEY");
  // The reset list spells the key out to avoid an import cycle (TDZ), so drift is possible.
  assert.match(
    GENERAL_TAB,
    new RegExp(`PREFS_KEYS[\\s\\S]*?"${key[1]}"`),
    `Reset All does not clear ${key[1]}`,
  );
});

test("only a row that could draw a bar watches the runtime store", () => {
  // The store ticks per streamed token, so ungated subscriptions multiply work by row count.
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
  // The GPU pin lives in the runtime store, so it must be in the snapshot to rerender.
  assert.match(
    HOOK,
    /pinSignature[\s\S]{0,160}selectedGpuIds/,
    "the epoch signature ignores the session GPU pin",
  );
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
