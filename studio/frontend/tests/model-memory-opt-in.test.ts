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
  assert.match(
    HOOK.slice(plan, plan + 200),
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

test("the settings request is gated on the same flag", () => {
  // The VRAM budget lookup is a second request; it must respect the opt-in too.
  const effect = HOOK.match(
    /useEffect\(\(\) => \{\s*if \(!enabled\) return;[\s\S]*?loadVramBudgetSettings/,
  );
  assert.ok(
    effect,
    "the vram budget request is not gated on the memory bar being enabled",
  );
});
