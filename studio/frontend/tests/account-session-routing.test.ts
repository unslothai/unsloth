// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as Session from "../src/features/auth/session.ts";

for (const [desktop, loginMode, change, expected] of [
  [true, "single", true, "/chat"],
  [true, "single", false, "/chat"],
  [true, "multi", true, "/change-password"],
  [true, "multi", false, "/chat"],
  [false, "single", true, "/change-password"],
  [false, "multi", true, "/change-password"],
] as const) {
  test(`${desktop ? "desktop" : "browser"} ${loginMode} session with change=${change} routes to ${expected}`, (t) => {
    const previousWindow = globalThis.window;
    const previousStorage = globalThis.localStorage;
    const storage = {
      getItem: () => (change ? "1" : null),
    } as unknown as Storage;
    globalThis.localStorage = storage;
    globalThis.window = {
      localStorage: storage,
      addEventListener() {},
    } as unknown as Window & typeof globalThis;
    t.after(() => {
      globalThis.window = previousWindow;
      globalThis.localStorage = previousStorage;
    });
    let listening = 0;
    const session = loadWithStubs<typeof Session>(
      new URL("../src/features/auth/session.ts", import.meta.url),
      {
        "@/lib/api-base": { isTauri: desktop },
        "@/lib/account-transition": {
          installAccountTransitionListener: () => {
            listening++;
          },
        },
        "./login-client": { getLoginMode: () => loginMode },
        "./session-events": {},
      },
    );
    assert.equal(session.getPostAuthRoute(), expected);
    assert.equal(listening, 1);
  });
}

// Account settings are read once per session, and a session still owing a password change is
// refused them with a 403. The bootstrap sign-in on /change-password stores its tokens before the
// change, so "has a token" is not "may read settings": a read in that gap leaves personalization
// unhydrated for the whole session. The gate agrees with the route the session is sent to.
for (const [desktop, loginMode, token, change, expected] of [
  [false, "single", false, false, false],
  [false, "single", true, true, false],
  [false, "multi", true, true, false],
  [false, "single", true, false, true],
  [true, "multi", true, true, false],
  [true, "single", true, true, true],
  [true, "single", false, false, false],
] as const) {
  test(`${desktop ? "desktop" : "browser"} ${loginMode} session with token=${token} change=${change} is settled=${expected}`, (t) => {
    const previousWindow = globalThis.window;
    const previousStorage = globalThis.localStorage;
    const storage = {
      getItem: (key: string) => {
        if (key === "unsloth_auth_token") return token ? "a.b.c" : null;
        if (key === "unsloth_auth_must_change_password") return change ? "1" : null;
        return null;
      },
    } as unknown as Storage;
    globalThis.localStorage = storage;
    globalThis.window = {
      localStorage: storage,
      addEventListener() {},
    } as unknown as Window & typeof globalThis;
    t.after(() => {
      globalThis.window = previousWindow;
      globalThis.localStorage = previousStorage;
    });
    const session = loadWithStubs<typeof Session>(
      new URL("../src/features/auth/session.ts", import.meta.url),
      {
        "@/lib/api-base": { isTauri: desktop },
        "@/lib/account-transition": { installAccountTransitionListener: () => {} },
        "./login-client": { getLoginMode: () => loginMode },
        "./session-events": {},
      },
    );
    assert.equal(session.hasSettledAuthSession(), expected);
  });
}
