// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  readUseTunnelPref,
  subscribeUseTunnelPref,
  writeUseTunnelPref,
} from "../src/features/settings/components/tunnel-preference.ts";

test("a failed storage write still publishes the current tunnel choice", () => {
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  let notifications = 0;
  const unsubscribe = subscribeUseTunnelPref(() => {
    notifications += 1;
  });

  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem: () => "true",
        setItem: () => {
          throw new Error("storage blocked");
        },
      },
    },
  });

  try {
    assert.equal(readUseTunnelPref(), true);
    writeUseTunnelPref(false);
    assert.equal(readUseTunnelPref(), false);
    assert.equal(notifications, 1);
  } finally {
    unsubscribe();
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});
