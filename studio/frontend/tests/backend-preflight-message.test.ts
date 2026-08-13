// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// use-tauri-backend.ts pulls in React and the Tauri APIs, so the message choice
// lives in its own module and is driven directly here.
import {
  WORKING_DIRECTORY_UNAVAILABLE,
  preflightStaleMessage,
} from "../src/hooks/backend-preflight-message.ts";

const UNREACHABLE_PROFILE = /cannot reach your user folder/;
const UPDATE_ADVICE = /unsloth studio update/;
const MANAGED_TOO_OLD = /Managed Unsloth install is too old/;
const OWNED_TOO_OLD = /Desktop-owned Unsloth backend is too old/;
const TOO_OLD = /too old/;

test("an unreachable profile is not reported as an outdated install", () => {
  for (const disposition of ["managed_stale", "owned_stale"]) {
    const message = preflightStaleMessage(
      disposition,
      WORKING_DIRECTORY_UNAVAILABLE,
    );
    assert.match(message, UNREACHABLE_PROFILE);
    // Updating needs the same folder, so it must not be the advice given.
    assert.doesNotMatch(message, UPDATE_ADVICE);
  }
});

test("a genuinely stale install still says to update", () => {
  assert.match(
    preflightStaleMessage("managed_stale", "old cli"),
    MANAGED_TOO_OLD,
  );
  assert.match(
    preflightStaleMessage("owned_stale", "backend_outdated"),
    OWNED_TOO_OLD,
  );
  assert.match(preflightStaleMessage("managed_stale", null), TOO_OLD);
});

test("the reason string matches the one the Rust side sends", () => {
  assert.equal(WORKING_DIRECTORY_UNAVAILABLE, "working_directory_unavailable");
});

test("the roaming-profile cause is offered on Windows and withheld elsewhere", () => {
  // `home_dir_available()` is called ungated from the preflight probe, so this
  // reason reaches Linux and macOS too. There the same symptom is an unmounted
  // home or a permissions problem, and a roaming profile is a Windows concept.
  const original = globalThis.navigator;
  const withPlatform = (platform: string) => {
    Object.defineProperty(globalThis, "navigator", {
      value: { platform },
      configurable: true,
    });
    return preflightStaleMessage("managed_stale", WORKING_DIRECTORY_UNAVAILABLE);
  };
  try {
    assert.match(withPlatform("Win32"), /roaming profile/);
    for (const platform of ["Linux x86_64", "MacIntel"]) {
      const message = withPlatform(platform);
      assert.doesNotMatch(message, /roaming profile/);
      // The symptom and the remedy still have to survive the trim.
      assert.match(message, UNREACHABLE_PROFILE);
      assert.match(message, /Reconnect and try again/);
    }
  } finally {
    Object.defineProperty(globalThis, "navigator", {
      value: original,
      configurable: true,
    });
  }
});
