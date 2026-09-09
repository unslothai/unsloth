// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// use-tauri-backend.ts pulls in React and the Tauri APIs, so the message choice
// lives in its own module.
import {
  LLAMA_RUNTIME_REASONS,
  WORKING_DIRECTORY_UNAVAILABLE,
  PATH_SETTING_UNRESOLVABLE,
  preflightStaleMessage,
} from "../src/hooks/backend-preflight-message.ts";

const UNREACHABLE_PROFILE = /cannot reach your user folder/;
const UPDATE_ADVICE = /unsloth studio update/;
const MANAGED_TOO_OLD = /Managed Unsloth install is too old/;
const OWNED_TOO_OLD = /Desktop-owned Unsloth backend is too old/;
const TOO_OLD = /too old/;
const RUNTIME_MISSING_FILES = /llama\.cpp runtime is missing files/;

test("an unreachable profile is not reported as an outdated install", () => {
  for (const disposition of ["managed_stale", "owned_stale"]) {
    const message = preflightStaleMessage(
      disposition,
      WORKING_DIRECTORY_UNAVAILABLE,
    );
    assert.match(message, UNREACHABLE_PROFILE);
    // Updating needs the same folder, so it must not be the advice.
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
  // The probe is ungated, so this reason reaches Linux and macOS, where the cause
  // is an unmounted home or permissions rather than a roaming profile.
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
      // Symptom and remedy still survive the trim.
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

test("a path setting that cannot be resolved is told apart from an unreachable folder", () => {
  const message = preflightStaleMessage("managed_stale", PATH_SETTING_UNRESOLVABLE);
  assert.match(message, /folder settings/);
  // Not the profile, and not fixable by an update: the value is the fix.
  assert.doesNotMatch(message, /user folder/);
  assert.doesNotMatch(message, /update/);
});

test("the setting that could not be resolved is named", () => {
  // "one of Unsloth's folder settings" is not actionable, so the backend appends
  // the name and the message uses it.
  const named = preflightStaleMessage(
    "managed_stale",
    `${PATH_SETTING_UNRESOLVABLE}:HF_HOME`,
  );
  assert.match(named, /^HF_HOME points somewhere that cannot be resolved/);
  assert.match(named, /full path/);
  // Without a name it still reads as a sentence, and still is not an update.
  const unnamed = preflightStaleMessage("managed_stale", PATH_SETTING_UNRESOLVABLE);
  assert.match(unnamed, /One of Unsloth's folder settings points/);
  assert.doesNotMatch(unnamed, UPDATE_ADVICE);
});

test("a quarantined llama.cpp runtime is not reported as an outdated install", () => {
  // "too old" would send the user to an update that reports they are up to date.
  // Every reason the health check can produce must reach the new message.
  for (const reason of LLAMA_RUNTIME_REASONS) {
    for (const disposition of ["managed_stale", "owned_stale"]) {
      const message = preflightStaleMessage(disposition, reason);
      assert.match(message, RUNTIME_MISSING_FILES);
      assert.doesNotMatch(message, TOO_OLD);
      // A reinstall is the fix here, so the update command must survive.
      assert.match(message, UPDATE_ADVICE);
      // A reinstall into the same quarantine needs an exclusion, not a retry.
      assert.match(message, /quarantined/);
    }
  }
});

test("the llama runtime reasons match the ones the Python and Rust sides send", () => {
  // installed_runtime_health() returns the first three; managed.rs substitutes the
  // fourth when the CLI reports llama_runtime_ok false with an empty reason. A
  // string that drifts on either side silently falls through to "too old".
  assert.deepEqual(LLAMA_RUNTIME_REASONS, [
    "llama_runtime_dir_missing",
    "llama_runtime_payload_incomplete",
    "llama_runtime_binaries_missing",
    "llama_runtime_incomplete",
  ]);
});

test("an unrelated stale reason is still reported as an outdated install", () => {
  // Only a reason that names the runtime may change the message.
  for (const reason of [
    "backend_outdated",
    "studio_install_incomplete",
    "desktop_backend_ownership_unsupported",
    // Near misses: the match is on the whole token before the colon.
    "llama_runtime",
    "not_llama_runtime_dir_missing",
    "llama_runtime_dir_missing_extra",
  ]) {
    const message = preflightStaleMessage("managed_stale", reason);
    assert.match(message, MANAGED_TOO_OLD);
    assert.doesNotMatch(message, RUNTIME_MISSING_FILES);
  }
  assert.match(
    preflightStaleMessage("owned_stale", "backend_outdated"),
    OWNED_TOO_OLD,
  );
});

test("a suffix on a llama runtime reason does not change the message", () => {
  // The split applies to every reason, so a runtime reason that ever gains a
  // suffix must still match on the token before the colon.
  for (const reason of LLAMA_RUNTIME_REASONS) {
    assert.match(
      preflightStaleMessage("managed_stale", `${reason}:llama-server.exe`),
      RUNTIME_MISSING_FILES,
    );
    // split(":", 2) drops anything after the second colon, harmless since the
    // message ignores the suffix.
    assert.match(
      preflightStaleMessage("managed_stale", `${reason}:C:\\quarantine`),
      RUNTIME_MISSING_FILES,
    );
  }
});

test("an absent or empty reason still falls through to the old behaviour", () => {
  // reason is optional and the CLI defaults it to "", so neither may be mistaken
  // for a runtime reason.
  for (const reason of [null, "", ":", ":llama_runtime_dir_missing"]) {
    assert.match(
      preflightStaleMessage("managed_stale", reason),
      MANAGED_TOO_OLD,
    );
    assert.match(preflightStaleMessage("owned_stale", reason), OWNED_TOO_OLD);
  }
});

test("the context reasons still win over the runtime reason", () => {
  // Both are checked first, and both would be made worse by advice to reinstall
  // the runtime into them.
  const unreachable = preflightStaleMessage(
    "managed_stale",
    WORKING_DIRECTORY_UNAVAILABLE,
  );
  assert.match(unreachable, UNREACHABLE_PROFILE);
  assert.doesNotMatch(unreachable, RUNTIME_MISSING_FILES);

  const unresolvable = preflightStaleMessage(
    "managed_stale",
    `${PATH_SETTING_UNRESOLVABLE}:HF_HOME`,
  );
  assert.match(unresolvable, /HF_HOME points somewhere/);
  assert.doesNotMatch(unresolvable, RUNTIME_MISSING_FILES);
});
