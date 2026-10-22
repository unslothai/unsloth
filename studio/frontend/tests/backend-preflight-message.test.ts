// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  LLAMA_RUNTIME_REASONS,
  MANAGED_ENVIRONMENT_BUSY,
  MANAGED_ENVIRONMENT_UPDATING,
  WORKING_DIRECTORY_UNAVAILABLE,
  PATH_SETTING_UNRESOLVABLE,
  preflightStaleMessage,
  runtimeRepairFailureMessage,
  runtimeRepairRecurrenceMessage,
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
  assert.doesNotMatch(message, /user folder/);
  assert.doesNotMatch(message, /update/);
});

test("the setting that could not be resolved is named", () => {
  const named = preflightStaleMessage(
    "managed_stale",
    `${PATH_SETTING_UNRESOLVABLE}:HF_HOME`,
  );
  assert.match(named, /^HF_HOME points somewhere that cannot be resolved/);
  assert.match(named, /full path/);
  const unnamed = preflightStaleMessage("managed_stale", PATH_SETTING_UNRESOLVABLE);
  assert.match(unnamed, /One of Unsloth's folder settings points/);
  assert.doesNotMatch(unnamed, UPDATE_ADVICE);
});

test("a quarantined llama.cpp runtime is not reported as an outdated install", () => {
  for (const reason of LLAMA_RUNTIME_REASONS) {
    for (const disposition of ["managed_stale", "owned_stale"]) {
      const message = preflightStaleMessage(disposition, reason);
      assert.match(message, RUNTIME_MISSING_FILES);
      assert.doesNotMatch(message, TOO_OLD);
      assert.match(message, UPDATE_ADVICE);
      assert.match(message, /quarantined/);
      assert.match(message, /\.unsloth[\\/]llama\.cpp/);
    }
  }
});

test("a failed repair and recurring damage show antivirus advice with the runtime folder", () => {
  const failure = runtimeRepairFailureMessage("download blocked");
  assert.match(failure, /Antivirus may be blocking the download/);
  assert.match(failure, /or check your connection/);
  assert.match(failure, /Repair error: download blocked/);
  assert.match(failure, /\.unsloth[\\/]llama\.cpp/);

  const recurring = runtimeRepairRecurrenceMessage();
  assert.match(recurring, /missing files again soon after a repair/);
  assert.match(recurring, /antivirus/);
  assert.match(recurring, /press Retry to reinstall it/);
  assert.doesNotMatch(recurring, /unsloth studio update/);
  assert.match(recurring, /\.unsloth[\\/]llama\.cpp/);
});

test("the folder to exclude is spelled the way the platform spells it", () => {
  const original = globalThis.navigator;
  const withPlatform = (platform: string) => {
    Object.defineProperty(globalThis, "navigator", {
      value: { platform },
      configurable: true,
    });
    return preflightStaleMessage("managed_stale", "llama_runtime_payload_incomplete");
  };
  try {
    const windows = withPlatform("Win32");
    assert.match(windows, /%USERPROFILE%\\\.unsloth\\llama\.cpp/);
    assert.doesNotMatch(windows, /~\//);
    for (const platform of ["Linux x86_64", "MacIntel"]) {
      const message = withPlatform(platform);
      assert.match(message, /~\/\.unsloth\/llama\.cpp/);
      assert.doesNotMatch(message, /%USERPROFILE%/);
    }
  } finally {
    Object.defineProperty(globalThis, "navigator", {
      value: original,
      configurable: true,
    });
  }
});

test("a missing navigator does not cost the folder advice", () => {
  const original = globalThis.navigator;
  try {
    Object.defineProperty(globalThis, "navigator", {
      value: undefined,
      configurable: true,
    });
    assert.match(
      preflightStaleMessage("managed_stale", "llama_runtime_binaries_missing"),
      /~\/\.unsloth\/llama\.cpp/,
    );
  } finally {
    Object.defineProperty(globalThis, "navigator", {
      value: original,
      configurable: true,
    });
  }
});

test("the llama runtime reasons match the ones the Python and Rust sides send", () => {
  // installed_runtime_health() returns the first three; managed.rs substitutes the fourth.
  // Keep both sides in sync or messages fall through to "too old".
  assert.deepEqual(LLAMA_RUNTIME_REASONS, [
    "llama_runtime_dir_missing",
    "llama_runtime_payload_incomplete",
    "llama_runtime_binaries_missing",
    "llama_runtime_incomplete",
  ]);
});

test("an unrelated stale reason is still reported as an outdated install", () => {
  for (const reason of [
    "backend_outdated",
    "studio_install_incomplete",
    "desktop_backend_ownership_unsupported",
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
  for (const reason of LLAMA_RUNTIME_REASONS) {
    assert.match(
      preflightStaleMessage("managed_stale", `${reason}:llama-server.exe`),
      RUNTIME_MISSING_FILES,
    );
    assert.match(
      preflightStaleMessage("managed_stale", `${reason}:C:\\quarantine`),
      RUNTIME_MISSING_FILES,
    );
  }
});

test("an absent or empty reason still falls through to the old behaviour", () => {
  for (const reason of [null, "", ":", ":llama_runtime_dir_missing"]) {
    assert.match(
      preflightStaleMessage("managed_stale", reason),
      MANAGED_TOO_OLD,
    );
    assert.match(preflightStaleMessage("owned_stale", reason), OWNED_TOO_OLD);
  }
});

test("the context reasons still win over the runtime reason", () => {
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

test("the busy reasons are spelled the same in Rust", async () => {
  const native = await readFile(
    new URL("../../src-tauri/src/preflight/managed.rs", import.meta.url),
    "utf8",
  );
  assert.ok(
    native.includes(`MANAGED_ENVIRONMENT_BUSY: &str = "${MANAGED_ENVIRONMENT_BUSY}"`),
  );
  assert.ok(
    native.includes(
      `MANAGED_ENVIRONMENT_UPDATING: &str = "${MANAGED_ENVIRONMENT_UPDATING}"`,
    ),
  );
});
