// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a spawned backend holds UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET, so `isTauri` alone
// is not enough. No React renderer here, so this asserts on source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

const HOOK = readSrc("features/rag/components/use-linked-folders.ts");

const MANAGER = readSrc("features/rag/components/linked-folders-manager.tsx");

const COMMANDS = readText("../../src-tauri/src/commands.rs");

const READINESS = readSrc("features/native-intents/use-native-readiness.ts");

const PREFLIGHT = readText("../../src-tauri/src/preflight.rs");

test("the picker is gated on the backend capability, not on isTauri alone", () => {
  assert.match(
    HOOK,
    /desktopSupported:\s*isTauri\s*&&\s*nativePathLeasesSupported/,
    "desktopSupported must require the lease capability as well as Tauri",
  );
  assert.ok(
    HOOK.includes("useNativePathLeasesSupported()"),
    "the capability must come from the shared readiness hook, not a local fetch",
  );
});

test("link() refuses without the capability, not just the button", () => {
  const body = HOOK.slice(
    HOOK.indexOf("const link = useCallback("),
    HOOK.indexOf("const run = useCallback("),
  );
  assert.ok(body.length > 0, "link() should still exist");
  assert.match(
    body,
    /if\s*\([^)]*!nativePathLeasesSupported[^)]*\)\s*return/,
    "link() must bail before pickNativeDocumentFolder when leases are unsupported",
  );
  assert.ok(
    body.includes("nativePathLeasesSupported") &&
      HOOK.slice(HOOK.indexOf("const link = useCallback("))
        .includes("nativePathLeasesSupported,"),
    "nativePathLeasesSupported must be in link()'s dependency list",
  );
});

test("the unsupported branch names the managed backend, not the desktop app", () => {
  assert.ok(
    !MANAGER.includes("link new folders in the desktop app"),
    "the unsupported copy must not tell a desktop user to use the desktop app",
  );
  assert.ok(
    MANAGER.includes("managed desktop backend"),
    "the unsupported copy and tooltip should name the managed backend",
  );
});

test("the capability is read from /api/health and only latches on true", () => {
  assert.match(
    READINESS,
    /native_path_leases_supported\s*!==\s*true/,
    "an absent field on an older backend must not read as supported",
  );
  assert.ok(
    READINESS.includes("useState(false)"),
    "the hook must start unsupported, so the picker is never live before it is known",
  );
});

test("the health bit alone does not enable the picker inside the app", () => {
  // An adopted survivor holds its own lease key, so it answers true while every grant fails.
  assert.ok(
    READINESS.includes("native_path_leases_usable"),
    "the hook must also ask the app whether the live backend is one it spawned",
  );
  const gate = READINESS.slice(READINESS.indexOf("native_path_leases_usable"));
  assert.match(
    gate,
    /if\s*\(usable\)\s*setSupported\(true\)/,
    "setSupported must be reached only when the app confirms the backend is ours",
  );
});

test("an adopted backend keeps running and only loses lease-backed actions", () => {
  // owned_stale routes through startRepair(), a network update, so a key mismatch must not force one.
  assert.ok(
    !PREFLIGHT.includes("native_path_lease_secret_not_persisted") &&
      !PREFLIGHT.includes("native_path_lease_secret_not_shared"),
    "an adopted survivor must not be forced stale over the lease key",
  );
  assert.match(
    COMMANDS,
    /fn native_path_leases_usable[\s\S]*?Some\(snapshot\) if !snapshot\.is_adopted/,
    "usable must require a spawned, non-adopted backend, not merely the absence " +
      "of an adopted one: attached_ready installs no snapshot at all",
  );
});
