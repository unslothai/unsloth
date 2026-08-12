// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Issue 8416: gated on `isTauri` alone, the picker stayed live when the app
// attached to a backend it had not spawned, and only a spawned backend holds
// UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET, so the lease came back 400. The
// button must be dead without the capability AND `link()` must refuse anyway.
// No React renderer here, so this asserts on source, like ~50 sibling tests.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HOOK = readFileSync(
  fileURLToPath(
    new URL("../src/features/rag/components/use-linked-folders.ts", import.meta.url),
  ),
  "utf8",
);

const MANAGER = readFileSync(
  fileURLToPath(
    new URL("../src/features/rag/components/linked-folders-manager.tsx", import.meta.url),
  ),
  "utf8",
);

const READINESS = readFileSync(
  fileURLToPath(
    new URL("../src/features/native-intents/use-native-readiness.ts", import.meta.url),
  ),
  "utf8",
);

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
  // A stale closure would restore the old behaviour once the capability flips.
  assert.ok(
    body.includes("nativePathLeasesSupported") &&
      HOOK.slice(HOOK.indexOf("const link = useCallback("))
        .includes("nativePathLeasesSupported,"),
    "nativePathLeasesSupported must be in link()'s dependency list",
  );
});

test("the unsupported branch names the managed backend, not the desktop app", () => {
  // The old copy told a desktop user to use the desktop app they were already in.
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
    /native_path_leases_supported\s*===\s*true/,
    "an absent field on an older backend must not read as supported",
  );
  assert.ok(
    READINESS.includes("useState(false)"),
    "the hook must start unsupported, so the picker is never live before it is known",
  );
});
