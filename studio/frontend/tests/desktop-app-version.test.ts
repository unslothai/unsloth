// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadDesktopAppVersion } from "../src/features/settings/desktop-app-version.ts";

test("loads the Tauri SemVer displayed by About", async () => {
  assert.equal(
    await loadDesktopAppVersion(() => Promise.resolve("1.8.3")),
    "1.8.3",
  );
});

test("reports an unavailable desktop version when the Tauri API rejects", async () => {
  const version = await loadDesktopAppVersion(() =>
    Promise.reject(new Error("Tauri app API unavailable")),
  );

  assert.equal(version, null);
});
