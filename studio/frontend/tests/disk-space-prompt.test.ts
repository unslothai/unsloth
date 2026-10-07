// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  diskCleanupAgentPrompt,
  isDiskFullInstallError,
} from "../src/components/tauri/disk-space-prompt.ts";

test("a full disk is recognized from the installer summary", () => {
  const error = "Installation failed: install unsloth failed (exit code 1): No space left on device (os error 28)";
  assert.equal(isDiskFullInstallError(error), true);
  assert.equal(
    isDiskFullInstallError("There is not enough space on the disk. (os error 112)"),
    true,
  );
  assert.equal(isDiskFullInstallError("There is not enough space on the disk. (os error 39)"), true);
  assert.equal(isDiskFullInstallError(null), false);
  assert.equal(
    isDiskFullInstallError("Installation failed: install unsloth failed (exit code 1): `sounddevice`"),
    false,
  );
  assert.equal(isDiskFullInstallError("enospc"), false);
  assert.equal(isDiskFullInstallError("Disk quota exceeded"), false);
  assert.equal(isDiskFullInstallError("os error 280"), false);
});

test("the cleanup prompt carries the error and waits for a choice", () => {
  const error = "Installation failed: install unsloth failed (exit code 1): No space left on device (os error 28)";
  const prompt = diskCleanupAgentPrompt(error);
  assert.match(prompt, /No space left on device/);
  assert.match(prompt, /Do not delete anything until I tell you which group to clear/);
  assert.match(prompt, /node_modules/);
});
