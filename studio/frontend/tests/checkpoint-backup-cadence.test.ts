import assert from "node:assert/strict";
import test from "node:test";

import {
  backupIntervalError,
  effectiveBackupSteps,
} from "../src/features/studio/sections/checkpoint-backup-cadence.ts";

test("checkpoint multipliers derive their effective step interval", () => {
  assert.equal(effectiveBackupSteps(200, 1), 200);
  assert.equal(effectiveBackupSteps(200, 2), 400);
  assert.equal(effectiveBackupSteps(200, 3), 600);
  assert.equal(effectiveBackupSteps(300, 2), 600);
});

test("custom checkpoint counts must be bounded positive integers", () => {
  assert.equal(backupIntervalError(1), null);
  assert.equal(backupIntervalError(4), null);
  assert.equal(backupIntervalError(0), "Enter at least 1 checkpoint.");
  assert.equal(backupIntervalError(1.5), "Enter at least 1 checkpoint.");
  assert.equal(backupIntervalError(1001), "Enter at least 1 checkpoint.");
});
