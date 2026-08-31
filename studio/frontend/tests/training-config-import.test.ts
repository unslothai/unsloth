// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_TRAINING_CONFIG_BYTES,
  TRAINING_CONFIG_TOO_LARGE_ERROR_KEY,
  TrainingConfigFileError,
  readBrowserTrainingConfig,
} from "../src/features/studio/wizard/training-config-file.ts";

test("browser training config imports enforce the native size limit", async () => {
  let read = false;
  const oversized = {
    size: MAX_TRAINING_CONFIG_BYTES + 1,
    text: () => {
      read = true;
      return Promise.resolve("model_name: unsloth/test");
    },
  };

  await assert.rejects(
    readBrowserTrainingConfig(oversized),
    (error: unknown) =>
      error instanceof TrainingConfigFileError &&
      error.translationKey === TRAINING_CONFIG_TOO_LARGE_ERROR_KEY,
  );
  assert.equal(read, false);

  assert.equal(
    await readBrowserTrainingConfig({
      size: MAX_TRAINING_CONFIG_BYTES,
      text: () => Promise.resolve("model_name: unsloth/test"),
    }),
    "model_name: unsloth/test",
  );
});
