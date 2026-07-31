// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_TRAINING_CONFIG_BYTES,
  readBrowserTrainingConfig,
} from "../src/features/studio/wizard/training-config-file.ts";

const MAXIMUM_SIZE_ERROR = /maximum 1 MiB/;

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
    MAXIMUM_SIZE_ERROR,
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
