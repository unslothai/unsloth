// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { presentedProgress } from "../src/features/hub/download-manager/download-presentation.ts";
import { pendingDrafterPresentation } from "../src/features/model-picker/components/model-selector/variant-download-presentation.ts";

test("a pending MTP file becomes the download manager presentation", () => {
  assert.deepEqual(
    pendingDrafterPresentation({
      filename: "Qwen3.8-Flash-Next-Q4.gguf",
      quant: "Q4",
      size_bytes: 100,
      pending_drafter_filename:
        "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
      pending_drafter_size_bytes: 20,
    }),
    {
      label: "MTP companion",
      filename: "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
      expectedBytes: 20,
    },
  );
});

test("companion presentation subtracts the cached main model from progress", () => {
  const mainBytes = 112_238_658_784;
  const mtpBytes = 2_786_568_256;
  const mtpTransferred = 540_000_000;

  assert.deepEqual(
    presentedProgress({
      expectedBytes: mainBytes + mtpBytes,
      downloadedBytes: mainBytes + mtpTransferred,
      fraction: (mainBytes + mtpTransferred) / (mainBytes + mtpBytes),
      presentation: {
        label: "MTP companion",
        filename: "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        expectedBytes: mtpBytes,
      },
    }),
    {
      expectedBytes: mtpBytes,
      downloadedBytes: mtpTransferred,
      fraction: mtpTransferred / mtpBytes,
    },
  );
});

test("ordinary downloads retain their plan-wide counters", () => {
  assert.deepEqual(
    presentedProgress({
      expectedBytes: 100,
      downloadedBytes: 25,
      fraction: 0.25,
    }),
    { expectedBytes: 100, downloadedBytes: 25, fraction: 0.25 },
  );
});
