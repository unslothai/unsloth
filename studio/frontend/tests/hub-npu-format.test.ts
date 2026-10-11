// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { matchesFormat } from "../src/features/hub/lib/format-filters.ts";

test("the NPU format holds no Hub repo or scanned folder: those models come from Lemonade", () => {
  for (const format of [
    true,
    false,
    "gguf",
    "safetensors",
    "checkpoint",
    "mlx",
    "adapter",
    "unknown",
    null,
  ] as const) {
    assert.equal(matchesFormat(format, "npu"), false, String(format));
  }
});
