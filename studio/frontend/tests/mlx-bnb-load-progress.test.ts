// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const SOURCE = readFileSync(
  new URL(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
    import.meta.url,
  ),
  "utf8",
);

const VALIDATION = SOURCE.slice(
  SOURCE.indexOf("if (validation.mlx_loads_base_model)"),
  SOURCE.indexOf("// Upgrade consent runs before the security dialogs"),
);
const DOWNLOAD_POLL = SOURCE.slice(
  SOURCE.indexOf("const pollDownload = async () =>"),
  SOURCE.indexOf("const pollLoad = async () =>"),
);

test("MLX base substitution resets progress to the downloaded repository", () => {
  assert.match(
    VALIDATION,
    /progressModelId = validation\.mlx_loads_base_model;/,
  );
  assert.match(VALIDATION, /downloadComplete = false;/);
  assert.match(DOWNLOAD_POLL, /getDownloadProgress\(progressModelId\)/);
  assert.doesNotMatch(DOWNLOAD_POLL, /getDownloadProgress\(modelId\)/);
});

test("a cached substituted base advances directly to model startup", () => {
  assert.match(
    DOWNLOAD_POLL,
    /hasShownProgress \|\| progressModelId !== modelId/,
  );
});

test("LoRA substitution says the adapter still loads", () => {
  assert.match(VALIDATION, /Loading the adapter with/);
  assert.match(VALIDATION, /in place of its bitsandbytes base/);
});
