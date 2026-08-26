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
  SOURCE.indexOf("const validation = await validateModel"),
  SOURCE.indexOf("// Upgrade consent runs before the security dialogs"),
);
const DOWNLOAD_POLL = SOURCE.slice(
  SOURCE.indexOf("const pollDownload = async () =>"),
  SOURCE.indexOf("const pollLoad = async () =>"),
);

test("MLX base substitution tracks every downloaded repository", () => {
  assert.match(
    VALIDATION,
    /progressModelIds = isLora/,
  );
  assert.match(
    VALIDATION,
    /\[modelId, validation\.mlx_loads_base_model\]/,
  );
  assert.match(VALIDATION, /\[validation\.mlx_loads_base_model\]/);
  assert.match(VALIDATION, /downloadComplete = false;/);
  assert.match(
    DOWNLOAD_POLL,
    /progressModelIdsAtRequest\.map\(\(progressModelId\) =>/,
  );
  assert.match(DOWNLOAD_POLL, /getDownloadProgress\(progressModelId\)/);
});

test("validated LoRA status controls download tracking", () => {
  assert.match(SOURCE, /let isLora =/);
  assert.match(VALIDATION, /isLora = validation\.is_lora \?\? isLora;/);
  assert.ok(
    VALIDATION.indexOf("isLora = validation.is_lora ?? isLora;") <
      VALIDATION.indexOf("if (validation.mlx_loads_base_model)"),
  );
});

test("local LoRA tracks only the substituted base download", () => {
  assert.match(VALIDATION, /progressModelIds = isLora && !isLocal/);
});

test("a pre-substitution progress response cannot complete the base download", () => {
  assert.match(
    DOWNLOAD_POLL,
    /const progressModelIdsAtRequest = \[\.\.\.progressModelIds\];/,
  );
  assert.match(
    DOWNLOAD_POLL,
    /progressModelIdsAtRequest\.length !== progressModelIds\.length/,
  );
  assert.match(
    DOWNLOAD_POLL,
    /progressModelIdsAtRequest\.some\(/,
  );
});

test("all substituted downloads must finish before model startup", () => {
  assert.match(
    DOWNLOAD_POLL,
    /const allDownloadsComplete = progressResponses\.every\(/,
  );
  assert.match(DOWNLOAD_POLL, /allDownloadsComplete &&/);
  assert.match(
    DOWNLOAD_POLL,
    /progressModelIds\.some\(/,
  );
});

test("LoRA substitution says the adapter still loads", () => {
  assert.match(VALIDATION, /Loading the adapter with/);
  assert.match(VALIDATION, /in place of its bitsandbytes base/);
});
