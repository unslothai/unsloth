// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

test("a switch validates the GPUs the load will use", () => {
  assert.doesNotMatch(runtime, /validateGpuIds/);
  const reset = runtime.indexOf("loadSelectedGpuIds = stagedGpuIds;");
  const request = runtime.indexOf(
    "const loadRequest = (): LoadModelRequest =>",
  );
  const validate = runtime.indexOf("await validateModel({");
  assert.ok(reset > 0 && reset < request && request < validate);
  assert.match(
    runtime.slice(request, validate),
    /gpu_ids: loadSelectedGpuIds \?\? undefined,/,
  );
  assert.match(runtime, /await validateModel\(\{\s*\.\.\.loadRequest\(\),/);
  assert.match(runtime, /await loadModel\(\{\s*\.\.\.loadRequest\(\),/);
});
