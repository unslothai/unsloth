// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  checkpointQuantFormat,
  matchesFormat,
  matchesRecommended,
} from "../src/features/hub/lib/format-filters.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const gguf = { id: "unsloth/Qwen3.6-27B-GGUF", isGguf: true };
const fp8 = { id: "unsloth/Qwen3.6-27B-FP8", isGguf: false };
const fp8Dynamic = {
  id: "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
  isGguf: false,
};
const nvfp4 = { id: "unsloth/Qwen3.6-27B-NVFP4", isGguf: false };
const bf16 = { id: "unsloth/Qwen3.6-27B", isGguf: false };

test("checkpoint quant format comes from the repo name or the fp8 quant config", () => {
  assert.equal(checkpointQuantFormat(fp8), "fp8");
  assert.equal(checkpointQuantFormat(fp8Dynamic), "fp8");
  assert.equal(checkpointQuantFormat(nvfp4), "nvfp4");
  assert.equal(checkpointQuantFormat(bf16), null);
  assert.equal(checkpointQuantFormat({ ...bf16, quantMethod: "fp8" }), "fp8");
  // FP8 bases re-quantized to another scheme are not FP8 checkpoints.
  for (const id of [
    "tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ",
    "StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe",
    "someone/Llama-FP8-AWQ",
  ]) {
    assert.equal(checkpointQuantFormat({ id, isGguf: false }), null, id);
  }
  assert.equal(checkpointQuantFormat({ ...fp8, quantMethod: "gptq" }), null);
  // A GGUF repo named after an FP8 base is still a GGUF.
  assert.equal(
    checkpointQuantFormat({ id: "unsloth/X-FP8-GGUF", isGguf: true }),
    null,
  );
});

test("Recommended lists every GGUF and only the checkpoints this GPU runs natively", () => {
  assert.equal(matchesRecommended(gguf, []), true);
  assert.equal(matchesRecommended(bf16, ["fp8", "nvfp4"]), false);
  // CPU, Mac, Ampere: no FP8 / NVFP4.
  assert.equal(matchesRecommended(fp8, []), false);
  assert.equal(matchesRecommended(nvfp4, []), false);
  // Ada / Hopper: FP8 only.
  assert.equal(matchesRecommended(fp8, ["fp8"]), true);
  assert.equal(matchesRecommended(nvfp4, ["fp8"]), false);
  // Blackwell: both.
  assert.equal(matchesRecommended(nvfp4, ["fp8", "nvfp4"]), true);
});

test("the GGUF filter stays a plain format match", () => {
  assert.equal(matchesFormat("gguf", "gguf"), true);
  assert.equal(matchesFormat("safetensors", "gguf"), false);
  assert.equal(matchesFormat("gguf", "recommended"), true);
});

test("the likes threshold only curates Recommended, and Recommended is the default", async () => {
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  assert.match(
    page,
    /effectiveDiscoverFormat !== "recommended" \|\|\s*hasQuery \|\|\s*resolveOwnerProviderLogo/,
  );
  assert.doesNotMatch(page, /preset\?\.format \?\? "gguf"/);
  const channels = await readSrcAsync("features/hub/lib/channels.ts");
  assert.match(
    channels,
    /id: "unsloth-trending",[\s\S]*?format: "recommended"/,
  );
});

test("the chat picker lists FP8 / NVFP4 checkpoints the GPU runs, whatever their size", async () => {
  const { isCapableCheckpointQuant } = await import(
    "../src/features/model-picker/components/model-selector/recommended-fit.ts"
  );
  assert.equal(isCapableCheckpointQuant(fp8.id, undefined, ["fp8"]), true);
  assert.equal(isCapableCheckpointQuant(fp8.id, undefined, []), false);
  assert.equal(isCapableCheckpointQuant(nvfp4.id, undefined, ["fp8"]), false);
  assert.equal(
    isCapableCheckpointQuant("unsloth/X-FP8-GGUF", undefined, ["fp8"]),
    false,
  );

  const picker = await readSrcAsync(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  // Past the runtime gate, the GGUF-only rule, the FP8 name filter and the fit filter.
  assert.match(picker, /matchesTask: chatSupportedCheckpointQuant\(r\)/);
  // Only the quant-method check is waived: an FP8 image model is still not a chat model.
  assert.match(
    picker,
    /runsCheckpointQuant\(r\) &&\s*isChatSupported\(\{ \.\.\.r, quantMethod: undefined \}\)/,
  );
  assert.match(
    picker,
    /cachedIdFor\(r\.id\) !== null \|\|\s*runsCheckpointQuant\(r\) \|\|/,
  );
  assert.equal(
    picker.match(/runsCheckpointQuant\(\{ id \}\) \|\| !\/-FP8/g)?.length,
    2,
  );
  // Task pages (images, audio) keep their own formats.
  assert.match(
    picker,
    /\(r: \{ id: string; quantMethod\?: string \}\) =>\s*!task &&/,
  );
});
