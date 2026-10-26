// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  checkpointQuantFormat,
  hasRecommendedCheckpointLikes,
  matchesFormat,
  matchesRecommended,
  matchesRecommendedDiffusion,
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
  // A mixed NVFP4 / FP8 checkpoint needs NVFP4 hardware.
  const mixed = {
    id: "primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8",
    isGguf: false,
  };
  assert.equal(checkpointQuantFormat(mixed), "nvfp4");
  assert.equal(matchesRecommended(mixed, ["fp8"]), false);
  assert.equal(matchesRecommended(mixed, ["fp8", "nvfp4"]), true);
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

test("Recommended browsing lists checkpoints only past 30 likes", async () => {
  assert.equal(hasRecommendedCheckpointLikes({ ...gguf, likes: 0 }), true);
  assert.equal(hasRecommendedCheckpointLikes(fp8), false);
  assert.equal(hasRecommendedCheckpointLikes({ ...fp8, likes: 30 }), false);
  assert.equal(hasRecommendedCheckpointLikes({ ...fp8, likes: 31 }), true);
  // Safetensors repo cross-tagged "gguf" for a stray .gguf file (Nemotron-3.5-Bolt-74B-A9B).
  const crossTagged = {
    id: "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B",
    isGguf: true,
    tags: ["transformers", "safetensors", "gguf"],
  };
  assert.equal(
    hasRecommendedCheckpointLikes({ ...crossTagged, likes: 2 }),
    false,
  );
  assert.equal(
    hasRecommendedCheckpointLikes({ ...crossTagged, likes: 31 }),
    true,
  );
  assert.equal(
    hasRecommendedCheckpointLikes({ ...gguf, tags: ["safetensors", "gguf"] }),
    true,
  );
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  assert.match(
    page,
    /\(!browsing \|\|[\s\S]{0,240}hasRecommendedCheckpointLikes\(result\)\)\) &&\s*matchesRecommended\(result, quantFormats\)/,
  );
  assert.match(
    page,
    /matchesFormat\(row\.result\.isGguf, "gguf"\)\)[\s\S]{0,200}\.filter\(\s*\(row\) => isDiffusionRow\(row\) \|\| hasRecommendedCheckpointLikes\(row\.result\),?\s*\)/,
  );
});

test("image and video models skip the likes rules and list full-precision checkpoints", async () => {
  // FLUX.3 klein: a 23-like GGUF with no provider logo, and its BF16 diffusers repo.
  const fluxBf16 = { id: "unsloth/flux-3-image-klein", isGguf: false };
  assert.equal(matchesRecommendedDiffusion(fluxBf16, []), true);
  assert.equal(matchesRecommended(fluxBf16, []), false);
  assert.equal(
    matchesRecommendedDiffusion(
      { id: "unsloth/flux-3-image-klein-GGUF", isGguf: true },
      [],
    ),
    true,
  );
  // Quantized checkpoints still need a GPU that runs them natively.
  const fluxFp8 = { id: "unsloth/flux-3-image-klein-FP8", isGguf: false };
  assert.equal(matchesRecommendedDiffusion(fluxFp8, []), false);
  assert.equal(matchesRecommendedDiffusion(fluxFp8, ["fp8"]), true);
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  assert.match(
    page,
    /isDiffusionRow\(row\) \? diffusionQuantFormats : checkpointQuantFormats/,
  );
  assert.match(
    page,
    /searchBrowsing,\s*isDiffusionResult\(result\),\s*quantFormats,/,
  );
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
    /if \(format !== "recommended"\) \{\s*return matchesFormat\(detectResultFormat\(result\), format\);\s*\}\s*if \(diffusion\) return matchesRecommendedDiffusion\(result, quantFormats\);\s*return \(\s*\(!browsing \|\|\s*\(\(resolveOwnerProviderLogo/,
  );
  // The search's skip test follows the formats' contents: a system poll's new array must not restart it.
  assert.match(
    page,
    /const quantFormatsKey = checkpointQuantFormats\.join\(","\);[\s\S]{0,600}\[effectiveDiscoverFormat, searchBrowsing, quantFormatsKey\]/,
  );
  assert.match(
    page,
    /matchesDiscoverFormat\(\s*row\.result,\s*effectiveDiscoverFormat,\s*!hasQuery,\s*isDiffusionRow\(row\),/,
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
  // The listing's GGUF verdict wins over an id without a -GGUF suffix.
  assert.equal(
    isCapableCheckpointQuant("someone/Model-FP8", undefined, ["fp8"], true),
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
    /runsCheckpointQuant\(r\) &&\s*r\.quantMethod\?\.toLowerCase\(\) === "compressed-tensors" &&\s*isChatSupported\(\{ \.\.\.r, quantMethod: undefined \}\)/,
  );
  assert.match(
    picker,
    /cachedIdFor\(r\.id\) !== null \|\|\s*runsCheckpointQuant\(r\) \|\|/,
  );
  assert.equal(
    picker.match(/runsCheckpointQuant\(\{ id \}\) \|\| !\/-FP8/g)?.length,
    2,
  );
  // The curated-search fit gate waives size for them too.
  assert.match(
    picker,
    /cachedIdFor\(id\) !== null \|\|\s*runsCheckpointQuant\(\{ id \}\) \|\|[\s\S]{0,120}?searchRowFits\(\{\s*id\b/,
  );
  // Task pages (images, audio) keep their own formats.
  assert.match(
    picker,
    /\(r: \{ id: string; quantMethod\?: string; isGguf\?: boolean \}\) =>\s*!task &&/,
  );
});

test("a capable Recommended channel query drops the GGUF-only tag", async () => {
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  assert.match(
    page,
    /effectiveDiscoverFormat === "recommended" &&\s*checkpointQuantFormats\.length > 0/,
  );
  assert.match(
    page,
    /tags: widenRecommendedChannel\s*\?\s*withoutGgufTag\(liveListChannel\.tags\)\s*:\s*liveListChannel\.tags/,
  );
});
