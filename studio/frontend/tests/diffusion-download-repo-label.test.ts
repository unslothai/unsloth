// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A vendor id the backend downloads from its unsloth mirror must be labelled with that mirror while it downloads.

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { createPickToast } = await import("../src/lib/diffusion-pick-toast.ts");
const { stagedDownloadRepoId } = await import("../src/lib/diffusion-pipeline-load-target.ts");
const { IMAGE_CATALOG, VIDEO_CATALOG, downloadRepoIdFor } = await import(
  "../src/features/model-picker/components/model-selector/model-catalog.ts"
);
const { readSrc } = await import("./helpers/kit.ts");

const FLUX3 = "black-forest-labs/flux-3-image-klein";
const FLUX3_FP8 = "unsloth/flux-3-image-klein-FP8";

test("a catalog vendor id is labelled with the unsloth repo the backend fetches it from", () => {
  const cases: [string, typeof IMAGE_CATALOG, string][] = [
    [FLUX3, IMAGE_CATALOG, FLUX3_FP8],
    ["black-forest-labs/flux-3-image-klein-base", IMAGE_CATALOG, "unsloth/flux-3-image-klein-base-FP8"],
    ["OpenMOSS-Team/MOSS-Image-SFT", IMAGE_CATALOG, "unsloth/MOSS-Image-SFT-FP8"],
    ["OpenMOSS-Team/MOSS-Image-Base", IMAGE_CATALOG, "unsloth/MOSS-Image-Base-FP8"],
    ["Qwen/Qwen-Image-2.1-Turbo", IMAGE_CATALOG, "unsloth/Qwen-Image-2.1-Turbo-FP8"],
    ["Qwen/Qwen-Image-2.1", IMAGE_CATALOG, "unsloth/Qwen-Image-2.1"],
    ["unsloth/Qwen-Image-2.1", IMAGE_CATALOG, "unsloth/Qwen-Image-2.1"],
    ["unsloth/flux-3-image-klein-GGUF", IMAGE_CATALOG, "unsloth/flux-3-image-klein-GGUF"],
    ["MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "unsloth/MiniMax-H3"],
    ["Lightricks/LTX-2.3", VIDEO_CATALOG, "unsloth/LTX-2.3-FP8"],
    ["Lightricks/LTX-2", VIDEO_CATALOG, "unsloth/LTX-2-FP8"],
    ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", VIDEO_CATALOG, "unsloth/Wan2.2-TI2V-5B-FP8"],
    ["Wan-AI/Wan2.2-T2V-A14B-Diffusers", VIDEO_CATALOG, "unsloth/Wan2.2-T2V-A14B-FP8"],
    ["hunyuanvideo-community/HunyuanImage-2.1-Diffusers", IMAGE_CATALOG, "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"],
    ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers", VIDEO_CATALOG, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"],
    ["someone/not-in-the-catalog", IMAGE_CATALOG, "someone/not-in-the-catalog"],
  ];
  for (const [id, catalog, expected] of cases) {
    assert.equal(downloadRepoIdFor(id, catalog), expected, id);
  }
});

test("every non-unsloth catalog row either names its unsloth mirror or is a known exception", () => {
  // Vendor-fetched rows, or gated rows (fetchedFrom would make the network check judge the open mirror).
  const noMirror = new Set([
    "MiniMaxAI/MiniMax-H3-Image",
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    "QuantStack/FLUX.1-Krea-dev-GGUF",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "nvidia/Cosmos3-Nano",
    "nvidia/Cosmos3-Nano-FP8",
    "nvidia/Cosmos3-Edge",
    "nvidia/Cosmos3-Edge-FP8",
  ]);
  const gatedPlanLabelled = new Set([
    "ideogram-ai/ideogram-4-fp8",
    "ideogram-ai/ideogram-4-nf4-diffusers",
    "Lightricks/LTX-2.5-Diffusers",
    "Lightricks/LTX-2.5",
  ]);
  const unexplained: string[] = [];
  for (const group of [...IMAGE_CATALOG, ...VIDEO_CATALOG]) {
    for (const artifact of group.artifacts) {
      if (artifact.repoId.startsWith("unsloth/")) continue;
      if (artifact.fetchedFrom) {
        assert.match(artifact.fetchedFrom, /^unsloth\//, artifact.repoId);
        continue;
      }
      if (noMirror.has(artifact.repoId)) continue;
      if (gatedPlanLabelled.has(artifact.repoId) && artifact.gated) continue;
      unexplained.push(artifact.repoId);
    }
  }
  assert.deepEqual(unexplained, []);
});

test("a staged pick is labelled with the repo its plan downloads", () => {
  const entry = (repoId: string, bytes: number, checkpoint?: boolean) => ({ repoId, bytes, checkpoint });
  assert.equal(stagedDownloadRepoId([entry(FLUX3_FP8, 18e9, true)], FLUX3), FLUX3_FP8);
  assert.equal(
    stagedDownloadRepoId(
      [entry(FLUX3_FP8, 9e9, false), entry("unsloth/flux-3-image-klein-GGUF", 3e9, true)],
      "unsloth/flux-3-image-klein-GGUF",
    ),
    "unsloth/flux-3-image-klein-GGUF",
  );
  assert.equal(
    stagedDownloadRepoId(
      [entry("unsloth/Qwen3-4B-GGUF", 2e9), entry("unsloth/FLUX.2-dev", 3e8)],
      "unsloth/FLUX.2-klein-4B-GGUF",
    ),
    "unsloth/Qwen3-4B-GGUF",
  );
  assert.equal(stagedDownloadRepoId([], FLUX3), FLUX3);
});

test("a plan's repo renames the pick toast, and later phases keep it", () => {
  calls.length = 0;
  const pick = createPickToast({
    describe: (phase, _progress, modelName) => `${phase}:${modelName}`,
  });
  const id = pick.show(FLUX3);
  pick.setPhase(id, "downloading", 1, FLUX3_FP8);
  pick.progress({ downloadedBytes: 1, totalBytes: 2, plan: 1 });
  pick.setPhase(id, "waiting");
  const described = (calls as { options?: { description?: unknown } }[]).map((c) => c.options?.description);
  assert.deepEqual(described, [`preparing:${FLUX3}`, `downloading:${FLUX3_FP8}`, `downloading:${FLUX3_FP8}`, `waiting:${FLUX3_FP8}`]);
});

test("both pages name the pick toast for the downloaded repo, not the picked id", () => {
  for (const [page, catalog] of [
    ["features/images/images-page.tsx", "IMAGE_CATALOG"],
    ["features/video/video-page.tsx", "VIDEO_CATALOG"],
  ]) {
    const src = readSrc(page);
    assert.match(src, new RegExp(`pickToast\\.show\\(downloadRepoIdFor\\(opts\\.displayRepoId \\?\\? repoId, ${catalog}\\)\\)`), page);
    assert.doesNotMatch(src, /pickToast\.show\(opts\.displayRepoId \?\? repoId\)/, page);
    assert.match(src, /stagedDownloadRepoId\(entries, opts\.displayRepoId \?\? repoId\)/, page);
  }
});
