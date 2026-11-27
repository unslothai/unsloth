// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Video LoRAs bake into the denoiser at load (VideoLoadRequest.loras), so the Video page picks them on its
 * load panel, reloads to apply a change, and the Train panel deploys a video-family adapter to Video. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  bakedVideoLoraSpecs,
  MAX_VIDEO_LORAS,
  trainedLoraId,
  videoLoraCapable,
  videoLoraReloadNeeded,
  videoLoraSpecs,
} from "../src/features/video/video-lora.ts";
import type { VideoStatus } from "../src/features/video/api.ts";

import { readSrc } from "./helpers/kit.ts";

function status(over: Partial<VideoStatus>): VideoStatus {
  return {
    loaded: true,
    repo_id: "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    family: "wan2.2-ti2v-5b",
    base_repo: null,
    device: "cuda",
    dtype: "bfloat16",
    model_kind: "pipeline",
    vae_tiling: false,
    speed_optims: [],
    has_audio: false,
    supports_cfg: true,
    ...over,
  };
}

test("a plain selection sends no loras; ids are trimmed, zero weights and repeats dropped", () => {
  assert.equal(videoLoraSpecs([]), undefined);
  assert.equal(videoLoraSpecs(undefined), undefined);
  assert.equal(videoLoraSpecs([{ id: "  ", weight: 1 }, { id: "a", weight: 0 }]), undefined);
  assert.deepEqual(
    videoLoraSpecs([
      { id: " tarot ", weight: 1 },
      { id: "tarot", weight: 0.5 },
      { id: "style", weight: 0.7 },
    ]),
    [
      { id: "tarot", weight: 1 },
      { id: "style", weight: 0.7 },
    ],
  );
  const many = Array.from({ length: 12 }, (_, i) => ({ id: `l${i}`, weight: 1 }));
  assert.equal(videoLoraSpecs(many)?.length, MAX_VIDEO_LORAS);
});

test("the picker needs a reload exactly when it differs from what the resident build baked", () => {
  const baked = status({ baked_loras: ["tarot"], baked_lora_specs: [{ id: "tarot", weight: 1 }] });
  assert.deepEqual(bakedVideoLoraSpecs(baked), [{ id: "tarot", weight: 1 }]);
  assert.equal(videoLoraReloadNeeded([{ id: "tarot", weight: 1 }], baked), false);
  assert.equal(videoLoraReloadNeeded([{ id: " tarot", weight: 1 }], baked), false);
  assert.equal(videoLoraReloadNeeded([{ id: "tarot", weight: 0.8 }], baked), true);
  assert.equal(videoLoraReloadNeeded([], baked), true);
  // A zero-weight row bakes nothing, so it matches a plain build.
  const plain = status({});
  assert.equal(videoLoraReloadNeeded([{ id: "tarot", weight: 0 }], plain), false);
  assert.equal(videoLoraReloadNeeded([{ id: "tarot", weight: 1 }], plain), true);
  // Order-free: the merge is a sum.
  const two = status({
    baked_lora_specs: [
      { id: "a", weight: 1 },
      { id: "b", weight: 0.5 },
    ],
  });
  assert.equal(
    videoLoraReloadNeeded(
      [
        { id: "b", weight: 0.5 },
        { id: "a", weight: 1 },
      ],
      two,
    ),
    false,
  );
  // Nothing resident: nothing to reload.
  assert.equal(videoLoraReloadNeeded([{ id: "a", weight: 1 }], status({ loaded: false })), false);
  assert.equal(videoLoraReloadNeeded([{ id: "a", weight: 1 }], null), false);
});

test("only a full pipeline load offers LoRAs; an older backend reads as none baked", () => {
  assert.equal(videoLoraCapable(status({})), true);
  assert.equal(videoLoraCapable(status({ model_kind: "gguf" })), false);
  assert.equal(videoLoraCapable(status({ model_kind: "single_file" })), false);
  assert.equal(videoLoraCapable(status({ loaded: false })), false);
  // A two-expert family refuses LoRAs at load, so the picker stays hidden.
  assert.equal(videoLoraCapable(status({ lora_refusal: "LoRAs are not supported for the two-expert Wan2.2 A14B yet" })), false);
  assert.equal(videoLoraCapable(status({ lora_refusal: null })), true);
  assert.deepEqual(bakedVideoLoraSpecs(status({})), []);
  assert.deepEqual(bakedVideoLoraSpecs(status({ loaded: false, baked_lora_specs: [{ id: "a", weight: 1 }] })), []);
});

test("a trained adapter deploys under its catalog stem", () => {
  assert.equal(trainedLoraId("/home/u/.unsloth/studio/loras/diffusion/ltx2_tarot.safetensors"), "ltx2_tarot");
  assert.equal(trainedLoraId("C:\\studio\\loras\\diffusion\\wan_clip.SAFETENSORS"), "wan_clip");
  assert.equal(trainedLoraId("plain"), "plain");
});

test("the video load and Reapply carry the selection; GGUF and single-file loads never do", () => {
  const api = readSrc("features/video/api.ts");
  assert.match(api, /export interface VideoLoadRequest \{[\s\S]*?\n {2}loras\?: LoraSpecInput\[\];\n\}/);
  assert.match(api, /baked_loras\?: string\[\];\n {2}baked_lora_specs\?: LoraSpecInput\[\];/);

  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /type VideoLoadOptions = \{[\s\S]*?loras\?: LoraSpecInput\[\];\n\};/);
  const load = page.slice(page.indexOf("const startRequest = loadVideoModel({"));
  assert.match(
    load.slice(0, 1500),
    /\.\.\.\(opts\.kind === "pipeline" && videoLoraSpecs\(opts\.loras\) \? \{ loras: videoLoraSpecs\(opts\.loras\) \} : \{\}\),/,
  );
  // The hub pick's download plan sees the same selection, so a LoRA load never stages a hosted quantized denoiser.
  const plan = page.slice(page.indexOf("const plan = await getVideoDownloadPlan({\n          model_path: opts.displayRepoId ?? repoId,"));
  assert.match(
    plan.slice(0, 1400),
    /\.\.\.\(opts\.kind === "pipeline" && videoLoraSpecs\(opts\.loras\) \? \{ loras: videoLoraSpecs\(opts\.loras\) \} : \{\}\),\n {8}\}\);/,
  );
  const reapply = page.slice(page.indexOf("const handleReapply = useCallback("));
  assert.match(
    reapply.slice(0, 900),
    /loras: l\.kind === "pipeline" \? videoLoraSpecs\(loraSelection\) : undefined,[\s\S]*?\}, \[handleLoad, loraSelection\]\);/,
  );
});

test("the load panel shows the picker on pipeline loads, the baked set, and Reload to apply", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(
    page,
    /\{loraCapable && \(\s*<VideoLoraField\s+catalog=\{loraCapable \? loraCatalog : EMPTY_LORA_CATALOG\}[\s\S]*?baked=\{status\?\.baked_loras \?\? \[\]\}\s*reloadNeeded=\{loraReloadNeeded\}[\s\S]*?onReload=\{handleReapply\}/,
  );
  // Catalog filtered to the loaded family, as on Images.
  assert.match(page, /listDiffusionLoras\(loraFamily \?\? undefined\)/);
  const field = readSrc("features/video/video-lora-field.tsx");
  assert.match(field, /Reload to apply/);
  assert.match(field, /Baked into the loaded model: \{baked\.join\(", "\)\}/);
  assert.match(field, /<LoraAddMenu/);
});

test("video-family adapters get Deploy to Video; image families keep Deploy to Create", () => {
  const src = readSrc("features/images/train/diffusion-train-panel.tsx");
  assert.match(src, /\?\.is_video === true/);
  assert.match(src, /!isVideoFamily\(famName\) \|\| Boolean\(onDeployVideo\)/);
  assert.match(src, /isVideoFamily\(famName\) \? "Deploy to Video" : "Deploy to Create"/);
  assert.match(
    src,
    /viewRun\.catalog_path && canDeployFamily\(viewRun\.family \|\| ""\) && \(\s*<Button[\s\S]{0,400}\(isVideoFamily\(viewRun\.family \|\| ""\) \? onDeployVideo : onDeploy\)\?\.\(/,
  );
  assert.match(
    src,
    /status\?\.catalog_path && canDeployFamily\(status\.family \|\| family\?\.name \|\| ""\) && \(\s*<Button type="button" size="sm" onClick=\{onDeployClick\}>/,
  );
  assert.match(src, /\(isVideoFamily\(famName\) \? onDeployVideo : onDeploy\)\?\.\(\{/);
});

test("Deploy loads the family base on Video with the adapter at weight 1", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /onDeployVideo=\{handleDeployVideoAdapter\}/);
  const deploy = page.slice(page.indexOf("const handleDeployVideoAdapter = useCallback("));
  const body = deploy.slice(0, 2400);
  assert.match(body, /const loraId = trainedLoraId\(args\.catalogPath\);/);
  assert.match(body, /setPageMode\("create"\);/);
  assert.match(body, /const deployLoras: LoraSpecInput\[\] = \[\{ id: loraId, weight: 1 \}\];/);
  assert.match(body, /const opts: VideoLoadOptions = \{ kind: "pipeline", loras: deployLoras \};/);
  assert.match(body, /void loadOrStage\(args\.baseRepo, opts, "hub", token\)/);
  // A request handed over from Images is taken once Video is active.
  assert.match(
    page,
    /if \(!active \|\| !deployRequest\) return;\s*const request = useVideoDeployStore\.getState\(\)\.takeDeploy\(\);\s*if \(request\) handleDeployVideoAdapter\(request\);/,
  );
  const images = readSrc("features/images/images-page.tsx");
  assert.match(
    images,
    /onDeployVideo=\{\(args\) => \{\s*useVideoDeployStore\.getState\(\)\.requestDeploy\(args\);\s*void navigateSelf\(\{ to: "\/video" \}\);\s*\}\}/,
  );
  // The image Deploy path is unchanged.
  assert.match(images, /onDeploy=\{handleDeployAdapter\}/);
});
