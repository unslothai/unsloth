// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Video page seeds steps and guidance from a repo-id substring table before backend status
// answers; the test lifts the table and matcher out of the page and runs them on real checkpoint
// names, including separator spellings the backend family matcher already folds.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const VIDEO = readText("../src/features/video/video-page.tsx");

function sliceFrom(start: string, end: string): string {
  const from = VIDEO.indexOf(start);
  assert.ok(from >= 0, `${start} must exist`);
  const to = VIDEO.indexOf(end, from);
  assert.ok(to > from, `${end} must follow ${start}`);
  return VIDEO.slice(from, to);
}

const defaultsFor = (() => {
  const source = [
    sliceFrom("const DEFAULT_GEN", "\n"),
    sliceFrom("const MODEL_DEFAULTS", "];") + "];",
    sliceFrom("function defaultsFor", "\n}\n") + "\n}",
  ]
    .join("\n")
    .replace(/: Array<\{[^}]*\}>/, "")
    .replace(/\(repoId: string, family\?: string\): \{[^}]*\}/, "(repoId, family)")
    .replace(/\((\w+): string\)/g, "($1)");
  const pick = new Function(`${source}\nreturn defaultsFor;`)() as (
    repoId: string,
    family?: string,
  ) => { steps: number; guidance: number };
  return (repoId: string, family?: string) => {
    const { steps, guidance } = pick(repoId, family);
    return { steps, guidance };
  };
})();

test("every Wan2.1 14B spelling seeds the 14B recipe, every 1.3B spelling the 1.3B one", () => {
  for (const id of [
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Kijai/WanVideo_comfy/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors",
    "wan2.1_t2v_14B_bf16.safetensors",
    "city96/Wan2.1-T2V-14B-gguf/wan2.1-t2v-14b-Q4_K_M.gguf",
    "Wan21_T2V_14B_fp8.safetensors",
  ]) {
    assert.deepEqual(defaultsFor(id), { steps: 20, guidance: 6 }, id);
  }
  for (const id of [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan2.1_t2v_1.3B_fp16.safetensors",
    "Kijai/WanVideo_comfy/Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors",
    "wan21-t2v-1.3b-Q8_0.gguf",
  ]) {
    assert.deepEqual(defaultsFor(id), { steps: 30, guidance: 6 }, id);
  }
});

test("the other families keep their recipes", () => {
  const cases: Array<[string, number, number]> = [
    ["Wan-AI/Wan2.2-T2V-A14B-Diffusers", 20, 3.5],
    ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", 20, 5],
    ["MiniMaxAI/MiniMax-H3", 20, 1],
    ["MiniMaxAI/MiniMax_H3", 20, 1],
    ["Lightricks/LTX-2.3", 40, 4],
    ["Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors", 8, 1],
    ["Lightricks/LTX-2.5-Diffusers", 8, 1],
    ["Lightricks/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", 8, 1],
    ["Lightricks/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", 30, 3],
    // Every variant name the backend's ltx25_is_dev accepts, with or without the size tag.
    ["someone/LTX-2.5-GGUF/ltx-2.5-dev.safetensors", 30, 3],
    ["/models/ltx_2.5_full_bf16.safetensors", 30, 3],
    ["someone/LTX-2.5-sft", 30, 3],
    ["someone/LTX-2.5/ltx-2.5-transformer_full.safetensors", 30, 3],
    // An opaque file name falls back to the repo: a Hub file's repo segment, a local file's own folder (the picker
    // loads a local file from its directory), as the backend gets them.
    ["/models/LTX-2.5-dev/model.safetensors", 30, 3],
    ["C:\\models\\LTX-2.5-dev\\model.safetensors", 30, 3],
    ["someone/LTX-2.5-sft/diffusion_models/model.safetensors", 30, 3],
    ["/models/LTX-2.5-sft/diffusion_models/model.safetensors", 8, 1],
    ["/home/dev/models/LTX-2.5/model.safetensors", 8, 1],
    // Any non-alphanumeric boundary, as the backend's token regex.
    ["/models/LTX-2.5/ltx-2.5-(dev).safetensors", 30, 3],
    ["/models/LTX-2.5/ltx-2.5+sft.safetensors", 30, 3],
    // Only the file and its repo count: an older LTX folder further up does not.
    ["/models/LTX-2.3-archive/LTX-2.5-dev/model.safetensors", 30, 3],
    ["/models/LTX-2.5-archive/LTX-2.3/model.safetensors", 40, 4],
    ["/models/LTX-2.5-dev/ltx-2.5-distilled.safetensors", 8, 1],
    // The file decides over the repo, and "distilled" wins inside one name.
    ["someone/LTX-2.5-dev/ltx-2.5-distilled.safetensors", 8, 1],
    ["someone/LTX-2.5/ltx-2.5-dev-distilled.safetensors", 8, 1],
    ["someone/LTX-2.5/ltx-2.5-developer.safetensors", 8, 1],
    // The compact spellings the backend's ltx_name_version accepts, and diffusers' pre-release "2.4".
    ["someone/LTX25", 8, 1],
    ["someone/ltx2.5", 8, 1],
    ["someone/LTX-Video-2.5", 8, 1],
    ["someone/LTXV-2.5-dev", 30, 3],
    ["/models/ltx25_dev.safetensors", 30, 3],
    ["someone/LTX-2.4-Diffusers", 8, 1],
    // Neither the old releases nor a future one take the LTX-2.5 recipe.
    ["someone/LTX2", 40, 4],
    ["someone/ltx-2.3-dev", 40, 4],
    ["someone/LTX-2.6", 40, 4],
    ["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", 20, 6],
    ["someone/unknown-model", 8, 1],
    ["wan21-lab/HunyuanVideo-1.5-Diffusers-480p_t2v", 20, 6],
    ["/home/wan2.1/models/HunyuanVideo-1.5-Diffusers-480p_t2v", 20, 6],
    ["wan-lab/LTX-2.3", 40, 4],
    ["QuantStack/Wan2.1_T2V_14B_FusionX-GGUF/Wan2.1_T2V_14B_FusionX-Q4_K_M.gguf", 8, 1],
    ["fusionx-lab/HunyuanVideo-1.5-Diffusers-480p_t2v", 20, 6],
    ["/models/FusionX/wan2.1-t2v-14b-Q4_K_M.gguf", 20, 6],
    ["/home/user/ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", 20, 5],
    ["/models/ltx/Cosmos3-Nano", 35, 6],
    ["/home/distilled/Cosmos3-Nano", 35, 6],
    ["/data/fusionx/Cosmos3-Edge", 20, 6],
    ["/models/a14b/Cosmos3-Edge", 20, 6],
    ["/models/hunyuanvideo/Cosmos3-Nano", 35, 6],
    ["cosmos3-nano-lab/HunyuanVideo-1.5-Diffusers-480p_t2v", 20, 6],
    ["/home/cosmos3-nano/models/HunyuanVideo-1.5-Diffusers-480p_t2v", 20, 6],
    ["/models/Cosmos3-Nano/wan2.1_t2v_14B_bf16.safetensors", 20, 6],
    ["cosmos3-edge-lab/Wan2.1-T2V-1.3B-Diffusers", 30, 6],
  ];
  for (const [id, steps, guidance] of cases) {
    assert.deepEqual(defaultsFor(id), { steps, guidance }, id);
  }
});

test("an explicit LTX-2.5 family still reads the variant from the file and repo names", () => {
  assert.deepEqual(defaultsFor("/models/opaque/transformer_full.safetensors", "ltx-2.5"), { steps: 30, guidance: 3 });
  assert.deepEqual(defaultsFor("/models/LTX-dev-run/model.safetensors", "ltx-2.5"), { steps: 30, guidance: 3 });
  assert.deepEqual(defaultsFor("/models/opaque/model.safetensors", "ltx-2.5"), { steps: 8, guidance: 1 });
  assert.deepEqual(defaultsFor("/models/opaque/model.safetensors"), { steps: 8, guidance: 1 });
});

test("every Cosmos3 spelling seeds the recipe its backend family reports: Nano 35 steps, Edge 20, CFG 6", () => {
  const families = readText("../../backend/core/inference/video_families.py");
  const backend = (constant: string) => {
    const row = families.slice(families.indexOf(`name = "${constant}",`));
    const steps = row.match(/default_steps = (\d+),/);
    const guidance = row.match(/default_guidance = ([\d.]+),/);
    assert.ok(steps && guidance, constant);
    return { steps: Number(steps[1]), guidance: Number(guidance[1]) };
  };
  const nano = backend("cosmos3-nano");
  const edge = backend("cosmos3-edge");
  // The cards' Diffusers examples (nvidia/Cosmos3-Nano@e59a53c2, nvidia/Cosmos3-Edge@344d602b).
  assert.deepEqual([nano, edge], [{ steps: 35, guidance: 6 }, { steps: 20, guidance: 6 }]);
  const spellings: Array<[string, typeof nano]> = [
    ["nvidia/Cosmos3-Nano", nano],
    ["NVIDIA/COSMOS3-NANO", nano],
    ["someone/Cosmos3_Nano", nano],
    ["Cosmos3 Nano", nano],
    ["/home/me/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/e59a53c25979a090fa8706c9acc0c254a6e89b92", nano],
    ["/data/models/Cosmos3-Nano/", nano],
    ["~/models/cosmos3_nano", nano],
    ["cosmos3-nano", nano],
    ["nvidia/Cosmos3-Edge", edge],
    ["someone/cosmos3_edge", edge],
    ["COSMOS3.EDGE", edge],
    ["/home/u/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/344d602b128d1bbdacb43b08d0a3626f46343e29", edge],
    ["C:\\models\\Cosmos3-Edge", edge],
    ["cosmos3-edge", edge],
  ];
  for (const [id, recipe] of spellings) {
    assert.deepEqual(defaultsFor(id), recipe, id);
  }
});
