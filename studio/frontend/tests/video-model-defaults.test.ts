// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Video page seeds steps and guidance from a repo-id substring table before the backend status
// answers. The table and its matcher live inside the page component, so this lifts both out of the
// source and runs them on real checkpoint names, including separator spellings the backend's family
// matcher already folds together.

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
    .replace(/\(repoId: string\): \{[^}]*\}/, "(repoId)")
    .replace(/\((\w+): string\)/g, "($1)");
  const pick = new Function(`${source}\nreturn defaultsFor;`)() as (
    repoId: string,
  ) => { steps: number; guidance: number };
  return (repoId: string) => {
    const { steps, guidance } = pick(repoId);
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
    ["MiniMaxAI/MiniMax-H3", 30, 1],
    ["MiniMaxAI/MiniMax_H3", 30, 1],
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
  ];
  for (const [id, steps, guidance] of cases) {
    assert.deepEqual(defaultsFor(id), { steps, guidance }, id);
  }
});
