// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Maps a repo to its Unsloth guide. Pure and offline: the table is the whole source.

export interface ModelGuide {
  /** Shown on the link. */
  title: string;
  url: string;
}

const DOCS = "https://unsloth.ai/docs";

// Matched in order, so specific families precede general ones: qwen3.8-next before qwen3.8.
// Every URL was checked against the live docs; an unlisted family gets no link.
const GUIDES: ReadonlyArray<{
  match: RegExp;
  title: string;
  path: string;
}> = [
  // A distilled model carries its base family's name too, so the named family it actually is
  // must match first: DeepSeek-R1-0528-Qwen3-8B is an R1 release, not a Qwen3 one.
  // DeepSeek
  { match: /deepseek[-_]?v4/, title: "DeepSeek-V4", path: "models/deepseek-v4" },
  {
    match: /deepseek[-_]?ocr[-_]?2/,
    title: "DeepSeek-OCR 2",
    path: "models/tutorials/deepseek-ocr-2",
  },
  {
    match: /deepseek[-_]?ocr/,
    title: "DeepSeek-OCR",
    path: "models/tutorials/deepseek-ocr-how-to-run-and-fine-tune",
  },
  {
    match: /deepseek[-_]?v3\.1/,
    title: "DeepSeek-V3.1",
    path: "models/tutorials/deepseek-v3.1-how-to-run-locally",
  },
  {
    match: /deepseek[-_]?r1/,
    title: "DeepSeek-R1",
    path: "models/tutorials/deepseek-r1-how-to-run-locally",
  },
  // Qwen
  {
    match: /qwen-?image/,
    title: "Qwen-Image",
    path: "models/tutorials/qwen-image-2512",
  },
  {
    match: /qwen3\.8[-_]?next|qwen3\.8[-_]?flash[-_]?next/,
    title: "Qwen3.8-Flash-Next",
    path: "models/qwen3.8-next",
  },
  { match: /qwen3\.8/, title: "Qwen3.8", path: "models/qwen3.8" },
  { match: /qwen3\.6/, title: "Qwen3.6", path: "models/qwen3.6" },
  { match: /qwen3\.5/, title: "Qwen3.5", path: "models/qwen3.5" },
  {
    match: /qwen3[-_]?coder[-_]?next/,
    title: "Qwen3-Coder-Next",
    path: "models/qwen3-coder-next",
  },
  {
    match: /qwen3[-_]?coder/,
    title: "Qwen3-Coder",
    path: "models/tutorials/qwen3-coder-how-to-run-locally",
  },
  {
    match: /qwen3[-_]?next/,
    title: "Qwen3-Next",
    path: "models/tutorials/qwen3-next",
  },
  {
    match: /qwen3[-_]?vl/,
    title: "Qwen3-VL",
    path: "models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune",
  },
  {
    match: /qwen3/,
    title: "Qwen3",
    path: "models/tutorials/qwen3-how-to-run-and-fine-tune",
  },
  {
    match: /qwq/,
    title: "QwQ-32B",
    path: "models/tutorials/qwq-32b-how-to-run-effectively",
  },
  // gpt-oss
  {
    match: /gpt[-_]?oss/,
    title: "gpt-oss",
    path: "models/gpt-oss-how-to-run-and-fine-tune",
  },
  // Gemma
  {
    match: /diffusiongemma/,
    title: "DiffusionGemma",
    path: "models/diffusiongemma",
  },
  {
    match: /functiongemma/,
    title: "FunctionGemma",
    path: "models/tutorials/functiongemma",
  },
  { match: /gemma[-_]?4/, title: "Gemma 4", path: "models/gemma-4" },
  {
    match: /gemma[-_]?3n/,
    title: "Gemma 3n",
    path: "models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune",
  },
  {
    match: /gemma[-_]?3/,
    title: "Gemma 3",
    path: "models/tutorials/gemma-3-how-to-run-and-fine-tune",
  },
  // Llama
  {
    match: /llama[-_]?4/,
    title: "Llama 4",
    path: "models/tutorials/llama-4-how-to-run-and-fine-tune",
  },
  // GLM
  {
    match: /glm[-_]?5\.3[-_]?flash/,
    title: "GLM-5.3-Flash",
    path: "models/glm-5.3-flash",
  },
  { match: /glm[-_]?5\.3/, title: "GLM-5.3", path: "models/glm-5.3" },
  { match: /glm[-_]?5\.2/, title: "GLM-5.2", path: "models/glm-5.2" },
  { match: /glm[-_]?5\.1/, title: "GLM-5.1", path: "models/glm-5.1" },
  { match: /glm[-_]?5/, title: "GLM-5", path: "models/tutorials/glm-5" },
  {
    match: /glm[-_]?4\.7[-_]?flash/,
    title: "GLM-4.7-Flash",
    path: "models/tutorials/glm-4.7-flash",
  },
  { match: /glm[-_]?4\.7/, title: "GLM-4.7", path: "models/tutorials/glm-4.7" },
  {
    match: /glm[-_]?4\.6/,
    title: "GLM-4.6",
    path: "models/tutorials/glm-4.6-how-to-run-locally",
  },
  // Kimi
  { match: /kimi[-_]?k3/, title: "Kimi K3", path: "models/kimi-k3" },
  {
    match: /kimi[-_]?k2\.7/,
    title: "Kimi K2.7 Code",
    path: "models/kimi-k2.7-code",
  },
  { match: /kimi[-_]?k2\.6/, title: "Kimi K2.6", path: "models/kimi-k2.6" },
  {
    match: /kimi[-_]?k2\.5/,
    title: "Kimi K2.5",
    path: "models/tutorials/kimi-k2.5",
  },
  {
    match: /kimi[-_]?k2/,
    title: "Kimi K2 Thinking",
    path: "models/tutorials/kimi-k2-thinking-how-to-run-locally",
  },
  // MiniMax
  { match: /minimax[-_]?m3/, title: "MiniMax M3", path: "models/minimax-m3" },
  {
    match: /minimax[-_]?m2\.7/,
    title: "MiniMax-M2.7",
    path: "models/tutorials/minimax-m27",
  },
  {
    match: /minimax[-_]?m2\.5/,
    title: "MiniMax-M2.5",
    path: "models/tutorials/minimax-m25",
  },
  // Nemotron
  {
    match: /nemotron[-_]?3\.5/,
    title: "Nemotron 3.5 Lightning",
    path: "models/nemotron-3.5",
  },
  {
    match: /nemotron[-_]?3[-_]?ultra/,
    title: "Nemotron 3 Ultra",
    path: "models/nemotron-3-ultra",
  },
  {
    match: /nemotron[-_]?3[-_]?nano[-_]?omni/,
    title: "Nemotron 3 Nano Omni",
    path: "models/nemotron-3-nano-omni",
  },
  {
    match: /nemotron[-_]?3[-_]?super/,
    title: "Nemotron-3-Super",
    path: "models/nemotron-3/nemotron-3-super",
  },
  { match: /nemotron[-_]?3/, title: "Nemotron 3 Nano", path: "models/nemotron-3" },
  // Mistral
  { match: /mistral[-_]?3\.5/, title: "Mistral 3.5", path: "models/mistral-3.5" },
  {
    match: /devstral[-_]?2/,
    title: "Devstral 2",
    path: "models/tutorials/devstral-2",
  },
  {
    match: /devstral/,
    title: "Devstral",
    path: "models/tutorials/devstral-how-to-run-and-fine-tune",
  },
  {
    match: /ministral[-_]?3/,
    title: "Ministral 3",
    path: "models/tutorials/ministral-3",
  },
  {
    match: /magistral/,
    title: "Magistral",
    path: "models/tutorials/magistral-how-to-run-and-fine-tune",
  },
  // Others
  {
    match: /phi[-_]?4.*reason|phi[-_]?4[-_]?mini[-_]?reason/,
    title: "Phi-4 Reasoning",
    path: "models/tutorials/phi-4-reasoning-how-to-run-and-fine-tune",
  },
  {
    match: /granite[-_]?4\.1/,
    title: "IBM Granite 4.1",
    path: "models/ibm-granite-4.1",
  },
  {
    match: /granite[-_]?4/,
    title: "IBM Granite 4.0",
    path: "models/tutorials/ibm-granite-4.0",
  },
  { match: /muse[-_]?glimmer/, title: "Muse Glimmer", path: "models/muse-glimmer" },
  { match: /inkling/, title: "Inkling", path: "models/inkling" },
  { match: /grok[-_]?2/, title: "Grok 2", path: "models/tutorials/grok-2" },
  { match: /lfm2\.5/, title: "Liquid LFM2.5", path: "models/tutorials/lfm2.5" },
  {
    match: /cogito[-_]?v2/,
    title: "Cogito v2.1",
    path: "models/tutorials/cogito-v2-how-to-run-locally",
  },
];

// A quant suffix is a bit width, and a bit width is a digit followed by "bit" — which is
// indistinguishable from a family version to a pattern like /llama[-_]?4/ that has no trailing
// boundary. `nous-hermes-llama-4bit-v2` was handed the Llama 4 guide on the strength of its
// quantisation. Fixing it here rather than in each of the 58 entries keeps the table readable
// and cannot be forgotten by the next entry added. The canonical `unsloth/*-bnb-4bit` names
// survive it, because their family digit comes earlier: `llama-3-8b-bnb-4bit` still matches
// Llama 3 once `4bit` is blanked.
// `\b` cannot see the delimiter here: in `llama_4bit`, `_` and `4` are both word characters,
// so there is no boundary to match and the suffix survived, leaving `llama_4` to claim the
// Llama 4 guide. The delimiter is stated explicitly instead, and the trailing side likewise,
// so `4bit` is only stripped when it stands alone rather than inside a longer token.
const QUANT_SUFFIX = /(?<![a-z0-9])\d+[-_]?bit(?![a-z0-9])/g;

/** The Unsloth guide for `repoId`, or null when no family matches. */
export function modelGuide(repoId: string | null | undefined): ModelGuide | null {
  if (!repoId) return null;
  // Match the model name, not its owner, and not how it was quantised.
  const id = repoId
    .slice(repoId.lastIndexOf("/") + 1)
    .toLowerCase()
    .replace(QUANT_SUFFIX, " ");
  for (const guide of GUIDES) {
    if (guide.match.test(id)) {
      return { title: guide.title, url: `${DOCS}/${guide.path}` };
    }
  }
  return null;
}
