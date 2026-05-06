// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingMethod } from "@/types/training";

export type ExportMethod = "merged" | "lora" | "gguf";

export const EXPORT_METHODS: {
  value: ExportMethod;
  title: string;
  description: string;
  tooltip: string;
  badge?: string;
}[] = [
  {
    value: "merged",
    title: "Merged Model",
    description: "Full 16-bit model ready for inference.",
    tooltip:
      "Merges adapter weights into the base model. Best for direct deployment with vLLM or TGI.",
  },
  {
    value: "lora",
    title: "LoRA Only",
    description: "Lightweight adapter files (~100 MB). Needs base model.",
    tooltip:
      "Exports only the trained adapter. Pair with the base model at inference time to save storage.",
  },
  {
    value: "gguf",
    title: "GGUF / Llama.cpp",
    description: "Quantized formats for local AI runners.",
    tooltip:
      "Converts to GGUF for llama.cpp, Ollama, and other local runners. Pick a quantization level below.",
  },
];

export const QUANT_OPTIONS = [
  { value: "q2_k_l", label: "Q2_K_L", size: "~2.9 GB" },
  { value: "q3_k_m", label: "Q3_K_M", size: "~3.5 GB" },
  { value: "q4_k_m", label: "Q4_K_M", size: "~4.8 GB", recommended: true },
  { value: "q5_k_m", label: "Q5_K_M", size: "~5.6 GB" },
  { value: "q6_k", label: "Q6_K", size: "~6.6 GB" },
  { value: "q8_0", label: "Q8_0", size: "~8.2 GB" },
  { value: "bf16", label: "BF16", size: "~14.2 GB" },
  { value: "f16", label: "F16", size: "~14.2 GB" },
];

export function getEstimatedSize(
  method: ExportMethod | null,
  quantLevels: string[],
) {
  const sizeOf = (v: string) =>
    QUANT_OPTIONS.find((q) => q.value === v)?.size ?? "—";
  if (method === "gguf" && quantLevels.length > 0) {
    if (quantLevels.length === 1) {
      return sizeOf(quantLevels[0]);
    }
    const total = quantLevels
      .map((q) => Number.parseFloat(sizeOf(q).replace(/[^0-9.]/g, "")))
      .reduce((a, b) => a + b, 0);
    return `~${total.toFixed(1)} GB (${quantLevels.length} files)`;
  }
  if (method === "merged") {
    return "~14.2 GB";
  }
  if (method === "lora") {
    return "~100 MB";
  }
  return "—";
}

export const METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full Fine-tune",
  cpt: "Continued Pretraining",
};

export const GUIDE_STEPS = [
  "Select a training checkpoint to export from",
  "Choose an export method based on your use case",
  "Pick quantization levels if using GGUF",
  "Click Export and choose your destination",
  "Test your model and compare outputs in Chat",
];
