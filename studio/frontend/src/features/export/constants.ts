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

export const QUANT_OPTIONS: {
  value: string;
  label: string;
  recommended?: boolean;
}[] = [
  { value: "q2_k_l", label: "Q2_K_L" },
  { value: "q3_k_m", label: "Q3_K_M" },
  { value: "q4_k_m", label: "Q4_K_M", recommended: true },
  { value: "q5_k_m", label: "Q5_K_M" },
  { value: "q6_k", label: "Q6_K" },
  { value: "q8_0", label: "Q8_0" },
  { value: "bf16", label: "BF16" },
  { value: "f16", label: "F16" },
];

/**
 * Canonical llama.cpp effective bits-per-weight per quant type. GGUF sizes
 * scale from the model's real fp16/bf16 size: `bytes ~= fp16_bytes * bpw / 16`.
 * F16/BF16 = 16 (no quantization). The K-quant values are published average
 * effective bit-rates across the mixed-precision tensors; `Q2_K_L` is an
 * Unsloth preset (Q2_K with Q8_0 embeddings/output). These are intentionally
 * approximate ("~") estimates, not exact file sizes.
 */
export const GGUF_BPW: Record<string, number> = {
  q2_k_l: 3.35,
  q3_k_m: 3.91,
  q4_k_m: 4.83,
  q5_k_m: 5.67,
  q6_k: 6.56,
  q8_0: 8.5,
  bf16: 16,
  f16: 16,
};

/** fp16/bf16 reference bit-rate (2 bytes per weight). */
const FP16_BPW = 16;

/**
 * Human-readable size in base-1024 units ("67 GB", "2.9 GB"), matching the
 * model-selector picker so an export estimate reads the same as the downloaded
 * model size. Do NOT swap in the base-1000 `hub/lib/format.ts::formatBytes`
 * here -- it would render the same model as "72 GB" and disagree with the picker.
 */
export function formatModelSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** i;
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

/** Estimated on-disk bytes for one GGUF quant, scaled from the real fp16 size. */
export function estimateQuantBytes(
  fp16Bytes: number | null | undefined,
  quant: string,
): number | null {
  if (!fp16Bytes || fp16Bytes <= 0) {
    return null;
  }
  const bpw = GGUF_BPW[quant];
  if (bpw == null) {
    return null;
  }
  return fp16Bytes * (bpw / FP16_BPW);
}

/** "~X GB" label for a quant, or "" when the real model size is unknown. */
export function formatQuantSize(
  fp16Bytes: number | null | undefined,
  quant: string,
): string {
  const bytes = estimateQuantBytes(fp16Bytes, quant);
  return bytes == null ? "" : `~${formatModelSize(bytes)}`;
}

/** value -> "~X GB" for every quant option (blank when size unknown). */
export function buildQuantSizeLabels(
  fp16Bytes: number | null | undefined,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const q of QUANT_OPTIONS) {
    out[q.value] = formatQuantSize(fp16Bytes, q.value);
  }
  return out;
}

/**
 * Estimated total export size for the summary line. Scales from the selected
 * model's real fp16 size (`fp16Bytes`); returns "" when unknown so the UI can
 * hide the estimate rather than show a misleading fixed number.
 */
export function getEstimatedSize(
  method: ExportMethod | null,
  quantLevels: string[],
  fp16Bytes: number | null | undefined,
): string {
  if (method === "gguf" && quantLevels.length > 0) {
    const perQuant = quantLevels.map((q) => estimateQuantBytes(fp16Bytes, q));
    if (perQuant.some((b) => b == null)) {
      return ""; // unknown size -> blank, never a wrong fixed number
    }
    let total = 0;
    for (const b of perQuant) {
      total += b ?? 0;
    }
    const label = `~${formatModelSize(total)}`;
    return quantLevels.length === 1
      ? label
      : `${label} (${quantLevels.length} files)`;
  }
  if (method === "merged") {
    return fp16Bytes && fp16Bytes > 0 ? `~${formatModelSize(fp16Bytes)}` : "";
  }
  if (method === "lora") {
    // Adapter size is bounded by LoRA rank, not the base model size.
    return "~100 MB";
  }
  return "";
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
