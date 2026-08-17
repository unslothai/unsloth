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
  imatrix?: boolean; // IQ quants require an importance matrix (the imatrix toggle below)
}[] = [
  { value: "iq2_xxs", label: "IQ2_XXS", imatrix: true },
  { value: "iq2_m", label: "IQ2_M", imatrix: true },
  { value: "iq3_xxs", label: "IQ3_XXS", imatrix: true },
  { value: "iq4_xs", label: "IQ4_XS", imatrix: true },
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
 * Merged-export precision formats, sorted by bit width. Three backends:
 *   - "plain":      standard save (16-bit); `formatType` is the backend `format_type`.
 *   - "compressed": llm-compressor compressed-tensors (vLLM), NVIDIA-only; `value` is the alias.
 *   - "torchao":    portable FP8/INT8, no NVIDIA GPU needed; `value` is the alias.
 * `common` entries are quick pills, the rest the "More formats" dropdown; `needsNvidia` entries
 * are hidden on non-NVIDIA hardware.
 */
export type MergedBackend = "plain" | "compressed" | "torchao";

export type MergedFormatOption = {
  value: string;
  label: string;
  bits: number;
  backend: MergedBackend;
  group: string;
  common: boolean;
  needsNvidia: boolean;
  needsCalibration?: boolean;
  hint: string;
  /** Backend `format_type` for a "plain" save (unused for compressed/torchao). */
  formatType?: string;
};

/** Kept as a string alias for back-compat with callers that typed the old union. */
export type MergedFormat = string;

export const MERGED_FORMATS: MergedFormatOption[] = [
  // 16-bit
  {
    value: "16-bit",
    label: "16-bit",
    bits: 16,
    backend: "plain",
    group: "16-bit",
    common: true,
    needsNvidia: false,
    hint: "Full precision, runs anywhere.",
    formatType: "16-bit (FP16)",
  },
  // 8-bit
  {
    value: "fp8",
    label: "FP8",
    bits: 8,
    backend: "compressed",
    group: "FP8",
    common: true,
    needsNvidia: true,
    hint: "Dynamic per-token FP8 (W8A8) for vLLM. Data-free.",
  },
  {
    value: "torchao_fp8",
    label: "FP8 (portable)",
    bits: 8,
    backend: "torchao",
    group: "Portable",
    common: true,
    needsNvidia: false,
    hint: "Device-agnostic FP8 (torchao). Produces on any hardware; loads in vLLM.",
  },
  {
    value: "w8a8",
    label: "INT8 (W8A8)",
    bits: 8,
    backend: "compressed",
    group: "INT",
    common: true,
    needsNvidia: true,
    hint: "8-bit weights and 8-bit activations for vLLM. Data-free.",
  },
  {
    value: "torchao_int8",
    label: "INT8 (portable)",
    bits: 8,
    backend: "torchao",
    group: "Portable",
    common: true,
    needsNvidia: false,
    hint: "Device-agnostic INT8 (torchao). Produces on any hardware; loads in vLLM.",
  },
  {
    value: "fp8_static",
    label: "FP8 Static",
    bits: 8,
    backend: "compressed",
    group: "FP8",
    common: false,
    needsNvidia: true,
    needsCalibration: true,
    hint: "Static per-tensor FP8. Calibrates on data.",
  },
  {
    value: "w8a16",
    label: "INT8 (W8A16)",
    bits: 8,
    backend: "compressed",
    group: "INT",
    common: false,
    needsNvidia: true,
    hint: "8-bit weight-only. Data-free.",
  },
  {
    value: "mxfp8",
    label: "MXFP8",
    bits: 8,
    backend: "compressed",
    group: "MXFP",
    common: false,
    needsNvidia: true,
    hint: "Microscaling FP8. Needs a newer compressed-tensors stack.",
  },
  // 4-bit
  {
    value: "w4a16",
    label: "INT4 (W4A16)",
    bits: 4,
    backend: "compressed",
    group: "INT",
    common: true,
    needsNvidia: true,
    hint: "4-bit weight-only (GPTQ-style) for vLLM. Data-free.",
  },
  {
    value: "mxfp4",
    label: "MXFP4",
    bits: 4,
    backend: "compressed",
    group: "MXFP",
    common: true,
    needsNvidia: true,
    hint: "Microscaling FP4 (W4A4) for vLLM. Data-free.",
  },
  {
    value: "nvfp4",
    label: "NVFP4",
    bits: 4,
    backend: "compressed",
    group: "FP4",
    common: true,
    needsNvidia: true,
    needsCalibration: true,
    hint: "NVIDIA FP4 (W4A4) for vLLM. Calibrates on data.",
  },
];

/** Look up a merged format option by its stable value. */
export function findMergedFormat(value: string): MergedFormatOption | undefined {
  return MERGED_FORMATS.find((f) => f.value === value);
}

/** Backend payload for one merged format: plain -> formatType, compressed/torchao -> the alias. */
export function mergedFormatPayload(value: string): {
  formatType: string;
  compressedMethod: string | null;
} {
  const opt = findMergedFormat(value);
  if (!opt || opt.backend === "plain") {
    return {
      formatType: opt?.formatType ?? "16-bit (FP16)",
      compressedMethod: null,
    };
  }
  return { formatType: "16-bit (FP16)", compressedMethod: opt.value };
}

/**
 * llama.cpp effective bits-per-weight per quant; GGUF size ~= fp16_bytes * bpw / 16.
 * K-quant values are published average bit-rates (Q2_K_L = Unsloth Q2_K + Q8_0
 * embeddings). Approximate ("~"), not exact file sizes.
 */
export const GGUF_BPW: Record<string, number> = {
  iq2_xxs: 2.06,
  iq2_m: 2.7,
  iq3_xxs: 3.06,
  iq4_xs: 4.25,
  q2_k_l: 3.35,
  q3_k_m: 3.91,
  q4_k_m: 4.83,
  q5_k_m: 5.67,
  q6_k: 6.56,
  q8_0: 8.5,
  bf16: 16,
  f16: 16,
};

const FP16_BPW = 16;

/**
 * Human-readable base-1024 size ("67 GB"), matching the model-selector picker.
 * Do NOT use the base-1000 hub formatBytes here -- it would disagree ("72 GB").
 */
export function formatModelSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  // clamp: bytes < 1 would give a negative index
  const i = Math.max(
    0,
    Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1),
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
 * Estimated total export size for the summary line; scales from the model's
 * real fp16 size, returns "" when unknown so the UI can hide a wrong number.
 */
export function getEstimatedSize(
  method: ExportMethod | null,
  quantLevels: string[],
  fp16Bytes: number | null | undefined,
): string {
  if (method === "gguf" && quantLevels.length > 0) {
    const perQuant = quantLevels.map((q) => estimateQuantBytes(fp16Bytes, q));
    if (perQuant.some((b) => b == null)) {
      return ""; // unknown -> blank
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
