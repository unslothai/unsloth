// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FormatFilterModelFormat =
  | "gguf"
  | "safetensors"
  | "adapter"
  | "checkpoint"
  | "mlx"
  | "unknown";

export type FormatFilterValue =
  | "all"
  | "recommended"
  | "gguf"
  | "checkpoint"
  | "mlx"
  | "npu";

export function matchesFormat(
  modelFormat: boolean | FormatFilterModelFormat | null | undefined,
  formatFilter: FormatFilterValue,
): boolean {
  if (formatFilter === "all") return true;
  const normalized =
    typeof modelFormat === "boolean"
      ? modelFormat
        ? "gguf"
        : "safetensors"
      : modelFormat;
  // Recommended's FP8 / NVFP4 checkpoints need the repo id: see matchesRecommended.
  if (formatFilter === "gguf" || formatFilter === "recommended") {
    return normalized === "gguf";
  }
  if (formatFilter === "mlx") return normalized === "mlx";
  // NPU models come from Lemonade's catalog, never from a Hub repo or the disk scan.
  if (formatFilter === "npu") return false;
  return normalized === "safetensors" || normalized === "checkpoint";
}

export function detectResultFormat(result: {
  isGguf: boolean;
  tags?: string[];
  libraryName?: string;
}): FormatFilterModelFormat {
  if (result.isGguf) return "gguf";
  if (
    result.libraryName?.toLowerCase() === "mlx" ||
    result.tags?.some((tag) => tag.toLowerCase() === "mlx")
  ) {
    return "mlx";
  }
  return "safetensors";
}

const CHECKPOINT_QUANT_NAME: ReadonlyArray<readonly [string, RegExp]> = [
  ["fp8", /(?:^|[-_/.])fp8(?:[-_/.]|$)/i],
  ["nvfp4", /(?:^|[-_/.])nvfp4(?:[-_/.]|$)/i],
];
// A second quant token (an FP8 base re-quantized to GPTQ / IQ4 / MXFP4) means the repo is not an FP8 checkpoint.
const OTHER_QUANT_NAME =
  /(?:^|[-_/.])(?:gptq|awq|exl2|mxfp4|int4|int8|w4a16|w8a16|bnb|iq\d\w*|q\d_\w+)(?:[-_/.]|$)/i;
const OTHER_QUANT_METHOD = new Set([
  "gptq",
  "awq",
  "exl2",
  "aqlm",
  "hqq",
  "quanto",
  "bitsandbytes",
]);

/** The prequantized checkpoint format ("fp8" / "nvfp4") a non-GGUF repo ships, from its name or
 *  quant config, or null. */
export function checkpointQuantFormat(result: {
  id: string;
  isGguf: boolean;
  quantMethod?: string;
}): string | null {
  if (result.isGguf || OTHER_QUANT_NAME.test(result.id)) return null;
  if (OTHER_QUANT_METHOD.has(result.quantMethod?.toLowerCase() ?? ""))
    return null;
  for (const [format, pattern] of CHECKPOINT_QUANT_NAME) {
    if (pattern.test(result.id)) return format;
  }
  return result.quantMethod?.toLowerCase() === "fp8" ? "fp8" : null;
}

/** Recommended: every GGUF, plus FP8 / NVFP4 checkpoints only where every GPU runs them natively. */
export function matchesRecommended(
  result: { id: string; isGguf: boolean; quantMethod?: string },
  checkpointQuantFormats: readonly string[],
): boolean {
  if (result.isGguf) return true;
  const format = checkpointQuantFormat(result);
  return format !== null && checkpointQuantFormats.includes(format);
}

// Inference-only quant formats Unsloth cannot fine-tune. Matched on the repo
// name since the search listing often omits the quant config.
const NON_FINETUNABLE_NAME =
  /(?:^|[-_/.])(?:fp8|nvfp4|mxfp4|w4a16|w8a8|w8a16|int4|int8|gptq|awq|mobile|litert|tflite)(?:[-_/.]|$)/i;
// Quant methods Unsloth can fine-tune: full precision (none) or bitsandbytes.
const FINETUNABLE_QUANT = new Set(["bitsandbytes", "bnb", "bnb_4bit"]);

export function isUnslothFinetunable(result: {
  id: string;
  quantMethod?: string;
}): boolean {
  if (NON_FINETUNABLE_NAME.test(result.id)) return false;
  const quant = result.quantMethod?.toLowerCase();
  return !quant || FINETUNABLE_QUANT.has(quant);
}
