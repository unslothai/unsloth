// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isGgufLike } from "./model-identifiers.ts";

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
  | "npu"
  | "unsloth";

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
  if (formatFilter === "npu") return false;
  // Where a model came from, not its format: the On Device list checks the row's source.
  if (formatFilter === "unsloth") return false;
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

// Strictest first: a mixed "NVFP4-FP8" checkpoint still holds NVFP4 weights and needs Blackwell.
const CHECKPOINT_QUANT_NAME: ReadonlyArray<readonly [string, RegExp]> = [
  ["nvfp4", /(?:^|[-_/.])nvfp4(?:[-_/.]|$)/i],
  ["fp8", /(?:^|[-_/.])fp8(?:[-_/.]|$)/i],
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

/** Recommended for an image / video model: also full-precision checkpoints, since diffusers repos are the main way to run them. */
export function matchesRecommendedDiffusion(
  result: { id: string; isGguf: boolean; quantMethod?: string },
  checkpointQuantFormats: readonly string[],
): boolean {
  return (
    matchesRecommended(result, checkpointQuantFormats) ||
    checkpointQuantFormat(result) === null
  );
}

// Recommended browsing lists every GGUF but only checkpoints with more than this many likes.
export const MIN_RECOMMENDED_CHECKPOINT_LIKES = 30;

export function hasRecommendedCheckpointLikes(result: {
  id: string;
  isGguf: boolean;
  likes?: number;
  tags?: readonly string[];
}): boolean {
  // A safetensors repo that also ships a stray .gguf is still tagged "gguf"; its weights are safetensors.
  const isGgufRepo =
    isGgufLike(result.id) ||
    (result.isGguf &&
      !result.tags?.some((tag) => tag.toLowerCase() === "safetensors"));
  return isGgufRepo || (result.likes ?? 0) > MIN_RECOMMENDED_CHECKPOINT_LIKES;
}

// Inference-only quant formats Unsloth cannot fine-tune. Matched on the repo
// name since the search listing often omits the quant config.
const NON_FINETUNABLE_NAME =
  /(?:^|[-_/.])(?:fp8|nvfp4|mxfp4|w4a16|w8a8|w8a16|int4|int8|gptq|awq|mobile|litert|tflite)(?:[-_/.]|$)/i;
const FINETUNABLE_QUANT = new Set(["bitsandbytes", "bnb", "bnb_4bit"]);

export function isUnslothFinetunable(result: {
  id: string;
  quantMethod?: string;
}): boolean {
  if (NON_FINETUNABLE_NAME.test(result.id)) return false;
  const quant = result.quantMethod?.toLowerCase();
  return !quant || FINETUNABLE_QUANT.has(quant);
}
