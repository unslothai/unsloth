// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FormatFilterModelFormat =
  | "gguf"
  | "safetensors"
  | "adapter"
  | "checkpoint"
  | "mlx"
  | "unknown";

export type FormatFilterValue = "all" | "gguf" | "checkpoint" | "mlx";

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
  if (formatFilter === "gguf") return normalized === "gguf";
  if (formatFilter === "mlx") return normalized === "mlx";
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
