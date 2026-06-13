// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type FormatFilterModelFormat =
  | "gguf"
  | "mlx"
  | "safetensors"
  | "adapter"
  | "checkpoint"
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

// Discover results carry an isGguf flag plus HF tags/library. MLX repos are
// tagged "mlx" or use library_name "mlx", so detect those before falling back
// to safetensors.
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
