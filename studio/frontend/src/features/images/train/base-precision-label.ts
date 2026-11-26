// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type BasePrecisionMode = "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto";

/** Train-panel label of a frozen-base precision; NF4 is named apart from NVFP4, which training does not offer. */
export function basePrecisionLabel(m: BasePrecisionMode): string {
  if (m === "auto") return "Auto";
  if (m === "nf4") return "4-bit NF4 (not NVFP4)";
  if (m === "bf16") return "bf16 (fastest)";
  if (m === "int8") return "int8";
  if (m === "mxfp8") return "mxfp8 (Blackwell)";
  return "fp8 (experimental)";
}
