// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SCHEMES = ["fp8", "int8", "nvfp4", "mxfp8"] as const;

/**
 * The Precision select value that reproduces a saved clip: its engaged transformer scheme, "none" when the clip
 * recorded a dense run, and null when the clip carries no precision record (keep the current pick).
 *
 * The gallery API serialises a missing `transformer_quant` as null, the same as a recorded dense run, so null only
 * means dense when the clip also carries `model_kind`, which the sidecar writes alongside it from the engaged state.
 */
export function clipPrecisionSelection(
  transformerQuant: string | null | undefined,
  modelKind?: string | null,
): (typeof SCHEMES)[number] | "none" | null {
  const value = (transformerQuant ?? "").trim().toLowerCase();
  if (value === "off" || value === "none") return "none";
  if (value === "") return modelKind ? "none" : null;
  return SCHEMES.find((s) => s === value) ?? null;
}
