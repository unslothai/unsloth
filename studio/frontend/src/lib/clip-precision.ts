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

const TE_SCHEMES = ["fp8", "fp8_dynamic", "int8", "nvfp4"] as const;

/**
 * The Text encoder precision select value that reproduces a saved clip: its engaged encoder scheme, "none" when the
 * clip recorded a dense encoder, and null when the clip carries no precision record (keep the current pick).
 *
 * The sidecar writes the ENGAGED encoder scheme from the loaded state, so a family default that loaded a quantized
 * encoder is recorded as that scheme, and null next to `model_kind` is a dense run (as for the transformer above).
 */
export function clipTextEncoderPrecisionSelection(
  textEncoderQuant: string | null | undefined,
  modelKind?: string | null,
): (typeof TE_SCHEMES)[number] | "none" | null {
  const value = (textEncoderQuant ?? "").trim().toLowerCase();
  if (value === "off" || value === "none") return "none";
  if (value === "") return modelKind ? "none" : null;
  return TE_SCHEMES.find((s) => s === value) ?? null;
}
