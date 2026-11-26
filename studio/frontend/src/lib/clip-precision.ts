// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const SCHEMES = ["fp8", "int8", "nvfp4", "mxfp8"] as const;

/**
 * The Precision select value that reproduces a saved clip: its engaged transformer scheme, "none" when the clip
 * recorded a dense run (null / "off" / "none"), and null when the sidecar predates the field (keep the current pick).
 */
export function clipPrecisionSelection(
  transformerQuant: string | null | undefined,
): (typeof SCHEMES)[number] | "none" | null {
  if (transformerQuant === undefined) return null;
  const value = (transformerQuant ?? "").trim().toLowerCase();
  if (value === "" || value === "off" || value === "none") return "none";
  return SCHEMES.find((s) => s === value) ?? null;
}
