// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which transformer Precision options can load, before the load is tried: the host's probed scheme ladder
// (/api/system dense_quant_schemes) and the family deny table (status transformer_quant_denials).

/** The schemes /api/system dense_quant_schemes can list (the auto ladder by GPU capability); nvfp4 is never in it. */
const LADDER_SCHEMES = new Set(["int8", "fp8", "mxfp8"]);

export type TransformerQuantDenials = Record<string, Record<string, string>>;

export interface PrecisionAvailabilityContext {
  /** /api/system dense_quant_supported: a CUDA bf16 host on the torchao path. */
  denseQuantSupported: boolean;
  /** /api/system dense_quant_schemes. Empty is unknown (older backend, MPS, CPU), never "nothing loads". */
  denseQuantSchemes: readonly string[];
  /** CUDA cards on the host: with more than one the list is their intersection, not the selected card's. */
  gpuCount?: number;
  /** The family the pick loads as, or null when it is not known before the load. */
  family: string | null | undefined;
  denials: TransformerQuantDenials | null | undefined;
}

function norm(value: string | null | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

/** Why `value` cannot load for this pick on this host, or null when nothing rules it out. */
export function precisionUnavailableReason(value: string, ctx: PrecisionAvailabilityContext): string | null {
  const scheme = norm(value);
  if (!scheme || scheme === "auto" || scheme === "none" || scheme === "off") return null;
  const family = norm(ctx.family);
  if (family && ctx.denials) {
    const reasons = Object.entries(ctx.denials).find(([name]) => norm(name) === family)?.[1];
    const reason = reasons
      ? Object.entries(reasons).find(([name]) => norm(name) === scheme)?.[1]
      : undefined;
    if (reason) return reason;
  }
  const schemes = ctx.denseQuantSchemes.map(norm).filter(Boolean);
  const singleCard = (ctx.gpuCount ?? 1) <= 1;
  if (
    ctx.denseQuantSupported &&
    singleCard &&
    schemes.length > 0 &&
    LADDER_SCHEMES.has(scheme) &&
    !schemes.includes(scheme)
  ) {
    return `${scheme.toUpperCase()} did not pass this GPU and PyTorch build's quantize and matmul check, or the card lacks its kernels.`;
  }
  return null;
}

/** Each option's unavailability reason, keyed by option value; only options that are ruled out appear. */
export function precisionDisabledReasons(
  options: readonly (readonly [string, string])[],
  ctx: PrecisionAvailabilityContext,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [value] of options) {
    const reason = precisionUnavailableReason(value, ctx);
    if (reason) out[value] = reason;
  }
  return out;
}

/** The resident family's denials for the options, as advice only: the select also configures the NEXT pick, which may
 * be another family, so they are shown (tooltip and a note when selected) but never disable an option. */
export function precisionAdvisoryReasons(
  options: readonly (readonly [string, string])[],
  residentFamily: string | null | undefined,
  denials: TransformerQuantDenials | null | undefined,
): Record<string, string> {
  return precisionDisabledReasons(options, {
    denseQuantSupported: false,
    denseQuantSchemes: [],
    family: residentFamily,
    denials,
  });
}

/** The family a pick loads as: an explicit override, else the resident family when the pick is the resident model. */
export function pickFamily(
  familyOverride: string | null | undefined,
  pickIsResident: boolean,
  residentFamily: string | null | undefined,
): string | null {
  const explicit = norm(familyOverride);
  if (explicit && explicit !== "auto") return explicit;
  return pickIsResident ? norm(residentFamily) || null : null;
}
