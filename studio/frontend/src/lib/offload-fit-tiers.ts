// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One backend-reported offload fit tier in the picker's units: total VRAM of the load device in
 *  GiB and AVAILABLE system RAM in GiB. Same shape as the catalog's `OffloadFitTier`. */
export interface ReportedOffloadFitTier {
  gpuGb: number;
  systemRamGb: number;
  requiresQuantisedStreaming?: boolean;
}

/** `/api/system.diffusers_offload_tiers` as lower-cased repo id -> tiers. Anything malformed is
 *  dropped, never coerced: a tier the picker cannot read must not admit a load. A tier missing
 *  `requires_quantised_streaming` is treated as requiring it, the conservative reading. */
export function normalizeReportedOffloadFitTiers(
  raw: unknown,
): Record<string, ReportedOffloadFitTier[]> {
  const out: Record<string, ReportedOffloadFitTier[]> = {};
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return out;
  for (const [repoId, tiers] of Object.entries(
    raw as Record<string, unknown>,
  )) {
    if (!Array.isArray(tiers)) continue;
    const kept: ReportedOffloadFitTier[] = [];
    for (const tier of tiers) {
      if (!tier || typeof tier !== "object") continue;
      const t = tier as Record<string, unknown>;
      const gpuGb = t.gpu_gb;
      const systemRamGb = t.system_ram_gb;
      if (
        typeof gpuGb !== "number" ||
        typeof systemRamGb !== "number" ||
        !Number.isFinite(gpuGb) ||
        !Number.isFinite(systemRamGb) ||
        gpuGb <= 0 ||
        systemRamGb < 0
      )
        continue;
      kept.push({
        gpuGb,
        systemRamGb,
        requiresQuantisedStreaming: t.requires_quantised_streaming !== false,
      });
    }
    if (kept.length) out[repoId.trim().toLowerCase()] = kept;
  }
  return out;
}
