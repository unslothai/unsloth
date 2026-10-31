// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Total load-device VRAM and AVAILABLE system RAM, in GiB. */
export interface ReportedOffloadFitTier {
  gpuGb: number;
  systemRamGb: number;
  requiresQuantisedStreaming?: boolean;
}

/** Malformed tiers are dropped, never coerced; missing streaming flag means required. */
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
