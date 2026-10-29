// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  gpuMemoryTotalsGb,
  resolveGpuVramUsedGb,
  sharesHostMemoryDevice,
  type VramReportingGpu,
} from "../hooks/gpu-vram.ts";
import type { GpuDevice } from "../hooks/use-system.ts";

export interface CardUsage {
  key: string;
  name: string;
  usedGb: number | null;
  totalGb: number;
}

export interface VramCards {
  /** Dedicated cards; every usage device when none is dedicated. */
  cards: GpuDevice[];
  totalGb: number;
  /** null = unknown: one card without a reading must not read as 0. */
  usedGb: number | null;
  /** One row per card, only when there is more than one. */
  rows: CardUsage[];
}

/**
 * The VRAM row counts dedicated cards. An iGPU beside them borrows system RAM, which the RAM row
 * already shows, and its missing reading would otherwise blank the whole total (Windows ROCm).
 */
export function vramCards(
  devices: GpuDevice[],
  usageGpu: VramReportingGpu | null | undefined,
): VramCards {
  const dedicated = devices.filter((device) => !sharesHostMemoryDevice(device));
  const cards = dedicated.length > 0 ? dedicated : devices;
  const usedGb =
    cards.length === devices.length
      ? resolveGpuVramUsedGb(usageGpu)
      : resolveGpuVramUsedGb({ devices: cards });
  const rows =
    cards.length > 1
      ? cards.map((card, position) => ({
          key: `${card.index ?? position}`,
          name: `${card.index ?? position} · ${card.name ?? "GPU"}`,
          usedGb: Number.isFinite(card.vram_used_gb) ? (card.vram_used_gb as number) : null,
          totalGb: gpuMemoryTotalsGb([card]).total,
        }))
      : [];
  return { cards, totalGb: gpuMemoryTotalsGb(cards).total, usedGb, rows };
}
