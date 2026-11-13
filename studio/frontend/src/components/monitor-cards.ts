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
  label: string;
  name: string;
  title: string;
  usedGb: number | null;
  totalGb: number;
}

const VENDOR_PREFIX = /^(?:AMD\s+Radeon(?:\(TM\))?|NVIDIA(?:\s+GeForce)?|Intel(?:\(R\))?\s+Arc(?:\(TM\))?)\s+/i;

export function shortCardName(name: string | undefined): string {
  if (!name) return "GPU";
  return name.replace(VENDOR_PREFIX, "") || name;
}

export interface VramCards {
  cards: GpuDevice[];
  totalGb: number;
  /** null = unknown: one card without a reading must not read as 0. */
  usedGb: number | null;
  rows: CardUsage[];
  showTotal: boolean;
}

/** Dedicated cards only: an iGPU borrows system RAM, and its missing reading would blank the total. */
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
          // The ordinal the detail rows and Settings use, so a masked device keeps one name.
          label: `GPU ${card.visible_ordinal ?? card.index ?? position}`,
          name: shortCardName(card.name),
          title: card.name ?? "GPU",
          usedGb: Number.isFinite(card.vram_used_gb) ? (card.vram_used_gb as number) : null,
          totalGb: gpuMemoryTotalsGb([card]).total,
        }))
      : [];
  const showTotal =
    rows.length === 0 || (usedGb !== null && rows.some((row) => row.usedGb === null));
  return { cards, totalGb: gpuMemoryTotalsGb(cards).total, usedGb, rows, showTotal };
}

export function holdsModelsInVram(cards: readonly GpuDevice[]): boolean {
  return cards.some((card) => card.shared_memory !== true && card.unified_memory !== true);
}
