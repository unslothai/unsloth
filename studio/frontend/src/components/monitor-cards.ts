// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  carveOutVramGb,
  gpuMemoryTotalsGb,
  resolveGpuVramUsedGb,
  sharesHostMemoryDevice,
  type VramReportingGpu,
} from "../hooks/gpu-vram.ts";
import type { GpuDevice } from "../hooks/use-system.ts";
import type { LoadedModelEntry } from "../features/loaded-models/loaded-models-sources.ts";
import type { NpuStatus } from "../features/npu/api.ts";

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

// A carve-out APU's shared part is system RAM, which the RAM row already shows.
function vramTotalGb(cards: GpuDevice[]): number {
  return (
    carveOutVramGb(cards, null, null)?.totalGb ??
    gpuMemoryTotalsGb(cards).total
  );
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
          totalGb: vramTotalGb([card]),
        }))
      : [];
  const showTotal =
    rows.length === 0 || (usedGb !== null && rows.some((row) => row.usedGb === null));
  return { cards, totalGb: vramTotalGb(cards), usedGb, rows, showTotal };
}

export function holdsModelsInVram(cards: readonly GpuDevice[]): boolean {
  return cards.some((card) => card.shared_memory !== true && card.unified_memory !== true);
}

export interface NpuCard {
  name: string;
  driver: string | null;
  model: string | null;
  residentGb: number | null;
  loadPercent: number | null;
}

/** null hides the card. The NPU has no memory of its own: its model's size is system RAM. */
export function npuCard(
  status: Pick<NpuStatus, "hardware"> | null,
  entries: readonly LoadedModelEntry[],
  loadPercent: number | null | undefined,
): NpuCard | null {
  const hardware = status?.hardware;
  if (!hardware?.supported) return null;
  // describeInferenceStatus leads an NPU row's detail with its runtime.
  const row = entries.find(
    (entry) => !entry.loading && entry.detail.split(" · ")[0] === "NPU",
  );
  const driver = [
    hardware.driver === "installed" ? null : hardware.driver,
    hardware.driver_version,
  ]
    .filter(Boolean)
    .join(" ");
  return {
    name: hardware.name || "NPU",
    driver: driver || null,
    model: row ? row.label || row.name : null,
    residentGb: row?.residentMib ? row.residentMib / 1024 : null,
    loadPercent: Number.isFinite(loadPercent) ? (loadPercent as number) : null,
  };
}
