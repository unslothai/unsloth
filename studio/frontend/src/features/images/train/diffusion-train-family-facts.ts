// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DiffusionTrainingFacts = {
  params?: string | null;
  qlora_vram_gb?: number | null;
  gated?: boolean | null;
  note?: string | null;
};

type FamilyWithBaseFacts = DiffusionTrainingFacts & {
  base_specs?: Record<string, DiffusionTrainingFacts>;
};

/** Resolve the chips for the selected checkpoint, falling back to its family facts. */
export function resolveDiffusionTrainingFacts(
  family: FamilyWithBaseFacts,
  baseModel?: string | null,
): DiffusionTrainingFacts {
  const wanted = (baseModel ?? "").trim().toLowerCase();
  const override = Object.entries(family.base_specs ?? {}).find(
    ([repo]) => repo.trim().toLowerCase() === wanted,
  )?.[1];
  if (!override) return family;
  return {
    params: override.params ?? family.params,
    qlora_vram_gb: override.qlora_vram_gb ?? family.qlora_vram_gb,
    gated: override.gated ?? family.gated,
    note: override.note ?? family.note,
  };
}
