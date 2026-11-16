// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type NpuModel, npuDisplayName, npuRowsFor } from "@/features/npu";
import { DEVICE_TAB, type PickerRow, RECOMMENDED_TAB } from "./picker-model";

export type NpuPickerRow = PickerRow & { npu: NpuModel };

/** The Hub's NPU names and order: downloaded models On Device, the rest offered in Discover. */
export function npuPickerRows(models: NpuModel[] | null): {
  device: NpuPickerRow[];
  offers: NpuPickerRow[];
} {
  const row = (model: NpuModel): NpuPickerRow => ({
    key: `${model.downloaded ? "npu" : "npu-offer"}:${model.id}`,
    id: model.model_path,
    label: npuDisplayName(model),
    tab: model.downloaded ? DEVICE_TAB : RECOMMENDED_TAB,
    kind: "npu",
    section: "npu",
    npu: model,
    sizeBytes:
      model.downloaded && model.size_gb != null ? model.size_gb * 1e9 : undefined,
  });
  const rows = npuRowsFor(models, { onDevice: false, query: "" }).map(row);
  return {
    device: rows.filter((r) => r.npu.downloaded),
    offers: rows.filter((r) => !r.npu.downloaded),
  };
}
