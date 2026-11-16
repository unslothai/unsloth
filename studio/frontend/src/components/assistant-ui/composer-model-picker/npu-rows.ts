// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type NpuModel, npuDisplayName, npuRowsFor } from "@/features/npu";
import { type PickerRow, RECOMMENDED_TAB } from "./picker-model";

/** Discover's NPU section: models not on this device, named and ordered as the Hub lists them. */
export function npuOfferRows(
  models: NpuModel[] | null,
): (PickerRow & { npu: NpuModel })[] {
  return npuRowsFor(models, { onDevice: false, query: "" })
    .filter((model) => !model.downloaded)
    .map((model) => ({
      key: `npu-offer:${model.id}`,
      id: model.model_path,
      label: npuDisplayName(model),
      tab: RECOMMENDED_TAB,
      kind: "npu",
      section: "npu",
      npu: model,
    }));
}
