// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isNpuModelId,
  type NpuModel,
  npuDisplayName,
  npuRowsFor,
} from "@/features/npu";
import { DEVICE_TAB, type PickerRow, RECOMMENDED_TAB } from "./picker-model";

export type NpuPickerRow = PickerRow & { npu: NpuModel };

/** The Hub's NPU names and order; `id` stays the `lemonade:` path that loads. */
export function npuPickerRows(
  models: NpuModel[] | null,
  onDevice: boolean,
): NpuPickerRow[] {
  return npuRowsFor(models, { onDevice, query: "" })
    .filter((model) => model.downloaded === onDevice)
    .map((model, rank) => ({
      key: `${onDevice ? "npu" : "npu-offer"}:${model.id}`,
      id: model.model_path,
      label: npuDisplayName(model),
      tab: onDevice ? DEVICE_TAB : RECOMMENDED_TAB,
      kind: "npu",
      // A Q4NX folder from the Export page sits with the other exports.
      section: model.local ? "exported" : "npu",
      npu: model,
      rank,
      sizeBytes:
        onDevice && model.size_gb != null ? model.size_gb * 1e9 : undefined,
    }));
}

/** Whether Discover shows the NPU notice: setup before the runtime is ready, its spinner while the
 * first listing loads (else the section vanishes right after Enable), or a failed listing. */
export function npuNoticeShown(
  catalog: {
    ready: boolean;
    listError: string | null;
    models: readonly unknown[] | null;
  } | null,
): boolean {
  return (
    catalog !== null &&
    (!catalog.ready || catalog.listError !== null || catalog.models === null)
  );
}

/** Org whose Hugging Face avatar marks an NPU row: its repo's owner, else FastFlowLM. */
export function npuMarkOwner(row?: {
  kind?: string;
  id?: string;
  npu?: Pick<NpuModel, "hf_repo">;
}): string | null {
  if (row?.kind !== "npu" && !isNpuModelId(row?.id)) return null;
  return row?.npu?.hf_repo?.split("/")[0]?.trim() || "FastFlowLM";
}
