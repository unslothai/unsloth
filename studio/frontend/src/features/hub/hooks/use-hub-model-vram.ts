// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import type { GpuInfo } from "@/hooks/use-gpu-info";
import {
  type VramFitStatus,
  checkVramFit,
  estimateLoadingVram,
} from "@/lib/vram";
import type { SelectedModelView } from "../types";

export interface ModelVramInfo {
  est: number;
  status: VramFitStatus;
}

export function useHubModelVram(
  selectedModel: SelectedModelView | null,
  gpu: GpuInfo,
): { vramInfo: ModelVramInfo | null } {
  const vramInfo = useMemo<ModelVramInfo | null>(() => {
    if (!selectedModel || selectedModel.isGguf || !selectedModel.totalParams) {
      return null;
    }
    const est = estimateLoadingVram(selectedModel.totalParams, "qlora");
    const status = checkVramFit(est, gpu.available ? gpu.memoryTotalGb : 0);
    return status ? { est, status } : null;
  }, [gpu, selectedModel]);

  return { vramInfo };
}
