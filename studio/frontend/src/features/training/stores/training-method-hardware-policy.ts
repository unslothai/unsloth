// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { TrainingMethod } from "@/types/training";

export async function selectTrainingMethodForHardware(
  modelSizeBytes: number,
  contextLength: number,
  signal: AbortSignal,
): Promise<TrainingMethod | null> {
  try {
    const response = await authFetch("/api/system/hardware", { signal });
    if (!response.ok) {
      return null;
    }
    const hardware = await response.json();
    const freeVramGb: number | null = hardware?.gpu?.vram_free_gb ?? null;
    if (freeVramGb == null) {
      return null;
    }

    let contextScale = 1;
    if (contextLength >= 32768) {
      contextScale = 4;
    } else if (contextLength >= 16384) {
      contextScale = 2;
    } else if (contextLength > 8192) {
      contextScale = 1.7;
    }

    const modelSizeGb = modelSizeBytes / 1024 ** 3;
    const estimatedUsageGb = modelSizeGb * 1.5 * contextScale;
    return estimatedUsageGb <= freeVramGb ? "lora" : "qlora";
  } catch {
    return null;
  }
}
