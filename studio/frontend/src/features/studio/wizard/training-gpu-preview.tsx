// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingParallelismMode } from "@/features/training/types/config";
import type { SystemGpuDevice } from "@/hooks/gpu-selection";
import type { ReactElement } from "react";

export function TrainingGpuPreview({
  mode,
  devices,
  gpuAvailable,
  totalMemoryGb,
  labels,
}: {
  mode: TrainingParallelismMode;
  devices: readonly SystemGpuDevice[];
  gpuAvailable: boolean;
  totalMemoryGb: number;
  labels: {
    hardware: string;
    vram: string;
    automatic: string;
    unavailable: string;
    noSelection: string;
    device: (index: number, name: string) => string;
    total: (memoryGb: string) => string;
  };
}): ReactElement {
  const hardwareValue =
    devices.length > 0 ? (
      <span className="flex min-w-0 flex-col items-end gap-1 text-right">
        {devices.map((device) => (
          <span key={device.index} className="max-w-full truncate">
            {labels.device(device.index, device.name)}
            {devices.length > 1 && (
              <span className="ml-1.5 text-muted-foreground/70">
                {device.memoryTotalGb.toFixed(1)} GiB
              </span>
            )}
          </span>
        ))}
        {mode === "model_parallel" && devices.length > 1 && (
          <span className="text-muted-foreground/70">
            {labels.total(totalMemoryGb.toFixed(1))}
          </span>
        )}
      </span>
    ) : (
      <span>{emptyHardwareLabel(mode, gpuAvailable, labels)}</span>
    );

  return (
    <>
      <div className="flex items-baseline justify-between gap-3">
        <span className="shrink-0 text-ui-11p5 text-muted-foreground/85">
          {labels.hardware}
        </span>
        <span className="min-w-0 break-words text-right text-ui-12p5 text-foreground/90">
          {hardwareValue}
        </span>
      </div>
      {devices.length === 1 && (
        <div className="flex items-baseline justify-between gap-3">
          <span className="shrink-0 text-ui-11p5 text-muted-foreground/85">
            {labels.vram}
          </span>
          <span
            className="min-w-0 truncate text-ui-12p5 text-foreground/90"
            title={`${totalMemoryGb.toFixed(1)} GiB`}
          >
            {totalMemoryGb.toFixed(1)} GiB
          </span>
        </div>
      )}
    </>
  );
}

function emptyHardwareLabel(
  mode: TrainingParallelismMode,
  gpuAvailable: boolean,
  labels: { automatic: string; unavailable: string; noSelection: string },
): string {
  if (!gpuAvailable) return labels.unavailable;
  if (mode === "auto") return labels.automatic;
  return labels.noSelection;
}
