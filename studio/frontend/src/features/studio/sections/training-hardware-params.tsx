// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TabsContent } from "@/components/ui/tabs";
import { useTrainingConfigStore } from "@/features/training";
import { reconcileTrainingGpuSelection } from "@/features/training/lib/training-gpu-selection";
import type { TrainingParallelismMode } from "@/features/training/types/config";
import {
  cachedTrainingGpuIndices,
  useTrainingGpuDevices,
  useGpuInfo,
} from "@/hooks/use-gpu-info";
import { useT } from "@/i18n";
import { useEffect, type ReactElement } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow } from "./params-section-controls";

export function TrainingHardwareParams(): ReactElement {
  const t = useT();
  const devices = useTrainingGpuDevices();
  const gpu = useGpuInfo();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      parallelismMode: state.parallelismMode,
      selectedGpuIds: state.selectedGpuIds,
      setGpuSelection: state.setGpuSelection,
    })),
  );
  // Training consumes physical torch device IDs; inference's `pinnable` also
  // encodes llama.cpp/GGUF support and incorrectly excludes valid XPU devices.
  const selectable = devices.filter(
    (device) => device.indexKind === "physical",
  );
  const selected = store.selectedGpuIds ?? [];
  useEffect(() => {
    const next = reconcileTrainingGpuSelection(
      store.parallelismMode,
      store.selectedGpuIds,
      cachedTrainingGpuIndices(),
    );
    if (
      next.parallelismMode !== store.parallelismMode ||
      JSON.stringify(next.selectedGpuIds) !==
        JSON.stringify(store.selectedGpuIds)
    ) {
      store.setGpuSelection(next.parallelismMode, next.selectedGpuIds);
    }
  }, [
    // Re-run when async device discovery updates its rendered inventory.
    devices,
    store.parallelismMode,
    store.selectedGpuIds,
    store.setGpuSelection,
  ]);
  const canSelect = selectable.length > 0;
  const modeDescription: Record<TrainingParallelismMode, string> = {
    auto: t("studio.params.hardwareModeAuto"),
    single: t("studio.params.hardwareModeSingle"),
    model_parallel: t("studio.params.hardwareModeSharding"),
    ddp: t("studio.params.hardwareModeDdp"),
  };
  const memoryLabel = (free: number, total: number, freeKnown?: boolean) =>
    freeKnown === false
      ? t("studio.params.memoryUnknown", { total: total.toFixed(1) })
      : t("studio.params.memoryAvailable", {
          free: free.toFixed(1),
          total: total.toFixed(1),
        });

  const chooseMode = (mode: TrainingParallelismMode) => {
    if (mode === "auto") {
      store.setGpuSelection(mode, null);
      return;
    }
    const retained = selected.filter((id) =>
      selectable.some((device) => device.index === id),
    );
    if (mode === "single") {
      const first = retained[0] ?? selectable[0]?.index;
      store.setGpuSelection(mode, first === undefined ? [] : [first]);
      return;
    }
    store.setGpuSelection(
      mode,
      retained.length >= 2
        ? retained
        : selectable.map((device) => device.index),
    );
  };

  const toggleDevice = (id: number, checked: boolean) => {
    if (store.parallelismMode === "single") {
      if (checked) store.setGpuSelection("single", [id]);
      return;
    }
    const next = checked
      ? [...new Set([...selected, id])].sort((a, b) => a - b)
      : selected.filter((selectedId) => selectedId !== id);
    store.setGpuSelection(store.parallelismMode, next);
  };

  const renderDeviceSelection = (): ReactElement => {
    if (!canSelect) {
      return (
        <p className="text-xs text-destructive">
          {t("studio.params.noSelectableGpu")}
        </p>
      );
    }
    if (store.parallelismMode === "auto") {
      return (
        <div className="flex flex-col gap-1 text-xs text-muted-foreground">
          <span>{t("studio.params.liveInventory")}</span>
          {selectable.map((device) => (
            <span key={device.index}>
              GPU {device.index}: {device.name} —{" "}
              {memoryLabel(
                device.memoryFreeGb,
                device.memoryTotalGb,
                device.memoryFreeKnown,
              )}
            </span>
          ))}
        </div>
      );
    }
    return (
      <div className="flex flex-col gap-2">
        {selectable.map((device) => {
          const checked = selected.includes(device.index);
          return (
            <label
              key={device.index}
              className="flex cursor-pointer items-center justify-between rounded-md border px-3 py-2 text-xs"
            >
              <span className="min-w-0 truncate">
                GPU {device.index} · {device.name}
              </span>
              <span className="ml-3 flex shrink-0 items-center gap-2 text-muted-foreground">
                {memoryLabel(
                  device.memoryFreeGb,
                  device.memoryTotalGb,
                  device.memoryFreeKnown,
                )}
                <Checkbox
                  checked={checked}
                  disabled={store.parallelismMode === "single" && checked}
                  onCheckedChange={(value) =>
                    toggleDevice(device.index, value === true)
                  }
                />
              </span>
            </label>
          );
        })}
        {(store.parallelismMode === "model_parallel" ||
          store.parallelismMode === "ddp") &&
          selected.length < 2 && (
            <p className="text-xs text-destructive">
              {t("studio.params.multipleGpusRequired")}
            </p>
          )}
      </div>
    );
  };

  return (
    <TabsContent value="hardware" className="mt-3 flex flex-col gap-3">
      <ParamsRow
        label={t("studio.params.gpuPlacement")}
        tooltip={t("studio.params.gpuPlacementTooltip")}
      >
        <Select
          value={store.parallelismMode}
          onValueChange={(value) =>
            chooseMode(value as TrainingParallelismMode)
          }
          disabled={!canSelect}
        >
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">
              {t("studio.params.hardwareModeAutoLabel")}
            </SelectItem>
            <SelectItem value="single">
              {t("studio.params.hardwareModeSingleLabel")}
            </SelectItem>
            <SelectItem value="model_parallel" disabled={selectable.length < 2}>
              {t("studio.params.hardwareModeShardingLabel")}
            </SelectItem>
            <SelectItem
              value="ddp"
              disabled={selectable.length < 2 || gpu.backend !== "cuda"}
            >
              {t("studio.params.hardwareModeDdpLabel")}
            </SelectItem>
          </SelectContent>
        </Select>
      </ParamsRow>
      <p className="text-xs text-muted-foreground">
        {modeDescription[store.parallelismMode]}
      </p>
      {renderDeviceSelection()}
    </TabsContent>
  );
}
