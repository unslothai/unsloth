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
import type { TrainingParallelismMode } from "@/features/training/types/config";
import { useGpuDevices, useGpuInfo } from "@/hooks/use-gpu-info";
import type { ReactElement } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow } from "./params-section-controls";

const modeDescription: Record<TrainingParallelismMode, string> = {
  auto: "Studio selects the least busy compatible GPU automatically.",
  single: "One model copy on one selected GPU.",
  model_parallel:
    "One model sharded across the selected GPUs. This combines VRAM; it is not DDP.",
  ddp:
    "One model replica per selected GPU. Gradients are synchronized; each selected GPU must fit the model.",
};

function memoryLabel(free: number, total: number): string {
  return `${free.toFixed(1)} / ${total.toFixed(1)} GiB free`;
}

export function TrainingHardwareParams(): ReactElement {
  const devices = useGpuDevices();
  const gpu = useGpuInfo();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      parallelismMode: state.parallelismMode,
      selectedGpuIds: state.selectedGpuIds,
      setGpuSelection: state.setGpuSelection,
    })),
  );
  const selectable = devices.filter((device) => device.pinnable);
  const selected = store.selectedGpuIds ?? [];
  const canSelect = selectable.length > 0;

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
      retained.length >= 2 ? retained : selectable.map((device) => device.index),
    );
  };

  const toggleDevice = (id: number, checked: boolean) => {
    if (store.parallelismMode === "single") {
      store.setGpuSelection("single", checked ? [id] : []);
      return;
    }
    const next = checked
      ? [...new Set([...selected, id])].sort((a, b) => a - b)
      : selected.filter((selectedId) => selectedId !== id);
    store.setGpuSelection(store.parallelismMode, next);
  };

  return (
    <TabsContent value="hardware" className="mt-3 flex flex-col gap-3">
      <ParamsRow
        label="GPU placement"
        tooltip="Choose automatic selection, one GPU, model sharding, or data-parallel DDP. DDP replicates the model on each selected GPU."
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
            <SelectItem value="auto">Automatic</SelectItem>
            <SelectItem value="single">Single GPU</SelectItem>
            <SelectItem
              value="model_parallel"
              disabled={selectable.length < 2}
            >
              Model sharding
            </SelectItem>
            <SelectItem
              value="ddp"
              disabled={selectable.length < 2 || gpu.backend !== "cuda"}
            >
              DDP (data parallel)
            </SelectItem>
          </SelectContent>
        </Select>
      </ParamsRow>
      <p className="text-xs text-muted-foreground">
        {modeDescription[store.parallelismMode]}
      </p>
      {!canSelect ? (
        <p className="text-xs text-destructive">
          No GPU with a stable physical index is available for explicit selection.
        </p>
      ) : store.parallelismMode === "auto" ? (
        <div className="flex flex-col gap-1 text-xs text-muted-foreground">
          <span>Live inventory:</span>
          {selectable.map((device) => (
            <span key={device.index}>
              GPU {device.index}: {device.name} —{" "}
              {memoryLabel(device.memoryFreeGb, device.memoryTotalGb)}
            </span>
          ))}
        </div>
      ) : (
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
                  {memoryLabel(device.memoryFreeGb, device.memoryTotalGb)}
                  <Checkbox
                    checked={checked}
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
                {store.parallelismMode === "ddp" ? "DDP" : "Model sharding"}{" "}
                requires at least two selected GPUs.
              </p>
            )}
        </div>
      )}
    </TabsContent>
  );
}
