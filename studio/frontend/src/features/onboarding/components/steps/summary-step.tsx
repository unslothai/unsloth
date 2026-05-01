// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useTrainingConfigStore } from "@/features/training";
import { useHardwareInfo } from "@/hooks";
import { isAdapterMethod } from "@/types/training";
import { ChipIcon, Database02Icon, GpuIcon, Settings04Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";

function Row({
  label,
  value,
  mono,
  capitalize,
  uppercase,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
  capitalize?: boolean;
  uppercase?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={
          mono
            ? "font-mono text-xs"
            : capitalize
              ? "capitalize"
              : uppercase
                ? "uppercase"
                : undefined
        }
      >
        {value}
      </span>
    </div>
  );
}

export function SummaryStep() {
  const hw = useHardwareInfo();
  const {
    modelType,
    selectedModel,
    trainingMethod,
    datasetSource,
    datasetFormat,
    dataset,
    datasetSubset,
    datasetSplit,
    uploadedFile,
    epochs,
    contextLength,
    learningRate,
    loraRank,
    loraAlpha,
    loraDropout,
  } = useTrainingConfigStore(
    useShallow(
      ({
        modelType,
        selectedModel,
        trainingMethod,
        datasetSource,
        datasetFormat,
        dataset,
        datasetSubset,
        datasetSplit,
        uploadedFile,
        epochs,
        contextLength,
        learningRate,
        loraRank,
        loraAlpha,
        loraDropout,
      }) => ({
        modelType,
        selectedModel,
        trainingMethod,
        datasetSource,
        datasetFormat,
        dataset,
        datasetSubset,
        datasetSplit,
        uploadedFile,
        epochs,
        contextLength,
        learningRate,
        loraRank,
        loraAlpha,
        loraDropout,
      }),
    ),
  );

  const showLoraParams = isAdapterMethod(trainingMethod);
  const datasetName = datasetSource === "upload" ? uploadedFile : dataset;

  return (
    <div className="grid grid-cols-2 gap-3">
      <Card size="sm" className="flex flex-col rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            System
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
              <HugeiconsIcon icon={GpuIcon} className="size-4 text-emerald-600" />
            </div>
            <div className="flex flex-1 flex-col">
              <span className="text-xs text-muted-foreground">GPU</span>
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">{hw.gpuName ?? "---"}</span>
                <Badge variant="secondary">{hw.vramTotalGb != null ? `${hw.vramTotalGb} GB` : "---"}</Badge>
              </div>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="space-y-1 text-sm">
            <Row label="unsloth" value={hw.unsloth ?? "---"} mono />
            <Row label="torch" value={hw.torch ?? "---"} mono />
            <Row label="transformers" value={hw.transformers ?? "---"} mono />
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="flex flex-col rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">Model</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
              <HugeiconsIcon icon={ChipIcon} className="size-4 text-emerald-600" />
            </div>
            <div className="flex flex-1 flex-col overflow-hidden">
              <span className="text-xs text-muted-foreground">Model</span>
              <span className="truncate text-sm font-medium">{selectedModel ?? "---"}</span>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="space-y-1 text-sm">
            <Row label="Type" value={modelType} capitalize />
            <Row label="Method" value={trainingMethod === "qlora" ? "QLoRA" : trainingMethod === "lora" ? "LoRA" : "Full"} />
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="flex flex-col rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            Dataset
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-indigo-500/10">
              <HugeiconsIcon icon={Database02Icon} className="size-4 text-indigo-600" />
            </div>
            <div className="flex flex-1 flex-col overflow-hidden">
              <span className="text-xs text-muted-foreground">Dataset</span>
              <span className="truncate text-sm font-medium">{datasetName ?? "---"}</span>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="space-y-1 text-sm">
            <Row label="Source" value={datasetSource} capitalize />
            {datasetSubset && (
              <Row label="Subset" value={datasetSubset} mono />
            )}
            {datasetSplit && (
              <Row label="Split" value={datasetSplit} mono />
            )}
            <Row label="Format" value={datasetFormat} capitalize />
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="flex flex-col rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            Hyperparameters
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-orange-500/10">
              <HugeiconsIcon icon={Settings04Icon} className="size-4 text-orange-600" />
            </div>
            <div className="flex flex-1 flex-col">
              <span className="text-xs text-muted-foreground">Training</span>
              <span className="text-sm font-medium">
                {trainingMethod === "qlora" ? "QLoRA" : trainingMethod === "lora" ? "LoRA" : "Full"}
              </span>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
            <Row label="Epochs" value={epochs} mono />
            <Row label="Context" value={contextLength.toLocaleString()} mono />
            <Row label="LR" value={learningRate.toExponential()} mono />
            {showLoraParams && (
              <>
                <Row label="Rank" value={loraRank} mono />
                <Row label="Alpha" value={loraAlpha} mono />
                <Row label="Dropout" value={loraDropout} mono />
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
