// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useTrainingConfigStore } from "@/features/training";
import { useHardwareInfo } from "@/hooks";
import { isAdapterMethod } from "@/types/training";
import { GpuIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";

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
    <div className="grid grid-cols-2 gap-4">
      <Card size="sm" className="rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            System
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="size-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <HugeiconsIcon icon={GpuIcon} className="size-4 text-primary" />
            </div>
            <div className="flex flex-col flex-1">
              <span className="text-xs text-muted-foreground">GPU</span>
              <span className="text-sm font-medium">{hw.gpuName ?? "—"}</span>
            </div>
            <Badge variant="secondary">{hw.vramTotalGb != null ? `${hw.vramTotalGb} GB` : "—"}</Badge>
          </div>
          <Separator />
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">unsloth</span>
              <span className="font-mono text-xs">{hw.unsloth ?? "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">torch</span>
              <span className="font-mono text-xs">{hw.torch ?? "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">transformers</span>
              <span className="font-mono text-xs">
                {hw.transformers ?? "—"}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">Model</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium truncate">
              {selectedModel ?? "—"}
            </span>
          </div>
          <Separator />
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Type</span>
            <span className="capitalize">{modelType}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Method</span>
            <span className="uppercase">{trainingMethod}</span>
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            Dataset
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="flex flex-col flex-1">
              <span className="text-sm font-medium truncate">
                {datasetName ?? "—"}
              </span>
            </div>
          </div>
          <Separator />
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Source</span>
            <span className="capitalize">{datasetSource}</span>
          </div>
          {datasetSubset && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Subset</span>
              <span className="font-mono text-xs">{datasetSubset}</span>
            </div>
          )}
          {datasetSplit && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Split</span>
              <span className="font-mono text-xs">{datasetSplit}</span>
            </div>
          )}
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Format</span>
            <span className="capitalize">{datasetFormat}</span>
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            Hyperparameters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Epochs</span>
              <span className="font-mono text-xs">{epochs}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Context</span>
              <span className="font-mono text-xs">
                {contextLength.toLocaleString()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">LR</span>
              <span className="font-mono text-xs">{learningRate}</span>
            </div>
            {showLoraParams && (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Rank</span>
                  <span className="font-mono text-xs">{loraRank}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Alpha</span>
                  <span className="font-mono text-xs">{loraAlpha}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Dropout</span>
                  <span className="font-mono text-xs">{loraDropout}</span>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
