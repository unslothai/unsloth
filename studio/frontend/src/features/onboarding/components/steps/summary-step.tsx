// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useI18n } from "@/features/i18n";
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
  const { t } = useI18n();
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
            {t("onboarding.summary.system")}
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
          <CardTitle className="text-sm text-muted-foreground">{t("onboarding.summary.model")}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10">
              <HugeiconsIcon icon={ChipIcon} className="size-4 text-emerald-600" />
            </div>
            <div className="flex flex-1 flex-col overflow-hidden">
              <span className="text-xs text-muted-foreground">{t("onboarding.summary.model")}</span>
              <span className="truncate text-sm font-medium">{selectedModel ?? "---"}</span>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="space-y-1 text-sm">
            <Row label={t("onboarding.summary.type")} value={modelType} capitalize />
            <Row label={t("onboarding.summary.method")} value={trainingMethod === "qlora" ? "QLoRA" : trainingMethod === "lora" ? "LoRA" : t("studio.training.method.full")} />
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="flex flex-col rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            {t("onboarding.summary.dataset")}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-indigo-500/10">
              <HugeiconsIcon icon={Database02Icon} className="size-4 text-indigo-600" />
            </div>
            <div className="flex flex-1 flex-col overflow-hidden">
              <span className="text-xs text-muted-foreground">{t("onboarding.summary.dataset")}</span>
              <span className="truncate text-sm font-medium">{datasetName ?? "---"}</span>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="space-y-1 text-sm">
            <Row label={t("onboarding.summary.source")} value={datasetSource} capitalize />
            {datasetSubset && (
              <Row label={t("onboarding.summary.subset")} value={datasetSubset} mono />
            )}
            {datasetSplit && (
              <Row label={t("onboarding.summary.split")} value={datasetSplit} mono />
            )}
            <Row label={t("onboarding.summary.format")} value={datasetFormat} capitalize />
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="flex flex-col rounded-2xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            {t("onboarding.summary.hyperparameters")}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-orange-500/10">
              <HugeiconsIcon icon={Settings04Icon} className="size-4 text-orange-600" />
            </div>
            <div className="flex flex-1 flex-col">
              <span className="text-xs text-muted-foreground">{t("onboarding.summary.training")}</span>
              <span className="text-sm font-medium">
                {trainingMethod === "qlora" ? "QLoRA" : trainingMethod === "lora" ? "LoRA" : t("studio.training.method.full")}
              </span>
            </div>
          </div>
          <Separator className="my-2" />
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
            <Row label={t("onboarding.hyper.epochs")} value={epochs} mono />
            <Row label={t("onboarding.summary.context")} value={contextLength.toLocaleString()} mono />
            <Row label="LR" value={learningRate.toExponential()} mono />
            {showLoraParams && (
              <>
                <Row label={t("onboarding.hyper.rank")} value={loraRank} mono />
                <Row label={t("onboarding.hyper.alpha")} value={loraAlpha} mono />
                <Row label={t("onboarding.hyper.dropout")} value={loraDropout} mono />
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
