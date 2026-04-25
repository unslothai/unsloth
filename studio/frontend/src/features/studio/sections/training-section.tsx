// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { ChartContainer } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  parseYamlConfig,
  serializeConfigToYaml,
  useTrainingActions,
  useTrainingConfigStore,
  validateTrainingConfig,
} from "@/features/training";
import { useI18n } from "@/features/i18n";
import {
  Archive04Icon,
  ChartAverageIcon,
  CleanIcon,
  CloudUploadIcon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef } from "react";
import { toast } from "sonner";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

const placeholderData = [
  { step: 0, loss: 2.5 },
  { step: 10, loss: 2.1 },
  { step: 20, loss: 1.7 },
  { step: 30, loss: 1.3 },
  { step: 40, loss: 1.0 },
  { step: 50, loss: 0.8 },
];

export function TrainingSection() {
  const { t } = useI18n();
  const store = useTrainingConfigStore();
  const { isStarting, startError, startTrainingRun } = useTrainingActions();
  const isLoadingModel = store.isLoadingModelDefaults || store.isCheckingVision;
  const isModelCapabilitiesSettled = !!store.selectedModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled &&
    ((!store.isVisionModel && store.isDatasetImage === true) ||
      (!store.isAudioModel && store.isDatasetAudio === true));
  const configValidation = validateTrainingConfig(store);
  const hasMessage = !!(startError || isIncompatible || (!configValidation.ok && configValidation.message));
  const fileInputRef = useRef<HTMLInputElement>(null);

  const chartConfig = {
    loss: { label: t("studio.training.metric.loss"), color: "#3b82f6" },
  } satisfies ChartConfig;

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const config = parseYamlConfig(reader.result as string);
        store.applyConfigPatch(config);
        toast.success(t("studio.training.toast.configLoaded"), { description: file.name });
      } catch (err) {
        toast.error(t("studio.training.toast.configLoadFailed"), {
          description:
            err instanceof Error ? err.message : t("studio.training.toast.invalidYaml"),
        });
      }
    };
    reader.onerror = () => {
      toast.error(t("studio.training.toast.readFailed"));
    };
    reader.readAsText(file);
  };

  const handleSaveConfig = () => {
    const yamlStr = serializeConfigToYaml(store, store.isVisionModel);
    const blob = new Blob([yamlStr], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;

    const model = (store.selectedModel ?? "model").split("/").pop();
    const method = store.trainingMethod ?? "qlora";
    const dataset = (store.dataset ?? "dataset").split("/").pop();
    const timestamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
    a.download = `${model}_${method}_${dataset}_${timestamp}.yaml`;

    a.click();
    URL.revokeObjectURL(url);
  };

  const handleResetConfig = () => {
    store.resetToModelDefaults();
    toast.success(t("studio.training.toast.resetDefaults"));
  };

  return (
    <div data-tour="studio-training" className="min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
        title={t("studio.training.title")}
        description={t("studio.training.description")}
        accent="blue"
        className={hasMessage ? "min-h-studio-config-column" : "h-studio-config-column"}
      >
        <div className="flex flex-col gap-4">
        {/* Loss chart */}
        <div className="relative  ">
          <ChartContainer
            config={chartConfig}
            className="h-[180px] w-full relative right-8 blur"
          >
            <LineChart data={placeholderData} accessibilityLayer={true}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="step"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize={10}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-loss)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
            <HugeiconsIcon
              icon={ChartAverageIcon}
              className="size-5 text-muted-foreground/50"
            />
            <p className="text-sm font-medium text-muted-foreground">
              {t("studio.training.chart.noData")}
            </p>
            <p className="text-xs text-muted-foreground/60">
              {t("studio.training.chart.startHint")}
            </p>
          </div>
        </div>

        {/* Start/Stop */}
        <Button
          data-tour="studio-start"
          className="w-full cursor-pointer bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600"
          onClick={() => void startTrainingRun()}
          disabled={isStarting || isIncompatible || store.isCheckingDataset || isLoadingModel || !configValidation.ok}
        >
          <HugeiconsIcon icon={Rocket01Icon} className="size-4" />
          {isStarting
            ? t("studio.training.starting")
            : isLoadingModel
              ? t("studio.training.loadingModel")
              : store.isCheckingDataset
                ? t("studio.training.checkingDataset")
                : t("studio.training.start")}
        </Button>
        {startError && (
          <p className="text-xs text-red-500 leading-relaxed">{startError}</p>
        )}
        {isIncompatible && (
          <p className="text-xs text-red-500 leading-relaxed">
            {!store.isAudioModel && store.isDatasetAudio === true
              ? t("studio.training.error.audioUnsupported")
              : t("studio.training.error.multimodalUnsupported")}
          </p>
        )}
        {!configValidation.ok && configValidation.message && !isIncompatible && (
          <p className="text-xs text-red-500 leading-relaxed">{configValidation.message}</p>
        )}

        {/* Upload / Save / Reset */}
        <p className="text-xs text-muted-foreground">{t("studio.training.config.title")}</p>
        <div className="grid grid-cols-3 gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
                {t("studio.training.config.upload")}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.training.config.uploadHint")}</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                data-tour="studio-save"
                variant="outline"
                size="sm"
                className="cursor-pointer"
                onClick={handleSaveConfig}
              >
                <HugeiconsIcon icon={Archive04Icon} className="size-3.5" />
                {t("studio.training.config.save")}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.training.config.saveHint")}</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="cursor-pointer"
                onClick={handleResetConfig}
                disabled={!store.selectedModel}
              >
                <HugeiconsIcon icon={CleanIcon} className="size-3.5" />
                {t("studio.training.config.reset")}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.training.config.resetHint")}</TooltipContent>
          </Tooltip>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".yaml,.yml"
          className="hidden"
          onChange={handleFileUpload}
        />
        </div>
      </SectionCard>
    </div>
  );
}
