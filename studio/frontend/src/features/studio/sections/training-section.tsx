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
import {
  Archive04Icon,
  ChartAverageIcon,
  CleanIcon,
  CloudUploadIcon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef } from "react";
import { toast } from "@/lib/toast";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { useT } from "@/i18n";

const placeholderData = [
  { step: 0, loss: 2.5 },
  { step: 10, loss: 2.1 },
  { step: 20, loss: 1.7 },
  { step: 30, loss: 1.3 },
  { step: 40, loss: 1.0 },
  { step: 50, loss: 0.8 },
];

export function TrainingSection() {
  const t = useT();
  const chartConfig = {
    loss: { label: t("studio.charts.loss"), color: "#3b82f6" },
  } satisfies ChartConfig;
  const store = useTrainingConfigStore();
  const { isStarting, startError, startTrainingRun } = useTrainingActions();
  const isLoadingModel = store.isLoadingModelDefaults || store.isCheckingVision;
  const isModelCapabilitiesSettled = !!store.selectedModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled &&
    ((!store.isVisionModel && store.isDatasetImage === true) ||
      (!store.isAudioModel && store.isDatasetAudio === true));
  const configValidation = validateTrainingConfig(store);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const config = parseYamlConfig(reader.result as string);
        store.applyConfigPatch(config);
        toast.success(t("studio.training.configLoaded"), { description: file.name });
      } catch (err) {
        toast.error(t("studio.training.failedToLoadConfig"), {
          description:
            err instanceof Error ? err.message : t("studio.training.invalidYamlFile"),
        });
      }
    };
    reader.onerror = () => {
      toast.error(t("studio.training.failedToReadFile"));
    };
    reader.readAsText(file);
  };

  const handleSaveConfig = () => {
    // isDatasetImage is null before a dataset check completes, after edits, and
    // on import. Treat all three as "save it" so the user's choice isn't dropped
    // while the type is unconfirmed. Only a confirmed text-only dataset
    // (=== false) suppresses the vision fields.
    const includeVisionFields =
      store.isVisionModel && store.isDatasetImage !== false;
    // DeepSeek OCR ignores vision_image_size; don't emit it, or a later import
    // on a non-DeepSeek model would activate the stale value.
    const selectedModelLower = (store.selectedModel ?? "").toLowerCase();
    const isDeepseekOcr =
      selectedModelLower.includes("deepseek") &&
      selectedModelLower.includes("ocr");
    const yamlStr = serializeConfigToYaml(
      store,
      includeVisionFields,
      includeVisionFields && !isDeepseekOcr,
    );
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
    toast.success(t("studio.training.parametersReset"));
  };

  return (
    <div data-tour="studio-training" className="min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
        title={t("studio.training.title")}
        description={t("studio.training.description")}
        accent="blue"
        className="min-h-studio-config-column"
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
                fontSize="calc(10px * var(--ui-font-scale, 1))"
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                fontSize="calc(10px * var(--ui-font-scale, 1))"
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
              {t("studio.training.chartNoDataTitle")}
            </p>
            <p className="text-xs text-muted-foreground/60">
              {t("studio.training.chartNoDataDescription")}
            </p>
          </div>
        </div>

        {/* Start/Stop */}
        <Button
          data-tour="studio-start"
          className="w-full cursor-pointer bg-primary text-primary-foreground hover:bg-primary/90"
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
                : t("studio.training.startTraining")}
        </Button>
        {startError && (
          <p className="text-xs text-red-500 leading-relaxed">{startError}</p>
        )}
        {isIncompatible && (
          <p className="text-xs text-red-500 leading-relaxed">
            {!store.isAudioModel && store.isDatasetAudio === true
              ? t("studio.training.audioIncompatible")
              : t("studio.training.visionIncompatible")}
          </p>
        )}
        {!configValidation.ok && configValidation.message && !isIncompatible && (
          <p className="text-xs text-red-500 leading-relaxed">{configValidation.message}</p>
        )}

        {/* Upload / Save / Reset */}
        <p className="text-xs text-muted-foreground">{t("studio.training.configLabel")}</p>
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
                {t("studio.training.upload")}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.training.uploadConfigTooltip")}</TooltipContent>
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
                {t("studio.training.save")}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.training.saveConfigTooltip")}</TooltipContent>
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
                {t("studio.training.reset")}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("studio.training.resetConfigTooltip")}</TooltipContent>
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
