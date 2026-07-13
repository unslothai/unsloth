// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { ChartContainer } from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type BackendModelConfig,
  buildPortableTrainingConfig,
  parseYamlConfig,
  serializeConfigToYaml,
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingPresets,
  validateTrainingConfig,
} from "@/features/training";
import { UserAssetApiError } from "@/features/user-assets";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  Archive04Icon,
  ChartAverageIcon,
  CleanIcon,
  CloudUploadIcon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef, useState } from "react";
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
  const t = useT();
  const chartConfig = {
    loss: { label: t("studio.charts.loss"), color: "#3b82f6" },
  } satisfies ChartConfig;
  const store = useTrainingConfigStore();
  const presetState = useTrainingPresets();
  const [presetName, setPresetName] = useState("");
  const { isStarting, startError, startTrainingRun } = useTrainingActions();
  const isLoadingModel = store.isLoadingModelDefaults || store.isCheckingVision;
  const isModelCapabilitiesSettled = !!store.selectedModel && !isLoadingModel;
  const isIncompatible =
    isModelCapabilitiesSettled &&
    ((!store.isVisionModel && store.isDatasetImage === true) ||
      (!store.isAudioModel && store.isDatasetAudio === true));
  const configValidation = validateTrainingConfig(store);
  const hasMessage = !!(
    startError ||
    isIncompatible ||
    (!configValidation.ok && configValidation.message)
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  const portableConfig = () => {
    const includeVisionFields =
      store.isVisionModel && store.isDatasetImage !== false;
    const selectedModelLower = (store.selectedModel ?? "").toLowerCase();
    const isDeepseekOcr =
      selectedModelLower.includes("deepseek") &&
      selectedModelLower.includes("ocr");
    return buildPortableTrainingConfig(
      store,
      includeVisionFields,
      includeVisionFields && !isDeepseekOcr,
    );
  };

  const loadPresetConfig = (config: ReturnType<typeof portableConfig>) => {
    store.applyConfigPatch(config as BackendModelConfig);
  };

  const handleLoadPreset = () => {
    if (!presetState.selected) return;
    loadPresetConfig(presetState.selected.config);
    toast.success(t("studio.trainingPresets.title"), {
      description: presetState.selected.name,
    });
  };

  const handleSavePreset = async (saveAs: boolean) => {
    const name =
      presetName.trim() || presetState.selected?.name || "Training preset";
    try {
      const saved = saveAs
        ? await presetState.saveAs(name, portableConfig())
        : await presetState.save(name, portableConfig());
      setPresetName(saved.name);
      toast.success(t("studio.trainingPresets.save"));
    } catch (error) {
      toast.error(t("studio.trainingPresets.saveError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const handleDeletePreset = async () => {
    if (!presetState.selected) return;
    if (!window.confirm(t("studio.trainingPresets.deleteConfirmDescription")))
      return;
    try {
      await presetState.remove();
      setPresetName("");
    } catch (error) {
      toast.error(t("studio.trainingPresets.deleteError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const handleReloadPreset = async () => {
    try {
      const record = await presetState.reloadSelected();
      if (!record) return;
      setPresetName(record.name);
      loadPresetConfig(record.config);
    } catch (error) {
      toast.error(
        t(
          error instanceof UserAssetApiError && error.status === 404
            ? "studio.trainingPresets.reloadUnavailable"
            : "studio.trainingPresets.reloadError",
        ),
        { description: error instanceof Error ? error.message : undefined },
      );
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const config = parseYamlConfig(reader.result as string);
        store.applyConfigPatch(config);
        toast.success(t("studio.training.configLoaded"), {
          description: file.name,
        });
      } catch (err) {
        toast.error(t("studio.training.failedToLoadConfig"), {
          description:
            err instanceof Error
              ? err.message
              : t("studio.training.invalidYamlFile"),
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
    const timestamp = new Date()
      .toISOString()
      .replace(/[:T]/g, "-")
      .slice(0, 19);
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
        className={
          hasMessage ? "min-h-studio-config-column" : "h-studio-config-column"
        }
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
            className="w-full cursor-pointer bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600"
            onClick={() => void startTrainingRun()}
            disabled={
              isStarting ||
              isIncompatible ||
              store.isCheckingDataset ||
              isLoadingModel ||
              !configValidation.ok
            }
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
          {!configValidation.ok &&
            configValidation.message &&
            !isIncompatible && (
              <p className="text-xs text-red-500 leading-relaxed">
                {configValidation.message}
              </p>
            )}

          {/* Upload / Save / Reset */}
          <p className="text-xs text-muted-foreground">
            {t("studio.training.configLabel")}
          </p>
          <div className="space-y-2 rounded-lg border bg-muted/10 p-2">
            <div className="flex gap-2">
              <select
                className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-xs"
                value={presetState.selectedId ?? ""}
                disabled={presetState.loading}
                onChange={(event) => {
                  const id = event.target.value || null;
                  presetState.setSelectedId(id);
                  const preset = presetState.presets.find(
                    (item) => item.id === id,
                  );
                  setPresetName(preset?.name ?? "");
                }}
                aria-label={t("studio.trainingPresets.selectPlaceholder")}
              >
                <option value="">
                  {presetState.loading
                    ? t("studio.trainingPresets.loading")
                    : t("studio.trainingPresets.selectPlaceholder")}
                </option>
                {presetState.presets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.name}
                  </option>
                ))}
              </select>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={!presetState.selected}
                onClick={handleLoadPreset}
              >
                {t("studio.trainingPresets.load")}
              </Button>
            </div>
            <div className="flex gap-2">
              <Input
                className="h-8 min-w-0 flex-1 text-xs"
                value={presetName}
                maxLength={200}
                placeholder={t("studio.trainingPresets.title")}
                onChange={(event) => setPresetName(event.target.value)}
              />
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => void handleSavePreset(false)}
              >
                {t("studio.trainingPresets.save")}
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => void handleSavePreset(true)}
              >
                {t("studio.trainingPresets.saveAs")}
              </Button>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                disabled={!presetState.selected}
                onClick={() => void handleDeletePreset()}
              >
                {t("studio.trainingPresets.delete")}
              </Button>
            </div>
            {presetState.error ? (
              <div className="flex items-center justify-between gap-2 text-xs text-destructive">
                <span>{t("studio.trainingPresets.loadError")}</span>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={() => void presetState.refresh()}
                >
                  {t("dataRecipes.server.retry")}
                </Button>
              </div>
            ) : null}
            {presetState.conflict ? (
              <div className="space-y-2 text-xs text-destructive">
                <p>{t("studio.trainingPresets.conflictDescription")}</p>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => void handleReloadPreset()}
                  >
                    {t("studio.trainingPresets.reload")}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => void handleSavePreset(true)}
                  >
                    {t("studio.trainingPresets.saveAsNew")}
                  </Button>
                </div>
              </div>
            ) : null}
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <Button
                  variant="outline"
                  size="sm"
                  className="cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
                  {t("studio.trainingPresets.importYaml")}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {t("studio.training.uploadConfigTooltip")}
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <Button
                  data-tour="studio-save"
                  variant="outline"
                  size="sm"
                  className="cursor-pointer"
                  onClick={handleSaveConfig}
                >
                  <HugeiconsIcon icon={Archive04Icon} className="size-3.5" />
                  {t("studio.trainingPresets.exportYaml")}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {t("studio.training.saveConfigTooltip")}
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild={true}>
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
              <TooltipContent>
                {t("studio.training.resetConfigTooltip")}
              </TooltipContent>
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
