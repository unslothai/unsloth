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
import { toast } from "sonner";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

const chartConfig = {
  loss: { label: "Loss", color: "#3b82f6" },
} satisfies ChartConfig;

const placeholderData = [
  { step: 0, loss: 2.5 },
  { step: 10, loss: 2.1 },
  { step: 20, loss: 1.7 },
  { step: 30, loss: 1.3 },
  { step: 40, loss: 1.0 },
  { step: 50, loss: 0.8 },
];

export function TrainingSection() {
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

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const config = parseYamlConfig(reader.result as string);
        store.applyConfigPatch(config);
        toast.success("Config loaded", { description: file.name });
      } catch (err) {
        toast.error("Failed to load config", {
          description:
            err instanceof Error ? err.message : "Invalid YAML file",
        });
      }
    };
    reader.onerror = () => {
      toast.error("Failed to read file");
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
    toast.success("Parameters reset to model defaults");
  };

  return (
    <div data-tour="studio-training" className="min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
        title="Training"
        description="Monitor and control training"
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
              No training data yet
            </p>
            <p className="text-xs text-muted-foreground/60">
              Start training to see loss progress
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
          {isStarting ? "Starting..." : isLoadingModel ? "Loading model..." : store.isCheckingDataset ? "Checking dataset..." : "Start Training"}
        </Button>
        {startError && (
          <p className="text-xs text-red-500 leading-relaxed">{startError}</p>
        )}
        {isIncompatible && (
          <p className="text-xs text-red-500 leading-relaxed">
            {!store.isAudioModel && store.isDatasetAudio === true
              ? "This model does not support audio. Switch to an audio-capable model or choose a non-audio dataset."
              : "Text model is not compatible with a multimodal dataset. Switch to a vision model or choose a text-only dataset."}
          </p>
        )}
        {!configValidation.ok && configValidation.message && !isIncompatible && (
          <p className="text-xs text-red-500 leading-relaxed">{configValidation.message}</p>
        )}

        {/* Upload / Save / Reset */}
        <p className="text-xs text-muted-foreground">Training Config</p>
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
                Upload
              </Button>
            </TooltipTrigger>
            <TooltipContent>Load a saved YAML config</TooltipContent>
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
                Save
              </Button>
            </TooltipTrigger>
            <TooltipContent>Download current config as YAML</TooltipContent>
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
                Reset
              </Button>
            </TooltipTrigger>
            <TooltipContent>Reset to model defaults</TooltipContent>
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
