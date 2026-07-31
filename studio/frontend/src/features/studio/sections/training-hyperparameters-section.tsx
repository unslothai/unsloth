// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedTabsList } from "@/components/segmented-tabs";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { usePlatformStore } from "@/config/env";
import {
  LR_SCHEDULER_OPTIONS,
  MLX_OPTIMIZER_OPTIONS,
  OPTIMIZER_OPTIONS,
} from "@/config/training";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow, ParamsSliderRow } from "./params-section-controls";
import { TrainingMemoryParams } from "./training-memory-params";

type HyperparameterTab = "optimization" | "schedule" | "memory";
type StudioT = ReturnType<typeof useT>;

function formatOptimizerLabel(
  value: string,
  fallback: string,
  t: StudioT,
): string {
  switch (value) {
    case "adamw_8bit":
      return t("studio.params.optimizerOptions.adamw8bit");
    case "paged_adamw_8bit":
      return t("studio.params.optimizerOptions.pagedAdamw8bit");
    case "adamw_bnb_8bit":
      return t("studio.params.optimizerOptions.adamwBnb8bit");
    case "paged_adamw_32bit":
      return t("studio.params.optimizerOptions.pagedAdamw32bit");
    case "adamw_torch":
      return t("studio.params.optimizerOptions.adamwTorch");
    case "adamw_torch_fused":
      return t("studio.params.optimizerOptions.adamwTorchFused");
    default:
      return fallback;
  }
}

function formatSchedulerLabel(
  value: string,
  fallback: string,
  t: StudioT,
): string {
  switch (value) {
    case "linear":
      return t("studio.params.lrSchedulerOptions.linear");
    case "cosine":
      return t("studio.params.lrSchedulerOptions.cosine");
    default:
      return fallback;
  }
}

export function TrainingHyperparametersSection({
  useEpochs,
  epochsSliderMax,
}: {
  useEpochs: boolean;
  epochsSliderMax: number;
}): ReactElement {
  const t = useT();
  const platformDeviceType = usePlatformStore((state) => state.deviceType);
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      epochs: state.epochs,
      optimizerType: state.optimizerType,
      lrSchedulerType: state.lrSchedulerType,
      batchSize: state.batchSize,
      gradientAccumulation: state.gradientAccumulation,
      weightDecay: state.weightDecay,
      warmupSteps: state.warmupSteps,
      saveSteps: state.saveSteps,
      evalSteps: state.evalSteps,
      randomSeed: state.randomSeed,
      setOptimizerType: state.setOptimizerType,
      setLrSchedulerType: state.setLrSchedulerType,
      setBatchSize: state.setBatchSize,
      setGradientAccumulation: state.setGradientAccumulation,
      setWeightDecay: state.setWeightDecay,
      setWarmupSteps: state.setWarmupSteps,
      setEpochs: state.setEpochs,
      setSaveSteps: state.setSaveSteps,
      setEvalSteps: state.setEvalSteps,
      setRandomSeed: state.setRandomSeed,
    })),
  );
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<HyperparameterTab>("optimization");
  const isMac = platformDeviceType === "mac";
  const optimizerOptions = isMac ? MLX_OPTIMIZER_OPTIONS : OPTIMIZER_OPTIONS;
  const isCudaAliasOptimizer = OPTIMIZER_OPTIONS.some(
    (option) => option.value === store.optimizerType,
  );
  const selectedOptimizer =
    isMac && isCudaAliasOptimizer ? "adamw" : store.optimizerType;
  const tabs = [
    { value: "optimization", label: t("studio.params.optimization") },
    { value: "schedule", label: t("studio.params.schedule") },
    { value: "memory", label: t("studio.params.memory") },
  ] as const;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {t("studio.params.trainingHyperparameters")}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
        <Tabs
          value={tab}
          onValueChange={(value) => setTab(value as HyperparameterTab)}
          className="w-full"
        >
          <SegmentedTabsList
            value={tab}
            options={tabs}
            ariaLabel={t("studio.params.trainingHyperparameters")}
            size="compact"
          />

          <TabsContent
            value="optimization"
            className="mt-3 flex flex-col gap-3"
          >
            <ParamsRow
              label={t("studio.params.optimizer")}
              tooltip={
                <>
                  {t(
                    isMac
                      ? "studio.params.optimizerTooltipMlx"
                      : "studio.params.optimizerTooltip",
                  )}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
            >
              <Select
                value={selectedOptimizer}
                onValueChange={store.setOptimizerType}
              >
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {optimizerOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {formatOptimizerLabel(option.value, option.label, t)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </ParamsRow>
            <ParamsRow
              label={t("studio.params.lrScheduler")}
              tooltip={
                <>
                  {t("studio.params.lrSchedulerTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
            >
              <Select
                value={store.lrSchedulerType}
                onValueChange={store.setLrSchedulerType}
              >
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LR_SCHEDULER_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {formatSchedulerLabel(option.value, option.label, t)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </ParamsRow>
            <ParamsSliderRow
              label={t("studio.params.batchSize")}
              tooltip={
                <>
                  {t("studio.params.batchSizeTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
              value={store.batchSize}
              onChange={store.setBatchSize}
              min={1}
              max={32}
              step={1}
            />
            <ParamsSliderRow
              label={t("studio.params.gradAccum")}
              tooltip={
                <>
                  {t("studio.params.gradAccumTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
              value={store.gradientAccumulation}
              onChange={store.setGradientAccumulation}
              min={1}
              max={64}
              step={1}
            />
            <ParamsRow
              label={t("studio.params.weightDecay")}
              tooltip={
                <>
                  {t("studio.params.weightDecayTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
            >
              <Input
                type="number"
                step="0.001"
                value={store.weightDecay}
                onChange={(event) =>
                  store.setWeightDecay(Number(event.target.value))
                }
                className="w-28 font-mono"
              />
            </ParamsRow>
          </TabsContent>

          <TabsContent value="schedule" className="mt-3 flex flex-col gap-3">
            <ParamsSliderRow
              label={t("studio.params.warmupSteps")}
              tooltip={
                <>
                  {t("studio.params.warmupStepsTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
              value={store.warmupSteps}
              onChange={store.setWarmupSteps}
              min={0}
              max={100}
              step={1}
            />
            {!useEpochs && (
              <ParamsSliderRow
                label={t("studio.params.epochs")}
                tooltip={
                  <>
                    {t("studio.params.scheduleEpochsTooltip")}{" "}
                    <a
                      href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline"
                    >
                      {t("studio.params.readMore")}
                    </a>
                  </>
                }
                value={store.epochs}
                onChange={store.setEpochs}
                min={0}
                max={epochsSliderMax}
                step={1}
              />
            )}
            <ParamsRow
              label={t("studio.params.saveSteps")}
              tooltip={
                <>
                  {t("studio.params.saveStepsTooltip")}{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    {t("studio.params.readMore")}
                  </a>
                </>
              }
            >
              <Input
                type="number"
                value={store.saveSteps}
                onChange={(event) =>
                  store.setSaveSteps(Number(event.target.value))
                }
                className="w-28 font-mono"
              />
            </ParamsRow>
            <ParamsRow
              label={t("studio.params.evalSteps")}
              tooltip={t("studio.params.evalStepsTooltip")}
            >
              <Input
                type="number"
                step="0.01"
                min="0.0"
                max="1.0"
                value={store.evalSteps}
                onChange={(event) =>
                  store.setEvalSteps(Number(event.target.value))
                }
                className="w-28 font-mono"
              />
            </ParamsRow>
            <ParamsRow
              label={t("studio.params.seed")}
              tooltip={t("studio.params.seedTooltip")}
            >
              <Input
                type="number"
                value={store.randomSeed}
                onChange={(event) =>
                  store.setRandomSeed(Number(event.target.value))
                }
                className="w-28 font-mono"
              />
            </ParamsRow>
          </TabsContent>

          <TrainingMemoryParams />
        </Tabs>
      </CollapsibleContent>
    </Collapsible>
  );
}
