// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CONTEXT_LENGTHS } from "@/config/training";
import {
  useMaxStepsEpochsToggle,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { LoraParamsSection } from "./lora-params-section";
import { TrainingHyperparametersSection } from "./training-hyperparameters-section";
import { useMlxTrainingConfigPolicy } from "./use-mlx-training-config-policy";

export function ParamsSection({
  mode = "advanced",
}: {
  mode?: "simple" | "advanced";
}): ReactElement {
  const t = useT();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      projectName: state.projectName,
      trainingMethod: state.trainingMethod,
      epochs: state.epochs,
      contextLength: state.contextLength,
      learningRate: state.learningRate,
      embeddingLearningRate: state.embeddingLearningRate,
      maxSteps: state.maxSteps,
      saveSteps: state.saveSteps,
      setProjectName: state.setProjectName,
      setEpochs: state.setEpochs,
      setContextLength: state.setContextLength,
      setLearningRate: state.setLearningRate,
      setEmbeddingLearningRate: state.setEmbeddingLearningRate,
      setMaxSteps: state.setMaxSteps,
      setSaveSteps: state.setSaveSteps,
    })),
  );
  useMlxTrainingConfigPolicy();

  const isCpt = store.trainingMethod === "cpt";
  const showAdvanced = mode === "advanced";
  const [contextDraft, setContextDraft] = useState(() => ({
    contextLength: store.contextLength,
    value: String(store.contextLength),
  }));
  const contextInput =
    contextDraft.contextLength === store.contextLength
      ? contextDraft.value
      : String(store.contextLength);
  const setContextInput = (value: string) => {
    setContextDraft({ contextLength: store.contextLength, value });
  };
  const contextAnchorRef = useRef<HTMLDivElement>(null);
  const contextItems = CONTEXT_LENGTHS.map(String);
  const [learningRateDraft, setLearningRateDraft] = useState(() => ({
    learningRate: store.learningRate,
    value: String(store.learningRate),
  }));
  const learningRateInput =
    learningRateDraft.learningRate === store.learningRate
      ? learningRateDraft.value
      : String(store.learningRate);

  const trySetContextLength = (input: string): number | null => {
    const value = Number(input);
    if (Number.isInteger(value) && value > 0) {
      store.setContextLength(value);
      return value;
    }
    return null;
  };

  const { useEpochs, toggleUseEpochs } = useMaxStepsEpochsToggle({
    maxSteps: store.maxSteps,
    epochs: store.epochs,
    saveSteps: store.saveSteps,
    setMaxSteps: store.setMaxSteps,
    setEpochs: store.setEpochs,
    setSaveSteps: store.setSaveSteps,
  });

  const maxStepsSliderMax = Math.max(500, store.maxSteps, 30);
  const epochsSliderMax = Math.max(20, store.epochs, 1);
  const durationKey = useEpochs ? "epochs" : "steps";
  const durationValue = useEpochs ? store.epochs : store.maxSteps;
  const [durationDraft, setDurationDraft] = useState(() => ({
    key: durationKey,
    committedValue: durationValue,
    value: String(durationValue),
  }));
  const durationInput =
    durationDraft.key === durationKey &&
    durationDraft.committedValue === durationValue
      ? durationDraft.value
      : String(durationValue);

  return (
    <div className="min-w-0">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            {t("studio.params.projectName")}
            <span className="text-ui-10 font-normal text-muted-foreground/70">
              {t("studio.params.optional")}
            </span>
          </span>
          <Input
            value={store.projectName || ""}
            onChange={(event) => store.setProjectName(event.target.value)}
            placeholder="customer-support-lora"
            maxLength={80}
          />
          <p className="text-ui-10 text-muted-foreground">
            {t("studio.params.projectNameDescription")}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <div
            key={useEpochs ? "epochs" : "steps"}
            className="flex flex-col gap-2 animate-in fade-in-0 slide-in-from-bottom-1 duration-200"
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                {useEpochs
                  ? t("studio.params.epochs")
                  : t("studio.params.maxSteps")}
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      className="text-foreground/70 hover:text-foreground"
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {useEpochs
                      ? t("studio.params.epochsTooltip")
                      : t("studio.params.maxStepsTooltip")}{" "}
                    <a
                      href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary underline"
                    >
                      {t("studio.params.readMore")}
                    </a>
                  </TooltipContent>
                </Tooltip>
              </span>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={toggleUseEpochs}
                  className="text-xs text-primary underline cursor-pointer"
                >
                  {useEpochs
                    ? t("studio.params.useMaxSteps")
                    : t("studio.params.useEpochs")}
                </button>
                <input
                  type="number"
                  value={durationInput}
                  onChange={(event) => {
                    const raw = event.target.value;
                    setDurationDraft({
                      key: durationKey,
                      committedValue: durationValue,
                      value: raw,
                    });
                    const value = Number(raw);
                    if (raw === "" || !Number.isFinite(value) || value < 1) {
                      return;
                    }
                    if (useEpochs) {
                      store.setEpochs(value);
                    } else {
                      store.setMaxSteps(value);
                    }
                    setDurationDraft({
                      key: durationKey,
                      committedValue: value,
                      value: raw,
                    });
                  }}
                  onBlur={() => {
                    const value = Number(durationInput);
                    if (
                      durationInput === "" ||
                      !Number.isFinite(value) ||
                      value < 1
                    ) {
                      setDurationDraft({
                        key: durationKey,
                        committedValue: durationValue,
                        value: String(durationValue),
                      });
                    }
                  }}
                  min={1}
                  max={useEpochs ? epochsSliderMax : maxStepsSliderMax}
                  step={1}
                  className="w-16 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-ring [&::-webkit-inner-spin-button]:appearance-none"
                />
              </div>
            </div>
            <Slider
              value={[
                useEpochs
                  ? Math.min(epochsSliderMax, Math.max(1, store.epochs))
                  : Math.min(maxStepsSliderMax, Math.max(1, store.maxSteps)),
              ]}
              onValueChange={([value]) =>
                useEpochs ? store.setEpochs(value) : store.setMaxSteps(value)
              }
              min={1}
              max={useEpochs ? epochsSliderMax : maxStepsSliderMax}
              step={1}
            />
            <p className="text-ui-10 text-muted-foreground">
              {useEpochs
                ? t("studio.params.epochsDescription")
                : t("studio.params.maxStepsDescription")}
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            {t("studio.params.contextLength")}
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-foreground/70 hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                {t("studio.params.contextLengthTooltip")}{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  {t("studio.params.readMore")}
                </a>
              </TooltipContent>
            </Tooltip>
          </span>
          <div ref={contextAnchorRef}>
            <Combobox
              items={contextItems}
              filteredItems={contextItems}
              filter={null}
              value={String(store.contextLength)}
              onValueChange={(value) => {
                if (value && trySetContextLength(value)) {
                  setContextInput(value);
                }
              }}
              onInputValueChange={setContextInput}
              itemToStringValue={(id) => Number(id).toLocaleString()}
              autoHighlight={false}
            >
              <ComboboxInput
                placeholder={String(store.contextLength)}
                className="w-full font-mono"
                onBlur={() => {
                  trySetContextLength(contextInput);
                  setContextInput(String(store.contextLength));
                }}
                onKeyDown={(event) => {
                  if (event.key !== "Enter") {
                    return;
                  }
                  const value = trySetContextLength(contextInput);
                  if (value === null) {
                    return;
                  }
                  if (!contextItems.includes(contextInput.trim())) {
                    event.stopPropagation();
                    event.preventDefault();
                  }
                  setContextInput(String(value));
                }}
              />
              <ComboboxContent anchor={contextAnchorRef}>
                <ComboboxEmpty>
                  {t("studio.params.customContextLength")}
                </ComboboxEmpty>
                <ComboboxList className="p-1">
                  {(id: string) => (
                    <ComboboxItem key={id} value={id} className="font-mono">
                      {Number(id).toLocaleString()}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>
          <p className="text-ui-10 text-muted-foreground">
            {t("studio.params.contextLengthDescription")}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            {t("studio.params.learningRate")}
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-foreground/70 hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                {t("studio.params.learningRateTooltip")}{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  {t("studio.params.readMore")}
                </a>
              </TooltipContent>
            </Tooltip>
          </span>
          <Input
            type="number"
            step="0.00001"
            min="0"
            value={learningRateInput}
            onChange={(event) => {
              const input = event.target.value;
              setLearningRateDraft({
                learningRate: store.learningRate,
                value: input,
              });
              const learningRate = Number(input);
              if (
                input !== "" &&
                Number.isFinite(learningRate)
              ) {
                store.setLearningRate(learningRate);
                setLearningRateDraft({ learningRate, value: input });
              }
            }}
            onBlur={() => {
              const learningRate = Number(learningRateInput);
              if (
                learningRateInput === "" ||
                !Number.isFinite(learningRate) ||
                learningRate <= 0
              ) {
                setLearningRateDraft({
                  learningRate: store.learningRate,
                  value: String(store.learningRate),
                });
              }
            }}
            className="w-full font-mono"
          />
          <p className="text-ui-10 text-muted-foreground">
            {t("studio.params.learningRateDescription")}
          </p>
        </div>

        {isCpt && (
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              {t("studio.params.embeddingLearningRate")}
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-foreground/70 hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  {t("studio.params.embeddingLearningRateTooltip")}
                </TooltipContent>
              </Tooltip>
            </span>
            <Input
              type="number"
              step="0.00001"
              min="0"
              max="1"
              placeholder={`auto (${(store.learningRate / 10).toExponential(1)})`}
              value={store.embeddingLearningRate ?? ""}
              onChange={(event) => {
                const raw = event.target.value;
                if (raw === "") {
                  store.setEmbeddingLearningRate(null);
                  return;
                }
                const value = Number(raw);
                store.setEmbeddingLearningRate(
                  Number.isFinite(value) ? value : null,
                );
              }}
              className="w-full font-mono"
            />
            <p className="text-ui-10 text-muted-foreground">
              {t("studio.params.embeddingLearningRateDescription")}
            </p>
          </div>
        )}

        {showAdvanced && <LoraParamsSection />}
        {showAdvanced && (
          <TrainingHyperparametersSection
            useEpochs={useEpochs}
            epochsSliderMax={epochsSliderMax}
          />
        )}
      </div>
    </div>
  );
}
