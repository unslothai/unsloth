// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { SectionCard } from "@/components/section-card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  CONTEXT_LENGTHS,
  LR_SCHEDULER_OPTIONS,
  OPTIMIZER_OPTIONS,
  TARGET_MODULES,
} from "@/config/training";
import { useMaxStepsEpochsToggle, useTrainingConfigStore } from "@/features/training";
import type { GradientCheckpointing } from "@/types/training";
import {
  ArrowDown01Icon,
  InformationCircleIcon,
  Settings04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useEffect, useRef, useState } from "react";

function Row({
  label,
  tooltip,
  children,
}: { label: string; tooltip?: ReactNode; children: ReactNode }): ReactElement {
  return (
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        {tooltip && (
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
            <TooltipContent>{tooltip}</TooltipContent>
          </Tooltip>
        )}
      </span>
      {children}
    </div>
  );
}

function SliderRow({
  label,
  tooltip,
  value,
  onChange,
  min,
  max,
  step,
  format,
}: {
  label: string;
  tooltip?: ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
}): ReactElement {
  return (
    <Row label={label} tooltip={tooltip}>
      <div className="flex items-center gap-3">
        <Slider
          value={[value]}
          onValueChange={([v]) => onChange(v)}
          min={min}
          max={max}
          step={step}
          className="w-32"
        />
        <input
          type="number"
          value={format ? format(value) : value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>
    </Row>
  );
}

export function ParamsSection(): ReactElement {
  const store = useTrainingConfigStore();
  const platformDeviceType = usePlatformStore((s) => s.deviceType);
  const isLora = store.trainingMethod !== "full";
  const showVisionLora = store.isVisionModel && store.isDatasetImage === true;
  const [loraOpen, setLoraOpen] = useState(false);
  const [hyperOpen, setHyperOpen] = useState(false);
  const [ctxInput, setCtxInput] = useState(String(store.contextLength));
  const ctxAnchorRef = useRef<HTMLDivElement>(null);
  const ctxItems = CONTEXT_LENGTHS.map(String);

  // Keep input in sync when the store value changes externally
  // (e.g. model defaults being applied after model selection).
  useEffect(() => {
    setCtxInput(String(store.contextLength));
  }, [store.contextLength]);

  const trySetContextLength = (input: string): number | null => {
    const n = Number(input);
    if (Number.isInteger(n) && n > 0) {
      store.setContextLength(n);
      return n;
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

  return (
    <div data-tour="studio-params" className="min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={Settings04Icon} className="size-5" />}
        title="Parameters"
        description="Configure training hyperparameters"
        accent="orange"
        className={`${(isLora && loraOpen) || hyperOpen
          ? "min-h-studio-config-column"
          : "h-studio-config-column"} duration-150`}
      >
        <div className="flex flex-col gap-4">
          {/* Max Steps / Epochs */}
          <div className="flex flex-col gap-2">
            <div
              key={useEpochs ? "epochs" : "steps"}
              className="flex flex-col gap-2 animate-in fade-in-0 slide-in-from-bottom-1 duration-200"
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  {useEpochs ? "Epochs" : "Max Steps"}
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
                        ? "Number of full passes over the dataset."
                        : "Override total optimizer steps."}{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
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
                    {useEpochs ? "Use Max Steps" : "Use Epochs"}
                  </button>
                  <input
                    type="number"
                    value={useEpochs ? store.epochs : store.maxSteps}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === "") return;

                      const value = Number(raw);
                      if (!Number.isFinite(value) || value < 1) return;

                      if (useEpochs) {
                        store.setEpochs(value);
                      } else {
                        store.setMaxSteps(value);
                      }
                    }}
                    min={1}
                    max={useEpochs ? epochsSliderMax : maxStepsSliderMax}
                    step={1}
                    className="w-16 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>
              <Slider
                value={[
                  useEpochs
                    ? Math.min(epochsSliderMax, Math.max(1, store.epochs))
                    : Math.min(maxStepsSliderMax, Math.max(1, store.maxSteps)),
                ]}
                onValueChange={([v]) =>
                  useEpochs ? store.setEpochs(v) : store.setMaxSteps(v)
                }
                min={1}
                max={useEpochs ? epochsSliderMax : maxStepsSliderMax}
                step={1}
              />
              <p className="text-[10px] text-muted-foreground">
                {useEpochs
                  ? "Each epoch is one full pass over your dataset."
                  : "Limits training to a fixed number of optimizer steps."}
              </p>
            </div>
          </div>

          {/* Context length */}
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              Context Length
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
                  Maximum number of tokens per training sample.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
            <div ref={ctxAnchorRef}>
              <Combobox
                items={ctxItems}
                filteredItems={ctxItems}
                filter={null}
                value={String(store.contextLength)}
                onValueChange={(v) => {
                  if (v && trySetContextLength(v)) {
                    setCtxInput(v);
                  }
                }}
                onInputValueChange={setCtxInput}
                itemToStringValue={(id) => Number(id).toLocaleString()}
                autoHighlight={false}
              >
                <ComboboxInput
                  placeholder={String(store.contextLength)}
                  className="w-full font-mono"
                  onBlur={() => {
                    trySetContextLength(ctxInput);
                    setCtxInput(String(store.contextLength));
                  }}
                  onKeyDown={(e) => {
                    if (e.key !== "Enter") { return; }
                    const n = trySetContextLength(ctxInput);
                    if (n === null) { return; }
                    if (!ctxItems.includes(ctxInput.trim())) {
                      e.stopPropagation();
                      e.preventDefault();
                    }
                    setCtxInput(String(n));
                  }}
                />
                <ComboboxContent anchor={ctxAnchorRef}>
                  <ComboboxEmpty>Enter a custom value</ComboboxEmpty>
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
            <p className="text-[10px] text-muted-foreground">
              Max sequence length for training samples
            </p>
          </div>

          {/* Learning Rate */}
          <div className="flex flex-col gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              Learning Rate
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
                  Step size for weight updates. Lower values train slower but more
                  stably.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
            <Input
              type="number"
              step="0.00001"
              value={store.learningRate}
              onChange={(e) => store.setLearningRate(Number(e.target.value))}
              className="w-full font-mono"
            />
            <p className="text-[10px] text-muted-foreground">
              Recommended: 2e-4 for LoRA, 2e-5 for full fine-tune
            </p>
          </div>

          {/* LoRA Settings */}
          {isLora && (
            <Collapsible open={loraOpen} onOpenChange={setLoraOpen}>
              <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
                <HugeiconsIcon
                  icon={ArrowDown01Icon}
                  className={`size-3.5 transition-transform ${loraOpen ? "rotate-180" : ""}`}
                />
                LoRA Settings
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
                <div className="pt-1.5 flex flex-col gap-4">
                <SliderRow
                  label="Rank"
                  tooltip={
                    <>
                      Dimension of the low-rank matrices. Higher = more capacity.{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </>
                  }
                  value={store.loraRank}
                  onChange={store.setLoraRank}
                  min={4}
                  max={128}
                  step={4}
                />
                <SliderRow
                  label="Alpha"
                  tooltip={
                    <>
                      Scaling factor for LoRA updates. Usually 2x rank.{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </>
                  }
                  value={store.loraAlpha}
                  onChange={store.setLoraAlpha}
                  min={4}
                  max={256}
                  step={4}
                />
                <SliderRow
                  label="Dropout"
                  tooltip={
                    <>
                      Dropout probability for LoRA layers to reduce overfitting.{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline"
                      >
                        Read more
                      </a>
                    </>
                  }
                  value={store.loraDropout}
                  onChange={store.setLoraDropout}
                  min={0}
                  max={0.5}
                  step={0.01}
                  format={(v) => v.toFixed(2)}
                />

                {/* Vision checkboxes */}
                {showVisionLora && (
                  <div className="flex flex-col gap-2 pt-1">
                    {(
                      [
                        [
                          "finetuneVisionLayers",
                          "Vision layers",
                          store.finetuneVisionLayers,
                          store.setFinetuneVisionLayers,
                        ],
                        [
                          "finetuneLanguageLayers",
                          "Language layers",
                          store.finetuneLanguageLayers,
                          store.setFinetuneLanguageLayers,
                        ],
                        [
                          "finetuneAttentionModules",
                          "Attention modules",
                          store.finetuneAttentionModules,
                          store.setFinetuneAttentionModules,
                        ],
                        [
                          "finetuneMLPModules",
                          "MLP modules",
                          store.finetuneMLPModules,
                          store.setFinetuneMLPModules,
                        ],
                      ] as const
                    ).map(([key, label, value, setter]) => (
                      <div key={key} className="flex items-center gap-2">
                        <Checkbox
                          id={key}
                          checked={value as boolean}
                          onCheckedChange={(v) =>
                            (setter as (v: boolean) => void)(!!v)
                          }
                        />
                        <label
                          htmlFor={key}
                          className="text-xs cursor-pointer text-muted-foreground"
                        >
                          {label}
                        </label>
                      </div>
                    ))}
                  </div>
                )}

                {/* Text target modules */}
                {!showVisionLora && (
                  <div className="flex flex-col gap-2 pt-1">
                    <span className="text-xs font-medium text-muted-foreground">
                      Target Modules
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {TARGET_MODULES.map((mod) => {
                        const active = store.targetModules.includes(mod);
                        return (
                          <button
                            key={mod}
                            type="button"
                            onClick={() => {
                              store.setTargetModules(
                                active
                                  ? store.targetModules.filter((m) => m !== mod)
                                  : [...store.targetModules, mod],
                              );
                            }}
                            className={`cursor-pointer rounded-full border px-2.5 py-0.5 text-[11px] font-mono transition-colors ${active
                                ? "border-orange-300 bg-orange-50 text-orange-700 dark:border-orange-700 dark:bg-orange-950 dark:text-orange-300"
                                : "text-muted-foreground hover:bg-muted/50"
                              }`}
                          >
                            {mod}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* LoRA variant */}
                <div className="flex gap-2">
                  {(
                    [
                      {
                        value: "lora",
                        label: "Enable LoRA",
                        desc: "Train with LoRA",
                      },
                      { value: "rslora", label: "RS-LoRA", desc: "Stable Rank" },
                      {
                        value: "loftq",
                        label: "LoftQ",
                        desc: "Memory Efficient",
                      },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => store.setLoraVariant(opt.value)}
                      className={`flex-1 corner-squircle rounded-xl border px-3 py-2 text-left transition-colors cursor-pointer ${store.loraVariant === opt.value
                          ? "border-primary/50 bg-primary/5 ring-1 ring-primary/20"
                          : "border-border hover:border-foreground/20"
                        }`}
                    >
                      <p className="text-xs font-medium">{opt.label}</p>
                      <p className="text-[10px] text-muted-foreground">
                        {opt.desc}
                      </p>
                    </button>
                  ))}
                </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Training Hyperparams */}
          <Collapsible open={hyperOpen} onOpenChange={setHyperOpen}>
            <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                className={`size-3.5 transition-transform ${hyperOpen ? "rotate-180" : ""}`}
              />
              Training Hyperparameters
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
              <Tabs defaultValue="optimization" className="w-full">
                <TabsList className="w-full">
                  <TabsTrigger
                    value="optimization"
                    className="flex-1 !corner-squircle text-xs cursor-pointer"
                  >
                    Optimization
                  </TabsTrigger>
                  <TabsTrigger
                    value="schedule"
                    className="flex-1 text-xs cursor-pointer"
                  >
                    Schedule
                  </TabsTrigger>
                  <TabsTrigger
                    value="memory"
                    className="flex-1 text-xs cursor-pointer"
                  >
                    Memory
                  </TabsTrigger>
                </TabsList>

                <TabsContent
                  value="optimization"
                  className="mt-3 flex flex-col gap-3"
                >
                  <Row
                    label="Optimizer"
                    tooltip={
                      <>
                        Optimization algorithm. 8-bit variants reduce memory usage.
                        Fused is recommended for vision models.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                  >
                    <Select
                      value={store.optimizerType}
                      onValueChange={(v) => store.setOptimizerType(v)}
                    >
                      <SelectTrigger className="w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {OPTIMIZER_OPTIONS.map((opt) => (
                          <SelectItem
                            key={opt.value}
                            value={opt.value}
                          >
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Row>
                  <Row
                    label="LR scheduler"
                    tooltip={
                      <>
                        How the learning rate changes over training. Linear decays
                        steadily; cosine decays in a curve.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                  >
                    <Select
                      value={store.lrSchedulerType}
                      onValueChange={(v) => store.setLrSchedulerType(v)}
                    >
                      <SelectTrigger className="w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {LR_SCHEDULER_OPTIONS.map((opt) => (
                          <SelectItem
                            key={opt.value}
                            value={opt.value}
                          >
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Row>
                  <SliderRow
                    label="Batch Size"
                    tooltip={
                      <>
                        Samples processed per step. Higher uses more VRAM.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                    value={store.batchSize}
                    onChange={store.setBatchSize}
                    min={1}
                    max={32}
                    step={1}
                  />
                  <SliderRow
                    label="Grad Accum"
                    tooltip={
                      <>
                        Simulates larger batch sizes without extra VRAM.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                    value={store.gradientAccumulation}
                    onChange={store.setGradientAccumulation}
                    min={1}
                    max={64}
                    step={1}
                  />
                  <Row
                    label="Weight Decay"
                    tooltip={
                      <>
                        L2 regularization to prevent overfitting.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                  >
                    <Input
                      type="number"
                      step="0.001"
                      value={store.weightDecay}
                      onChange={(e) =>
                        store.setWeightDecay(Number(e.target.value))
                      }
                      className="w-28 font-mono"
                    />
                  </Row>
                </TabsContent>

                <TabsContent
                  value="schedule"
                  className="mt-3 flex flex-col gap-3"
                >
                  <SliderRow
                    label="Warmup Steps"
                    tooltip={
                      <>
                        Gradually increase LR at training start for stability.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
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
                    <SliderRow
                      label="Epochs"
                      tooltip={
                        <>
                          Number of full passes over the dataset. Set 0 to run by
                          max steps.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-primary underline"
                          >
                            Read more
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
                  <Row
                    label="Save Steps"
                    tooltip={
                      <>
                        Save a checkpoint every N steps. 0 to disable.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                  >
                    <Input
                      type="number"
                      value={store.saveSteps}
                      onChange={(e) => store.setSaveSteps(Number(e.target.value))}
                      className="w-28 font-mono"
                    />
                  </Row>
                  <Row
                    label="Eval Steps"
                    tooltip="Fraction of total training steps between evaluations (0-1). Set to 0 to disable evaluation. E.g. 0.01 = evaluate every 1% of steps."
                  >
                    <Input
                      type="number"
                      step="0.01"
                      min="0.0"
                      max="1.0"
                      value={store.evalSteps}
                      onChange={(e) => store.setEvalSteps(Number(e.target.value))}
                      className="w-28 font-mono"
                    />
                  </Row>
                  <Row label="Seed" tooltip="Random seed for reproducibility.">
                    <Input
                      type="number"
                      value={store.randomSeed}
                      onChange={(e) =>
                        store.setRandomSeed(Number(e.target.value))
                      }
                      className="w-28 font-mono"
                    />
                  </Row>
                </TabsContent>

                <TabsContent value="memory" className="mt-3 flex flex-col gap-3">
                  <Row
                    label="Grad Checkpoint"
                    tooltip={
                      <>
                        Trade compute for memory by recomputing activations.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </>
                    }
                  >
                    <Select
                      value={store.gradientCheckpointing}
                      onValueChange={(v) =>
                        store.setGradientCheckpointing(v as GradientCheckpointing)
                      }
                    >
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="true">Standard</SelectItem>
                        {platformDeviceType === "mac" ? (
                          <SelectItem value="mlx">MLX</SelectItem>
                        ) : (
                          <SelectItem value="unsloth">Unsloth</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </Row>
                  {!showVisionLora && !store.isEmbeddingModel && (
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="packing"
                        checked={store.packing}
                        onCheckedChange={(v) => store.setPacking(!!v)}
                      />
                      <label
                        htmlFor="packing"
                        className="text-xs cursor-pointer text-muted-foreground"
                      >
                        Enable packing
                      </label>
                    </div>
                  )}
                  {!store.isEmbeddingModel && (
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="trainOnCompletions"
                        checked={store.trainOnCompletions}
                        onCheckedChange={(v) => store.setTrainOnCompletions(!!v)}
                      />
                      <label
                        htmlFor="trainOnCompletions"
                        className="text-xs cursor-pointer text-muted-foreground"
                      >
                        Assistant completions only
                      </label>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </SectionCard>
    </div>
  );
}
