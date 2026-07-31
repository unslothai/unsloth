// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  CONTEXT_LENGTHS,
  CPT_TARGET_MODULES,
  LR_SCHEDULER_OPTIONS,
  MLX_OPTIMIZER_OPTIONS,
  OPTIMIZER_OPTIONS,
  TARGET_MODULES,
} from "@/config/training";
import {
  isRawTextDatasetFormat,
  useMaxStepsEpochsToggle,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { isAdapterMethod } from "@/types/training";
import type { GradientCheckpointing } from "@/types/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";

type StudioT = ReturnType<typeof useT>;
type HyperparameterTab = "optimization" | "schedule" | "memory";

function selectableOptionStateClassName(selected: boolean): string {
  return selected
    ? "border-ring-strong bg-primary/5"
    : "border-border hover:border-foreground/20";
}

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
          className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-ring [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>
    </Row>
  );
}

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

export function ParamsSection({
  mode = "advanced",
}: {
  mode?: "simple" | "advanced";
}): ReactElement {
  const t = useT();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      selectedModel: state.selectedModel,
      projectName: state.projectName,
      trainingMethod: state.trainingMethod,
      datasetFormat: state.datasetFormat,
      datasetStreaming: state.datasetStreaming,
      epochs: state.epochs,
      contextLength: state.contextLength,
      learningRate: state.learningRate,
      embeddingLearningRate: state.embeddingLearningRate,
      optimizerType: state.optimizerType,
      lrSchedulerType: state.lrSchedulerType,
      loraRank: state.loraRank,
      loraAlpha: state.loraAlpha,
      loraDropout: state.loraDropout,
      loraVariant: state.loraVariant,
      batchSize: state.batchSize,
      gradientAccumulation: state.gradientAccumulation,
      weightDecay: state.weightDecay,
      warmupSteps: state.warmupSteps,
      maxSteps: state.maxSteps,
      saveSteps: state.saveSteps,
      evalSteps: state.evalSteps,
      packing: state.packing,
      trainOnCompletions: state.trainOnCompletions,
      gradientCheckpointing: state.gradientCheckpointing,
      randomSeed: state.randomSeed,
      isVisionModel: state.isVisionModel,
      isEmbeddingModel: state.isEmbeddingModel,
      isDatasetImage: state.isDatasetImage,
      finetuneVisionLayers: state.finetuneVisionLayers,
      finetuneLanguageLayers: state.finetuneLanguageLayers,
      finetuneAttentionModules: state.finetuneAttentionModules,
      finetuneMLPModules: state.finetuneMLPModules,
      targetModules: state.targetModules,
      visionImageSize: state.visionImageSize,
      setProjectName: state.setProjectName,
      setEpochs: state.setEpochs,
      setContextLength: state.setContextLength,
      setVisionImageSize: state.setVisionImageSize,
      setLearningRate: state.setLearningRate,
      setEmbeddingLearningRate: state.setEmbeddingLearningRate,
      setOptimizerType: state.setOptimizerType,
      setLrSchedulerType: state.setLrSchedulerType,
      setLoraRank: state.setLoraRank,
      setLoraAlpha: state.setLoraAlpha,
      setLoraDropout: state.setLoraDropout,
      setLoraVariant: state.setLoraVariant,
      setBatchSize: state.setBatchSize,
      setGradientAccumulation: state.setGradientAccumulation,
      setWeightDecay: state.setWeightDecay,
      setWarmupSteps: state.setWarmupSteps,
      setMaxSteps: state.setMaxSteps,
      setSaveSteps: state.setSaveSteps,
      setEvalSteps: state.setEvalSteps,
      setPacking: state.setPacking,
      setTrainOnCompletions: state.setTrainOnCompletions,
      setGradientCheckpointing: state.setGradientCheckpointing,
      setRandomSeed: state.setRandomSeed,
      setFinetuneVisionLayers: state.setFinetuneVisionLayers,
      setFinetuneLanguageLayers: state.setFinetuneLanguageLayers,
      setFinetuneAttentionModules: state.setFinetuneAttentionModules,
      setFinetuneMLPModules: state.setFinetuneMLPModules,
      setTargetModules: state.setTargetModules,
    })),
  );
  const platformDeviceType = usePlatformStore((s) => s.deviceType);
  const isLora = isAdapterMethod(store.trainingMethod);
  const isCpt = store.trainingMethod === "cpt";
  const isRawText = isRawTextDatasetFormat(store.datasetFormat);
  const showVisionLora = store.isVisionModel && store.isDatasetImage === true;
  // DeepSeek OCR uses a coupled preset; backend ignores user image size.
  const _selectedModelLower = (store.selectedModel ?? "").toLowerCase();
  const isDeepseekOcr =
    _selectedModelLower.includes("deepseek") &&
    _selectedModelLower.includes("ocr");
  const showVisionImageSize = showVisionLora && !isDeepseekOcr;
  const [loraOpen, setLoraOpen] = useState(false);
  const [hyperOpen, setHyperOpen] = useState(false);
  const [hyperparameterTab, setHyperparameterTab] =
    useState<HyperparameterTab>("optimization");
  const showAdvanced = mode === "advanced";
  const hyperparameterTabs = [
    {
      value: "optimization",
      label: t("studio.params.optimization"),
    },
    { value: "schedule", label: t("studio.params.schedule") },
    { value: "memory", label: t("studio.params.memory") },
  ] as const;
  const [ctxDraft, setCtxDraft] = useState(() => ({
    contextLength: store.contextLength,
    value: String(store.contextLength),
  }));
  const ctxInput =
    ctxDraft.contextLength === store.contextLength
      ? ctxDraft.value
      : String(store.contextLength);
  const setCtxInput = (value: string) => {
    setCtxDraft({ contextLength: store.contextLength, value });
  };
  const ctxAnchorRef = useRef<HTMLDivElement>(null);
  const ctxItems = CONTEXT_LENGTHS.map(String);
  // Backend validator allows [256, 2048]; offer the full span.
  const visionImageSizePresets = [256, 384, 512, 768, 1024, 1536, 2048];

  // Apple Silicon (MLX) supports a different optimizer set than the CUDA list.
  const isMac = platformDeviceType === "mac";
  const optimizerOptions = isMac ? MLX_OPTIMIZER_OPTIONS : OPTIMIZER_OPTIONS;

  // On Mac the MLX backend remaps CUDA optimizers to AdamW, so label those as
  // AdamW; other values (MLX or imported) show as-is. Non-Mac unchanged.
  const isCudaAliasOptimizer = OPTIMIZER_OPTIONS.some(
    (o) => o.value === store.optimizerType,
  );
  const selectedOptimizer =
    isMac && isCudaAliasOptimizer ? "adamw" : store.optimizerType;

  // LoftQ and DoRA are not supported on MLX (the backend rejects both), so
  // clear a stale selection to lora on Apple Silicon -- whether persisted,
  // applied from a model default, or imported -- so the backend never
  // receives it.
  useEffect(() => {
    if (
      isMac &&
      (store.loraVariant === "loftq" || store.loraVariant === "dora")
    ) {
      useTrainingConfigStore.setState({ loraVariant: "lora" });
    }
  }, [isMac, store.loraVariant]);

  // Packing is unsupported on MLX; clear it on Apple Silicon (checkbox disabled).
  useEffect(() => {
    if (isMac && store.packing) {
      useTrainingConfigStore.setState({ packing: false });
    }
  }, [isMac, store.packing]);

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

        {/* Max Steps / Epochs */}
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
              onValueChange={([v]) =>
                useEpochs ? store.setEpochs(v) : store.setMaxSteps(v)
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

        {/* Context length */}
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
                  if (e.key !== "Enter") {
                    return;
                  }
                  const n = trySetContextLength(ctxInput);
                  if (n === null) {
                    return;
                  }
                  if (!ctxItems.includes(ctxInput.trim())) {
                    e.stopPropagation();
                    e.preventDefault();
                  }
                  setCtxInput(String(n));
                }}
              />
              <ComboboxContent anchor={ctxAnchorRef}>
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

        {/* Learning Rate */}
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
            value={store.learningRate}
            onChange={(e) => store.setLearningRate(Number(e.target.value))}
            className="w-full font-mono"
          />
          <p className="text-ui-10 text-muted-foreground">
            {t("studio.params.learningRateDescription")}
          </p>
        </div>

        {/* Embedding Learning Rate (CPT only) */}
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
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "") {
                  store.setEmbeddingLearningRate(null);
                  return;
                }
                const n = Number(raw);
                store.setEmbeddingLearningRate(Number.isFinite(n) ? n : null);
              }}
              className="w-full font-mono"
            />
            <p className="text-ui-10 text-muted-foreground">
              {t("studio.params.embeddingLearningRateDescription")}
            </p>
          </div>
        )}

        {/* LoRA Settings */}
        {showAdvanced && isLora && (
          <Collapsible open={loraOpen} onOpenChange={setLoraOpen}>
            <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                className={`size-3.5 transition-transform ${loraOpen ? "rotate-180" : ""}`}
              />
              {t("studio.params.loraSettings")}
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
              <div className="pt-1.5 flex flex-col gap-4">
                <SliderRow
                  label={t("studio.params.rank")}
                  tooltip={
                    <>
                      {t("studio.params.rankTooltip")}{" "}
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
                  value={store.loraRank}
                  onChange={store.setLoraRank}
                  min={4}
                  max={128}
                  step={4}
                />
                <SliderRow
                  label={t("studio.params.alpha")}
                  tooltip={
                    <>
                      {t("studio.params.alphaTooltip")}{" "}
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
                  value={store.loraAlpha}
                  onChange={store.setLoraAlpha}
                  min={4}
                  max={256}
                  step={4}
                />
                <SliderRow
                  label={t("studio.params.dropout")}
                  tooltip={
                    <>
                      {t("studio.params.dropoutTooltip")}{" "}
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
                          t("studio.params.visionLayers"),
                          store.finetuneVisionLayers,
                          store.setFinetuneVisionLayers,
                        ],
                        [
                          "finetuneLanguageLayers",
                          t("studio.params.languageLayers"),
                          store.finetuneLanguageLayers,
                          store.setFinetuneLanguageLayers,
                        ],
                        [
                          "finetuneAttentionModules",
                          t("studio.params.attentionModules"),
                          store.finetuneAttentionModules,
                          store.setFinetuneAttentionModules,
                        ],
                        [
                          "finetuneMLPModules",
                          t("studio.params.mlpModules"),
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
                      {t("studio.params.targetModules")}
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {(isCpt ? CPT_TARGET_MODULES : TARGET_MODULES).map(
                        (mod) => {
                          const active = store.targetModules.includes(mod);
                          return (
                            <button
                              key={mod}
                              type="button"
                              aria-pressed={active}
                              onClick={() => {
                                store.setTargetModules(
                                  active
                                    ? store.targetModules.filter(
                                        (m) => m !== mod,
                                      )
                                    : [...store.targetModules, mod],
                                );
                              }}
                              className={`cursor-pointer rounded-full border px-2.5 py-0.5 text-ui-11 font-mono transition-colors ${selectableOptionStateClassName(active)} ${
                                active
                                  ? "text-foreground"
                                  : "text-muted-foreground"
                              }`}
                            >
                              {mod}
                            </button>
                          );
                        },
                      )}
                    </div>
                  </div>
                )}

                {/* LoRA variant */}
                <div className="grid grid-cols-2 gap-2">
                  {(
                    [
                      {
                        value: "lora",
                        label: t("studio.params.enableLora"),
                        desc: t("studio.params.trainWithLora"),
                      },
                      {
                        value: "rslora",
                        label: "RS-LoRA",
                        desc: t("studio.params.stableRank"),
                      },
                      {
                        value: "loftq",
                        label: "LoftQ",
                        desc: t("studio.params.memoryEfficient"),
                      },
                      {
                        value: "dora",
                        label: "DoRA",
                        desc: t("studio.params.weightDecomposed"),
                      },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      disabled={
                        isMac && (opt.value === "loftq" || opt.value === "dora")
                      }
                      aria-pressed={store.loraVariant === opt.value}
                      onClick={() => store.setLoraVariant(opt.value)}
                      className={`flex-1 corner-squircle rounded-xl border px-3 py-2 text-left transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-60 ${selectableOptionStateClassName(store.loraVariant === opt.value)}`}
                    >
                      <p className="text-xs font-medium">{opt.label}</p>
                      <p className="text-ui-10 text-muted-foreground">
                        {isMac &&
                        (opt.value === "loftq" || opt.value === "dora")
                          ? "Not supported on Apple Silicon"
                          : opt.desc}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Training Hyperparams */}
        {showAdvanced && (
          <Collapsible open={hyperOpen} onOpenChange={setHyperOpen}>
            <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                className={`size-3.5 transition-transform ${hyperOpen ? "rotate-180" : ""}`}
              />
              {t("studio.params.trainingHyperparameters")}
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
              <Tabs
                value={hyperparameterTab}
                onValueChange={(value) =>
                  setHyperparameterTab(value as HyperparameterTab)
                }
                className="w-full"
              >
                <SegmentedTabsList
                  value={hyperparameterTab}
                  options={hyperparameterTabs}
                  ariaLabel={t("studio.params.trainingHyperparameters")}
                  size="compact"
                />

                <TabsContent
                  value="optimization"
                  className="mt-3 flex flex-col gap-3"
                >
                  <Row
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
                      onValueChange={(v) => store.setOptimizerType(v)}
                    >
                      <SelectTrigger className="w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {optimizerOptions.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {formatOptimizerLabel(opt.value, opt.label, t)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Row>
                  <Row
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
                      onValueChange={(v) => store.setLrSchedulerType(v)}
                    >
                      <SelectTrigger className="w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {LR_SCHEDULER_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {formatSchedulerLabel(opt.value, opt.label, t)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Row>
                  <SliderRow
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
                  <SliderRow
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
                  <Row
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
                    <SliderRow
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
                  <Row
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
                      onChange={(e) =>
                        store.setSaveSteps(Number(e.target.value))
                      }
                      className="w-28 font-mono"
                    />
                  </Row>
                  <Row
                    label={t("studio.params.evalSteps")}
                    tooltip={t("studio.params.evalStepsTooltip")}
                  >
                    <Input
                      type="number"
                      step="0.01"
                      min="0.0"
                      max="1.0"
                      value={store.evalSteps}
                      onChange={(e) =>
                        store.setEvalSteps(Number(e.target.value))
                      }
                      className="w-28 font-mono"
                    />
                  </Row>
                  <Row
                    label={t("studio.params.seed")}
                    tooltip={t("studio.params.seedTooltip")}
                  >
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

                <TabsContent
                  value="memory"
                  className="mt-3 flex flex-col gap-3"
                >
                  {showVisionImageSize && (
                    <Row
                      label="Image Size"
                      tooltip={
                        <>
                          Resize images by maximum side length. Default uses the
                          model image size. Larger images use up more context.
                          Does not upscale or change aspect ratio.{" "}
                          <a
                            href="https://unsloth.ai/docs/basics/vision-fine-tuning"
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
                        value={
                          store.visionImageSize == null
                            ? "default"
                            : String(store.visionImageSize)
                        }
                        onValueChange={(value) => {
                          if (value === "default") {
                            store.setVisionImageSize(null);
                            return;
                          }
                          store.setVisionImageSize(Number(value));
                        }}
                      >
                        <SelectTrigger className="w-32">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="default">Default</SelectItem>
                          {store.visionImageSize != null &&
                            !visionImageSizePresets.includes(
                              store.visionImageSize,
                            ) && (
                              <SelectItem value={String(store.visionImageSize)}>
                                {store.visionImageSize}
                              </SelectItem>
                            )}
                          {visionImageSizePresets.map((size) => (
                            <SelectItem key={size} value={String(size)}>
                              {size}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Row>
                  )}
                  <Row
                    label={t("studio.params.gradCheckpoint")}
                    tooltip={
                      <>
                        {t("studio.params.gradCheckpointTooltip")}{" "}
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
                      value={store.gradientCheckpointing}
                      onValueChange={(v) =>
                        store.setGradientCheckpointing(
                          v as GradientCheckpointing,
                        )
                      }
                    >
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">
                          {t("studio.params.none")}
                        </SelectItem>
                        <SelectItem value="true">
                          {t("studio.params.standard")}
                        </SelectItem>
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
                        disabled={isMac}
                        onCheckedChange={(v) => store.setPacking(!!v)}
                      />
                      <label
                        htmlFor="packing"
                        className={`text-xs text-muted-foreground ${
                          isMac
                            ? "cursor-not-allowed opacity-60"
                            : "cursor-pointer"
                        }`}
                      >
                        {t("studio.params.enablePacking")}
                      </label>
                      {isMac && (
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
                            Packing is not supported on Apple Silicon (MLX).
                          </TooltipContent>
                        </Tooltip>
                      )}
                    </div>
                  )}
                  {!store.isEmbeddingModel && !isCpt && !isRawText && (
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="trainOnCompletions"
                        checked={store.trainOnCompletions}
                        disabled={store.datasetStreaming}
                        onCheckedChange={(v) =>
                          store.setTrainOnCompletions(!!v)
                        }
                      />
                      <label
                        htmlFor="trainOnCompletions"
                        aria-disabled={store.datasetStreaming || undefined}
                        title={
                          store.datasetStreaming
                            ? t(
                                "studio.dataset.streaming.completionsUnavailable",
                              )
                            : undefined
                        }
                        className={`text-xs text-muted-foreground ${
                          store.datasetStreaming
                            ? "cursor-not-allowed opacity-60"
                            : "cursor-pointer"
                        }`}
                      >
                        {t("studio.params.assistantCompletionsOnly")}
                      </label>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}
