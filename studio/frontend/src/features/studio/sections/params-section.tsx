// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Checkbox } from "@/components/ui/checkbox";
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
  OPTIMIZER_OPTIONS,
  TARGET_MODULES,
} from "@/config/training";
import {
  isRawTextDatasetFormat,
  type TrainingConfigStore,
  useMaxStepsEpochsToggle,
  useTrainingConfigStore,
} from "@/features/training";
import { cn } from "@/lib/utils";
import { isAdapterMethod, type GradientCheckpointing } from "@/types/training";
import {
  FlashIcon,
  InformationCircleIcon,
  Layers01Icon,
  Settings04Icon,
  Timer01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type CSSProperties,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";

const CONTEXT_LENGTH_ITEMS = CONTEXT_LENGTHS.map(String);

const LORA_VARIANTS = [
  { value: "lora", label: "Enable LoRA", desc: "Train with LoRA" },
  { value: "rslora", label: "RS-LoRA", desc: "Stable Rank" },
  { value: "loftq", label: "LoftQ", desc: "Memory Efficient" },
] as const;

type ParamMode = "simple" | "advanced";

function FieldLabel({
  label,
  tooltip,
}: {
  label: string;
  tooltip?: ReactNode;
}): ReactElement {
  return (
    <span className="field-label">
      {label}
      {tooltip && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="text-muted-foreground/50 hover:text-foreground"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent>{tooltip}</TooltipContent>
        </Tooltip>
      )}
    </span>
  );
}

function Field({
  label,
  tooltip,
  hint,
  children,
}: {
  label: string;
  tooltip?: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <FieldLabel label={label} tooltip={tooltip} />
      <div className="min-w-0">{children}</div>
      {hint && (
        <p className="mt-0.5 text-[11px] leading-[16px] text-muted-foreground/65">
          {hint}
        </p>
      )}
    </div>
  );
}

function FieldGrid({
  children,
  cols = 2,
}: {
  children: ReactNode;
  cols?: 2 | 3;
}): ReactElement {
  return (
    <div
      className={cn(
        "grid grid-cols-1 items-start gap-x-4 gap-y-3",
        cols === 3 ? "sm:grid-cols-3" : "sm:grid-cols-2",
      )}
    >
      {children}
    </div>
  );
}

function Subsection({
  title,
  icon,
  accent,
  children,
}: {
  title: string;
  icon?: IconSvgElement;
  accent?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <section className="flex flex-col gap-4">
      <h4 className="flex items-center gap-2 text-[11.5px] font-medium tracking-[0.04em] text-foreground">
        {icon && (
          <span
            className="train-section-chip inline-flex size-5 shrink-0 items-center justify-center rounded-full"
            style={
              accent ? ({ "--chip-color": accent } as CSSProperties) : undefined
            }
          >
            <HugeiconsIcon
              icon={icon}
              strokeWidth={1.75}
              className="size-3"
            />
          </span>
        )}
        {title}
      </h4>
      <div className="flex flex-col gap-5">{children}</div>
    </section>
  );
}

function linkTooltip(text: string): ReactNode {
  return (
    <>
      {text}{" "}
      <a
        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
        target="_blank"
        rel="noopener noreferrer"
        className="underline"
      >
        Read more
      </a>
    </>
  );
}

function SliderField({
  label,
  tooltip,
  value,
  onChange,
  min,
  max,
  step,
  format,
  inputClassName = "w-16",
  labelAction,
}: {
  label: string;
  tooltip?: ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  inputClassName?: string;
  labelAction?: ReactNode;
}): ReactElement {
  const frameRef = useRef<number | null>(null);
  const latestValueRef = useRef(value);
  const [draftState, setDraftState] = useState({ source: value, value });
  const draft = draftState.source === value ? draftState.value : value;

  const setDraft = useCallback(
    (next: number) => {
      latestValueRef.current = next;
      setDraftState({ source: value, value: next });
    },
    [value],
  );

  const cancelScheduledChange = useCallback(() => {
    if (frameRef.current === null || typeof window === "undefined") return;
    window.cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  }, []);

  const commitNow = useCallback(
    (next: number) => {
      cancelScheduledChange();
      setDraft(next);
      onChange(next);
    },
    [cancelScheduledChange, onChange, setDraft],
  );

  const scheduleChange = useCallback(
    (next: number) => {
      setDraft(next);
      if (typeof window === "undefined") {
        onChange(next);
        return;
      }
      if (frameRef.current !== null) return;
      frameRef.current = window.requestAnimationFrame(() => {
        frameRef.current = null;
        onChange(latestValueRef.current);
      });
    },
    [onChange, setDraft],
  );

  useEffect(
    () => () => {
      cancelScheduledChange();
    },
    [cancelScheduledChange],
  );

  return (
    <div className="slider-field flex min-w-0 flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <FieldLabel label={label} tooltip={tooltip} />
        <div className="flex items-center gap-1.5">
          {labelAction}
          <input
            type="number"
            value={format ? format(draft) : draft}
            onChange={(event) => {
              const raw = event.currentTarget.value;
              if (raw === "") return;
              const next = Number(raw);
              if (!Number.isFinite(next)) return;
              commitNow(Math.min(max, Math.max(min, next)));
            }}
            min={min}
            max={max}
            step={step}
            aria-label={label}
            className={cn("panel-number-input", inputClassName)}
          />
        </div>
      </div>
      <Slider
        value={[draft]}
        onValueChange={([next]) => {
          if (typeof next === "number") scheduleChange(next);
        }}
        onValueCommit={([next]) => {
          if (typeof next === "number") commitNow(next);
        }}
        min={min}
        max={max}
        step={step}
        className="panel-slider"
      />
    </div>
  );
}

function SliderParamField({
  valueSelector,
  setterSelector,
  ...props
}: Omit<Parameters<typeof SliderField>[0], "value" | "onChange"> & {
  valueSelector: (state: TrainingConfigStore) => number;
  setterSelector: (state: TrainingConfigStore) => (value: number) => void;
}): ReactElement {
  const value = useTrainingConfigStore(valueSelector);
  const setter = useTrainingConfigStore(setterSelector);
  return <SliderField {...props} value={value} onChange={setter} />;
}

function MaxStepsEpochsField(): ReactElement {
  const {
    maxSteps,
    epochs,
    saveSteps,
    setMaxSteps,
    setEpochs,
    setSaveSteps,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      maxSteps: state.maxSteps,
      epochs: state.epochs,
      saveSteps: state.saveSteps,
      setMaxSteps: state.setMaxSteps,
      setEpochs: state.setEpochs,
      setSaveSteps: state.setSaveSteps,
    })),
  );
  const { useEpochs, toggleUseEpochs } = useMaxStepsEpochsToggle({
    maxSteps,
    epochs,
    saveSteps,
    setMaxSteps,
    setEpochs,
    setSaveSteps,
  });
  const max = useEpochs
    ? Math.max(20, epochs, 1)
    : Math.max(500, maxSteps, 30);
  const value = useEpochs
    ? Math.min(max, Math.max(1, epochs))
    : Math.min(max, Math.max(1, maxSteps));
  const setValue = useEpochs ? setEpochs : setMaxSteps;
  const label = useEpochs ? "Epochs" : "Max Steps";

  return (
    <div
      key={useEpochs ? "epochs" : "steps"}
      className="animate-in fade-in-0 slide-in-from-bottom-1 duration-200"
    >
      <SliderField
        label={label}
        tooltip={linkTooltip(
          useEpochs
            ? "Number of full passes over the dataset."
            : "Override total optimizer steps.",
        )}
        value={value}
        onChange={setValue}
        min={1}
        max={max}
        step={1}
        inputClassName="w-14"
        labelAction={
          <button
            type="button"
            onClick={toggleUseEpochs}
            aria-label={`Switch to ${useEpochs ? "max steps" : "epochs"}`}
            className="cursor-pointer text-[10px] uppercase tracking-wide text-muted-foreground/70 transition-colors hover:text-foreground"
          >
            ⇄ {useEpochs ? "Steps" : "Epochs"}
          </button>
        }
      />
    </div>
  );
}

function ContextLengthField(): ReactElement {
  const contextLength = useTrainingConfigStore((state) => state.contextLength);
  const setContextLength = useTrainingConfigStore(
    (state) => state.setContextLength,
  );
  const ctxAnchorRef = useRef<HTMLDivElement>(null);
  const [draftState, setDraftState] = useState({
    source: contextLength,
    value: String(contextLength),
  });
  const ctxInput =
    draftState.source === contextLength
      ? draftState.value
      : String(contextLength);

  const setCtxInput = useCallback(
    (value: string) => {
      setDraftState({ source: contextLength, value });
    },
    [contextLength],
  );

  const trySetContextLength = useCallback(
    (input: string): number | null => {
      const next = Number(input);
      if (!Number.isInteger(next) || next <= 0) return null;
      setContextLength(next);
      setDraftState({ source: next, value: String(next) });
      return next;
    },
    [setContextLength],
  );

  return (
    <Field
      label="Context Length"
      tooltip={linkTooltip("Maximum number of tokens per training sample.")}
    >
      <div ref={ctxAnchorRef}>
        <Combobox
          items={CONTEXT_LENGTH_ITEMS}
          filteredItems={CONTEXT_LENGTH_ITEMS}
          filter={null}
          value={String(contextLength)}
          onValueChange={(value) => {
            if (value) trySetContextLength(value);
          }}
          onInputValueChange={setCtxInput}
          itemToStringValue={(id) => Number(id).toLocaleString()}
          autoHighlight={false}
        >
          <ComboboxInput
            placeholder={String(contextLength)}
            className="field-trigger field-soft field-pill w-full tabular-nums"
            onBlur={() => {
              if (trySetContextLength(ctxInput) === null) {
                setDraftState({
                  source: contextLength,
                  value: String(contextLength),
                });
              }
            }}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              const next = trySetContextLength(ctxInput);
              if (next === null) return;
              if (!CONTEXT_LENGTH_ITEMS.includes(ctxInput.trim())) {
                event.stopPropagation();
                event.preventDefault();
              }
              setDraftState({ source: next, value: String(next) });
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
    </Field>
  );
}

function LearningRateField(): ReactElement {
  const learningRate = useTrainingConfigStore((state) => state.learningRate);
  const setLearningRate = useTrainingConfigStore(
    (state) => state.setLearningRate,
  );

  return (
    <Field
      label="Learning Rate"
      tooltip={linkTooltip(
        "Step size for weight updates. Lower values train slower but more stably.",
      )}
    >
      <Input
        type="number"
        step="0.00001"
        value={learningRate}
        onChange={(event) => setLearningRate(Number(event.currentTarget.value))}
        className="field-trigger field-soft field-pill w-full tabular-nums"
      />
    </Field>
  );
}

function CoreParams(): ReactElement {
  return (
    <FieldGrid cols={3}>
      <MaxStepsEpochsField />
      <ContextLengthField />
      <LearningRateField />
    </FieldGrid>
  );
}

function EmbeddingLearningRateField(): ReactElement {
  const { learningRate, embeddingLearningRate, setEmbeddingLearningRate } =
    useTrainingConfigStore(
      useShallow((state) => ({
        learningRate: state.learningRate,
        embeddingLearningRate: state.embeddingLearningRate,
        setEmbeddingLearningRate: state.setEmbeddingLearningRate,
      })),
    );

  return (
    <Field
      label="Embedding Learning Rate"
      tooltip={
        <>
          Only used when CPT is training <code>embed_tokens</code>. Embeddings
          are easier to destabilize than LoRA weights, so they usually need a
          smaller LR. Leave blank to use <code>lr/10</code>; typical working
          range is 2x-10x smaller than the main LR.
        </>
      }
      hint="Leave blank to use lr/10 (recommended). Typical range is 2x-10x smaller than the main learning rate."
    >
      <Input
        type="number"
        step="0.00001"
        min="0"
        max="1"
        placeholder={`auto (${(learningRate / 10).toExponential(1)})`}
        value={embeddingLearningRate ?? ""}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          if (raw === "") {
            setEmbeddingLearningRate(null);
            return;
          }
          const next = Number(raw);
          setEmbeddingLearningRate(Number.isFinite(next) ? next : null);
        }}
        className="field-trigger field-soft field-pill w-full tabular-nums"
      />
    </Field>
  );
}

function VisionFineTuneFields(): ReactElement {
  const config = useTrainingConfigStore(
    useShallow((state) => ({
      finetuneVisionLayers: state.finetuneVisionLayers,
      setFinetuneVisionLayers: state.setFinetuneVisionLayers,
      finetuneLanguageLayers: state.finetuneLanguageLayers,
      setFinetuneLanguageLayers: state.setFinetuneLanguageLayers,
      finetuneAttentionModules: state.finetuneAttentionModules,
      setFinetuneAttentionModules: state.setFinetuneAttentionModules,
      finetuneMLPModules: state.finetuneMLPModules,
      setFinetuneMLPModules: state.setFinetuneMLPModules,
    })),
  );
  const options = [
    [
      "finetuneVisionLayers",
      "Vision layers",
      config.finetuneVisionLayers,
      config.setFinetuneVisionLayers,
    ],
    [
      "finetuneLanguageLayers",
      "Language layers",
      config.finetuneLanguageLayers,
      config.setFinetuneLanguageLayers,
    ],
    [
      "finetuneAttentionModules",
      "Attention modules",
      config.finetuneAttentionModules,
      config.setFinetuneAttentionModules,
    ],
    [
      "finetuneMLPModules",
      "MLP modules",
      config.finetuneMLPModules,
      config.setFinetuneMLPModules,
    ],
  ] as const;

  return (
    <Field label="Fine-tune">
      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {options.map(([key, label, checked, setter]) => (
          <div key={key} className="flex items-center gap-2">
            <Checkbox
              id={key}
              checked={checked}
              onCheckedChange={(value) => setter(!!value)}
            />
            <label
              htmlFor={key}
              className="cursor-pointer text-[12px] text-muted-foreground"
            >
              {label}
            </label>
          </div>
        ))}
      </div>
    </Field>
  );
}

function TargetModulesField({ isCpt }: { isCpt: boolean }): ReactElement {
  const { targetModules, setTargetModules } = useTrainingConfigStore(
    useShallow((state) => ({
      targetModules: state.targetModules,
      setTargetModules: state.setTargetModules,
    })),
  );
  const modules = isCpt ? CPT_TARGET_MODULES : TARGET_MODULES;

  return (
    <Field label="Target Modules">
      <div className="flex flex-wrap gap-1.5">
        {modules.map((mod) => {
          const active = targetModules.includes(mod);
          return (
            <button
              key={mod}
              type="button"
              onClick={() => {
                setTargetModules(
                  active
                    ? targetModules.filter((item) => item !== mod)
                    : [...targetModules, mod],
                );
              }}
              className={cn(
                "cursor-pointer rounded-full border px-2.5 py-0.5 font-mono text-[11px] transition-colors",
                active
                  ? "border-foreground/30 bg-foreground/[0.08] text-foreground"
                  : "border-border text-muted-foreground hover:bg-foreground/[0.04]",
              )}
            >
              {mod}
            </button>
          );
        })}
      </div>
    </Field>
  );
}

function LoraVariantField(): ReactElement {
  const loraVariant = useTrainingConfigStore((state) => state.loraVariant);
  const setLoraVariant = useTrainingConfigStore(
    (state) => state.setLoraVariant,
  );

  return (
    <Field label="Variant">
      <div className="grid grid-cols-3 gap-2">
        {LORA_VARIANTS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => setLoraVariant(option.value)}
            className={cn(
              "corner-squircle cursor-pointer rounded-xl border px-3 py-2 text-left transition-colors",
              loraVariant === option.value
                ? "border-foreground/30 bg-foreground/[0.05] ring-1 ring-foreground/15"
                : "border-border hover:border-foreground/20",
            )}
          >
            <p className="text-[12px] font-medium tracking-nav">
              {option.label}
            </p>
            <p className="text-[10.5px] text-muted-foreground">
              {option.desc}
            </p>
          </button>
        ))}
      </div>
    </Field>
  );
}

function LoraSection({
  isCpt,
  showVisionLora,
}: {
  isCpt: boolean;
  showVisionLora: boolean;
}): ReactElement {
  return (
    <Subsection title="LoRA" icon={Layers01Icon} accent="#9b89d4">
      <div className="flex flex-col gap-4">
        <FieldGrid cols={3}>
          <SliderParamField
            label="Rank"
            tooltip={linkTooltip(
              "Dimension of the low-rank matrices. Higher = more capacity.",
            )}
            valueSelector={(state) => state.loraRank}
            setterSelector={(state) => state.setLoraRank}
            min={4}
            max={128}
            step={4}
          />
          <SliderParamField
            label="Alpha"
            tooltip={linkTooltip(
              "Scaling factor for LoRA updates. Usually 2x rank.",
            )}
            valueSelector={(state) => state.loraAlpha}
            setterSelector={(state) => state.setLoraAlpha}
            min={4}
            max={256}
            step={4}
          />
          <SliderParamField
            label="Dropout"
            tooltip={linkTooltip(
              "Dropout probability for LoRA layers to reduce overfitting.",
            )}
            valueSelector={(state) => state.loraDropout}
            setterSelector={(state) => state.setLoraDropout}
            min={0}
            max={0.5}
            step={0.01}
            format={(value) => value.toFixed(2)}
          />
        </FieldGrid>

        {showVisionLora ? (
          <VisionFineTuneFields />
        ) : (
          <TargetModulesField isCpt={isCpt} />
        )}

        <LoraVariantField />
      </div>
    </Subsection>
  );
}

function OptimizerField(): ReactElement {
  const optimizerType = useTrainingConfigStore((state) => state.optimizerType);
  const setOptimizerType = useTrainingConfigStore(
    (state) => state.setOptimizerType,
  );

  return (
    <Field
      label="Optimizer"
      tooltip={linkTooltip(
        "Optimization algorithm. 8-bit variants reduce memory usage. Fused is recommended for vision models.",
      )}
    >
      <Select value={optimizerType} onValueChange={setOptimizerType}>
        <SelectTrigger className="field-trigger field-soft w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {OPTIMIZER_OPTIONS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}

function LrSchedulerField(): ReactElement {
  const lrSchedulerType = useTrainingConfigStore(
    (state) => state.lrSchedulerType,
  );
  const setLrSchedulerType = useTrainingConfigStore(
    (state) => state.setLrSchedulerType,
  );

  return (
    <Field
      label="LR Scheduler"
      tooltip={linkTooltip(
        "How the learning rate changes over training. Linear decays steadily; cosine decays in a curve.",
      )}
    >
      <Select value={lrSchedulerType} onValueChange={setLrSchedulerType}>
        <SelectTrigger className="field-trigger field-soft w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {LR_SCHEDULER_OPTIONS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}

function GradientCheckpointingField({
  platformDeviceType,
}: {
  platformDeviceType: string | null;
}): ReactElement {
  const gradientCheckpointing = useTrainingConfigStore(
    (state) => state.gradientCheckpointing,
  );
  const setGradientCheckpointing = useTrainingConfigStore(
    (state) => state.setGradientCheckpointing,
  );

  return (
    <Field
      label="Gradient Checkpointing"
      tooltip={linkTooltip("Trade compute for memory by recomputing activations.")}
    >
      <Select
        value={gradientCheckpointing}
        onValueChange={(value) =>
          setGradientCheckpointing(value as GradientCheckpointing)
        }
      >
        <SelectTrigger className="field-trigger field-soft w-full">
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
    </Field>
  );
}

function WeightDecayField(): ReactElement {
  const weightDecay = useTrainingConfigStore((state) => state.weightDecay);
  const setWeightDecay = useTrainingConfigStore((state) => state.setWeightDecay);

  return (
    <Field
      label="Weight Decay"
      tooltip={linkTooltip("L2 regularization to prevent overfitting.")}
    >
      <Input
        type="number"
        step="0.001"
        value={weightDecay}
        onChange={(event) => setWeightDecay(Number(event.currentTarget.value))}
        className="field-trigger field-soft field-pill w-full tabular-nums"
      />
    </Field>
  );
}

function OptimizationFlags({
  showPacking,
  showTrainOnCompletions,
}: {
  showPacking: boolean;
  showTrainOnCompletions: boolean;
}): ReactElement | null {
  const { packing, setPacking, trainOnCompletions, setTrainOnCompletions } =
    useTrainingConfigStore(
      useShallow((state) => ({
        packing: state.packing,
        setPacking: state.setPacking,
        trainOnCompletions: state.trainOnCompletions,
        setTrainOnCompletions: state.setTrainOnCompletions,
      })),
    );

  if (!showPacking && !showTrainOnCompletions) return null;

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {showPacking && (
        <div className="flex items-center gap-2">
          <Checkbox
            id="packing"
            checked={packing}
            onCheckedChange={(value) => setPacking(!!value)}
          />
          <label
            htmlFor="packing"
            className="cursor-pointer text-[12px] text-muted-foreground"
          >
            Enable packing
          </label>
        </div>
      )}
      {showTrainOnCompletions && (
        <div className="flex items-center gap-2">
          <Checkbox
            id="trainOnCompletions"
            checked={trainOnCompletions}
            onCheckedChange={(value) => setTrainOnCompletions(!!value)}
          />
          <label
            htmlFor="trainOnCompletions"
            className="cursor-pointer text-[12px] text-muted-foreground"
          >
            Assistant completions only
          </label>
        </div>
      )}
    </div>
  );
}

function OptimizationSection({
  platformDeviceType,
  showVisionLora,
  isEmbeddingModel,
  isCpt,
  isRawText,
}: {
  platformDeviceType: string | null;
  showVisionLora: boolean;
  isEmbeddingModel: boolean;
  isCpt: boolean;
  isRawText: boolean;
}): ReactElement {
  return (
    <Subsection title="Optimization" icon={FlashIcon} accent="#d4a566">
      <div className="flex flex-col gap-4">
        <FieldGrid cols={3}>
          <OptimizerField />
          <LrSchedulerField />
          <GradientCheckpointingField platformDeviceType={platformDeviceType} />
        </FieldGrid>
        <FieldGrid cols={3}>
          <SliderParamField
            label="Batch Size"
            tooltip={linkTooltip("Samples processed per step. Higher uses more VRAM.")}
            valueSelector={(state) => state.batchSize}
            setterSelector={(state) => state.setBatchSize}
            min={1}
            max={32}
            step={1}
          />
          <SliderParamField
            label="Grad Accum"
            tooltip={linkTooltip(
              "Simulates larger batch sizes without extra VRAM.",
            )}
            valueSelector={(state) => state.gradientAccumulation}
            setterSelector={(state) => state.setGradientAccumulation}
            min={1}
            max={64}
            step={1}
          />
          <WeightDecayField />
        </FieldGrid>
        <OptimizationFlags
          showPacking={!showVisionLora && !isEmbeddingModel}
          showTrainOnCompletions={!isEmbeddingModel && !isCpt && !isRawText}
        />
      </div>
    </Subsection>
  );
}

function SaveStepsField(): ReactElement {
  const saveSteps = useTrainingConfigStore((state) => state.saveSteps);
  const setSaveSteps = useTrainingConfigStore((state) => state.setSaveSteps);

  return (
    <Field
      label="Save Steps"
      tooltip={linkTooltip("Save a checkpoint every N steps. 0 to disable.")}
    >
      <Input
        type="number"
        value={saveSteps}
        onChange={(event) => setSaveSteps(Number(event.currentTarget.value))}
        className="field-trigger field-soft field-pill w-full tabular-nums"
      />
    </Field>
  );
}

function EvalStepsField(): ReactElement {
  const evalSteps = useTrainingConfigStore((state) => state.evalSteps);
  const setEvalSteps = useTrainingConfigStore((state) => state.setEvalSteps);

  return (
    <Field
      label="Eval Steps"
      tooltip="Fraction of total training steps between evaluations (0-1). Set to 0 to disable evaluation. E.g. 0.01 = evaluate every 1% of steps."
    >
      <Input
        type="number"
        step="0.01"
        min="0.0"
        max="1.0"
        value={evalSteps}
        onChange={(event) => setEvalSteps(Number(event.currentTarget.value))}
        className="field-trigger field-soft field-pill w-full tabular-nums"
      />
    </Field>
  );
}

function SeedField(): ReactElement {
  const randomSeed = useTrainingConfigStore((state) => state.randomSeed);
  const setRandomSeed = useTrainingConfigStore((state) => state.setRandomSeed);

  return (
    <Field label="Seed" tooltip="Random seed for reproducibility.">
      <Input
        type="number"
        value={randomSeed}
        onChange={(event) => setRandomSeed(Number(event.currentTarget.value))}
        className="field-trigger field-soft field-pill w-full tabular-nums"
      />
    </Field>
  );
}

function ScheduleSection(): ReactElement {
  return (
    <Subsection title="Schedule & checkpoints" icon={Timer01Icon} accent="#82a8c5">
      <div className="flex flex-col gap-4">
        <SliderParamField
          label="Warmup Steps"
          tooltip={linkTooltip(
            "Gradually increase LR at training start for stability.",
          )}
          valueSelector={(state) => state.warmupSteps}
          setterSelector={(state) => state.setWarmupSteps}
          min={0}
          max={100}
          step={1}
        />
        <FieldGrid cols={3}>
          <SaveStepsField />
          <EvalStepsField />
          <SeedField />
        </FieldGrid>
      </div>
    </Subsection>
  );
}

export function ParamsSection({
  mode = "simple",
}: {
  mode?: ParamMode;
} = {}): ReactElement {
  const platformDeviceType = usePlatformStore((state) => state.deviceType);
  const {
    trainingMethod,
    datasetFormat,
    isVisionModel,
    isDatasetImage,
    isEmbeddingModel,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      trainingMethod: state.trainingMethod,
      datasetFormat: state.datasetFormat,
      isVisionModel: state.isVisionModel,
      isDatasetImage: state.isDatasetImage,
      isEmbeddingModel: state.isEmbeddingModel,
    })),
  );
  const isLora = isAdapterMethod(trainingMethod);
  const isCpt = trainingMethod === "cpt";
  const isRawText = isRawTextDatasetFormat(datasetFormat);
  const showVisionLora = isVisionModel && isDatasetImage === true;
  const showAdvanced = mode === "advanced";

  return (
    <div data-tour="studio-params" className="min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={Settings04Icon} className="size-5" />}
        title="Parameters"
        description="Configure training hyperparameters"
        accent="orange"
      >
        <div className="flex flex-col gap-9">
          <CoreParams />

          {isCpt && <EmbeddingLearningRateField />}

          {showAdvanced && (
            <>
              {isLora && (
                <LoraSection isCpt={isCpt} showVisionLora={showVisionLora} />
              )}
              <OptimizationSection
                platformDeviceType={platformDeviceType}
                showVisionLora={showVisionLora}
                isEmbeddingModel={isEmbeddingModel}
                isCpt={isCpt}
                isRawText={isRawText}
              />
              <ScheduleSection />
            </>
          )}
        </div>
      </SectionCard>
    </div>
  );
}
