// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { SectionCard } from "@/components/section-card";
import { Checkbox } from "@/components/ui/checkbox";
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  CONTEXT_LENGTHS,
  CPT_TARGET_MODULES,
  LR_SCHEDULER_OPTIONS,
  OPTIMIZER_OPTIONS,
  TARGET_MODULES,
} from "@/config/training";
import { useMaxStepsEpochsToggle, useTrainingConfigStore } from "@/features/training";
import { isRawTextDatasetFormat } from "@/features/training/lib/training-methods";
import { cn } from "@/lib/utils";
import { isAdapterMethod } from "@/types/training";
import type { GradientCheckpointing } from "@/types/training";
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
  useEffect,
  useRef,
  useState,
} from "react";

/** Label + optional tooltip — shared by Field and inline rows. Visual
 *  styling lives in the `.field-label` CSS class (index.css) so the
 *  same label rules drive both this component and the dataset-selectors. */
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

/** Vertical field wrapper: label on top, control underneath taking the full
 *  column width. Stops the old `justify-between` "label hugging the left,
 *  control hugging the right" layout that left a huge dead gap. */
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

/** Slider field — same shape as the chat right-panel ParamSlider:
 *  label + editable value on top row, full-width slider below. Uses the
 *  shared `panel-slider` + `panel-number-input` utilities from index.css so
 *  the visual system matches the chat panel exactly. */
function SliderField({
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
  hint?: ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
}): ReactElement {
  return (
    <div className="slider-field flex min-w-0 flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <FieldLabel label={label} tooltip={tooltip} />
        <input
          type="number"
          value={format ? format(value) : value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          aria-label={label}
          className="panel-number-input w-16"
        />
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        className="panel-slider"
      />
    </div>
  );
}

/** Responsive grid for short rows of controls. Drops to a single column on
 *  mobile; expands to `cols` (2 by default, optionally 3) on `sm+`. Items
 *  align to the top so mixed-height controls (sliders next to inputs) stay
 *  flush at the row's start rather than vertically centring. */
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
  description?: string;
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

type ParamMode = "simple" | "advanced";

export function ParamsSection({
  mode = "simple",
}: {
  mode?: ParamMode;
} = {}): ReactElement {
  const store = useTrainingConfigStore();
  const platformDeviceType = usePlatformStore((s) => s.deviceType);
  const isLora = isAdapterMethod(store.trainingMethod);
  const isCpt = store.trainingMethod === "cpt";
  const isRawText = isRawTextDatasetFormat(store.datasetFormat);
  const showVisionLora = store.isVisionModel && store.isDatasetImage === true;
  const showAdvanced = mode === "advanced";
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
      >
        <div className="flex flex-col gap-9">
          <FieldGrid cols={3}>
            <div
              key={useEpochs ? "epochs" : "steps"}
              className="slider-field flex min-w-0 flex-col gap-2 animate-in fade-in-0 slide-in-from-bottom-1 duration-200"
            >
              <div className="flex items-center justify-between gap-2">
                <FieldLabel
                  label={useEpochs ? "Epochs" : "Max Steps"}
                  tooltip={
                    <>
                      {useEpochs
                        ? "Number of full passes over the dataset."
                        : "Override total optimizer steps."}{" "}
                      <a
                        href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline"
                      >
                        Read more
                      </a>
                    </>
                  }
                />
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={toggleUseEpochs}
                    aria-label={`Switch to ${useEpochs ? "max steps" : "epochs"}`}
                    className="text-[10px] uppercase tracking-wide text-muted-foreground/70 hover:text-foreground cursor-pointer transition-colors"
                  >
                    ⇄ {useEpochs ? "Steps" : "Epochs"}
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
                    aria-label={useEpochs ? "Epochs" : "Max Steps"}
                    className="panel-number-input w-14"
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
                className="panel-slider"
              />
            </div>
            <Field
              label="Context Length"
              tooltip={
                <>
                  Maximum number of tokens per training sample.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  >
                    Read more
                  </a>
                </>
              }
            >
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
                    className="field-trigger field-soft field-pill w-full tabular-nums"
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

            <Field
              label="Learning Rate"
              tooltip={
                <>
                  Step size for weight updates. Lower values train slower but more
                  stably.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  >
                    Read more
                  </a>
                </>
              }
            >
              <Input
                type="number"
                step="0.00001"
                value={store.learningRate}
                onChange={(e) => store.setLearningRate(Number(e.target.value))}
                className="field-trigger field-soft field-pill w-full tabular-nums"
              />
            </Field>
          </FieldGrid>

          {/* Embedding Learning Rate (CPT only) */}
          {isCpt && (
            <Field
              label="Embedding Learning Rate"
              tooltip={
                <>
                  Only used when CPT is training <code>embed_tokens</code>.
                  Embeddings are easier to destabilize than LoRA weights, so
                  they usually need a smaller LR. Leave blank to use
                  <code>lr/10</code>; typical working range is 2x-10x smaller
                  than the main LR.
                </>
              }
              hint="Leave blank to use lr/10 (recommended). Typical range is 2x-10x smaller than the main learning rate."
            >
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
                className="field-trigger field-soft field-pill w-full tabular-nums"
              />
            </Field>
          )}

          {showAdvanced && (
            <>
          {isLora && (
            <Subsection
              title="LoRA"
              description="Low-rank adapter shape"
              icon={Layers01Icon}
              accent="#9b89d4"
            >
              <div className="flex flex-col gap-4">
                  <FieldGrid cols={3}>
                    <SliderField
                      label="Rank"
                      tooltip={
                        <>
                          Dimension of the low-rank matrices. Higher = more
                          capacity.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                    <SliderField
                      label="Alpha"
                      tooltip={
                        <>
                          Scaling factor for LoRA updates. Usually 2x rank.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                    <SliderField
                      label="Dropout"
                      tooltip={
                        <>
                          Dropout probability for LoRA layers to reduce overfitting.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                  </FieldGrid>

                  {showVisionLora && (
                    <Field label="Fine-tune">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
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
                              className="cursor-pointer text-[12px] text-muted-foreground"
                            >
                              {label}
                            </label>
                          </div>
                        ))}
                      </div>
                    </Field>
                  )}

                  {!showVisionLora && (
                    <Field label="Target Modules">
                      <div className="flex flex-wrap gap-1.5">
                        {(isCpt ? CPT_TARGET_MODULES : TARGET_MODULES).map((mod) => {
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
                              className={`cursor-pointer rounded-full border px-2.5 py-0.5 text-[11px] font-mono transition-colors ${
                                active
                                  ? "border-foreground/30 bg-foreground/[0.08] text-foreground"
                                  : "border-border text-muted-foreground hover:bg-foreground/[0.04]"
                              }`}
                            >
                              {mod}
                            </button>
                          );
                        })}
                      </div>
                    </Field>
                  )}

                  <Field label="Variant">
                    <div className="grid grid-cols-3 gap-2">
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
                          className={`corner-squircle rounded-xl border px-3 py-2 text-left transition-colors cursor-pointer ${
                            store.loraVariant === opt.value
                              ? "border-foreground/30 bg-foreground/[0.05] ring-1 ring-foreground/15"
                              : "border-border hover:border-foreground/20"
                          }`}
                        >
                          <p className="text-[12px] font-medium tracking-nav">
                            {opt.label}
                          </p>
                          <p className="text-[10.5px] text-muted-foreground">
                            {opt.desc}
                          </p>
                        </button>
                      ))}
                    </div>
                  </Field>
              </div>
            </Subsection>
          )}

          <Subsection
            title="Optimization"
            description="Optimizer + batching + memory"
            icon={FlashIcon}
            accent="#d4a566"
          >
            <div className="flex flex-col gap-4">
                  <FieldGrid cols={3}>
                    <Field
                      label="Optimizer"
                      tooltip={
                        <>
                          Optimization algorithm. 8-bit variants reduce memory
                          usage. Fused is recommended for vision models.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                        <SelectTrigger className="field-trigger field-soft w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {OPTIMIZER_OPTIONS.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value}>
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Field>
                    <Field
                      label="LR Scheduler"
                      tooltip={
                        <>
                          How the learning rate changes over training. Linear
                          decays steadily; cosine decays in a curve.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                        <SelectTrigger className="field-trigger field-soft w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {LR_SCHEDULER_OPTIONS.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value}>
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </Field>
                    <Field
                      label="Gradient Checkpointing"
                      tooltip={
                        <>
                          Trade compute for memory by recomputing activations.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                  </FieldGrid>
                  <FieldGrid cols={3}>
                    <SliderField
                      label="Batch Size"
                      tooltip={
                        <>
                          Samples processed per step. Higher uses more VRAM.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                    <SliderField
                      label="Grad Accum"
                      tooltip={
                        <>
                          Simulates larger batch sizes without extra VRAM.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                    <Field
                      label="Weight Decay"
                      tooltip={
                        <>
                          L2 regularization to prevent overfitting.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
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
                        className="field-trigger field-soft field-pill w-full tabular-nums"
                      />
                    </Field>
                  </FieldGrid>
                  {((!showVisionLora && !store.isEmbeddingModel) ||
                    (!store.isEmbeddingModel && !isCpt && !isRawText)) && (
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {!showVisionLora && !store.isEmbeddingModel && (
                        <div className="flex items-center gap-2">
                          <Checkbox
                            id="packing"
                            checked={store.packing}
                            onCheckedChange={(v) => store.setPacking(!!v)}
                          />
                          <label
                            htmlFor="packing"
                            className="cursor-pointer text-[12px] text-muted-foreground"
                          >
                            Enable packing
                          </label>
                        </div>
                      )}
                      {!store.isEmbeddingModel && !isCpt && !isRawText && (
                        <div className="flex items-center gap-2">
                          <Checkbox
                            id="trainOnCompletions"
                            checked={store.trainOnCompletions}
                            onCheckedChange={(v) =>
                              store.setTrainOnCompletions(!!v)
                            }
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
                  )}
            </div>
          </Subsection>

          <Subsection
            title="Schedule & checkpoints"
            description="LR warmup, eval, checkpoint cadence"
            icon={Timer01Icon}
            accent="#82a8c5"
          >
            <div className="flex flex-col gap-4">
                  <SliderField
                    label="Warmup Steps"
                    tooltip={
                      <>
                        Gradually increase LR at training start for stability.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline"
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
                  <FieldGrid cols={3}>
                    <Field
                      label="Save Steps"
                      tooltip={
                        <>
                          Save a checkpoint every N steps. 0 to disable.{" "}
                          <a
                            href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline"
                          >
                            Read more
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
                        className="field-trigger field-soft field-pill w-full tabular-nums"
                      />
                    </Field>
                    <Field
                      label="Eval Steps"
                      tooltip="Fraction of total training steps between evaluations (0-1). Set to 0 to disable evaluation. E.g. 0.01 = evaluate every 1% of steps."
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
                        className="field-trigger field-soft field-pill w-full tabular-nums"
                      />
                    </Field>
                    <Field label="Seed" tooltip="Random seed for reproducibility.">
                      <Input
                        type="number"
                        value={store.randomSeed}
                        onChange={(e) =>
                          store.setRandomSeed(Number(e.target.value))
                        }
                        className="field-trigger field-soft field-pill w-full tabular-nums"
                      />
                    </Field>
                  </FieldGrid>
            </div>
          </Subsection>

            </>
          )}
        </div>
      </SectionCard>
    </div>
  );
}
