// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CONTEXT_LENGTHS } from "@/config/training";
import { useMaxStepsEpochsToggle, useTrainingConfigStore } from "@/features/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";

/** Format a number in scientific notation like 2e-4, 5e-3, etc. */
function formatLR(value: number): string {
  if (value <= 0) return "0";
  const exp = Math.floor(Math.log10(value));
  const mantissa = value / 10 ** exp;
  const rounded = Math.round(mantissa * 10) / 10;
  if (rounded === 10) return `1e${exp + 1}`;
  if (rounded === Math.round(rounded)) return `${Math.round(rounded)}e${exp}`;
  return `${rounded}e${exp}`;
}

/**
 * Step learning rate up in a scientific-notation-friendly sequence:
 * 1e-4 -> 2e-4 -> 3e-4 -> ... -> 9e-4 -> 1e-3 -> 2e-3 -> ...
 */
function stepLR(value: number, direction: 1 | -1): number {
  if (value <= 0) return 1e-5;
  const exp = Math.floor(Math.log10(value) + 1e-9);
  const mantissa = Math.round(value / 10 ** exp);
  let newMantissa = mantissa + direction;
  let newExp = exp;
  if (newMantissa > 9) {
    newMantissa = 1;
    newExp = exp + 1;
  } else if (newMantissa < 1) {
    newMantissa = 9;
    newExp = exp - 1;
  }
  return newMantissa * 10 ** newExp;
}

export function HyperparametersStep() {
  const {
    trainingMethod,
    maxSteps,
    setMaxSteps,
    epochs,
    setEpochs,
    saveSteps,
    setSaveSteps,
    contextLength,
    setContextLength,
    learningRate,
    setLearningRate,
    loraRank,
    setLoraRank,
    loraAlpha,
    setLoraAlpha,
    loraDropout,
    setLoraDropout,
    maxPositionEmbeddings,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      trainingMethod: s.trainingMethod,
      maxSteps: s.maxSteps,
      setMaxSteps: s.setMaxSteps,
      epochs: s.epochs,
      setEpochs: s.setEpochs,
      saveSteps: s.saveSteps,
      setSaveSteps: s.setSaveSteps,
      contextLength: s.contextLength,
      setContextLength: s.setContextLength,
      learningRate: s.learningRate,
      setLearningRate: s.setLearningRate,
      loraRank: s.loraRank,
      setLoraRank: s.setLoraRank,
      loraAlpha: s.loraAlpha,
      setLoraAlpha: s.setLoraAlpha,
      loraDropout: s.loraDropout,
      setLoraDropout: s.setLoraDropout,
      maxPositionEmbeddings: s.maxPositionEmbeddings,
    })),
  );

  const showLoraParams =
    trainingMethod === "lora" || trainingMethod === "qlora";
  const { useEpochs, toggleUseEpochs } = useMaxStepsEpochsToggle({
    maxSteps,
    epochs,
    saveSteps,
    setMaxSteps,
    setEpochs,
    setSaveSteps,
  });

  const maxStepsSliderMax = Math.max(500, maxSteps, 30);
  const epochsSliderMax = Math.max(10, epochs, 1);

  // Use model's max_position_embeddings to cap context length options.
  // Fall back to 65536 (64K) if not available.
  const maxCtx = maxPositionEmbeddings ?? 65536;
  const contextLengthOptions = useMemo(
    () => CONTEXT_LENGTHS.filter((len) => len <= maxCtx),
    [maxCtx],
  );

  return (
    <FieldGroup>
      <FieldSet>
        <FieldLegend variant="label">Choose your training parameters</FieldLegend>
        <div className="flex flex-col gap-4">
          <div
            key={useEpochs ? "epochs" : "steps"}
            className="flex flex-col gap-2 animate-in fade-in-1 slide-in-from-bottom-1 duration-200"
          >
            <div className="flex items-center justify-between">
              <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                {useEpochs ? "Epochs" : "Max Steps"}
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      className="text-muted-foreground/50 hover:text-muted-foreground"
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3.5"
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
              </FieldLabel>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={toggleUseEpochs}
                  className="text-xs text-primary underline cursor-pointer"
                >
                  {useEpochs ? "Use Max Steps" : "Use Epochs"}
                </button>
                <Slider
                  value={[
                    useEpochs
                      ? Math.min(epochsSliderMax, Math.max(1, epochs))
                      : Math.min(maxStepsSliderMax, Math.max(1, maxSteps)),
                  ]}
                  onValueChange={([v]) =>
                    useEpochs ? setEpochs(v) : setMaxSteps(v)
                  }
                  min={1}
                  max={useEpochs ? epochsSliderMax : maxStepsSliderMax}
                  step={1}
                  className="w-40"
                />
                <input
                  type="number"
                  value={useEpochs ? epochs : maxSteps}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") return;

                    const value = Number(raw);
                    if (!Number.isFinite(value) || value < 1) return;

                    if (useEpochs) {
                      setEpochs(value);
                    } else {
                      setMaxSteps(value);
                    }
                  }}
                  min={1}
                  max={useEpochs ? epochsSliderMax : maxStepsSliderMax}
                  step={1}
                  className="w-16 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
              Context Length
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-muted-foreground/50 hover:text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3.5"
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
            </FieldLabel>
            <Select
              value={String(contextLength)}
              onValueChange={(v) => setContextLength(Number(v))}
            >
              <SelectTrigger className="w-32 font-mono">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {contextLengthOptions.map((len) => (
                  <SelectItem key={len} value={String(len)}>
                    {len.toLocaleString()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
              Learning Rate
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-muted-foreground/50 hover:text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3.5"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Step size for weight updates. Lower = slower but more stable.{" "}
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
            </FieldLabel>
            <div className="flex items-center gap-1">
              <button
                type="button"
                className="flex size-7 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-muted cursor-pointer"
                onClick={() => setLearningRate(stepLR(learningRate, -1))}
              >
                -
              </button>
              <span className="w-16 text-center font-mono text-sm">
                {formatLR(learningRate)}
              </span>
              <button
                type="button"
                className="flex size-7 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-muted cursor-pointer"
                onClick={() => setLearningRate(stepLR(learningRate, 1))}
              >
                +
              </button>
            </div>
          </div>

        </div>
      </FieldSet>

      {showLoraParams && (
        <>
          <Separator />
          <FieldSet>
            <FieldLegend variant="label">LoRA Parameters</FieldLegend>
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                  Rank
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Dimension of the low-rank matrices. Higher = more
                      capacity.{" "}
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
                </FieldLabel>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[loraRank]}
                    onValueChange={([v]) => setLoraRank(v)}
                    min={4}
                    max={128}
                    step={4}
                    className="w-40"
                  />
                  <input
                    type="number"
                    value={loraRank}
                    onChange={(e) => setLoraRank(Number(e.target.value))}
                    min={4}
                    max={128}
                    step={4}
                    className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                  Alpha
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Scaling factor. Typically set to 2x the rank value.{" "}
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
                </FieldLabel>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[loraAlpha]}
                    onValueChange={([v]) => setLoraAlpha(v)}
                    min={8}
                    max={256}
                    step={8}
                    className="w-40"
                  />
                  <input
                    type="number"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    min={8}
                    max={256}
                    step={8}
                    className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <FieldLabel className="flex items-center gap-1.5 !text-sm text-muted-foreground">
                  Dropout
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Probability of dropping neurons during training for
                      regularization.{" "}
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
                </FieldLabel>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[loraDropout]}
                    onValueChange={([v]) => setLoraDropout(v)}
                    min={0}
                    max={0.5}
                    step={0.01}
                    className="w-40"
                  />
                  <input
                    type="number"
                    value={loraDropout}
                    onChange={(e) => setLoraDropout(Number(e.target.value))}
                    min={0}
                    max={0.5}
                    step={0.01}
                    className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
                  />
                </div>
              </div>
            </div>
          </FieldSet>
        </>
      )}
    </FieldGroup>
  );
}
