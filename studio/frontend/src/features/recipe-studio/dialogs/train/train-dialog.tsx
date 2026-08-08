// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Switch } from "@/components/ui/switch";
import {
  LR_SCHEDULER_OPTIONS,
  OPTIMIZER_OPTIONS,
  PRIORITY_TRAINING_MODELS,
} from "@/config/training";
import type {
  DatasetFormat,
  GradientCheckpointing,
  TrainingMethod,
} from "@/types/training";
import {
  type ChangeEvent,
  type ReactElement,
  useEffect,
  useState,
} from "react";
import type {
  TrainingCardConfig,
  TrainingCardDatasetSource,
  TrainingCardLoraVariant,
} from "../../types";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type TrainDialogProps = {
  config: TrainingCardConfig;
  onUpdate: (patch: Partial<TrainingCardConfig>) => void;
};

const METHOD_OPTIONS: ReadonlyArray<{ value: TrainingMethod; label: string }> =
  [
    { value: "qlora", label: "QLoRA (4-bit, lowest memory)" },
    { value: "lora", label: "LoRA (16-bit adapters)" },
    { value: "full", label: "Full fine-tuning" },
    { value: "cpt", label: "Continued pretraining" },
  ];

const LORA_VARIANT_OPTIONS: ReadonlyArray<{
  value: TrainingCardLoraVariant;
  label: string;
}> = [
  { value: "lora", label: "Standard" },
  { value: "rslora", label: "Rank-stabilized (rsLoRA)" },
  { value: "loftq", label: "LoftQ" },
];

const DATASET_FORMAT_OPTIONS: ReadonlyArray<{
  value: DatasetFormat;
  label: string;
}> = [
  { value: "auto", label: "Auto-detect" },
  { value: "alpaca", label: "Alpaca" },
  { value: "chatml", label: "ChatML" },
  { value: "sharegpt", label: "ShareGPT" },
  { value: "raw", label: "Raw text" },
];

const GRADIENT_CHECKPOINTING_OPTIONS: ReadonlyArray<{
  value: GradientCheckpointing;
  label: string;
}> = [
  { value: "unsloth", label: "Unsloth (recommended)" },
  { value: "true", label: "Standard" },
  { value: "none", label: "Off" },
];

/** Numeric input that keeps a text draft while focused, committing on blur. */
function NumberField({
  id,
  value,
  onCommit,
  disabled,
  placeholder,
}: {
  id: string;
  value: number;
  onCommit: (value: number) => void;
  disabled?: boolean;
  placeholder?: string;
}): ReactElement {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commit = () => {
    const parsed = Number.parseFloat(draft);
    if (Number.isFinite(parsed) && parsed !== value) {
      onCommit(parsed);
    } else {
      setDraft(String(value));
    }
  };

  return (
    <Input
      id={id}
      inputMode="decimal"
      className="nodrag"
      value={draft}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(event: ChangeEvent<HTMLInputElement>) =>
        setDraft(event.target.value)
      }
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

export function TrainDialog({
  config,
  onUpdate,
}: TrainDialogProps): ReactElement {
  const advancedOpen = config.advancedOpen ?? false;
  const idFor = (suffix: string) => `${config.id}-${suffix}`;

  return (
    <div className="space-y-4">
      <NameField
        label="Train step name"
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />

      <div className="corner-squircle rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
        <p className="text-sm font-semibold text-foreground">
          Fine-tune a model on this recipe's dataset
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Connect the recipe output into this card. When the graph runs, the
          generated dataset is trained on with the settings below.
        </p>
      </div>

      {/* Model + method */}
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Base model"
            htmlFor={idFor("model")}
            hint="Hugging Face model id to fine-tune, e.g. unsloth/Llama-3.2-3B-Instruct."
          />
          <Input
            id={idFor("model")}
            className="nodrag"
            list={idFor("model-suggestions")}
            placeholder="unsloth/Llama-3.2-3B-Instruct"
            value={config.baseModel}
            onChange={(event) => onUpdate({ baseModel: event.target.value })}
          />
          <datalist id={idFor("model-suggestions")}>
            {PRIORITY_TRAINING_MODELS.map((model) => (
              <option key={model} value={model} />
            ))}
          </datalist>
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Training method"
            htmlFor={idFor("method")}
            hint="QLoRA is the most memory efficient. Full fine-tuning updates all weights."
          />
          <Select
            value={config.trainingMethod}
            onValueChange={(value) =>
              onUpdate({ trainingMethod: value as TrainingMethod })
            }
          >
            <SelectTrigger id={idFor("method")} className="nodrag">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {METHOD_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Dataset source */}
      <div className="grid gap-1.5">
        <FieldLabel
          label="Dataset"
          htmlFor={idFor("dataset-source")}
          hint="Auto-wire the generated dataset from the connected recipe, or override it."
        />
        <Select
          value={config.datasetSource}
          onValueChange={(value) =>
            onUpdate({ datasetSource: value as TrainingCardDatasetSource })
          }
        >
          <SelectTrigger id={idFor("dataset-source")} className="nodrag">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="recipe">Auto-wire from recipe output</SelectItem>
            <SelectItem value="huggingface">
              Hugging Face dataset (override)
            </SelectItem>
            <SelectItem value="upload">Local file path (override)</SelectItem>
          </SelectContent>
        </Select>
        {config.datasetSource === "recipe" && (
          <p className="text-xs text-muted-foreground">
            Uses the artifact from the connected recipe's most recent completed
            run.
          </p>
        )}
      </div>

      {config.datasetSource === "huggingface" && (
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="grid gap-1.5 sm:col-span-3">
            <FieldLabel label="Dataset id" htmlFor={idFor("hf-dataset")} />
            <Input
              id={idFor("hf-dataset")}
              className="nodrag"
              placeholder="unsloth/OpenMathReasoning-mini"
              value={config.hfDataset}
              onChange={(event) => onUpdate({ hfDataset: event.target.value })}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel label="Subset" htmlFor={idFor("hf-subset")} />
            <Input
              id={idFor("hf-subset")}
              className="nodrag"
              placeholder="default"
              value={config.hfSubset}
              onChange={(event) => onUpdate({ hfSubset: event.target.value })}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel label="Split" htmlFor={idFor("hf-split")} />
            <Input
              id={idFor("hf-split")}
              className="nodrag"
              placeholder="train"
              value={config.hfSplit}
              onChange={(event) => onUpdate({ hfSplit: event.target.value })}
            />
          </div>
        </div>
      )}

      {config.datasetSource === "upload" && (
        <div className="grid gap-1.5">
          <FieldLabel
            label="Dataset file path"
            htmlFor={idFor("upload")}
            hint="Absolute path to a local JSON, JSONL, or CSV dataset file."
          />
          <Input
            id={idFor("upload")}
            className="nodrag"
            placeholder="/path/to/dataset.jsonl"
            value={config.uploadedFile}
            onChange={(event) => onUpdate({ uploadedFile: event.target.value })}
          />
        </div>
      )}

      <div className="grid gap-1.5">
        <FieldLabel
          label="Dataset format"
          htmlFor={idFor("dataset-format")}
          hint="How rows map to the chat template. Auto-detect works for most datasets."
        />
        <Select
          value={config.datasetFormat}
          onValueChange={(value) =>
            onUpdate({ datasetFormat: value as DatasetFormat })
          }
        >
          <SelectTrigger id={idFor("dataset-format")} className="nodrag">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DATASET_FORMAT_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Core hyperparameters */}
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <FieldLabel label="Epochs" htmlFor={idFor("epochs")} />
          <NumberField
            id={idFor("epochs")}
            value={config.epochs}
            onCommit={(value) => onUpdate({ epochs: value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Max sequence length"
            htmlFor={idFor("seq")}
            hint="Longest input the model trains on, in tokens."
          />
          <NumberField
            id={idFor("seq")}
            value={config.contextLength}
            onCommit={(value) => onUpdate({ contextLength: value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="LoRA rank"
            htmlFor={idFor("rank")}
            hint="Higher rank = more trainable parameters."
          />
          <NumberField
            id={idFor("rank")}
            value={config.loraRank}
            onCommit={(value) => onUpdate({ loraRank: value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel label="Batch size" htmlFor={idFor("batch")} />
          <NumberField
            id={idFor("batch")}
            value={config.batchSize}
            onCommit={(value) => onUpdate({ batchSize: value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel label="Learning rate" htmlFor={idFor("lr")} />
          <NumberField
            id={idFor("lr")}
            value={config.learningRate}
            onCommit={(value) => onUpdate({ learningRate: value })}
          />
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Output adapter name"
            htmlFor={idFor("output")}
            hint="Names the run and the saved adapter directory."
          />
          <Input
            id={idFor("output")}
            className="nodrag"
            placeholder="my-finetune"
            value={config.outputName}
            onChange={(event) => onUpdate({ outputName: event.target.value })}
          />
        </div>
      </div>

      {/* Advanced */}
      <Collapsible
        open={advancedOpen}
        onOpenChange={(open) => onUpdate({ advancedOpen: open })}
      >
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Advanced training settings"
            open={advancedOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <FieldLabel label="LoRA alpha" htmlFor={idFor("alpha")} />
              <NumberField
                id={idFor("alpha")}
                value={config.loraAlpha}
                onCommit={(value) => onUpdate({ loraAlpha: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel label="LoRA dropout" htmlFor={idFor("dropout")} />
              <NumberField
                id={idFor("dropout")}
                value={config.loraDropout}
                onCommit={(value) => onUpdate({ loraDropout: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel label="LoRA variant" htmlFor={idFor("variant")} />
              <Select
                value={config.loraVariant}
                onValueChange={(value) =>
                  onUpdate({ loraVariant: value as TrainingCardLoraVariant })
                }
              >
                <SelectTrigger id={idFor("variant")} className="nodrag">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LORA_VARIANT_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-1.5">
              <FieldLabel
                label="Gradient accumulation"
                htmlFor={idFor("grad-accum")}
              />
              <NumberField
                id={idFor("grad-accum")}
                value={config.gradientAccumulation}
                onCommit={(value) => onUpdate({ gradientAccumulation: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel label="Warmup steps" htmlFor={idFor("warmup")} />
              <NumberField
                id={idFor("warmup")}
                value={config.warmupSteps}
                onCommit={(value) => onUpdate({ warmupSteps: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel
                label="Max steps"
                htmlFor={idFor("max-steps")}
                hint="Overrides epochs when greater than zero."
              />
              <NumberField
                id={idFor("max-steps")}
                value={config.maxSteps}
                onCommit={(value) => onUpdate({ maxSteps: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel
                label="Weight decay"
                htmlFor={idFor("weight-decay")}
              />
              <NumberField
                id={idFor("weight-decay")}
                value={config.weightDecay}
                onCommit={(value) => onUpdate({ weightDecay: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel label="Random seed" htmlFor={idFor("seed")} />
              <NumberField
                id={idFor("seed")}
                value={config.randomSeed}
                onCommit={(value) => onUpdate({ randomSeed: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel label="Optimizer" htmlFor={idFor("optimizer")} />
              <Select
                value={config.optimizerType}
                onValueChange={(value) => onUpdate({ optimizerType: value })}
              >
                <SelectTrigger id={idFor("optimizer")} className="nodrag">
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
            </div>
            <div className="grid gap-1.5">
              <FieldLabel label="LR scheduler" htmlFor={idFor("scheduler")} />
              <Select
                value={config.lrSchedulerType}
                onValueChange={(value) => onUpdate({ lrSchedulerType: value })}
              >
                <SelectTrigger id={idFor("scheduler")} className="nodrag">
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
            </div>
            <div className="grid gap-1.5">
              <FieldLabel
                label="Gradient checkpointing"
                htmlFor={idFor("grad-ckpt")}
              />
              <Select
                value={config.gradientCheckpointing}
                onValueChange={(value) =>
                  onUpdate({
                    gradientCheckpointing: value as GradientCheckpointing,
                  })
                }
              >
                <SelectTrigger id={idFor("grad-ckpt")} className="nodrag">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {GRADIENT_CHECKPOINTING_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid gap-3">
            <label
              htmlFor={idFor("packing")}
              className="flex items-center justify-between gap-3"
            >
              <span className="text-xs font-semibold uppercase text-muted-foreground">
                Sequence packing
              </span>
              <Switch
                id={idFor("packing")}
                checked={config.packing}
                onCheckedChange={(value) => onUpdate({ packing: value })}
              />
            </label>
            <label
              htmlFor={idFor("train-completions")}
              className="flex items-center justify-between gap-3"
            >
              <span className="text-xs font-semibold uppercase text-muted-foreground">
                Train on completions only
              </span>
              <Switch
                id={idFor("train-completions")}
                checked={config.trainOnCompletions}
                onCheckedChange={(value) =>
                  onUpdate({ trainOnCompletions: value })
                }
              />
            </label>
          </div>

          <div className="grid gap-1.5">
            <FieldLabel
              label="Hugging Face token"
              htmlFor={idFor("hf-token")}
              hint="Needed for gated base models or private datasets."
            />
            <Input
              id={idFor("hf-token")}
              type="password"
              className="nodrag"
              placeholder="hf_..."
              value={config.hfToken}
              onChange={(event) => onUpdate({ hfToken: event.target.value })}
            />
          </div>

          <div className="grid gap-3">
            <label
              htmlFor={idFor("wandb")}
              className="flex items-center justify-between gap-3"
            >
              <span className="text-xs font-semibold uppercase text-muted-foreground">
                Log to Weights & Biases
              </span>
              <Switch
                id={idFor("wandb")}
                checked={config.enableWandb}
                onCheckedChange={(value) => onUpdate({ enableWandb: value })}
              />
            </label>
            {config.enableWandb && (
              <div className="grid gap-1.5">
                <FieldLabel
                  label="W&B project"
                  htmlFor={idFor("wandb-project")}
                />
                <Input
                  id={idFor("wandb-project")}
                  className="nodrag"
                  value={config.wandbProject}
                  onChange={(event) =>
                    onUpdate({ wandbProject: event.target.value })
                  }
                />
              </div>
            )}
            <label
              htmlFor={idFor("tensorboard")}
              className="flex items-center justify-between gap-3"
            >
              <span className="text-xs font-semibold uppercase text-muted-foreground">
                Log to TensorBoard
              </span>
              <Switch
                id={idFor("tensorboard")}
                checked={config.enableTensorboard}
                onCheckedChange={(value) =>
                  onUpdate({ enableTensorboard: value })
                }
              />
            </label>
            {config.enableTensorboard && (
              <div className="grid gap-1.5">
                <FieldLabel
                  label="TensorBoard directory"
                  htmlFor={idFor("tensorboard-dir")}
                />
                <Input
                  id={idFor("tensorboard-dir")}
                  className="nodrag"
                  value={config.tensorboardDir}
                  onChange={(event) =>
                    onUpdate({ tensorboardDir: event.target.value })
                  }
                />
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
