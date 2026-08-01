// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TrainModelSelector } from "@/features/train-model-picker";
import { useTrainingConfigStore } from "@/features/training";
import type { TrainingMethod } from "@/types/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";
import { HfTokenField } from "../hf-token-field";

export function ModelSelectionStep() {
  const { selectedModel, trainingMethod, setTrainingMethod } =
    useTrainingConfigStore(
      useShallow((state) => ({
        selectedModel: state.selectedModel,
        trainingMethod: state.trainingMethod,
        setTrainingMethod: state.setTrainingMethod,
      })),
    );

  return (
    <FieldGroup>
      <HfTokenField />

      <Field>
        <FieldLabel>Choose a model</FieldLabel>
        <FieldDescription>
          Search Hugging Face or choose a trainable model already on this
          device.
        </FieldDescription>
        <TrainModelSelector />
      </Field>

      {selectedModel && (
        <Field>
          <div className="flex items-center justify-between gap-4">
            <div>
              <FieldLabel className="flex items-center gap-1.5">
                Training method
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
                  <TooltipContent className="max-w-xs">
                    QLoRA uses 4-bit quantization for the lowest VRAM use. LoRA
                    uses 16-bit weights, while full fine-tuning updates every
                    weight.{" "}
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
              <FieldDescription>
                Choose how to fine-tune {selectedModel}
              </FieldDescription>
            </div>
            <Select
              value={trainingMethod}
              onValueChange={(value) =>
                setTrainingMethod(value as TrainingMethod)
              }
            >
              <SelectTrigger className="w-44 shrink-0">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="qlora">QLoRA (4-bit)</SelectItem>
                <SelectItem value="lora">LoRA (16-bit)</SelectItem>
                <SelectItem value="full">Full Fine-tune</SelectItem>
                <SelectItem value="cpt">Continued Pretraining</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Field>
      )}
    </FieldGroup>
  );
}
