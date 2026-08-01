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
import { useT } from "@/i18n";
import type { TrainingMethod } from "@/types/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";
import { HfTokenField } from "../hf-token-field";

export function ModelSelectionStep() {
  const t = useT();
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
        <FieldLabel>{t("studio.training.chooseModel")}</FieldLabel>
        <FieldDescription>
          {t("studio.wizard.modelPickerDescription")}
        </FieldDescription>
        <TrainModelSelector />
      </Field>

      {selectedModel && (
        <Field>
          <div className="flex items-center justify-between gap-4">
            <div>
              <FieldLabel className="flex items-center gap-1.5">
                {t("studio.wizard.trainingMethod")}
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label={t("studio.wizard.trainingMethod")}
                      className="text-muted-foreground/50 hover:text-muted-foreground"
                    >
                      <HugeiconsIcon
                        icon={InformationCircleIcon}
                        className="size-3.5"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    {t("studio.wizard.trainingMethodTooltip")}{" "}
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
              </FieldLabel>
              <FieldDescription>
                {t("studio.wizard.trainingMethodDescription", {
                  model: selectedModel,
                })}
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
                <SelectItem value="qlora">
                  {t("studio.methods.qlora.label")} (
                  {t("studio.methods.qlora.note")})
                </SelectItem>
                <SelectItem value="lora">
                  {t("studio.methods.lora.label")} (
                  {t("studio.methods.lora.note")})
                </SelectItem>
                <SelectItem value="full">
                  {t("studio.methods.full.label")}
                </SelectItem>
                <SelectItem value="cpt">
                  {t("studio.methods.cpt.label")}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Field>
      )}
    </FieldGroup>
  );
}
