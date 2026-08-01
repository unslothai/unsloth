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
import {
  TRAINING_METHOD_META,
  TRAINING_METHOD_ORDER,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import type { TrainingMethod } from "@/types/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { HfTokenField } from "../hf-token-field";

export function ModelSelectionStep() {
  const t = useT();
  const {
    ensureModelDefaultsLoaded,
    modelType,
    selectedModel,
    trainingMethod,
    setTrainingMethod,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      ensureModelDefaultsLoaded: state.ensureModelDefaultsLoaded,
      modelType: state.modelType,
      selectedModel: state.selectedModel,
      trainingMethod: state.trainingMethod,
      setTrainingMethod: state.setTrainingMethod,
    })),
  );

  useEffect(() => {
    if (selectedModel) {
      ensureModelDefaultsLoaded();
    }
  }, [ensureModelDefaultsLoaded, selectedModel]);

  return (
    <FieldGroup>
      <HfTokenField />

      <Field>
        <FieldLabel>{t("studio.training.chooseModel")}</FieldLabel>
        <FieldDescription>
          {t("studio.wizard.modelPickerDescription")}
        </FieldDescription>
        <TrainModelSelector requiredModelType={modelType ?? undefined} />
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
                {TRAINING_METHOD_ORDER.map((method) => {
                  const meta = TRAINING_METHOD_META[method];
                  const note =
                    method === "qlora" || method === "lora"
                      ? t(meta.noteKey)
                      : null;
                  return (
                    <SelectItem key={method} value={method}>
                      {t(meta.labelKey)}
                      {note ? ` (${note})` : null}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>
        </Field>
      )}
    </FieldGroup>
  );
}
