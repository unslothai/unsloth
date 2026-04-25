// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { type ReactElement, useState } from "react";
import { useI18n } from "@/features/i18n";
import type { SamplerConfig } from "../../types";
import { ChipInput } from "../../components/chip-input";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type CategoryDialogProps = {
  config: SamplerConfig;
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

function addChipWithWeight(
  values: string[] | undefined,
  weights: Array<number | null> | undefined,
  value: string,
): { values: string[]; weights: Array<number | null> } {
  return {
    values: [...(values ?? []), value],
    weights: [...(weights ?? []), null],
  };
}

function removeChipWithWeight(
  values: string[] | undefined,
  weights: Array<number | null> | undefined,
  index: number,
): { values: string[]; weights: Array<number | null> } {
  const nextValues = [...(values ?? [])];
  const nextWeights = [...(weights ?? [])];
  nextValues.splice(index, 1);
  nextWeights.splice(index, 1);
  return { values: nextValues, weights: nextWeights };
}

export function CategoryDialog({
  config,
  onUpdate,
}: CategoryDialogProps): ReactElement {
  const { t } = useI18n();
  const [conditionDraft, setConditionDraft] = useState("");
  const advancedOpen = config.advancedOpen === true;
  const conditionInputId = `${config.id}-conditional-rule`;
  const conditional = config.conditional_params ?? {};
  const conditionalCount = Object.keys(conditional).length;

  const handleAddCondition = () => {
    const condition = conditionDraft.trim();
    if (!condition || conditional[condition]) {
      return;
    }
    onUpdate({
      // biome-ignore lint/style/useNamingConvention: api schema
      conditional_params: {
        ...conditional,
        [condition]: {
          // biome-ignore lint/style/useNamingConvention: api schema
          sampler_type: "category",
          values: [],
          weights: [],
        },
      },
    });
    setConditionDraft("");
  };

  const removeCondition = (condition: string) => {
    const next = { ...conditional };
    delete next[condition];
    onUpdate({
      // biome-ignore lint/style/useNamingConvention: api schema
      conditional_params: Object.keys(next).length > 0 ? next : undefined,
    });
  };

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="space-y-3">
        <div className="grid gap-1.5">
          <FieldLabel
            label={t("recipe.sampler.category.values")}
            hint={t("recipe.sampler.category.valuesHint")}
          />
          <ChipInput
            values={config.values ?? []}
            onAdd={(value) => {
              const { values, weights } = addChipWithWeight(
                config.values,
                config.weights,
                value,
              );
              onUpdate({ values, weights });
            }}
            onRemove={(index) => {
              const { values, weights } = removeChipWithWeight(
                config.values,
                config.weights,
                index,
              );
              onUpdate({ values, weights });
            }}
            placeholder={t("recipe.sampler.category.valuesPlaceholder")}
          />
        </div>
      </div>
      <Collapsible
        open={advancedOpen}
        onOpenChange={(open) => onUpdate({ advancedOpen: open })}
      >
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label={t("recipe.sampler.category.advanced")}
            open={advancedOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-3">
            <div className="grid gap-1.5">
              <FieldLabel
                label={t("recipe.sampler.category.weights")}
                hint={t("recipe.sampler.category.weightsHint")}
              />
              {(config.values ?? []).length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  {t("recipe.sampler.category.weightsEmpty")}
                </p>
              ) : (
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {(config.values ?? []).map((value, index) => (
                    <div key={`${value}-weight`} className="space-y-1">
                      <p
                        className="truncate text-xs text-muted-foreground"
                        title={value}
                      >
                        {value}
                      </p>
                      <Input
                        type="number"
                        className="nodrag w-full"
                        placeholder={t("recipe.sampler.category.weightPlaceholder")}
                        value={config.weights?.[index] ?? ""}
                        onChange={(event) => {
                          const weights = [...(config.weights ?? [])];
                          weights[index] = event.target.value
                            ? Number(event.target.value)
                            : null;
                          onUpdate({ weights });
                        }}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="flex items-center justify-between gap-2">
              <FieldLabel
                label={t("recipe.sampler.category.conditional")}
                hint={t("recipe.sampler.category.conditionalHint")}
              />
              <span className="text-xs text-muted-foreground">
                {conditionalCount} {t("recipe.sampler.category.rules")}
              </span>
            </div>
            <div className="flex gap-2">
              <Input
                id={conditionInputId}
                className="nodrag"
                placeholder={t("recipe.sampler.category.conditionPlaceholder")}
                value={conditionDraft}
                onChange={(event) => setConditionDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    handleAddCondition();
                  }
                }}
              />
              <Button type="button" size="sm" onClick={handleAddCondition}>
                {t("recipe.sampler.category.addRule")}
              </Button>
            </div>
            {Object.entries(conditional).map(([condition, params]) => (
              <div
                key={condition}
                className="space-y-3 rounded-2xl border border-border/60 p-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-semibold text-foreground">{condition}</p>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={() => removeCondition(condition)}
                  >
                    {t("recipe.sampler.category.remove")}
                  </Button>
                </div>
                <ChipInput
                  values={params.values ?? []}
                  onAdd={(value) => {
                    const { values, weights } = addChipWithWeight(
                      params.values,
                      params.weights,
                      value,
                    );
                    onUpdate({
                      // biome-ignore lint/style/useNamingConvention: api schema
                      conditional_params: {
                        ...conditional,
                        [condition]: { ...params, values, weights },
                      },
                    });
                  }}
                  onRemove={(index) => {
                    const { values, weights } = removeChipWithWeight(
                      params.values,
                      params.weights,
                      index,
                    );
                    onUpdate({
                      // biome-ignore lint/style/useNamingConvention: api schema
                      conditional_params: {
                        ...conditional,
                        [condition]: { ...params, values, weights },
                      },
                    });
                  }}
                  placeholder={t("recipe.sampler.category.conditionalValuePlaceholder")}
                />
                <div className="grid gap-1.5">
                  <p className="text-xs font-semibold uppercase text-muted-foreground">
                    {t("recipe.sampler.category.ruleWeights")}
                  </p>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {(params.values ?? []).map((value, index) => (
                      <div
                        key={`${condition}-${value}-${index}-weight`}
                        className="space-y-1"
                      >
                        <p
                          className="truncate text-xs text-muted-foreground"
                          title={value}
                        >
                          {value}
                        </p>
                        <Input
                          type="number"
                          className="nodrag"
                          placeholder={t("recipe.sampler.category.weightPlaceholder")}
                          value={params.weights?.[index] ?? ""}
                          onChange={(event) => {
                            const weights = [
                              ...(params.weights ??
                                Array.from(
                                  { length: (params.values ?? []).length },
                                  () => null,
                                )),
                            ];
                            weights[index] = event.target.value
                              ? Number(event.target.value)
                              : null;
                            onUpdate({
                              // biome-ignore lint/style/useNamingConvention: api schema
                              conditional_params: {
                                ...conditional,
                                [condition]: { ...params, weights },
                              },
                            });
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
