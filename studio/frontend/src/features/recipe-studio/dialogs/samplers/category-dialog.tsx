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
            label="Values"
            hint="Define allowed categorical values for this column."
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
            placeholder="Type a value and press Enter"
          />
        </div>
      </div>
      <Collapsible
        open={advancedOpen}
        onOpenChange={(open) => onUpdate({ advancedOpen: open })}
      >
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Advanced list settings"
            open={advancedOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-3">
            <div className="grid gap-1.5">
              <FieldLabel
                label="Weights (optional)"
                hint="Set selection probability per value."
              />
              {(config.values ?? []).length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  Add values first, then set optional weights.
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
                        placeholder="Weight"
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
                label="Conditional params (category)"
                hint="Override category values/weights when condition matches."
              />
              <span className="text-xs text-muted-foreground">
                {conditionalCount} rules
              </span>
            </div>
            <div className="flex gap-2">
              <Input
                id={conditionInputId}
                className="nodrag"
                placeholder="Condition (e.g., {{ region }} == 'US')"
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
                Add rule
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
                    Remove
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
                  placeholder="Type a conditional value and press Enter"
                />
                <div className="grid gap-1.5">
                  <p className="text-xs font-semibold uppercase text-muted-foreground">
                    Rule weights (optional)
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
                          placeholder="Weight"
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
