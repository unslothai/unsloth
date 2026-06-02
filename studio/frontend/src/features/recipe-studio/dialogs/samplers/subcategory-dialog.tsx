// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type ReactElement, useCallback, useEffect, useMemo } from "react";
import type { SamplerConfig } from "../../types";
import { ChipInput } from "../../components/chip-input";
import { NameField } from "../shared/name-field";
import { FieldLabel } from "../shared/field-label";

type SubcategoryDialogProps = {
  config: SamplerConfig;
  categoryOptions: SamplerConfig[];
  onUpdate: (patch: Partial<SamplerConfig>) => void;
};

export function SubcategoryDialog({
  config,
  categoryOptions,
  onUpdate,
}: SubcategoryDialogProps): ReactElement {
  const parentSelectId = `${config.id}-parent-category`;
  const updateField = useCallback(
    <K extends keyof SamplerConfig>(key: K, value: SamplerConfig[K]) => {
      onUpdate({ [key]: value } as Partial<SamplerConfig>);
    },
    [onUpdate],
  );
  const parent = useMemo(
    () =>
      categoryOptions.find(
        (option) => option.name === config.subcategory_parent,
      ) ?? null,
    [categoryOptions, config.subcategory_parent],
  );
  const categoryValues = parent?.values ?? [];
  const mapping = config.subcategory_mapping ?? {};

  const ensureMapping = useCallback(
    (nextParent?: SamplerConfig | null) => {
      const values = nextParent?.values ?? [];
      const nextMapping: Record<string, string[]> = {};
      for (const value of values) {
        nextMapping[value] = config.subcategory_mapping?.[value] ?? [];
      }
      const currentKeys = Object.keys(config.subcategory_mapping ?? {});
      const nextKeys = Object.keys(nextMapping);
      const changed =
        currentKeys.length !== nextKeys.length ||
        currentKeys.some((key) => !nextKeys.includes(key));
      if (changed) {
        updateField("subcategory_mapping", nextMapping);
      }
    },
    [config.subcategory_mapping, updateField],
  );

  useEffect(() => {
    if (parent) {
      ensureMapping(parent);
    }
  }, [ensureMapping, parent]);

  return (
    <div className="space-y-4">
      <NameField
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="space-y-3">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Parent category column"
            htmlFor={parentSelectId}
            hint="Category column this block maps from."
          />
          <Select
            value={config.subcategory_parent ?? ""}
            onValueChange={(value) => {
              const nextParent =
                categoryOptions.find((option) => option.name === value) ?? null;
              updateField("subcategory_parent", value);
              ensureMapping(nextParent);
            }}
          >
            <SelectTrigger className="nodrag w-full" id={parentSelectId}>
              <SelectValue placeholder="Select category column" />
            </SelectTrigger>
            <SelectContent>
              {categoryOptions.map((option) => (
                <SelectItem key={option.id} value={option.name}>
                  {option.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Map each parent category value to its subcategory options below.
          </p>
        </div>
        {categoryValues.length > 0 && (
          <div className="grid gap-4">
            {categoryValues.map((value) => (
              <div key={value}>
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-foreground">
                    {value}
                  </p>
                  <span className="text-xs text-muted-foreground">
                    {mapping[value]?.length ?? 0} subvalues
                  </span>
                </div>
                <ChipInput
                  values={mapping[value] ?? []}
                  onAdd={(item) => {
                    const next = { ...mapping };
                    const list = next[value] ? [...next[value]] : [];
                    list.push(item);
                    next[value] = list;
                    updateField("subcategory_mapping", next);
                  }}
                  onRemove={(index) => {
                    const next = { ...mapping };
                    const list = [...(next[value] ?? [])];
                    list.splice(index, 1);
                    next[value] = list;
                    updateField("subcategory_mapping", next);
                  }}
                  placeholder="Type subcategory and press Enter"
                />
                {(mapping[value] ?? []).length === 0 && (
                  <p className="mt-2 text-xs text-rose-500">
                    Add at least 1 subcategory.
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
