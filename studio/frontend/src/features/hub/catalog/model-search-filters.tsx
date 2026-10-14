// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  AiChipIcon,
  Database02Icon,
  FilterIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useId, useState } from "react";
import {
  type ModelSearchFilters,
  hasModelSearchFilters,
} from "../lib/model-search-filters";
import type { ResourceTypeFilter } from "../types";

/** The Hub's filter popover: Models or Datasets, then (where the Hub search takes them) size,
 *  context and publisher limits. */
export function ModelSearchFiltersControl({
  value,
  onChange,
  resourceType,
  onResourceTypeChange,
  showModelFilters,
  className,
}: {
  value: ModelSearchFilters;
  onChange: (value: ModelSearchFilters) => void;
  resourceType: ResourceTypeFilter;
  onResourceTypeChange: (value: ResourceTypeFilter) => void;
  showModelFilters: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(value);
  const id = useId();
  const invalid =
    (draft.minParams !== undefined &&
      draft.maxParams !== undefined &&
      draft.minParams > draft.maxParams) ||
    (draft.minContext !== undefined &&
      draft.maxContext !== undefined &&
      draft.minContext > draft.maxContext);
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (next) {
          setDraft(value);
        }
        setOpen(next);
      }}
    >
      {/* Icon-sized so the Discover toolbar keeps its existing single-row layout. */}
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <PopoverTrigger asChild={true}>
            <button type="button" className={className} aria-label="Filters">
              <HugeiconsIcon
                icon={FilterIcon}
                strokeWidth={1.75}
                className="size-4"
              />
              {((showModelFilters && hasModelSearchFilters(value)) ||
                resourceType === "datasets") && (
                <span
                  aria-hidden="true"
                  className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-primary"
                />
              )}
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6}>
          Filters
        </TooltipContent>
      </Tooltip>
      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-80 max-w-[calc(100vw-2rem)]"
      >
        <div
          role="radiogroup"
          aria-label="Show"
          className={cn(
            "grid grid-cols-2 gap-1 rounded-full bg-muted p-1",
            showModelFilters && "mb-4",
          )}
        >
          {(
            [
              ["models", "Models", AiChipIcon],
              ["datasets", "Datasets", Database02Icon],
            ] as const
          ).map(([kind, label, icon]) => (
            <button
              key={kind}
              type="button"
              role="radio"
              aria-checked={resourceType === kind}
              onClick={() => onResourceTypeChange(kind)}
              className={cn(
                "inline-flex h-8 cursor-pointer items-center justify-center gap-1.5 rounded-full text-sm transition-colors",
                resourceType === kind
                  ? "bg-background text-foreground shadow-sm dark:bg-card"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={icon}
                strokeWidth={1.75}
                className="size-4"
              />
              {label}
            </button>
          ))}
        </div>
        {showModelFilters && (
          <form
            className="flex flex-col gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              if (invalid) {
                return;
              }
              onChange(draft);
              setOpen(false);
            }}
          >
            {(
              [
                ["Parameters (billions)", "minParams", "maxParams", 1e9],
                ["Context length (tokens)", "minContext", "maxContext", 1],
              ] as const
            ).map(([label, min, max, scale]) => (
              <fieldset key={min} className="space-y-2">
                <legend className="text-sm font-medium">{label}</legend>
                <div className="grid grid-cols-2 gap-2">
                  {([min, max] as const).map((key, index) => (
                    <label
                      key={key}
                      htmlFor={`${id}-${key}`}
                      className="space-y-1 text-xs text-muted-foreground"
                    >
                      <span>{index === 0 ? "Minimum" : "Maximum"}</span>
                      <Input
                        id={`${id}-${key}`}
                        aria-label={`${index === 0 ? "Minimum" : "Maximum"} ${label.toLowerCase()}`}
                        type="number"
                        min={0}
                        step={scale === 1 ? 1 : "any"}
                        placeholder="Any"
                        value={
                          draft[key] === undefined ? "" : draft[key] / scale
                        }
                        onChange={(event) =>
                          setDraft({
                            ...draft,
                            [key]:
                              event.target.value === ""
                                ? undefined
                                : Math.round(
                                    event.target.valueAsNumber * scale,
                                  ),
                          })
                        }
                      />
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={draft.organizationsOnly ?? false}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    organizationsOnly: event.target.checked,
                  })
                }
              />
              Organizations only (hide personal accounts)
            </label>
            <p className="text-xs text-muted-foreground">
              Ranges include both limits. Models without the requested metadata
              are excluded. Context is the published limit, not your device's
              capacity.
            </p>
            {invalid && (
              <p role="alert" className="text-xs text-destructive">
                Minimum must not exceed maximum.
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                className="rounded-full px-3 py-2 text-sm"
                onClick={() => {
                  setDraft({});
                  onChange({});
                  setOpen(false);
                }}
              >
                Clear
              </button>
              <button
                type="submit"
                disabled={invalid}
                className="rounded-full bg-primary px-3 py-2 text-sm text-primary-foreground disabled:opacity-50"
              >
                Apply filters
              </button>
            </div>
          </form>
        )}
      </PopoverContent>
    </Popover>
  );
}
