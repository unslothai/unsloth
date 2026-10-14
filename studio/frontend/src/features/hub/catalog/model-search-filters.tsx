// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
        collisionPadding={12}
        onCloseAutoFocus={(event) => event.preventDefault()}
        className="hub-menu-instant menu-soft-surface w-[calc(288px*var(--ui-space-scale,1))] max-w-[calc(100vw-1rem)] rounded-[18px] p-1.5 ring-0"
      >
        <div
          role="radiogroup"
          aria-label="Show"
          className="grid grid-cols-2 gap-1 rounded-full bg-muted p-1"
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
                "inline-flex h-7 cursor-pointer items-center justify-center gap-1.5 rounded-full text-ui-12p5 transition-colors",
                resourceType === kind
                  ? "bg-background text-foreground shadow-sm dark:bg-card"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
              {label}
            </button>
          ))}
        </div>
        {showModelFilters && (
          <form
            className="flex flex-col gap-3 pt-3"
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
                ["Parameters", "billions", "minParams", "maxParams", 1e9],
                ["Context length", "tokens", "minContext", "maxContext", 1],
              ] as const
            ).map(([label, unit, min, max, scale]) => (
              <fieldset key={min} className="flex flex-col gap-1.5">
                <legend className="mb-1.5 flex w-full items-baseline justify-between px-0.5 text-ui-12p5 font-medium text-foreground">
                  {label}
                  <span className="text-ui-11 font-normal text-muted-foreground">
                    {unit}
                  </span>
                </legend>
                <div className="grid grid-cols-2 gap-1.5">
                  {([min, max] as const).map((key, index) => (
                    <Input
                      key={key}
                      id={`${id}-${key}`}
                      aria-label={`${index === 0 ? "Minimum" : "Maximum"} ${label.toLowerCase()}`}
                      type="number"
                      min={0}
                      step={scale === 1 ? 1 : "any"}
                      placeholder={index === 0 ? "Min" : "Max"}
                      className="h-8 pl-3 text-ui-12p5 md:text-ui-12p5"
                      value={draft[key] === undefined ? "" : draft[key] / scale}
                      onChange={(event) =>
                        setDraft({
                          ...draft,
                          [key]:
                            event.target.value === ""
                              ? undefined
                              : Math.round(event.target.valueAsNumber * scale),
                        })
                      }
                    />
                  ))}
                </div>
              </fieldset>
            ))}
            <button
              type="button"
              role="checkbox"
              aria-checked={draft.organizationsOnly ?? false}
              onClick={(event) => {
                // Inside a form, Radix's Checkbox syncs a hidden input with a dispatched click that
                // bubbles here; toggling on it again looped until React gave up (error #185).
                if (!event.isTrusted) return;
                setDraft({
                  ...draft,
                  organizationsOnly: !(draft.organizationsOnly ?? false),
                });
              }}
              className="-mx-1.5 flex cursor-pointer select-none items-center gap-2 rounded-[10px] px-1.5 py-1.5 text-left text-ui-12p5 text-muted-foreground transition-colors hover:text-foreground"
            >
              <Checkbox
                checked={draft.organizationsOnly ?? false}
                tabIndex={-1}
                aria-hidden={true}
                className="pointer-events-none size-3.5 rounded-full dark:border-[color-mix(in_oklab,var(--foreground)_calc(60%*var(--contrast-edge-gain,1)),transparent)] dark:data-checked:border-primary [&_svg]:!size-2.5"
              />
              Organizations only (hide personal accounts)
            </button>
            <p className="px-0.5 text-ui-11 leading-snug text-muted-foreground">
              Ranges include both limits. Models without the requested metadata
              are excluded. Context is the published limit, not your device's
              capacity.
            </p>
            {invalid && (
              <p role="alert" className="px-0.5 text-ui-11 text-destructive">
                Minimum must not exceed maximum.
              </p>
            )}
            <div className="-mx-1.5 flex justify-end gap-1.5 border-t border-border/60 px-1.5 pb-0.5 pt-2.5">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  setDraft({});
                  onChange({});
                  setOpen(false);
                }}
              >
                Clear
              </Button>
              <Button type="submit" size="sm" disabled={invalid}>
                Apply filters
              </Button>
            </div>
          </form>
        )}
      </PopoverContent>
    </Popover>
  );
}
