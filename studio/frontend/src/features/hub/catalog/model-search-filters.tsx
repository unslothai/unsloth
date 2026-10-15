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
  FilterMailIcon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useId, useState } from "react";
import {
  HUB_UPLOADERS,
  type HubUploader,
  type ModelSearchFilters,
  hasModelSearchFilters,
} from "../lib/model-search-filters";
import type { HubCapabilityFilter } from "../lib/model-capabilities";
import { CAPABILITY_FILTER_OPTIONS } from "../lib/view-models";

const UPLOADER_LABELS: Record<HubUploader, string> = {
  unsloth: "Unsloth",
  organization: "Organizations",
  personal: "Non-organization",
};

const ROUND_CHECKBOX =
  "pointer-events-none size-3.5 rounded-full dark:border-[color-mix(in_oklab,var(--foreground)_calc(60%*var(--contrast-edge-gain,1)),transparent)] dark:data-checked:border-primary [&_svg]:!size-2.5";

/** The Hub's filter popover: capabilities, then (where the Hub search takes them) size and
 *  context limits and the uploader. Ticks apply at once; the limits on Apply. */
export function ModelSearchFiltersControl({
  value,
  onChange,
  capabilities,
  onCapabilitiesChange,
  uploaders,
  uploadersChanged,
  onUploadersChange,
  showModelFilters,
  className,
}: {
  value: ModelSearchFilters;
  onChange: (value: ModelSearchFilters) => void;
  /** Empty means every box ticked (no filter); otherwise a model with any of the picks is listed. */
  capabilities: HubCapabilityFilter[];
  onCapabilitiesChange: (value: HubCapabilityFilter[]) => void;
  uploaders: readonly HubUploader[];
  /** False while the default applies, which leaves the icon without its dot. */
  uploadersChanged: boolean;
  onUploadersChange: (value: HubUploader[]) => void;
  showModelFilters: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(value);
  const id = useId();
  const formId = `${id}-limits`;
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
                icon={FilterMailIcon}
                strokeWidth={1.75}
                className="size-5"
              />
              {((showModelFilters &&
                (hasModelSearchFilters(value) || uploadersChanged)) ||
                capabilities.length > 0) && (
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
        className="hub-menu-instant menu-soft-surface w-[calc(288px*var(--ui-space-scale,1))] max-w-[calc(100vw-1rem)] rounded-[18px] p-3 pb-2.5 ring-0"
      >
        <fieldset className="flex flex-col gap-1">
          <legend className="mb-1.5 flex w-full items-baseline justify-between px-0.5 text-ui-12p5 font-medium text-foreground">
            Capabilities
            {capabilities.length > 0 && (
              <button
                type="button"
                onClick={() => onCapabilitiesChange([])}
                className="cursor-pointer text-ui-11 font-normal text-muted-foreground transition-colors hover:text-foreground"
              >
                Select all
              </button>
            )}
          </legend>
          <div className="grid grid-cols-2 gap-x-2">
            {CAPABILITY_FILTER_OPTIONS.map((option) => {
              const ticked = (key: HubCapabilityFilter) =>
                capabilities.length === 0 || capabilities.includes(key);
              const checked = ticked(option.value);
              // Kept in the list's order, so it doesn't depend on click order.
              const next = CAPABILITY_FILTER_OPTIONS.map((o) => o.value).filter(
                (key) => (key === option.value ? !checked : ticked(key)),
              );
              // The last tick stays: none would list nothing.
              const locked = next.length === 0;
              return (
                <button
                  key={option.value}
                  type="button"
                  role="checkbox"
                  aria-checked={checked}
                  aria-disabled={locked || undefined}
                  onClick={() => {
                    if (locked) return;
                    onCapabilitiesChange(
                      next.length === CAPABILITY_FILTER_OPTIONS.length
                        ? []
                        : next,
                    );
                  }}
                  className={cn(
                    "flex cursor-pointer select-none items-center gap-2 rounded-[10px] px-0.5 py-1.5 text-left text-ui-12p5 transition-colors hover:text-foreground aria-disabled:cursor-default",
                    checked ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  <Checkbox
                    checked={checked}
                    tabIndex={-1}
                    aria-hidden={true}
                    className={ROUND_CHECKBOX}
                  />
                  {option.label}
                </button>
              );
            })}
          </div>
        </fieldset>
        {showModelFilters && (
          <form
            id={formId}
            className="-mx-3 mt-2.5 flex flex-col gap-3 border-t border-border/60 px-3 pt-3"
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
                [
                  "Parameters",
                  "billions",
                  "minParams",
                  "maxParams",
                  1e9,
                  "Both limits are included. Models that don't publish a parameter count are left out.",
                ],
                [
                  "Context length",
                  "tokens",
                  "minContext",
                  "maxContext",
                  1,
                  "The model's published limit, not what your device can fit. Both limits are included; models that don't publish one are left out.",
                ],
              ] as const
            ).map(([label, unit, min, max, scale, about]) => (
              <fieldset key={min} className="flex flex-col gap-1.5">
                <legend className="mb-1.5 flex w-full items-center justify-between px-0.5 text-ui-12p5 font-medium text-foreground">
                  <span className="flex items-center gap-1">
                    {label}
                    <Tooltip>
                      <TooltipTrigger asChild={true}>
                        <button
                          type="button"
                          aria-label={`About ${label.toLowerCase()}`}
                          className="inline-flex cursor-help items-center text-muted-foreground transition-colors hover:text-foreground"
                        >
                          <HugeiconsIcon
                            icon={InformationCircleIcon}
                            strokeWidth={1.75}
                            className="size-3.5"
                          />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent
                        side="top"
                        sideOffset={6}
                        className="max-w-[calc(240px*var(--ui-space-scale,1))]"
                      >
                        {about}
                      </TooltipContent>
                    </Tooltip>
                  </span>
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
            {invalid && (
              <p role="alert" className="px-0.5 text-ui-11 text-destructive">
                Minimum must not exceed maximum.
              </p>
            )}
          </form>
        )}
        {showModelFilters && (
          <>
            <div className="-mx-3 mt-3 border-t border-border/60 px-3 pt-3">
              <fieldset className="flex flex-col">
                <legend className="mb-1.5 px-0.5 text-ui-12p5 font-medium text-foreground">
                  Uploader
                </legend>
                {HUB_UPLOADERS.map((uploader) => {
                  const checked = uploaders.includes(uploader);
                  const next = HUB_UPLOADERS.filter((key) =>
                    key === uploader ? !checked : uploaders.includes(key),
                  );
                  // The last tick stays: none would list nothing.
                  const locked = next.length === 0;
                  return (
                    <button
                      key={uploader}
                      type="button"
                      role="checkbox"
                      aria-checked={checked}
                      aria-disabled={locked || undefined}
                      onClick={() => {
                        if (!locked) onUploadersChange(next);
                      }}
                      className={cn(
                        "flex cursor-pointer select-none items-center gap-2 rounded-[10px] px-0.5 py-1.5 text-left text-ui-12p5 transition-colors hover:text-foreground aria-disabled:cursor-default",
                        checked ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      <Checkbox
                        checked={checked}
                        tabIndex={-1}
                        aria-hidden={true}
                        className={ROUND_CHECKBOX}
                      />
                      {UPLOADER_LABELS[uploader]}
                    </button>
                  );
                })}
              </fieldset>
            </div>
            <div className="-mx-3 flex justify-end gap-1.5 border-t border-border/60 px-3 pt-2.5">
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
              <Button type="submit" form={formId} size="sm" disabled={invalid}>
                Apply filters
              </Button>
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
