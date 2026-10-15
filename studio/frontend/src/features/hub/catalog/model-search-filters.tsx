// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
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
  BrainIcon,
  FilterMailIcon,
  FlimSlateIcon,
  HeadphonesIcon,
  ImageIcon,
  InformationCircleIcon,
  SparklesIcon,
  TextFontIcon,
  Tick02Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useId, useState } from "react";
import {
  HUB_UPLOADERS,
  type HubUploader,
  type ModelSearchFilters,
  hasModelSearchFilters,
} from "../lib/model-search-filters";
import type { HubCapabilityFilter } from "../lib/model-capabilities";
import { CAPABILITY_FILTER_OPTIONS } from "../lib/view-models";

// The Hub rows' capability icons (catalog/shared.tsx), with Image gen on diffusion's and Video gen
// on the Video page's.
const CAPABILITY_ICONS: Record<HubCapabilityFilter, IconSvgElement> = {
  text: TextFontIcon,
  reasoning: BrainIcon,
  vision: ViewIcon,
  audio: HeadphonesIcon,
  embedding: SparklesIcon,
  "image-gen": ImageIcon,
  "video-gen": FlimSlateIcon,
};

const UPLOADER_LABELS: Record<HubUploader, string> = {
  unsloth: "Unsloth",
  organization: "Organizations",
  personal: "Non-organization",
};

// The composer menu's label and separator.
const SECTION_LABEL =
  "flex w-full items-center justify-between px-3 pt-1.5 pb-1 text-ui-12 text-muted-foreground";
const SEPARATOR = "mx-1 my-1 h-px bg-border/50";

/** A composer-menu row that ticks: a check on the right when on. */
function MenuTick({
  icon,
  label,
  checked,
  locked,
  onToggle,
}: {
  icon?: IconSvgElement;
  label: string;
  checked: boolean;
  /** The last tick of a group, which stays. */
  locked: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-disabled={locked || undefined}
      onClick={() => {
        if (!locked) onToggle();
      }}
      className={cn(
        "hub-menu-row relative flex w-full cursor-pointer select-none items-center gap-3 py-2 pr-9 pl-3 text-left text-sm aria-disabled:cursor-default",
      )}
    >
      {icon && (
        <HugeiconsIcon
          icon={icon}
          strokeWidth={2}
          className="size-4 shrink-0"
        />
      )}
      <span className="truncate">{label}</span>
      {checked && (
        <HugeiconsIcon
          icon={Tick02Icon}
          strokeWidth={2}
          className="pointer-events-none absolute right-3 size-4"
        />
      )}
    </button>
  );
}

/** An i that shows its note while hovered, not on focus, so opening the popover doesn't pop it. */
function HoverInfo({ label, children }: { label: string; children: string }) {
  const [open, setOpen] = useState(false);
  return (
    <Tooltip open={open}>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={`About ${label.toLowerCase()}`}
          onPointerEnter={() => setOpen(true)}
          onPointerLeave={() => setOpen(false)}
          className="inline-flex cursor-help items-center transition-colors hover:text-foreground"
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
        {children}
      </TooltipContent>
    </Tooltip>
  );
}

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
        className="hub-menu-instant hub-composer-menu menu-soft-surface gap-0 w-[calc(344px*var(--ui-space-scale,1))] max-w-[calc(100vw-1rem)] ring-0"
      >
        <fieldset>
          <legend className={SECTION_LABEL}>
            Capabilities
            {capabilities.length > 0 && (
              <button
                type="button"
                onClick={() => onCapabilitiesChange([])}
                className="cursor-pointer transition-colors hover:text-foreground"
              >
                Select all
              </button>
            )}
          </legend>
          <div className="grid grid-cols-2 gap-x-1">
            {CAPABILITY_FILTER_OPTIONS.map((option) => {
              const ticked = (key: HubCapabilityFilter) =>
                capabilities.length === 0 || capabilities.includes(key);
              const checked = ticked(option.value);
              // Kept in the list's order, so it doesn't depend on click order.
              const next = CAPABILITY_FILTER_OPTIONS.map((o) => o.value).filter(
                (key) => (key === option.value ? !checked : ticked(key)),
              );
              return (
                <MenuTick
                  key={option.value}
                  icon={CAPABILITY_ICONS[option.value]}
                  label={option.label}
                  checked={checked}
                  // The last tick stays: none would list nothing.
                  locked={next.length === 0}
                  onToggle={() =>
                    onCapabilitiesChange(
                      next.length === CAPABILITY_FILTER_OPTIONS.length
                        ? []
                        : next,
                    )
                  }
                />
              );
            })}
          </div>
        </fieldset>
        {showModelFilters && (
          <>
            <div aria-hidden={true} className={SEPARATOR} />
            <form
              id={formId}
              className="flex flex-col gap-2"
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
                <fieldset key={min}>
                  <legend className={SECTION_LABEL}>
                    <span className="flex items-center gap-1">
                      {label}
                      <HoverInfo label={label}>{about}</HoverInfo>
                    </span>
                    {unit}
                  </legend>
                  <div className="grid grid-cols-2 gap-1.5 px-3">
                    {([min, max] as const).map((key, index) => (
                      <Input
                        key={key}
                        id={`${id}-${key}`}
                        aria-label={`${index === 0 ? "Minimum" : "Maximum"} ${label.toLowerCase()}`}
                        type="number"
                        min={0}
                        step={scale === 1 ? 1 : "any"}
                        placeholder={index === 0 ? "Min" : "Max"}
                        className="h-8 pl-3 text-sm md:text-sm"
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
                    ))}
                  </div>
                </fieldset>
              ))}
              {invalid && (
                <p role="alert" className="px-3 text-ui-12 text-destructive">
                  Minimum must not exceed maximum.
                </p>
              )}
            </form>
            <div aria-hidden={true} className={cn(SEPARATOR, "mt-2")} />
            <fieldset>
              <legend className={SECTION_LABEL}>Uploader</legend>
              {HUB_UPLOADERS.map((uploader) => {
                const checked = uploaders.includes(uploader);
                const next = HUB_UPLOADERS.filter((key) =>
                  key === uploader ? !checked : uploaders.includes(key),
                );
                return (
                  <MenuTick
                    key={uploader}
                    label={UPLOADER_LABELS[uploader]}
                    checked={checked}
                    locked={next.length === 0}
                    onToggle={() => onUploadersChange(next)}
                  />
                );
              })}
            </fieldset>
            <div aria-hidden={true} className={SEPARATOR} />
            <div className="flex justify-end gap-1.5 px-3 pt-1">
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
