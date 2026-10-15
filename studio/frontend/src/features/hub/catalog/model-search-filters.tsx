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
import { BulbIcon } from "@/lib/bulb-icon";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  FilterMailIcon,
  FlimSlateIcon,
  ImageIcon,
  InformationCircleIcon,
  LayoutTable02Icon,
  TextFontIcon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactNode, useId, useState } from "react";
import type { HubCapabilityFilter } from "../lib/model-capabilities";
import {
  HUB_UPLOADERS,
  type HubUploader,
  type ModelSearchFilters,
  hasModelSearchFilters,
} from "../lib/model-search-filters";
import { CAPABILITY_FILTER_OPTIONS } from "../lib/view-models";

// The Hub rows' capability icons (catalog/shared.tsx), with Image gen on diffusion's and Video gen
// on the Video page's.
const CAPABILITY_ICONS: Record<HubCapabilityFilter, IconSvgElement | null> = {
  text: TextFontIcon,
  // The composer's thinking bulb, which isn't a Hugeicons glyph.
  reasoning: null,
  vision: ViewIcon,
  audio: AiSpeechIcon,
  embedding: LayoutTable02Icon,
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
  "flex w-full items-center justify-between px-3 pt-2 pb-1.5 text-ui-12 text-muted-foreground";
// my-2 keeps text 18px from each line: rows bring 2px margin and 8px of hover pill, the labels pt-2.
const SEPARATOR = "mx-1 my-2 h-px bg-border/50";

const LIMIT_KEYS = [
  "minParams",
  "maxParams",
  "minContext",
  "maxContext",
] as const;

/** A composer-menu row that ticks: a check on the right when on, and on hover an "Only" that
 *  keeps just this row ticked. */
function MenuTick({
  icon,
  label,
  checked,
  locked,
  onToggle,
  onOnly,
}: {
  icon?: ReactNode;
  label: string;
  checked: boolean;
  /** The last tick of a group, which stays. */
  locked: boolean;
  onToggle: () => void;
  /** Absent while this row is already the only one ticked. */
  onOnly?: () => void;
}) {
  return (
    <div className="hub-menu-row group/tick relative">
      <button
        type="button"
        role="checkbox"
        aria-checked={checked}
        aria-disabled={locked || undefined}
        onClick={() => {
          if (!locked) onToggle();
        }}
        className="flex w-full cursor-pointer select-none items-center gap-3 rounded-[inherit] py-2 pr-9 pl-3 text-left text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40 aria-disabled:cursor-default"
      >
        {icon}
        <span className="truncate">{label}</span>
        {checked && (
          <HugeiconsIcon
            icon={Tick02Icon}
            strokeWidth={2}
            className={cn(
              "pointer-events-none absolute right-3 size-4",
              onOnly &&
                "group-hover/tick:opacity-0 group-has-[[data-only]:focus-visible]/tick:opacity-0",
            )}
          />
        )}
      </button>
      {onOnly && (
        <button
          type="button"
          data-only=""
          aria-label={`Show only ${label}`}
          onClick={onOnly}
          className="absolute top-1/2 right-1.5 -translate-y-1/2 cursor-pointer rounded-full px-1.5 py-0.5 text-ui-12 text-muted-foreground opacity-0 outline-none transition-colors hover:text-foreground focus-visible:opacity-100 group-hover/tick:opacity-100"
        >
          Only
        </button>
      )}
    </div>
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
  onReset,
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
  /** Back to no filters: every capability, the default uploaders, no limits. */
  onReset: () => void;
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
  const dirty = LIMIT_KEYS.some((key) => draft[key] !== value[key]);
  const filtered =
    capabilities.length > 0 ||
    (showModelFilters && (hasModelSearchFilters(value) || uploadersChanged));
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
              {filtered && (
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
              const glyph = CAPABILITY_ICONS[option.value];
              // Kept in the list's order, so it doesn't depend on click order.
              const next = CAPABILITY_FILTER_OPTIONS.map((o) => o.value).filter(
                (key) => (key === option.value ? !checked : ticked(key)),
              );
              return (
                <MenuTick
                  key={option.value}
                  icon={
                    glyph ? (
                      <HugeiconsIcon
                        icon={glyph}
                        strokeWidth={2}
                        className="size-4 shrink-0"
                      />
                    ) : (
                      <BulbIcon className="size-4 shrink-0" />
                    )
                  }
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
                  onOnly={
                    capabilities.length === 1 && checked
                      ? undefined
                      : () => onCapabilitiesChange([option.value])
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
              className="flex flex-col gap-2.5 pb-0.5"
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
                    "minParams",
                    "maxParams",
                    1e9,
                    "Both limits are included. Models that don't publish a parameter count are left out.",
                  ],
                  [
                    "Context length",
                    "minContext",
                    "maxContext",
                    1,
                    "In tokens. The model's published limit, not what your device can fit. Both limits are included; models that don't publish one are left out.",
                  ],
                ] as const
              ).map(([label, min, max, scale, about]) => (
                <fieldset key={min}>
                  <legend className={SECTION_LABEL}>
                    <span className="flex items-center gap-1">
                      {label}
                      <HoverInfo label={label}>{about}</HoverInfo>
                    </span>
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
            <div aria-hidden={true} className={SEPARATOR} />
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
                    onOnly={
                      uploaders.length === 1 && checked
                        ? undefined
                        : () => onUploadersChange([uploader])
                    }
                  />
                );
              })}
            </fieldset>
          </>
        )}
        <div aria-hidden={true} className={SEPARATOR} />
        <div className="flex items-center justify-between gap-1.5 px-3 pt-0.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={!(filtered || dirty)}
            onClick={() => {
              setDraft({});
              onReset();
            }}
          >
            Reset all
          </Button>
          {showModelFilters && (
            <Button
              type="submit"
              form={formId}
              size="sm"
              disabled={invalid || !dirty}
            >
              Apply limits
            </Button>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
