// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
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
import { cn } from "@/lib/utils";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { Tick02Icon } from "@/lib/tick-icon";
import { Slider as SliderPrimitive } from "radix-ui";
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
import { type ReactNode, useEffect, useState } from "react";
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

// The composer menu's label and separator.
const SECTION_LABEL =
  "flex w-full items-center justify-between px-3 pt-2 pb-1.5 text-ui-12 text-muted-foreground";
// my-2 keeps text 18px from each line: rows bring 2px margin and 8px of hover pill, the labels pt-2.
const SEPARATOR = "mx-1 my-2 h-px bg-border/50";

type LimitKey = "minParams" | "maxParams" | "minContext" | "maxContext";

interface RangeSpec {
  label: string;
  about: string;
  min: LimitKey;
  max: LimitKey;
  /** The first stop as a minimum and the last as a maximum mean no limit. */
  stops: readonly number[];
  /** Index -> label; unlabelled stops show only a tick. */
  labels: readonly (string | null)[];
}

const B = 1e9;
const K = 1024;
const RANGES: readonly RangeSpec[] = [
  {
    label: "Parameters",
    about:
      "Both limits are included. Models that don't publish a parameter count are left out.",
    min: "minParams",
    max: "maxParams",
    stops: [1 * B, 3 * B, 6 * B, 12 * B, 32 * B, 70 * B, 128 * B, 500 * B],
    labels: ["<1B", null, "6B", "12B", "32B", null, "128B", ">500B"],
  },
  {
    label: "Context length",
    about:
      "In tokens. The model's published limit, not what your device can fit. Both limits are included; models that don't publish one are left out.",
    min: "minContext",
    max: "maxContext",
    stops: [
      4 * K,
      8 * K,
      16 * K,
      32 * K,
      64 * K,
      128 * K,
      256 * K,
      512 * K,
      K * K,
    ],
    labels: ["<4K", null, "16K", null, "64K", null, "256K", null, "1M+"],
  },
];

const THUMB = 14;

function stopIndex(
  stops: readonly number[],
  value: number | undefined,
  end: number,
) {
  if (value === undefined) {
    return end;
  }
  let best = 0;
  for (let i = 1; i < stops.length; i++) {
    if (Math.abs(stops[i] - value) < Math.abs(stops[best] - value)) {
      best = i;
    }
  }
  return best;
}

/** A two-thumb range over fixed stops. The value chips ride the thumbs and stand in for the
 *  tick labels under them; the limits apply on release. */
function RangeStops({
  spec,
  value,
  onChange,
}: {
  spec: RangeSpec;
  value: ModelSearchFilters;
  onChange: (value: ModelSearchFilters) => void;
}) {
  const last = spec.stops.length - 1;
  const committed: [number, number] = [
    stopIndex(spec.stops, value[spec.min], 0),
    stopIndex(spec.stops, value[spec.max], last),
  ];
  // Only while dragging. Radix can report a keyboard step after its commit, so a new committed
  // value (or a reset elsewhere) clears it too.
  const [dragging, setDragging] = useState<[number, number] | null>(null);
  const minValue = value[spec.min];
  const maxValue = value[spec.max];
  // biome-ignore lint/correctness/useExhaustiveDependencies: clears on a committed change
  useEffect(() => setDragging(null), [minValue, maxValue]);
  const [lo, hi] = dragging ?? committed;
  const set = lo !== 0 || hi !== last;
  // Thumbs stay inside the track, so stop i sits half a thumb in from each end.
  // A chip stands in for the label under it; one hugging an end also covers its neighbour.
  const covered = (i: number) =>
    i === lo ||
    i === hi ||
    (lo === 0 && i === 1) ||
    (hi === last && i === last - 1);
  // The end labels and chips hug the track's ends instead of hanging past them.
  const edge = (i: number) =>
    i === 0 ? { left: 0 } : i === last ? { right: 0 } : undefined;
  const at = (i: number) =>
    `calc(${THUMB / 2}px + (100% - ${THUMB}px) * ${i / last})`;
  const chip = (i: number) =>
    spec.labels[i] ??
    (spec.stops[i] >= B ? `${spec.stops[i] / B}B` : `${spec.stops[i] / K}K`);
  return (
    <fieldset>
      <legend className={SECTION_LABEL}>
        <span className="flex items-center gap-1">
          {spec.label}
          <HoverInfo label={spec.label}>{spec.about}</HoverInfo>
        </span>
        {set && (
          <button
            type="button"
            onClick={() =>
              onChange({
                ...value,
                [spec.min]: undefined,
                [spec.max]: undefined,
              })
            }
            className="cursor-pointer transition-colors hover:text-foreground"
          >
            Reset
          </button>
        )}
      </legend>
      <div className="relative mx-3 pt-7 pb-1">
        <div aria-hidden={true} className="absolute inset-x-0 top-0 h-5">
          {spec.labels.map((label, i) =>
            label === null || covered(i) ? null : (
              <span
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed stops
                key={i}
                style={edge(i) ?? { left: at(i) }}
                className={cn(
                  "absolute top-0.5 whitespace-nowrap text-ui-12 text-muted-foreground tabular-nums",
                  !edge(i) && "-translate-x-1/2",
                )}
              >
                {label}
              </span>
            ),
          )}
        </div>
        <SliderPrimitive.Root
          min={0}
          max={last}
          step={1}
          minStepsBetweenThumbs={1}
          value={[lo, hi]}
          onValueChange={(next) => setDragging([next[0], next[1]])}
          onValueCommit={(next) => {
            setDragging(null);
            onChange({
              ...value,
              [spec.min]: next[0] === 0 ? undefined : spec.stops[next[0]],
              [spec.max]: next[1] === last ? undefined : spec.stops[next[1]],
            });
          }}
          className="relative flex h-4 w-full touch-none select-none items-center"
        >
          <SliderPrimitive.Track className="relative h-1 w-full grow rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]">
            {spec.stops.map((_, i) =>
              i === 0 || i === last ? null : (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: fixed stops
                  key={i}
                  aria-hidden={true}
                  style={{ left: at(i) }}
                  className="absolute top-1/2 h-2 w-px -translate-x-1/2 -translate-y-1/2 bg-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-wash-gain,1)),transparent)]"
                />
              ),
            )}
            <SliderPrimitive.Range className="absolute h-full rounded-full bg-foreground" />
          </SliderPrimitive.Track>
          {([lo, hi] as const).map((index, thumb) => (
            <SliderPrimitive.Thumb
              key={thumb === 0 ? "min" : "max"}
              aria-label={`${thumb === 0 ? "Minimum" : "Maximum"} ${spec.label.toLowerCase()}`}
              aria-valuetext={chip(index)}
              style={{ width: THUMB, height: THUMB }}
              className="relative block cursor-grab rounded-full bg-foreground shadow-sm outline-none ring-background ring-2 focus-visible:ring-4 focus-visible:ring-ring/50 active:cursor-grabbing"
            >
              <span
                aria-hidden={true}
                className={cn(
                  "pointer-events-none absolute bottom-full mb-2 whitespace-nowrap rounded-md bg-foreground px-1.5 py-0.5 text-ui-12 font-medium text-background tabular-nums",
                  index === 0
                    ? "left-0"
                    : index === last
                      ? "right-0"
                      : "left-1/2 -translate-x-1/2",
                )}
              >
                {chip(index)}
              </span>
            </SliderPrimitive.Thumb>
          ))}
        </SliderPrimitive.Root>
      </div>
    </fieldset>
  );
}

/** A composer-menu row that ticks: a check on the right when on. */
function MenuTick({
  icon,
  label,
  checked,
  locked,
  onToggle,
}: {
  icon?: ReactNode;
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
      className="hub-menu-row relative flex w-full cursor-pointer select-none items-center gap-3 py-2 pr-9 pl-3 text-left text-sm aria-disabled:cursor-default"
    >
      {icon}
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
  isolateFirstCapability,
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
  /** True until the first change from the defaults: that capability pick keeps just that one. */
  isolateFirstCapability: boolean;
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
  const unslothOnly = uploaders.length === 1 && uploaders[0] === "unsloth";
  const filtered =
    capabilities.length > 0 ||
    (showModelFilters && (hasModelSearchFilters(value) || uploadersChanged));
  return (
    <Popover open={open} onOpenChange={setOpen}>
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
        // The sections scroll inside the padding, so a short window keeps the corners round and
        // Reset all in view; flex children would otherwise shrink the separators away.
        className="hub-menu-instant hub-composer-menu menu-soft-surface gap-0 overflow-hidden w-[calc(392px*var(--ui-space-scale,1))] max-w-[calc(100vw-1rem)] ring-0"
      >
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-width:thin]">
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
                const next = CAPABILITY_FILTER_OPTIONS.map(
                  (o) => o.value,
                ).filter((key) =>
                  key === option.value ? !checked : ticked(key),
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
                        capabilities.length === 0 && isolateFirstCapability
                          ? [option.value]
                          : next.length === CAPABILITY_FILTER_OPTIONS.length
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
              <div className="flex flex-col gap-2.5 pb-0.5">
                {RANGES.map((spec) => (
                  <RangeStops
                    key={spec.label}
                    spec={spec}
                    value={value}
                    onChange={onChange}
                  />
                ))}
              </div>
              <div aria-hidden={true} className={SEPARATOR} />
              <div className="flex items-center justify-between gap-3 px-3 py-0.5">
                <span className="text-ui-12 text-muted-foreground">
                  Uploader
                </span>
                <div
                  role="radiogroup"
                  aria-label="Uploader"
                  className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
                >
                  {(
                    [
                      ["Unsloth", true],
                      ["All uploaders", false],
                    ] as const
                  ).map(([label, unslothOption]) => {
                    const active = unslothOnly === unslothOption;
                    return (
                      <button
                        key={label}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        onClick={() => {
                          if (!active) {
                            onUploadersChange(
                              unslothOption ? ["unsloth"] : [...HUB_UPLOADERS],
                            );
                          }
                        }}
                        className={cn(
                          "relative flex h-8 cursor-pointer items-center rounded-full px-3 text-xs font-medium transition-colors",
                          active
                            ? "hub-tab-toggle-pill text-foreground"
                            : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        <span className="relative z-10">{label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
        <div aria-hidden={true} className={cn(SEPARATOR, "shrink-0")} />
        <div className="flex shrink-0 items-center justify-end px-3 pt-0.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={!filtered}
            onClick={onReset}
          >
            Reset all
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
