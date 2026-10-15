// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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
  ChartLineData01Icon,
  CubeIcon,
  File01Icon,
  FilterMailIcon,
  FlimSlateIcon,
  ImageIcon,
  InformationCircleIcon,
  LayoutTable02Icon,
  Location01Icon,
  Table01Icon,
  TextFontIcon,
  UndoIcon,
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
import {
  DATASET_MODALITIES,
  DATASET_ROW_STOPS,
  type DatasetModality,
} from "../lib/dataset-filters";
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
// Each section's content sits 17px from the lines on both sides; the instances trim or pad my-2 to it.
const SEPARATOR = "mx-1 my-2 h-px bg-border";
// A picked capability: a flat grey a few steps down in light, a lift toward white in dark.
const PICKED_FILL =
  "bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),var(--popover))] dark:bg-[color-mix(in_srgb,var(--accent),white_8%)]";
// An unpicked capability: a faint wash in light, so a picked one stands off it; field-soft's tones in dark.
const UNPICKED_FILL =
  "bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),var(--popover))] hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),var(--popover))] dark:bg-[color-mix(in_srgb,var(--accent)_60%,transparent)] dark:hover:bg-(--accent)";
// The sections' spacing without the line: only Reset all sits under one.
const SECTION_GAP = "mx-1 my-2 h-px";

type LimitKey =
  | "minParams"
  | "maxParams"
  | "minContext"
  | "maxContext"
  | "minRows"
  | "maxRows";

interface RangeSpec {
  label: string;
  about: string;
  min: LimitKey;
  max: LimitKey;
  /** The first stop as a minimum and the last as a maximum mean no limit. */
  stops: readonly number[];
  /** Index -> label; unlabelled stops show only a tick. */
  labels: readonly (string | null)[];
  /** An unlabelled stop's pill text. */
  format: (value: number) => string;
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
    // Labelled stops sit evenly with one step between each pair, as on the Hugging Face Hub.
    stops: [1, 3, 6, 9, 12, 24, 32, 64, 128, 256, 500].map((n) => n * B),
    labels: [
      "<1B",
      null,
      "6B",
      null,
      "12B",
      null,
      "32B",
      null,
      "128B",
      null,
      ">500B",
    ],
    format: (n) => `${n / B}B`,
  },
  {
    label: "Context length",
    about:
      "In tokens. The model's published limit, not what your device can fit. Both limits are included; models that don't publish one are left out.",
    min: "minContext",
    max: "maxContext",
    // Like Parameters, one step between each labelled pair.
    stops: [16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024].map((n) => n * K),
    labels: [
      "<16K",
      null,
      "32K",
      null,
      "64K",
      null,
      "128K",
      null,
      "256K",
      null,
      ">1M",
    ],
    format: (n) => `${n / K}K`,
  },
];

const ROW_UNITS: readonly [number, string][] = [
  [1e12, "T"],
  [1e9, "B"],
  [1e6, "M"],
  [1e3, "K"],
];
const DATASET_ROWS: RangeSpec = {
  label: "Size (rows)",
  about:
    "The Hub's size category for the dataset. Datasets that don't publish one are left out.",
  min: "minRows",
  max: "maxRows",
  stops: DATASET_ROW_STOPS,
  labels: ["<1K", null, null, "1M", null, null, "1B", null, null, ">1T"],
  format: (n) => {
    const [scale, unit] = ROW_UNITS.find(([size]) => n >= size) ?? [1, ""];
    return `${n / scale}${unit}`;
  },
};

const MODALITY_LABELS: Record<DatasetModality, string> = {
  "3d": "3D",
  audio: "Audio",
  document: "Document",
  geospatial: "Geospatial",
  image: "Image",
  tabular: "Tabular",
  text: "Text",
  timeseries: "Time-series",
  video: "Video",
};

const MODALITY_ICONS: Record<DatasetModality, IconSvgElement> = {
  "3d": CubeIcon,
  audio: AiSpeechIcon,
  document: File01Icon,
  geospatial: Location01Icon,
  image: ImageIcon,
  tabular: Table01Icon,
  text: TextFontIcon,
  timeseries: ChartLineData01Icon,
  video: FlimSlateIcon,
};

const THUMB = 12;
// The range, thumbs and value pills: a dark grey in light, a soft white in dark.
const SLIDER_FG =
  "[--hub-slider-fg:color-mix(in_oklab,var(--foreground)_68%,var(--background))] dark:[--hub-slider-fg:color-mix(in_oklab,var(--foreground)_86%,var(--background))]";
const RANGE_FILL = "bg-(--hub-slider-fg)";

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
  // A chip stands in for the label under it.
  const covered = (i: number) => i === lo || i === hi;
  // The end labels and chips hug the track's ends instead of hanging past them.
  const edge = (i: number) =>
    i === 0 ? { left: 0 } : i === last ? { right: 0 } : undefined;
  const at = (i: number) =>
    `calc(${THUMB / 2}px + (100% - ${THUMB}px) * ${i / last})`;
  const chip = (i: number) => spec.labels[i] ?? spec.format(spec.stops[i]);
  return (
    <fieldset>
      <legend className={SECTION_LABEL}>
        <HoverInfo label={spec.label}>{spec.about}</HoverInfo>
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
            className="flex cursor-pointer items-center gap-1 transition-colors hover:text-foreground"
          >
            <HugeiconsIcon
              icon={UndoIcon}
              strokeWidth={1.75}
              className="size-3"
            />
            Reset
          </button>
        )}
      </legend>
      {/* Inset by an end pill's overhang, so the pills line up with the labels above. */}
      <div className={cn("relative mx-4.5 pt-7 pb-1", SLIDER_FG)}>
        <div aria-hidden={true} className="absolute inset-x-0 top-0 h-5">
          {spec.labels.map((label, i) =>
            label === null && !covered(i) ? null : (
              <span
                // biome-ignore lint/suspicious/noArrayIndexKey: fixed stops
                key={i}
                style={edge(i) ?? { left: at(i) }}
                className={cn(
                  // A thumb's stop wears a pill; the padding sits outside the text so it stays put.
                  "absolute top-0.5 whitespace-nowrap text-ui-11 tabular-nums",
                  !edge(i) && "-translate-x-1/2",
                  covered(i)
                    ? cn(
                        "-mx-1 -my-0.5 rounded-md px-1 py-0.5 font-semibold text-background",
                        RANGE_FILL,
                      )
                    : "text-muted-foreground",
                )}
              >
                {chip(i)}
              </span>
            ),
          )}
        </div>
        <SliderPrimitive.Root
          min={0}
          max={last}
          step={1}
          minStepsBetweenThumbs={0}
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
          className="group/range relative flex h-4 w-full touch-none select-none items-center"
        >
          {/* The run settings track: a faint wash rather than a bar. */}
          <SliderPrimitive.Track className="relative h-[3px] w-full grow rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]">
            {spec.stops.map((_, i) =>
              // Under the range, which covers them where it runs, so its fill stays unbroken.
              i === 0 || i === last ? null : (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: fixed stops
                  key={i}
                  aria-hidden={true}
                  style={{ left: at(i) }}
                  className="absolute top-1/2 h-3 w-px -translate-x-1/2 -translate-y-1/2 bg-[color-mix(in_oklab,var(--foreground)_calc(28%*var(--contrast-wash-gain,1)),transparent)]"
                />
              ),
            )}
            <SliderPrimitive.Range
              className={cn("absolute h-full rounded-full", RANGE_FILL)}
            />
          </SliderPrimitive.Track>
          {([lo, hi] as const).map((index, thumb) => (
            <SliderPrimitive.Thumb
              key={thumb === 0 ? "min" : "max"}
              aria-label={`${thumb === 0 ? "Minimum" : "Maximum"} ${spec.label.toLowerCase()}`}
              aria-valuetext={chip(index)}
              style={{ width: THUMB, height: THUMB }}
              // The run settings thumb's halo on hover, keyboard focus and press.
              className={cn(
                "relative block cursor-grab rounded-full outline-none transition-shadow duration-150 hover:shadow-[0_0_0_8px_color-mix(in_srgb,var(--hub-slider-fg)_18%,transparent)] focus-visible:shadow-[0_0_0_8px_color-mix(in_srgb,var(--hub-slider-fg)_18%,transparent)] active:cursor-grabbing active:shadow-[0_0_0_10px_color-mix(in_srgb,var(--hub-slider-fg)_22%,transparent)]",
                RANGE_FILL,
              )}
            />
          ))}
        </SliderPrimitive.Root>
      </div>
    </fieldset>
  );
}

/** A tick option as a pill in the Hub's control look: the picked ones raised like a selected
 *  segment, with a check on the right. */
function OptionTile({
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
        if (!locked) {
          onToggle();
        }
      }}
      className={cn(
        "relative flex w-full cursor-pointer select-none items-center gap-2.5 rounded-full py-2 pr-8 pl-3.5 text-left text-sm transition-colors aria-disabled:cursor-default",
        checked
          ? cn(PICKED_FILL, "text-foreground")
          : cn(UNPICKED_FILL, "text-muted-foreground hover:text-foreground"),
      )}
    >
      {icon}
      <span className="truncate">{label}</span>
      {checked && (
        <HugeiconsIcon
          icon={Tick02Icon}
          strokeWidth={2}
          className="pointer-events-none absolute right-2.5 size-4"
        />
      )}
    </button>
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

/** A labelled switch; its note shows on hovering the label. */
function SwitchField({
  label,
  about,
  checked,
  onCheckedChange,
}: {
  label: string;
  about: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3 px-3">
      <span className="pt-0.5 text-ui-12 text-muted-foreground">
        <HoverInfo label={label}>{about}</HoverInfo>
      </span>
      <Switch
        aria-label={label}
        checked={checked}
        onCheckedChange={onCheckedChange}
      />
    </div>
  );
}

/** A labelled pill toggle, the Hugging Face / ModelScope switch's look. */
function SegmentedRow<T extends string>({
  label,
  about,
  value,
  options,
  onChange,
}: {
  label: string;
  about: string;
  value: T;
  options: readonly (readonly [T, string])[];
  onChange: (value: T) => void;
}) {
  return (
    // One row, the title on the control's top edge rather than its middle.
    <div className="flex items-start justify-between gap-3 px-3">
      <span className="pt-0.5 text-ui-12 text-muted-foreground">
        <HoverInfo label={label}>{about}</HoverInfo>
      </span>
      <div
        role="radiogroup"
        aria-label={label}
        // Light keeps the segmented toggle's raised white pick; dark takes the pills' fills, the track
        // at field-soft's rest.
        className="hub-tab-toggle inline-flex h-8 items-center rounded-full dark:[--hub-tab-track:color-mix(in_srgb,var(--accent)_60%,transparent)]"
      >
        {options.map(([key, text]) => {
          const active = value === key;
          return (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => {
                if (!active) {
                  onChange(key);
                }
              }}
              className={cn(
                "relative flex h-8 cursor-pointer items-center rounded-full px-3 text-xs font-medium transition-colors",
                active
                  ? "hub-tab-toggle-pill text-foreground dark:bg-[color-mix(in_srgb,var(--accent),white_8%)]!"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="relative z-10">{text}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** A section label with an i. Its note shows while the label is hovered, not on focus (opening the
 *  popover doesn't pop it), and to the left of the popover so it never covers the other section. */
function HoverInfo({ label, children }: { label: string; children: string }) {
  const [open, setOpen] = useState(false);
  return (
    <Tooltip open={open}>
      <TooltipTrigger asChild={true}>
        <span
          onPointerEnter={() => setOpen(true)}
          onPointerLeave={() => setOpen(false)}
          className="group/title flex cursor-help items-center gap-1 transition-colors hover:text-foreground"
        >
          {label}
          <button
            type="button"
            aria-label={`About ${label.toLowerCase()}`}
            // Shown only while the title is hovered.
            className="inline-flex cursor-help items-center opacity-0 transition-opacity focus-visible:opacity-100 group-hover/title:opacity-100"
          >
            <HugeiconsIcon
              icon={InformationCircleIcon}
              strokeWidth={1.75}
              className="size-3"
            />
          </button>
        </span>
      </TooltipTrigger>
      <TooltipContent
        side="left"
        sideOffset={24}
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
  datasets,
  onDatasetsChange,
  modelSections,
  datasetSections,
  modalities,
  onModalitiesChange,
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
  /** The list shows datasets. */
  datasets: boolean;
  /** Capabilities, ranges and uploader apply (Discover models). */
  modelSections: boolean;
  /** Modalities and size apply (Discover datasets). */
  datasetSections: boolean;
  /** Empty means every modality ticked (no filter). */
  modalities: DatasetModality[];
  onModalitiesChange: (value: DatasetModality[]) => void;
  onDatasetsChange: (value: boolean) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const unslothOnly = uploaders.length === 1 && uploaders[0] === "unsloth";
  const filtered =
    modalities.length > 0 ||
    capabilities.length > 0 ||
    uploadersChanged ||
    ((showModelFilters || datasetSections) && hasModelSearchFilters(value));
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
                // Its lines narrow downward, so the box's centre reads high: a pixel down evens it.
                className="size-5 translate-y-px"
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
          {modelSections && (
            <>
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
                <div className="grid grid-cols-2 gap-1.5 px-3 pb-0.5">
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
                      <OptionTile
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
                  <div
                    aria-hidden={true}
                    className={cn(
                      SECTION_GAP,
                      "mt-[calc(5px*var(--ui-space-scale,1))]",
                    )}
                  />
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
                </>
              )}
              <div
                aria-hidden={true}
                className={cn(
                  SECTION_GAP,
                  showModelFilters
                    ? "mt-0"
                    : "mt-[calc(5px*var(--ui-space-scale,1))]",
                )}
              />
            </>
          )}
          {datasetSections && (
            <>
              <fieldset>
                <legend className={SECTION_LABEL}>
                  Modalities
                  {modalities.length > 0 && (
                    <button
                      type="button"
                      onClick={() => onModalitiesChange([])}
                      className="cursor-pointer transition-colors hover:text-foreground"
                    >
                      Select all
                    </button>
                  )}
                </legend>
                <div className="grid grid-cols-2 gap-x-1">
                  {DATASET_MODALITIES.map((modality) => {
                    const ticked = (key: DatasetModality) =>
                      modalities.length === 0 || modalities.includes(key);
                    const checked = ticked(modality);
                    const next = DATASET_MODALITIES.filter((key) =>
                      key === modality ? !checked : ticked(key),
                    );
                    return (
                      <MenuTick
                        key={modality}
                        icon={
                          <HugeiconsIcon
                            icon={MODALITY_ICONS[modality]}
                            strokeWidth={2}
                            className="size-4 shrink-0"
                          />
                        }
                        label={MODALITY_LABELS[modality]}
                        checked={checked}
                        locked={next.length === 0}
                        onToggle={() =>
                          onModalitiesChange(
                            // Like Capabilities: the first pick from the defaults keeps just that one.
                            modalities.length === 0 && isolateFirstCapability
                              ? [modality]
                              : next.length === DATASET_MODALITIES.length
                                ? []
                                : next,
                          )
                        }
                      />
                    );
                  })}
                </div>
              </fieldset>
              <div
                aria-hidden={true}
                className={cn(
                  SECTION_GAP,
                  "mt-[calc(5px*var(--ui-space-scale,1))]",
                )}
              />
              <div className="pb-0.5">
                <RangeStops
                  spec={DATASET_ROWS}
                  value={value}
                  onChange={onChange}
                />
              </div>
              <div aria-hidden={true} className={cn(SECTION_GAP, "mt-0")} />
            </>
          )}
          <div className="flex flex-col gap-3 pt-2 pb-[calc(7px*var(--ui-space-scale,1))]">
            {/* The Hub search's own scope, so it stays on ModelScope too; datasets take none. */}
            {modelSections && (
              <SegmentedRow
                label="Uploader"
                about="Unsloth lists Unsloth's own uploads. All uploaders searches every account on the Hub; a search starts there."
                value={unslothOnly ? "unsloth" : "all"}
                options={[
                  ["unsloth", "Unsloth"],
                  ["all", "All uploaders"],
                ]}
                onChange={(next) =>
                  onUploadersChange(
                    next === "unsloth" ? ["unsloth"] : [...HUB_UPLOADERS],
                  )
                }
              />
            )}
            <div>
              <SwitchField
                label="Datasets only"
                about="Lists datasets instead of models."
                checked={datasets}
                onCheckedChange={onDatasetsChange}
              />
            </div>
          </div>
        </div>
        <div aria-hidden={true} className={cn(SEPARATOR, "mb-1.5 shrink-0")} />
        <div className="flex shrink-0 items-center justify-end px-3 pb-0.5">
          <Button
            type="button"
            variant="ghost"
            size="xs"
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
