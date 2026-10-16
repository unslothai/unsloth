// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowLeftRightIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { ChevronDown } from "lucide-react";
import { type ReactNode, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/** Output size controls shared by the Images and Video Create panels. */

// Landscape ratios; Flip mirrors them to portrait.
export const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};
export const ASPECT_OPTIONS = ["custom", ...Object.keys(ASPECT_RATIOS)];
export const ASPECT_LABELS: Record<string, string> = {
  "1:1": "Square",
  "3:2": "Photo",
  "4:3": "Landscape",
  "16:9": "Widescreen",
  "21:9": "Ultrawide",
};

// The ratio key is compared long:short, so it survives orientation.
export function matchAspect(
  width: number,
  height: number,
): { key: string; portrait: boolean } {
  const target = Math.max(width, height) / Math.min(width, height);
  const found = Object.entries(ASPECT_RATIOS).find(
    ([, [a, b]]) => Math.abs(target - a / b) < 0.01,
  );
  return { key: found ? found[0] : "custom", portrait: height > width };
}

/** Aspect ratio select with the flip-orientation button beside it. */
export function AspectRatioSelect({
  value,
  onValueChange,
  portrait,
  onFlip,
  open,
  onOpenChange,
  extraItems,
}: {
  value: string;
  onValueChange: (key: string) => void;
  portrait: boolean;
  onFlip: () => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  // Items listed ahead of the ratios (Video's "Match source").
  extraItems?: ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      <Select
        value={value}
        onValueChange={onValueChange}
        open={open}
        onOpenChange={onOpenChange}
      >
        <SelectTrigger className="flex-1" aria-label="Aspect ratio">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {extraItems}
          {ASPECT_OPTIONS.map((key) => (
            <SelectItem key={key} value={key}>
              {key === "custom" ? "Custom" : `${ASPECT_LABELS[key]} (${key})`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <Button
            type="button"
            variant="secondary"
            size="icon"
            aria-label="Flip width and height"
            onClick={onFlip}
          >
            {/* Arrows turn with the orientation, showing which way it flips. */}
            <HugeiconsIcon
              icon={ArrowLeftRightIcon}
              className={cn(
                "size-4 transition-transform duration-200",
                portrait && "rotate-90",
              )}
            />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {portrait ? "Switch to landscape" : "Switch to portrait"}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

/** A width or height box: type a value, or pick one of `options` from the chevron menu. */
export function DimensionSelect({
  icon,
  label,
  value,
  open,
  onOpenChange,
  onChange,
  options,
  snap,
}: {
  icon: IconSvgElement;
  label: string;
  value: number;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onChange: (value: number) => void;
  options: number[];
  // Brings a typed value onto the model's grid and range.
  snap: (value: number) => number;
}) {
  // Typing is held in a draft so a half-entered number is not snapped mid-keystroke.
  const [draft, setDraft] = useState(String(value));
  const [lastValue, setLastValue] = useState(value);
  if (value !== lastValue) {
    setLastValue(value);
    setDraft(String(value));
  }
  const commit = () => {
    const typed = Number(draft);
    const next = snap(Number.isFinite(typed) && typed > 0 ? typed : value);
    setDraft(String(next));
    setLastValue(next);
    if (next !== value) onChange(next);
  };
  const pick = (n: number) => {
    setDraft(String(n));
    setLastValue(n);
    onChange(n);
  };
  return (
    <div className="flex h-9 flex-1 items-center gap-2 rounded-full border border-border bg-background px-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:focus-within:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-4 shrink-0 text-muted-foreground"
      />
      <input
        aria-label={label}
        inputMode="numeric"
        value={draft}
        onChange={(e) => setDraft(e.target.value.replace(/[^0-9]/g, ""))}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          }
        }}
        className="w-full min-w-0 bg-transparent text-sm tabular-nums outline-none"
      />
      <DropdownMenu open={open} onOpenChange={onOpenChange}>
        <DropdownMenuTrigger
          aria-label={`${label} presets`}
          className="-mr-1 shrink-0 cursor-pointer rounded-full p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ChevronDown className="size-4" />
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="max-h-[min(--spacing(72),var(--radix-dropdown-menu-content-available-height))] overflow-y-auto"
        >
          {options.map((n) => (
            <DropdownMenuItem key={n} onSelect={() => pick(n)}>
              <span className="tabular-nums">{n}</span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
