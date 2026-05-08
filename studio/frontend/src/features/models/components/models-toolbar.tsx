// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  Search01Icon,
  SortByDown01Icon,
  SortByUp01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { HfSortDirection, HfSortKey } from "@/hooks";
import type {
  CapabilityFilter,
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
} from "../types";
import {
  CAPABILITY_FILTER_OPTIONS,
  FORMAT_FILTER_OPTIONS,
  RESOURCE_TYPE_OPTIONS,
} from "../lib/view-models";

const SORT_OPTIONS: ReadonlyArray<{
  value: HfSortKey;
  label: string;
}> = [
  { value: "trendingScore", label: "Best match" },
  { value: "likes", label: "Most likes" },
  { value: "downloads", label: "Most downloads" },
  { value: "lastModified", label: "Recently updated" },
];

export function ModelsToolbar({
  tab,
  onTabChange,
  query,
  onQueryChange,
  isLoading,
  sortBy,
  direction,
  onSortChange,
  onDirectionToggle,
  resourceType,
  onResourceTypeChange,
  formatFilter,
  onFormatFilterChange,
  capabilityFilter,
  onCapabilityFilterChange,
  discoverCount,
  downloadedCount,
}: {
  tab: ModelsTab;
  onTabChange: (tab: ModelsTab) => void;
  query: string;
  onQueryChange: (value: string) => void;
  isLoading: boolean;
  sortBy: HfSortKey;
  direction: HfSortDirection;
  onSortChange: (value: HfSortKey) => void;
  onDirectionToggle: () => void;
  resourceType: ResourceTypeFilter;
  onResourceTypeChange: (value: ResourceTypeFilter) => void;
  formatFilter: ModelFormatFilter;
  onFormatFilterChange: (value: ModelFormatFilter) => void;
  capabilityFilter: CapabilityFilter;
  onCapabilityFilterChange: (value: CapabilityFilter) => void;
  discoverCount: number;
  downloadedCount: number;
}) {
  const isDataset = resourceType === "datasets";
  const surface =
    "border border-border/70 bg-background/70 dark:bg-white/[0.035]";
  const triggerBase = cn(
    "h-9 rounded-[10px] text-[12.5px] transition-colors",
    surface,
    "hover:bg-background dark:hover:bg-white/[0.06]",
  );
  return (
    <div className="flex flex-col gap-2.5 lg:flex-row lg:items-center">
      <div
        className={cn(
          "relative inline-flex h-9 w-[240px] shrink-0 items-center rounded-[11px] p-0.5",
          "border border-border/70 bg-foreground/[0.04] dark:bg-white/[0.03]",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute left-0.5 top-0.5 bottom-0.5 w-[calc(50%-2px)] rounded-[9px] transition-transform duration-200 ease-out",
            "bg-background border border-border/60 shadow-[0_1px_2px_rgba(0,0,0,0.06)]",
            "dark:bg-white/[0.08] dark:border-white/[0.07] dark:shadow-[0_1px_3px_rgba(0,0,0,0.35)]",
            tab === "downloaded" ? "translate-x-full" : "translate-x-0",
          )}
        />
        <button
          type="button"
          onClick={() => onTabChange("discover")}
          className={cn(
            "relative z-10 inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-[9px] px-3 text-[12.5px] font-medium transition-colors",
            tab === "discover"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          Discover
          <span className="text-[11px] tabular-nums opacity-70">
            {discoverCount}
          </span>
        </button>
        <button
          type="button"
          onClick={() => onTabChange("downloaded")}
          className={cn(
            "relative z-10 inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-[9px] px-3 text-[12.5px] font-medium transition-colors",
            tab === "downloaded"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          On Device
          <span className="text-[11px] tabular-nums opacity-70">
            {downloadedCount}
          </span>
        </button>
      </div>

      <div className="relative flex-1">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={1.8}
          className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder={
            tab === "discover"
              ? "Search Hugging Face — model, family, or publisher"
              : "Search cached, local, or LM Studio models"
          }
          className={cn(
            "h-9 rounded-[10px] pl-9 pr-10 text-[13px] placeholder:text-muted-foreground/70",
            surface,
            "focus-visible:bg-background dark:focus-visible:bg-white/[0.06]",
          )}
        />
        {tab === "discover" && isLoading && (
          <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Select
          value={resourceType}
          onValueChange={(value) =>
            onResourceTypeChange(value as ResourceTypeFilter)
          }
        >
          <SelectTrigger className={cn(triggerBase, "min-w-[112px]")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0">
            {RESOURCE_TYPE_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {!isDataset && (
          <Select
            value={formatFilter}
            onValueChange={(value) =>
              onFormatFilterChange(value as ModelFormatFilter)
            }
          >
            <SelectTrigger className={cn(triggerBase, "min-w-[124px]")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="menu-soft-surface ring-0">
              {FORMAT_FILTER_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {tab === "discover" && !isDataset && (
          <Select
            value={capabilityFilter}
            onValueChange={(value) =>
              onCapabilityFilterChange(value as CapabilityFilter)
            }
          >
            <SelectTrigger className={cn(triggerBase, "min-w-[136px]")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="menu-soft-surface ring-0">
              {CAPABILITY_FILTER_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {tab === "discover" && (
          <Select
            value={sortBy}
            onValueChange={(value) => onSortChange(value as HfSortKey)}
          >
            <SelectTrigger className={cn(triggerBase, "min-w-[140px]")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="menu-soft-surface ring-0">
              {SORT_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {tab === "discover" && sortBy !== "trendingScore" && (
          <button
            type="button"
            onClick={onDirectionToggle}
            className={cn(
              "inline-flex size-9 items-center justify-center rounded-[10px] text-muted-foreground transition-colors",
              surface,
              "hover:bg-background hover:text-foreground dark:hover:bg-white/[0.07]",
            )}
            aria-label={
              direction === "desc"
                ? "Switch to ascending order"
                : "Switch to descending order"
            }
          >
            <HugeiconsIcon
              icon={direction === "desc" ? SortByDown01Icon : SortByUp01Icon}
              strokeWidth={1.8}
              className="size-4"
            />
          </button>
        )}
      </div>
    </div>
  );
}
