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
  ArrowDown01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { HfSortKey } from "@/hooks";
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
  { value: "trendingScore", label: "Trending" },
  { value: "likes", label: "Most likes" },
  { value: "downloads", label: "Most downloads" },
  { value: "lastModified", label: "Recently updated" },
  { value: "createdAt", label: "Newest" },
];

export function ModelsToolbar({
  tab,
  onTabChange,
  query,
  onQueryChange,
  isLoading,
  sortBy,
  onSortChange,
  resourceType,
  onResourceTypeChange,
  formatFilter,
  onFormatFilterChange,
  capabilityFilter,
  onCapabilityFilterChange,
}: {
  tab: ModelsTab;
  onTabChange: (tab: ModelsTab) => void;
  query: string;
  onQueryChange: (value: string) => void;
  isLoading: boolean;
  sortBy: HfSortKey;
  onSortChange: (value: HfSortKey) => void;
  resourceType: ResourceTypeFilter;
  onResourceTypeChange: (value: ResourceTypeFilter) => void;
  formatFilter: ModelFormatFilter;
  onFormatFilterChange: (value: ModelFormatFilter) => void;
  capabilityFilter: CapabilityFilter;
  onCapabilityFilterChange: (value: CapabilityFilter) => void;
}) {
  const isDataset = resourceType === "datasets";
  const triggerBase = cn(
    "menu-trigger field-soft h-9 rounded-[12px] text-[12.5px] text-muted-foreground transition-colors",
    "focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-border",
  );
  return (
    <div className="flex flex-col gap-2 lg:flex-row lg:items-center">
      <div
        className={cn(
          "menu-trigger tab-toggle relative inline-flex h-9 w-[240px] shrink-0 items-center rounded-full p-0.5",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "tab-toggle-pill",
            "pointer-events-none absolute left-0.5 top-0.5 bottom-0.5 w-[calc(50%-2px)] rounded-full transition-transform duration-200 ease-out",
            tab === "downloaded" ? "translate-x-full" : "translate-x-0",
          )}
        />
        <button
          type="button"
          onClick={() => onTabChange("discover")}
          className={cn(
            "relative z-10 inline-flex h-8 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
            tab === "discover"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          Discover
        </button>
        <button
          type="button"
          onClick={() => onTabChange("downloaded")}
          className={cn(
            "relative z-10 inline-flex h-8 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
            tab === "downloaded"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          On Device
        </button>
      </div>

      <Select
        value={resourceType}
        onValueChange={(value) =>
          onResourceTypeChange(value as ResourceTypeFilter)
        }
      >
        <SelectTrigger
          animateRadius={false}
          icon={ArrowDown01Icon}
          iconStrokeWidth={1.25}
          iconClassName="size-3.5"
          className={cn(
            triggerBase,
            "w-[112px] lg:w-[88px] xl:w-[128px] 2xl:w-[168px]",
          )}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent
          position="popper"
          side="bottom"
          align="start"
          sideOffset={8}
          avoidCollisions={false}
          className="menu-instant menu-soft-surface rounded-[14px] ring-0"
        >
          {RESOURCE_TYPE_OPTIONS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="relative flex-1">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={1.8}
          className="pointer-events-none absolute left-3 top-1/2 size-[18px] -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder={
            tab === "discover"
              ? "Search Hugging Face: model, family, or publisher"
              : "Search cached or local models"
          }
          className={cn(
            "field-soft h-9 rounded-full pl-10 pr-10 text-[13.5px] placeholder:text-muted-foreground/80",
            "focus-visible:bg-background dark:focus-visible:bg-white/[0.06]",
          )}
        />
        {tab === "discover" && isLoading && (
          <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {!isDataset && (
          <Select
            value={formatFilter}
            onValueChange={(value) =>
              onFormatFilterChange(value as ModelFormatFilter)
            }
          >
            <SelectTrigger
              animateRadius={false}
              icon={ArrowDown01Icon}
              iconStrokeWidth={1.25}
              iconClassName="size-3.5"
              className={cn(triggerBase, "min-w-[124px]")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent
            position="popper"
            side="bottom"
            align="start"
            sideOffset={8}
            avoidCollisions={false}
            className="menu-instant menu-soft-surface rounded-[14px] ring-0"
          >
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
            <SelectTrigger
              animateRadius={false}
              icon={ArrowDown01Icon}
              iconStrokeWidth={1.25}
              iconClassName="size-3.5"
              className={cn(triggerBase, "min-w-[136px]")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent
            position="popper"
            side="bottom"
            align="start"
            sideOffset={8}
            avoidCollisions={false}
            className="menu-instant menu-soft-surface rounded-[14px] ring-0"
          >
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
            <SelectTrigger
              animateRadius={false}
              icon={ArrowDown01Icon}
              iconStrokeWidth={1.25}
              iconClassName="size-3.5"
              className={cn(triggerBase, "min-w-[140px]")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent
            position="popper"
            side="bottom"
            align="start"
            sideOffset={8}
            avoidCollisions={false}
            className="menu-instant menu-soft-surface rounded-[14px] ring-0"
          >
              {SORT_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

      </div>
    </div>
  );
}
