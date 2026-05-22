// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  AiChipIcon,
  Database02Icon,
  RefreshIcon,
  Search01Icon,
  Tick02Icon,
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
  CHANNEL_PRESETS,
  type ChannelId,
  findChannel,
} from "../lib/channels";
import {
  CAPABILITY_FILTER_OPTIONS,
  FORMAT_FILTER_OPTIONS,
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
  activeChannelId,
  onChannelSelect,
  onRefresh,
}: {
  tab: ModelsTab;
  onTabChange: (tab: ModelsTab) => void;
  query: string;
  onQueryChange: (value: string) => void;
  isLoading: boolean;
  onRefresh: () => void;
  sortBy: HfSortKey;
  onSortChange: (value: HfSortKey) => void;
  resourceType: ResourceTypeFilter;
  onResourceTypeChange: (value: ResourceTypeFilter) => void;
  formatFilter: ModelFormatFilter;
  onFormatFilterChange: (value: ModelFormatFilter) => void;
  capabilityFilter: CapabilityFilter;
  onCapabilityFilterChange: (value: CapabilityFilter) => void;
  activeChannelId: ChannelId | null;
  onChannelSelect: (id: ChannelId | null) => void;
}) {
  const activeChannel = findChannel(activeChannelId);
  const isDataset = resourceType === "datasets";
  const triggerBase = cn(
    "field-trigger menu-trigger field-soft transition-colors",
    "focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-border",
  );
  return (
    <div className="flex flex-col gap-2 lg:flex-row lg:items-center">
      <div
        className={cn(
          "menu-trigger tab-toggle relative inline-flex h-9 w-full shrink-0 items-center rounded-full lg:w-[240px]",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "tab-toggle-pill",
            "pointer-events-none absolute inset-y-0 left-0 w-1/2 rounded-full transition-transform duration-200 ease-out",
            tab === "downloaded" ? "translate-x-full" : "translate-x-0",
          )}
        />
        <button
          type="button"
          onClick={() => onTabChange("discover")}
          className={cn(
            "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
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
            "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
            tab === "downloaded"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          On Device
        </button>
      </div>

      <div className="flex items-center gap-2 lg:contents">
      <div
        className="menu-trigger tab-toggle relative inline-flex h-9 shrink-0 items-center rounded-full"
        role="radiogroup"
        aria-label="Resource type"
      >
        <span
          aria-hidden="true"
          className={cn(
            "tab-toggle-pill",
            "pointer-events-none absolute inset-y-0 left-0 w-1/2 rounded-full transition-transform duration-200 ease-out",
            resourceType === "datasets" ? "translate-x-full" : "translate-x-0",
          )}
        />
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              role="radio"
              aria-checked={resourceType === "models"}
              aria-label="Models"
              onClick={() => onResourceTypeChange("models")}
              className={cn(
                "relative z-10 inline-flex h-9 w-9 items-center justify-center rounded-full transition-colors",
                resourceType === "models"
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={AiChipIcon}
                strokeWidth={1.75}
                className="size-4"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom" sideOffset={6}>
            Models
          </TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              role="radio"
              aria-checked={resourceType === "datasets"}
              aria-label="Datasets"
              onClick={() => onResourceTypeChange("datasets")}
              className={cn(
                "relative z-10 inline-flex h-9 w-9 items-center justify-center rounded-full transition-colors",
                resourceType === "datasets"
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={Database02Icon}
                strokeWidth={1.75}
                className="size-4"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom" sideOffset={6}>
            Datasets
          </TooltipContent>
        </Tooltip>
      </div>

      <div className="relative min-w-0 flex-1">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={1.8}
          className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search Hugging Face"
          className="field-soft h-9 rounded-full pl-10 pr-10 text-[12.5px] placeholder:text-muted-foreground/80"
        />
        {tab === "discover" && isLoading && (
          <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        )}
      </div>
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
              className={cn(triggerBase, "field-filter min-w-[124px]")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent
              position="popper"
              side="bottom"
              align="start"
              sideOffset={8}
              avoidCollisions={false}
              onCloseAutoFocus={(e) => e.preventDefault()}
              className="menu-instant menu-soft-surface rounded-[14px] ring-0"
            >
              {FORMAT_FILTER_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.value === "gguf" && (
                    <span className="inline-block size-1.5 shrink-0 rounded-full bg-format-gguf" />
                  )}
                  {option.value === "checkpoint" && (
                    <span className="inline-block size-1.5 shrink-0 rounded-full bg-format-checkpoint" />
                  )}
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
              className={cn(triggerBase, "field-filter min-w-[136px]")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent
            position="popper"
            side="bottom"
            align="start"
            sideOffset={8}
            avoidCollisions={false}
            onCloseAutoFocus={(e) => e.preventDefault()}
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
              className={cn(triggerBase, "field-filter min-w-[140px]")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent
            position="popper"
            side="bottom"
            align="start"
            sideOffset={8}
            avoidCollisions={false}
            onCloseAutoFocus={(e) => e.preventDefault()}
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

        {tab === "discover" && !isDataset && (
          <DropdownMenu>
            <Tooltip>
              <TooltipTrigger asChild>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    aria-label="Curated channels"
                    className={cn(
                      triggerBase,
                      "field-filter inline-flex size-9 shrink-0 items-center justify-center rounded-full p-0",
                    )}
                  >
                    <span
                      aria-hidden="true"
                      className="verified-badge size-4 text-primary"
                    />
                  </button>
                </DropdownMenuTrigger>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={6}>
                {activeChannel ? activeChannel.label : "Curated channels"}
              </TooltipContent>
            </Tooltip>
            <DropdownMenuContent
              align="end"
              sideOffset={8}
              onCloseAutoFocus={(e) => e.preventDefault()}
              className="menu-instant menu-soft-surface min-w-[220px] rounded-[14px] ring-0 p-1"
            >
              <DropdownMenuItem
                onSelect={() => onChannelSelect(null)}
                className="relative flex w-full cursor-pointer items-center gap-2.5 rounded-xl corner-squircle py-2 pr-8 pl-3 text-sm select-none"
              >
                All models
                {activeChannelId === null && (
                  <span className="pointer-events-none absolute right-2 flex size-4 items-center justify-center">
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      strokeWidth={2}
                      className="size-4"
                    />
                  </span>
                )}
              </DropdownMenuItem>
              {CHANNEL_PRESETS.map((preset) => (
                <DropdownMenuItem
                  key={preset.id}
                  onSelect={() => onChannelSelect(preset.id)}
                  className="relative flex w-full cursor-pointer items-center gap-2.5 rounded-xl corner-squircle py-2 pr-8 pl-3 text-sm select-none"
                >
                  {preset.label}
                  {activeChannelId === preset.id && (
                    <span className="pointer-events-none absolute right-2 flex size-4 items-center justify-center">
                      <HugeiconsIcon
                        icon={Tick02Icon}
                        strokeWidth={2}
                        className="size-4"
                      />
                    </span>
                  )}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}

        {tab === "discover" && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label="Refresh from Hugging Face"
                onClick={onRefresh}
                className={cn(
                  triggerBase,
                  "field-filter inline-flex size-9 shrink-0 items-center justify-center rounded-full p-0",
                )}
              >
                <HugeiconsIcon
                  icon={RefreshIcon}
                  strokeWidth={1.75}
                  className={cn("size-4", isLoading && "animate-spin")}
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" sideOffset={6}>
              Refresh from Hugging Face
            </TooltipContent>
          </Tooltip>
        )}

      </div>
    </div>
  );
}
