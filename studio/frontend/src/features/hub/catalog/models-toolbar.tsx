// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
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
  FolderSearchIcon,
  Refresh01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { HfSortKey } from "@/features/hub/hooks/use-hub-model-search";
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
import { HubOptionMenu, type HubOption } from "./hub-option-menu";
import { memo, useMemo } from "react";

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

type ChannelOptionValue = "all" | ChannelId;

export const ModelsToolbar = memo(function ModelsToolbar({
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
  onManageLocalFolders,
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
  onManageLocalFolders: () => void;
}) {
  const activeChannel = findChannel(activeChannelId);
  const isDataset = resourceType === "datasets";
  const channelValue: ChannelOptionValue = activeChannelId ?? "all";
  const formatOptions = useMemo<HubOption<ModelFormatFilter>[]>(
    () =>
      FORMAT_FILTER_OPTIONS.map((option) => ({
        value: option.value,
        triggerLabel: option.label,
        label: (
          <>
            {option.value === "gguf" && (
              <span className="inline-block size-1.5 shrink-0 rounded-full bg-format-gguf" />
            )}
            {option.value === "checkpoint" && (
              <span className="inline-block size-1.5 shrink-0 rounded-full bg-format-checkpoint" />
            )}
            {option.label}
          </>
        ),
      })),
    [],
  );
  const capabilityOptions = useMemo<HubOption<CapabilityFilter>[]>(
    () =>
      CAPABILITY_FILTER_OPTIONS.map((option) => ({
        value: option.value,
        label: option.label,
      })),
    [],
  );
  const sortOptions = useMemo<HubOption<HfSortKey>[]>(
    () =>
      SORT_OPTIONS.map((option) => ({
        value: option.value,
        label: option.label,
      })),
    [],
  );
  const channelOptions = useMemo<HubOption<ChannelOptionValue>[]>(
    () => [
      { value: "all", label: "All models" },
      ...CHANNEL_PRESETS.map((preset) => ({
        value: preset.id,
        checkClassName: "text-primary",
        label: (
          <span className="flex w-full min-w-0 items-start gap-2">
            <HugeiconsIcon
              icon={preset.icon}
              strokeWidth={1.75}
              className="mt-0.5 size-4 shrink-0 text-muted-foreground"
            />
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="min-w-0 truncate">{preset.label}</span>
              <span className="min-w-0 whitespace-normal break-words text-[11px] leading-snug text-muted-foreground">
                {preset.hint}
              </span>
            </span>
          </span>
        ),
      })),
    ],
    [],
  );
  const triggerBase = cn(
    "field-trigger hub-menu-trigger field-soft transition-colors",
    "focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-border",
  );
  return (
    <div className="flex min-w-0 flex-col gap-2 lg:flex-row lg:flex-wrap lg:items-center">
      <div
        className={cn(
          "hub-menu-trigger hub-tab-toggle relative inline-flex h-9 w-full shrink-0 items-center rounded-full lg:w-[240px]",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "hub-tab-toggle-pill",
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

      <div className="flex min-w-0 items-center gap-2 lg:flex-[1_1_280px]">
        <div
          className="hub-menu-trigger hub-tab-toggle relative inline-flex h-9 shrink-0 items-center rounded-full"
          role="radiogroup"
          aria-label="Resource type"
        >
          <span
            aria-hidden="true"
            className={cn(
              "hub-tab-toggle-pill",
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
            placeholder={
              tab === "downloaded"
                ? `Search on-device ${isDataset ? "datasets" : "models"}`
                : "Search Hugging Face"
            }
            className="field-soft h-9 rounded-full pl-10 pr-10 text-[12.5px] placeholder:text-muted-foreground/80"
          />
          {tab === "discover" && isLoading && (
            <Spinner className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          )}
        </div>
      </div>

      <div className="flex min-w-0 flex-wrap items-center gap-2 lg:flex-[0_1_auto] lg:justify-end">
        {tab === "downloaded" && !isDataset && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={onManageLocalFolders}
                className={cn(
                  triggerBase,
                  "field-filter inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3 text-[12.5px]",
                )}
              >
                <HugeiconsIcon
                  icon={FolderSearchIcon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                Add folder
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" sideOffset={6}>
              Manage local model folders
            </TooltipContent>
          </Tooltip>
        )}

        {!isDataset && (
          <HubOptionMenu
            value={formatFilter}
            options={formatOptions}
            onValueChange={onFormatFilterChange}
            ariaLabel="Format filter"
            className={cn(triggerBase, "min-w-[124px]")}
          />
        )}

        {tab === "discover" && !isDataset && (
          <HubOptionMenu
            value={capabilityFilter}
            options={capabilityOptions}
            onValueChange={onCapabilityFilterChange}
            ariaLabel="Capability filter"
            className={cn(triggerBase, "min-w-[136px]")}
          />
        )}

        {tab === "discover" && (
          <HubOptionMenu
            value={sortBy}
            options={sortOptions}
            onValueChange={onSortChange}
            ariaLabel="Sort models"
            className={cn(triggerBase, "min-w-[140px]")}
          />
        )}

        {tab === "discover" && !isDataset && (
          <HubOptionMenu
            value={channelValue}
            options={channelOptions}
            onValueChange={(next) =>
              onChannelSelect(next === "all" ? null : next)
            }
            ariaLabel="Curated channels"
            title={activeChannel ? activeChannel.label : "Curated channels"}
            align="end"
            showChevron={false}
            triggerContent={
              <span
                aria-hidden="true"
                className={cn(
                  "hub-verified-badge size-4",
                  activeChannel ? "text-primary" : "text-muted-foreground",
                )}
              />
            }
            className={cn(
              triggerBase,
              "field-filter inline-flex size-9 shrink-0 items-center justify-center rounded-full p-0",
            )}
            contentClassName="w-[calc(100vw-1.5rem)] sm:w-[320px]"
          />
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
                  icon={Refresh01Icon}
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
});
