// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { HfSortKey } from "@/features/hub/hooks/use-hub-model-search";
import { cn } from "@/lib/utils";
import {
  AiChipIcon,
  CancelCircleIcon,
  Database02Icon,
  FolderSearchIcon,
  Search01Icon,
  SlidersHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useMemo, useState } from "react";
import {
  clearRecentSearches,
  recordRecentSearch,
  removeRecentSearch,
  useRecentSearches,
} from "../lib/recent-searches";
import {
  CAPABILITY_FILTER_OPTIONS,
  FORMAT_FILTER_OPTIONS,
} from "../lib/view-models";
import type {
  CapabilityFilter,
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
} from "../types";
import { type HubOption, HubOptionMenu } from "./hub-option-menu";
import { RecentSearches } from "./recent-searches";

// Widened so the format dropdown can carry the "Fine-tune ready" pseudo-option,
// which opens the curated channel instead of becoming the active format filter.
type FormatMenuValue = ModelFormatFilter | "finetune";

const SORT_OPTIONS: ReadonlyArray<{
  value: HfSortKey;
  label: string;
}> = [
  { value: "createdAt", label: "Newest" },
  { value: "trendingScore", label: "Trending" },
  { value: "downloads", label: "Most downloads" },
  { value: "lastModified", label: "Recently updated" },
  { value: "likes", label: "Most likes" },
];

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
  fitOnDeviceOnly,
  onFitOnDeviceOnlyChange,
  onManageLocalFolders,
  onOpenFineTune,
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
  /** Shared with the chat model selector: hide models over the device budget. */
  fitOnDeviceOnly: boolean;
  onFitOnDeviceOnlyChange: (value: boolean) => void;
  onManageLocalFolders: () => void;
  /** Opens the curated "Fine-tune ready" channel (discover only). Exposed as a
   *  format-dropdown option rather than a standalone feed section. */
  onOpenFineTune: () => void;
}) {
  // Recent searches surface while the empty search field is focused, only on
  // Discover (on-device search is a local filter and isn't recorded).
  const recentSearches = useRecentSearches();
  const [searchFocused, setSearchFocused] = useState(false);
  const isDiscover = tab === "discover";
  const showRecentSearches =
    isDiscover &&
    searchFocused &&
    query.trim() === "" &&
    recentSearches.length > 0;

  const isDataset = resourceType === "datasets";
  const hasTrailing = Boolean(query) || (isDiscover && isLoading);
  const formatOptions = useMemo<HubOption<FormatMenuValue>[]>(() => {
    const options: HubOption<FormatMenuValue>[] = FORMAT_FILTER_OPTIONS.filter(
      (option) => option.value !== "mlx" || tab === "discover",
    ).map((option) => ({
      value: option.value,
      triggerLabel: option.label,
      label: (
        <>
          <span className="flex size-3.5 shrink-0 items-center justify-center">
            {option.value === "gguf" && (
              <span className="size-1.5 rounded-full bg-format-gguf" />
            )}
            {option.value === "checkpoint" && (
              <span className="size-1.5 rounded-full bg-format-checkpoint" />
            )}
            {option.value === "mlx" && (
              <span className="size-1.5 rounded-full bg-format-mlx" />
            )}
          </span>
          {option.label}
        </>
      ),
    }));
    // "Fine-tune ready" is a curated channel (bnb-4bit checkpoints), not a
    // format: it opens the channel rather than setting the filter (onValueChange).
    if (tab === "discover") {
      options.push({
        value: "finetune",
        triggerLabel: "Fine-tune ready",
        label: (
          <>
            <HugeiconsIcon
              icon={SlidersHorizontalIcon}
              strokeWidth={1.75}
              className="size-3.5 shrink-0 text-muted-foreground"
            />
            Fine-tune ready
          </>
        ),
      });
    }
    return options;
  }, [tab]);
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
  const triggerBase = cn(
    "field-trigger hub-menu-trigger field-soft transition-colors",
    "focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-border",
  );
  return (
    <div className="flex min-w-0 flex-col gap-2 lg:flex-row lg:flex-nowrap lg:items-center">
      <div
        className={cn(
          "hub-menu-trigger hub-tab-toggle relative inline-flex h-9 w-full shrink-0 items-center rounded-full lg:w-[280px]",
        )}
        role="radiogroup"
        aria-label="View"
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
          role="radio"
          aria-checked={tab === "discover"}
          onClick={() => onTabChange("discover")}
          className={cn(
            "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[0.78125rem] transition-colors",
            tab === "discover"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          Discover
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={tab === "downloaded"}
          onClick={() => onTabChange("downloaded")}
          className={cn(
            "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[0.78125rem] transition-colors",
            tab === "downloaded"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          On Device
        </button>
      </div>

      <div className="relative min-w-0 flex-1 lg:flex-[1_1_360px]">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={1.8}
          className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          // `type="search"` plus these flags stop password managers and noisy
          // text assistance from acting on this field.
          type="search"
          name="hub-search"
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          spellCheck={false}
          enterKeyHint="search"
          data-1p-ignore={true}
          data-lpignore={true}
          data-form-type="other"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onFocus={() => setSearchFocused(true)}
          onBlur={() => {
            setSearchFocused(false);
            if (isDiscover) {
              recordRecentSearch(query);
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && isDiscover) {
              recordRecentSearch(query);
            } else if (event.key === "Escape" && showRecentSearches) {
              event.currentTarget.blur();
            }
          }}
          placeholder={
            tab === "downloaded"
              ? `Search on-device ${isDataset ? "datasets" : "models"}`
              : isDataset
                ? "Search datasets"
                : "Search all models"
          }
          className={cn(
            "field-soft h-9 rounded-full !border-0 pl-10 text-[0.8125rem] placeholder:text-muted-foreground/80 focus-visible:!ring-0",
            hasTrailing ? "pr-10" : "pr-4",
          )}
        />
        {query ? (
          <button
            type="button"
            aria-label="Clear search"
            // Keep focus on the input so clearing reveals recent searches
            // rather than dismissing the field.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => onQueryChange("")}
            className="absolute right-2.5 top-1/2 inline-flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:text-foreground"
          >
            <HugeiconsIcon
              icon={CancelCircleIcon}
              strokeWidth={1.75}
              className="size-[18px]"
            />
          </button>
        ) : isDiscover && isLoading ? (
          <Spinner className="pointer-events-none absolute right-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        ) : null}
        {showRecentSearches && (
          <RecentSearches
            searches={recentSearches}
            onSelect={(value) => {
              recordRecentSearch(value);
              onQueryChange(value);
            }}
            onRemove={removeRecentSearch}
            onClear={clearRecentSearches}
          />
        )}
      </div>

      <div className="flex min-w-0 flex-wrap items-center gap-2 lg:flex-[0_0_auto] lg:flex-nowrap lg:justify-end">
        {tab === "downloaded" && !isDataset && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                onClick={onManageLocalFolders}
                className={cn(
                  triggerBase,
                  "field-filter inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3 text-[0.78125rem]",
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
          <HubOptionMenu<FormatMenuValue>
            value={formatFilter}
            options={formatOptions}
            onValueChange={(value) => {
              if (value === "finetune") {
                onOpenFineTune();
              } else {
                onFormatFilterChange(value);
              }
            }}
            ariaLabel="Format filter"
            className={cn(triggerBase, "w-[128px]")}
          />
        )}

        {tab === "discover" && !isDataset && (
          <HubOptionMenu
            value={capabilityFilter}
            options={capabilityOptions}
            onValueChange={onCapabilityFilterChange}
            ariaLabel="Capability filter"
            className={cn(triggerBase, "w-[128px]")}
          />
        )}

        {tab === "discover" && (
          <HubOptionMenu
            value={sortBy}
            options={sortOptions}
            onValueChange={onSortChange}
            ariaLabel="Sort models"
            className={cn(triggerBase, "w-[128px]")}
            footer={
              isDataset ? undefined : (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      role="checkbox"
                      aria-checked={fitOnDeviceOnly}
                      onClick={() => onFitOnDeviceOnlyChange(!fitOnDeviceOnly)}
                      className="flex w-full cursor-pointer select-none items-center gap-2 rounded-[10px] px-3 py-2 text-left text-[0.78125rem] text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <Checkbox
                        checked={fitOnDeviceOnly}
                        tabIndex={-1}
                        aria-hidden={true}
                        className="pointer-events-none size-3.5 rounded-full [&_svg]:!size-2.5"
                      />
                      Only show models that fit
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    Hides models larger than this device's memory budget.
                    Downloaded models stay visible.
                  </TooltipContent>
                </Tooltip>
              )
            }
          />
        )}

        <div
          className="hub-menu-trigger hub-tab-toggle relative inline-flex h-8 shrink-0 items-center rounded-full"
          role="radiogroup"
          aria-label="Resource type"
        >
          <span
            aria-hidden="true"
            className={cn(
              "hub-tab-toggle-pill",
              // Height-filling circle (matches the 3-view-tab toggle) that slides
              // one button-width (w-8 = 32px) between the two options.
              "pointer-events-none absolute inset-y-0 left-0 w-8 rounded-full transition-transform duration-200 ease-out",
              resourceType === "datasets" ? "translate-x-8" : "translate-x-0",
            )}
          />
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                role="radio"
                aria-checked={resourceType === "models"}
                aria-label="Models"
                onClick={() => onResourceTypeChange("models")}
                className={cn(
                  "relative z-10 inline-flex size-8 items-center justify-center rounded-full transition-colors",
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
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                role="radio"
                aria-checked={resourceType === "datasets"}
                aria-label="Datasets"
                onClick={() => onResourceTypeChange("datasets")}
                className={cn(
                  "relative z-10 inline-flex size-8 items-center justify-center rounded-full transition-colors",
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
      </div>
    </div>
  );
});
