// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { HfSortKey } from "@/features/hub/hooks/use-hub-model-search";
import { useHubSource } from "@/lib/hf-endpoint";
import { cn } from "@/lib/utils";
import {
  ArrowUpDownIcon,
  CancelCircleIcon,
  Delete02Icon,
  FolderSearchIcon,
  Search01Icon,
  SlidersHorizontalIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { HubCapabilityFilter } from "../lib/model-capabilities";
import type {
  HubUploader,
  ModelSearchFilters,
} from "../lib/model-search-filters";
import {
  clearRecentSearches,
  recordRecentSearch,
  removeRecentSearch,
  useRecentSearches,
} from "../lib/recent-searches";
import type { DatasetModality } from "../lib/dataset-filters";
import { FORMAT_FILTER_OPTIONS } from "../lib/view-models";
import type {
  ModelFormatFilter,
  ModelsTab,
  ResourceTypeFilter,
} from "../types";
import { type HubOption, HubOptionMenu } from "./hub-option-menu";
import {
  ModelSearchFiltersControl,
  type OnDeviceFilters,
} from "./model-search-filters";
import type { InventorySort } from "./models-table";
import { RecentSearches } from "./recent-searches";

// Includes the "Fine-tune ready" pseudo-option, which opens a channel rather than filtering.
type FormatMenuValue = ModelFormatFilter | "finetune";

const SORT_OPTIONS: ReadonlyArray<{
  value: HfSortKey;
  label: string;
}> = [
  { value: "fresh", label: "Latest" },
  { value: "createdAt", label: "Newest" },
  { value: "trendingScore", label: "Trending" },
  { value: "downloads", label: "Most downloads" },
  { value: "lastModified", label: "Recently updated" },
  { value: "likes", label: "Most likes" },
];

const INVENTORY_SORT_OPTIONS: HubOption<InventorySort>[] = [
  { value: "recent", label: "Recent" },
  { value: "name", label: "Name" },
  { value: "size", label: "Size" },
  { value: "format", label: "Format" },
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
  capabilityFilters,
  onCapabilityFiltersChange,
  isolateFirstCapability,
  datasetModalities,
  onDatasetModalitiesChange,
  fitOnDeviceOnly,
  onFitOnDeviceOnlyChange,
  onManageLocalFolders,
  onFreeUpSpace,
  onOpenFineTune,
  npuAvailable = false,
  searchFilters,
  onSearchFiltersChange,
  uploaders,
  uploadersChanged,
  onUploadersChange,
  onResetFilters,
  inventorySort,
  onInventorySortChange,
  inventoryCapabilities,
  onInventoryCapabilitiesChange,
  isolateFirstInventoryCapability,
  onResetInventoryFilters,
}: {
  searchFilters: ModelSearchFilters;
  onSearchFiltersChange: (value: ModelSearchFilters) => void;
  uploaders: readonly HubUploader[];
  /** False while the default applies (Unsloth's uploads, or every uploader for a search). */
  uploadersChanged: boolean;
  onUploadersChange: (value: HubUploader[]) => void;
  onResetFilters: () => void;
  inventorySort: InventorySort;
  onInventorySortChange: (value: InventorySort) => void;
  inventoryCapabilities: HubCapabilityFilter[];
  onInventoryCapabilitiesChange: (value: HubCapabilityFilter[]) => void;
  isolateFirstInventoryCapability: boolean;
  onResetInventoryFilters: () => void;
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
  /** None picked means every capability. */
  capabilityFilters: HubCapabilityFilter[];
  onCapabilityFiltersChange: (value: HubCapabilityFilter[]) => void;
  /** Until the first change from the defaults, a capability pick keeps just that one. */
  isolateFirstCapability: boolean;
  /** Empty means every modality. */
  datasetModalities: DatasetModality[];
  onDatasetModalitiesChange: (value: DatasetModality[]) => void;
  /** Shared with the chat model selector: hide models over the device budget. */
  fitOnDeviceOnly: boolean;
  onFitOnDeviceOnlyChange: (value: boolean) => void;
  onManageLocalFolders: () => void;
  onFreeUpSpace: () => void;
  /** Opens the curated "Fine-tune ready" channel (discover only). */
  onOpenFineTune: () => void;
  npuAvailable?: boolean;
}) {
  const viewPillId = useId();
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

  const toolbarRef = useRef<HTMLDivElement | null>(null);
  const searchWrapRef = useRef<HTMLDivElement | null>(null);
  const [recentPanelTop, setRecentPanelTop] = useState<number | undefined>();
  useLayoutEffect(() => {
    if (!showRecentSearches) {
      return;
    }
    const measure = () => {
      const toolbar = toolbarRef.current;
      const wrap = searchWrapRef.current;
      if (!(toolbar && wrap)) {
        return;
      }
      setRecentPanelTop(
        toolbar.getBoundingClientRect().bottom -
          wrap.getBoundingClientRect().top,
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(toolbarRef.current as HTMLDivElement);
    return () => observer.disconnect();
  }, [showRecentSearches]);

  const isDataset = resourceType === "datasets";
  const npuList = formatFilter === "npu" && !isDataset;
  const hasTrailing = Boolean(query) || (isDiscover && isLoading);
  const formatOptions = useMemo<HubOption<FormatMenuValue>[]>(() => {
    const options: HubOption<FormatMenuValue>[] = FORMAT_FILTER_OPTIONS.filter(
      (option) =>
        option.value === "unsloth"
          ? tab === "downloaded"
          : option.value !== "recommended" || tab === "discover",
    ).map((option) => ({
      value: option.value,
      triggerLabel: option.label,
      label: (
        <>
          <span className="flex size-3.5 shrink-0 items-center justify-center">
            {option.value === "recommended" && (
              <HugeiconsIcon
                icon={SparklesIcon}
                strokeWidth={1.75}
                className="size-3.5 text-muted-foreground"
              />
            )}
            {option.value === "gguf" && (
              <span className="size-1.5 rounded-full bg-format-gguf" />
            )}
            {option.value === "checkpoint" && (
              <span className="size-1.5 rounded-full bg-format-checkpoint" />
            )}
            {option.value === "mlx" && (
              <span className="size-1.5 rounded-full bg-format-mlx" />
            )}
            {option.value === "unsloth" && (
              <span className="size-1.5 rounded-full bg-primary" />
            )}
          </span>
          {option.label}
        </>
      ),
    }));
    if (npuAvailable) {
      options.push({
        value: "npu",
        triggerLabel: "NPU",
        label: (
          <>
            <span className="flex size-3.5 shrink-0 items-center justify-center">
              <span className="size-1.5 rounded-full bg-format-npu" />
            </span>
            NPU
          </>
        ),
      });
    }
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
  }, [tab, npuAvailable]);
  const onDeviceFilters = useMemo<OnDeviceFilters | undefined>(
    () =>
      tab === "downloaded"
        ? {
            capabilities: inventoryCapabilities,
            onCapabilitiesChange: onInventoryCapabilitiesChange,
            isolateFirstCapability: isolateFirstInventoryCapability,
            onReset: onResetInventoryFilters,
          }
        : undefined,
    [
      tab,
      inventoryCapabilities,
      onInventoryCapabilitiesChange,
      isolateFirstInventoryCapability,
      onResetInventoryFilters,
    ],
  );
  const hubSource = useHubSource();
  const showCapabilities = isDiscover && !isDataset && !npuList;
  // The size and context limits read Hugging Face metadata, so only there.
  const showModelFilters = showCapabilities && hubSource !== "modelscope";
  const sortOptions = useMemo<HubOption<HfSortKey>[]>(
    () =>
      SORT_OPTIONS.filter(
        (option) =>
          // Latest ranks Hugging Face models; datasets and ModelScope have no such ranking.
          !(option.value === "fresh" && (isDataset || hubSource === "modelscope")) &&
          (hubSource !== "modelscope" || option.value !== "createdAt"),
      ).map((option) => ({
        value: option.value,
        label: option.label,
      })),
    [hubSource, isDataset],
  );
  // pl-4 / pr-3.5: the model picker's dropdown insets, so the two read alike.
  const triggerBase = cn(
    "field-trigger hub-menu-trigger field-soft transition-colors pl-4 pr-3.5",
    "focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-border",
  );
  return (
    <div
      ref={toolbarRef}
      className="flex min-w-0 flex-col gap-2 lg:flex-row lg:flex-wrap lg:items-center"
    >
      <div
        data-tour="hub-tabs"
        className={cn(
          "hub-menu-trigger hub-tab-toggle relative inline-flex h-9 w-full shrink-0 items-center rounded-full lg:w-[calc(280px*var(--ui-space-scale,1))]",
        )}
        role="radiogroup"
        aria-label="View"
      >
        <button
          type="button"
          role="radio"
          aria-checked={tab === "discover"}
          onClick={() => onTabChange("discover")}
          className={cn(
            "relative inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-ui-12p5 transition-colors",
            tab === "discover"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {tab === "discover" && <SegmentedPill group={viewPillId} />}
          Discover
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={tab === "downloaded"}
          onClick={() => onTabChange("downloaded")}
          className={cn(
            "relative inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-ui-12p5 transition-colors",
            tab === "downloaded"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {tab === "downloaded" && <SegmentedPill group={viewPillId} />}
          On Device
        </button>
      </div>

      <div
        ref={searchWrapRef}
        data-tour="hub-search"
        className="relative min-w-0 flex-1 lg:min-w-[calc(220px*var(--ui-space-scale,1))] lg:flex-[1_1_220px]"
      >
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={1.8}
          className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          // Stops password managers and text assistance from acting on this field.
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
            "field-soft h-9 rounded-full !border-0 pl-10 text-ui-13 placeholder:text-muted-foreground/80 focus-visible:!ring-0",
            hasTrailing ? "pr-10" : "pr-4",
          )}
        />
        {query ? (
          <button
            type="button"
            aria-label="Clear search"
            // Keep focus on the input so clearing reveals recent searches.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => onQueryChange("")}
            className="absolute right-2.5 top-1/2 inline-flex size-6 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:text-foreground"
          >
            <HugeiconsIcon
              icon={CancelCircleIcon}
              strokeWidth={1.75}
              className="size-[calc(18px*var(--ui-space-scale,1))]"
            />
          </button>
        ) : isDiscover && isLoading ? (
          <Spinner className="pointer-events-none absolute right-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        ) : null}
        {showRecentSearches && (
          <RecentSearches
            top={recentPanelTop}
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
        <ModelSearchFiltersControl
          value={searchFilters}
          onChange={onSearchFiltersChange}
          capabilities={capabilityFilters}
          onCapabilitiesChange={onCapabilityFiltersChange}
          isolateFirstCapability={isolateFirstCapability}
          uploaders={uploaders}
          uploadersChanged={uploadersChanged}
          onUploadersChange={onUploadersChange}
          onReset={onResetFilters}
          showModelFilters={showModelFilters}
          datasets={isDataset}
          modelSections={showCapabilities}
          datasetSections={isDiscover && isDataset}
          modalities={datasetModalities}
          onModalitiesChange={onDatasetModalitiesChange}
          onDatasetsChange={(on) =>
            onResourceTypeChange(on ? "datasets" : "models")
          }
          onDevice={onDeviceFilters}
          className={cn(
            triggerBase,
            "relative inline-flex size-9 shrink-0 items-center justify-center rounded-full p-0",
          )}
        />
        {tab === "downloaded" && !isDataset && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                onClick={onManageLocalFolders}
                aria-label="Model folders"
                className={cn(
                  triggerBase,
                  "field-filter relative inline-flex size-9 shrink-0 items-center justify-center rounded-full p-0",
                )}
              >
                <HugeiconsIcon
                  icon={FolderSearchIcon}
                  strokeWidth={1.75}
                  className="size-[calc(18px*var(--ui-space-scale,1))]"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" sideOffset={6}>
              Model folders
            </TooltipContent>
          </Tooltip>
        )}

        {tab === "downloaded" && !isDataset && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                onClick={onFreeUpSpace}
                data-testid="free-up-space-trigger"
                aria-label="Free up space"
                className={cn(
                  triggerBase,
                  "field-filter relative inline-flex size-9 shrink-0 items-center justify-center rounded-full p-0",
                )}
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  strokeWidth={1.75}
                  className="size-[calc(18px*var(--ui-space-scale,1))]"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" sideOffset={6}>
              Free up space: remove shared assets no installed model needs
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
            className={cn(
              triggerBase,
              "w-[calc(160px*var(--ui-space-scale,1))]",
            )}
          />
        )}

        {tab === "downloaded" && (
          <HubOptionMenu<InventorySort>
            value={inventorySort}
            options={INVENTORY_SORT_OPTIONS}
            onValueChange={onInventorySortChange}
            ariaLabel="Sort on-device models"
            className={cn(
              triggerBase,
              "w-[calc(160px*var(--ui-space-scale,1))]",
            )}
            triggerContent={
              <span className="flex items-center gap-1">
                <HugeiconsIcon
                  icon={ArrowUpDownIcon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0"
                />
                <span className="truncate">
                  {INVENTORY_SORT_OPTIONS.find(
                    (option) => option.value === inventorySort,
                  )?.label ?? inventorySort}
                </span>
              </span>
            }
          />
        )}

        {tab === "discover" && !npuList && (
          <HubOptionMenu
            value={sortBy}
            options={sortOptions}
            onValueChange={onSortChange}
            ariaLabel="Sort models"
            className={cn(
              triggerBase,
              "w-[calc(160px*var(--ui-space-scale,1))]",
            )}
            triggerContent={
              <span className="flex items-center gap-1">
                <HugeiconsIcon
                  icon={ArrowUpDownIcon}
                  strokeWidth={1.75}
                  className="size-3.5 shrink-0"
                />
                <span className="truncate">
                  {SORT_OPTIONS.find((option) => option.value === sortBy)
                    ?.label ?? sortBy}
                </span>
              </span>
            }
            footer={
              isDataset ? undefined : (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      role="checkbox"
                      aria-checked={fitOnDeviceOnly}
                      onClick={() => onFitOnDeviceOnlyChange(!fitOnDeviceOnly)}
                      className="hub-menu-row relative flex w-full cursor-pointer select-none items-center gap-2.5 py-2 pr-3 pl-3 text-left text-sm"
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

      </div>
    </div>
  );
});
