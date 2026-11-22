// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
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
  ClockIcon,
  SearchIcon,
  SparklesIcon,
  TagIcon,
  XIcon,
  ZapIcon,
} from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { fetchDeepInfraModels } from "./api/providers-api";
import {
  type DeepInfraRawModel,
  type ModelCatalogItem,
  type SortDirection,
  type SortField,
  filterModels,
  formatContextTokens,
  formatPricePerMillion,
  parseDeepInfraCatalog,
  parseModelIds,
  sortModels,
} from "./deepinfra-model-catalog";

export interface DeepInfraModelSelectionPanelProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  onEnterPress?: () => void;
  defaultModelIds?: string;
  onCatalogLoaded?: (models: string[]) => void;
}

const DEEPINFRA_MODELS_URL = "https://api.deepinfra.com/models/list";
let deepinfraCache: ModelCatalogItem[] | null = null;

export function DeepInfraModelSelectionPanel({
  value,
  onChange,
  disabled = false,
  onEnterPress,
  defaultModelIds = "deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1, meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen2.5-Coder-32B-Instruct",
  onCatalogLoaded,
}: DeepInfraModelSelectionPanelProps): ReactNode {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState("all");
  const [selectedQuant, setSelectedQuant] = useState("all");
  const [minContext, setMinContext] = useState<number>(0);
  const [featuredOnly, setFeaturedOnly] = useState(false);
  const [discountOnly, setDiscountOnly] = useState(false);
  const [priorityOnly, setPriorityOnly] = useState(false);
  const [flexOnly, setFlexOnly] = useState(false);
  const [sortField, setSortField] = useState<SortField>("name");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [loading, setLoading] = useState(() => !deepinfraCache);
  const [models, setModels] = useState<ModelCatalogItem[]>(
    () => deepinfraCache ?? [],
  );

  const onCatalogLoadedRef = useRef(onCatalogLoaded);
  useEffect(() => {
    onCatalogLoadedRef.current = onCatalogLoaded;
  }, [onCatalogLoaded]);

  useEffect(() => {
    if (deepinfraCache) {
      onCatalogLoadedRef.current?.(deepinfraCache.map((m) => m.id));
      return;
    }

    let isMounted = true;
    const controller = new AbortController();

    const loadModels = async () => {
      try {
        let rawList: DeepInfraRawModel[] | null = null;
        // 1. Try daemon backend proxy endpoint first
        try {
          rawList = (await fetchDeepInfraModels()) as DeepInfraRawModel[];
        } catch {
          // Backend proxy endpoint not reachable or network error, fallback to direct API
        }

        // 2. If backend endpoint didn't provide a list, try direct API
        if (!rawList || !Array.isArray(rawList)) {
          const res = await fetch(DEEPINFRA_MODELS_URL, {
            signal: controller.signal,
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          rawList = (await res.json()) as DeepInfraRawModel[];
        }

        if (!isMounted) return;
        const catalog = parseDeepInfraCatalog(rawList);

        deepinfraCache = catalog;
        setModels(catalog);
        onCatalogLoadedRef.current?.(catalog.map((m) => m.id));
      } catch (err: unknown) {
        if (controller.signal.aborted) return;
        console.warn(
          "[DeepInfraModelSelectionPanel] Failed to fetch DeepInfra models:",
          err,
        );
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    void loadModels();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, []);

  const selectedIds = useMemo(() => parseModelIds(value), [value]);

  const availableTypes = useMemo(() => {
    const types = new Set<string>();
    for (const m of models) {
      if (m.type) types.add(m.type);
    }
    return ["all", ...Array.from(types).sort()];
  }, [models]);

  const availableQuants = useMemo(() => {
    const quants = new Set<string>();
    for (const m of models) {
      if (m.quantization) quants.add(m.quantization);
    }
    return ["all", ...Array.from(quants).sort()];
  }, [models]);

  const filteredModels = useMemo(() => {
    return filterModels(models, {
      searchQuery,
      selectedType,
      selectedQuant,
      minContext,
      featuredOnly,
      discountOnly,
      priorityOnly,
      flexOnly,
    });
  }, [
    models,
    searchQuery,
    selectedType,
    selectedQuant,
    featuredOnly,
    discountOnly,
    priorityOnly,
    flexOnly,
    minContext,
  ]);

  const sortedModels = useMemo(() => {
    return sortModels(filteredModels, sortField, sortDirection);
  }, [filteredModels, sortField, sortDirection]);

  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) {
        setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortField(field);
        setSortDirection("asc");
      }
    },
    [sortField],
  );

  const toggleModel = useCallback(
    (modelId: string) => {
      if (disabled) return;
      const current = parseModelIds(value);
      const next = current.includes(modelId)
        ? current.filter((id) => id !== modelId)
        : [...current, modelId];
      onChange(next.join(", "));
    },
    [disabled, onChange, value],
  );

  const removeModel = useCallback(
    (modelId: string) => {
      if (disabled) return;
      const current = parseModelIds(value);
      onChange(current.filter((id) => id !== modelId).join(", "));
    },
    [disabled, onChange, value],
  );

  const handleSelectAllFiltered = useCallback(() => {
    if (disabled) return;
    const current = parseModelIds(value);
    const combined = [
      ...new Set([...current, ...sortedModels.map((m) => m.id)]),
    ];
    onChange(combined.join(", "));
  }, [disabled, sortedModels, onChange, value]);

  const handleClearSelection = useCallback(() => {
    if (disabled) return;
    onChange("");
  }, [disabled, onChange]);

  return (
    <div className="flex w-full flex-col gap-3 font-sans">
      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Search */}
        <div className="relative min-w-[160px] flex-1">
          <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          <Input
            className="pl-9 text-xs"
            placeholder="Search model name..."
            value={searchQuery}
            disabled={disabled}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        {/* Type Filter */}
        <div className="w-[130px] shrink-0">
          <Select
            value={selectedType}
            disabled={disabled}
            onValueChange={setSelectedType}
          >
            <SelectTrigger aria-label="Filter by type" className="h-9 text-xs">
              <SelectValue placeholder="All types" />
            </SelectTrigger>
            <SelectContent>
              {availableTypes.map((type) => (
                <SelectItem key={type} value={type} className="text-xs">
                  {type === "all" ? "All Types" : type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Quantization Filter */}
        <div className="w-[120px] shrink-0">
          <Select
            value={selectedQuant}
            disabled={disabled}
            onValueChange={setSelectedQuant}
          >
            <SelectTrigger
              aria-label="Filter by quantization"
              className="h-9 text-xs"
            >
              <SelectValue placeholder="All Quants" />
            </SelectTrigger>
            <SelectContent>
              {availableQuants.map((quant) => (
                <SelectItem key={quant} value={quant} className="text-xs">
                  {quant === "all" ? "All Quants" : quant}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Context Filter */}
        <div className="w-[130px] shrink-0">
          <Select
            value={String(minContext)}
            disabled={disabled}
            onValueChange={(val) => setMinContext(Number(val))}
          >
            <SelectTrigger
              aria-label="Filter by context window"
              className="h-9 text-xs"
            >
              <SelectValue placeholder="All contexts" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0" className="text-xs">
                All Contexts
              </SelectItem>
              <SelectItem value="32000" className="text-xs">
                ≥ 32k tokens
              </SelectItem>
              <SelectItem value="64000" className="text-xs">
                ≥ 64k tokens
              </SelectItem>
              <SelectItem value="128000" className="text-xs">
                ≥ 128k tokens
              </SelectItem>
              <SelectItem value="200000" className="text-xs">
                ≥ 200k tokens
              </SelectItem>
              <SelectItem value="1000000" className="text-xs">
                ≥ 1024k tokens
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Toggle Buttons Group: Featured / Discount / Priority / Flex */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            className={cn(
              "inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground transition-all hover:bg-secondary hover:text-foreground",
              featuredOnly &&
                "border-purple-500/60 bg-purple-500/15 font-medium text-purple-400 dark:text-purple-300",
            )}
            onClick={() => setFeaturedOnly((prev) => !prev)}
            disabled={disabled}
            title="Filter featured models"
          >
            <SparklesIcon className="size-3.5 text-purple-400" />
            <span>Featured</span>
          </button>

          <button
            type="button"
            className={cn(
              "inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground transition-all hover:bg-secondary hover:text-foreground",
              discountOnly &&
                "border-emerald-500/60 bg-emerald-500/15 font-medium text-emerald-400 dark:text-emerald-300",
            )}
            onClick={() => setDiscountOnly((prev) => !prev)}
            disabled={disabled}
            title="Filter discounted models"
          >
            <TagIcon className="size-3.5 text-emerald-400" />
            <span>Discount</span>
          </button>

          <button
            type="button"
            className={cn(
              "inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground transition-all hover:bg-secondary hover:text-foreground",
              priorityOnly &&
                "border-amber-500/60 bg-amber-500/15 font-medium text-amber-400 dark:text-amber-300",
            )}
            onClick={() => setPriorityOnly((prev) => !prev)}
            disabled={disabled}
            title="Filter models supporting priority service tier"
          >
            <ZapIcon className="size-3.5 text-amber-400" />
            <span>Priority</span>
          </button>

          <button
            type="button"
            className={cn(
              "inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground transition-all hover:bg-secondary hover:text-foreground",
              flexOnly &&
                "border-blue-500/60 bg-blue-500/15 font-medium text-blue-400 dark:text-blue-300",
            )}
            onClick={() => setFlexOnly((prev) => !prev)}
            disabled={disabled}
            title="Filter models supporting flex service tier"
          >
            <ClockIcon className="size-3.5 text-blue-400" />
            <span>Flex</span>
          </button>
        </div>
      </div>

      {/* Model Table List */}
      <div className="max-h-[320px] overflow-y-auto rounded-md border border-border bg-background">
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-xs text-muted-foreground">
            <Spinner className="size-4" />
            <span>Loading models...</span>
          </div>
        ) : sortedModels.length === 0 ? (
          <div className="py-8 text-center text-xs text-muted-foreground">
            <span>No models found matching criteria.</span>
          </div>
        ) : (
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="sticky top-0 z-10 border-b border-border bg-secondary text-foreground">
                <th className="w-9 px-2.5 py-2"></th>
                <th
                  className="px-2.5 py-2 font-semibold"
                  aria-sort={
                    sortField === "name"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className="inline-flex cursor-pointer items-center gap-1 hover:text-primary"
                    onClick={() => handleSort("name")}
                  >
                    <span>Model</span>
                    {sortField === "name" ? (
                      <span className="text-primary font-bold">
                        {sortDirection === "asc" ? "↑" : "↓"}
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40">↕</span>
                    )}
                  </button>
                </th>
                <th
                  className="px-2.5 py-2 font-semibold"
                  aria-sort={
                    sortField === "context"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className="inline-flex cursor-pointer items-center gap-1 hover:text-primary"
                    onClick={() => handleSort("context")}
                  >
                    <span>Context</span>
                    {sortField === "context" ? (
                      <span className="text-primary font-bold">
                        {sortDirection === "asc" ? "↑" : "↓"}
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40">↕</span>
                    )}
                  </button>
                </th>
                <th
                  className="px-2.5 py-2 font-semibold"
                  aria-sort={
                    sortField === "quant"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className="inline-flex cursor-pointer items-center gap-1 hover:text-primary"
                    onClick={() => handleSort("quant")}
                  >
                    <span>Quant</span>
                    {sortField === "quant" ? (
                      <span className="text-primary font-bold">
                        {sortDirection === "asc" ? "↑" : "↓"}
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40">↕</span>
                    )}
                  </button>
                </th>
                <th
                  className="px-2.5 py-2 font-semibold"
                  aria-sort={
                    sortField === "priceIn"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className="inline-flex cursor-pointer items-center gap-1 hover:text-primary"
                    onClick={() => handleSort("priceIn")}
                    title="Sort by input price per million tokens"
                  >
                    <span>In ($/Mt)</span>
                    {sortField === "priceIn" ? (
                      <span className="text-primary font-bold">
                        {sortDirection === "asc" ? "↑" : "↓"}
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40">↕</span>
                    )}
                  </button>
                </th>
                <th
                  className="px-2.5 py-2 font-semibold"
                  aria-sort={
                    sortField === "priceOut"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className="inline-flex cursor-pointer items-center gap-1 hover:text-primary"
                    onClick={() => handleSort("priceOut")}
                    title="Sort by output price per million tokens"
                  >
                    <span>Out ($/Mt)</span>
                    {sortField === "priceOut" ? (
                      <span className="text-primary font-bold">
                        {sortDirection === "asc" ? "↑" : "↓"}
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40">↕</span>
                    )}
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((model) => {
                const isSelected = selectedIds.includes(model.id);
                return (
                  <tr
                    key={model.id}
                    className={cn(
                      "cursor-pointer border-b border-border transition-colors hover:bg-secondary/60",
                      isSelected && "bg-primary/10 hover:bg-primary/15",
                    )}
                    onClick={() => toggleModel(model.id)}
                  >
                    <td
                      className="px-2.5 py-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Checkbox
                        checked={isSelected}
                        disabled={disabled}
                        onCheckedChange={() => toggleModel(model.id)}
                        aria-label={`Select ${model.name}`}
                      />
                    </td>
                    <td className="px-2.5 py-2">
                      <div className="flex min-w-[180px] flex-col gap-0.5">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="font-semibold text-foreground break-all">
                            {model.name}
                          </span>
                          {model.isFeatured && (
                            <span
                              title="Featured model"
                              className="inline-flex items-center"
                            >
                              <SparklesIcon
                                className="size-3 text-purple-400"
                                aria-label="Featured model"
                              />
                            </span>
                          )}
                          {model.hasDiscount && (
                            <span
                              title={
                                model.discountPercent != null
                                  ? `Discounted (${model.discountPercent}% off)`
                                  : "Discounted pricing available"
                              }
                              className="inline-flex items-center"
                            >
                              <TagIcon
                                className="size-3 text-emerald-400"
                                aria-label="Discount available"
                              />
                            </span>
                          )}
                          {model.hasPriority && (
                            <span
                              title="Supports Priority Service Tier"
                              className="inline-flex items-center"
                            >
                              <ZapIcon
                                className="size-3.5 text-amber-400"
                                aria-label="Priority tier"
                              />
                            </span>
                          )}
                          {model.hasFlex && (
                            <span
                              title="Supports Flex Service Tier"
                              className="inline-flex items-center"
                            >
                              <ClockIcon
                                className="size-3 text-blue-400"
                                aria-label="Flex tier"
                              />
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                          <span>{model.type}</span>
                        </div>
                      </div>
                    </td>
                    <td className="px-2.5 py-2 font-mono text-[11px]">
                      {formatContextTokens(model.maxTokens)}
                    </td>
                    <td className="px-2.5 py-2">
                      {model.quantization ? (
                        <span className="inline-flex rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                          {model.quantization}
                        </span>
                      ) : (
                        <span className="text-muted-foreground/60">—</span>
                      )}
                    </td>
                    <td className="px-2.5 py-2 font-mono text-[11px] whitespace-nowrap text-muted-foreground">
                      {model.centsPerInputToken != null ? (
                        formatPricePerMillion(model.centsPerInputToken)
                      ) : (
                        <span className="text-muted-foreground/60">—</span>
                      )}
                    </td>
                    <td className="px-2.5 py-2 font-mono text-[11px] whitespace-nowrap text-foreground">
                      {model.centsPerOutputToken != null ? (
                        formatPricePerMillion(model.centsPerOutputToken)
                      ) : (
                        <span className="text-muted-foreground/60">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Selection Summary and Batch Actions */}
      <div className="flex flex-wrap items-center justify-between gap-2 py-1 text-xs">
        <div className="flex items-center gap-2.5">
          <span className="font-medium text-foreground">
            {selectedIds.length}{" "}
            {selectedIds.length === 1 ? "model selected" : "models selected"}
          </span>
          {sortedModels.length > 0 && (
            <button
              type="button"
              className="cursor-pointer border-0 bg-transparent p-0 text-xs text-blue-500 hover:underline"
              onClick={handleSelectAllFiltered}
              disabled={disabled}
            >
              Select all visible ({sortedModels.length})
            </button>
          )}
          {selectedIds.length > 0 && (
            <button
              type="button"
              className="cursor-pointer border-0 bg-transparent p-0 text-xs text-muted-foreground hover:text-foreground hover:underline"
              onClick={handleClearSelection}
              disabled={disabled}
            >
              Clear all
            </button>
          )}
        </div>

        {selectedIds.length > 0 && (
          <div className="flex max-h-16 flex-wrap gap-1.5 overflow-y-auto">
            {selectedIds.map((id) => (
              <span
                key={id}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] text-foreground"
              >
                <span>{id}</span>
                <XIcon
                  className="size-3 cursor-pointer opacity-60 hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeModel(id);
                  }}
                />
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Manual Input Field (Maintains full backward compatibility & custom models) */}
      <div className="flex flex-col gap-1.5 pt-1">
        <div className="text-xs text-muted-foreground">
          Enter model IDs separated by commas. Examples: {defaultModelIds}
        </div>
        <label
          htmlFor="deepinfra-model-ids-input"
          className="text-xs font-medium text-muted-foreground"
        >
          Selected Model IDs (comma-separated or type custom model):
        </label>
        <Input
          id="deepinfra-model-ids-input"
          value={value}
          aria-label="Model IDs"
          disabled={disabled}
          placeholder="deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1, meta-llama/Llama-3.3-70B-Instruct"
          className="rounded-md text-xs font-mono"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              event.stopPropagation();
              onEnterPress?.();
            }
          }}
        />
      </div>
    </div>
  );
}
