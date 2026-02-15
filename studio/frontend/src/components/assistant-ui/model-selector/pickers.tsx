import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { useDebouncedValue, useHfModelSearch, useInfiniteScroll } from "@/hooks";
import { cn, formatCompact } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState, type ReactNode } from "react";
import type {
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./types";

function dedupe(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function ListLabel({ children }: { children: ReactNode }) {
  return (
    <div className="px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

function ModelRow({
  label,
  meta,
  selected,
  onClick,
}: {
  label: string;
  meta?: string;
  selected?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent",
        selected && "bg-accent/60",
      )}
    >
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {meta ? (
        <span className="shrink-0 text-[10px] text-muted-foreground">{meta}</span>
      ) : null}
    </button>
  );
}

export function HubModelPicker({
  models,
  value,
  onSelect,
}: {
  models: ModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
}) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  const { results, isLoading, isLoadingMore, fetchMore } = useHfModelSearch(
    debouncedQuery,
  );

  const recommendedIds = useMemo(
    () => dedupe([...models.map((model) => model.id), value ?? ""]),
    [models, value],
  );

  const showHfSection = debouncedQuery.trim().length > 0;
  const recommendedSet = useMemo(() => new Set(recommendedIds), [recommendedIds]);

  const hfIds = useMemo(() => {
    if (!showHfSection) return [];
    return results
      .map((result) => result.id)
      .filter((id) => !recommendedSet.has(id));
  }, [recommendedSet, results, showHfSection]);

  const metricsById = useMemo(
    () =>
      new Map(
        results.map((result) => [
          result.id,
          result.totalParams
            ? formatCompact(result.totalParams)
            : `↓${formatCompact(result.downloads)}`,
        ]),
      ),
    [results],
  );

  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore, results.length);

  return (
    <div className="space-y-2">
      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search Hugging Face models"
          className="h-9 pl-8 pr-8"
        />
        {isLoading && (
          <Spinner className="pointer-events-none absolute right-2.5 top-2.5 size-4 text-muted-foreground" />
        )}
      </div>

      <div ref={scrollRef} className="max-h-64 overflow-y-auto">
        <div className="p-1">
          {!showHfSection ? (
            <>
              <ListLabel>Recommended</ListLabel>
              {recommendedIds.length === 0 ? (
                <div className="px-2.5 py-2 text-xs text-muted-foreground">
                  No default models.
                </div>
              ) : (
                recommendedIds.map((id) => (
                  <ModelRow
                    key={id}
                    label={id}
                    selected={value === id}
                    onClick={() => onSelect(id, { source: "hub", isLora: false })}
                  />
                ))
              )}
            </>
          ) : null}

          {showHfSection ? (
            <>
              <ListLabel>Hugging Face</ListLabel>
              {hfIds.length === 0 && !isLoading ? (
                <div className="px-2.5 py-2 text-xs text-muted-foreground">
                  No matching models.
                </div>
              ) : (
                hfIds.map((id) => (
                  <ModelRow
                    key={id}
                    label={id}
                    meta={metricsById.get(id)}
                    selected={value === id}
                    onClick={() => onSelect(id, { source: "hub", isLora: false })}
                  />
                ))
              )}
              <div ref={sentinelRef} className="h-px" />
              {isLoadingMore ? (
                <div className="flex items-center justify-center py-2">
                  <Spinner className="size-3.5 text-muted-foreground" />
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function LoraModelPicker({
  loraModels,
  value,
  onSelect,
}: {
  loraModels: LoraModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
}) {
  const [query, setQuery] = useState("");

  const normalized = useMemo(
    () =>
      loraModels
        .map((model) => ({
          ...model,
          baseModel: model.baseModel || model.description || "Unknown base model",
        }))
        .sort((a, b) => {
          const aTime = a.updatedAt ?? -1;
          const bTime = b.updatedAt ?? -1;
          if (aTime !== bTime) return bTime - aTime;
          const baseCmp = a.baseModel.localeCompare(b.baseModel);
          if (baseCmp !== 0) return baseCmp;
          return a.name.localeCompare(b.name);
        }),
    [loraModels],
  );

  const grouped = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const out = new Map<string, LoraModelOption[]>();

    for (const model of normalized) {
      const searchText = `${model.name} ${model.baseModel} ${model.id}`.toLowerCase();
      if (needle && !searchText.includes(needle)) continue;

      const key = model.baseModel || "Unknown base model";
      const prev = out.get(key) ?? [];
      prev.push(model);
      out.set(key, prev);
    }

    return [...out.entries()].sort((a, b) => {
      const aLatest = Math.max(...a[1].map((model) => model.updatedAt ?? -1));
      const bLatest = Math.max(...b[1].map((model) => model.updatedAt ?? -1));
      if (aLatest !== bLatest) return bLatest - aLatest;
      return a[0].localeCompare(b[0]);
    });
  }, [normalized, query]);

  return (
    <div className="space-y-2">
      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search local adapters"
          className="h-9 pl-8"
        />
      </div>

      <div className="max-h-64 overflow-y-auto">
        <div className="p-1">
          {grouped.length === 0 ? (
            <div className="px-2.5 py-2 text-xs text-muted-foreground">
              No adapters found.
            </div>
          ) : (
            grouped.map(([baseModel, adapters], index) => (
              <div key={baseModel}>
                {index > 0 ? <div className="my-1" /> : null}
                <ListLabel>{baseModel}</ListLabel>
                {adapters.map((adapter) => (
                  <ModelRow
                    key={adapter.id}
                    label={adapter.name}
                    meta="LoRA"
                    selected={value === adapter.id}
                    onClick={() => onSelect(adapter.id, { source: "lora", isLora: true })}
                  />
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

