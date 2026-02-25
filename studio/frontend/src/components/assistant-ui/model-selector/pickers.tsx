import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useDebouncedValue,
  useGpuInfo,
  useHfModelSearch,
  useInfiniteScroll,
  useRecommendedModelVram,
} from "@/hooks";
import { cn, formatCompact } from "@/lib/utils";
import type { VramFitStatus } from "@/lib/vram";
import { checkVramFit, estimateLoadingVram } from "@/lib/vram";
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
  vramStatus,
  vramEst,
  gpuGb,
  tooltipText,
}: {
  label: string;
  meta?: string;
  selected?: boolean;
  onClick: () => void;
  vramStatus?: VramFitStatus | null;
  vramEst?: number;
  gpuGb?: number;
  tooltipText?: ReactNode;
}) {
  const exceeds = vramStatus === "exceeds";
  const showVramTooltip =
    vramEst != null && vramEst > 0 && gpuGb != null && gpuGb > 0;
  const vramTooltipText =
    showVramTooltip && vramStatus
      ? exceeds
        ? `Needs ~${vramEst}GB VRAM (GPU: ${gpuGb}GB)`
        : vramStatus === "tight"
          ? `~${vramEst}GB VRAM (tight fit on ${gpuGb}GB)`
          : `~${vramEst}GB VRAM`
      : null;

  const content = (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent",
        selected && "bg-accent/60",
        exceeds && "opacity-50",
      )}
    >
      <span
        className={cn(
          "min-w-0 flex-1 truncate",
          exceeds && "line-through decoration-muted-foreground/50",
        )}
      >
        {label}
      </span>
      <span className="flex items-center gap-1.5 shrink-0">
        {vramStatus === "exceeds" && (
          <span className="text-[9px] font-medium text-red-400">OOM</span>
        )}
        {vramStatus === "tight" && (
          <span className="text-[9px] font-medium text-amber-400">TIGHT</span>
        )}
        {vramStatus === "fits" && (
          <span className="text-[9px] font-medium text-emerald-500/90">FIT</span>
        )}
        {meta ? (
          <span className="text-[10px] text-muted-foreground">{meta}</span>
        ) : null}
      </span>
    </button>
  );

  if (vramTooltipText) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{content}</TooltipTrigger>
        <TooltipContent side="left" className="max-w-xs break-all">
          {label}
          <span className="block text-[10px] mt-1">{vramTooltipText}</span>
        </TooltipContent>
      </Tooltip>
    );
  }

  if (tooltipText) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{content}</TooltipTrigger>
        <TooltipContent side="left" className="max-w-xs break-all">
          {tooltipText}
        </TooltipContent>
      </Tooltip>
    );
  }
  return content;
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
  const gpu = useGpuInfo();
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  const { results, isLoading, isLoadingMore, fetchMore } = useHfModelSearch(
    debouncedQuery,
  );

  const recommendedIds = useMemo(
    () => dedupe([...models.map((model) => model.id), value ?? ""]),
    [models, value],
  );

  const { paramCountById: recommendedParamCountById } =
    useRecommendedModelVram(recommendedIds);

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
        results
          .filter((result) => result.totalParams)
          .map((result) => [result.id, formatCompact(result.totalParams!)]),
      ),
    [results],
  );

  const vramMap = useMemo(() => {
    const map = new Map<
      string,
      { est: number; status: VramFitStatus | null; detail: string | null }
    >();
    for (const r of results) {
      const detail = r.totalParams ? formatCompact(r.totalParams) : null;
      if (r.totalParams) {
        const est = estimateLoadingVram(r.totalParams, "qlora");
        const status = gpu.available
          ? checkVramFit(est, gpu.memoryTotalGb)
          : null;
        map.set(r.id, { est, status, detail });
      } else {
        map.set(r.id, { est: 0, status: null, detail });
      }
    }
    return map;
  }, [results, gpu]);

  const recommendedVramMap = useMemo(() => {
    const map = new Map<
      string,
      { est: number; status: VramFitStatus | null; detail: string | null }
    >();
    for (const id of recommendedIds) {
      const totalParams = recommendedParamCountById.get(id);
      if (totalParams) {
        const est = estimateLoadingVram(totalParams, "qlora");
        const status = gpu.available
          ? checkVramFit(est, gpu.memoryTotalGb)
          : null;
        const detail = formatCompact(totalParams);
        map.set(id, { est, status, detail });
      }
    }
    return map;
  }, [recommendedIds, recommendedParamCountById, gpu]);

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
                recommendedIds.map((id) => {
                  const vram = recommendedVramMap.get(id);
                  return (
                    <ModelRow
                      key={id}
                      label={id}
                      meta={vram?.detail ?? undefined}
                      selected={value === id}
                      onClick={() =>
                        onSelect(id, { source: "hub", isLora: false })
                      }
                      vramStatus={vram?.status ?? null}
                      vramEst={vram?.est}
                      gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                    />
                  );
                })
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
                hfIds.map((id) => {
                  const vram = vramMap.get(id);
                  return (
                    <ModelRow
                      key={id}
                      label={id}
                      meta={metricsById.get(id)}
                      selected={value === id}
                      onClick={() =>
                        onSelect(id, { source: "hub", isLora: false })
                      }
                      vramStatus={vram?.status ?? null}
                      vramEst={vram?.est}
                      gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                    />
                  );
                })
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
                {adapters.map((adapter) => {
                  const isExported = adapter.source === "exported";
                  const isMerged = adapter.exportType === "merged";
                  const tag = isExported
                    ? isMerged ? "Merged" : "LoRA"
                    : "LoRA";
                  const meta = isExported ? `${tag} · Exported` : tag;
                  return (
                    <ModelRow
                      key={adapter.id}
                      label={adapter.name}
                      meta={meta}
                      selected={value === adapter.id}
                      onClick={() => onSelect(adapter.id, {
                        source: isExported ? "exported" : "lora",
                        isLora: !isMerged,
                      })}
                      tooltipText={
                        <>
                          <span className="block break-words">{adapter.name}</span>
                          <span className="block mt-1 text-[10px] text-muted-foreground break-all">
                            {adapter.id}
                          </span>
                        </>
                      }
                    />
                  );
                })}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

