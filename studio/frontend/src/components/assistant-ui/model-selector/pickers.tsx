// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { deleteCachedModel, listCachedGguf, listCachedModels, listGgufVariants } from "@/features/chat/api/chat-api";
import type { CachedGgufRepo, CachedModelRepo } from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import { usePlatformStore } from "@/config/env";
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
import { Trash2Icon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
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

/** Format bytes to a human-readable size string. */
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / 1024 ** i;
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
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
        "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent",
        selected && "bg-accent/60",
        exceeds && "opacity-50",
      )}
    >
      <span
        className={cn(
          "block min-w-0 flex-1 truncate",
          exceeds && "line-through decoration-muted-foreground/50",
        )}
      >
        {label}
      </span>
      <span className="ml-auto flex items-center gap-1.5 shrink-0">
        {vramStatus === "exceeds" && (
          <span className="text-[9px] font-medium text-red-400">OOM</span>
        )}
        {vramStatus === "tight" && (
          <span className="text-[9px] font-medium text-amber-400">TIGHT</span>
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

// ── GGUF Variant Expander ────────────────────────────────────

function GgufVariantExpander({
  repoId,
  onSelect,
  gpuGb,
  systemRamGb,
  onDeleteVariant,
}: {
  repoId: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  gpuGb?: number;
  systemRamGb?: number;
  onDeleteVariant?: (quant: string) => void;
}) {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [defaultVariant, setDefaultVariant] = useState<string | null>(null);
  const [hasVision, setHasVision] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;
    setLoading(true);
    setError(null);

    listGgufVariants(repoId)
      .then((res) => {
        if (canceled) return;
        setVariants(res.variants);
        setDefaultVariant(res.default_variant);
        setHasVision(res.has_vision);
      })
      .catch((err) => {
        if (canceled) return;
        setError(err instanceof Error ? err.message : "Failed to load variants");
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });

    return () => {
      canceled = true;
    };
  }, [repoId]);

  const handleVariantClick = useCallback(
    (quant: string, downloaded?: boolean, sizeBytes?: number) => {
      onSelect(repoId, {
        source: "hub",
        isLora: false,
        ggufVariant: quant,
        isDownloaded: downloaded,
        expectedBytes: sizeBytes,
      });
    },
    [repoId, onSelect],
  );

  // GGUF fit classification matching llama-server's _select_gpus logic:
  //   fits  = model <= 0.7 * total GPU memory
  //   tight = model > 0.7 * GPU but <= 0.7 * GPU + 0.7 * system RAM (--fit uses CPU offload)
  //   oom   = model > 0.7 * GPU + 0.7 * system RAM
  const gpuBudgetGb = (gpuGb ?? 0) * 0.70;
  const totalBudgetGb = gpuBudgetGb + (systemRamGb ?? 0) * 0.70;

  const getGgufFit = useCallback(
    (sizeBytes: number): "fits" | "tight" | "oom" => {
      if (!gpuGb || gpuGb <= 0) return "fits";
      const gb = sizeBytes / (1024 ** 3);
      if (gb <= 0 || gb <= gpuBudgetGb) return "fits";
      if (gb <= totalBudgetGb) return "tight";
      return "oom";
    },
    [gpuGb, gpuBudgetGb, totalBudgetGb],
  );

  // If the backend-recommended variant is OOM, pick the largest fitting
  // variant instead; if all are OOM, recommend the smallest one.
  const effectiveRecommended = useMemo(() => {
    if (!variants || !gpuGb || gpuGb <= 0) return defaultVariant;
    const defaultV = variants.find((v) => v.quant === defaultVariant);
    if (defaultV && getGgufFit(defaultV.size_bytes) !== "oom") return defaultVariant;
    // Default is OOM -- pick largest non-OOM variant (best quality that fits)
    const fitting = variants.filter((v) => getGgufFit(v.size_bytes) !== "oom");
    if (fitting.length > 0) {
      fitting.sort((a, b) => b.size_bytes - a.size_bytes);
      return fitting[0].quant;
    }
    // All OOM -- recommend smallest (most likely to partially run)
    const sorted = [...variants].sort((a, b) => a.size_bytes - b.size_bytes);
    return sorted[0].quant;
  }, [variants, defaultVariant, gpuGb, getGgufFit]);

  const sortedVariants = useMemo(() => {
    if (!variants) return variants;
    // Tier: 0 = downloaded+fits, 1 = downloaded+tight, 2 = fits, 3 = tight, 4 = OOM
    const tierOf = (v: GgufVariantDetail) => {
      const f = getGgufFit(v.size_bytes);
      if (f === "oom") return 4;
      const base = f === "fits" ? 0 : 1;
      return v.downloaded ? base : base + 2;
    };
    return [...variants].sort((a, b) => {
      const aTier = tierOf(a);
      const bTier = tierOf(b);
      if (aTier !== bTier) return aTier - bTier;

      // Within the same tier, recommended goes first
      const aIsRec = a.quant === effectiveRecommended;
      const bIsRec = b.quant === effectiveRecommended;
      if (aIsRec !== bIsRec) return aIsRec ? -1 : 1;

      // fits: largest first (best quality that fits in GPU)
      // tight/OOM: smallest first (closest to fitting, fastest to run)
      const fitsInGpu = aTier === 0 || aTier === 2;
      return fitsInGpu ? b.size_bytes - a.size_bytes : a.size_bytes - b.size_bytes;
    });
  }, [variants, effectiveRecommended, getGgufFit]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-5 py-2">
        <Spinner className="size-3 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Loading variants…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-5 py-2 text-xs text-destructive">{error}</div>
    );
  }

  if (!sortedVariants || sortedVariants.length === 0) {
    return (
      <div className="px-5 py-2 text-xs text-muted-foreground">
        No GGUF variants found.
      </div>
    );
  }

  return (
    <div className="pl-4 border-l-2 border-accent/50 ml-3 my-1">
      <div className="px-2 py-1 flex items-center gap-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Quantizations
        </span>
        {hasVision && (
          <span className="text-[9px] font-medium text-blue-400">Vision</span>
        )}
      </div>
      {sortedVariants.map((v) => {
        const fit = getGgufFit(v.size_bytes);
        const oom = fit === "oom";
        const tight = fit === "tight";
        return (
          <div key={v.filename} className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => handleVariantClick(v.quant, v.downloaded, v.size_bytes)}
              className={cn(
                "flex min-w-0 flex-1 items-center justify-between gap-2 rounded-md px-2.5 py-1 text-left text-sm transition-colors hover:bg-accent",
              )}
            >
              <span className="min-w-0 flex-1 truncate font-mono text-xs">
                {v.quant}
                {v.downloaded ? (
                  <span className="ml-1.5 text-[9px] font-sans font-medium text-green-400">
                    downloaded
                  </span>
                ) : v.quant === effectiveRecommended ? (
                  <span className="ml-1.5 text-[9px] font-sans font-medium text-primary/70">
                    recommended
                  </span>
                ) : null}
              </span>
              <span className="flex items-center gap-1.5 shrink-0">
                {oom && (
                  <span className="text-[9px] font-medium text-red-400">OOM</span>
                )}
                {tight && (
                  <span className="text-[9px] font-medium text-amber-400">TIGHT</span>
                )}
                <span className="text-[10px] text-muted-foreground">
                  {formatBytes(v.size_bytes)}
                </span>
              </span>
            </button>
            {v.downloaded && onDeleteVariant && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onDeleteVariant(v.quant); }}
                className="shrink-0 rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-destructive/10 hover:text-destructive"
              >
                <Trash2Icon className="size-3" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Detect GGUF repos by naming convention ────────────────────

function isGgufRepo(id: string): boolean {
  return id.toUpperCase().includes("-GGUF");
}

/** Extract param count label from model name (e.g. "Qwen3-0.6B" -> "0.6B"). */
function extractParamLabel(id: string): string | undefined {
  // Match patterns like "0.6B", "1B", "4B", "3.5B", "70B", "1.5B" etc.
  const name = id.split("/").pop() ?? id;
  const match = name.match(/(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/);
  return match ? `${match[1]}B` : undefined;
}

// Module-level caches so re-mounting the popover shows results instantly
let _cachedGgufCache: CachedGgufRepo[] = [];
let _cachedModelsCache: CachedModelRepo[] = [];

// ── Hub Model Picker ──────────────────────────────────────────

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

  // Track which GGUF repo is expanded for variant selection
  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);

  // Delete confirmation dialog state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Cached (already downloaded) repos -- use module-level cache so
  // re-mounting the popover does not flash an empty "Downloaded" section.
  const [cachedGguf, setCachedGguf] = useState<CachedGgufRepo[]>(_cachedGgufCache);
  const [cachedModels, setCachedModels] = useState<CachedModelRepo[]>(_cachedModelsCache);
  const alreadyCached = _cachedGgufCache.length > 0 || _cachedModelsCache.length > 0;
  const [cachedReady, setCachedReady] = useState(alreadyCached);

  const refreshCachedLists = useCallback(() => {
    listCachedGguf().then((v) => { _cachedGgufCache = v; setCachedGguf(v); }).catch(() => {});
    listCachedModels().then((v) => { _cachedModelsCache = v; setCachedModels(v); }).catch(() => {});
  }, []);

  useEffect(() => {
    if (alreadyCached) return;
    let done = 0;
    const check = () => { if (++done >= 2) setCachedReady(true); };
    listCachedGguf().then((v) => { _cachedGgufCache = v; setCachedGguf(v); }).catch(() => {}).finally(check);
    listCachedModels().then((v) => { _cachedModelsCache = v; setCachedModels(v); }).catch(() => {}).finally(check);
  }, [alreadyCached]);

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      // deleteTarget is "repo_id" or "repo_id::variant"
      const sepIdx = deleteTarget.indexOf("::");
      const repoId = sepIdx >= 0 ? deleteTarget.slice(0, sepIdx) : deleteTarget;
      const variant = sepIdx >= 0 ? deleteTarget.slice(sepIdx + 2) : undefined;
      await deleteCachedModel(repoId, variant);
      toast.success(`Deleted ${variant ? `${repoId} ${variant}` : repoId}`);
      refreshCachedLists();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete model");
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }, [deleteTarget, refreshCachedLists]);

  // Deduplicate: don't show downloaded models in the recommended list.
  // Compare case-insensitively since HF cache lowercases repo IDs.
  const downloadedSet = useMemo(() => {
    const s = new Set<string>();
    for (const c of cachedGguf) s.add(c.repo_id.toLowerCase());
    for (const c of cachedModels) s.add(c.repo_id.toLowerCase());
    return s;
  }, [cachedGguf, cachedModels]);

  const chatOnly = usePlatformStore((s) => s.isChatOnly());

  const recommendedIds = useMemo(() => {
    const all = dedupe([...models.map((model) => model.id), value ?? ""])
      .filter((id) => !downloadedSet.has(id.toLowerCase()))
      .filter((id) => !chatOnly || isGgufRepo(id));
    // Sort: GGUFs first, then hub models
    const gguf: string[] = [];
    const hub: string[] = [];
    for (const id of all) {
      if (isGgufRepo(id)) gguf.push(id);
      else hub.push(id);
    }
    return [...gguf, ...hub];
  }, [models, value, downloadedSet, chatOnly]);

  // Infinite scroll paging for the recommended section
  const [recommendedPage, setRecommendedPage] = useState(1);
  // Reset page when the underlying list changes
  useEffect(() => { setRecommendedPage(1); }, [models, chatOnly]);

  const visibleRecommendedIds = useMemo(() => {
    const hubStartIndex = recommendedIds.findIndex((id) => !isGgufRepo(id));
    const allGguf = hubStartIndex === -1 ? recommendedIds : recommendedIds.slice(0, hubStartIndex);
    const allHub = hubStartIndex === -1 ? [] : recommendedIds.slice(hubStartIndex);
    // Interleave in chunks of 4: [4 gguf, 4 hub, 4 gguf, 4 hub, ...]
    const result: string[] = [];
    for (let p = 0; p < recommendedPage; p++) {
      result.push(...allGguf.slice(p * 4, (p + 1) * 4));
      result.push(...allHub.slice(p * 4, (p + 1) * 4));
    }
    return result;
  }, [recommendedIds, recommendedPage]);

  const hasMoreRecommended = visibleRecommendedIds.length < recommendedIds.length;

  // Fetch VRAM info for the full pool once (recommendedIds is stable across
  // page increments) so we don't re-fetch on every scroll.
  const { paramCountById: recommendedParamCountById } =
    useRecommendedModelVram(recommendedIds);

  const showHfSection = debouncedQuery.trim().length > 0;
  const recommendedSet = useMemo(() => new Set(visibleRecommendedIds), [visibleRecommendedIds]);

  const hfIds = useMemo(() => {
    if (!showHfSection) return [];
    return results
      .map((result) => result.id)
      .filter((id) => !recommendedSet.has(id))
      .filter((id) => !chatOnly || isGgufRepo(id));
  }, [recommendedSet, results, showHfSection, chatOnly]);

  const metricsById = useMemo(
    () =>
      new Map(
        results
          .filter((result) => result.totalParams || result.estimatedSizeBytes)
          .map((result) => [
            result.id,
            result.estimatedSizeBytes
              ? `~${formatBytes(result.estimatedSizeBytes)}`
              : formatCompact(result.totalParams!),
          ]),
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
    for (const id of visibleRecommendedIds) {
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
  }, [visibleRecommendedIds, recommendedParamCountById, gpu]);

  const { scrollRef, sentinelRef } = useInfiniteScroll(fetchMore, results.length);

  // Sentinel + IntersectionObserver for recommended infinite scroll.
  // We disconnect after each fire so the observer doesn't loop while
  // React re-renders; the effect re-creates it on the next page.
  // Uses a callback ref for the sentinel so we detect mount/unmount reliably.
  const [recommendedSentinel, setRecommendedSentinel] = useState<HTMLDivElement | null>(null);
  const recommendedSentinelRef = useCallback((node: HTMLDivElement | null) => {
    setRecommendedSentinel(node);
  }, []);
  useEffect(() => {
    if (!recommendedSentinel || !hasMoreRecommended) return;
    const root = scrollRef.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          obs.disconnect();
          setRecommendedPage((p) => p + 1);
        }
      },
      { threshold: 0, root },
    );
    // Small delay so the browser finishes layout after the previous page render
    const timer = setTimeout(() => obs.observe(recommendedSentinel), 100);
    return () => { clearTimeout(timer); obs.disconnect(); };
  }, [recommendedSentinel, hasMoreRecommended, recommendedPage, scrollRef]);

  /** Handle clicking a model row — GGUF repos expand, others load directly. */
  const handleModelClick = useCallback(
    (id: string) => {
      if (isGgufRepo(id)) {
        // Toggle GGUF variant expander
        setExpandedGguf((prev) => (prev === id ? null : id));
      } else {
        onSelect(id, { source: "hub", isLora: false });
      }
    },
    [onSelect],
  );

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
          {!cachedReady && !showHfSection ? (
            <div className="flex items-center gap-2 px-5 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Loading models…</span>
            </div>
          ) : !showHfSection && (cachedGguf.length > 0 || (!chatOnly && cachedModels.length > 0)) ? (
            <>
              <ListLabel>{"\uD83E\uDDA5"} Downloaded</ListLabel>
              {cachedGguf.map((c) => (
                <div key={c.repo_id}>
                  <ModelRow
                    label={c.repo_id}
                    meta={`GGUF · ${formatBytes(c.size_bytes)}`}
                    selected={value === c.repo_id}
                    onClick={() => handleModelClick(c.repo_id)}
                    vramStatus={null}
                  />
                  {expandedGguf === c.repo_id && (
                    <GgufVariantExpander
                      repoId={c.repo_id}
                      onSelect={onSelect}
                      gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                      systemRamGb={gpu.available ? gpu.systemRamAvailableGb : undefined}
                      onDeleteVariant={(quant) => setDeleteTarget(`${c.repo_id}::${quant}`)}
                    />
                  )}
                </div>
              ))}
              {!chatOnly && cachedModels.map((c) => (
                <div key={c.repo_id} className="flex items-center gap-0.5">
                  <div className="min-w-0 flex-1">
                    <ModelRow
                      label={c.repo_id}
                      meta={formatBytes(c.size_bytes)}
                      selected={value === c.repo_id}
                      onClick={() => onSelect(c.repo_id, { source: "hub", isLora: false, isDownloaded: true })}
                      vramStatus={null}
                    />
                  </div>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); setDeleteTarget(c.repo_id); }}
                    className="shrink-0 rounded-md p-1.5 text-muted-foreground/60 transition-colors hover:bg-destructive/10 hover:text-destructive"
                  >
                    <Trash2Icon className="size-3.5" />
                  </button>
                </div>
              ))}
            </>
          ) : null}

          {!showHfSection && cachedReady ? (
            <>
              <ListLabel>{"\uD83E\uDDA5"} Recommended</ListLabel>
              {visibleRecommendedIds.length === 0 ? (
                <div className="px-2.5 py-2 text-xs text-muted-foreground">
                  No default models.
                </div>
              ) : (
                visibleRecommendedIds.map((id) => {
                  const vram = recommendedVramMap.get(id);
                  return (
                    <div key={id}>
                      <ModelRow
                        label={id}
                        meta={
                          isGgufRepo(id)
                            ? "GGUF"
                            : vram?.detail ?? extractParamLabel(id)
                        }
                        selected={value === id}
                        onClick={() => handleModelClick(id)}
                        vramStatus={isGgufRepo(id) ? null : vram?.status ?? null}
                        vramEst={isGgufRepo(id) ? undefined : vram?.est}
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                      />
                      {expandedGguf === id && (
                        <GgufVariantExpander repoId={id} onSelect={onSelect} gpuGb={gpu.available ? gpu.memoryTotalGb : undefined} systemRamGb={gpu.available ? gpu.systemRamAvailableGb : undefined} />
                      )}
                    </div>
                  );
                })
              )}
              {hasMoreRecommended && (
                <>
                  <div ref={recommendedSentinelRef} className="h-px" />
                  <div className="flex items-center justify-center py-2">
                    <Spinner className="size-3.5 text-muted-foreground" />
                  </div>
                </>
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
                    <div key={id}>
                      <ModelRow
                        label={id}
                        meta={
                          isGgufRepo(id)
                            ? "GGUF"
                            : metricsById.get(id) ?? extractParamLabel(id)
                        }
                        selected={value === id}
                        onClick={() => handleModelClick(id)}
                        vramStatus={isGgufRepo(id) ? null : vram?.status ?? null}
                        vramEst={isGgufRepo(id) ? undefined : vram?.est}
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                      />
                      {expandedGguf === id && (
                        <GgufVariantExpander repoId={id} onSelect={onSelect} gpuGb={gpu.available ? gpu.memoryTotalGb : undefined} systemRamGb={gpu.available ? gpu.systemRamAvailableGb : undefined} />
                      )}
                    </div>
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

      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open && !deleting) setDeleteTarget(null); }}>
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete cached model?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove <span className="font-medium text-foreground">{deleteTarget?.includes("::") ? `${deleteTarget.split("::")[0]} (${deleteTarget.split("::")[1]})` : deleteTarget}</span> from disk. You can re-download it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>No</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={(e) => { e.preventDefault(); handleDeleteConfirm(); }}
            >
              {deleting ? "Deleting..." : "Yes"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
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
                  const isGguf = adapter.exportType === "gguf";
                  const tag = isGguf
                    ? "GGUF"
                    : isExported
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
                        isLora: !isMerged && !isGguf,
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
