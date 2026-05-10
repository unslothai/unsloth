// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  deleteCachedModel,
  deleteFineTunedModel,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  listLocalModels,
} from "@/features/chat/api/chat-api";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  LocalModelInfo,
} from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import { useGpuInfo } from "@/hooks";
import { cn } from "@/lib/utils";
import type { VramFitStatus } from "@/lib/vram";
import {
  ArrowRight01Icon,
  CubeIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { ModelDeleteAction } from "./model-delete-action";
import { ChevronDownIcon, ChevronRightIcon, DownloadIcon } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type {
  DeletedModelRef,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./types";

/** Normalize a string for fuzzy search: lowercase, strip separators. */
function normalizeForSearch(s: string): string {
  return s.toLowerCase().replace(/[\s\-_\.]/g, "");
}

function ListLabel({
  children,
  icon,
  collapsed,
  onToggle,
}: {
  children: ReactNode;
  icon?: ReactNode;
  collapsed?: boolean;
  onToggle?: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-1 px-2.5 py-1.5">
      <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {icon}
        {children}
      </span>
      {onToggle && (
        <button
          type="button"
          onClick={onToggle}
          aria-label={collapsed ? "Expand section" : "Collapse section"}
          className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
        >
          {collapsed
            ? <ChevronRightIcon className="size-3" />
            : <ChevronDownIcon className="size-3" />}
        </button>
      )}
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
        "flex w-full items-center gap-2 rounded-[6px] px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035]",
        selected && "bg-[#ececec] dark:bg-[#2e3035]",
      )}
    >
      <span
        className={cn(
          "block min-w-0 flex-1 truncate",
          exceeds && "!text-gray-500 dark:!text-gray-400",
        )}
      >
        {label}
      </span>
      <span className="ml-auto flex items-center gap-1.5 shrink-0">
        {vramStatus === "exceeds" && (
          <span className="text-[9px] font-medium !text-red-700 !bg-red-50 dark:!text-red-400 dark:!bg-red-950 px-1.5 py-0.5 rounded">OOM</span>
        )}
        {vramStatus === "tight" && (
          <span className="text-[9px] font-medium !text-amber-400">TIGHT</span>
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
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
        <TooltipContent
          side="left"
          className="tooltip-compact max-w-xs break-all"
        >
          {label}
          <span className="block text-[10px] mt-1">{vramTooltipText}</span>
        </TooltipContent>
      </Tooltip>
    );
  }

  if (tooltipText) {
    return (
      <Tooltip>
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
        <TooltipContent
          side="left"
          className="tooltip-compact max-w-xs break-all"
        >
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
  sourceOverride,
  deleteVariantTitle = "Delete cached model?",
  renderDeleteVariantDescription,
  getDeleteVariantSuccessMessage,
  deleteDisabled = false,
}: {
  repoId: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  gpuGb?: number;
  systemRamGb?: number;
  onDeleteVariant?: (quant: string) => Promise<void> | void;
  sourceOverride?: ModelSelectorChangeMeta["source"];
  deleteVariantTitle?: string;
  renderDeleteVariantDescription?: (quant: string) => ReactNode;
  getDeleteVariantSuccessMessage?: (quant: string) => string;
  deleteDisabled?: boolean;
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
        setError(
          err instanceof Error ? err.message : "Failed to load variants",
        );
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });

    return () => {
      canceled = true;
    };
  }, [repoId]);

  // Covers Unix absolute (/), Windows drive (C:\, D:/), UNC (\\server), relative (./, ../), tilde (~/)
  const isLocalPath = /^(\/|\.{1,2}[\\\/]|~[\\\/]|[A-Za-z]:[\\\/]|\\\\)/.test(
    repoId,
  );

  const handleVariantClick = useCallback(
    (quant: string, downloaded?: boolean, sizeBytes?: number) => {
      onSelect(repoId, {
        source: sourceOverride ?? (isLocalPath ? "local" : "hub"),
        isLora: false,
        ggufVariant: quant,
        isDownloaded: isLocalPath ? true : downloaded,
        expectedBytes: sizeBytes,
      });
    },
    [repoId, isLocalPath, onSelect, sourceOverride],
  );

  // GGUF fit classification matching llama-server's _select_gpus logic:
  //   fits  = model <= 0.7 * total GPU memory
  //   tight = model > 0.7 * GPU but <= 0.7 * GPU + 0.7 * system RAM (--fit uses CPU offload)
  //   oom   = model > 0.7 * GPU + 0.7 * system RAM
  const gpuBudgetGb = (gpuGb ?? 0) * 0.7;
  const totalBudgetGb = gpuBudgetGb + (systemRamGb ?? 0) * 0.7;

  const getGgufFit = useCallback(
    (sizeBytes: number): "fits" | "tight" | "oom" => {
      if (!gpuGb || gpuGb <= 0) return "fits";
      const gb = sizeBytes / 1024 ** 3;
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
    if (defaultV && getGgufFit(defaultV.size_bytes) !== "oom")
      return defaultVariant;
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
      return fitsInGpu
        ? b.size_bytes - a.size_bytes
        : a.size_bytes - b.size_bytes;
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
    return <div className="px-5 py-2 text-xs text-destructive">{error}</div>;
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
              onClick={() =>
                handleVariantClick(v.quant, v.downloaded, v.size_bytes)
              }
              className={cn(
                "flex min-w-0 flex-1 items-center justify-between gap-2 rounded-[6px] px-2.5 py-1 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035]",
              )}
            >
              <span className="min-w-0 flex-1 truncate font-mono text-xs">
                <span className={cn(oom && "!text-gray-500 dark:!text-gray-400")}>{v.quant}</span>
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
                  <span className="text-[9px] font-medium !text-red-700 !bg-red-50 dark:!text-red-400 dark:!bg-red-950 px-1.5 py-0.5 rounded">
                    OOM
                  </span>
                )}
                {tight && (
                  <span className="text-[9px] font-medium !text-amber-400">
                    TIGHT
                  </span>
                )}
                <span className="text-[10px] text-muted-foreground">
                  {formatBytes(v.size_bytes)}
                </span>
              </span>
            </button>
            {v.downloaded && onDeleteVariant && (
              <ModelDeleteAction
                ariaLabel={`Delete ${repoId} ${v.quant}`}
                title={deleteVariantTitle}
                description={
                  renderDeleteVariantDescription?.(v.quant) ?? (
                    <>
                      This will remove{" "}
                      <span className="font-medium text-foreground">
                        {repoId} ({v.quant})
                      </span>{" "}
                      from disk. You can re-download it later.
                    </>
                  )
                }
                successMessage={
                  getDeleteVariantSuccessMessage?.(v.quant) ??
                  `Deleted ${repoId} ${v.quant}`
                }
                buttonClassName="p-1"
                iconClassName="size-3"
                disabled={deleteDisabled}
                onConfirm={() => onDeleteVariant(v.quant)}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Detect GGUF repos by naming convention or hub tag ────────────────────

function hasGgufSuffix(id: string): boolean {
  return /-GGUF(?:$|-)/i.test(id);
}

function isGgufRepo(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || hasGgufSuffix(id);
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
let _lmStudioCache: LocalModelInfo[] = [];

/** Sort external (lmstudio-source) models with unsloth publisher first. */
function sortLmStudio(models: LocalModelInfo[]): LocalModelInfo[] {
  return [...models].sort((a, b) => {
    const aUnsloth = (a.model_id ?? "").startsWith("unsloth/") ? 0 : 1;
    const bUnsloth = (b.model_id ?? "").startsWith("unsloth/") ? 0 : 1;
    if (aUnsloth !== bUnsloth) return aUnsloth - bUnsloth;
    return (a.model_id ?? a.display_name).localeCompare(
      b.model_id ?? b.display_name,
    );
  });
}

// ── Hub Model Picker ──────────────────────────────────────────

export function HubModelPicker({
  value,
  onSelect,
}: {
  models: ModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onFoldersChange?: () => void;
}) {
  const gpu = useGpuInfo();
  const navigate = useNavigate();
  const chatOnly = usePlatformStore((s) => s.isChatOnly());

  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  const [downloadedCollapsed, setDownloadedCollapsed] = useState(false);
  const [lmStudioCollapsed, setLmStudioCollapsed] = useState(false);
  const [filter, setFilter] = useState("");

  const [cachedGguf, setCachedGguf] =
    useState<CachedGgufRepo[]>(_cachedGgufCache);
  const [cachedModels, setCachedModels] =
    useState<CachedModelRepo[]>(_cachedModelsCache);
  const alreadyCached =
    _cachedGgufCache.length > 0 || _cachedModelsCache.length > 0;
  const [cachedReady, setCachedReady] = useState(alreadyCached);
  const [lmStudioModels, setLmStudioModels] =
    useState<LocalModelInfo[]>(_lmStudioCache);

  const refreshLocalModelsList = useCallback(() => {
    listLocalModels()
      .then((res) => {
        const lm = sortLmStudio(
          res.models.filter((m) => m.source === "lmstudio"),
        );
        _lmStudioCache = lm;
        setLmStudioModels(lm);
      })
      .catch(() => {});
  }, []);

  const refreshCachedLists = useCallback(() => {
    listCachedGguf()
      .then((v) => {
        _cachedGgufCache = v;
        setCachedGguf(v);
      })
      .catch(() => {});
    listCachedModels()
      .then((v) => {
        _cachedModelsCache = v;
        setCachedModels(v);
      })
      .catch(() => {});
    refreshLocalModelsList();
  }, [refreshLocalModelsList]);

  useEffect(() => {
    refreshLocalModelsList();
    let done = 0;
    const check = () => {
      if (++done >= 2) setCachedReady(true);
    };
    listCachedGguf()
      .then((v) => {
        _cachedGgufCache = v;
        setCachedGguf(v);
      })
      .catch(() => {})
      .finally(check);
    listCachedModels()
      .then((v) => {
        _cachedModelsCache = v;
        setCachedModels(v);
      })
      .catch(() => {})
      .finally(check);
  }, [refreshLocalModelsList]);

  const filterNeedle = normalizeForSearch(filter.trim());
  const matchesFilter = useCallback(
    (id: string) =>
      filterNeedle.length === 0 ||
      normalizeForSearch(id).includes(filterNeedle),
    [filterNeedle],
  );
  const filteredGguf = useMemo(
    () => cachedGguf.filter((c) => matchesFilter(c.repo_id)),
    [cachedGguf, matchesFilter],
  );
  const filteredModels = useMemo(
    () => cachedModels.filter((c) => matchesFilter(c.repo_id)),
    [cachedModels, matchesFilter],
  );
  const filteredLmStudio = useMemo(
    () =>
      lmStudioModels.filter((m) =>
        matchesFilter(m.model_id ?? m.display_name),
      ),
    [lmStudioModels, matchesFilter],
  );

  const hasFilteredDownloaded =
    filteredGguf.length > 0 || (!chatOnly && filteredModels.length > 0);
  const hasFilteredLmStudio = chatOnly && filteredLmStudio.length > 0;

  return (
    <div className="space-y-2">
      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground"
        />
        <Input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Filter downloaded models"
          className="h-9 pl-8"
        />
      </div>

      {!chatOnly && (
        <button
          type="button"
          onClick={() => navigate({ to: "/models" })}
          className="group flex w-full items-center gap-2 rounded-[8px] border border-dashed border-border/60 bg-muted/30 px-2.5 py-2 text-left text-[12.5px] font-medium text-muted-foreground transition-colors hover:border-border hover:bg-muted/60 hover:text-foreground"
        >
          <HugeiconsIcon
            icon={CubeIcon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0"
          />
          <span className="flex-1">Discover & download models</span>
          <HugeiconsIcon
            icon={ArrowRight01Icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0 transition-transform group-hover:translate-x-0.5"
          />
        </button>
      )}

      <div className="max-h-64 overflow-y-auto">
        <div className="p-1">
          {!cachedReady ? (
            <div className="flex items-center gap-2 px-5 py-3">
              <Spinner className="size-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                Loading models…
              </span>
            </div>
          ) : (
            <>
              {hasFilteredDownloaded && (
                <>
                  <ListLabel
                    icon={<DownloadIcon className="size-3" />}
                    collapsed={downloadedCollapsed}
                    onToggle={() => setDownloadedCollapsed((v) => !v)}
                  >
                    Downloaded
                  </ListLabel>
                  {!downloadedCollapsed &&
                    filteredGguf.map((c) => (
                      <div key={c.repo_id}>
                        <ModelRow
                          label={c.repo_id}
                          meta={`GGUF · ${formatBytes(c.size_bytes)}`}
                          selected={value === c.repo_id}
                          onClick={() =>
                            setExpandedGguf((prev) =>
                              prev === c.repo_id ? null : c.repo_id,
                            )
                          }
                          vramStatus={null}
                        />
                        {expandedGguf === c.repo_id && (
                          <GgufVariantExpander
                            repoId={c.repo_id}
                            onSelect={onSelect}
                            gpuGb={
                              gpu.available ? gpu.memoryTotalGb : undefined
                            }
                            systemRamGb={
                              gpu.available
                                ? gpu.systemRamAvailableGb
                                : undefined
                            }
                            onDeleteVariant={async (quant) => {
                              await deleteCachedModel(c.repo_id, quant);
                              refreshCachedLists();
                            }}
                          />
                        )}
                      </div>
                    ))}
                  {!downloadedCollapsed &&
                    !chatOnly &&
                    filteredModels.map((c) => (
                      <div
                        key={c.repo_id}
                        className="flex items-center gap-0.5"
                      >
                        <div className="min-w-0 flex-1">
                          <ModelRow
                            label={c.repo_id}
                            meta={formatBytes(c.size_bytes)}
                            selected={value === c.repo_id}
                            onClick={() =>
                              onSelect(c.repo_id, {
                                source: "hub",
                                isLora: false,
                                isDownloaded: true,
                              })
                            }
                            vramStatus={null}
                          />
                        </div>
                        <ModelDeleteAction
                          ariaLabel={`Delete ${c.repo_id}`}
                          title="Delete cached model?"
                          description={
                            <>
                              This will remove{" "}
                              <span className="font-medium text-foreground">
                                {c.repo_id}
                              </span>{" "}
                              from disk. You can re-download it later.
                            </>
                          }
                          successMessage={`Deleted ${c.repo_id}`}
                          onConfirm={() => deleteCachedModel(c.repo_id)}
                          onDeleted={refreshCachedLists}
                        />
                      </div>
                    ))}
                </>
              )}

              {hasFilteredLmStudio && (
                <>
                  <ListLabel
                    collapsed={lmStudioCollapsed}
                    onToggle={() => setLmStudioCollapsed((v) => !v)}
                  >
                    External
                  </ListLabel>
                  {!lmStudioCollapsed &&
                    filteredLmStudio.map((m) => {
                      const isGguf =
                        isGgufRepo(m.id) || isGgufRepo(m.display_name);
                      return (
                        <div key={m.id}>
                          <ModelRow
                            label={m.model_id ?? m.display_name}
                            meta={
                              isGguf || m.path.toLowerCase().endsWith(".gguf")
                                ? "GGUF"
                                : "Local"
                            }
                            selected={value === m.id}
                            onClick={() => {
                              if (isGguf) {
                                setExpandedGguf((prev) =>
                                  prev === m.id ? null : m.id,
                                );
                              } else {
                                onSelect(m.id, {
                                  source: "local",
                                  isLora: false,
                                  isDownloaded: true,
                                });
                              }
                            }}
                            vramStatus={null}
                          />
                          {expandedGguf === m.id && (
                            <GgufVariantExpander
                              repoId={m.id}
                              onSelect={onSelect}
                              gpuGb={
                                gpu.available ? gpu.memoryTotalGb : undefined
                              }
                              systemRamGb={
                                gpu.available
                                  ? gpu.systemRamAvailableGb
                                  : undefined
                              }
                            />
                          )}
                        </div>
                      );
                    })}
                </>
              )}

              {!hasFilteredDownloaded && !hasFilteredLmStudio && (
                <div className="px-2.5 py-3 text-xs text-muted-foreground">
                  {filter.trim().length > 0
                    ? "No matching models."
                    : chatOnly
                      ? "No external models found."
                      : "No downloaded models yet."}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function LoraModelPicker({
  loraModels,
  value,
  onSelect,
  onModelsChange,
  deleteDisabled = false,
}: {
  loraModels: LoraModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  const gpu = useGpuInfo();

  const normalized = useMemo(
    () =>
      loraModels
        .map((model) => ({
          ...model,
          baseModel:
            model.baseModel || model.description || "Unknown base model",
        }))
        .sort((a, b) => {
          const baseCmp = a.baseModel.localeCompare(b.baseModel);
          if (baseCmp !== 0) return baseCmp;
          // Prioritize unsloth publisher within External (lmstudio) group
          if (a.baseModel === "External" && b.baseModel === "External") {
            const aUnsloth = a.name.startsWith("unsloth/") ? 0 : 1;
            const bUnsloth = b.name.startsWith("unsloth/") ? 0 : 1;
            if (aUnsloth !== bUnsloth) return aUnsloth - bUnsloth;
          }
          const aTime = a.updatedAt ?? -1;
          const bTime = b.updatedAt ?? -1;
          if (aTime !== bTime) return bTime - aTime;
          return a.name.localeCompare(b.name);
        }),
    [loraModels],
  );

  const grouped = useMemo(() => {
    const needle = normalizeForSearch(query.trim());
    const out = new Map<string, LoraModelOption[]>();

    for (const model of normalized) {
      const searchText = normalizeForSearch(
        `${model.name} ${model.baseModel} ${model.id}`,
      );
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
          placeholder="Search trained models"
          className="h-9 pl-8"
        />
      </div>

      <div className="max-h-64 overflow-y-auto">
        <div className="p-1">
          {grouped.length === 0 ? (
            <div className="px-2.5 py-2 text-xs text-muted-foreground">
              No trained models found.
            </div>
          ) : (
            grouped.map(([baseModel, adapters], index) => (
              <div key={baseModel}>
                {index > 0 ? <div className="my-1" /> : null}
                <ListLabel>{baseModel}</ListLabel>
                {adapters.map((adapter) => {
                  const isLocal = adapter.source === "local";
                  const isTraining = adapter.source === "training";
                  const isExported = adapter.source === "exported";
                  const isMerged = adapter.exportType === "merged";
                  const isGguf = adapter.exportType === "gguf";
                  const isExportedGguf = isExported && isGguf;
                  const canDelete = (isTraining || isExported) && !isExportedGguf;
                  const isTrainingFull = isTraining && isMerged;
                  const isLocalGgufDir =
                    isLocal &&
                    (isGgufRepo(adapter.id) || isGgufRepo(adapter.name));
                  const tag = isLocal
                    ? isLocalGgufDir
                      ? "GGUF"
                      : "Local"
                    : isGguf
                      ? "GGUF"
                      : isTrainingFull
                        ? "Full"
                      : isExported
                        ? isMerged
                          ? "Merged"
                          : "LoRA"
                        : "LoRA";
                  const meta = isLocal
                    ? isLocalGgufDir
                      ? "GGUF"
                      : "Local"
                    : isTrainingFull
                      ? "Full finetune"
                    : isExported
                      ? `${tag} · Exported`
                      : tag;
                  return (
                    <div key={adapter.id}>
                      <div className="flex items-center gap-0.5">
                        <div className="min-w-0 flex-1">
                          <ModelRow
                            label={adapter.name}
                            meta={meta}
                            selected={value === adapter.id}
                            onClick={() => {
                              if (isLocalGgufDir || isExportedGguf) {
                                setExpandedGguf((prev) =>
                                  prev === adapter.id ? null : adapter.id,
                                );
                              } else {
                                onSelect(adapter.id, {
                                  source: isLocal
                                    ? "local"
                                    : isExported
                                      ? "exported"
                                      : "lora",
                                  isLora: !isLocal && !isMerged && !isGguf,
                                  isDownloaded: true,
                                });
                              }
                            }}
                            tooltipText={
                              <>
                                <span className="block break-words">
                                  {adapter.name}
                                </span>
                                <span className="block mt-1 text-[10px] text-muted-foreground break-all">
                                  {adapter.id}
                                </span>
                              </>
                            }
                          />
                        </div>
                        {canDelete && (
                          <ModelDeleteAction
                            ariaLabel={`Delete ${adapter.name}`}
                            title="Delete fine-tuned model?"
                            description={
                              <>
                                This will remove{" "}
                                <span className="font-medium text-foreground">
                                  {adapter.name}
                                </span>{" "}
                                from disk. This cannot be undone.
                              </>
                            }
                            successMessage={`Deleted ${adapter.name}`}
                            disabled={deleteDisabled}
                            onConfirm={() =>
                              deleteFineTunedModel({
                                modelPath: adapter.id,
                                source: isExported ? "exported" : "training",
                                exportType: adapter.exportType,
                              })
                            }
                            onDeleted={() =>
                              onModelsChange?.({ id: adapter.id })
                            }
                          />
                        )}
                      </div>
                      {expandedGguf === adapter.id && (
                        <GgufVariantExpander
                          repoId={adapter.id}
                          onSelect={onSelect}
                          gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                          systemRamGb={
                            gpu.available ? gpu.systemRamAvailableGb : undefined
                          }
                          sourceOverride={isExportedGguf ? "exported" : undefined}
                          deleteVariantTitle="Delete exported GGUF variant?"
                          renderDeleteVariantDescription={(quant) => (
                            <>
                              This will remove{" "}
                              <span className="font-medium text-foreground">
                                {adapter.name} ({quant})
                              </span>{" "}
                              from disk. This cannot be undone.
                            </>
                          )}
                          getDeleteVariantSuccessMessage={(quant) =>
                            `Deleted ${adapter.name} ${quant}`
                          }
                          deleteDisabled={deleteDisabled}
                          onDeleteVariant={
                            isExportedGguf
                              ? async (quant) => {
                                  await deleteFineTunedModel({
                                    modelPath: adapter.id,
                                    source: "exported",
                                    exportType: "gguf",
                                    ggufVariant: quant,
                                  });
                                  onModelsChange?.({
                                    id: adapter.id,
                                    ggufVariant: quant,
                                  });
                                }
                              : undefined
                          }
                        />
                      )}
                    </div>
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
