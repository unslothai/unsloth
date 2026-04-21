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
import { usePlatformStore } from "@/config/env";
import {
  type ScanFolderInfo,
  addScanFolder,
  deleteCachedModel,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  listLocalModels,
  listRecommendedFolders,
  listScanFolders,
  removeScanFolder,
} from "@/features/chat/api/chat-api";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  LocalModelInfo,
} from "@/features/chat/api/chat-api";
import type { GgufVariantDetail } from "@/features/chat/types/api";
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
import { Add01Icon, Cancel01Icon, Folder02Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { FolderBrowser } from "./folder-browser";
import { ChevronDownIcon, ChevronRightIcon, DownloadIcon, StarIcon, Trash2Icon } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { toast } from "sonner";
import type {
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./types";

function dedupe(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

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
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
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
        source: isLocalPath ? "local" : "hub",
        isLora: false,
        ggufVariant: quant,
        isDownloaded: isLocalPath ? true : downloaded,
        expectedBytes: sizeBytes,
      });
    },
    [repoId, isLocalPath, onSelect],
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
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteVariant(v.quant);
                }}
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
let _customFolderCache: LocalModelInfo[] = [];
let _scanFoldersCache: ScanFolderInfo[] = [];

/** Sort LM Studio models with unsloth publisher first. */
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
  models,
  value,
  onSelect,
  onFoldersChange,
}: {
  models: ModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onFoldersChange?: () => void;
}) {
  const gpu = useGpuInfo();
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  const { results, isLoading, isLoadingMore, fetchMore } =
    useHfModelSearch(debouncedQuery);

  // Sets of lowercased repo ids that the store or HF search have
  // confirmed are GGUF. Absence means "no hint" and lets hasGgufSuffix
  // take over as fallback, rather than conflating unknown with known-
  // not-GGUF. Keys are lowercased so that store IDs and HF search IDs
  // that differ only by casing still match the same hint.
  const modelGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const model of models) {
      if (model.isGguf) ids.add(model.id.toLowerCase());
    }
    return ids;
  }, [models]);
  const resultGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const result of results) {
      if (result.isGguf) ids.add(result.id.toLowerCase());
    }
    return ids;
  }, [results]);
  const isKnownGgufRepo = useCallback(
    (id: string): boolean => {
      const key = id.toLowerCase();
      return isGgufRepo(id, resultGgufIds.has(key) || modelGgufIds.has(key));
    },
    [modelGgufIds, resultGgufIds],
  );

  // Track which GGUF repo is expanded for variant selection
  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);

  // Delete confirmation dialog state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [downloadedCollapsed, setDownloadedCollapsed] = useState(false);
  const [customFoldersCollapsed, setCustomFoldersCollapsed] = useState(false);
  const [recommendedCollapsed, setRecommendedCollapsed] = useState(false);

  // Cached (already downloaded) repos -- use module-level cache so
  // re-mounting the popover does not flash an empty "Downloaded" section.
  const [cachedGguf, setCachedGguf] =
    useState<CachedGgufRepo[]>(_cachedGgufCache);
  const [cachedModels, setCachedModels] =
    useState<CachedModelRepo[]>(_cachedModelsCache);
  const alreadyCached =
    _cachedGgufCache.length > 0 || _cachedModelsCache.length > 0;
  const [cachedReady, setCachedReady] = useState(alreadyCached);

  // LM Studio local models -- module-level cache so re-mounting the
  // popover does not flash an empty section (same pattern as GGUF/models).
  const [lmStudioModels, setLmStudioModels] =
    useState<LocalModelInfo[]>(_lmStudioCache);
  const [customFolderModels, setCustomFolderModels] =
    useState<LocalModelInfo[]>(_customFolderCache);

  // Custom scan folders management
  const [scanFolders, setScanFolders] = useState<ScanFolderInfo[]>(_scanFoldersCache);
  const [folderInput, setFolderInput] = useState("");
  const [folderError, setFolderError] = useState<string | null>(null);
  const [showFolderInput, setShowFolderInput] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);
  const [recommendedFolders, setRecommendedFolders] = useState<string[]>([]);

  const refreshLocalModelsList = useCallback(() => {
    listLocalModels()
      .then((res) => {
        const lm = sortLmStudio(
          res.models.filter((m) => m.source === "lmstudio"),
        );
        _lmStudioCache = lm;
        setLmStudioModels(lm);
        const cf = res.models.filter((m) => m.source === "custom");
        _customFolderCache = cf;
        setCustomFolderModels(cf);
      })
      .catch(() => {});
  }, []);

  const refreshScanFolders = useCallback(() => {
    listScanFolders()
      .then((v) => {
        _scanFoldersCache = v;
        setScanFolders(v);
      })
      .catch(() => {});
  }, []);

  const handleAddFolder = useCallback(async (overridePath?: string) => {
    // Accept an explicit path so the folder browser can submit the
    // chosen path in the same tick it calls `setFolderInput`; reading
    // `folderInput` alone would race the state update.
    const raw = overridePath !== undefined ? overridePath : folderInput;
    const trimmed = raw.trim();
    if (!trimmed || folderLoading) return;
    setFolderError(null);
    setFolderLoading(true);
    // True when the request originated from the folder browser's
    // ``onSelect`` (one-click "Use this folder"). In that flow the
    // typed-input panel is closed, so the inline ``folderError``
    // paragraph is invisible. Surface failures via toast instead so
    // the action doesn't appear to silently no-op when the backend
    // rejects (denylisted path, sandbox 403, etc.).
    const fromBrowser = overridePath !== undefined;
    try {
      const created = await addScanFolder(trimmed);
      // Backend returns existing row for duplicates, so deduplicate
      const next = _scanFoldersCache.some((f) => f.id === created.id || f.path === created.path)
        ? _scanFoldersCache
        : [..._scanFoldersCache, created];
      _scanFoldersCache = next;
      setScanFolders(next);
      setFolderInput("");
      setShowFolderInput(false);
      refreshLocalModelsList();
      onFoldersChange?.();
      // Background reconciliation with the server
      void refreshScanFolders();
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to add folder";
      setFolderError(message);
      if (fromBrowser) {
        toast.error("Couldn't add folder", { description: message });
      }
    } finally {
      setFolderLoading(false);
    }
  }, [folderInput, folderLoading, refreshScanFolders, refreshLocalModelsList, onFoldersChange]);

  const handleRemoveFolder = useCallback(async (id: number) => {
    try {
      await removeScanFolder(id);
      // Optimistic update so the folder disappears immediately
      const next = _scanFoldersCache.filter((f) => f.id !== id);
      _scanFoldersCache = next;
      setScanFolders(next);
      refreshScanFolders();
      refreshLocalModelsList();
      onFoldersChange?.();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to remove folder");
      refreshScanFolders();
    }
  }, [refreshScanFolders, refreshLocalModelsList, onFoldersChange]);

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
    // Always refresh LM Studio + custom folder models (not gated by alreadyCached)
    refreshLocalModelsList();
    refreshScanFolders();
    listRecommendedFolders()
      .then(setRecommendedFolders)
      .catch(() => {});

    // Always refetch cached GGUF/model lists. The module-level caches give
    // an instant render with stale data (no spinner flash), but newly
    // downloaded repos won't appear unless we re-hit the backend on every
    // mount.  Initial state already has cachedReady=alreadyCached, so the
    // background refresh is invisible when we already had data.
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
  }, [refreshLocalModelsList, refreshScanFolders]);

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
      toast.error(
        err instanceof Error ? err.message : "Failed to delete model",
      );
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
      .filter((id) => !chatOnly || isKnownGgufRepo(id))
      .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id));
    // Sort: GGUFs first, then hub models
    const gguf: string[] = [];
    const hub: string[] = [];
    for (const id of all) {
      if (isKnownGgufRepo(id)) gguf.push(id);
      else hub.push(id);
    }
    return [...gguf, ...hub];
  }, [models, value, downloadedSet, chatOnly, isKnownGgufRepo]);

  // Infinite scroll paging for the recommended section
  const [recommendedPage, setRecommendedPage] = useState(1);
  // Reset page when the underlying list changes
  useEffect(() => {
    setRecommendedPage(1);
  }, [models, chatOnly]);

  const visibleRecommendedIds = useMemo(() => {
    const hubStartIndex = recommendedIds.findIndex((id) => !isKnownGgufRepo(id));
    const allGguf =
      hubStartIndex === -1
        ? recommendedIds
        : recommendedIds.slice(0, hubStartIndex);
    const allHub =
      hubStartIndex === -1 ? [] : recommendedIds.slice(hubStartIndex);
    // Interleave in chunks of 4: [4 gguf, 4 hub, 4 gguf, 4 hub, ...]
    const result: string[] = [];
    for (let p = 0; p < recommendedPage; p++) {
      result.push(...allGguf.slice(p * 4, (p + 1) * 4));
      result.push(...allHub.slice(p * 4, (p + 1) * 4));
    }
    return result;
  }, [recommendedIds, recommendedPage, isKnownGgufRepo]);

  const hasMoreRecommended =
    visibleRecommendedIds.length < recommendedIds.length;

  const showHfSection = debouncedQuery.trim().length > 0;

  // Recommended models that match the current search query
  const filteredRecommendedIds = useMemo(() => {
    if (!showHfSection) return [];
    const q = normalizeForSearch(debouncedQuery.trim());
    return recommendedIds.filter((id) => normalizeForSearch(id).includes(q));
  }, [showHfSection, debouncedQuery, recommendedIds]);

  // Fetch VRAM info for visible models, plus any models surfaced by a search
  // query so that filtered recommended models also show VRAM badges.
  // Skip GGUF repos: they have no safetensors metadata and the render layer
  // already shows a static "GGUF" badge instead of VRAM data.
  const idsForVram = useMemo(() => {
    const ids = showHfSection
      ? [...new Set([...visibleRecommendedIds, ...filteredRecommendedIds])]
      : visibleRecommendedIds;
    return ids.filter((id) => !isKnownGgufRepo(id));
  }, [visibleRecommendedIds, showHfSection, filteredRecommendedIds, isKnownGgufRepo]);
  const { paramCountById: recommendedParamCountById } =
    useRecommendedModelVram(idsForVram);

  const recommendedSet = useMemo(
    () =>
      new Set(showHfSection ? filteredRecommendedIds : visibleRecommendedIds),
    [showHfSection, filteredRecommendedIds, visibleRecommendedIds],
  );

  const hfIds = useMemo(() => {
    if (!showHfSection) return [];
    return results
      .map((result) => result.id)
      .filter((id) => !recommendedSet.has(id))
      .filter((id) => !chatOnly || isKnownGgufRepo(id))
      .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id));
  }, [recommendedSet, results, showHfSection, chatOnly, isKnownGgufRepo]);

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
    const ids = showHfSection ? filteredRecommendedIds : visibleRecommendedIds;
    for (const id of ids) {
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
  }, [
    showHfSection,
    filteredRecommendedIds,
    visibleRecommendedIds,
    recommendedParamCountById,
    gpu,
  ]);

  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    results.length,
  );

  // Sentinel + IntersectionObserver for recommended infinite scroll.
  // We disconnect after each fire so the observer doesn't loop while
  // React re-renders; the effect re-creates it on the next page.
  // Uses a callback ref for the sentinel so we detect mount/unmount reliably.
  const [recommendedSentinel, setRecommendedSentinel] =
    useState<HTMLDivElement | null>(null);
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
    return () => {
      clearTimeout(timer);
      obs.disconnect();
    };
  }, [recommendedSentinel, hasMoreRecommended, recommendedPage, scrollRef]);

  /** Handle clicking a model row — GGUF repos expand, others load directly. */
  const handleModelClick = useCallback(
    (id: string) => {
      if (isKnownGgufRepo(id)) {
        // Toggle GGUF variant expander
        setExpandedGguf((prev) => (prev === id ? null : id));
      } else {
        onSelect(id, { source: "hub", isLora: false });
      }
    },
    [onSelect, isKnownGgufRepo],
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
              <span className="text-xs text-muted-foreground">
                Loading models…
              </span>
            </div>
          ) : !showHfSection &&
            (cachedGguf.length > 0 ||
              (!chatOnly && cachedModels.length > 0)) ? (
            <>
              <ListLabel
                icon={<DownloadIcon className="size-3" />}
                collapsed={downloadedCollapsed}
                onToggle={() => setDownloadedCollapsed((v) => !v)}
              >Downloaded</ListLabel>
              {!downloadedCollapsed && cachedGguf.map((c) => (
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
                      gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                      systemRamGb={
                        gpu.available ? gpu.systemRamAvailableGb : undefined
                      }
                      onDeleteVariant={(quant) =>
                        setDeleteTarget(`${c.repo_id}::${quant}`)
                      }
                    />
                  )}
                </div>
              ))}
              {!downloadedCollapsed && !chatOnly &&
                cachedModels.map((c) => (
                  <div key={c.repo_id} className="flex items-center gap-0.5">
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
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteTarget(c.repo_id);
                      }}
                      className="shrink-0 rounded-md p-1.5 text-muted-foreground/60 transition-colors hover:bg-destructive/10 hover:text-destructive"
                    >
                      <Trash2Icon className="size-3.5" />
                    </button>
                  </div>
                ))}
            </>
          ) : null}

          {!showHfSection && chatOnly && lmStudioModels.length > 0 ? (
            <>
              <ListLabel>LM Studio</ListLabel>
              {lmStudioModels.map((m) => {
                const isGguf = isGgufRepo(m.id) || isGgufRepo(m.display_name);
                return (
                  <div key={m.id}>
                    <ModelRow
                      label={m.model_id ?? m.display_name}
                      meta={
                        isGguf || m.path.toLowerCase().endsWith(".gguf") ? "GGUF" : "Local"
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
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                        systemRamGb={
                          gpu.available ? gpu.systemRamAvailableGb : undefined
                        }
                      />
                    )}
                  </div>
                );
              })}
            </>
          ) : null}

          {!showHfSection ? (
            <>
              <div className="flex items-center gap-1 px-2.5 py-1.5">
                <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <HugeiconsIcon icon={Folder02Icon} className="size-3" />
                  Custom Folders
                </span>
                <div className="flex items-center gap-0.5">
                  <button
                    type="button"
                    aria-label={showFolderInput ? "Cancel adding folder" : "Add scan folder by path"}
                    title={showFolderInput ? "Cancel" : "Add by typing a path"}
                    onClick={() => {
                      setShowFolderInput((open) => {
                        if (open) { setFolderInput(""); setFolderError(null); }
                        return !open;
                      });
                    }}
                    className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                  >
                    <HugeiconsIcon icon={showFolderInput ? Cancel01Icon : Add01Icon} className="size-3" />
                  </button>
                  <button
                    type="button"
                    aria-label="Browse for a folder on the server"
                    title="Browse folders on the server"
                    onClick={() => setShowFolderBrowser(true)}
                    className="shrink-0 rounded p-0.5 text-muted-foreground/60 transition-colors hover:text-foreground"
                  >
                    <HugeiconsIcon icon={Search01Icon} className="size-2.5" />
                  </button>
                </div>
                <div className="ml-auto">
                  <button
                    type="button"
                    aria-label={customFoldersCollapsed ? "Expand custom folders" : "Collapse custom folders"}
                    title={customFoldersCollapsed ? "Expand" : "Collapse"}
                    onClick={() => setCustomFoldersCollapsed((v) => !v)}
                    className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                  >
                    {customFoldersCollapsed
                      ? <ChevronRightIcon className="size-3" />
                      : <ChevronDownIcon className="size-3" />}
                  </button>
                </div>
              </div>

              {/* Folder paths */}
              {!customFoldersCollapsed && scanFolders.map((f) => (
                <div
                  key={f.id}
                  className="group flex items-center gap-1.5 px-2.5 py-0.5"
                >
                  <HugeiconsIcon icon={Folder02Icon} className="size-3 shrink-0 text-muted-foreground/40" />
                  <span
                    className="min-w-0 flex-1 truncate font-mono text-[10px] text-muted-foreground/70"
                    title={f.path}
                  >
                    {f.path}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleRemoveFolder(f.id)}
                    aria-label={`Remove folder ${f.path}`}
                    className="shrink-0 rounded p-1 text-foreground/70 transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:bg-destructive/10 focus-visible:text-destructive"
                  >
                    <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                  </button>
                </div>
              ))}

              {/* Recommended folders */}
              {!customFoldersCollapsed && (() => {
                const registered = new Set(scanFolders.map((f) => f.path));
                const unregistered = recommendedFolders.filter((p) => !registered.has(p));
                if (unregistered.length === 0) return null;
                return (
                  <div className="flex flex-wrap gap-1 px-2.5 pb-0.5">
                    {unregistered.map((p) => (
                      <button
                        key={p}
                        type="button"
                        onClick={() => void handleAddFolder(p)}
                        disabled={folderLoading}
                        title={`Add ${p}`}
                        className="rounded-full border border-dashed border-border/50 px-2 py-0.5 font-mono text-[10px] text-muted-foreground/70 transition-colors hover:border-foreground/30 hover:bg-accent hover:text-foreground disabled:opacity-40"
                      >
                        <span className="text-[11px] font-semibold">+</span> {p.length > 30 ? `...${p.slice(-27)}` : p}
                      </button>
                    ))}
                  </div>
                );
              })()}

              {/* Add folder input */}
              {!customFoldersCollapsed && showFolderInput && (
                <div className="px-2.5 pb-1 pt-0.5">
                  <div className="flex items-center gap-1">
                    <HugeiconsIcon icon={Folder02Icon} className="size-3 shrink-0 text-muted-foreground/40" />
                    <input
                      value={folderInput}
                      onChange={(e) => { setFolderInput(e.target.value); setFolderError(null); }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") { e.preventDefault(); handleAddFolder(); }
                        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setShowFolderInput(false); setFolderInput(""); setFolderError(null); }
                      }}
                      placeholder="/path/to/models"
                      className="h-6 min-w-0 flex-1 rounded border border-border/50 bg-transparent px-1.5 font-mono text-[10px] text-foreground outline-none placeholder:text-muted-foreground/40 focus:border-foreground/20"
                      disabled={folderLoading}
                      autoFocus={true}
                    />
                    <button
                      type="button"
                      onClick={() => setShowFolderBrowser(true)}
                      disabled={folderLoading}
                      aria-label="Browse for folder"
                      title="Browse folders on the server"
                      className="flex h-6 shrink-0 items-center justify-center rounded border border-border/50 px-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
                    >
                      <HugeiconsIcon icon={Search01Icon} className="size-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => { void handleAddFolder(); }}
                      disabled={folderLoading || !folderInput.trim()}
                      className="h-6 shrink-0 rounded border border-border/50 px-1.5 text-[10px] text-muted-foreground transition-colors hover:bg-accent disabled:opacity-40"
                    >
                      Add
                    </button>
                  </div>
                  {folderError && (
                    <p className="px-0.5 pt-0.5 text-[10px] text-destructive">{folderError}</p>
                  )}
                </div>
              )}

              <FolderBrowser
                open={showFolderBrowser}
                onOpenChange={setShowFolderBrowser}
                initialPath={folderInput.trim() || undefined}
                onSelect={(picked) => {
                  setFolderInput(picked);
                  setFolderError(null);
                  // One-click UX: the "Use this folder" button submits
                  // the scan folder directly. Pass the path explicitly
                  // because `folderInput` state hasn't flushed yet.
                  void handleAddFolder(picked);
                }}
              />


              {/* Models from custom folders */}
              {!customFoldersCollapsed && customFolderModels.map((m) => {
                const isGgufFile = m.path.toLowerCase().endsWith(".gguf");
                const isGguf =
                  isGgufFile ||
                  isGgufRepo(m.id) ||
                  isGgufRepo(m.display_name);
                // Single .gguf files (e.g. Ollama blobs) load directly;
                // GGUF repos/directories expand to pick a variant.
                const isDirectGguf = isGgufFile;
                return (
                  <div key={m.id}>
                    <ModelRow
                      label={m.model_id ?? m.display_name}
                      meta={isGguf ? "GGUF" : "Local"}
                      selected={value === m.id}
                      onClick={() => {
                        if (isDirectGguf) {
                          onSelect(m.id, {
                            source: "local",
                            isLora: false,
                            isDownloaded: true,
                          });
                        } else if (isGguf) {
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
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                        systemRamGb={
                          gpu.available ? gpu.systemRamAvailableGb : undefined
                        }
                      />
                    )}
                  </div>
                );
              })}
            </>
          ) : null}

          {!showHfSection && cachedReady ? (
            <>
              <ListLabel
                icon={<StarIcon className="size-3" />}
                collapsed={recommendedCollapsed}
                onToggle={() => setRecommendedCollapsed((v) => !v)}
              >Recommended</ListLabel>
              {recommendedCollapsed ? null : visibleRecommendedIds.length === 0 ? (
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
                          isKnownGgufRepo(id)
                            ? "GGUF"
                            : (vram?.detail ?? extractParamLabel(id))
                        }
                        selected={value === id}
                        onClick={() => {
                          if (isKnownGgufRepo(id)) {
                            setExpandedGguf((prev) => (prev === id ? null : id));
                          } else {
                            handleModelClick(id);
                          }
                        }}
                        vramStatus={
                          isKnownGgufRepo(id) ? null : (vram?.status ?? null)
                        }
                        vramEst={isKnownGgufRepo(id) ? undefined : vram?.est}
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                      />
                      {expandedGguf === id && (
                        <GgufVariantExpander
                          repoId={id}
                          onSelect={onSelect}
                          gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                          systemRamGb={
                            gpu.available ? gpu.systemRamAvailableGb : undefined
                          }
                        />
                      )}
                    </div>
                  );
                })
              )}
              {!recommendedCollapsed && hasMoreRecommended && (
                <>
                  <div ref={recommendedSentinelRef} className="h-px" />
                  <div className="flex items-center justify-center py-2">
                    <Spinner className="size-3.5 text-muted-foreground" />
                  </div>
                </>
              )}
            </>
          ) : null}

          {showHfSection && filteredRecommendedIds.length > 0 ? (
            <>
              <ListLabel icon={<StarIcon className="size-3" />}>Recommended</ListLabel>
              {filteredRecommendedIds.map((id) => {
                const vram = recommendedVramMap.get(id);
                return (
                  <div key={id}>
                    <ModelRow
                      label={id}
                      meta={
                        isKnownGgufRepo(id)
                          ? "GGUF"
                          : (vram?.detail ?? extractParamLabel(id))
                      }
                      selected={value === id}
                      onClick={() => {
                        if (isKnownGgufRepo(id)) {
                          setExpandedGguf((prev) => (prev === id ? null : id));
                        } else {
                          handleModelClick(id);
                        }
                      }}
                      vramStatus={
                        isKnownGgufRepo(id) ? null : (vram?.status ?? null)
                      }
                      vramEst={isKnownGgufRepo(id) ? undefined : vram?.est}
                      gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                    />
                    {expandedGguf === id && (
                      <GgufVariantExpander
                        repoId={id}
                        onSelect={onSelect}
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                        systemRamGb={
                          gpu.available ? gpu.systemRamAvailableGb : undefined
                        }
                      />
                    )}
                  </div>
                );
              })}
            </>
          ) : null}

          {showHfSection ? (
            <>
              {(hfIds.length > 0 || isLoading) && (
                <ListLabel>Hugging Face</ListLabel>
              )}
              {hfIds.length === 0 && !isLoading ? (
                filteredRecommendedIds.length === 0 ? (
                  <div className="px-2.5 py-2 text-xs text-muted-foreground">
                    No matching models.
                  </div>
                ) : null
                ) : (
                hfIds.map((id) => {
                  const vram = vramMap.get(id);
                  const isSearchGguf = isKnownGgufRepo(id);
                  return (
                    <div key={id}>
                      <ModelRow
                        label={id}
                        meta={
                          isSearchGguf
                            ? "GGUF"
                            : (metricsById.get(id) ?? extractParamLabel(id))
                        }
                        selected={value === id}
                        onClick={() => {
                          if (isSearchGguf) {
                            setExpandedGguf((prev) => (prev === id ? null : id));
                          } else {
                            handleModelClick(id);
                          }
                        }}
                        vramStatus={
                          isSearchGguf ? null : (vram?.status ?? null)
                        }
                        vramEst={isSearchGguf ? undefined : vram?.est}
                        gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                      />
                      {expandedGguf === id && (
                        <GgufVariantExpander
                          repoId={id}
                          onSelect={onSelect}
                          gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                          systemRamGb={
                            gpu.available ? gpu.systemRamAvailableGb : undefined
                          }
                        />
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

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open && !deleting) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete cached model?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove{" "}
              <span className="font-medium text-foreground">
                {deleteTarget?.includes("::")
                  ? `${deleteTarget.split("::")[0]} (${deleteTarget.split("::")[1]})`
                  : deleteTarget}
              </span>{" "}
              from disk. You can re-download it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>No</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={(e) => {
                e.preventDefault();
                handleDeleteConfirm();
              }}
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
          // Prioritize unsloth publisher within LM Studio group
          if (a.baseModel === "LM Studio" && b.baseModel === "LM Studio") {
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
                      <ModelRow
                        label={adapter.name}
                        meta={meta}
                        selected={value === adapter.id}
                        onClick={() => {
                          if (isLocalGgufDir) {
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
                      {expandedGguf === adapter.id && (
                        <GgufVariantExpander
                          repoId={adapter.id}
                          onSelect={onSelect}
                          gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                          systemRamGb={
                            gpu.available ? gpu.systemRamAvailableGb : undefined
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
