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
import { ApiProviderLogo } from "@/features/chat/api-provider-logo";
import {
  type ScanFolderInfo,
  addScanFolder,
  deleteCachedModel,
  deleteFineTunedModel,
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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import type { GgufVariantDetail } from "@/features/chat/types/api";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import {
  type HubOption,
  HubOptionMenu,
} from "@/features/hub/catalog/hub-option-menu";
import { TrainIcon } from "@/features/hub/components/train-icon";
import { useHubInfiniteScroll } from "@/features/hub/hooks/use-hub-infinite-scroll";
import {
  type HfModelResult,
  type HfSortKey,
  useHubModelSearch,
} from "@/features/hub/hooks/use-hub-model-search";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { classifyUnslothSupport } from "@/features/hub/lib/unsloth-support";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { useDebouncedValue, useGpuInfo } from "@/hooks";
import { extractParamLabel } from "@/lib/model-size";
import { toast } from "@/lib/toast";
import { cn, formatCompact } from "@/lib/utils";
import type { VramFitStatus } from "@/lib/vram";
import { checkVramFit, estimateLoadingVram } from "@/lib/vram";
import {
  Add01Icon,
  AudioWave01Icon,
  Cancel01Icon,
  DashboardCircleIcon,
  Download01Icon,
  Folder02Icon,
  RemoveCircleIcon,
  Search01Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import {
  type Dispatch,
  type KeyboardEvent,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { FolderBrowser } from "./folder-browser";
import {
  type ModelCapabilities,
  detectCapabilities,
  hasAnyCapability,
} from "./model-capabilities";
import { ModelDeleteAction } from "./model-delete-action";
import { ModelLoadSettingsAction } from "./model-load-settings-action";
import {
  type ModelLoadTimes,
  loadedAt,
  useModelLoadTimes,
} from "./model-usage";
import {
  type FormatFilter,
  estimateQuantBytes,
  fitsDevice,
  isMlxId,
  isMobileVariant,
  isRecommendableFormat,
  matchesFormatFilter,
  paramsFromId,
} from "./recommended-fit";
import { parseMetaTokens, splitRepoLabel } from "./row-meta";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./types";

function dedupe(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

/** Repos published by Unsloth; the rest group under the "Other models" section. */
function isUnslothRepoId(repoId: string): boolean {
  return repoId.toLowerCase().startsWith("unsloth/");
}

/** Lowercase and strip separators for fuzzy search. */
function normalizeForSearch(s: string): string {
  return s.toLowerCase().replace(/[\s_.-]/g, "");
}

function makeModelOptionKey(section: string, id: string): string {
  return `${section}::${id}`;
}

function makeModelOptionChildrenId(optionKey: string): string {
  return `model-picker-children-${optionKey.replace(/[^A-Za-z0-9_-]/g, "-")}`;
}

function focusFirstChildOption(optionKey: string): boolean {
  const childList = document.getElementById(
    makeModelOptionChildrenId(optionKey),
  );
  const option = childList?.querySelector<HTMLElement>(
    "[data-model-picker-option]",
  );
  if (!option) {
    return false;
  }
  option.focus();
  return true;
}

type ModelRowOptionProps = {
  id: string;
  tabIndex: number;
  onFocus: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) => void;
  "data-model-picker-option": true;
  "data-model-picker-active-option"?: "true";
  "aria-current"?: "true";
};

function useRovingModelList({
  label,
  optionKeys,
  selectedOptionKey,
  onNavigatePastStart,
  onNavigatePastEnd,
}: {
  label: string;
  optionKeys: string[];
  selectedOptionKey?: string;
  onNavigatePastStart?: () => void;
  onNavigatePastEnd?: () => void;
}) {
  const rawListboxId = useId();
  const listboxId = `model-picker-${rawListboxId.replace(/:/g, "")}`;
  const [rovingOptionKey, setRovingOptionKey] = useState<string | null>(null);

  const preferredOptionKey =
    selectedOptionKey && optionKeys.includes(selectedOptionKey)
      ? selectedOptionKey
      : (optionKeys[0] ?? null);
  const activeOptionKey =
    rovingOptionKey && optionKeys.includes(rovingOptionKey)
      ? rovingOptionKey
      : preferredOptionKey;

  const getOptionDomId = useCallback(
    (optionKey: string) => {
      const index = optionKeys.indexOf(optionKey);
      return index === -1 ? undefined : `${listboxId}-option-${index}`;
    },
    [listboxId, optionKeys],
  );

  const focusOption = useCallback(
    (optionKey: string) => {
      const id = getOptionDomId(optionKey);
      if (!id) {
        return;
      }
      document.getElementById(id)?.focus();
    },
    [getOptionDomId],
  );

  const moveFocus = useCallback(
    (
      fromOptionKey: string,
      direction: "next" | "previous" | "first" | "last",
    ) => {
      if (optionKeys.length === 0) {
        return;
      }

      const currentIndex = optionKeys.indexOf(fromOptionKey);
      let nextIndex = currentIndex === -1 ? 0 : currentIndex;
      if (direction === "next") {
        if (currentIndex >= optionKeys.length - 1) {
          onNavigatePastEnd?.();
          return;
        }
        nextIndex = Math.min(optionKeys.length - 1, nextIndex + 1);
      } else if (direction === "previous") {
        if (currentIndex <= 0) {
          onNavigatePastStart?.();
          return;
        }
        nextIndex = Math.max(0, nextIndex - 1);
      } else if (direction === "first") {
        nextIndex = 0;
      } else {
        nextIndex = optionKeys.length - 1;
      }

      const nextOptionKey = optionKeys[nextIndex];
      setRovingOptionKey(nextOptionKey);
      focusOption(nextOptionKey);
    },
    [focusOption, onNavigatePastEnd, onNavigatePastStart, optionKeys],
  );

  const getOptionProps = useCallback(
    (optionKey: string, selected: boolean): ModelRowOptionProps => ({
      id: getOptionDomId(optionKey) ?? `${listboxId}-option-missing`,
      tabIndex: 0,
      onFocus: () => {
        setRovingOptionKey(optionKey);
      },
      onKeyDown: (event) => {
        if (event.key === "ArrowDown") {
          event.preventDefault();
          moveFocus(optionKey, "next");
        } else if (event.key === "ArrowUp") {
          event.preventDefault();
          moveFocus(optionKey, "previous");
        } else if (event.key === "Home") {
          event.preventDefault();
          moveFocus(optionKey, "first");
        } else if (event.key === "End") {
          event.preventDefault();
          moveFocus(optionKey, "last");
        }
      },
      "data-model-picker-option": true,
      "data-model-picker-active-option":
        optionKey === activeOptionKey ? "true" : undefined,
      "aria-current": selected ? "true" : undefined,
    }),
    [activeOptionKey, getOptionDomId, listboxId, moveFocus],
  );

  return {
    activeOptionKey,
    focusOption,
    getOptionProps,
    moveFocus,
    listboxProps: {
      id: listboxId,
      "data-model-picker-list": true,
      "aria-label": label,
    },
  };
}

function ListLabel({
  children,
  icon,
  action,
  collapsed,
  onToggle,
}: {
  children: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
  collapsed?: boolean;
  onToggle?: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-1 px-2.5 pb-1.5 pt-3">
      <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {icon}
        {children}
      </span>
      {(action || onToggle) && (
        <div className="flex items-center gap-0.5">
          {action}
          {onToggle && (
            <button
              type="button"
              onClick={onToggle}
              aria-label={collapsed ? "Expand section" : "Collapse section"}
              className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
            >
              {collapsed ? (
                <ChevronRightIcon className="size-3" />
              ) : (
                <ChevronDownIcon className="size-3" />
              )}
            </button>
          )}
        </div>
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

// Small icon badges for what a model can do (vision / reasoning / audio).
// Vision and reasoning badges were dropped to keep rows uncluttered.
const CAPABILITY_BADGES = [
  { key: "audio" as const, icon: AudioWave01Icon, title: "Audio" },
];

function CapabilityIcons({ caps }: { caps: ModelCapabilities }) {
  return (
    <>
      {CAPABILITY_BADGES.filter((b) => caps[b.key]).map((b) => (
        <span
          key={b.key}
          title={b.title}
          aria-label={b.title}
          className="flex size-[18px] shrink-0 items-center justify-center rounded-md border border-border/60 text-muted-foreground"
        >
          <HugeiconsIcon icon={b.icon} className="size-3" strokeWidth={1.8} />
        </span>
      ))}
    </>
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
  optionProps,
  onArrowDownIntoChildren,
  capabilities,
  hideOwner,
  downloaded,
  showVision,
}: {
  label: string;
  meta?: string | null;
  selected?: boolean;
  onClick: () => void;
  vramStatus?: VramFitStatus | null;
  vramEst?: number;
  gpuGb?: number;
  tooltipText?: ReactNode;
  optionProps?: ModelRowOptionProps;
  onArrowDownIntoChildren?: () => boolean;
  /** Capability override (HF rows have tags); falls back to name detection. */
  capabilities?: ModelCapabilities;
  /** Hide the "owner/" prefix (e.g. Recommended, where all are unsloth). */
  hideOwner?: boolean;
  /** Mark a row already on disk (shown in Recommended instead of being hidden). */
  downloaded?: boolean;
  /** Show a Vision badge on the name (On Device, read from GGUF metadata). */
  showVision?: boolean;
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

  const { owner, name } = splitRepoLabel(label);
  const parsed = parseMetaTokens(meta);
  // Param chip from meta, else derived from the name so GGUF rows show it too.
  const paramLabel = parsed.param ?? extractParamLabel(name) ?? null;
  // Use the passed-in capabilities (tag-aware) or infer from the repo name.
  const caps = capabilities ?? detectCapabilities({ id: label });
  const showCaps = hasAnyCapability(caps);

  const content = (
    <button
      type="button"
      {...optionProps}
      onKeyDown={(event) => {
        if (event.key === "ArrowDown" && onArrowDownIntoChildren?.()) {
          event.preventDefault();
          return;
        }
        optionProps?.onKeyDown(event);
      }}
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] focus-visible:bg-[#ececec] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45 dark:hover:bg-[var(--sidebar-accent)] dark:focus-visible:bg-[var(--sidebar-accent)]",
        selected && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
      )}
    >
      <span className="flex min-w-0 flex-1 items-baseline">
        {owner && !hideOwner ? (
          <span className="inline-flex min-w-0 max-w-[45%] shrink items-baseline text-[13px] text-muted-foreground/90">
            <span className="truncate">{owner}</span>
            <span className="shrink-0 text-muted-foreground/45">/</span>
          </span>
        ) : null}
        <span className="min-w-0 flex-1 truncate">{name}</span>
      </span>
      <span className="ml-auto flex shrink-0 items-center gap-1.5">
        {showCaps && <CapabilityIcons caps={caps} />}
        {showVision && (
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild={true}>
              <span
                aria-label="Vision"
                className="flex h-[18px] shrink-0 items-center justify-center rounded-md border border-border/60 px-1.5 text-indigo-700 dark:text-indigo-300"
              >
                <HugeiconsIcon
                  icon={ViewIcon}
                  className="size-3"
                  strokeWidth={1.8}
                />
              </span>
            </TooltipTrigger>
            <TooltipContent side="top" className="tooltip-compact">
              This model can process image inputs
            </TooltipContent>
          </Tooltip>
        )}
        {selected && (
          <DotTag
            tone="success"
            label="Loaded"
            className="h-[18px] gap-1 rounded-md px-1.5"
            dotClassName="size-[5px]"
          />
        )}
        {downloaded && !selected && (
          <span
            title="Already downloaded"
            aria-label="Already downloaded"
            className="flex h-[18px] shrink-0 items-center justify-center rounded-md border border-border/60 px-1.5 text-muted-foreground"
          >
            <HugeiconsIcon
              icon={Download01Icon}
              className="size-3"
              strokeWidth={1.8}
            />
          </span>
        )}
        {vramStatus === "exceeds" && (
          <span className="text-[9px] font-medium !text-red-700 !bg-red-50 dark:!text-red-300 dark:!bg-red-500/15 px-1.5 py-0.5 rounded">
            OOM
          </span>
        )}
        {vramStatus === "tight" && (
          <span className="text-[9px] font-medium !text-amber-400">TIGHT</span>
        )}
        {paramLabel ? (
          <span className="rounded-md border border-border/60 px-1.5 py-px text-[10px] font-medium text-muted-foreground tabular-nums">
            {paramLabel}
          </span>
        ) : null}
        {parsed.formats.map((f) => (
          <DotTag
            key={f.label}
            tone={f.tone}
            label={f.label}
            className="h-[18px] gap-[3px] rounded-md px-1.5"
            dotClassName="size-[5px]"
          />
        ))}
        {parsed.texts.map((text) => (
          <span key={text} className="text-[10px] text-muted-foreground">
            {text}
          </span>
        ))}
        {parsed.size ? (
          <span className="text-[10px] text-muted-foreground tabular-nums">
            {parsed.size}
          </span>
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
  parentOptionKey,
  onNavigatePastStart,
  onNavigatePastEnd,
  onDeleteVariant,
  sourceOverride,
  deleteVariantTitle = "Delete cached model?",
  renderDeleteVariantDescription,
  getDeleteVariantSuccessMessage,
  deleteDisabled = false,
  onDevice = false,
  onHasVision,
}: {
  repoId: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  gpuGb?: number;
  systemRamGb?: number;
  parentOptionKey?: string;
  onNavigatePastStart?: () => void;
  onNavigatePastEnd?: () => void;
  onDeleteVariant?: (quant: string) => Promise<void> | void;
  sourceOverride?: ModelSelectorChangeMeta["source"];
  deleteVariantTitle?: string;
  renderDeleteVariantDescription?: (quant: string) => ReactNode;
  getDeleteVariantSuccessMessage?: (quant: string) => string;
  deleteDisabled?: boolean;
  /** On Device rows honor the Show all quantizations setting; Recommended and
   *  other browse lists always show every quant. */
  onDevice?: boolean;
  /** Report GGUF vision support up so the parent row can badge it. */
  onHasVision?: (hasVision: boolean) => void;
}) {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [defaultVariant, setDefaultVariant] = useState<string | null>(null);
  const [hasVision, setHasVision] = useState(false);
  // Native max context (GGUF metadata); only set once a variant is downloaded.
  const [nativeContext, setNativeContext] = useState<number | null>(null);
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
        onHasVision?.(res.has_vision);
        setNativeContext(res.context_length ?? null);
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
        contextLength: nativeContext,
      });
    },
    [repoId, isLocalPath, onSelect, sourceOverride, nativeContext],
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

  // If the recommended variant is OOM, pick the largest fitting one;
  // if all are OOM, recommend the smallest.
  const effectiveRecommended = useMemo(() => {
    if (!variants || !gpuGb || gpuGb <= 0) return defaultVariant;
    const defaultV = variants.find((v) => v.quant === defaultVariant);
    if (defaultV && getGgufFit(defaultV.size_bytes) !== "oom")
      return defaultVariant;
    // Largest non-OOM variant (best quality that fits)
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

  // On Device only: when Show all quantizations is off, list quants already on
  // disk. Recommended and other browse lists always show every quant.
  const showAllQuantizations = useChatRuntimeStore(
    (s) => s.showAllQuantizations,
  );
  const displayVariants = useMemo(() => {
    if (!sortedVariants) return sortedVariants;
    return showAllQuantizations || !onDevice
      ? sortedVariants
      : sortedVariants.filter((v) => v.downloaded);
  }, [sortedVariants, showAllQuantizations, onDevice]);

  const variantOptionKeys = useMemo(
    () =>
      (displayVariants ?? []).map((variant) =>
        makeModelOptionKey("gguf-variant", `${repoId}:${variant.filename}`),
      ),
    [repoId, displayVariants],
  );
  const variantList = useRovingModelList({
    label: `${repoId} quantizations`,
    optionKeys: variantOptionKeys,
    onNavigatePastStart,
    onNavigatePastEnd,
  });

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

  if (!displayVariants || displayVariants.length === 0) {
    return (
      <div className="px-5 py-2 text-xs text-muted-foreground">
        No GGUF variants found.
      </div>
    );
  }

  return (
    <div
      {...variantList.listboxProps}
      id={
        parentOptionKey
          ? makeModelOptionChildrenId(parentOptionKey)
          : variantList.listboxProps.id
      }
      className="pl-4 border-l-2 border-accent/50 ml-3 my-1"
    >
      {/* On Device shows the model name above, so the Quantizations heading is
          redundant; its Vision badge is relayed to the name instead. */}
      {!onDevice && (
        <div className="px-2 py-1 flex items-center gap-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Quantizations
          </span>
          {hasVision && (
            <span className="flex items-center gap-0.5 text-[9px] font-medium text-indigo-700 dark:text-indigo-300">
              <HugeiconsIcon
                icon={ViewIcon}
                className="size-3"
                strokeWidth={1.8}
              />
              Vision
            </span>
          )}
        </div>
      )}
      {displayVariants.map((v) => {
        const fit = getGgufFit(v.size_bytes);
        const oom = fit === "oom";
        const tight = fit === "tight";
        const keyBase = `${repoId}:${v.filename}`;
        const variantOptionKey = makeModelOptionKey("gguf-variant", keyBase);
        return (
          <div key={v.filename} className="flex items-center gap-0.5">
            <button
              type="button"
              {...variantList.getOptionProps(variantOptionKey, false)}
              onClick={() =>
                handleVariantClick(v.quant, v.downloaded, v.size_bytes)
              }
              className={cn(
                "flex min-w-0 flex-1 items-center justify-between gap-2 rounded-full px-2 py-1 text-left text-sm transition-colors hover:bg-[#ececec] focus-visible:bg-[#ececec] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45 dark:hover:bg-[var(--sidebar-accent)] dark:focus-visible:bg-[var(--sidebar-accent)]",
              )}
            >
              <span className="min-w-0 flex-1 truncate font-mono text-xs">
                <span
                  className={cn(oom && "!text-gray-500 dark:!text-gray-400")}
                >
                  {v.quant}
                </span>
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
                  <span className="text-[9px] font-medium !text-red-700 !bg-red-50 dark:!text-red-300 dark:!bg-red-500/15 px-1.5 py-0.5 rounded">
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
            {v.downloaded && (
              <ModelLoadSettingsAction
                ariaLabel={`Inference settings for ${repoId} ${v.quant}`}
                repoId={repoId}
                quant={v.quant}
                maxContext={nativeContext}
              />
            )}
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

// Module-level caches so re-mounting the popover shows results instantly
let _cachedGgufCache: CachedGgufRepo[] = [];
let _cachedModelsCache: CachedModelRepo[] = [];
let _lmStudioCache: LocalModelInfo[] = [];
let _localDirCache: LocalModelInfo[] = [];
let _customFolderCache: LocalModelInfo[] = [];
let _scanFoldersCache: ScanFolderInfo[] = [];

/** True when any on-device model (downloaded GGUF, cached repo, LM Studio, or
 * custom-folder model) is known. Reads the module caches, which persist across
 * popover mounts, so the selector can default to the On Device tab. */
export function hasDownloadedModels(): boolean {
  return (
    _cachedGgufCache.length > 0 ||
    _cachedModelsCache.length > 0 ||
    _lmStudioCache.length > 0 ||
    _localDirCache.length > 0 ||
    _customFolderCache.length > 0
  );
}

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

function canDeleteLoraModel(model: LoraModelOption): boolean {
  const isTraining = model.source === "training";
  const isExported = model.source === "exported";
  const isExportedGguf = isExported && model.exportType === "gguf";
  return (isTraining || isExported) && !isExportedGguf;
}

// ── Hub Model Picker ──────────────────────────────────────────

// Recommended section sort. "recommended" = newly created unsloth GGUF/MLX that
// fit the device; the rest are plain HF sort keys over all unsloth models.
type RecommendedSortKey = "recommended" | "trendingScore" | "lastModified";

const RECOMMENDED_SORT_OPTIONS: HubOption<RecommendedSortKey>[] = [
  { value: "recommended", label: "Recommended" },
  { value: "trendingScore", label: "Trending" },
  { value: "lastModified", label: "Recent" },
];

// Sort for the On Device / Custom (local) lists. "recent" = last loaded;
// "downloaded" = file download date.
type LocalSortKey = "recent" | "downloaded" | "size" | "name";

const LOCAL_SORT_OPTIONS: HubOption<LocalSortKey>[] = [
  { value: "recent", label: "Recent" },
  { value: "size", label: "Size" },
  { value: "name", label: "Name" },
  { value: "downloaded", label: "Downloaded" },
];

// Format filter dropdown for the Unsloth listing. Plain labels are reused in
// the empty-state copy below.
const FORMAT_FILTER_LABELS: Record<FormatFilter, string> = {
  all: "All",
  gguf: "GGUF",
  mlx: "MLX",
  safetensors: "Safetensors",
};

// Dot colors match the row format tags: gguf blue, mlx amber, safetensors pink.
const FORMAT_FILTER_DOTS: Partial<Record<FormatFilter, string>> = {
  gguf: "bg-format-gguf",
  mlx: "bg-format-mlx",
  safetensors: "bg-format-checkpoint",
};

const FORMAT_FILTER_OPTIONS: HubOption<FormatFilter>[] = (
  Object.keys(FORMAT_FILTER_LABELS) as FormatFilter[]
).map((value) => {
  const dot = FORMAT_FILTER_DOTS[value];
  return {
    value,
    label: dot ? (
      <span className="flex items-center gap-2">
        <span
          className={cn("inline-block size-1.5 shrink-0 rounded-full", dot)}
        />
        {FORMAT_FILTER_LABELS[value]}
      </span>
    ) : (
      FORMAT_FILTER_LABELS[value]
    ),
  };
});

/** Sort cached repos: by last-loaded, download date, size desc, or name. */
function sortCachedRepos<
  T extends { repo_id: string; size_bytes: number; last_modified?: number },
>(rows: T[], key: LocalSortKey, loadTimes: ModelLoadTimes): T[] {
  const byDate = (a: T, b: T) =>
    (b.last_modified ?? -1) - (a.last_modified ?? -1) ||
    a.repo_id.localeCompare(b.repo_id);
  return [...rows].sort((a, b) => {
    if (key === "name") return a.repo_id.localeCompare(b.repo_id);
    if (key === "size") {
      return b.size_bytes - a.size_bytes || a.repo_id.localeCompare(b.repo_id);
    }
    if (key === "recent") {
      const d = loadedAt(loadTimes, b.repo_id) - loadedAt(loadTimes, a.repo_id);
      return d !== 0 ? d : byDate(a, b);
    }
    return byDate(a, b); // "downloaded"
  });
}

/** Sort local-provider models. They carry no size, so "size" falls back to name. */
function sortLocalModels(
  rows: LocalModelInfo[],
  key: LocalSortKey,
  loadTimes: ModelLoadTimes,
): LocalModelInfo[] {
  const name = (m: LocalModelInfo) => m.model_id ?? m.display_name ?? m.id;
  const byDate = (a: LocalModelInfo, b: LocalModelInfo) =>
    (b.updated_at ?? -1) - (a.updated_at ?? -1) ||
    name(a).localeCompare(name(b));
  return [...rows].sort((a, b) => {
    if (key === "recent") {
      const d = loadedAt(loadTimes, a.id) - loadedAt(loadTimes, b.id);
      return d !== 0 ? -d : byDate(a, b);
    }
    if (key === "downloaded") return byDate(a, b);
    return name(a).localeCompare(name(b)); // "size" (no size) and "name"
  });
}

/** GGUF detection for a local model by name or file path. */
function localModelIsGguf(m: LocalModelInfo): boolean {
  return (
    isGgufRepo(m.id) ||
    isGgufRepo(m.display_name) ||
    m.path.toLowerCase().endsWith(".gguf")
  );
}

/** Whether a local model matches the format toggle (GGUF detected by name/path). */
function localModelMatchesFormat(
  m: LocalModelInfo,
  filter: FormatFilter,
): boolean {
  return matchesFormatFilter(
    m.model_id ?? m.display_name ?? m.id,
    localModelIsGguf(m),
    filter,
  );
}

export function HubModelPicker({
  models,
  loraModels = [],
  externalModels = [],
  value,
  onSelect,
  onFoldersChange,
  onBrowseHub,
  onModelsChange,
  deleteDisabled = false,
  section = "downloaded",
  sectionToggle,
  onEject,
}: {
  models: ModelOption[];
  /** Fine-tuned models, shown as a section in the On Device view. */
  loraModels?: LoraModelOption[];
  /** Connected provider models, shown in the Connected section. */
  externalModels?: ExternalModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onFoldersChange?: () => void;
  /** Open the full Hub page to browse more models. */
  onBrowseHub?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  /** Section shown when not searching. Search spans all sections. */
  section?: "downloaded" | "recommended" | "custom" | "connected";
  /** Section toggle rendered under the search bar. */
  sectionToggle?: ReactNode;
  /** Eject the loaded model. Rendered as the last list row when set. */
  onEject?: () => void;
}) {
  const gpu = useGpuInfo();
  // Last-loaded timestamps power the "Recent" sort (vs "Downloaded" = file date).
  const loadTimes = useModelLoadTimes(value);
  // Fade the list's top edge once scrolled, and its bottom edge while more
  // rows sit below the fold.
  const [listScrolled, setListScrolled] = useState(false);
  const [listMoreBelow, setListMoreBelow] = useState(false);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  // Shared Hub search stack (the same hooks the Hub page uses) so the picker
  // and Hub run one implementation. Scoped to unsloth like the old listing.
  const online = useOnlineStatus();
  const accessToken = useHfTokenStore((s) => s.token) || undefined;
  // Recommended section: a live unsloth listing sorted by the dropdown. The
  // same sort drives the search results so the dropdown works while searching.
  const [recommendedSort, setRecommendedSort] =
    useState<RecommendedSortKey>("trendingScore");
  // "recommended" surfaces the most recently created Unsloth repos.
  const recommendedSortBy: HfSortKey =
    recommendedSort === "recommended" ? "createdAt" : recommendedSort;
  const {
    results,
    isLoading,
    isLoadingMore,
    fetchMore,
    scannedCount,
    hasMore,
  } = useHubModelSearch(debouncedQuery, {
    ownerScope: "unsloth",
    sortBy: recommendedSortBy,
    sortDirection: "desc",
    pinUnslothFirst: true,
    keepUnsupportedTags: true,
    accessToken,
    enabled: online,
  });
  const recommendedSearch = useHubModelSearch("", {
    ownerScope: "unsloth",
    sortBy: recommendedSortBy,
    sortDirection: "desc",
    pinUnslothFirst: true,
    keepUnsupportedTags: true,
    accessToken,
    enabled: online,
  });

  // Lowercased repo ids confirmed GGUF by the store or HF search.
  // Absence means "no hint" -> hasGgufSuffix is the fallback (don't
  // conflate unknown with known-not-GGUF). Lowercased so store and HF
  // IDs differing only by casing match the same hint.
  const modelGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const model of models) {
      if (model.isGguf) ids.add(model.id.toLowerCase());
    }
    return ids;
  }, [models]);
  // Both listings contribute GGUF hints so a tag-only GGUF (no "-GGUF" suffix)
  // in Recommended still expands variants instead of loading as a checkpoint.
  const resultGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const result of [...results, ...recommendedSearch.results]) {
      if (result.isGguf) ids.add(result.id.toLowerCase());
    }
    return ids;
  }, [results, recommendedSearch.results]);
  const isKnownGgufRepo = useCallback(
    (id: string): boolean => {
      const key = id.toLowerCase();
      return isGgufRepo(id, resultGgufIds.has(key) || modelGgufIds.has(key));
    },
    [modelGgufIds, resultGgufIds],
  );

  // Track which GGUF repo is expanded for variant selection
  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  // GGUF vision support per repo, reported by the expander once it has read the
  // metadata, so On Device rows can show a Vision badge on the name.
  const [visionByRepo, setVisionByRepo] = useState<Record<string, boolean>>({});
  const reportVision = useCallback((repoId: string, hasVision: boolean) => {
    setVisionByRepo((prev) =>
      prev[repoId] === hasVision ? prev : { ...prev, [repoId]: hasVision },
    );
  }, []);
  // When on, On Device GGUF repos show their quantizations without a click.
  const expandQuantizations = useChatRuntimeStore((s) => s.expandQuantizations);
  // Repos the user clicked to collapse while expand-by-default is on. Kept in
  // memory only, so it resets on reload (and when the setting is toggled).
  const [collapsedGguf, setCollapsedGguf] = useState<Set<string>>(
    () => new Set(),
  );
  useEffect(() => {
    setCollapsedGguf(new Set());
  }, [expandQuantizations]);
  const isGgufExpanded = useCallback(
    (id: string) =>
      expandQuantizations ? !collapsedGguf.has(id) : expandedGguf === id,
    [expandQuantizations, collapsedGguf, expandedGguf],
  );
  // Toggle a repo's quantizations: flip the collapse set when expand-by-default
  // is on, otherwise drive the single-open expandedGguf state.
  const toggleGgufExpanded = useCallback(
    (id: string) => {
      if (expandQuantizations) {
        setCollapsedGguf((prev) => {
          const next = new Set(prev);
          if (next.has(id)) next.delete(id);
          else next.add(id);
          return next;
        });
      } else {
        setExpandedGguf((prev) => (prev === id ? null : id));
      }
    },
    [expandQuantizations],
  );

  const [downloadedCollapsed, setDownloadedCollapsed] = useState(false);
  const [otherModelsCollapsed, setOtherModelsCollapsed] = useState(false);
  const [customFoldersCollapsed, setCustomFoldersCollapsed] = useState(false);
  const [fineTunedCollapsed, setFineTunedCollapsed] = useState(false);
  const [lmStudioCollapsed, setLmStudioCollapsed] = useState(false);
  const [localDirCollapsed, setLocalDirCollapsed] = useState(false);
  // The Fine-tuned section header; the train icon on the Unsloth header scrolls
  // here so users can jump to their trained models.
  const fineTunedSectionRef = useRef<HTMLDivElement>(null);
  const scrollToFineTuned = useCallback(() => {
    setFineTunedCollapsed(false);
    // Two frames so the expand renders before we scroll the section to the top
    // of the list.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fineTunedSectionRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    });
  }, []);
  // The Custom Folders header; the folder icon on the Unsloth header scrolls
  // here instead of opening the browse popup.
  const customFolderSectionRef = useRef<HTMLDivElement>(null);
  const scrollToCustomFolders = useCallback(() => {
    setCustomFoldersCollapsed(false);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        customFolderSectionRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    });
  }, []);

  // Cached (downloaded) repos -- module-level cache avoids flashing an
  // empty "Downloaded" section when the popover re-mounts.
  const [cachedGguf, setCachedGguf] =
    useState<CachedGgufRepo[]>(_cachedGgufCache);
  const [cachedModels, setCachedModels] =
    useState<CachedModelRepo[]>(_cachedModelsCache);
  const alreadyCached =
    _cachedGgufCache.length > 0 || _cachedModelsCache.length > 0;
  const [cachedReady, setCachedReady] = useState(alreadyCached);

  // LM Studio local models -- module-level cache, same pattern as above.
  const [lmStudioModels, setLmStudioModels] =
    useState<LocalModelInfo[]>(_lmStudioCache);
  // Models found under the local models directory (./models), so they stay
  // selectable on the On Device tab after leaving the Fine-tuned tab.
  const [localDirModels, setLocalDirModels] =
    useState<LocalModelInfo[]>(_localDirCache);
  const [customFolderModels, setCustomFolderModels] =
    useState<LocalModelInfo[]>(_customFolderCache);

  // Custom scan folders management
  const [scanFolders, setScanFolders] =
    useState<ScanFolderInfo[]>(_scanFoldersCache);
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
        const ld = res.models.filter((m) => m.source === "models_dir");
        _localDirCache = ld;
        setLocalDirModels(ld);
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

  const handleAddFolder = useCallback(
    async (overridePath?: string) => {
      // Explicit path lets the folder browser submit in the same tick it
      // calls `setFolderInput`; reading `folderInput` would race the update.
      const raw = overridePath !== undefined ? overridePath : folderInput;
      const trimmed = raw.trim();
      if (!trimmed || folderLoading) return;
      setFolderError(null);
      setFolderLoading(true);
      // From the folder browser's one-click "Use this folder": the typed-
      // input panel is closed, so the inline folderError is invisible.
      // Surface failures (denylisted path, sandbox 403, etc.) via toast.
      const fromBrowser = overridePath !== undefined;
      try {
        const created = await addScanFolder(trimmed);
        // Backend returns the existing row for duplicates, so dedupe.
        const next = _scanFoldersCache.some(
          (f) => f.id === created.id || f.path === created.path,
        )
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
    },
    [
      folderInput,
      folderLoading,
      refreshScanFolders,
      refreshLocalModelsList,
      onFoldersChange,
    ],
  );

  const handleRemoveFolder = useCallback(
    async (id: number) => {
      try {
        await removeScanFolder(id);
        // Optimistic: drop it immediately.
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
    },
    [refreshScanFolders, refreshLocalModelsList, onFoldersChange],
  );

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
    // Always refresh LM Studio + custom folder models (not gated by alreadyCached).
    refreshLocalModelsList();
    refreshScanFolders();
    listRecommendedFolders()
      .then(setRecommendedFolders)
      .catch(() => {});

    // Always refetch cached GGUF/model lists. The module-level caches render
    // instantly with stale data (no spinner flash), but newly downloaded
    // repos need a fresh backend hit. cachedReady=alreadyCached initially,
    // so the background refresh is invisible when we already had data.
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

  // Hide downloaded models from the recommended list. Case-insensitive
  // since the HF cache lowercases repo IDs.
  const downloadedSet = useMemo(() => {
    const s = new Set<string>();
    for (const c of cachedGguf) s.add(c.repo_id.toLowerCase());
    for (const c of cachedModels) s.add(c.repo_id.toLowerCase());
    return s;
  }, [cachedGguf, cachedModels]);

  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const deviceType = usePlatformStore((s) => s.deviceType);
  const isMac = deviceType === "mac";

  // Drop models Studio can't run for chat (diffusion / image / video / etc.)
  // using the Hub's classifier on the tags the listing already carries.
  const isChatSupported = useCallback(
    (r: HfModelResult) =>
      classifyUnslothSupport({
        modelId: r.id,
        pipelineTag: r.pipelineTag,
        tags: r.tags,
        libraryName: r.libraryName,
        quantMethod: r.quantMethod,
        deviceType,
      }).status !== "unsupported",
    [deviceType],
  );

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

  const showHfSection = debouncedQuery.trim().length > 0;

  // Independent sort for each local section's inline dropdown.
  const [downloadedSort, setDownloadedSort] = useState<LocalSortKey>("recent");
  const [customSort, setCustomSort] = useState<LocalSortKey>("recent");
  // Format filter toggle for the Unsloth listing.
  const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");

  // Recommended only suggests GGUF (anywhere) and MLX (Mac only); safetensors
  // are never recommended. The "recommended" sort also drops models too big for
  // the device. Already-downloaded models stay visible (badged), never hidden.
  const recommendedRows = useMemo(() => {
    // Never list mobile-targeted builds in the Unsloth section.
    let rows = recommendedSearch.results.filter((r) => !isMobileVariant(r.id));
    // Drop models Studio can't run for chat (diffusion / image / video / etc.).
    rows = rows.filter(isChatSupported);
    // Keep only formats we recommend for this device.
    rows = rows.filter((r) => isRecommendableFormat(r.id, r.isGguf, isMac));
    // The format toggle narrows further.
    if (formatFilter !== "all") {
      rows = rows.filter((r) =>
        matchesFormatFilter(r.id, r.isGguf, formatFilter),
      );
    }
    if (recommendedSort !== "recommended") return rows;
    return rows.filter((r) => {
      // Downloaded models always show, regardless of device fit.
      if (downloadedSet.has(r.id.toLowerCase())) return true;
      if (!gpu.available) return true;
      // GGUF/MLX repos rarely expose safetensors metadata, so fall back to the
      // GGUF param count, then the repo name, for a size estimate. Anything we
      // still cannot size is hidden (requireKnown) so over-budget models like a
      // 1T GGUF don't slip into Recommended.
      const params = r.totalParams ?? paramsFromId(r.id);
      const sizeBytes =
        r.estimatedSizeBytes ??
        (params ? estimateQuantBytes(params) : undefined);
      return fitsDevice({
        sizeBytes,
        gpuGb: gpu.memoryTotalGb,
        systemRamGb: gpu.systemRamAvailableGb,
        requireKnown: true,
      });
    });
  }, [
    recommendedSearch.results,
    downloadedSet,
    recommendedSort,
    formatFilter,
    isMac,
    gpu,
    isChatSupported,
  ]);

  // Per-row meta + VRAM badge from the recommended listing's own metadata.
  const recommendedMeta = useMemo(() => {
    const map = new Map<
      string,
      { meta: string | null; status: VramFitStatus | null; est: number }
    >();
    for (const r of recommendedSearch.results) {
      const isG = isKnownGgufRepo(r.id);
      // GGUF param count comes from the repo name or the GGUF metadata, so even
      // repos with no "<n>B" token (Kimi, MiniMax) show a param chip.
      const ggufParams = r.totalParams ?? paramsFromId(r.id);
      const meta = isG
        ? [
            ggufParams ? formatCompact(ggufParams) : null,
            "GGUF",
            r.estimatedSizeBytes ? formatBytes(r.estimatedSizeBytes) : null,
          ]
            .filter(Boolean)
            .join(" · ")
        : [
            r.totalParams
              ? formatCompact(r.totalParams)
              : extractParamLabel(r.id),
            // MLX and safetensors get a format pill like GGUF.
            isMlxId(r.id) ? "MLX" : "Safetensors",
            r.estimatedSizeBytes ? formatBytes(r.estimatedSizeBytes) : null,
          ]
            .filter(Boolean)
            .join(" · ") || null;
      if (isG) {
        // GGUF fit is size-based: flag OOM when even the smallest quant we can
        // size exceeds the device budget. Repos we cannot size show no badge.
        const params = ggufParams;
        const sizeBytes =
          r.estimatedSizeBytes ??
          (params ? estimateQuantBytes(params) : undefined);
        const exceeds =
          gpu.available &&
          sizeBytes != null &&
          !fitsDevice({
            sizeBytes,
            gpuGb: gpu.memoryTotalGb,
            systemRamGb: gpu.systemRamAvailableGb,
          });
        map.set(r.id, {
          meta,
          status: exceeds ? "exceeds" : null,
          est: sizeBytes ? Math.round(sizeBytes / 1024 ** 3) : 0,
        });
        continue;
      }
      const est = r.totalParams
        ? estimateLoadingVram(r.totalParams, "qlora")
        : 0;
      const status =
        est > 0 && gpu.available ? checkVramFit(est, gpu.memoryTotalGb) : null;
      map.set(r.id, { meta, status, est });
    }
    return map;
  }, [recommendedSearch.results, isKnownGgufRepo, gpu]);

  // Tag-accurate capabilities keyed by repo id, pooled from both HF listings.
  // Rows look it up by id and fall back to name detection when absent.
  const capsById = useMemo(() => {
    const map = new Map<string, ModelCapabilities>();
    for (const r of [...results, ...recommendedSearch.results]) {
      if (map.has(r.id)) continue;
      map.set(
        r.id,
        detectCapabilities({
          id: r.id,
          tags: r.tags,
          pipelineTag: r.pipelineTag,
        }),
      );
    }
    return map;
  }, [results, recommendedSearch.results]);

  // Ordered by the On Device dropdown (recent/download date/size/name).
  const sortedCachedGguf = useMemo(
    () => sortCachedRepos(cachedGguf, downloadedSort, loadTimes),
    [cachedGguf, downloadedSort, loadTimes],
  );
  const sortedCachedModels = useMemo(
    () => sortCachedRepos(cachedModels, downloadedSort, loadTimes),
    [cachedModels, downloadedSort, loadTimes],
  );
  // Each local section's search is scoped to its own models (matched by name).
  const localQuery = normalizeForSearch(debouncedQuery.trim());
  const matchesLocalQuery = (m: LocalModelInfo) =>
    !localQuery ||
    normalizeForSearch(
      `${m.model_id ?? ""} ${m.display_name} ${m.id}`,
    ).includes(localQuery);
  const sortedLmStudio = useMemo(
    () =>
      sortLocalModels(
        lmStudioModels.filter(
          (m) =>
            localModelMatchesFormat(m, formatFilter) && matchesLocalQuery(m),
        ),
        downloadedSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lmStudioModels, downloadedSort, formatFilter, loadTimes, localQuery],
  );
  // Local ./models entries. Chat-only Studio runs GGUF/MLX only, so raw
  // checkpoints there are hidden (mirrors the cached non-GGUF rule).
  const sortedLocalDir = useMemo(
    () =>
      sortLocalModels(
        localDirModels.filter(
          (m) =>
            (!chatOnly || localModelIsGguf(m)) &&
            localModelMatchesFormat(m, formatFilter) &&
            matchesLocalQuery(m),
        ),
        downloadedSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      localDirModels,
      downloadedSort,
      formatFilter,
      loadTimes,
      localQuery,
      chatOnly,
    ],
  );
  const sortedCustomFolderModels = useMemo(
    () =>
      sortLocalModels(
        customFolderModels.filter(
          (m) =>
            localModelMatchesFormat(m, formatFilter) && matchesLocalQuery(m),
        ),
        customSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [customFolderModels, customSort, formatFilter, loadTimes, localQuery],
  );

  // Fine-tuned models for the On Device "Fine-tuned" section: flat, query-
  // filtered, newest first.
  const fineTunedRows = useMemo(() => {
    const needle = normalizeForSearch(debouncedQuery.trim());
    return loraModels
      .filter((m) => {
        const text = normalizeForSearch(
          `${m.name} ${m.baseModel ?? ""} ${m.id}`,
        );
        return !needle || text.includes(needle);
      })
      .slice()
      .sort((a, b) => {
        const aTime = a.updatedAt ?? -1;
        const bTime = b.updatedAt ?? -1;
        if (aTime !== bTime) return bTime - aTime;
        return a.name.localeCompare(b.name);
      });
  }, [loraModels, debouncedQuery]);

  // While searching, filter Downloaded by the query instead of hiding it, so a
  // downloaded model the user is searching for stays visible.
  const visibleCachedGguf = useMemo(() => {
    if (!showHfSection)
      return sortedCachedGguf.filter((c) =>
        matchesFormatFilter(c.repo_id, true, formatFilter),
      );
    const q = normalizeForSearch(debouncedQuery.trim());
    return sortedCachedGguf.filter((c) =>
      normalizeForSearch(c.repo_id).includes(q),
    );
  }, [sortedCachedGguf, showHfSection, debouncedQuery, formatFilter]);
  const visibleCachedModels = useMemo(() => {
    if (!showHfSection)
      return sortedCachedModels.filter((c) =>
        matchesFormatFilter(c.repo_id, false, formatFilter),
      );
    const q = normalizeForSearch(debouncedQuery.trim());
    return sortedCachedModels.filter((c) =>
      normalizeForSearch(c.repo_id).includes(q),
    );
  }, [sortedCachedModels, showHfSection, debouncedQuery, formatFilter]);

  // Non-GGUF cached rows are not shown in chat-only mode, so the empty-state
  // logic must use this (not visibleCachedModels) or the picker can go blank.
  const visibleCachedModelRows = chatOnly ? [] : visibleCachedModels;

  // Split downloaded models so non-Unsloth repos get their own "Other models"
  // section above Fine-tuned.
  const unslothCachedGguf = useMemo(
    () => visibleCachedGguf.filter((c) => isUnslothRepoId(c.repo_id)),
    [visibleCachedGguf],
  );
  const otherCachedGguf = useMemo(
    () => visibleCachedGguf.filter((c) => !isUnslothRepoId(c.repo_id)),
    [visibleCachedGguf],
  );
  const unslothCachedModelRows = useMemo(
    () => visibleCachedModelRows.filter((c) => isUnslothRepoId(c.repo_id)),
    [visibleCachedModelRows],
  );
  const otherCachedModelRows = useMemo(
    () => visibleCachedModelRows.filter((c) => !isUnslothRepoId(c.repo_id)),
    [visibleCachedModelRows],
  );

  // Recommended models that match the current search query
  const filteredRecommendedIds = useMemo(() => {
    if (!showHfSection) return [];
    const q = normalizeForSearch(debouncedQuery.trim());
    return recommendedIds
      .filter((id) => normalizeForSearch(id).includes(q))
      .filter((id) =>
        matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
      );
  }, [
    showHfSection,
    debouncedQuery,
    recommendedIds,
    formatFilter,
    isKnownGgufRepo,
  ]);

  // Param counts come straight off the unsloth listings the picker already
  // loaded, so no extra per-id fetch is needed for the VRAM badges.
  const recommendedParamCountById = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of [...results, ...recommendedSearch.results]) {
      if (r.totalParams) map.set(r.id, r.totalParams);
    }
    return map;
  }, [results, recommendedSearch.results]);

  const recommendedSet = useMemo(
    () => new Set(filteredRecommendedIds),
    [filteredRecommendedIds],
  );

  const hfIds = useMemo(() => {
    // Only the Unsloth tab searches the HF listing, and only Unsloth models.
    if (!showHfSection || section !== "recommended") return [];
    return results
      .filter(isChatSupported)
      .map((result) => result.id)
      .filter((id) => id.toLowerCase().startsWith("unsloth/"))
      .filter((id) => !recommendedSet.has(id))
      .filter((id) => !chatOnly || isKnownGgufRepo(id))
      .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id))
      .filter((id) =>
        matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
      );
  }, [
    recommendedSet,
    results,
    showHfSection,
    section,
    chatOnly,
    isKnownGgufRepo,
    isChatSupported,
    formatFilter,
  ]);

  const hubOptionKeys = useMemo(() => {
    const keys: string[] = [];

    // Downloaded (Unsloth) rows (query-filtered) on the On Device tab only.
    if (
      section === "downloaded" &&
      cachedReady &&
      !downloadedCollapsed &&
      (unslothCachedGguf.length > 0 || unslothCachedModelRows.length > 0)
    ) {
      keys.push(
        ...unslothCachedGguf.map((model) =>
          makeModelOptionKey("downloaded-gguf", model.repo_id),
        ),
      );
      keys.push(
        ...unslothCachedModelRows.map((model) =>
          makeModelOptionKey("downloaded-model", model.repo_id),
        ),
      );
    }

    // Unsloth-tab search keys (curated matches + HF unsloth results).
    if (showHfSection && section === "recommended") {
      keys.push(
        ...filteredRecommendedIds.map((id) =>
          makeModelOptionKey("search-recommended", id),
        ),
      );
      keys.push(...hfIds.map((id) => makeModelOptionKey("search-hf", id)));
      return keys;
    }

    // Other (non-Unsloth) downloaded rows sit just above Fine-tuned.
    if (
      section === "downloaded" &&
      cachedReady &&
      !otherModelsCollapsed &&
      (otherCachedGguf.length > 0 || otherCachedModelRows.length > 0)
    ) {
      keys.push(
        ...otherCachedGguf.map((model) =>
          makeModelOptionKey("downloaded-gguf", model.repo_id),
        ),
      );
      keys.push(
        ...otherCachedModelRows.map((model) =>
          makeModelOptionKey("downloaded-model", model.repo_id),
        ),
      );
    }

    // Fine-tuned models sit below downloaded, above custom folders.
    if (section === "downloaded" && !fineTunedCollapsed) {
      keys.push(...fineTunedRows.map((m) => makeModelOptionKey("lora", m.id)));
    }

    // Custom folders sit right below the downloaded models on On Device.
    if (section === "downloaded" && !customFoldersCollapsed) {
      keys.push(
        ...sortedCustomFolderModels.map((model) =>
          makeModelOptionKey("custom-folder", model.id),
        ),
      );
    }

    if (section === "downloaded" && !lmStudioCollapsed) {
      keys.push(
        ...sortedLmStudio.map((model) =>
          makeModelOptionKey("lm-studio", model.id),
        ),
      );
    }

    if (section === "downloaded" && !localDirCollapsed) {
      keys.push(
        ...sortedLocalDir.map((model) =>
          makeModelOptionKey("local-dir", model.id),
        ),
      );
    }

    if (section === "recommended") {
      keys.push(
        ...recommendedRows.map((r) => makeModelOptionKey("recommended", r.id)),
      );
    }

    return keys;
  }, [
    cachedReady,
    chatOnly,
    sortedCustomFolderModels,
    customFoldersCollapsed,
    downloadedCollapsed,
    fineTunedRows,
    fineTunedCollapsed,
    filteredRecommendedIds,
    hfIds,
    sortedLmStudio,
    lmStudioCollapsed,
    recommendedRows,
    section,
    showHfSection,
    sortedLocalDir,
    localDirCollapsed,
    unslothCachedGguf,
    unslothCachedModelRows,
    otherCachedGguf,
    otherCachedModelRows,
    otherModelsCollapsed,
  ]);

  const selectedHubOptionKey = useMemo(
    () =>
      value
        ? hubOptionKeys.find((optionKey) => optionKey.endsWith(`::${value}`))
        : undefined,
    [hubOptionKeys, value],
  );
  const hubModelList = useRovingModelList({
    label: "Hub models",
    optionKeys: hubOptionKeys,
    selectedOptionKey: selectedHubOptionKey,
  });

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
    for (const id of filteredRecommendedIds) {
      // GGUF fit is size-based and badged elsewhere; skip the qlora estimate.
      if (isKnownGgufRepo(id)) continue;
      const totalParams = recommendedParamCountById.get(id) ?? paramsFromId(id);
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
  }, [filteredRecommendedIds, recommendedParamCountById, isKnownGgufRepo, gpu]);

  const { scrollRef, sentinelRef } = useHubInfiniteScroll(
    fetchMore,
    scannedCount,
    {
      enabled: online && hasMore,
      isFetching: isLoading || isLoadingMore,
      resultCount: results.length,
      resetKey: debouncedQuery,
    },
  );

  // Recompute the top/bottom edge fades from the scroll position.
  const updateListFades = useCallback((el: HTMLDivElement) => {
    const scrolled = el.scrollTop > 0;
    setListScrolled((prev) => (prev === scrolled ? prev : scrolled));
    const moreBelow = el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setListMoreBelow((prev) => (prev === moreBelow ? prev : moreBelow));
  }, []);

  // Keep the fades in sync when rows are added, removed, or filtered.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    updateListFades(el);
    const observer = new ResizeObserver(() => updateListFades(el));
    observer.observe(el);
    if (el.firstElementChild) observer.observe(el.firstElementChild);
    return () => observer.disconnect();
  }, [scrollRef, updateListFades]);

  // Sentinel + IntersectionObserver for recommended infinite scroll. Re-running
  // on each loaded page (results length) re-attaches the observer so a heavily
  // filtered list keeps paging until the viewport fills or the listing ends;
  // fetchMore is a no-op while a page is in flight. Callback ref tracks mount.
  const [recommendedSentinel, setRecommendedSentinel] =
    useState<HTMLDivElement | null>(null);
  const recommendedSentinelRef = useCallback((node: HTMLDivElement | null) => {
    setRecommendedSentinel(node);
  }, []);
  useEffect(() => {
    if (!recommendedSentinel || !recommendedSearch.hasMore) return;
    const root = scrollRef.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) recommendedSearch.fetchMore();
      },
      { threshold: 0, root },
    );
    obs.observe(recommendedSentinel);
    return () => obs.disconnect();
  }, [
    recommendedSentinel,
    recommendedSearch.hasMore,
    recommendedSearch.fetchMore,
    recommendedSearch.results.length,
    scrollRef,
  ]);

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

  // On Device owns the downloaded and custom-folder models; the Unsloth tab
  // searches the HF listing (below). Both filter locally by the query.
  const showDownloaded = section === "downloaded";
  const showCustom = section === "downloaded";
  const showRecommendedSection = !showHfSection && section === "recommended";
  const downloadedEmpty =
    visibleCachedGguf.length === 0 &&
    visibleCachedModelRows.length === 0 &&
    sortedLmStudio.length === 0 &&
    sortedLocalDir.length === 0;

  // Sort dropdown shown inline to the right of the section toggle. Options
  // depend on the tab and stay visible while searching so results can be
  // sorted. Fixed width matching the Search Hub button so it and the format
  // dropdown always line up; text-xs matches that button too. The trigger label
  // clips (no ellipsis) when long; the open menu expands to show it in full.
  const sortTriggerClassName =
    "w-[110px] shrink-0 justify-between pr-2.5 !border-0 text-xs [&>span]:!text-clip";
  // Tighter menu like the Projects activity Select: less left/top padding and
  // text-xs to match the trigger. Keep the option's right padding so the
  // selected-item checkmark never overlaps the label.
  const sortMenuContentClassName =
    "!p-1 !rounded-[14px] [&_[role=option]]:!pl-2 [&_[role=option]]:!py-1.5 [&_[role=option]]:!text-xs [&_[role=option]]:!rounded-[10px]";
  const sectionSortDropdown =
    section === "recommended" ? (
      <HubOptionMenu
        value={recommendedSort}
        options={RECOMMENDED_SORT_OPTIONS}
        onValueChange={setRecommendedSort}
        ariaLabel="Sort Unsloth models"
        align="end"
        className={sortTriggerClassName}
        contentClassName={sortMenuContentClassName}
      />
    ) : section === "downloaded" ? (
      <HubOptionMenu
        value={downloadedSort}
        options={LOCAL_SORT_OPTIONS}
        onValueChange={setDownloadedSort}
        ariaLabel="Sort downloaded models"
        align="end"
        className={sortTriggerClassName}
        contentClassName={sortMenuContentClassName}
      />
    ) : (
      <HubOptionMenu
        value={customSort}
        options={LOCAL_SORT_OPTIONS}
        onValueChange={setCustomSort}
        ariaLabel="Sort custom models"
        align="end"
        className={sortTriggerClassName}
        contentClassName={sortMenuContentClassName}
      />
    );

  // Connected models grouped by provider, filtered by the shared search query.
  const connectedGroups = useMemo(() => {
    const needle = normalizeForSearch(debouncedQuery.trim());
    const byProvider = new Map<
      string,
      {
        providerId: string;
        providerName: string;
        providerType: string;
        models: ExternalModelOption[];
      }
    >();
    for (const model of externalModels) {
      const text = normalizeForSearch(
        `${model.name} ${model.providerName} ${model.id}`,
      );
      if (needle && !text.includes(needle)) continue;
      const prev = byProvider.get(model.providerId);
      if (prev) {
        prev.models.push(model);
      } else {
        byProvider.set(model.providerId, {
          providerId: model.providerId,
          providerName: model.providerName,
          providerType: model.providerType,
          models: [model],
        });
      }
    }
    return [...byProvider.values()]
      .map((group) => ({
        ...group,
        models: group.models.sort((a, b) => a.name.localeCompare(b.name)),
      }))
      .sort((a, b) => a.providerName.localeCompare(b.providerName));
  }, [externalModels, debouncedQuery]);
  const showConnected = section === "connected";
  // With a Connected tab present the box is wider, so right-align the dropdowns
  // and drop the search inset to keep Search Hub on the last dropdown's edge.
  const hasConnected = externalModels.length > 0;

  // Shared row renderers so Downloaded (Unsloth) and Other models render alike.
  const renderDownloadedGgufRow = (c: (typeof visibleCachedGguf)[number]) => {
    const optionKey = makeModelOptionKey("downloaded-gguf", c.repo_id);
    return (
      <div key={c.repo_id}>
        <ModelRow
          label={c.repo_id}
          meta="GGUF"
          showVision={c.has_vision ?? visionByRepo[c.repo_id]}
          selected={value === c.repo_id}
          optionProps={hubModelList.getOptionProps(
            optionKey,
            value === c.repo_id,
          )}
          onClick={() => toggleGgufExpanded(c.repo_id)}
          onArrowDownIntoChildren={
            isGgufExpanded(c.repo_id)
              ? () => focusFirstChildOption(optionKey)
              : undefined
          }
          vramStatus={null}
        />
        {isGgufExpanded(c.repo_id) && (
          <GgufVariantExpander
            repoId={c.repo_id}
            onDevice={true}
            onHasVision={(v) => reportVision(c.repo_id, v)}
            onSelect={onSelect}
            parentOptionKey={optionKey}
            onNavigatePastStart={() => hubModelList.focusOption(optionKey)}
            onNavigatePastEnd={() => hubModelList.moveFocus(optionKey, "next")}
            gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
            systemRamGb={gpu.available ? gpu.systemRamAvailableGb : undefined}
            onDeleteVariant={async (quant) => {
              await deleteCachedModel(c.repo_id, quant);
              refreshCachedLists();
            }}
          />
        )}
      </div>
    );
  };
  const renderDownloadedModelRow = (
    c: (typeof visibleCachedModelRows)[number],
  ) => {
    const optionKey = makeModelOptionKey("downloaded-model", c.repo_id);
    return (
      <div key={c.repo_id} className="flex items-center gap-0.5">
        <div className="min-w-0 flex-1">
          <ModelRow
            label={c.repo_id}
            meta={formatBytes(c.size_bytes)}
            selected={value === c.repo_id}
            optionProps={hubModelList.getOptionProps(
              optionKey,
              value === c.repo_id,
            )}
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
              <span className="font-medium text-foreground">{c.repo_id}</span>{" "}
              from disk. You can re-download it later.
            </>
          }
          successMessage={`Deleted ${c.repo_id}`}
          onConfirm={() => deleteCachedModel(c.repo_id)}
          onDeleted={refreshCachedLists}
        />
      </div>
    );
  };

  return (
    <div className="relative space-y-2">
      {/* Inset the right so Search Hub lands on the Trending dropdown's edge.
          The wider Connected layout right-aligns the dropdowns, so drop it. */}
      <div
        className={cn("flex items-center gap-2 pb-1", !hasConnected && "pr-2")}
      >
        <div className="relative flex-1">
          <HugeiconsIcon
            icon={Search01Icon}
            className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground"
          />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search models"
            data-model-picker-search-input={true}
            className="field-soft h-9 border-0 pl-8 pr-8"
          />
          {isLoading && (
            <Spinner className="pointer-events-none absolute right-2.5 top-2.5 size-4 text-muted-foreground" />
          )}
        </div>
        {onBrowseHub ? (
          <button
            type="button"
            onClick={onBrowseHub}
            aria-label="Search more models on the Hub"
            className="hub-tab-toggle-pill flex h-9 w-[110px] shrink-0 items-center justify-center gap-[5px] rounded-full border-0 text-xs text-foreground transition-colors"
          >
            <HugeiconsIcon icon={DashboardCircleIcon} className="size-4" />
            Search Hub
          </button>
        ) : null}
      </div>

      {/* Section tabs then the format and sort dropdowns. Normally packed left
          with one uniform gap; the wider Connected layout right-aligns them so
          Trending meets Search Hub. The dropdowns hide on the Connected view. */}
      <div
        className={cn(
          "flex items-center gap-2",
          hasConnected && "justify-between",
        )}
      >
        {sectionToggle}
        {showConnected ? null : (
          <div className="flex items-center gap-2">
            <HubOptionMenu
              value={formatFilter}
              options={FORMAT_FILTER_OPTIONS}
              onValueChange={setFormatFilter}
              ariaLabel="Filter by format"
              align="end"
              className={sortTriggerClassName}
              contentClassName={sortMenuContentClassName}
            />
            {sectionSortDropdown}
          </div>
        )}
      </div>

      <div
        ref={scrollRef}
        onScroll={(e) => updateListFades(e.currentTarget)}
        className={cn(
          // List sits within the menu padding so left and right gaps match.
          // Height tracks the content up to the cap, so short lists do not
          // leave white space.
          "model-list-scroll max-h-[21rem] overflow-y-auto pr-0.5 mr-1",
          listScrolled && "is-scrolled",
          listMoreBelow && "is-bottom-faded",
        )}
        {...hubModelList.listboxProps}
      >
        <div className="py-1 pr-0">
          {showConnected ? (
            connectedGroups.length === 0 ? (
              <div className="px-2.5 py-2 text-xs leading-relaxed text-muted-foreground">
                {externalModels.length === 0
                  ? "No models from your connections. Set up in Settings then Connections."
                  : "No models match your search."}
              </div>
            ) : (
              connectedGroups.map((group) => (
                <div key={group.providerId}>
                  <div className="flex items-center gap-2 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    <ApiProviderLogo
                      providerType={group.providerType}
                      className="size-3.5"
                      title={group.providerName}
                    />
                    <span className="min-w-0 truncate">
                      {group.providerName}
                    </span>
                  </div>
                  {group.models.map((model) => (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() =>
                        onSelect(model.id, {
                          source: "external",
                          isLora: false,
                        })
                      }
                      className={cn(
                        "flex w-full items-center rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent",
                        value === model.id && "bg-accent/60",
                      )}
                    >
                      <span className="min-w-0 truncate">{model.name}</span>
                    </button>
                  ))}
                </div>
              ))
            )
          ) : (
            <>
              {/* First-load spinner only when nothing cached is shown yet. */}
              {showDownloaded &&
              !cachedReady &&
              !showHfSection &&
              downloadedEmpty ? (
                <div className="flex items-center gap-2 px-5 py-3">
                  <Spinner className="size-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">
                    Loading models…
                  </span>
                </div>
              ) : null}

              {/* Empty On Device: a search miss vs nothing downloaded yet. Hidden
              when custom folders below still have matches. */}
              {showDownloaded &&
              cachedReady &&
              downloadedEmpty &&
              sortedCustomFolderModels.length === 0 ? (
                <div className="px-2.5 py-2 text-xs text-muted-foreground">
                  {showHfSection
                    ? "No matching models on device."
                    : formatFilter === "all"
                      ? "No downloaded models yet. Search above or pick Recommended."
                      : `No downloaded ${FORMAT_FILTER_LABELS[formatFilter]} models yet.`}
                </div>
              ) : null}

              {/* Downloaded (Unsloth) stays visible (filtered) while searching. */}
              {showDownloaded &&
              (unslothCachedGguf.length > 0 ||
                unslothCachedModelRows.length > 0) ? (
                <>
                  <ListLabel
                    collapsed={downloadedCollapsed}
                    onToggle={() => setDownloadedCollapsed((v) => !v)}
                    action={
                      <>
                        <Tooltip delayDuration={0}>
                          <TooltipTrigger asChild={true}>
                            <button
                              type="button"
                              onClick={scrollToFineTuned}
                              aria-label="Go to fine-tuned models"
                              className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                            >
                              <HugeiconsIcon
                                icon={TrainIcon}
                                className="size-3"
                              />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent
                            side="bottom"
                            className="tooltip-compact"
                          >
                            Go to fine-tuned models
                          </TooltipContent>
                        </Tooltip>
                        <Tooltip delayDuration={0}>
                          <TooltipTrigger asChild={true}>
                            <button
                              type="button"
                              onClick={scrollToCustomFolders}
                              aria-label="Go to custom folders"
                              className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                            >
                              <HugeiconsIcon
                                icon={Folder02Icon}
                                className="size-3"
                              />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent
                            side="bottom"
                            className="tooltip-compact"
                          >
                            Go to custom folders
                          </TooltipContent>
                        </Tooltip>
                      </>
                    }
                  >
                    {/* When other providers (LM Studio/Ollama) also show here, name
                    this group "Unsloth" so the two are easy to tell apart. */}
                    {sortedLmStudio.length > 0 ? "Unsloth" : "Downloaded"}
                  </ListLabel>
                  {!downloadedCollapsed &&
                    unslothCachedGguf.map(renderDownloadedGgufRow)}
                  {!downloadedCollapsed &&
                    unslothCachedModelRows.map(renderDownloadedModelRow)}
                </>
              ) : null}

              {/* Other models: non-Unsloth downloads, grouped just above
              Fine-tuned. */}
              {showDownloaded &&
              (otherCachedGguf.length > 0 ||
                otherCachedModelRows.length > 0) ? (
                <>
                  <ListLabel
                    collapsed={otherModelsCollapsed}
                    onToggle={() => setOtherModelsCollapsed((v) => !v)}
                  >
                    Other models
                  </ListLabel>
                  {!otherModelsCollapsed &&
                    otherCachedGguf.map(renderDownloadedGgufRow)}
                  {!otherModelsCollapsed &&
                    otherCachedModelRows.map(renderDownloadedModelRow)}
                </>
              ) : null}

              {/* Fine-tuned models: a section above Custom Folders. Always shown on
              On Device so the train shortcut always has a target, with an empty
              state when none exist. */}
              {section === "downloaded" ? (
                <>
                  <div
                    ref={fineTunedSectionRef}
                    className="flex items-center gap-1 px-2.5 pb-1.5 pt-3"
                  >
                    <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      <HugeiconsIcon icon={TrainIcon} className="size-3.5" />
                      Fine-tuned
                    </span>
                    <div className="ml-auto">
                      <button
                        type="button"
                        aria-label={
                          fineTunedCollapsed
                            ? "Expand fine-tuned models"
                            : "Collapse fine-tuned models"
                        }
                        title={fineTunedCollapsed ? "Expand" : "Collapse"}
                        onClick={() => setFineTunedCollapsed((v) => !v)}
                        className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                      >
                        {fineTunedCollapsed ? (
                          <ChevronRightIcon className="size-3" />
                        ) : (
                          <ChevronDownIcon className="size-3" />
                        )}
                      </button>
                    </div>
                  </div>
                  {!fineTunedCollapsed && fineTunedRows.length > 0 && (
                    <FineTunedRows
                      adapters={fineTunedRows}
                      value={value}
                      onSelect={onSelect}
                      onModelsChange={onModelsChange}
                      deleteDisabled={deleteDisabled}
                      loraModelList={hubModelList}
                      expandedGguf={expandedGguf}
                      setExpandedGguf={setExpandedGguf}
                      gpu={gpu}
                    />
                  )}
                </>
              ) : null}

              {showCustom ? (
                <>
                  <div
                    ref={customFolderSectionRef}
                    className="flex items-center gap-1 px-2.5 pb-1.5 pt-3"
                  >
                    <button
                      type="button"
                      onClick={() => setShowFolderBrowser(true)}
                      title="Browse folders on the server"
                      className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground"
                    >
                      <HugeiconsIcon icon={Folder02Icon} className="size-3.5" />
                      Custom Folders
                    </button>
                    <div className="flex items-center gap-0.5">
                      <button
                        type="button"
                        aria-label={
                          showFolderInput
                            ? "Cancel adding folder"
                            : "Add scan folder by path"
                        }
                        title={
                          showFolderInput ? "Cancel" : "Add by typing a path"
                        }
                        onClick={() => {
                          setShowFolderInput((open) => {
                            if (open) {
                              setFolderInput("");
                              setFolderError(null);
                            }
                            return !open;
                          });
                        }}
                        className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={showFolderInput ? Cancel01Icon : Add01Icon}
                          className="size-3"
                        />
                      </button>
                      <button
                        type="button"
                        aria-label="Browse for a folder on the server"
                        title="Browse folders on the server"
                        onClick={() => setShowFolderBrowser(true)}
                        className="shrink-0 rounded p-0.5 text-muted-foreground/60 transition-colors hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={Search01Icon}
                          className="size-2.5"
                        />
                      </button>
                    </div>
                    <div className="ml-auto">
                      <button
                        type="button"
                        aria-label={
                          customFoldersCollapsed
                            ? "Expand custom folders"
                            : "Collapse custom folders"
                        }
                        title={customFoldersCollapsed ? "Expand" : "Collapse"}
                        onClick={() => setCustomFoldersCollapsed((v) => !v)}
                        className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                      >
                        {customFoldersCollapsed ? (
                          <ChevronRightIcon className="size-3" />
                        ) : (
                          <ChevronDownIcon className="size-3" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Folder paths */}
                  {!customFoldersCollapsed &&
                    scanFolders.map((f) => (
                      <div
                        key={f.id}
                        className="group flex items-center gap-1.5 px-2.5 py-0.5"
                      >
                        <HugeiconsIcon
                          icon={Folder02Icon}
                          className="size-3 shrink-0 text-muted-foreground/40"
                        />
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
                          <HugeiconsIcon
                            icon={Cancel01Icon}
                            className="size-3"
                          />
                        </button>
                      </div>
                    ))}

                  {/* Recommended folders */}
                  {!customFoldersCollapsed &&
                    (() => {
                      const registered = new Set(
                        scanFolders.map((f) => f.path),
                      );
                      const unregistered = recommendedFolders.filter(
                        (p) => !registered.has(p),
                      );
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
                              <span className="text-[11px] font-semibold">
                                +
                              </span>{" "}
                              {p.length > 30 ? `...${p.slice(-27)}` : p}
                            </button>
                          ))}
                        </div>
                      );
                    })()}

                  {/* Add folder input */}
                  {!customFoldersCollapsed && showFolderInput && (
                    <div className="px-2.5 pb-1 pt-0.5">
                      <div className="flex items-center gap-1">
                        <HugeiconsIcon
                          icon={Folder02Icon}
                          className="size-3 shrink-0 text-muted-foreground/40"
                        />
                        <input
                          value={folderInput}
                          onChange={(e) => {
                            setFolderInput(e.target.value);
                            setFolderError(null);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              handleAddFolder();
                            }
                            if (e.key === "Escape") {
                              e.preventDefault();
                              e.stopPropagation();
                              setShowFolderInput(false);
                              setFolderInput("");
                              setFolderError(null);
                            }
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
                          <HugeiconsIcon
                            icon={Search01Icon}
                            className="size-3"
                          />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            void handleAddFolder();
                          }}
                          disabled={folderLoading || !folderInput.trim()}
                          className="h-6 shrink-0 rounded border border-border/50 px-1.5 text-[10px] text-muted-foreground transition-colors hover:bg-accent disabled:opacity-40"
                        >
                          Add
                        </button>
                      </div>
                      {folderError && (
                        <p className="px-0.5 pt-0.5 text-[10px] text-destructive">
                          {folderError}
                        </p>
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
                      // Pass the path explicitly: `folderInput` state hasn't
                      // flushed yet when "Use this folder" submits.
                      void handleAddFolder(picked);
                    }}
                  />

                  {/* Models from custom folders */}
                  {!customFoldersCollapsed &&
                    sortedCustomFolderModels.map((m) => {
                      const isGgufFile = m.path.toLowerCase().endsWith(".gguf");
                      const isGguf =
                        isGgufFile ||
                        isGgufRepo(m.id) ||
                        isGgufRepo(m.display_name);
                      // Single .gguf files (e.g. Ollama blobs) load directly;
                      // GGUF repos/directories expand to pick a variant.
                      const isDirectGguf = isGgufFile;
                      const optionKey = makeModelOptionKey(
                        "custom-folder",
                        m.id,
                      );
                      return (
                        <div key={m.id}>
                          <ModelRow
                            label={m.model_id ?? m.display_name}
                            meta={isGguf ? "GGUF" : "Local"}
                            selected={value === m.id}
                            optionProps={hubModelList.getOptionProps(
                              optionKey,
                              value === m.id,
                            )}
                            onClick={() => {
                              if (isDirectGguf) {
                                onSelect(m.id, {
                                  source: "local",
                                  isLora: false,
                                  isDownloaded: true,
                                });
                              } else if (isGguf) {
                                toggleGgufExpanded(m.id);
                              } else {
                                onSelect(m.id, {
                                  source: "local",
                                  isLora: false,
                                  isDownloaded: true,
                                });
                              }
                            }}
                            onArrowDownIntoChildren={
                              isGgufExpanded(m.id)
                                ? () => {
                                    const focused =
                                      focusFirstChildOption(optionKey);
                                    return focused;
                                  }
                                : undefined
                            }
                            vramStatus={null}
                          />
                          {isGgufExpanded(m.id) && (
                            <GgufVariantExpander
                              repoId={m.id}
                              onDevice={true}
                              onSelect={onSelect}
                              parentOptionKey={optionKey}
                              onNavigatePastStart={() =>
                                hubModelList.focusOption(optionKey)
                              }
                              onNavigatePastEnd={() =>
                                hubModelList.moveFocus(optionKey, "next")
                              }
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
                  {!customFoldersCollapsed &&
                  showHfSection &&
                  sortedCustomFolderModels.length === 0 ? (
                    <div className="px-2.5 py-2 text-xs text-muted-foreground">
                      No matching models in custom folders.
                    </div>
                  ) : null}
                </>
              ) : null}

              {section === "downloaded" && sortedLmStudio.length > 0 ? (
                <>
                  <ListLabel
                    collapsed={lmStudioCollapsed}
                    onToggle={() => setLmStudioCollapsed((v) => !v)}
                  >
                    LM Studio
                  </ListLabel>
                  {!lmStudioCollapsed &&
                    sortedLmStudio.map((m) => {
                      const isGgufFile = m.path.toLowerCase().endsWith(".gguf");
                      const isGguf =
                        isGgufRepo(m.id) || isGgufRepo(m.display_name);
                      const optionKey = makeModelOptionKey("lm-studio", m.id);
                      return (
                        <div key={m.id}>
                          <ModelRow
                            label={m.model_id ?? m.display_name}
                            meta={isGguf || isGgufFile ? "GGUF" : "Local"}
                            selected={value === m.id}
                            optionProps={hubModelList.getOptionProps(
                              optionKey,
                              value === m.id,
                            )}
                            onClick={() => {
                              if (isGguf) {
                                toggleGgufExpanded(m.id);
                              } else {
                                onSelect(m.id, {
                                  source: "local",
                                  isLora: false,
                                  isDownloaded: true,
                                  isGguf: isGgufFile,
                                });
                              }
                            }}
                            onArrowDownIntoChildren={
                              isGgufExpanded(m.id)
                                ? () => {
                                    const focused =
                                      focusFirstChildOption(optionKey);
                                    return focused;
                                  }
                                : undefined
                            }
                            vramStatus={null}
                          />
                          {isGgufExpanded(m.id) && (
                            <GgufVariantExpander
                              repoId={m.id}
                              onDevice={true}
                              onSelect={onSelect}
                              parentOptionKey={optionKey}
                              onNavigatePastStart={() =>
                                hubModelList.focusOption(optionKey)
                              }
                              onNavigatePastEnd={() =>
                                hubModelList.moveFocus(optionKey, "next")
                              }
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
              ) : null}

              {section === "downloaded" && sortedLocalDir.length > 0 ? (
                <>
                  <ListLabel
                    collapsed={localDirCollapsed}
                    onToggle={() => setLocalDirCollapsed((v) => !v)}
                  >
                    Local models
                  </ListLabel>
                  {!localDirCollapsed &&
                    sortedLocalDir.map((m) => {
                      const isGguf = localModelIsGguf(m);
                      const optionKey = makeModelOptionKey("local-dir", m.id);
                      return (
                        <div key={m.id}>
                          <ModelRow
                            label={m.model_id ?? m.display_name}
                            meta={isGguf ? "GGUF" : "Local"}
                            selected={value === m.id}
                            optionProps={hubModelList.getOptionProps(
                              optionKey,
                              value === m.id,
                            )}
                            onClick={() => {
                              if (isGguf) {
                                toggleGgufExpanded(m.id);
                              } else {
                                onSelect(m.id, {
                                  source: "local",
                                  isLora: false,
                                  isDownloaded: true,
                                });
                              }
                            }}
                            onArrowDownIntoChildren={
                              isGgufExpanded(m.id)
                                ? () => focusFirstChildOption(optionKey)
                                : undefined
                            }
                            vramStatus={null}
                          />
                          {isGgufExpanded(m.id) && (
                            <GgufVariantExpander
                              repoId={m.id}
                              onDevice={true}
                              onSelect={onSelect}
                              parentOptionKey={optionKey}
                              onNavigatePastStart={() =>
                                hubModelList.focusOption(optionKey)
                              }
                              onNavigatePastEnd={() =>
                                hubModelList.moveFocus(optionKey, "next")
                              }
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
              ) : null}

              {showRecommendedSection ? (
                <>
                  {recommendedSearch.isLoading &&
                  recommendedRows.length === 0 ? (
                    <div className="flex items-center gap-2 px-5 py-3">
                      <Spinner className="size-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        Loading models…
                      </span>
                    </div>
                  ) : recommendedRows.length === 0 ? (
                    <div className="px-2.5 py-2 text-xs text-muted-foreground">
                      No models found.
                    </div>
                  ) : (
                    recommendedRows.map((r) => {
                      const id = r.id;
                      const info = recommendedMeta.get(id);
                      const isG = isKnownGgufRepo(id);
                      const optionKey = makeModelOptionKey("recommended", id);
                      return (
                        <div key={id}>
                          <ModelRow
                            label={id}
                            hideOwner={true}
                            downloaded={downloadedSet.has(id.toLowerCase())}
                            capabilities={capsById.get(id)}
                            meta={
                              info?.meta ??
                              (isG ? "GGUF" : extractParamLabel(id))
                            }
                            selected={value === id}
                            optionProps={hubModelList.getOptionProps(
                              optionKey,
                              value === id,
                            )}
                            onClick={() => {
                              if (isG) {
                                setExpandedGguf((prev) =>
                                  prev === id ? null : id,
                                );
                              } else {
                                handleModelClick(id);
                              }
                            }}
                            vramStatus={info?.status ?? null}
                            vramEst={info?.est}
                            gpuGb={
                              gpu.available ? gpu.memoryTotalGb : undefined
                            }
                            onArrowDownIntoChildren={
                              expandedGguf === id
                                ? () => focusFirstChildOption(optionKey)
                                : undefined
                            }
                          />
                          {expandedGguf === id && (
                            <GgufVariantExpander
                              repoId={id}
                              onSelect={onSelect}
                              parentOptionKey={optionKey}
                              onNavigatePastStart={() =>
                                hubModelList.focusOption(optionKey)
                              }
                              onNavigatePastEnd={() =>
                                hubModelList.moveFocus(optionKey, "next")
                              }
                              gpuGb={
                                gpu.available ? gpu.memoryTotalGb : undefined
                              }
                              systemRamGb={
                                gpu.available
                                  ? gpu.systemRamAvailableGb
                                  : undefined
                              }
                              onDeleteVariant={async (quant) => {
                                await deleteCachedModel(id, quant);
                                refreshCachedLists();
                              }}
                            />
                          )}
                        </div>
                      );
                    })
                  )}
                  {recommendedSearch.hasMore && (
                    <>
                      <div ref={recommendedSentinelRef} className="h-px" />
                      <div className="flex items-center justify-center py-2">
                        <Spinner className="size-3.5 text-muted-foreground" />
                      </div>
                    </>
                  )}
                </>
              ) : null}

              {showHfSection &&
              section === "recommended" &&
              filteredRecommendedIds.length > 0 ? (
                <>
                  {filteredRecommendedIds.map((id) => {
                    const vram = recommendedVramMap.get(id);
                    const optionKey = makeModelOptionKey(
                      "search-recommended",
                      id,
                    );
                    return (
                      <div key={id}>
                        <ModelRow
                          label={id}
                          capabilities={capsById.get(id)}
                          meta={
                            isKnownGgufRepo(id)
                              ? "GGUF"
                              : (vram?.detail ?? extractParamLabel(id))
                          }
                          selected={value === id}
                          optionProps={hubModelList.getOptionProps(
                            optionKey,
                            value === id,
                          )}
                          onClick={() => {
                            if (isKnownGgufRepo(id)) {
                              setExpandedGguf((prev) =>
                                prev === id ? null : id,
                              );
                            } else {
                              handleModelClick(id);
                            }
                          }}
                          vramStatus={
                            isKnownGgufRepo(id) ? null : (vram?.status ?? null)
                          }
                          vramEst={isKnownGgufRepo(id) ? undefined : vram?.est}
                          gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                          onArrowDownIntoChildren={
                            expandedGguf === id
                              ? () => {
                                  const focused =
                                    focusFirstChildOption(optionKey);
                                  return focused;
                                }
                              : undefined
                          }
                        />
                        {expandedGguf === id && (
                          <GgufVariantExpander
                            repoId={id}
                            onSelect={onSelect}
                            parentOptionKey={optionKey}
                            onNavigatePastStart={() =>
                              hubModelList.focusOption(optionKey)
                            }
                            onNavigatePastEnd={() =>
                              hubModelList.moveFocus(optionKey, "next")
                            }
                            gpuGb={
                              gpu.available ? gpu.memoryTotalGb : undefined
                            }
                            systemRamGb={
                              gpu.available
                                ? gpu.systemRamAvailableGb
                                : undefined
                            }
                            onDeleteVariant={async (quant) => {
                              await deleteCachedModel(id, quant);
                              refreshCachedLists();
                            }}
                          />
                        )}
                      </div>
                    );
                  })}
                </>
              ) : null}

              {showHfSection && section === "recommended" ? (
                <>
                  {hfIds.length === 0 && !isLoading ? (
                    filteredRecommendedIds.length === 0 ? (
                      <div className="px-2.5 py-2 text-xs text-muted-foreground">
                        No matching Unsloth models.
                      </div>
                    ) : null
                  ) : (
                    hfIds.map((id) => {
                      const vram = vramMap.get(id);
                      const isSearchGguf = isKnownGgufRepo(id);
                      const optionKey = makeModelOptionKey("search-hf", id);
                      return (
                        <div key={id}>
                          <ModelRow
                            label={id}
                            capabilities={capsById.get(id)}
                            meta={
                              isSearchGguf
                                ? "GGUF"
                                : [
                                    metricsById.get(id) ??
                                      extractParamLabel(id),
                                    isMlxId(id) ? "MLX" : "Safetensors",
                                  ]
                                    .filter(Boolean)
                                    .join(" · ")
                            }
                            selected={value === id}
                            optionProps={hubModelList.getOptionProps(
                              optionKey,
                              value === id,
                            )}
                            onClick={() => {
                              if (isSearchGguf) {
                                setExpandedGguf((prev) =>
                                  prev === id ? null : id,
                                );
                              } else {
                                handleModelClick(id);
                              }
                            }}
                            vramStatus={
                              isSearchGguf ? null : (vram?.status ?? null)
                            }
                            vramEst={isSearchGguf ? undefined : vram?.est}
                            gpuGb={
                              gpu.available ? gpu.memoryTotalGb : undefined
                            }
                            onArrowDownIntoChildren={
                              expandedGguf === id
                                ? () => {
                                    const focused =
                                      focusFirstChildOption(optionKey);
                                    return focused;
                                  }
                                : undefined
                            }
                          />
                          {expandedGguf === id && (
                            <GgufVariantExpander
                              repoId={id}
                              onSelect={onSelect}
                              parentOptionKey={optionKey}
                              onNavigatePastStart={() =>
                                hubModelList.focusOption(optionKey)
                              }
                              onNavigatePastEnd={() =>
                                hubModelList.moveFocus(optionKey, "next")
                              }
                              gpuGb={
                                gpu.available ? gpu.memoryTotalGb : undefined
                              }
                              systemRamGb={
                                gpu.available
                                  ? gpu.systemRamAvailableGb
                                  : undefined
                              }
                              onDeleteVariant={async (quant) => {
                                await deleteCachedModel(id, quant);
                                refreshCachedLists();
                              }}
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
            </>
          )}
        </div>
      </div>
      {/* Floating eject pill: overlaid on the list bottom, outside the scroll
          so the edge fade never touches it. Only the pill catches clicks. */}
      {onEject ? (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-end pr-3.5 pb-5">
          <button
            type="button"
            onClick={onEject}
            className="pointer-events-auto inline-flex items-center justify-center gap-2 rounded-md bg-popover px-3 py-2 text-[13px] text-destructive shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] transition-colors hover:bg-[color-mix(in_srgb,var(--destructive)_12%,var(--popover))] dark:bg-[color-mix(in_srgb,var(--foreground)_10%,var(--sidebar))] dark:shadow-none dark:hover:bg-[color-mix(in_srgb,var(--destructive)_22%,var(--sidebar))]"
            title="Eject model"
          >
            <HugeiconsIcon icon={RemoveCircleIcon} className="size-3.5" />
            Eject model
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** Fine-tuned model rows for the On Device tab's Fine-tuned section. Plugs into
 * that section's roving list and shared GGUF-expand state. */
function FineTunedRows({
  adapters,
  value,
  onSelect,
  onModelsChange,
  deleteDisabled = false,
  loraModelList,
  expandedGguf,
  setExpandedGguf,
  gpu,
}: {
  adapters: LoraModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  loraModelList: ReturnType<typeof useRovingModelList>;
  expandedGguf: string | null;
  setExpandedGguf: Dispatch<SetStateAction<string | null>>;
  gpu: {
    available: boolean;
    memoryTotalGb: number;
    systemRamAvailableGb: number;
  };
}) {
  return (
    <>
      {adapters.map((adapter) => {
        const isLocal = adapter.source === "local";
        const isTraining = adapter.source === "training";
        const isExported = adapter.source === "exported";
        const isMerged = adapter.exportType === "merged";
        const isGguf = adapter.exportType === "gguf";
        const isExportedGguf = isExported && isGguf;
        const canDelete = canDeleteLoraModel(adapter);
        const isTrainingFull = isTraining && isMerged;
        const isLocalGgufDir =
          isLocal && (isGgufRepo(adapter.id) || isGgufRepo(adapter.name));
        const optionKey = makeModelOptionKey("lora", adapter.id);
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
                  optionProps={loraModelList.getOptionProps(
                    optionKey,
                    value === adapter.id,
                  )}
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
                      <span className="block break-words">{adapter.name}</span>
                      <span className="block mt-1 text-[10px] text-muted-foreground break-all">
                        {adapter.id}
                      </span>
                    </>
                  }
                  onArrowDownIntoChildren={
                    expandedGguf === adapter.id
                      ? () => {
                          const focused = focusFirstChildOption(optionKey);
                          return focused;
                        }
                      : undefined
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
                  onDeleted={() => onModelsChange?.({ id: adapter.id })}
                />
              )}
            </div>
            {expandedGguf === adapter.id && (
              <GgufVariantExpander
                repoId={adapter.id}
                onSelect={onSelect}
                parentOptionKey={optionKey}
                onNavigatePastStart={() => loraModelList.focusOption(optionKey)}
                onNavigatePastEnd={() =>
                  loraModelList.moveFocus(optionKey, "next")
                }
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
    </>
  );
}
