// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
import { shouldRefreshPickerInventoryOnMount } from "@/components/resource-picker/picker-tab-policy";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { INVENTORY_FRESHNESS_WINDOW_MS } from "@/features/hub/inventory";
import { ApiProviderLogo } from "@/features/chat";
import {
  type ScanFolderInfo,
  addScanFolder,
  deleteFineTunedModel,
  listGgufVariants,
  listRecommendedFolders,
  listScanFolders,
  removeScanFolder,
} from "@/features/chat";
import {
  chatModelLoaded,
  isExternalModelId,
  useChatRuntimeStore,
} from "@/features/chat";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantDetail,
  LocalModelInfo,
} from "@/features/chat";
import {
  DotTag,
  type HubOption,
  HubOptionMenu,
  TrainIcon,
  TransportConflictDialog,
  deleteCachedModel,
  ggufVariantDisplayLabel,
  invalidateGgufVariantsCache,
  listGgufVariants as listGgufVariantsCached,
  useGgufVariantsCacheVersions,
  useHubInfiniteScroll,
} from "@/features/hub";
import {
  type HfModelResult,
  type HfSortKey,
  useHubModelSearch,
} from "@/features/hub";
import type { HfTaskFilter } from "@/features/hub/hooks/use-hub-model-search";
import {
  classifyUnslothSupport,
  downloadManager,
  hfApiToken,
  isHiddenModelId,
  jobKeyOf,
  useDownloadManagerStore,
  useHfTokenStore,
  useOnlineStatus,
} from "@/features/hub";
import { useDebouncedValue, useGpuInfo, useInferenceGpuInfo } from "@/hooks";
import { diffusionRouteSearch } from "@/lib/diffusion-route-search";
import { extractParamLabel } from "@/lib/model-size";
import { toast } from "@/lib/toast";
import { cn, formatCompact } from "@/lib/utils";
import type { VramFitStatus } from "@/lib/vram";
import { checkVramFit, estimateLoadingVram } from "@/lib/vram";
import {
  Add01Icon,
  ArrowUpDownIcon,
  AudioWave01Icon,
  Cancel01Icon,
  DashboardCircleIcon,
  Download01Icon,
  Flag01Icon,
  Folder02Icon,
  HelpCircleIcon,
  PinIcon,
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
import { useChatPickerInventory } from "../../inventory/use-chat-picker-inventory";
import { FolderBrowser } from "./folder-browser";
import {
  type ModelCapabilities,
  detectCapabilities,
  hasAnyCapability,
} from "./model-capabilities";
import { ModelDeleteAction } from "./model-delete-action";
import { ModelLoadSettingsAction } from "./model-load-settings-action";
import { ModelRowMenu } from "./model-row-menu";
import {
  type ModelLoadTimes,
  loadedAt,
  useModelLoadTimes,
} from "./model-usage";
import {
  makePinRank,
  pinKey,
  pinnedQuantEntries,
  usePinnedModelsStore,
} from "./pinned-models";
import {
  type FormatFilter,
  estimateQuantBytes,
  fitsDevice,
  hfModelFitsDevice,
  isMlxId,
  isMobileVariant,
  isRecommendableFormat,
  loadScopedGpu,
  matchesFormatFilter,
  orderRecommendedRows,
  paramsFromId,
  searchRowFitsDevice,
  searchableRecommendedIds,
} from "./recommended-fit";
import {
  ggufVariantsMatchForPicker,
  modelIdsMatchForPicker,
  soleQuantRowState,
} from "./row-identity";
import {
  type FormatTone,
  isUnslothOwner,
  parseMetaTokens,
  splitRepoLabel,
} from "./row-meta";
import {
  type SoleQuantEntry,
  type SoleQuantTarget,
  createSoleQuantReader,
  partitionSoleQuants,
  soleQuantFingerprint,
  soleQuantKey,
  takeDriftedRepos,
} from "./sole-quant-cache";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelDownloadFootprintResolver,
  ModelSelectorChangeMeta,
} from "./types";
import {
  type CatalogGroup,
  artifactForRepoId,
  curatedCapabilitiesFor,
  curatedDisplayNameFor,
  curatedSizeBytesFor,
  curatedTotalParamsFor,
} from "./model-catalog";
import { describeVariantListingError } from "./variant-listing-error";
import {
  shouldMountVariantExpander,
  toggleAutoExpandedRow,
  visibleGgufVariants,
} from "./variant-visibility";

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
  divider,
}: {
  children: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
  collapsed?: boolean;
  onToggle?: () => void;
  /** Draw a divider line above, evenly spaced, to separate it from the section
   *  above (omit on the first section). */
  divider?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-1 px-2.5 pb-1",
        divider ? "mt-3 border-t border-border/50 pt-3" : "pt-3",
      )}
    >
      <span className="flex items-center gap-1.5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
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
  // Guard non-positive / non-finite sizes so we never render "NaN undefined".
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  // Decimal (base-1000) units to match Hugging Face's reported file sizes (GPU-fit
  // math below stays base-1024 since VRAM is binary). Divide iteratively rather
  // than via Math.log, which has float error at exact powers of 1000 (mislabeling
  // 1 TB as "1000 GB") and could run off the end of units.
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let value = bytes;
  while (value >= 1000 && i < units.length - 1) {
    value /= 1000;
    i += 1;
  }
  // No space: "145MB" reads as one value beside the quant chip.
  return `${value.toFixed(value < 10 ? 1 : 0)}${units[i]}`;
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

/** "This model reads images", from the GGUF metadata (On Device rows). */
function VisionBadge() {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label="Vision"
          className="flex h-[18px] shrink-0 items-center justify-center rounded-md border border-border/60 px-1.5 text-indigo-700 dark:text-indigo-300"
        >
          <HugeiconsIcon icon={ViewIcon} className="size-3" strokeWidth={1.8} />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        This model can process image inputs
      </TooltipContent>
    </Tooltip>
  );
}

/** Parameter count chip ("27B"). */
function ParamChip({ label }: { label: string }) {
  return (
    <span className="whitespace-nowrap rounded-md border border-border/60 px-1.5 py-px text-ui-10 font-medium text-muted-foreground tabular-nums">
      {label}
    </span>
  );
}

// Format colours: gguf blue, mlx amber, safetensors/checkpoint pink, adapter.
const FORMAT_TONE_DOT: Record<FormatTone, string> = {
  gguf: "bg-format-gguf",
  mlx: "bg-format-mlx",
  checkpoint: "bg-format-checkpoint",
  adapter: "bg-format-adapter",
};

/** Format as a coloured dot ahead of the name, named on hover. A word like
 *  "Safetensors" is wide enough to shove the rest of the row around. */
function FormatTag({ tone, label }: { tone: FormatTone; label: string }) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        {/* Hit area, so hovering the dot is not a pixel hunt. */}
        <span
          aria-label={label}
          className="flex size-[14px] shrink-0 items-center justify-center"
        >
          <span
            aria-hidden="true"
            className={cn("size-[5px] rounded-full", FORMAT_TONE_DOT[tone])}
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

/** "Already on disk", shown on Hub rows that are also downloaded. */
function DownloadedBadge() {
  return (
    <span
      title="Already downloaded"
      aria-label="Already downloaded"
      className="flex h-[18px] shrink-0 items-center justify-center text-status-success"
    >
      <HugeiconsIcon
        icon={Download01Icon}
        className="size-3"
        strokeWidth={1.8}
      />
    </span>
  );
}

/** VRAM verdict for Hub rows: over budget, or a tight fit. */
function VramBadge({ status }: { status?: VramFitStatus | null }) {
  if (status === "exceeds") {
    return (
      <span className="whitespace-nowrap text-ui-9 font-medium !text-red-700 !bg-red-50 dark:!text-red-300 dark:!bg-red-500/15 px-1.5 py-0.5 rounded">
        OOM
      </span>
    );
  }
  if (status === "tight") {
    return (
      <span className="whitespace-nowrap text-ui-9 font-medium !text-amber-400">
        TIGHT
      </span>
    );
  }
  return null;
}

const SIZE_PARTS_RE = /^(~?)([\d.]+)\s*([A-Za-z]+)$/;

/** A size in mono: the dot pulled in, a hair of air before the unit. */
function SizeText({ value }: { value: string }) {
  const parts = SIZE_PARTS_RE.exec(value);
  if (!parts) {
    return <>{value}</>;
  }
  const [, approx, digits, unit] = parts;
  const [whole, fraction] = digits.split(".");
  return (
    <>
      {approx}
      {whole}
      {fraction === undefined ? null : (
        <>
          <span className="mx-[-0.1em]">.</span>
          {fraction}
        </>
      )}
      <span className="ml-[0.14em]">{unit}</span>
    </>
  );
}

/** Keep the row's size treatment consistent with every other model. Diffusion
 * GGUFs get one small explanation affordance because their checkpoint is only
 * part of what the loader must keep on disk. The icon is decorative: the
 * explanation hangs off the row button, the one focusable element here. */
export function GgufDownloadFootprint({
  checkpointBytes,
  companionBytes,
}: {
  checkpointBytes: number;
  companionBytes: number;
}) {
  const totalBytes = checkpointBytes + companionBytes;
  // Whole-GB rounding is too lossy for a sum: "2.6 GB + 8.2 GB = 11 GB"
  // looks contradictory. Keep one decimal for the aggregate through GB/TB.
  const totalLabel =
    totalBytes >= 1_000_000_000 && totalBytes < 1_000_000_000_000
      ? `${(totalBytes / 1_000_000_000).toFixed(1)} GB`
      : totalBytes >= 1_000_000_000_000
        ? `${(totalBytes / 1_000_000_000_000).toFixed(1)} TB`
        : formatBytes(totalBytes);
  return (
    <span
      data-model-download-footprint={true}
      className="flex items-center gap-1 whitespace-nowrap text-foreground/80"
    >
      <SizeText value={totalLabel} />
      <span
        data-model-download-footprint-help={true}
        aria-hidden={true}
        className="flex size-3.5 shrink-0 items-center justify-center text-muted-foreground/70"
      >
        <HugeiconsIcon icon={HelpCircleIcon} className="size-3" strokeWidth={1.8} />
      </span>
    </span>
  );
}

/** The checkpoint-versus-assets breakdown behind that aggregate. */
export function GgufDownloadFootprintExplanation({
  checkpointBytes,
  companionBytes,
}: {
  checkpointBytes: number;
  companionBytes: number;
}) {
  return (
    <>
      <span className="font-medium">Full required size</span>
      <span className="ml-1 text-muted-foreground">
        {formatBytes(checkpointBytes)} model + {formatBytes(companionBytes)} required assets
      </span>
    </>
  );
}

/** The one quant a row loads, as a compact mono chip. */
function QuantChip({ label }: { label: string }) {
  return (
    <span className="inline-flex h-[18px] max-w-full items-center overflow-hidden rounded-md bg-black/[0.06] px-1 font-mono text-ui-9 text-muted-foreground dark:bg-white/[0.1]">
      {label}
    </span>
  );
}

function isRuntimeLoadedModel(
  loadedModelId: string | undefined,
  activeGgufVariant: string | null | undefined,
  modelId: string,
  variantPolicy: "none" | "required" | "ignore",
): boolean {
  if (!modelIdsMatchForPicker(loadedModelId, modelId)) return false;
  if (variantPolicy === "ignore") return true;
  const hasActiveGgufVariant = !ggufVariantsMatchForPicker(
    activeGgufVariant,
    null,
  );
  return variantPolicy === "required"
    ? hasActiveGgufVariant
    : !hasActiveGgufVariant;
}

// Shared row columns, so the meta lines up down the list instead of drifting
// with each name's length. Widths are em of the slot's own text, so they
// follow the UI font scale, and collapse below the picker's full width.
// min-w-min means a width is the column held open, not a clamp: an outsized
// badge grows its own slot rather than spilling over the next one.
const META_COLUMN = {
  // Fits "UD-Q4_K_XL"; a hard cap, so longer quants clip.
  quant: "min-[560px]:w-[7.2em]",
  // One capability / vision / downloaded badge.
  badge: "min-w-min min-[560px]:w-[24px]",
  // The "OOM" pill, wider than bare "TIGHT" (Hub rows).
  vram: "min-w-min min-[560px]:w-[4em]",
  // "235B" on device rows; Hub rows report "2779.5B", hence paramWide.
  param: "min-w-min min-[560px]:w-[3.6em]",
  paramWide: "min-w-min min-[560px]:w-[5.2em]",
  // "536 MB".
  size: "min-w-min min-[560px]:w-[4.2em]",
  // The format dot that leads the row; the name lives in its tooltip.
  format: "min-[560px]:w-[14px]",
} as const;

// One gutter for every row, gear or no gear, so the columns never shift by a
// button. The buttons show on the hovered row, or while their menu is open;
// the gutter stays open so nothing moves as they appear.
const ROW_ACTIONS_CLASS =
  "mr-0.5 flex w-[38px] shrink-0 items-center justify-end -space-x-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100 has-[[data-state=open]]:opacity-100 [@media(hover:none)]:opacity-100";

function ModelRow({
  label,
  meta,
  selected,
  loaded = false,
  onClick,
  vramStatus,
  vramEst,
  gpuGb,
  tooltipText,
  hubUrl,
  optionProps,
  onArrowDownIntoChildren,
  capabilities,
  hideOwner,
  downloaded,
  showVision,
  quantChip,
  alignMeta,
  showSize,
  className,
}: {
  label: string;
  meta?: string | null;
  selected?: boolean;
  /** Override badge state when authoritative runtime state is available. */
  loaded?: boolean;
  onClick: () => void;
  vramStatus?: VramFitStatus | null;
  vramEst?: number;
  gpuGb?: number;
  tooltipText?: ReactNode;
  /** Hugging Face address (e.g. "huggingface.co/owner/name") for online/Hub
   * rows; surfaced on hover so their repo id / URL is discoverable the same
   * way local rows show an on-disk path. Omit to show no address line. */
  hubUrl?: string;
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
  /** Grey chip beside the name, for rows that load one specific quant. */
  quantChip?: string | null;
  /** Column layout (see META_COLUMN): "device" reserves the quant chip,
   *  "hub" the download and VRAM badges those lists carry instead. */
  alignMeta?: "device" | "hub";
  /** Hold the size column open. Hub rows pass this on the MLX and Safetensors
   *  filters, where a repo is one download with one size. */
  showSize?: boolean;
  className?: string;
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
  // Drop our own owner: the list is nearly all unsloth/, so it is noise.
  // Other owners still show, which is what tells the two apart.
  const showOwner = !!owner && !hideOwner && !isUnslothOwner(owner);
  const parsed = parseMetaTokens(meta);
  // Param chip from meta, else derived from the name so GGUF rows show it too.
  const paramLabel = parsed.param ?? extractParamLabel(name) ?? null;
  // Use the passed-in capabilities (tag-aware) or infer from the repo name.
  const caps = capabilities ?? detectCapabilities({ id: label });
  const showCaps = hasAnyCapability(caps);
  const aligned = alignMeta !== undefined;
  // One dot per row. A second format shares the first's colour anyway, so it
  // rides along in the tooltip instead of pushing the name out of line.
  const formatDot = parsed.formats[0]
    ? {
        tone: parsed.formats[0].tone,
        label: parsed.formats.map((f) => f.label).join(" · "),
      }
    : null;

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
        "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] focus-visible:bg-[#ececec] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:bg-[var(--sidebar-accent)] dark:focus-visible:bg-[var(--sidebar-accent)]",
        selected && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
        className,
      )}
    >
      <span className="flex min-w-0 flex-1 items-baseline">
        {/* Fixed slot, so names start on one line with or without a dot. */}
        {aligned ? (
          <span
            className={cn(
              "mr-1 flex shrink-0 items-center self-center",
              META_COLUMN.format,
            )}
          >
            {formatDot ? <FormatTag {...formatDot} /> : null}
          </span>
        ) : formatDot ? (
          <span className="mr-1 flex shrink-0 items-center self-center">
            <FormatTag {...formatDot} />
          </span>
        ) : null}
        {showOwner ? (
          <span className="inline-flex min-w-0 max-w-[45%] shrink items-baseline text-ui-13 text-muted-foreground/90">
            <span className="truncate">{owner}</span>
            <span className="shrink-0 text-muted-foreground/45">/</span>
          </span>
        ) : null}
        <span className="min-w-0 flex-1 truncate">{name}</span>
        {/* Here it eats name width instead of moving the meta columns. */}
        {aligned && loaded && (
          <DotTag
            tone="success"
            label="Loaded"
            className="ml-2 h-[18px] shrink-0 gap-1 rounded-md px-1.5"
            dotClassName="size-[5px]"
          />
        )}
        {alignMeta === "device" ? (
          <span
            className={cn(
              "ml-1.5 flex shrink-0 items-center self-center text-ui-9",
              META_COLUMN.quant,
            )}
          >
            {quantChip ? <QuantChip label={quantChip} /> : null}
          </span>
        ) : quantChip ? (
          <span className="ml-2 shrink-0 rounded-md bg-black/[0.06] px-1.5 py-px font-mono text-ui-10 text-muted-foreground dark:bg-white/[0.1]">
            {quantChip}
          </span>
        ) : null}
      </span>
      <span
        className={cn(
          "ml-auto flex shrink-0 items-center",
          aligned ? "gap-1" : "gap-1.5",
        )}
      >
        {/* Capabilities, vision and the Hub lists' "on disk" mark share one
            column; two of them widen the slot rather than overlap. */}
        {aligned ? (
          <span
            className={cn(
              "flex shrink-0 items-center justify-center gap-1 text-ui-10",
              META_COLUMN.badge,
            )}
          >
            {showCaps && <CapabilityIcons caps={caps} />}
            {showVision && <VisionBadge />}
            {downloaded && !loaded ? <DownloadedBadge /> : null}
          </span>
        ) : (
          <>
            {showCaps && <CapabilityIcons caps={caps} />}
            {showVision && <VisionBadge />}
            {loaded && (
              <DotTag
                tone="success"
                label="Loaded"
                className="h-[18px] gap-1 rounded-md px-1.5"
                dotClassName="size-[5px]"
              />
            )}
            {downloaded && !loaded ? <DownloadedBadge /> : null}
          </>
        )}
        {alignMeta === "hub" ? (
          <span
            className={cn(
              "flex shrink-0 items-center justify-end text-ui-9",
              META_COLUMN.vram,
            )}
          >
            <VramBadge status={vramStatus} />
          </span>
        ) : (
          <VramBadge status={vramStatus} />
        )}
        {aligned ? (
          <span
            className={cn(
              "flex shrink-0 justify-end text-ui-10",
              alignMeta === "hub" ? META_COLUMN.paramWide : META_COLUMN.param,
            )}
          >
            {paramLabel ? <ParamChip label={paramLabel} /> : null}
          </span>
        ) : paramLabel ? (
          <ParamChip label={paramLabel} />
        ) : null}
        {parsed.texts.map((text) => (
          <span key={text} className="text-ui-10 text-muted-foreground">
            {text}
          </span>
        ))}
        {/* GGUF repos hold several quants of different sizes, so their rows
            report one only once expanded, leaving the column an empty gap. */}
        {alignMeta === "device" || showSize ? (
          <span
            className={cn(
              "shrink-0 whitespace-nowrap text-right font-mono text-ui-10 text-muted-foreground tabular-nums",
              META_COLUMN.size,
            )}
          >
            {parsed.size === undefined ? null : (
              <SizeText value={parsed.size} />
            )}
          </span>
        ) : aligned ? null : parsed.size !== undefined ? (
          <span className="font-mono text-ui-10 text-muted-foreground tabular-nums">
            <SizeText value={parsed.size} />
          </span>
        ) : null}
      </span>
    </button>
  );

  // Optional Hugging Face address line for online/Hub rows, rendered under
  // whichever tooltip shows so the repo id / URL is always visible on hover.
  const hubUrlLine = hubUrl ? (
    <span className="block mt-1 text-ui-10 text-muted-foreground break-all">
      {hubUrl}
    </span>
  ) : null;

  // The dot names its format on hover only, which keyboard focus never
  // reaches, so the row tooltip carries it too.
  const formatLine = formatDot ? (
    <span className="block text-ui-10 mt-1">{formatDot.label}</span>
  ) : null;

  const tooltipBody = vramTooltipText ? (
    <>
      {label}
      <span className="block text-ui-10 mt-1">{vramTooltipText}</span>
      {formatLine}
      {hubUrlLine}
    </>
  ) : tooltipText ? (
    <>
      {tooltipText}
      {formatLine}
      {hubUrlLine}
    </>
  ) : hubUrl ? (
    <>
      <span className="block break-words">{label}</span>
      {formatLine}
      {hubUrlLine}
    </>
  ) : formatLine ? (
    <>
      <span className="block break-words">{label}</span>
      {formatLine}
    </>
  ) : null;

  if (tooltipBody) {
    return (
      <Tooltip delayDuration={700}>
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
        <TooltipContent
          side="left"
          className="tooltip-compact max-w-xs break-all"
        >
          {tooltipBody}
        </TooltipContent>
      </Tooltip>
    );
  }
  return content;
}

// ── GGUF Variant Expander ────────────────────────────────────

function isValidGgufVariant(variant: unknown): variant is GgufVariantDetail {
  if (!variant || typeof variant !== "object") return false;
  const candidate = variant as Partial<GgufVariantDetail>;
  return (
    typeof candidate.filename === "string" &&
    candidate.filename.length > 0 &&
    typeof candidate.quant === "string" &&
    candidate.quant.length > 0 &&
    typeof candidate.size_bytes === "number" &&
    Number.isFinite(candidate.size_bytes) &&
    candidate.size_bytes >= 0 &&
    (candidate.downloaded === undefined ||
      typeof candidate.downloaded === "boolean") &&
    // Carried through so each row can look up its own dependency group's
    // footprint. Absent or null on an older backend, which groups the repo as
    // one, so it must never reject the row.
    (candidate.dependency_key === undefined ||
      candidate.dependency_key === null ||
      typeof candidate.dependency_key === "string")
  );
}

function normalizeGgufVariantsResponse(
  res:
    | {
        variants?: unknown;
        default_variant?: unknown;
        has_vision?: unknown;
        context_length?: unknown;
        resolved_locally?: unknown;
      }
    | null
    | undefined,
): {
  variants: GgufVariantDetail[];
  defaultVariant: string | null;
  hasVision: boolean;
  contextLength: number | null;
  resolvedLocally: boolean;
} {
  const contextLength = res?.context_length;
  return {
    variants: (Array.isArray(res?.variants) ? res.variants : []).filter(
      isValidGgufVariant,
    ),
    defaultVariant:
      typeof res?.default_variant === "string" && res.default_variant.length > 0
        ? res.default_variant
        : null,
    hasVision: res?.has_vision === true,
    contextLength:
      typeof contextLength === "number" &&
      Number.isFinite(contextLength) &&
      contextLength >= 0
        ? contextLength
        : null,
    // The backend's own verdict, which resolves existence-first: a marker-less relative name
    // that exists on disk is a local model even though no path prefix says so. A server that
    // predates the field omits it, leaving the prefix test to answer alone as before.
    resolvedLocally: res?.resolved_locally === true,
  };
}

function ggufVariantExpectedBytes(variant: GgufVariantDetail): number {
  const downloadBytes = variant.download_size_bytes;
  return typeof downloadBytes === "number" &&
    Number.isFinite(downloadBytes) &&
    downloadBytes > 0
    ? downloadBytes
    : variant.size_bytes;
}

/** The one quant a repo holds, plus the vision flag read with it. The
 *  collapsed row never mounts the expander, so this is its only source. */
interface SoleDownloadedQuant {
  variant: GgufVariantDetail;
  hasVision: boolean;
}

/** The repo's one complete quant, or null when it holds none, holds several,
 *  or could not be read. Disk-only and client-cached: no remote listing. */
async function readSoleQuant(
  target: SoleQuantTarget,
  hfToken?: string,
): Promise<SoleDownloadedQuant | null> {
  try {
    const res = await listGgufVariantsCached(target.repoId, hfToken, {
      preferLocalCache: true,
      localPath: target.localSource,
    });
    const normalized = normalizeGgufVariantsResponse(res);
    const local = normalized.variants;
    // One file on disk and nothing torn beside it. A partial quant keeps the
    // expander, where it can be resumed.
    if (local.length !== 1 || local[0].downloaded !== true) return null;
    return { variant: local[0], hasVision: normalized.hasVision };
  } catch {
    return null;
  }
}

const EMPTY_SOLE_QUANT_ENTRIES: ReadonlyMap<
  string,
  SoleQuantEntry<SoleDownloadedQuant>
> = new Map();
// Reads run a few at a time, so a large cache doesn't fire one request per
// repo. A worker pool, not fixed batches: one slow repo holds up only itself.
const SOLE_QUANT_WORKERS = 6;

/** On Device repos holding exactly one quant on disk, keyed by repo id. With
 *  "Show all quantizations" off there is nothing else to pick, so those repos
 *  collapse into one pinned-style row. Results are kept per repo, so one
 *  repo's download or delete leaves every other row as it was. */
function useSoleDownloadedQuants(
  repos: readonly CachedGgufRepo[],
  { enabled, hfToken }: { enabled: boolean; hfToken?: string },
): {
  quants: ReadonlyMap<string, SoleDownloadedQuant>;
  pending: ReadonlySet<string>;
} {
  const repoIds = useMemo(() => repos.map((repo) => repo.repo_id), [repos]);
  // A download or delete invalidates one repo, so watch each repo's version.
  const variantsVersion = useGgufVariantsCacheVersions(repoIds);
  const targets = useMemo(() => {
    const versions = variantsVersion.split(",");
    return repos.map((repo, index) => {
      const localSource = repo.load_id || repo.cache_path || null;
      const fingerprint = soleQuantFingerprint(repo);
      return {
        repoId: repo.repo_id,
        localSource,
        fingerprint,
        key: soleQuantKey(versions[index], localSource, fingerprint),
      };
    });
  }, [repos, variantsVersion]);

  const [entries, setEntries] =
    useState<ReadonlyMap<string, SoleQuantEntry<SoleDownloadedQuant>>>(
      EMPTY_SOLE_QUANT_ENTRIES,
    );
  const { quants, pending, stale } = useMemo(
    () => partitionSoleQuants(targets, entries, { enabled }),
    [targets, entries, enabled],
  );

  // A change outside this tab, another window or the CLI, moves the row's
  // bytes without touching this instance's variants cache. Drop that repo's
  // cached listing so the read, and every other reader, sees disk again.
  const fingerprintsRef = useRef(new Map<string, string>());
  useEffect(() => {
    for (const repoId of takeDriftedRepos(targets, fingerprintsRef.current)) {
      invalidateGgufVariantsCache(repoId);
    }
  }, [targets]);

  // Reads outlive a render, so they run outside it. The token is read at call
  // time, so a change to it does not strand the reader.
  const hfTokenRef = useRef(hfToken);
  hfTokenRef.current = hfToken;
  const mountedRef = useRef(true);
  useEffect(() => {
    // Set on setup, not just cleared on teardown: StrictMode replays effects,
    // and a ref left false would discard every later read.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const readerRef = useRef<ReturnType<
    typeof createSoleQuantReader<SoleDownloadedQuant>
  > | null>(null);
  if (readerRef.current === null) {
    readerRef.current = createSoleQuantReader<SoleDownloadedQuant>({
      workers: SOLE_QUANT_WORKERS,
      read: (target) => readSoleQuant(target, hfTokenRef.current),
      commit: (target, quant) => {
        if (!mountedRef.current) return;
        setEntries((prev) => {
          const next = new Map(prev);
          next.set(target.repoId, { key: target.key, quant });
          return next;
        });
      },
    });
  }

  useEffect(() => {
    if (stale.length > 0) readerRef.current?.start(stale);
  }, [stale]);

  return { quants, pending };
}

function GgufVariantExpander({
  repoId,
  loadId,
  cachePath,
  onSelect,
  resolveDownloadFootprint,
  gpuGb,
  systemRamGb,
  budgetKnown = false,
  hfToken,
  parentOptionKey,
  onNavigatePastStart,
  onNavigatePastEnd,
  onConfigure,
  sourceOverride,
  variantActions,
  onDevice = false,
  allowPin = false,
  onHasVision,
}: {
  repoId: string;
  /** Snapshot the cached listing pinned this repo to, if any. */
  loadId?: string | null;
  /** Cache directory this downloaded row represents, if any. */
  cachePath?: string | null;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  /** HF token threaded into the variant fetch so private/gated repos resolve
   *  their GGUF variants (and update badges). */
  hfToken?: string;
  parentOptionKey?: string;
  onNavigatePastStart?: () => void;
  onNavigatePastEnd?: () => void;
  onConfigure?: (id: string, meta: ModelSelectorChangeMeta) => void;
  sourceOverride?: ModelSelectorChangeMeta["source"];
  /** Update/delete actions for cached variant rows. Omitted by browse-only
   *  expanders (Recommended, etc.) that don't manage on-disk variants. */
  variantActions?: {
    onUpdate?: (quant: string, expectedBytes: number) => Promise<void> | void;
    updateTitle?: string;
    renderUpdateDescription?: (quant: string) => ReactNode;
    getUpdateSuccessMessage?: (quant: string) => string;
    updateDisabled?: boolean;
    onDelete?: (quant: string) => Promise<void> | void;
    deleteTitle?: string;
    renderDeleteDescription?: (quant: string) => ReactNode;
    getDeleteSuccessMessage?: (quant: string) => string;
    deleteDisabled?: boolean;
  };
  /** On Device rows honor the Show all quantizations setting; Recommended and
   *  other browse lists always show every quant. */
  onDevice?: boolean;
  /** Only managed cached-Hub rows can surface quant pins in the Pinned
   *  section. Local-path expanders deliberately leave this false. */
  allowPin?: boolean;
  /** Report GGUF vision support up so the parent row can badge it. */
  onHasVision?: (hasVision: boolean) => void;
}) {
  const pinnedKeys = usePinnedModelsStore((s) => s.pinned);
  const togglePinnedQuant = usePinnedModelsStore((s) => s.togglePinned);
  const onUpdateVariant = variantActions?.onUpdate;
  const updateVariantTitle =
    variantActions?.updateTitle ?? "Update cached model?";
  const renderUpdateVariantDescription =
    variantActions?.renderUpdateDescription;
  const updateDisabled = variantActions?.updateDisabled ?? false;
  const onDeleteVariant = variantActions?.onDelete;
  const deleteVariantTitle =
    variantActions?.deleteTitle ?? "Delete cached model?";
  const renderDeleteVariantDescription =
    variantActions?.renderDeleteDescription;
  const getDeleteVariantSuccessMessage =
    variantActions?.getDeleteSuccessMessage;
  const deleteDisabled = variantActions?.deleteDisabled ?? false;
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [defaultVariant, setDefaultVariant] = useState<string | null>(null);
  const [hasVision, setHasVision] = useState(false);
  // Native max context (GGUF metadata); only set once a variant is downloaded.
  const [nativeContext, setNativeContext] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  // Whether the LISTING resolved this identifier off disk. Not derivable from loadId/cachePath,
  // which a downloaded hub model also carries.
  const [resolvedLocally, setResolvedLocally] = useState(false);
  const localSource = loadId || cachePath || null;

  useEffect(() => {
    let canceled = false;
    // Collapsing the row drops the request: a stalled one otherwise holds a
    // per-host connection, and enough of them stall download and load too.
    const controller = new AbortController();
    queueMicrotask(() => {
      if (canceled) return;
      setLoading(true);
      setError(null);
      // Belongs to the identifier being listed: carrying it over would apply the previous
      // row's locality to this one's footprint arithmetic.
      setResolvedLocally(false);
    });

    // The row's own directory, so disk contents count against that cache, not the active one. No
    // preferLocalCache: it answers from disk alone and drops the undownloaded.
    listGgufVariants(repoId, hfToken, {
      ...(localSource ? { localPath: localSource } : {}),
      signal: controller.signal,
    })
      .then((res) => {
        if (canceled) return;
        const normalized = normalizeGgufVariantsResponse(res);
        setVariants(normalized.variants);
        setDefaultVariant(normalized.defaultVariant);
        setHasVision(normalized.hasVision);
        onHasVision?.(normalized.hasVision);
        setNativeContext(normalized.contextLength);
        setResolvedLocally(normalized.resolvedLocally);
      })
      .catch((err) => {
        if (canceled) return;
        setError(describeVariantListingError(err));
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });

    return () => {
      canceled = true;
      controller.abort();
    };
  }, [repoId, localSource, refreshKey, hfToken]);

  // Covers Unix absolute (/), Windows drive (C:\, D:/), UNC (\\server), relative (./, ../), tilde (~/)
  const isLocalPath = /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(
    repoId,
  );
  // The prefix test cannot see a marker-less relative directory like "models/my-image-model",
  // which the backend loads off disk. Whether the checkpoint is on disk decides the footprint
  // arithmetic, so that question is asked of the listing, not of the spelling.
  const checkpointIsLocal = isLocalPath || resolvedLocally;

  const handleVariantClick = useCallback(
    // ``filename`` is required, not decorative: the diffusion pages load a quant with {kind: "gguf", filename} and gate that branch on meta.ggufFilename, so a quant label alone made every Images/Video GGUF pick a dead click.
    (quant: string, filename: string, downloaded?: boolean, sizeBytes?: number) => {
      const isAvailable = isLocalPath || downloaded === true;
      onSelect(repoId, {
        source: sourceOverride ?? (isLocalPath ? "local" : "hub"),
        isLora: false,
        // Only for a quant already in the pinned snapshot: a new download lands elsewhere.
        loadId: downloaded === true ? loadId : undefined,
        ggufVariant: quant,
        ggufFilename: filename,
        isDownloaded: isLocalPath ? true : downloaded,
        expectedBytes: sizeBytes,
        contextLength: isAvailable ? nativeContext : undefined,
        isGguf: true,
      });
    },
    [repoId, loadId, isLocalPath, onSelect, sourceOverride, nativeContext],
  );

  // GGUF fit classification matching llama-server's _select_gpus logic:
  //   fits  = model <= 0.7 * total GPU memory
  //   tight = model > 0.7 * GPU but <= 0.7 * GPU + 0.7 * system RAM (--fit uses CPU offload)
  //   oom   = model > 0.7 * GPU + 0.7 * system RAM
  const gpuBudgetGb = (gpuGb ?? 0) * 0.7;
  const totalBudgetGb = gpuBudgetGb + (systemRamGb ?? 0) * 0.7;

  const getGgufFit = useCallback(
    (sizeBytes: number): "fits" | "tight" | "oom" => {
      // Preserve permissive behavior only when no budget was measured. A known
      // zero Vulkan budget means every non-empty variant is OOM.
      if (totalBudgetGb <= 0) return budgetKnown ? "oom" : "fits";
      const gb = sizeBytes / 1024 ** 3;
      if (gb <= 0 || gb <= gpuBudgetGb) return "fits";
      // No-GPU / unified-memory hosts (Mac) have only the RAM budget, so the tier
      // collapses to fit-or-oom against system RAM.
      if (gpuBudgetGb <= 0) return gb <= totalBudgetGb ? "fits" : "oom";
      if (gb <= totalBudgetGb) return "tight";
      return "oom";
    },
    [budgetKnown, gpuBudgetGb, totalBudgetGb],
  );

  // If the recommended variant is OOM, pick the largest fitting one;
  // if all are OOM, recommend the smallest.
  const effectiveRecommended = useMemo(() => {
    if (
      !variants ||
      variants.length === 0 ||
      (totalBudgetGb <= 0 && !budgetKnown)
    ) {
      return defaultVariant;
    }
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
    return sorted[0]?.quant ?? defaultVariant;
  }, [variants, defaultVariant, totalBudgetGb, budgetKnown, getGgufFit]);

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
  // disk, torn ones included. Browse lists always show every quant.
  const showAllQuantizations = useChatRuntimeStore(
    (s) => s.showAllQuantizations,
  );
  const displayVariants = useMemo(() => {
    if (!sortedVariants) return sortedVariants;
    return visibleGgufVariants(sortedVariants, {
      onDevice,
      showAll: showAllQuantizations,
    });
  }, [sortedVariants, showAllQuantizations, onDevice]);

  // A diffusion GGUF is not self-contained: the loader also needs a text
  // encoder, VAE, tokenizer and configs. That companion set is NOT
  // repository-wide, so one representative's footprint cannot speak for the
  // whole listing: a neutral repo can hold GGUFs of different families with
  // different base repos, and FLUX.2-klein picks a different text encoder for
  // its 9B checkpoints than for its 4B ones. Both are folded into the
  // backend's dependency_key, so grouping by it is what keeps a non
  // representative row from advertising a GB-wrong total. One request per
  // distinct key: the ordinary repo has exactly one, which is the cost this
  // representative scheme exists to protect.
  const footprintVariants = useMemo(() => {
    const byKey = new Map<string, GgufVariantDetail>();
    for (const variant of displayVariants ?? []) {
      // An unkeyed repo (older backend, or no family resolved) collapses to one
      // group, which is exactly the previous repo-wide behavior.
      const key = variant.dependency_key ?? "";
      const current = byKey.get(key);
      if (current === undefined) {
        byKey.set(key, variant);
        continue;
      }
      // The recommended quant is the representative of its own group when it
      // has one; otherwise the group's first row stands.
      if (
        current.quant !== effectiveRecommended &&
        variant.quant === effectiveRecommended
      ) {
        byKey.set(key, variant);
      }
    }
    return Array.from(byKey.values());
  }, [displayVariants, effectiveRecommended]);
  const [companionBytesByKey, setCompanionBytesByKey] = useState<
    Map<string, number>
  >(() => new Map());
  useEffect(() => {
    let cancelled = false;
    setCompanionBytesByKey(new Map());
    // A local path is resolved too: only the CHECKPOINT is on disk. Its text encoder, VAE,
    // tokenizer and configs still come from the remote base, which is the larger half of the
    // footprint, so suppressing the request understated a local row by many gigabytes.
    if (!resolveDownloadFootprint) {
      return () => {
        cancelled = true;
      };
    }
    for (const footprintVariant of footprintVariants) {
      const dependencyKey = footprintVariant.dependency_key ?? "";
      const expectedBytes = ggufVariantExpectedBytes(footprintVariant);
      void resolveDownloadFootprint(repoId, {
        // Same source the row itself reports, so the plan describes the pick that would run.
        source: sourceOverride ?? (isLocalPath ? "local" : "hub"),
        isLora: false,
        ggufVariant: footprintVariant.quant,
        ggufFilename: footprintVariant.filename,
        isDownloaded: footprintVariant.downloaded,
        expectedBytes,
        isGguf: true,
      })
        .then((footprint) => {
          if (cancelled || !footprint) return;
          // A checkpoint already on disk is not part of required_bytes at all, so nothing may
          // be subtracted for it: the whole figure IS the remote companion set. Subtracting
          // anyway drove the total to zero and hid a multi-GB companion set behind the
          // checkpoint size. Only a hub pick carries its checkpoint inside the total, and
          // expectedBytes stands in when the planner could not size it.
          const checkpoint = checkpointIsLocal
            ? 0
            : footprint.checkpointBytes > 0
              ? footprint.checkpointBytes
              : expectedBytes;
          const companion = footprint.requiredBytes - checkpoint;
          if (Number.isFinite(companion) && companion > 0) {
            // A fresh Map per resolution: React compares state by identity, and
            // the groups resolve independently, so a mutation would drop the
            // rows whose request landed first.
            setCompanionBytesByKey((previous) => {
              const next = new Map(previous);
              next.set(dependencyKey, companion);
              return next;
            });
          }
        })
        .catch(() => {
          // The checkpoint size remains useful when an older backend or a Hub
          // metadata failure cannot provide the companion footprint.
        });
    }
    return () => {
      cancelled = true;
    };
  }, [
    checkpointIsLocal,
    footprintVariants,
    isLocalPath,
    repoId,
    resolveDownloadFootprint,
    sourceOverride,
  ]);

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
    return (
      <div className="flex flex-wrap items-center gap-2 px-5 py-2 text-xs text-destructive">
        <span>{error}</span>
        <button
          type="button"
          onClick={() => setRefreshKey((key) => key + 1)}
          className="rounded-full border border-destructive/40 px-2 py-0.5 font-medium text-destructive transition-colors hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          Retry
        </button>
      </div>
    );
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
          <span className="text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
            Quantizations
          </span>
          {hasVision && (
            <span className="flex items-center gap-0.5 text-ui-9 font-medium text-indigo-700 dark:text-indigo-300">
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
        const expectedBytes = ggufVariantExpectedBytes(v);
        // This row's own dependency group, never the listing's: see the
        // footprintVariants comment above.
        const companionBytes =
          companionBytesByKey.get(v.dependency_key ?? "") ?? null;
        // A folder has no download to resume; a quant short a shard has no files to load.
        const unusableLocal = isLocalPath && v.partial === true;
        const keyBase = `${repoId}:${v.filename}`;
        const variantOptionKey = makeModelOptionKey("gguf-variant", keyBase);
        const rowButton = (
          <button
            type="button"
            {...variantList.getOptionProps(variantOptionKey, false)}
            disabled={unusableLocal}
            onClick={() =>
              handleVariantClick(
                v.quant,
                v.filename,
                v.downloaded,
                expectedBytes,
              )
            }
            className={cn(
              "flex min-w-0 flex-1 items-center justify-between gap-2 rounded-full py-1 pl-2 pr-1.5 text-left text-sm transition-colors hover:bg-[#ececec] focus-visible:bg-[#ececec] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:bg-[var(--sidebar-accent)] dark:focus-visible:bg-[var(--sidebar-accent)]",
              unusableLocal &&
                "cursor-default opacity-50 hover:bg-transparent dark:hover:bg-transparent",
            )}
          >
            <span className="min-w-0 flex-1 truncate font-mono text-xs">
              <span
                className={cn(oom && "!text-gray-500 dark:!text-gray-400")}
              >
                {ggufVariantDisplayLabel(v)}
              </span>
              {unusableLocal ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">
                  incomplete
                </span>
              ) : v.downloaded ? (
                <>
                  <span className="ml-1.5 text-ui-9 font-sans font-medium text-green-600/90 dark:text-green-400/80">
                    downloaded
                  </span>
                  {v.update_available ? (
                    <span className="ml-1.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">
                      update available
                    </span>
                  ) : null}
                </>
              ) : v.quant === effectiveRecommended ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-primary/70">
                  recommended
                </span>
              ) : null}
            </span>
            <span className="flex items-center gap-1.5 shrink-0">
              {oom && (
                <span className="text-ui-9 font-medium !text-red-700 !bg-red-50 dark:!text-red-300 dark:!bg-red-500/15 px-1.5 py-0.5 rounded">
                  OOM
                </span>
              )}
              {tight && (
                <span className="text-ui-9 font-medium !text-amber-400">
                  TIGHT
                </span>
              )}
              <span className="font-mono text-ui-10 text-muted-foreground tabular-nums">
                {companionBytes === null ? (
                  <SizeText value={formatBytes(v.size_bytes)} />
                ) : (
                  <GgufDownloadFootprint
                    checkpointBytes={v.size_bytes}
                    companionBytes={companionBytes}
                  />
                )}
              </span>
            </span>
          </button>
        );
        return (
          <div key={v.filename} className="flex items-center">
            {/* The explanation rides the row button; nested button triggers are not accessible. */}
            {companionBytes === null ? (
              rowButton
            ) : (
              <Tooltip delayDuration={0}>
                <TooltipTrigger asChild={true}>{rowButton}</TooltipTrigger>
                <TooltipContent side="top" className="tooltip-compact">
                  <GgufDownloadFootprintExplanation
                    checkpointBytes={v.size_bytes}
                    companionBytes={companionBytes}
                  />
                </TooltipContent>
              </Tooltip>
            )}
            {v.downloaded && onConfigure && (
              <ModelLoadSettingsAction
                ariaLabel={`Inference settings for ${repoId} ${v.quant}`}
                className="relative left-0.5"
                onConfigure={() =>
                  onConfigure(repoId, {
                    source: sourceOverride ?? (isLocalPath ? "local" : "hub"),
                    isLora: false,
                    loadId,
                    ggufVariant: v.quant,
                    isDownloaded: true,
                    expectedBytes,
                    contextLength: nativeContext,
                    isGguf: true,
                  })
                }
              />
            )}
            {v.downloaded &&
              (allowPin ||
                (v.update_available && onUpdateVariant) ||
                onDeleteVariant ||
                !isLocalPath) && (
                <ModelRowMenu
                  ariaLabel={`More options for ${repoId} ${v.quant}`}
                  iconClassName="size-3"
                  cachePath={
                    isLocalPath ? undefined : { repoId, variant: v.quant }
                  }
                  pin={
                    allowPin
                      ? {
                          pinned: pinnedKeys.includes(pinKey(repoId, v.quant)),
                          pinLabel: "Pin to top",
                          unpinLabel: "Unpin",
                          onToggle: () => togglePinnedQuant(repoId, v.quant),
                        }
                      : undefined
                  }
                  update={
                    v.update_available && onUpdateVariant
                      ? {
                          title: updateVariantTitle,
                          description: renderUpdateVariantDescription?.(
                            v.quant,
                          ) ?? (
                            <>
                              This will update{" "}
                              <span className="font-medium text-foreground">
                                {repoId} ({v.quant})
                              </span>
                              {"."}
                            </>
                          ),
                          repoId,
                          variant: v.quant,
                          disabled: updateDisabled,
                          onConfirm: () =>
                            onUpdateVariant(v.quant, expectedBytes),
                          onUpdated: () => setRefreshKey((key) => key + 1),
                        }
                      : undefined
                  }
                  del={
                    onDeleteVariant
                      ? {
                          title: deleteVariantTitle,
                          impact: { repoId, variant: v.quant },
                          description: renderDeleteVariantDescription?.(
                            v.quant,
                          ) ?? (
                            <>
                              This will remove{" "}
                              <span className="font-medium text-foreground">
                                {repoId} ({v.quant})
                              </span>{" "}
                              from disk. You can re-download it later.
                            </>
                          ),
                          successMessage:
                            getDeleteVariantSuccessMessage?.(v.quant) ??
                            `Deleted ${repoId} ${v.quant}`,
                          disabled: deleteDisabled,
                          onConfirm: async () => {
                            await onDeleteVariant(v.quant);
                            // Drop the pin too: a pinned row for a deleted file
                            // would try to load something that no longer exists.
                            if (pinnedKeys.includes(pinKey(repoId, v.quant))) {
                              togglePinnedQuant(repoId, v.quant);
                            }
                            // Re-fetch this expander's variants so the deleted
                            // quant stops showing as downloaded (and clickable to
                            // reload) while the repo still has other cached quants.
                            setRefreshKey((key) => key + 1);
                          },
                        }
                      : undefined
                  }
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

// ── Task scoping: which pages own which pipeline tasks ───────────────────

// True when a repo's inferred task is within the picker's task filter (or no filter). Unknown task (null) passes only with no filter.
function taskMatchesFilter(
  repoTask: string | null | undefined,
  filter: HfTaskFilter,
): boolean {
  if (!filter) return true;
  const wanted = Array.isArray(filter) ? filter : [filter];
  return repoTask != null && (wanted as readonly string[]).includes(repoTask);
}

// Image-generation pipeline tasks: owned by the Images page, never chat-loadable. The backend reports "text-to-image" for diffusion-arch GGUFs, and the Images page reuses this as its picker `task` filter.
export const IMAGE_GEN_TASKS = [
  "text-to-image",
  "image-to-image",
  "image-text-to-image",
] as const;

// Video-generation pipeline tasks: owned by the Video page, never chat-loadable. The backend reports "text-to-video" for video-diffusion GGUFs.
// image-to-video is included because HF gives the LTX-2 family that pipeline_tag, so a text-to-video-only filter dropped it out of Video Hub search.
export const VIDEO_GEN_TASKS = ["text-to-video", "image-to-video"] as const;

// Diffusion GGUF archs the Images backend cannot assemble yet (SD/SDXL/PixArt/Wan/...). The backend tags them with this task so the chat picker hides them and the Images picker leaves them out (they would 400 on load).
const UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported";

// Generation tasks the Images / Video pages own. Not chat-loadable, so an on-device pick routes to its page instead.
const DIFFUSION_PAGE_TASKS: readonly string[] = [
  ...IMAGE_GEN_TASKS,
  ...VIDEO_GEN_TASKS,
];

/** The page that runs this task, or null when chat should handle the pick. */
function diffusionPageForTask(task: string | null | undefined): "images" | "video" | null {
  if (!task || !DIFFUSION_PAGE_TASKS.includes(task)) return null;
  return (VIDEO_GEN_TASKS as readonly string[]).includes(task) ? "video" : "images";
}

// Editing/inpaint checkpoints are tagged image-to-image but need an input image the text-to-image backend rejects (mirrors
// its _EDIT_KEYWORDS), so they are hidden by id. The task itself must stay: FLUX.2-klein carries it too. "layered" hides Qwen-Image-Layered, which needs a dedicated pipeline.
const IMAGE_EDIT_KEYWORDS = ["edit", "kontext", "inpaint", "layered"] as const;
// Editing families the backend now SUPPORTS (their own Edit workflow): not hidden despite the edit keyword. Mirrors the backend's qwen-image-edit family.
const SUPPORTED_EDIT_KEYWORDS = ["qwen-image-edit", "kontext"] as const;
// Match a keyword as a whole path/name segment, not a raw substring, so "edit" does not hide ".../edited/..." and "kontext"
// does not hide ".../kontextual/...". The keywords are [a-z-] literals, so no escaping. Mirrors _token_in_needle.
function idHasSegment(id: string, keyword: string): boolean {
  return new RegExp(`(?:^|[-_./\\\\])${keyword}(?:$|[-_./\\\\])`).test(id);
}
function isImageEditModel(repoId: string | null | undefined): boolean {
  if (!repoId) return false;
  const id = repoId.toLowerCase();
  if (SUPPORTED_EDIT_KEYWORDS.some((kw) => idHasSegment(id, kw))) return false;
  return IMAGE_EDIT_KEYWORDS.some((kw) => idHasSegment(id, kw));
}

// Gate an on-device model by the picker's task scope: with a filter (Images) keep only matching, non-editing tasks; with none (chat) drop image-generation models.
function passesTaskGate(
  repoTask: string | null | undefined,
  repoId: string | null | undefined,
  filter: HfTaskFilter,
): boolean {
  if (filter)
    return taskMatchesFilter(repoTask, filter) && !isImageEditModel(repoId);
  // Unfiltered (chat) picker: an on-device diffusion model stays listed and routes to the Images/Video page on click; only the never-loadable tag is hidden.
  return repoTask !== UNSUPPORTED_DIFFUSION_TASK;
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

/** GGUF detection for a local model by backend format hint, name, or file path. */
function localModelIsGguf(m: LocalModelInfo): boolean {
  return (
    m.model_format === "gguf" ||
    isGgufRepo(m.id) ||
    isGgufRepo(m.display_name) ||
    m.path.toLowerCase().endsWith(".gguf")
  );
}

function localPathTooltip(name: string, path: string): ReactNode {
  return (
    <>
      <span className="block break-words">{name}</span>
      <span className="block mt-1 text-ui-10 text-muted-foreground break-all">
        {path}
      </span>
    </>
  );
}

function localModelMeta(isGguf = false): ModelSelectorChangeMeta {
  return {
    source: "local",
    isLora: false,
    isDownloaded: true,
    ...(isGguf ? { isGguf: true } : {}),
  };
}

function localDirectGgufMeta(): ModelSelectorChangeMeta {
  return localModelMeta(true);
}

/** Hugging Face address for an online/Hub row, or undefined when the repo id is
 * missing so the row shows no (empty) address line on hover. */
function hubRepoUrl(id: string | null | undefined): string | undefined {
  const trimmed = id?.trim();
  return trimmed ? `huggingface.co/${trimmed}` : undefined;
}

/** Whether a local model is an MLX build (name hint). MLX runs on Mac only, so
 * callers gate visibility on the host being a Mac. */
function localModelIsMlx(m: LocalModelInfo): boolean {
  return isMlxId(m.id) || isMlxId(m.display_name) || isMlxId(m.model_id ?? "");
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
  onSelect: onSelectProp,
  resolveDownloadFootprint,
  onFoldersChange,
  onBrowseHub,
  onModelsChange,
  onConfigure,
  deleteDisabled = false,
  section = "downloaded",
  sectionToggle,
  onEject,
  task,
  catalog,
}: {
  models: ModelOption[];
  /** Fine-tuned models, shown as a section in the On Device view. */
  loraModels?: LoraModelOption[];
  /** Connected provider models, shown in the Connected section. */
  externalModels?: ExternalModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  onFoldersChange?: () => void;
  /** Open the full Hub page to browse more models. */
  onBrowseHub?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  onConfigure?: (id: string, meta: ModelSelectorChangeMeta) => void;
  deleteDisabled?: boolean;
  /** Section shown when not searching. Search spans all sections. */
  section?: "downloaded" | "recommended" | "custom" | "connected";
  /** Section toggle rendered under the search bar. */
  sectionToggle?: ReactNode;
  onEject?: () => void;
  /** Restrict results to a pipeline task (e.g. text-to-image for the Images page). Undefined = all tasks (the chat default). */
  task?: HfTaskFilter;
  /** Curated catalog for a task-scoped picker: one canonical row per model, with its published formats as the second level. */
  catalog?: CatalogGroup[];
}) {
  const gpu = useGpuInfo();
  const inferenceGpu = useInferenceGpuInfo();
  // What the backend actually holds, not the dropdown highlight, which can be a
  // staged pick. The selection alone was wrong: an image or video load evicts
  // the chat model and leaves the pick untouched, so its rows kept the "Loaded"
  // badge with nothing resident. Same predicate as the header tick, so the two
  // cannot disagree.
  const selectedCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const residentCheckpoint = useChatRuntimeStore((s) => s.residentCheckpoint);
  const loadedModelId = chatModelLoaded({
    checkpoint: selectedCheckpoint,
    isExternalModel: isExternalModelId(selectedCheckpoint),
    residentCheckpoint,
  })
    ? selectedCheckpoint
    : undefined;
  // Loaded GGUF quant of the active model; marks the matching pinned row.
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  // Last-loaded timestamps power the "Recent" sort (vs "Downloaded" = file date).
  const loadTimes = useModelLoadTimes(value);
  // Fade the list's top edge once scrolled, and its bottom edge while more
  // rows sit below the fold.
  const [listScrolled, setListScrolled] = useState(false);
  const [listMoreBelow, setListMoreBelow] = useState(false);
  const hfToken = useHfTokenStore((s) => s.token);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  // Shared Hub search stack (the same hooks the Hub page uses) so the picker
  // and Hub run one implementation. Scoped to unsloth like the old listing.
  const online = useOnlineStatus();
  // Sanitize to anonymous on a malformed token, matching the Hub page.
  const accessToken = hfApiToken(hfToken);
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
    // Only Recommended renders Hub results, so keep the Hub hooks idle on other
    // tabs to avoid needless requests and preserve offline-local behavior.
    enabled: online && section === "recommended",
  });
  const recommendedSearch = useHubModelSearch("", {
    ownerScope: "unsloth",
    sortBy: recommendedSortBy,
    sortDirection: "desc",
    pinUnslothFirst: true,
    keepUnsupportedTags: true,
    accessToken,
    enabled: online && section === "recommended",
  });

  // Lowercased repo ids confirmed GGUF by the store or HF search. Absence means
  // "no hint" -> hasGgufSuffix is the fallback (don't conflate unknown with
  // known-not-GGUF). Lowercased so store and HF IDs match regardless of casing.
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
  // Off: On Device lists only downloaded quants, so a repo holding one collapses
  // into a single row instead of hiding it behind an expander.
  const showAllQuantizations = useChatRuntimeStore(
    (s) => s.showAllQuantizations,
  );
  // Shared with the Hub page: list only models sized within the device budget.
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);
  // Repos the user clicked to collapse while expand-by-default is on, and the
  // ones they clicked back open. Kept in memory only, so both reset on reload
  // (and when the setting is toggled).
  const [collapsedGgufState, setCollapsedGgufState] = useState<{
    expandQuantizations: boolean;
    value: Set<string>;
    reopened: Set<string>;
  }>(() => ({ expandQuantizations, value: new Set(), reopened: new Set() }));
  const expansionMatchesSetting =
    collapsedGgufState.expandQuantizations === expandQuantizations;
  const collapsedGguf = expansionMatchesSetting
    ? collapsedGgufState.value
    : new Set<string>();
  const reopenedGguf = expansionMatchesSetting
    ? collapsedGgufState.reopened
    : new Set<string>();
  const isGgufExpanded = useCallback(
    (id: string) =>
      expandQuantizations ? !collapsedGguf.has(id) : expandedGguf === id,
    [expandQuantizations, collapsedGguf, expandedGguf],
  );
  // Toggle a repo's quantizations: flip the collapse set when expand-by-default
  // is on, otherwise drive the single-open expandedGguf state.
  const toggleGgufExpanded = useCallback(
    // `showing` is what the row actually renders, which is not the collapse
    // set alone: a row held back by its sole-quant probe shows nothing, and a
    // click on it should open it rather than collapse what is already hidden.
    (id: string, showing = isGgufExpanded(id)) => {
      if (!expandQuantizations) {
        setExpandedGguf((prev) => (prev === id ? null : id));
        return;
      }
      setCollapsedGgufState((prev) => {
        const matches = prev.expandQuantizations === expandQuantizations;
        const next = toggleAutoExpandedRow(
          {
            collapsed: matches ? prev.value : new Set(),
            reopened: matches ? prev.reopened : new Set(),
          },
          { repoId: id, showing },
        );
        return {
          expandQuantizations,
          value: next.collapsed,
          reopened: next.reopened,
        };
      });
    },
    [expandQuantizations, isGgufExpanded],
  );

  const [pinnedCollapsed, setPinnedCollapsed] = useState(false);
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
  // The Other models header; the directions icon on the Unsloth header scrolls
  // here.
  const otherModelsSectionRef = useRef<HTMLDivElement>(null);
  const scrollToOtherModels = useCallback(() => {
    setOtherModelsCollapsed(false);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        otherModelsSectionRef.current?.scrollIntoView({
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

  const pickerInventory = useChatPickerInventory({ enabled: true });
  const {
    cachedGguf,
    cachedModels,
    cachedReady,
    refreshInventory,
    refreshInventoryIfOlderThan,
  } = pickerInventory;
  const cachedReadyAtMount = useRef(cachedReady);
  const lmStudioModels = useMemo(
    () =>
      sortLmStudio(
        pickerInventory.localModels.filter((m) => m.source === "lmstudio"),
      ),
    [pickerInventory.localModels],
  );
  const localDirModels = useMemo(
    () => pickerInventory.localModels.filter((m) => m.source === "models_dir"),
    [pickerInventory.localModels],
  );
  const customFolderModels = useMemo(
    () => pickerInventory.localModels.filter((m) => m.source === "custom"),
    [pickerInventory.localModels],
  );
  useEffect(() => {
    _cachedGgufCache = cachedGguf;
    _cachedModelsCache = cachedModels;
    _lmStudioCache = lmStudioModels;
    _localDirCache = localDirModels;
    _customFolderCache = customFolderModels;
  }, [
    cachedGguf,
    cachedModels,
    lmStudioModels,
    localDirModels,
    customFolderModels,
  ]);
  const [updateConflictKey, setUpdateConflictKey] = useState<string | null>(
    null,
  );
  const updateTransportConflict = useDownloadManagerStore((state) =>
    updateConflictKey
      ? (state.conflicts[updateConflictKey]?.info ?? null)
      : null,
  );
  const cancelUpdateConflict = useCallback(() => {
    if (updateConflictKey) downloadManager.cancelConflict(updateConflictKey);
    setUpdateConflictKey(null);
  }, [updateConflictKey]);
  const resumeUpdateConflict = useCallback(() => {
    if (!updateConflictKey) return;
    downloadManager.resumeConflict(updateConflictKey);
    setUpdateConflictKey(null);
  }, [updateConflictKey]);
  const restartUpdateConflict = useCallback(() => {
    if (!updateConflictKey) return;
    downloadManager.restartConflict(updateConflictKey);
    setUpdateConflictKey(null);
  }, [updateConflictKey]);

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
    void pickerInventory.refreshInventory();
  }, [pickerInventory.refreshInventory]);

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
      // From the folder browser's "Use this folder": the typed-input panel is
      // closed, so surface failures (denylisted path, sandbox 403) via toast.
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
    void pickerInventory.refreshInventory();
  }, [pickerInventory.refreshInventory]);

  // Updates run as managed downloads (Downloads panel: progress + Cancel), not a blocking
  // call. The worker pulls only changed blobs, so the cached copy stays usable until done.
  const startManagedUpdate = useCallback(
    (repoId: string, variant: string, expectedBytes: number) => {
      return downloadManager
        .requestStart({
          kind: "model",
          repoId,
          variant,
          expectedBytes,
        })
        .then((outcome) => {
          if (outcome === "conflict") {
            setUpdateConflictKey(jobKeyOf("model", repoId, variant));
          } else if (outcome === "busy") {
            // A sibling variant/snapshot for this repo is already downloading,
            // so this update did not start. Say so instead of closing the
            // dialog as if it began and leaving the cached copy stale.
            toast.info("A download for this model is already in progress", {
              description: "Try updating again once it finishes.",
            });
          } else if (outcome === "error") {
            throw new Error("Failed to start update");
          }
        });
    },
    [],
  );

  const updateGgufVariant = useCallback(
    (repoId: string, quant: string, expectedBytes: number) =>
      startManagedUpdate(repoId, quant, expectedBytes),
    [startManagedUpdate],
  );

  useEffect(() => {
    refreshScanFolders();
    listRecommendedFolders()
      .then(setRecommendedFolders)
      .catch(() => {});
  }, [refreshScanFolders]);

  useEffect(() => {
    if (!shouldRefreshPickerInventoryOnMount(cachedReadyAtMount.current)) {
      return;
    }
    void refreshInventoryIfOlderThan(INVENTORY_FRESHNESS_WINDOW_MS);
  }, [refreshInventoryIfOlderThan]);

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

  // Drop models Unsloth cannot run for chat. A task-scoped picker wants exactly the tasks the chat classifier calls unsupported, so it gates on the task.
  const isChatSupported = useCallback(
    (r: HfModelResult) => {
      // Image/Video tab (task set): only task-matching, non-editing results.
      if (task) return taskMatchesFilter(r.pipelineTag, task) && !isImageEditModel(r.id);
      return classifyUnslothSupport({
        modelId: r.id,
        pipelineTag: r.pipelineTag,
        tags: r.tags,
        libraryName: r.libraryName,
        quantMethod: r.quantMethod,
        deviceType,
      }).status !== "unsupported";
    },
    [deviceType, task],
  );

  const recommendedIds = useMemo(() => {
    const all = dedupe([...models.map((model) => model.id), value ?? ""])
      .filter((id) => !isHiddenModelId(id))
      .filter((id) => !downloadedSet.has(id.toLowerCase()))
      // Task-scoped pages load single-file GGUF only; chat-only keeps runnable formats (GGUF anywhere, plus MLX/safetensors on Mac).
      // A curated artifact stays listed whatever its format: loadSpecFor knows how to load each, and a GGUF-only rule hid every non-GGUF curated model.
      .filter((id) =>
        task
          ? isKnownGgufRepo(id) || Boolean(catalog && artifactForRepoId(id, catalog))
          : !chatOnly || isRecommendableFormat(id, isKnownGgufRepo(id), isMac),
      )
      // Member repos of a catalog group would collapse into the canonical group row, but nothing renders those rows yet and a
      // task-scoped picker's `models` is exactly group members, so suppressing them emptied Recommended. Keep them until the grouped UI lands.
      .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id));
    // Sort: GGUFs first, then hub models
    const gguf: string[] = [];
    const hub: string[] = [];
    for (const id of all) {
      if (isKnownGgufRepo(id)) gguf.push(id);
      else hub.push(id);
    }
    return [...gguf, ...hub];
  }, [
    models,
    value,
    downloadedSet,
    chatOnly,
    isKnownGgufRepo,
    isMac,
    task,
    catalog,
  ]);

  const showHfSection = debouncedQuery.trim().length > 0;

  // Independent sort for each local section's inline dropdown.
  const [downloadedSort, setDownloadedSort] = useState<LocalSortKey>("recent");
  const [customSort, setCustomSort] = useState<LocalSortKey>("recent");
  // Format filter toggle for the Unsloth listing.
  const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");
  // MLX and Safetensors repos are one download, so a row can name its size.
  const hubRowsShowSize =
    formatFilter === "mlx" || formatFilter === "safetensors";

  // Paint curated rows before any request, so a task-scoped picker whose models
  // are already in memory does not sit on a spinner for a round trip.
  const catalogSeedRows = useMemo<HfModelResult[]>(() => {
    if (!task) return [];
    return dedupe(models.map((model) => model.id))
      .filter((id) => !isHiddenModelId(id))
      .filter((id) => !isMobileVariant(id))
      .filter((id) => !isImageEditModel(id))
      .filter((id) => {
        const isG = isKnownGgufRepo(id);
        return formatFilter === "all"
          ? isRecommendableFormat(id, isG, isMac)
          : matchesFormatFilter(id, isG, formatFilter);
      })
      .map((id) => ({
        id,
        downloads: 0,
        likes: 0,
        isGguf: isKnownGgufRepo(id),
        // Size from the catalog, not an id "<n>B" guess: the guess is missing for
        // most curated ids and wrong for others (Wan2.2-TI2V-5B is 30 GB, not 2),
        // and non-unsloth ids never get a listing row to correct it.
        curatedSizeBytes: catalog ? curatedSizeBytesFor(id, catalog) : undefined,
        // Same reason the size is curated: a seed the listing does not return has no
        // other source for its param chip, and most curated ids carry no "<n>B" token.
        totalParams: catalog ? curatedTotalParamsFor(id, catalog) : undefined,
      }));
  }, [catalog, models, formatFilter, isKnownGgufRepo, isMac, task]);

  const catalogSeedIds = useMemo(
    () => catalogSeedRows.map((row) => row.id),
    [catalogSeedRows],
  );

  // Recommended suggests GGUF anywhere; on Mac also MLX and safetensors. The
  // "recommended" sort also drops models too big for the device. Already-
  // downloaded models stay visible (badged), never hidden.
  const recommendedRows = useMemo(() => {
    const keep = (r: HfModelResult) =>
      !isHiddenModelId(r.id) &&
      !isMobileVariant(r.id) &&
      isChatSupported(r) &&
      // No pick: device-recommended formats (GGUF, plus MLX on Mac). A pick wins.
      (formatFilter === "all"
        ? isRecommendableFormat(r.id, r.isGguf, isMac)
        : matchesFormatFilter(r.id, r.isGguf, formatFilter)) &&
      // Task pages load single-file GGUF, plus curated artifacts in any format.
      (!task || r.isGguf || Boolean(catalog && artifactForRepoId(r.id, catalog)));
    // Members are not filtered here (see recommendedIds): it dropped them from
    // Hub search too. "recommended" always device-filters; the "Fits on device"
    // tick extends that to the other sorts.
    const deviceFiltered = recommendedSort === "recommended" || fitOnDeviceOnly;
    const taskScoped = Boolean(task);
    const rowGpu = loadScopedGpu(gpu, taskScoped);
    const rowInferenceGpu = loadScopedGpu(inferenceGpu, taskScoped);
    const fits = (r: HfModelResult) =>
      // Downloaded models show regardless of fit.
      downloadedSet.has(r.id.toLowerCase()) ||
      hfModelFitsDevice(r, r.isGguf ? rowInferenceGpu : rowGpu);
    return orderRecommendedRows({
      seeds: catalogSeedRows,
      results: recommendedSearch.results,
      keep,
      deviceFiltered,
      fits,
    });
  }, [
    recommendedSearch.results,
    catalogSeedRows,
    downloadedSet,
    recommendedSort,
    fitOnDeviceOnly,
    formatFilter,
    isMac,
    gpu,
    inferenceGpu,
    isChatSupported,
    task,
    catalog,
  ]);

  // Per-row meta + VRAM badge from the recommended listing's own metadata, with the
  // curated seeds behind it: a listing row wins wherever there is one, and a curated
  // row the listing never returns still gets its size chip instead of rendering bare.
  const recommendedMeta = useMemo(() => {
    const map = new Map<
      string,
      { meta: string | null; status: VramFitStatus | null; est: number }
    >();
    for (const r of [...recommendedSearch.results, ...catalogSeedRows]) {
      if (map.has(r.id)) continue;
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
        const hasDeviceBudget =
          inferenceGpu.budgetKnown ||
          inferenceGpu.memoryTotalGb > 0 ||
          inferenceGpu.systemRamAvailableGb > 0;
        const exceeds =
          hasDeviceBudget &&
          sizeBytes != null &&
          !fitsDevice({
            sizeBytes,
            gpuGb: inferenceGpu.memoryTotalGb,
            systemRamGb: inferenceGpu.systemRamAvailableGb,
            budgetKnown: inferenceGpu.budgetKnown,
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
  }, [
    recommendedSearch.results,
    catalogSeedRows,
    isKnownGgufRepo,
    gpu,
    inferenceGpu,
  ]);

  // Tag-accurate capabilities keyed by repo id, pooled from both HF listings, then the
  // catalog for curated ids neither listing returned. Rows look it up by id and fall
  // back to repo-name detection when absent, which cannot see an audio track a name
  // does not mention. Listings first: real tags outrank curated data.
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
    if (catalog) {
      for (const row of catalogSeedRows) {
        if (map.has(row.id)) continue;
        const curated = curatedCapabilitiesFor(row.id, catalog);
        if (curated) map.set(row.id, curated);
      }
    }
    return map;
  }, [results, recommendedSearch.results, catalog, catalogSeedRows]);

  // Ordered by the On Device dropdown (recent/download date/size/name). The gate keeps diffusion GGUFs in the Images/Video picker and out of chat.
  const sortedCachedGguf = useMemo(
    () =>
      sortCachedRepos(
        cachedGguf.filter((c) => passesTaskGate(c.task, c.repo_id, task)),
        downloadedSort,
        loadTimes,
      ),
    [cachedGguf, downloadedSort, loadTimes, task],
  );
  // Cached non-GGUF repos. In chat, passesTaskGate drops diffusers image repos; the Images picker keeps them, but only unsloth-hosted ones this backend can load. Base repos are cached as dependencies and fail the trust gate.
  const sortedCachedModels = useMemo(
    () =>
      sortCachedRepos(
        cachedModels.filter(
          (c) =>
            // A partially-downloaded snapshot is not on-device: listing it as loadable errors or triggers a silent multi-GB re-fetch.
            !c.partial &&
            passesTaskGate(c.task, c.repo_id, task) &&
            // Diffusion pickers: unsloth repos plus any repo the backend can LOAD. Gate on a curated ARTIFACT (what loadSpecFor resolves), not a group-key match: a base / uncurated-quant sibling matches the group by key but dead-ends at the trust gate.
            // An unsloth repo must also be a full pipeline: the fall-through loads uncataloged rows as "pipeline", and from_pretrained on a single-file checkpoint repo fails. Curated single-file artifacts stay, since loadSpecFor carries their filename.
            (!task ||
              (isUnslothRepoId(c.repo_id) && !c.single_file) ||
              (catalog ? artifactForRepoId(c.repo_id, catalog) !== null : false)),
        ),
        downloadedSort,
        loadTimes,
      ),
    [cachedModels, downloadedSort, loadTimes, task, catalog],
  );
  // Task-scoped loads put the whole pipeline on ONE device, so quant fit uses the device the load lands on (the lowest visible ordinal), not the multi-GPU sum or the largest card: sizing against the bigger card OOMs the smaller one. Chat keeps the sum.
  // The source is picked per row (a GGUF row sizes against the inference GPU, anything else against the system view); this only decides how much of it a row may claim.
  const expanderGpuGbFrom = (info: typeof inferenceGpu) =>
    info.available ? loadScopedGpu(info, Boolean(task)).memoryTotalGb : undefined;
  const expanderGpuGb = expanderGpuGbFrom(inferenceGpu);
  const expanderSystemGpuGb = expanderGpuGbFrom(gpu);

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
            // The backend tags every local model with its task for exactly this: on the Images/Video pages a chat GGUF must not be offered.
            passesTaskGate(m.task, m.model_id ?? m.id, task) &&
            localModelMatchesFormat(m, formatFilter) && matchesLocalQuery(m),
        ),
        downloadedSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lmStudioModels, downloadedSort, formatFilter, loadTimes, localQuery, task],
  );
  // Local ./models entries. Chat-only Unsloth runs GGUF (any host) and MLX (Mac only), so raw checkpoints there are hidden (mirrors the cached
  // non-GGUF rule); an MLX build a Mac user dropped in stays selectable. A task-scoped picker (Images) is exempt: the image backend loads local pipelines even there.
  const sortedLocalDir = useMemo(
    () =>
      sortLocalModels(
        localDirModels.filter(
          (m) =>
            passesTaskGate(m.task, m.model_id ?? m.id, task) &&
            (!chatOnly ||
              Boolean(task) ||
              localModelIsGguf(m) ||
              (isMac && localModelIsMlx(m))) &&
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
      isMac,
      loadTimes,
      localQuery,
      chatOnly,
      task,
    ],
  );
  const sortedCustomFolderModels = useMemo(
    () =>
      sortLocalModels(
        customFolderModels.filter(
          (m) =>
            passesTaskGate(m.task, m.model_id ?? m.id, task) &&
            localModelMatchesFormat(m, formatFilter) && matchesLocalQuery(m),
        ),
        customSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [customFolderModels, customSort, formatFilter, loadTimes, localQuery, task],
  );

  // Chat cannot load a diffusion model but the Images/Video pages can, so a pick routes to the page that runs it instead of hiding it or letting it 400. Task-scoped pickers select normally.
  const navigateToPage = useNavigate();
  const diffusionTaskById = useMemo(() => {
    const byId = new Map<string, string>();
    const put = (id: string | null | undefined, t: string | null | undefined) => {
      if (id && t) byId.set(id.toLowerCase(), t);
    };
    for (const c of cachedGguf) put(c.repo_id, c.task);
    for (const c of cachedModels) put(c.repo_id, c.task);
    // Both ids: a local row's click passes m.id (a filesystem path for models_dir / LM Studio entries) while m.model_id is its HF-style name, so keying on one alone makes the lookup below miss.
    const putLocal = (m: LocalModelInfo) => {
      put(m.id, m.task);
      put(m.model_id, m.task);
    };
    for (const m of lmStudioModels) putLocal(m);
    for (const m of localDirModels) putLocal(m);
    for (const m of customFolderModels) putLocal(m);
    return byId;
  }, [cachedGguf, cachedModels, lmStudioModels, localDirModels, customFolderModels]);

  const onSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      if (!task) {
        const page = diffusionPageForTask(diffusionTaskById.get(id.toLowerCase()));
        if (page) {
          void navigateToPage({
            to: `/${page}`,
            // `quant` is used verbatim as the gguf filename, so a label like "Q4_K_M" rides ggufQuant instead; dropping it
            // made every non-curated GGUF repo arrive as a bare repo id.
            search: diffusionRouteSearch(id, meta),
          });
          return;
        }
      }
      onSelectProp(id, meta);
    },
    [task, diffusionTaskById, navigateToPage, onSelectProp],
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
    // Keep the format filter active while searching so the dropdown stays
    // consistent with the no-query branch (Safetensors selected shouldn't show
    // GGUF downloads just because the user typed).
    return sortedCachedGguf.filter(
      (c) =>
        matchesFormatFilter(c.repo_id, true, formatFilter) &&
        normalizeForSearch(c.repo_id).includes(q),
    );
  }, [sortedCachedGguf, showHfSection, debouncedQuery, formatFilter]);
  const visibleCachedModels = useMemo(() => {
    if (!showHfSection)
      return sortedCachedModels.filter((c) =>
        matchesFormatFilter(c.repo_id, false, formatFilter),
      );
    const q = normalizeForSearch(debouncedQuery.trim());
    return sortedCachedModels.filter(
      (c) =>
        matchesFormatFilter(c.repo_id, false, formatFilter) &&
        normalizeForSearch(c.repo_id).includes(q),
    );
  }, [sortedCachedModels, showHfSection, debouncedQuery, formatFilter]);

  // Non-GGUF cached rows are not shown in chat-only mode, so the empty-state logic must use this (not visibleCachedModels) or the picker can go
  // blank. A task-scoped picker (Images) is exempt: the image backend loads local diffusers/safetensors pipelines even on chat-only hosts.
  const visibleCachedModelRows = chatOnly && !task ? [] : visibleCachedModels;

  // Unfiltered list, so typing a query doesn't re-run resolution.
  const soleQuants = useSoleDownloadedQuants(sortedCachedGguf, {
    enabled: section === "downloaded" && !showAllQuantizations,
    hfToken: hfToken || undefined,
  });

  // Pinned entries surface in their own section above the Unsloth heading.
  // GGUF quants pin individually and their repo stays listed below; non-GGUF
  // repos pin whole and leave the Unsloth / Other models groups.
  const pinnedIds = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const pinnedSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);

  // Candidate pins whose repo still exists in the cache. Per-quant validation
  // below is needed because deleting one variant can leave a sibling cached.
  const pinnedQuantCandidates = useMemo(() => {
    // The existence check ignores the text query (keeps the format filter) so a
    // pinned quant stays findable by quant name; querying visibleCachedGguf would
    // drop the repo before the `${repoId} ${quant}` predicate could surface it.
    const cached = new Set(
      sortedCachedGguf
        .filter((c) => matchesFormatFilter(c.repo_id, true, formatFilter))
        .map((c) => c.repo_id),
    );
    return pinnedQuantEntries(pinnedIds).filter((entry) =>
      cached.has(entry.repoId),
    );
  }, [pinnedIds, sortedCachedGguf, formatFilter]);
  const [pinnedQuantValidation, setPinnedQuantValidation] = useState<{
    validated: boolean;
    downloaded: ReadonlySet<string>;
  }>({ validated: false, downloaded: new Set() });
  const prunePinnedQuantValidation = useCallback(
    (repoId: string, quant: string) => {
      const key = pinKey(repoId, quant);
      setPinnedQuantValidation((prev) => {
        if (!prev.downloaded.has(key)) return prev;
        const downloaded = new Set(prev.downloaded);
        downloaded.delete(key);
        return { ...prev, downloaded };
      });
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    const repoIds = Array.from(
      new Set(pinnedQuantCandidates.map((entry) => entry.repoId)),
    );
    if (repoIds.length === 0) return;

    void Promise.all(
      repoIds.map(async (repoId) => {
        try {
          const response = await listGgufVariantsCached(
            repoId,
            hfToken || undefined,
            { preferLocalCache: true },
          );
          return normalizeGgufVariantsResponse(response)
            .variants.filter((variant) => variant.downloaded === true)
            .map((variant) => pinKey(repoId, variant.quant));
        } catch {
          // If the backend cannot verify a quant, hiding the direct-load row
          // is safer than claiming a missing file is downloaded.
          return [];
        }
      }),
    ).then((groups) => {
      if (!cancelled) {
        setPinnedQuantValidation({
          validated: true,
          downloaded: new Set(groups.flat()),
        });
      }
    });

    return () => {
      cancelled = true;
    };
  }, [hfToken, pinnedQuantCandidates]);
  const downloadedPinnedQuantKeys = useMemo<ReadonlySet<string>>(
    () =>
      pinnedQuantValidation.validated
        ? pinnedQuantValidation.downloaded
        : new Set(),
    [pinnedQuantValidation],
  );

  // Verified downloaded quants, in pin order and filtered by repo id or quant.
  const pinnedQuants = useMemo(() => {
    const q = normalizeForSearch(debouncedQuery.trim());
    return pinnedQuantCandidates.filter(
      (entry) =>
        downloadedPinnedQuantKeys.has(pinKey(entry.repoId, entry.quant)) &&
        (!q ||
          normalizeForSearch(`${entry.repoId} ${entry.quant}`).includes(q)),
    );
  }, [debouncedQuery, downloadedPinnedQuantKeys, pinnedQuantCandidates]);

  const pinnedCachedModelRows = useMemo(
    () =>
      visibleCachedModelRows.filter((c) => pinnedSet.has(pinKey(c.repo_id))),
    [visibleCachedModelRows, pinnedSet],
  );

  const pinnedRows = useMemo(() => {
    const rank = makePinRank(pinnedIds);
    const rows = [
      ...pinnedQuants.map((entry) => ({
        key: pinKey(entry.repoId, entry.quant),
        entry,
        model: null,
      })),
      ...pinnedCachedModelRows.map((model) => ({
        key: pinKey(model.repo_id),
        entry: null,
        model,
      })),
    ];
    rows.sort((a, b) => rank(a.key) - rank(b.key));
    return rows;
  }, [pinnedIds, pinnedQuants, pinnedCachedModelRows]);

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
    () =>
      visibleCachedModelRows.filter(
        (c) => isUnslothRepoId(c.repo_id) && !pinnedSet.has(pinKey(c.repo_id)),
      ),
    [visibleCachedModelRows, pinnedSet],
  );
  const otherCachedModelRows = useMemo(
    () =>
      visibleCachedModelRows.filter(
        (c) => !isUnslothRepoId(c.repo_id) && !pinnedSet.has(pinKey(c.repo_id)),
      ),
    [visibleCachedModelRows, pinnedSet],
  );

  // Param counts come straight off the unsloth listings the picker already
  // loaded, so no extra per-id fetch is needed for the VRAM badges.
  const recommendedParamCountById = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of [...results, ...recommendedSearch.results]) {
      if (r.totalParams) map.set(r.id, r.totalParams);
    }
    return map;
  }, [results, recommendedSearch.results]);

  // Shared by both search lists so a curated id one drops cannot return via the
  // other as a raw Hub row.
  const searchRowFits = useCallback(
    (row: {
      id: string;
      totalParams?: number;
      estimatedSizeBytes?: number;
      curatedSizeBytes?: number;
    }) =>
      searchRowFitsDevice(
        {
          ...row,
          // Curated params last, same rule as the curated size below: a listing
          // total wins, but a repo no listing returns must still be sizable or
          // `requireKnown` hides it from search while the unfiltered Recommended
          // list, which reads the seed row's own metadata, keeps painting it.
          totalParams:
            row.totalParams ??
            recommendedParamCountById.get(row.id) ??
            (catalog ? curatedTotalParamsFor(row.id, catalog) : undefined),
        },
        {
          isGguf: isKnownGgufRepo(row.id),
          curatedSizeBytes: catalog
            ? curatedSizeBytesFor(row.id, catalog)
            : undefined,
          gpu,
          inferenceGpu,
          taskScoped: Boolean(task),
        },
      ),
    [
      catalog,
      gpu,
      inferenceGpu,
      isKnownGgufRepo,
      recommendedParamCountById,
      task,
    ],
  );

  // Recommended models that match the current search query
  const filteredRecommendedIds = useMemo(() => {
    if (!showHfSection) return [];
    const q = normalizeForSearch(debouncedQuery.trim());
    return (
      // Seeds included: recommendedIds hides downloaded models, which the unfiltered
      // Recommended list still paints, so without them a curated pick vanishes from
      // search the moment it is on disk unless a Hub listing row happens to carry it.
      searchableRecommendedIds(catalogSeedIds, recommendedIds)
        .filter((id) => normalizeForSearch(id).includes(q))
        .filter((id) =>
          matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
        )
        // Curated defaults obey the fit toggle like the live HF rows, else large
        // defaults resurface in search results with the filter on.
        .filter(
          (id) =>
            !fitOnDeviceOnly ||
            downloadedSet.has(id.toLowerCase()) ||
            searchRowFits({ id }),
        )
    );
  }, [
    showHfSection,
    debouncedQuery,
    catalogSeedIds,
    recommendedIds,
    formatFilter,
    isKnownGgufRepo,
    fitOnDeviceOnly,
    downloadedSet,
    searchRowFits,
  ]);

  const recommendedSet = useMemo(
    () => new Set(filteredRecommendedIds),
    [filteredRecommendedIds],
  );

  const hfIds = useMemo(() => {
    // Only the Unsloth tab searches the HF listing, and only Unsloth models.
    if (!showHfSection || section !== "recommended") return [];
    return (
      results
        .filter(isChatSupported)
        .filter(
          (r) =>
            !fitOnDeviceOnly ||
            downloadedSet.has(r.id.toLowerCase()) ||
            searchRowFits(r),
        )
        .map((result) => result.id)
        .filter((id) => !isHiddenModelId(id))
        .filter((id) => id.toLowerCase().startsWith("unsloth/"))
        .filter((id) => !recommendedSet.has(id))
        // Chat-only keeps runnable formats: GGUF anywhere, plus MLX/safetensors
        // on Mac (matches the empty Recommended view so search stays consistent).
        .filter(
          (id) =>
            !chatOnly || isRecommendableFormat(id, isKnownGgufRepo(id), isMac),
        )
        .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id))
        .filter((id) =>
          matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
        )
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
    fitOnDeviceOnly,
    downloadedSet,
    searchRowFits,
    isMac,
  ]);

  const hubOptionKeys = useMemo(() => {
    const keys: string[] = [];

    // Pinned rows sit above the Unsloth heading on the On Device tab.
    if (
      section === "downloaded" &&
      cachedReady &&
      !pinnedCollapsed &&
      pinnedRows.length > 0
    ) {
      keys.push(
        ...pinnedRows.map((row) =>
          row.entry
            ? makeModelOptionKey("pinned-quant", row.key)
            : makeModelOptionKey("downloaded-model", row.model.repo_id),
        ),
      );
    }

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
    pinnedRows,
    pinnedCollapsed,
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
  // per loaded page re-attaches the observer so a heavily filtered list keeps
  // paging until the viewport fills; fetchMore is a no-op while a page is in flight.
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
        // Cached repos load now; uncached ones download via the Hub manager.
        onSelect(id, {
          source: "hub",
          isLora: false,
          isDownloaded: downloadedSet.has(id.toLowerCase()),
        });
      }
    },
    [onSelect, isKnownGgufRepo, downloadedSet],
  );

  // On Device owns the downloaded and custom-folder models; the Unsloth tab
  // searches the HF listing (below). Both filter locally by the query.
  const showDownloaded = section === "downloaded";
  const showCustom = section === "downloaded";
  const showRecommendedSection = !showHfSection && section === "recommended";
  const downloadedEmpty =
    pinnedRows.length === 0 &&
    visibleCachedGguf.length === 0 &&
    visibleCachedModelRows.length === 0 &&
    sortedLmStudio.length === 0 &&
    sortedLocalDir.length === 0 &&
    // Fine-tuned models are on-device too: don't show the empty state above a
    // non-empty Fine-tuned section.
    fineTunedRows.length === 0;

  // Sort dropdown inline right of the section toggle; options depend on the tab
  // and stay visible while searching. Fixed width matches the Search Hub button
  // so it and the format dropdown line up. Trigger label clips; the menu shows full.
  const sortTriggerClassName =
    "h-(--picker-control-h) w-(--picker-control-w) shrink-0 justify-between pr-2.5 !border-0 text-xs [&>span]:!text-clip";
  // Tighter menu (less padding, text-xs) matching the trigger. Keep the option's
  // right padding so the selected-item checkmark never overlaps the label.
  const sortMenuContentClassName =
    "!p-1 !rounded-[14px] [&_[role=option]]:!pl-2 [&_[role=option]]:!py-1.5 [&_[role=option]]:!text-xs [&_[role=option]]:!rounded-[10px]";
  // Device-fit toggle inside the sort menu (shared with the Hub page). The whole
  // row is the button: a Checkbox renders as a <button> and label-click forwarding
  // to it is unreliable, so the row owns the toggle and the Checkbox is presentational.
  const fitOnDeviceFooter = (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          role="checkbox"
          aria-checked={fitOnDeviceOnly}
          onClick={() => setFitOnDeviceOnly(!fitOnDeviceOnly)}
          className="flex w-full cursor-pointer select-none items-center gap-1.5 rounded-[10px] px-2 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
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
        Hides models larger than this device's memory budget. Downloaded models
        stay visible.
      </TooltipContent>
    </Tooltip>
  );
  // Sort icon + selected label inside the trigger pill.
  const sortTriggerContent = (label: ReactNode) => (
    <span className="flex items-center gap-1">
      <HugeiconsIcon
        icon={ArrowUpDownIcon}
        strokeWidth={1.75}
        className="size-3.5 shrink-0 text-muted-foreground"
      />
      <span className="truncate">{label}</span>
    </span>
  );
  // On Device / Custom rows are already on disk, so the device-fit filter
  // only applies to the Unsloth listing.
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
        triggerContent={sortTriggerContent(
          RECOMMENDED_SORT_OPTIONS.find((o) => o.value === recommendedSort)
            ?.label ?? recommendedSort,
        )}
        footer={fitOnDeviceFooter}
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
        triggerContent={sortTriggerContent(
          LOCAL_SORT_OPTIONS.find((o) => o.value === downloadedSort)?.label ??
            downloadedSort,
        )}
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
        triggerContent={sortTriggerContent(
          LOCAL_SORT_OPTIONS.find((o) => o.value === customSort)?.label ??
            customSort,
        )}
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
  // The Connected layout uses a wider box, so it drops the search inset to keep
  // Search Hub on the last dropdown's edge while the right gap matches the left.
  const hasConnected = externalModels.length > 0;
  // The Other models section and its shortcut only show with non-Unsloth downloads.
  const hasOtherModels =
    otherCachedGguf.length > 0 || otherCachedModelRows.length > 0;

  const downloadedRowButtonClassName =
    "bg-transparent pr-1 hover:bg-transparent focus-visible:bg-transparent dark:bg-transparent dark:hover:bg-transparent dark:focus-visible:bg-transparent";
  // Not focus-within: the dots menu returns focus to its trigger on close, so
  // the row stayed lit after the pointer left. Keyboard focus and an open menu
  // still light it.
  const downloadedRowShellClassName = (selected: boolean) =>
    cn(
      "group flex items-center rounded-full transition-colors hover:bg-[#ececec] has-[:focus-visible]:bg-[#ececec] has-[[data-state=open]]:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)] dark:has-[:focus-visible]:bg-[var(--sidebar-accent)] dark:has-[[data-state=open]]:bg-[var(--sidebar-accent)]",
      selected && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
    );

  // A pinned quant: repo name with the quant as a grey chip. One click loads
  // that quant directly, no expansion needed.
  const renderPinnedQuantRow = (entry: { repoId: string; quant: string }) => {
    const optionKey = makeModelOptionKey(
      "pinned-quant",
      pinKey(entry.repoId, entry.quant),
    );
    const isSelected =
      value === entry.repoId && activeGgufVariant === entry.quant;
    const isLoaded =
      modelIdsMatchForPicker(loadedModelId, entry.repoId) &&
      !ggufVariantsMatchForPicker(activeGgufVariant, null) &&
      ggufVariantsMatchForPicker(activeGgufVariant, entry.quant);
    return (
      <div key={optionKey} className={downloadedRowShellClassName(isSelected)}>
        {/* Through ModelRow, so a pinned quant lands in the same columns as
            the rows below it. */}
        <div className="min-w-0 flex-1">
          <ModelRow
            label={entry.repoId}
            tooltipText={`${entry.repoId} (${entry.quant})`}
            meta="GGUF"
            quantChip={entry.quant}
            alignMeta="device"
            selected={isSelected}
            loaded={isLoaded}
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() =>
              onSelect(entry.repoId, {
                source: "hub",
                isLora: false,
                ggufVariant: entry.quant,
                isDownloaded: true,
                // The row loads one quant, so it is a GGUF pick like the expander's; without this the pages asked for a
                // pipeline, which a GGUF repo rejects. No filename: the pin stores a label, resolved against the listing.
                isGguf: true,
              })
            }
            vramStatus={null}
            className={downloadedRowButtonClassName}
          />
        </div>
        <span className={ROW_ACTIONS_CLASS}>
          {onConfigure && (
            <ModelLoadSettingsAction
              ariaLabel={`Inference settings for ${entry.repoId} ${entry.quant}`}
              onConfigure={() =>
                onConfigure(entry.repoId, {
                  source: "hub",
                  isLora: false,
                  ggufVariant: entry.quant,
                  isDownloaded: true,
                  isGguf: true,
                })
              }
            />
          )}
          <ModelRowMenu
            ariaLabel={`More options for ${entry.repoId} ${entry.quant}`}
            cachePath={{ repoId: entry.repoId, variant: entry.quant }}
            pin={{
              pinned: true,
              pinLabel: "Pin to top",
              unpinLabel: "Unpin",
              onToggle: () => togglePinned(entry.repoId, entry.quant),
            }}
            del={{
              title: "Delete cached model?",
              // Same preview the Hub On Device row asks for, so a companion base an
              // installed image model still needs shows the reason and a disabled
              // Delete rather than an enabled one that comes back 400.
              impact: { repoId: entry.repoId, variant: entry.quant },
              description: (
                <>
                  This will remove{" "}
                  <span className="font-medium text-foreground">
                    {entry.repoId} ({entry.quant})
                  </span>{" "}
                  from disk. You can re-download it later.
                </>
              ),
              successMessage: `Deleted ${entry.repoId} ${entry.quant}`,
              disabled: deleteDisabled,
              onConfirm: async () => {
                await deleteCachedModel(
                  entry.repoId,
                  entry.quant,
                  hfToken || undefined,
                );
                refreshCachedLists();
                // The file is gone, so drop its pin too.
                togglePinned(entry.repoId, entry.quant);
              },
            }}
          />
        </span>
      </div>
    );
  };

  // One quant on disk with "Show all quantizations" off: the expander would
  // list just that quant, so the row carries it as a chip and loads it in one
  // click, like a pinned quant.
  const renderSoleQuantGgufRow = (
    c: (typeof visibleCachedGguf)[number],
    sole: SoleDownloadedQuant,
  ) => {
    const variant = sole.variant;
    const optionKey = makeModelOptionKey("downloaded-gguf", c.repo_id);
    // The row names one quant, so the repo running a different one is not it.
    const rowState = soleQuantRowState({
      pickerValue: value,
      repoId: c.repo_id,
      quant: variant.quant,
      loadedModelId,
      activeGgufVariant,
    });
    const isSelected = rowState.selected;
    const expectedBytes = ggufVariantExpectedBytes(variant);
    const isPinned = pinnedSet.has(pinKey(c.repo_id, variant.quant));
    const selectMeta: ModelSelectorChangeMeta = {
      source: "hub",
      isLora: false,
      loadId: c.load_id,
      ggufVariant: variant.quant,
      ggufFilename: variant.filename,
      isDownloaded: true,
      expectedBytes,
      isGguf: true,
    };
    return (
      <div key={c.repo_id} className={downloadedRowShellClassName(isSelected)}>
        <div className="min-w-0 flex-1">
          <ModelRow
            label={c.repo_id}
            tooltipText={localPathTooltip(c.repo_id, c.cache_path)}
            meta={`GGUF · ${formatBytes(variant.size_bytes)}`}
            quantChip={variant.quant}
            showVision={c.has_vision || sole.hasVision}
            selected={isSelected}
            loaded={rowState.loaded}
            alignMeta="device"
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() => onSelect(c.repo_id, selectMeta)}
            vramStatus={null}
            className={downloadedRowButtonClassName}
          />
        </div>
        <span className={ROW_ACTIONS_CLASS}>
          {onConfigure && (
            <ModelLoadSettingsAction
              ariaLabel={`Inference settings for ${c.repo_id} ${variant.quant}`}
              onConfigure={() => onConfigure(c.repo_id, selectMeta)}
            />
          )}
          <ModelRowMenu
            ariaLabel={`More options for ${c.repo_id} ${variant.quant}`}
            cachePath={{ repoId: c.repo_id, variant: variant.quant }}
            pin={{
              pinned: isPinned,
              pinLabel: "Pin to top",
              unpinLabel: "Unpin",
              onToggle: () => togglePinned(c.repo_id, variant.quant),
            }}
            del={{
              title: "Delete cached model?",
              impact: { repoId: c.repo_id, variant: variant.quant },
              description: (
                <>
                  This will remove{" "}
                  <span className="font-medium text-foreground">
                    {c.repo_id} ({variant.quant})
                  </span>{" "}
                  from disk. You can re-download it later.
                </>
              ),
              successMessage: `Deleted ${c.repo_id} ${variant.quant}`,
              disabled: deleteDisabled,
              onConfirm: async () => {
                await deleteCachedModel(
                  c.repo_id,
                  variant.quant,
                  hfToken || undefined,
                  c.cache_path || undefined,
                );
                // The file is gone, so drop its pin too.
                if (isPinned) togglePinned(c.repo_id, variant.quant);
                prunePinnedQuantValidation(c.repo_id, variant.quant);
                refreshCachedLists();
              },
            }}
          />
        </span>
      </div>
    );
  };

  // Shared row renderers so Downloaded (Unsloth) and Other models render alike.
  const renderDownloadedGgufRow = (c: (typeof visibleCachedGguf)[number]) => {
    const optionKey = makeModelOptionKey("downloaded-gguf", c.repo_id);
    const isSelected = value === c.repo_id;
    const soleQuant = soleQuants.quants.get(c.repo_id);
    if (soleQuant) return renderSoleQuantGgufRow(c, soleQuant);
    // Auto-expansion waits for the probe: expanding every row first would
    // mount an expander, and its remote listing, for repos about to collapse.
    const expanderOpen = shouldMountVariantExpander({
      expanded: isGgufExpanded(c.repo_id),
      autoExpand: expandQuantizations && !reopenedGguf.has(c.repo_id),
      soleQuantsPending: soleQuants.pending.has(c.repo_id),
    });
    return (
      <div key={c.repo_id}>
        <div className={downloadedRowShellClassName(isSelected)}>
          <div className="min-w-0 flex-1">
            <ModelRow
              label={c.repo_id}
              tooltipText={localPathTooltip(c.repo_id, c.cache_path)}
              meta="GGUF"
              showVision={c.has_vision ?? visionByRepo[c.repo_id]}
              alignMeta="device"
              selected={isSelected}
              loaded={isRuntimeLoadedModel(
                loadedModelId,
                activeGgufVariant,
                c.repo_id,
                "required",
              )}
              optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
              onClick={() => toggleGgufExpanded(c.repo_id, expanderOpen)}
              onArrowDownIntoChildren={
                expanderOpen
                  ? () => focusFirstChildOption(optionKey)
                  : undefined
              }
              vramStatus={null}
              className={downloadedRowButtonClassName}
            />
          </div>
          {/* Stands in for the other rows' buttons, so the tags line up. */}
          <span aria-hidden="true" className={cn(ROW_ACTIONS_CLASS, "h-6")} />
        </div>
        {expanderOpen && (
          <GgufVariantExpander
            repoId={c.repo_id}
            loadId={c.load_id}
            cachePath={c.cache_path}
            onDevice={true}
            allowPin={true}
            onHasVision={(v) => reportVision(c.repo_id, v)}
            onSelect={onSelect}
            resolveDownloadFootprint={resolveDownloadFootprint}
            onConfigure={onConfigure}
            hfToken={hfToken || undefined}
            parentOptionKey={optionKey}
            onNavigatePastStart={() => hubModelList.focusOption(optionKey)}
            onNavigatePastEnd={() => hubModelList.moveFocus(optionKey, "next")}
            gpuGb={expanderGpuGb}
            systemRamGb={inferenceGpu.systemRamAvailableGb || undefined}
            budgetKnown={inferenceGpu.budgetKnown}
            variantActions={{
              onUpdate: (quant, expectedBytes) =>
                updateGgufVariant(c.repo_id, quant, expectedBytes),
              updateDisabled: loadedModelId === c.repo_id,
              onDelete: async (quant) => {
                await deleteCachedModel(
                  c.repo_id,
                  quant,
                  hfToken || undefined,
                  c.cache_path || undefined,
                );
                prunePinnedQuantValidation(c.repo_id, quant);
                refreshCachedLists();
              },
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
    const isSelected = value === c.repo_id;
    return (
      <div key={c.repo_id} className={downloadedRowShellClassName(isSelected)}>
        <div className="min-w-0 flex-1">
          <ModelRow
            label={c.repo_id}
            hubUrl={hubRepoUrl(c.repo_id)}
            meta={`${isMlxId(c.repo_id) ? "MLX" : "Safetensors"} · ${formatBytes(
              c.size_bytes,
            )}`}
            selected={isSelected}
            alignMeta="device"
            loaded={isRuntimeLoadedModel(
              loadedModelId,
              activeGgufVariant,
              c.repo_id,
              "none",
            )}
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() =>
              onSelect(c.repo_id, {
                source: "hub",
                isLora: false,
                loadId: c.load_id,
                isDownloaded: true,
              })
            }
            vramStatus={null}
            className={downloadedRowButtonClassName}
          />
        </div>
        <span className={ROW_ACTIONS_CLASS}>
          {onConfigure && (
            <ModelLoadSettingsAction
              ariaLabel={`Inference settings for ${c.repo_id}`}
              onConfigure={() =>
                onConfigure(c.repo_id, {
                  source: "hub",
                  isLora: false,
                  loadId: c.load_id,
                  isDownloaded: true,
                  isGguf: false,
                })
              }
            />
          )}
          <ModelRowMenu
            ariaLabel={`More options for ${c.repo_id}`}
            cachePath={{ repoId: c.repo_id }}
            pin={{
              pinned: pinnedSet.has(pinKey(c.repo_id)),
              pinLabel: "Pin to top",
              unpinLabel: "Unpin",
              onToggle: () => togglePinned(c.repo_id),
            }}
            del={{
              title: "Delete cached model?",
              impact: { repoId: c.repo_id },
              description: (
                <>
                  This will remove{" "}
                  <span className="font-medium text-foreground">
                    {c.repo_id}
                  </span>{" "}
                  from disk. You can re-download it later.
                </>
              ),
              successMessage: `Deleted ${c.repo_id}`,
              disabled: deleteDisabled,
              onConfirm: async () => {
                await deleteCachedModel(
                  c.repo_id,
                  undefined,
                  hfToken || undefined,
                  c.cache_path || undefined,
                );
                if (pinnedSet.has(pinKey(c.repo_id))) {
                  togglePinned(c.repo_id);
                }
              },
              onDeleted: refreshCachedLists,
            }}
          />
        </span>
      </div>
    );
  };

  return (
    <>
      <div className="relative space-y-2">
        {/* A small right inset shortens the search bar so Search Hub lands on the
          last dropdown's right edge (none on the wider Connected box). */}
        <div
          className={cn(
            "flex items-center gap-2 pb-1",
            hasConnected ? "pr-0" : "pr-2",
          )}
        >
          <div className="relative flex-1">
            <HugeiconsIcon
              icon={Search01Icon}
              className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={
                section === "downloaded"
                  ? "Search local models"
                  : "Search Unsloth models"
              }
              data-model-picker-search-input={true}
              className="field-soft h-(--picker-control-h) border-0 pl-8 pr-8"
            />
            {isLoading && (
              <Spinner className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            )}
          </div>
          {onBrowseHub ? (
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  onClick={onBrowseHub}
                  aria-label="Search more models on the Hub"
                  className="hub-tab-toggle-pill flex h-(--picker-control-h) w-(--picker-control-w) shrink-0 items-center justify-center gap-[5px] rounded-full border-0 text-xs text-foreground transition-colors"
                >
                  <HugeiconsIcon
                    icon={DashboardCircleIcon}
                    className="size-4"
                  />
                  Search Hub
                </button>
              </TooltipTrigger>
              <TooltipContent>Search all models</TooltipContent>
            </Tooltip>
          ) : null}
        </div>

        {/* Keep the left-packed controls on one line while they fit, then wrap
          whole groups before their intrinsic widths cross the picker edge.
          Dropdowns hide on Connected. */}
        <div
          className={cn(
            "flex flex-wrap items-center gap-2",
            hasConnected ? "-mr-4" : "-mr-2",
          )}
        >
          {sectionToggle}
          {showConnected ? null : (
            <div className="flex max-w-full min-w-0 flex-wrap items-center gap-2">
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
            // List sits within the menu padding so gaps match; height tracks content
            // up to the cap. scroll-py + symmetric px keep the focus ring off the
            // overflow clip edges during keyboard nav.
            "model-list-scroll max-h-[335px] overflow-y-auto scroll-py-1.5 px-0.5 mr-1",
            listScrolled && "is-scrolled",
            listMoreBelow && "is-bottom-faded",
          )}
          {...hubModelList.listboxProps}
        >
          <div
            className={cn(
              // Keep row actions clear of overlay scrollbars, overflowing or not.
              "overlay-scrollbar-gutter",
              // On Device pulls the heading block tight to the controls; Recommended
              // keeps a little more top room above its first row.
              showDownloaded ? "pt-0" : "pt-[4px]",
              onEject ? "pb-[60px]" : "pb-4",
            )}
          >
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
                    {/* Wider than the On Device section labels: nothing
                        divides these groups but the gap. */}
                    <div className="flex items-center gap-2 px-2.5 pb-1 pt-5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
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
                          "flex w-full items-center rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
                          value === model.id &&
                            "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
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

                {/* Pinned quants and models sit above the Unsloth heading so
              favorites are always first. Filtered by the query like the
              sections below. */}
                {showDownloaded && pinnedRows.length > 0 ? (
                  <>
                    <ListLabel
                      icon={
                        <HugeiconsIcon icon={PinIcon} className="size-3.5" />
                      }
                      collapsed={pinnedCollapsed}
                      onToggle={() => setPinnedCollapsed((v) => !v)}
                    >
                      Pinned
                    </ListLabel>
                    {!pinnedCollapsed &&
                      pinnedRows.map((row) =>
                        row.entry
                          ? renderPinnedQuantRow(row.entry)
                          : renderDownloadedModelRow(row.model),
                      )}
                  </>
                ) : null}

                {/* Downloaded (Unsloth) stays visible (filtered) while searching. */}
                {showDownloaded &&
                (unslothCachedGguf.length > 0 ||
                  unslothCachedModelRows.length > 0) ? (
                  <>
                    <ListLabel
                      divider={pinnedRows.length > 0}
                      collapsed={downloadedCollapsed}
                      onToggle={() => setDownloadedCollapsed((v) => !v)}
                      action={
                        <>
                          {hasOtherModels ? (
                            <Tooltip delayDuration={0}>
                              <TooltipTrigger asChild={true}>
                                <button
                                  type="button"
                                  onClick={scrollToOtherModels}
                                  aria-label="Go to other models"
                                  className="shrink-0 rounded p-1 text-muted-foreground/60 transition-colors hover:text-foreground"
                                >
                                  <HugeiconsIcon
                                    icon={Flag01Icon}
                                    className="size-3"
                                  />
                                </button>
                              </TooltipTrigger>
                              <TooltipContent
                                side="bottom"
                                className="tooltip-compact"
                              >
                                Other non-Unsloth models
                              </TooltipContent>
                            </Tooltip>
                          ) : null}
                          {!task && (
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
                          )}
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
                      {/* Rows drop the unsloth/ prefix; the heading carries
                    it for the group. */}
                      Unsloth
                    </ListLabel>
                    {!downloadedCollapsed &&
                      unslothCachedGguf.map(renderDownloadedGgufRow)}
                    {!downloadedCollapsed &&
                      unslothCachedModelRows.map(renderDownloadedModelRow)}
                  </>
                ) : null}

                {/* Other models: non-Unsloth downloads, grouped just above
              Fine-tuned. Shown only when such models exist. */}
                {showDownloaded && hasOtherModels ? (
                  <div ref={otherModelsSectionRef}>
                    <ListLabel
                      divider={true}
                      icon={
                        <HugeiconsIcon icon={Flag01Icon} className="size-3.5" />
                      }
                      collapsed={otherModelsCollapsed}
                      onToggle={() => setOtherModelsCollapsed((v) => !v)}
                    >
                      Other models
                    </ListLabel>
                    {!otherModelsCollapsed &&
                      otherCachedGguf.map(renderDownloadedGgufRow)}
                    {!otherModelsCollapsed &&
                      otherCachedModelRows.map(renderDownloadedModelRow)}
                  </div>
                ) : null}

                {/* Fine-tuned models: a section above Custom Folders, always shown on On Device so the train shortcut has a target. Hidden under a task filter (e.g. Images). */}
                {section === "downloaded" && !task ? (
                  <>
                    <div
                      ref={fineTunedSectionRef}
                      className="mt-3 flex items-center gap-1 border-t border-border/50 px-2.5 pb-1 pt-3"
                    >
                      <span className="flex items-center gap-1.5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
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
                        loadedModelId={loadedModelId}
                        activeGgufVariant={activeGgufVariant}
                        onSelect={onSelect}
                        onConfigure={onConfigure}
                        onModelsChange={onModelsChange}
                        deleteDisabled={deleteDisabled}
                        loraModelList={hubModelList}
                        expandedGguf={expandedGguf}
                        setExpandedGguf={setExpandedGguf}
                        gpu={inferenceGpu}
                      />
                    )}
                  </>
                ) : null}

                {showCustom ? (
                  <>
                    <div
                      ref={customFolderSectionRef}
                      className="mt-3 flex items-center gap-1 border-t border-border/50 px-2.5 pb-1 pt-3"
                    >
                      <button
                        type="button"
                        onClick={() => setShowFolderBrowser(true)}
                        title="Browse folders on the server"
                        className="flex items-center gap-1.5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={Folder02Icon}
                          className="size-3.5"
                        />
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
                            className="min-w-0 flex-1 truncate font-mono text-ui-10 text-muted-foreground/70"
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
                                className="rounded-full border border-dashed border-border/50 px-2 py-0.5 font-mono text-ui-10 text-muted-foreground/70 transition-colors hover:border-foreground/30 hover:bg-accent hover:text-foreground disabled:opacity-40"
                              >
                                <span className="text-ui-11 font-semibold">
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
                            className="h-6 min-w-0 flex-1 rounded border border-border/50 bg-transparent px-1.5 font-mono text-ui-10 text-foreground outline-none placeholder:text-muted-foreground/40 focus:border-foreground/20"
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
                            className="h-6 shrink-0 rounded border border-border/50 px-1.5 text-ui-10 text-muted-foreground transition-colors hover:bg-accent disabled:opacity-40"
                          >
                            Add
                          </button>
                        </div>
                        {folderError && (
                          <p className="px-0.5 pt-0.5 text-ui-10 text-destructive">
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
                        const isGgufFile = m.path
                          .toLowerCase()
                          .endsWith(".gguf");
                        // Honor the backend model_format hint (suffixless GGUF
                        // folders) in addition to name/path so the row classifies
                        // and loads through the same GGUF path as the filter.
                        const isGguf = localModelIsGguf(m);
                        // Single .gguf files (e.g. Ollama blobs) load directly;
                        // GGUF repos/directories expand to pick a variant.
                        const isDirectGguf = isGgufFile;
                        const optionKey = makeModelOptionKey(
                          "custom-folder",
                          m.id,
                        );
                        return (
                          <div key={m.id}>
                            <div className="group flex items-center">
                              <div className="min-w-0 flex-1">
                                <ModelRow
                                  label={m.model_id ?? m.display_name}
                                  meta={isGguf ? "GGUF" : "Local"}
                                  tooltipText={localPathTooltip(
                                    m.model_id ?? m.display_name,
                                    m.path,
                                  )}
                                  selected={value === m.id}
                                  loaded={isRuntimeLoadedModel(
                                    loadedModelId,
                                    activeGgufVariant,
                                    m.id,
                                    isGgufFile
                                      ? "ignore"
                                      : isGguf
                                        ? "required"
                                        : "none",
                                  )}
                                  optionProps={hubModelList.getOptionProps(
                                    optionKey,
                                    value === m.id,
                                  )}
                                  onClick={() => {
                                    if (isDirectGguf) {
                                      onSelect(m.id, localDirectGgufMeta());
                                    } else if (isGguf) {
                                      toggleGgufExpanded(m.id);
                                    } else {
                                      onSelect(m.id, localModelMeta());
                                    }
                                  }}
                                  onArrowDownIntoChildren={
                                    isGguf &&
                                    !isDirectGguf &&
                                    isGgufExpanded(m.id)
                                      ? () => {
                                          const focused =
                                            focusFirstChildOption(optionKey);
                                          return focused;
                                        }
                                      : undefined
                                  }
                                  alignMeta="device"
                                  vramStatus={null}
                                />
                              </div>
                              <span className={ROW_ACTIONS_CLASS}>
                                {isDirectGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(m.id, localDirectGgufMeta())
                                    }
                                  />
                                )}
                                {!isGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(m.id, localModelMeta())
                                    }
                                  />
                                )}
                              </span>
                            </div>
                            {isGguf &&
                              !isDirectGguf &&
                              isGgufExpanded(m.id) && (
                                <GgufVariantExpander
                                  repoId={m.id}
                                  onDevice={true}
                                  onSelect={onSelect}
                                  resolveDownloadFootprint={resolveDownloadFootprint}
                                  onConfigure={onConfigure}
                                  parentOptionKey={optionKey}
                                  onNavigatePastStart={() =>
                                    hubModelList.focusOption(optionKey)
                                  }
                                  onNavigatePastEnd={() =>
                                    hubModelList.moveFocus(optionKey, "next")
                                  }
                                  gpuGb={
                                    inferenceGpu.available
                                      ? inferenceGpu.memoryTotalGb
                                      : undefined
                                  }
                                  systemRamGb={
                                    inferenceGpu.systemRamAvailableGb || undefined
                                  }
                                  budgetKnown={inferenceGpu.budgetKnown}
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
                      divider={true}
                      collapsed={lmStudioCollapsed}
                      onToggle={() => setLmStudioCollapsed((v) => !v)}
                    >
                      LM Studio
                    </ListLabel>
                    {!lmStudioCollapsed &&
                      sortedLmStudio.map((m) => {
                        const isGgufFile = m.path
                          .toLowerCase()
                          .endsWith(".gguf");
                        // LM Studio dirs are GGUF but rarely carry a -GGUF suffix;
                        // use the shared helper (model_format hint) so the row,
                        // filter, and load path agree.
                        const isGguf = localModelIsGguf(m);
                        const optionKey = makeModelOptionKey("lm-studio", m.id);
                        return (
                          <div key={m.id}>
                            <div className="group flex items-center">
                              <div className="min-w-0 flex-1">
                                <ModelRow
                                  label={m.model_id ?? m.display_name}
                                  meta={isGguf ? "GGUF" : "Local"}
                                  tooltipText={localPathTooltip(
                                    m.model_id ?? m.display_name,
                                    m.path,
                                  )}
                                  selected={value === m.id}
                                  loaded={isRuntimeLoadedModel(
                                    loadedModelId,
                                    activeGgufVariant,
                                    m.id,
                                    isGgufFile
                                      ? "ignore"
                                      : isGguf
                                        ? "required"
                                        : "none",
                                  )}
                                  optionProps={hubModelList.getOptionProps(
                                    optionKey,
                                    value === m.id,
                                  )}
                                  onClick={() => {
                                    if (isGgufFile) {
                                      onSelect(m.id, localDirectGgufMeta());
                                    } else if (isGguf) {
                                      toggleGgufExpanded(m.id);
                                    } else {
                                      onSelect(m.id, localModelMeta());
                                    }
                                  }}
                                  onArrowDownIntoChildren={
                                    isGguf &&
                                    !isGgufFile &&
                                    isGgufExpanded(m.id)
                                      ? () => {
                                          const focused =
                                            focusFirstChildOption(optionKey);
                                          return focused;
                                        }
                                      : undefined
                                  }
                                  alignMeta="device"
                                  vramStatus={null}
                                />
                              </div>
                              <span className={ROW_ACTIONS_CLASS}>
                                {isGgufFile && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(m.id, localDirectGgufMeta())
                                    }
                                  />
                                )}
                                {!isGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(m.id, localModelMeta())
                                    }
                                  />
                                )}
                              </span>
                            </div>
                            {isGguf && !isGgufFile && isGgufExpanded(m.id) && (
                              <GgufVariantExpander
                                repoId={m.id}
                                onDevice={true}
                                onSelect={onSelect}
                                resolveDownloadFootprint={resolveDownloadFootprint}
                                onConfigure={onConfigure}
                                parentOptionKey={optionKey}
                                onNavigatePastStart={() =>
                                  hubModelList.focusOption(optionKey)
                                }
                                onNavigatePastEnd={() =>
                                  hubModelList.moveFocus(optionKey, "next")
                                }
                                gpuGb={expanderGpuGb}
                                systemRamGb={
                                  inferenceGpu.systemRamAvailableGb || undefined
                                }
                                budgetKnown={inferenceGpu.budgetKnown}
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
                      divider={true}
                      collapsed={localDirCollapsed}
                      onToggle={() => setLocalDirCollapsed((v) => !v)}
                    >
                      Local models
                    </ListLabel>
                    {!localDirCollapsed &&
                      sortedLocalDir.map((m) => {
                        // A loose ./models/*.gguf loads directly; a GGUF repo dir
                        // expands to pick a variant. The variant scanner returns
                        // nothing for a config-less loose file, so expanding it would
                        // dead-end at "No GGUF variants".
                        const isGgufFile = m.path
                          .toLowerCase()
                          .endsWith(".gguf");
                        const isGguf = localModelIsGguf(m);
                        const optionKey = makeModelOptionKey("local-dir", m.id);
                        return (
                          <div key={m.id}>
                            <div className="group flex items-center">
                              <div className="min-w-0 flex-1">
                                <ModelRow
                                  label={m.model_id ?? m.display_name}
                                  meta={isGguf ? "GGUF" : "Local"}
                                  tooltipText={localPathTooltip(
                                    m.model_id ?? m.display_name,
                                    m.path,
                                  )}
                                  selected={value === m.id}
                                  loaded={isRuntimeLoadedModel(
                                    loadedModelId,
                                    activeGgufVariant,
                                    m.id,
                                    isGgufFile
                                      ? "ignore"
                                      : isGguf
                                        ? "required"
                                        : "none",
                                  )}
                                  optionProps={hubModelList.getOptionProps(
                                    optionKey,
                                    value === m.id,
                                  )}
                                  onClick={() => {
                                    if (isGgufFile) {
                                      onSelect(m.id, localDirectGgufMeta());
                                    } else if (isGguf) {
                                      toggleGgufExpanded(m.id);
                                    } else {
                                      onSelect(m.id, localModelMeta());
                                    }
                                  }}
                                  onArrowDownIntoChildren={
                                    isGguf &&
                                    !isGgufFile &&
                                    isGgufExpanded(m.id)
                                      ? () => focusFirstChildOption(optionKey)
                                      : undefined
                                  }
                                  alignMeta="device"
                                  vramStatus={null}
                                />
                              </div>
                              <span className={ROW_ACTIONS_CLASS}>
                                {isGgufFile && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(m.id, localDirectGgufMeta())
                                    }
                                  />
                                )}
                                {!isGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(m.id, localModelMeta())
                                    }
                                  />
                                )}
                              </span>
                            </div>
                            {isGguf && !isGgufFile && isGgufExpanded(m.id) && (
                              <GgufVariantExpander
                                repoId={m.id}
                                onDevice={true}
                                onSelect={onSelect}
                                resolveDownloadFootprint={resolveDownloadFootprint}
                                onConfigure={onConfigure}
                                parentOptionKey={optionKey}
                                onNavigatePastStart={() =>
                                  hubModelList.focusOption(optionKey)
                                }
                                onNavigatePastEnd={() =>
                                  hubModelList.moveFocus(optionKey, "next")
                                }
                                gpuGb={expanderGpuGb}
                                systemRamGb={
                                  inferenceGpu.systemRamAvailableGb || undefined
                                }
                                budgetKnown={inferenceGpu.budgetKnown}
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
                              label={
                                (catalog && curatedDisplayNameFor(id, catalog)) || id
                              }
                              hubUrl={hubRepoUrl(id)}
                              alignMeta="hub"
                              showSize={hubRowsShowSize}
                              hideOwner={true}
                              downloaded={downloadedSet.has(id.toLowerCase())}
                              capabilities={capsById.get(id)}
                              meta={
                                info?.meta ??
                                (isG ? "GGUF" : extractParamLabel(id))
                              }
                              selected={value === id}
                              loaded={isRuntimeLoadedModel(
                                loadedModelId,
                                activeGgufVariant,
                                id,
                                isG ? "required" : "none",
                              )}
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
                                isG ? expanderGpuGb : expanderSystemGpuGb
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
                                resolveDownloadFootprint={resolveDownloadFootprint}
                                onConfigure={onConfigure}
                                hfToken={hfToken || undefined}
                                parentOptionKey={optionKey}
                                onNavigatePastStart={() =>
                                  hubModelList.focusOption(optionKey)
                                }
                                onNavigatePastEnd={() =>
                                  hubModelList.moveFocus(optionKey, "next")
                                }
                                gpuGb={expanderGpuGb}
                                systemRamGb={
                                  inferenceGpu.systemRamAvailableGb || undefined
                                }
                                budgetKnown={inferenceGpu.budgetKnown}
                                variantActions={{
                                  onDelete: async (quant) => {
                                    await deleteCachedModel(
                                      id,
                                      quant,
                                      hfToken || undefined,
                                    );
                                    prunePinnedQuantValidation(id, quant);
                                    refreshCachedLists();
                                  },
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
                        {/* Only while a page is in flight; on hasMore it sat under a usable list. */}
                        {recommendedSearch.isLoadingMore ? (
                          <div className="flex items-center justify-center py-2">
                            <Spinner className="size-3.5 text-muted-foreground" />
                          </div>
                        ) : null}
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
                            label={
                              (catalog && curatedDisplayNameFor(id, catalog)) || id
                            }
                            hubUrl={hubRepoUrl(id)}
                            alignMeta="hub"
                            showSize={hubRowsShowSize}
                            downloaded={downloadedSet.has(id.toLowerCase())}
                            capabilities={capsById.get(id)}
                            // Same meta the unfiltered Recommended row shows, so a
                            // model does not lose its size chip just because it was
                            // reached by typing its name.
                            meta={
                              isKnownGgufRepo(id)
                                ? (recommendedMeta.get(id)?.meta ?? "GGUF")
                                : (vram?.detail ?? extractParamLabel(id))
                            }
                            selected={value === id}
                            loaded={isRuntimeLoadedModel(
                              loadedModelId,
                              activeGgufVariant,
                              id,
                              isKnownGgufRepo(id) ? "required" : "none",
                            )}
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
                              isKnownGgufRepo(id)
                                ? null
                                : (vram?.status ?? null)
                            }
                            vramEst={
                              isKnownGgufRepo(id) ? undefined : vram?.est
                            }
                            gpuGb={
                              isKnownGgufRepo(id) ? expanderGpuGb : expanderSystemGpuGb
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
                              resolveDownloadFootprint={resolveDownloadFootprint}
                              onConfigure={onConfigure}
                              hfToken={hfToken || undefined}
                              parentOptionKey={optionKey}
                              onNavigatePastStart={() =>
                                hubModelList.focusOption(optionKey)
                              }
                              onNavigatePastEnd={() =>
                                hubModelList.moveFocus(optionKey, "next")
                              }
                              gpuGb={expanderGpuGb}
                              systemRamGb={
                                inferenceGpu.systemRamAvailableGb || undefined
                              }
                              budgetKnown={inferenceGpu.budgetKnown}
                              variantActions={{
                                onDelete: async (quant) => {
                                  await deleteCachedModel(
                                    id,
                                    quant,
                                    hfToken || undefined,
                                  );
                                  prunePinnedQuantValidation(id, quant);
                                  refreshCachedLists();
                                },
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
                              label={
                                (catalog && curatedDisplayNameFor(id, catalog)) || id
                              }
                              hubUrl={hubRepoUrl(id)}
                              alignMeta="hub"
                              showSize={hubRowsShowSize}
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
                              loaded={isRuntimeLoadedModel(
                                loadedModelId,
                                activeGgufVariant,
                                id,
                                isSearchGguf ? "required" : "none",
                              )}
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
                                isSearchGguf ? expanderGpuGb : expanderSystemGpuGb
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
                                resolveDownloadFootprint={resolveDownloadFootprint}
                                onConfigure={onConfigure}
                                hfToken={hfToken || undefined}
                                parentOptionKey={optionKey}
                                onNavigatePastStart={() =>
                                  hubModelList.focusOption(optionKey)
                                }
                                onNavigatePastEnd={() =>
                                  hubModelList.moveFocus(optionKey, "next")
                                }
                                gpuGb={expanderGpuGb}
                                systemRamGb={
                                  inferenceGpu.systemRamAvailableGb || undefined
                                }
                                budgetKnown={inferenceGpu.budgetKnown}
                                variantActions={{
                                  onDelete: async (quant) => {
                                    await deleteCachedModel(
                                      id,
                                      quant,
                                      hfToken || undefined,
                                    );
                                    prunePinnedQuantValidation(id, quant);
                                    refreshCachedLists();
                                  },
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
        {onEject ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-end pr-3.5 pb-[19px]">
            <button
              type="button"
              onClick={onEject}
              className="pointer-events-auto inline-flex items-center justify-center gap-2 rounded-md bg-popover px-3 py-2 text-ui-13 font-medium text-destructive shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] transition-colors hover:bg-[color-mix(in_srgb,var(--destructive)_12%,var(--popover))] dark:bg-[color-mix(in_srgb,var(--foreground)_10%,var(--sidebar))] dark:shadow-none dark:hover:bg-[color-mix(in_srgb,var(--destructive)_22%,var(--sidebar))]"
              title="Eject model"
            >
              <HugeiconsIcon icon={RemoveCircleIcon} className="size-3.5" />
              Eject model
            </button>
          </div>
        ) : null}
      </div>
      <TransportConflictDialog
        conflict={updateTransportConflict}
        onCancel={cancelUpdateConflict}
        onKeepTransport={resumeUpdateConflict}
        onSwitchTransport={restartUpdateConflict}
      />
    </>
  );
}

/** Fine-tuned model rows for the On Device tab's Fine-tuned section. Plugs into
 * that section's roving list and shared GGUF-expand state. */
function FineTunedRows({
  adapters,
  value,
  loadedModelId,
  activeGgufVariant,
  onSelect,
  onConfigure,
  onModelsChange,
  deleteDisabled = false,
  loraModelList,
  expandedGguf,
  setExpandedGguf,
  gpu,
}: {
  adapters: LoraModelOption[];
  value?: string;
  loadedModelId?: string;
  activeGgufVariant?: string | null;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onConfigure?: (id: string, meta: ModelSelectorChangeMeta) => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  loraModelList: ReturnType<typeof useRovingModelList>;
  expandedGguf: string | null;
  setExpandedGguf: Dispatch<SetStateAction<string | null>>;
  gpu: {
    available: boolean;
    budgetKnown: boolean;
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
        const selectionMeta: ModelSelectorChangeMeta = {
          source: isLocal ? "local" : isExported ? "exported" : "lora",
          isLora: !isLocal && !isMerged && !isGguf,
          isDownloaded: true,
          isGguf: false,
        };
        const canConfigure = !(isLocalGgufDir || isExportedGguf);
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
            <div className="group flex items-center">
              <div className="min-w-0 flex-1">
                <ModelRow
                  label={adapter.name}
                  meta={meta}
                  selected={value === adapter.id}
                  loaded={isRuntimeLoadedModel(
                    loadedModelId,
                    activeGgufVariant,
                    adapter.id,
                    isLocalGgufDir || isExportedGguf ? "required" : "none",
                  )}
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
                      onSelect(adapter.id, selectionMeta);
                    }
                  }}
                  tooltipText={
                    <>
                      <span className="block break-words">{adapter.name}</span>
                      <span className="block mt-1 text-ui-10 text-muted-foreground break-all">
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
                  alignMeta="device"
                />
              </div>
              <span className={ROW_ACTIONS_CLASS}>
                {canConfigure && onConfigure && (
                  <ModelLoadSettingsAction
                    ariaLabel={`Inference settings for ${adapter.name}`}
                    onConfigure={() => onConfigure(adapter.id, selectionMeta)}
                  />
                )}
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
              </span>
            </div>
            {expandedGguf === adapter.id && (
              <GgufVariantExpander
                repoId={adapter.id}
                onSelect={onSelect}
                onConfigure={onConfigure}
                parentOptionKey={optionKey}
                onNavigatePastStart={() => loraModelList.focusOption(optionKey)}
                onNavigatePastEnd={() =>
                  loraModelList.moveFocus(optionKey, "next")
                }
                gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                systemRamGb={gpu.systemRamAvailableGb || undefined}
                budgetKnown={gpu.budgetKnown}
                sourceOverride={isExportedGguf ? "exported" : undefined}
                variantActions={{
                  deleteTitle: "Delete exported GGUF variant?",
                  renderDeleteDescription: (quant) => (
                    <>
                      This will remove{" "}
                      <span className="font-medium text-foreground">
                        {adapter.name} ({quant})
                      </span>{" "}
                      from disk. This cannot be undone.
                    </>
                  ),
                  getDeleteSuccessMessage: (quant) =>
                    `Deleted ${adapter.name} ${quant}`,
                  deleteDisabled: deleteDisabled,
                  onDelete: isExportedGguf
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
                    : undefined,
                }}
              />
            )}
          </div>
        );
      })}
    </>
  );
}
