// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isChatGgufTask,
  reconcileGgufPinsAfterDelete,
} from "./reconcile-gguf-pins";
import {
  createTaskLimiter,
  hubWithdrawsSoleQuant,
  loadPickerGgufVariants,
} from "./gguf-discovery";

import { ModelMemoryBar } from "@/components/model-memory-bar";
import { shouldRefreshPickerInventoryOnMount } from "@/components/resource-picker/picker-tab-policy";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { isTouchClick } from "@/components/ui/touch-click";
import { usePlatformStore } from "@/config/env";
import { ApiProviderLogo } from "@/features/chat";
import {
  type ScanFolderInfo,
  addScanFolder,
  deleteFineTunedModel,
  listGgufVariants,
  listRecommendedFolders,
  listScanFolders,
  listLoras,
  removeScanFolder,
  revealFineTunedModel,
} from "@/features/chat";
import {
  chatModelLoaded,
  DROP_CUE_CLASS,
  isExternalModelId,
  modelCatalogVersion,
  parseExternalModelId,
  subscribeModelCatalog,
  useChatRuntimeStore,
  useExternalProvidersStore,
} from "@/features/chat";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantDetail,
  LocalModelInfo,
} from "@/features/chat";
import type { ProviderApiType } from "@/features/chat/api/providers-api";
// eslint-disable-next-line no-restricted-imports -- Connection contract has no React dependencies.
import type { CustomReasoningConfig } from "@/features/chat/custom-reasoning";
import { normalizeGgufVisionCapability } from "@/features/chat/utils/model-vision-capability";
import {
  DotTag,
  type HubOption,
  HubOptionMenu,
  TrainIcon,
  TransportConflictDialog,
  deleteCachedModel,
  invalidateGgufVariantsCache,
  listGgufVariants as listGgufVariantsCached,
  useGgufVariantsCacheVersions,
  useHubInfiniteScroll,
  isHuggingFaceOffline,
} from "@/features/hub";
import { type HfModelResult, useHubModelSearch } from "@/features/hub";
import {
  classifyUnslothSupport,
  downloadManager,
  hfApiToken,
  isHiddenModelId,
  jobKeyOf,
  partialSetFromRows,
  pendingDrafterPresentation,
  scanFolderStatusCopy,
  useDownloadManagerStore,
  useHfTokenStore,
  HubFailureHint,
  useHubAvailability,
  useOnlineStatus,
} from "@/features/hub";
import type { HfTaskFilter } from "@/features/hub/hooks/use-hub-model-search";
import { INVENTORY_FRESHNESS_WINDOW_MS } from "@/features/hub/inventory";
import {
  type NpuModel,
  type NpuPickerSource,
  NpuSetupNotice,
  npuDownloadLabel,
  npuResumeLabel,
  npuRowsFor,
  npuSizeLabel,
  useNpuCatalog,
} from "@/features/npu";
import {
  useDebouncedValue,
  useDenseQuantSchemes,
  useGpuInfo,
  useHostClass,
  useInferenceGpuInfo,
} from "@/hooks";
import {
  type ModelMemorySource,
  useModelMemory,
} from "@/hooks/use-model-memory";
import { useVramBudgetFraction } from "@/hooks/use-vram-budget-fraction";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { diffusionRouteSearch } from "@/lib/diffusion-route-search";
import {
  type GgufFitClass,
  ggufVariantFitSizeBytes,
  requiredGgufMemoryGb,
} from "@/lib/gguf-fit";
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
  Copy01Icon,
  DashboardCircleIcon,
  Flag01Icon,
  FlimSlateIcon,
  Folder02Icon,
  HelpCircleIcon,
  Image03Icon,
  InformationCircleIcon,
  PinIcon,
  RemoveCircleIcon,
  Search01Icon,
  Settings02Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import {
  type Dispatch,
  type KeyboardEvent,
  type ReactNode,
  type SetStateAction,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { audioPickSearch } from "../../../audio/route-search.ts";
import { useChatPickerInventory } from "../../inventory/use-chat-picker-inventory";
import {
  type CommunityModelPolicy,
  allowedHiddenModelIdMatches,
  audioPickIsRoutable,
  audioPipelineTagFor,
  communityAudioRowIsRunnable,
  curatedAudioInventoryMatches,
  curatedAudioInventoryTask,
  filesystemRowsSupportedForTask,
  localAudioRowIsUndecodableGguf,
  macTtsHubRowIsRunnable,
  nativeAudioCheckpointIsLoadable,
  shouldDiscoverCommunityModels,
  shouldRecommendCommunityModels,
  taskCatalogFormatMatches,
  taskForMediaPick,
  taskPickerRowMatches,
} from "./audio-picker-policy";
import { ConnectedModelInfoDialog } from "./connected-model-info-dialog";
import {
  connectedModelMarks,
} from "./connected-model-meta";
import { ConnectedModelSettingsDialog } from "./connected-model-settings-dialog";
import { FolderBrowser } from "./folder-browser";
import { curatedArtifactIsOfferable } from "./host-artifact-policy";
import { localGgufKindFor } from "./local-gguf-policy";
import {
  type ModelCapabilities,
  detectCapabilities,
} from "./model-capabilities";
import {
  AUDIO_CATALOG,
  type CatalogGroup,
  type DeviceBudget,
  artifactForRepoId,
  classifyGgufFit,
  classifyMediaGgufFit,
  curatedArtifactFit,
  curatedCapabilitiesFor,
  curatedRowLabelFor,
  curatedSizeBytesFor,
  curatedTotalParamsFor,
  groupForRepoId,
  recommendedQuantForDevice,
} from "./model-catalog";
import { ModelDeleteAction } from "./model-delete-action";
import { ModelLoadSettingsAction } from "./model-load-settings-action";
import { ModelRowMenu } from "./model-row-menu";
import {
  type ModelLoadTimes,
  loadedAt,
  useModelLoadTimes,
} from "./model-usage";
import { usePinnedConnectedModelsStore } from "./pinned-connected-models";
import {
  makePinRank,
  pinDropAnchor,
  pinKey,
  pinnedQuantEntries,
  usePinnedModelsStore,
} from "./pinned-models";
import {
  type CuratedBudget,
  type FormatFilter,
  curatedBudget,
  curatedBudgetText,
  estimateQuantBytes,
  hfModelFitsDevice,
  isMlxId,
  isMobileVariant,
  isRecommendableFormat,
  loadScopedGpu,
  matchesFormatFilter,
  orderRecommendedRows,
  paramsFromId,
  recommendedEmptyState,
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
  verifiedSoleHubVariant,
} from "./sole-quant-cache";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelDownloadFootprintResolver,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./types";
import { type PinnedDropEdge, usePinnedRowDrag } from "./use-pinned-row-drag";
import { describeVariantListingError } from "./variant-listing-error";
import {
  ggufQuantChipLabel,
  ggufQuantDetailLabel,
  ggufVariantPickerLabel,
  ggufVariantContextLength,
  groupGgufVariantsForPicker,
  h3PickerHasOnlyPrunedBuilds,
  preferredGgufVariantByGroup,
} from "./variant-presentation";
import {
  shouldMountVariantExpander,
  toggleAutoExpandedRow,
  visibleGgufVariants,
} from "./variant-visibility";

function dedupe(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

/** The primary namespace used by runtime trust gates. */
function isUnslothRepoId(repoId: string): boolean {
  return repoId.toLowerCase().startsWith("unsloth/");
}

/** Official publisher namespaces, used only for visual On Device grouping. */
function isUnslothPublisherRepoId(repoId: string): boolean {
  return isUnslothOwner(splitRepoLabel(repoId).owner);
}

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
  divider?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-1 px-2.5 pb-1",
        divider ? "mt-3 border-t border-border pt-3" : "pt-3",
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
              className="-mr-0.5 shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
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

function formatBytes(bytes: number, unitSeparator = ""): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  // Decimal units to match Hugging Face sizes (GPU-fit math stays base-1024); Math.log is off at
  // exact powers of 1000.
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let value = bytes;
  while (value >= 1000 && i < units.length - 1) {
    value /= 1000;
    i += 1;
  }
  // No space: "145MB" reads as one value beside the quant chip.
  return `${value.toFixed(value < 10 ? 1 : 0)}${unitSeparator}${units[i]}`;
}

// Most distinguishing first, since only the first MAX_CAPABILITY_BADGES are drawn.
const CAPABILITY_BADGES: {
  key: keyof ModelCapabilities;
  title: string;
  tone: string;
  Glyph: (props: { className: string }) => ReactNode;
}[] = [
  {
    key: "videoGen",
    title: "Generates video",
    tone: "text-[oklch(0.5_0.1_55)] dark:text-[oklch(0.78_0.09_60)]",
    Glyph: (props) => (
      <HugeiconsIcon icon={FlimSlateIcon} strokeWidth={1.8} {...props} />
    ),
  },
  {
    key: "imageGen",
    title: "Generates images",
    tone: "text-pink-700 dark:text-pink-300",
    Glyph: (props) => (
      <HugeiconsIcon icon={Image03Icon} strokeWidth={1.8} {...props} />
    ),
  },
  {
    key: "audio",
    // Direction-neutral: `audio` covers ASR as well as synthesis.
    title: "Audio",
    tone: "text-[oklch(0.5_0.08_190)] dark:text-[oklch(0.78_0.08_190)]",
    Glyph: (props) => (
      <HugeiconsIcon icon={AudioWave01Icon} strokeWidth={1.8} {...props} />
    ),
  },
];

/** Capability glyphs worth drawing in this picker; null draws all. A media picker's own kind
 *  is not information. */
const CapabilityScope = createContext<readonly (keyof ModelCapabilities)[] | null>(
  null,
);

// The row reserves a fixed slot (META_COLUMN.badge), so the cap keeps later columns aligned.
const MAX_CAPABILITY_BADGES = 3;

function visibleCapabilityBadges(
  caps: ModelCapabilities,
  scope: readonly (keyof ModelCapabilities)[] | null,
) {
  return CAPABILITY_BADGES.filter(
    (b) => caps[b.key] && (scope?.includes(b.key) ?? true),
  ).slice(0, MAX_CAPABILITY_BADGES);
}

function CapabilityIcons({ caps }: { caps: ModelCapabilities }) {
  const scope = useContext(CapabilityScope);
  return (
    <>
      {visibleCapabilityBadges(caps, scope).map(({ key, title, tone, Glyph }) => (
        <span
          key={key}
          title={title}
          aria-label={title}
          className={cn(
            "flex h-[calc(18px*var(--ui-space-scale,1))] shrink-0 items-center justify-center rounded-md border border-border px-1.5",
            tone,
          )}
        >
          <Glyph className="size-3" />
        </span>
      ))}
    </>
  );
}

function VisionBadge() {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label="Vision"
          className="flex h-[calc(18px*var(--ui-space-scale,1))] shrink-0 items-center justify-center rounded-md border border-border px-1.5 text-indigo-700 dark:text-indigo-300"
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

function ParamChip({ label }: { label: string }) {
  return (
    // Fixed height like every other chip in the row; py-px scaled with --ui-font-scale instead.
    <span className="inline-flex h-[calc(18px*var(--ui-space-scale,1))] shrink-0 items-center whitespace-nowrap rounded-md border border-border px-1.5 text-ui-10 font-medium text-muted-foreground tabular-nums">
      {label}
    </span>
  );
}

const FORMAT_TONE_DOT: Record<FormatTone, string> = {
  gguf: "bg-format-gguf",
  mlx: "bg-format-mlx",
  checkpoint: "bg-format-checkpoint",
  adapter: "bg-format-adapter",
  npu: "bg-format-npu",
};

/** Format as a coloured dot, named on hover, since a word would shove the row around. */
function FormatTag({ tone, label }: { tone: FormatTone; label: string }) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label={label}
          className="flex size-[calc(14px*var(--ui-space-scale,1))] shrink-0 items-center justify-center"
        >
          <span
            aria-hidden="true"
            className={cn("size-[calc(5px*var(--ui-space-scale,1))] rounded-full", FORMAT_TONE_DOT[tone])}
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

/** "Already on disk" for Hub rows, whose download arrow would read as "click to fetch". */
function DownloadedBadge() {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label="On device"
          className="flex h-[calc(18px*var(--ui-space-scale,1))] w-[calc(14px*var(--ui-space-scale,1))] shrink-0 items-center justify-center"
        >
          <span
            aria-hidden="true"
            className="size-[calc(5px*var(--ui-space-scale,1))] rounded-full bg-status-success"
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        On device
      </TooltipContent>
    </Tooltip>
  );
}

/** An incomplete download; the row selects with isDownloaded: false so the click resumes the
 *  download instead of loading incomplete weights. */
function PartialBadge({ resumable }: { resumable?: boolean }) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label="Partial download"
          className="flex h-[calc(18px*var(--ui-space-scale,1))] w-[calc(14px*var(--ui-space-scale,1))] shrink-0 items-center justify-center"
        >
          <span
            aria-hidden="true"
            className="size-[calc(5px*var(--ui-space-scale,1))] rounded-full bg-status-warning"
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {resumable
          ? "Partial download. Select to resume it, or delete it."
          : "Partial download. Select to continue it, or delete it."}
      </TooltipContent>
    </Tooltip>
  );
}

interface FitVerdict {
  label: string;
  tone: string;
  hint: string;
}

const AMBER = "!text-yellow-600 dark:!text-yellow-400";
const ORANGE = "!text-orange-600 dark:!text-orange-300";

/** Over the VRAM Budget but under the card: `_select_gpus` hands it to --fit every time. */
const MIGHT_FIT: FitVerdict = {
  label: "Over budget",
  tone: AMBER,
  hint: "Larger than your VRAM Budget allows, so part of it offloads even on an idle GPU. It is still smaller than the card, so raising the budget can keep it resident.",
};
/** Past the card; llama-server never refuses a GGUF on size, --fit offloads. */
const OFFLOADS: FitVerdict = {
  label: "Does not fit",
  tone: ORANGE,
  hint: "Model may not fit but still works with offloading. Expect slower inference.",
};
/** checkVramFit's 75-100% band, the only source of `tight`: fits on the card with no --fit. */
const DEVICE_TIGHT: FitVerdict = {
  label: "Tight fit",
  tone: AMBER,
  hint: "Uses nearly all your VRAM, with little headroom for anything else.",
};
/** A torch load has no --fit: the pipeline goes wholly on the device. */
const WONT_FIT: FitVerdict = {
  label: "Does not fit",
  tone: ORANGE,
  // "memory", not "VRAM": this also covers diffusion refusals on a shared pool.
  hint: "Needs more memory than this device has. This model will not load.",
};

/** Fit verdict marks keyed by the Hub's classes; `tight`/`exceeds` come only from torch or
 *  QLoRA estimates, which never offload. */
const VRAM_VERDICT: Record<GgufFitClass | VramFitStatus, FitVerdict | null> = {
  fits: null,
  marginal: MIGHT_FIT,
  tight: DEVICE_TIGHT,
  partial: OFFLOADS,
  ram: {
    label: "RAM fallback",
    tone: ORANGE,
    hint: "No GPU detected. Runs on system RAM and CPU. Expect much slower inference.",
  },
  oom: OFFLOADS,
  exceeds: WONT_FIT,
};

/** On a shared pool a diffusion `oom` is a refusal (MPS high-watermark disabled, OS kills the
 *  process). `hostPooled` folds in unified_memory since shared_memory is set only on Windows. */
function diffusionRefuses(
  fit: GgufFitClass,
  diffusionLoad: boolean,
  hostPooled: boolean,
): boolean {
  return fit === "oom" && diffusionLoad && hostPooled;
}

/** Zero on a host pool, where offload frees nothing. llama.cpp differs: GGUFs spill to host RAM. */
function mediaRamBudgetGb(systemRamGb: number, hostPooled: boolean): number {
  return hostPooled ? 0 : systemRamGb;
}

/** `marginal` is not over budget: it is a full GPU load with little room to spare. */
function isOverBudget(status?: GgufFitClass | VramFitStatus | null): boolean {
  return (
    status === "partial" ||
    status === "ram" ||
    status === "oom" ||
    status === "exceeds"
  );
}

function VramBadge({
  status,
  /** Model rows hold the mark in the layout and paint it on hover; variant rows always show it. */
  revealOnHover = false,
}: {
  status?: GgufFitClass | VramFitStatus | null;
  revealOnHover?: boolean;
}) {
  const verdict = status ? VRAM_VERDICT[status] : null;
  if (!verdict) return null;
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        {/* biome-ignore lint/a11y/useKeyWithClickEvents: the handler suppresses the enclosing
            button, it does not add an interaction of its own */}
        <span
          aria-label={verdict.label}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
          }}
          className={cn(
            "flex size-[calc(18px*var(--ui-space-scale,1))] shrink-0 items-center justify-center",
            verdict.tone,
            revealOnHover &&
              "opacity-0 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100",
          )}
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            className="size-3.5"
            strokeWidth={1.8}
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {verdict.hint}
      </TooltipContent>
    </Tooltip>
  );
}

const SIZE_PARTS_RE = /^(~?)([\d.]+)\s*([A-Za-z]+)$/;

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

/** One decimal through GB/TB, so a total and its model + assets breakdown visibly add up. */
export function formatFootprintBytes(bytes: number): string {
  return bytes >= 1_000_000_000 && bytes < 1_000_000_000_000
    ? `${(bytes / 1_000_000_000).toFixed(1)} GB`
    : bytes >= 1_000_000_000_000
      ? `${(bytes / 1_000_000_000_000).toFixed(1)} TB`
      : formatBytes(bytes, " ");
}

/** Diffusion GGUFs get an explanation, since the checkpoint is only part of what is kept on disk. */
export function GgufDownloadFootprint({
  checkpointBytes,
  companionBytes,
}: {
  checkpointBytes: number;
  companionBytes: number;
}) {
  const totalLabel = formatFootprintBytes(checkpointBytes + companionBytes);
  return (
    <span
      data-model-download-footprint={true}
      className="flex items-center gap-1 whitespace-nowrap text-muted-foreground"
    >
      {/* Keep flex gaps out of SizeText's fragments. */}
      <span>
        <SizeText value={totalLabel} />
      </span>
      <span
        data-model-download-footprint-help={true}
        aria-hidden={true}
        className="flex size-3.5 shrink-0 -translate-y-[0.25em] items-center justify-center text-muted-foreground/80"
      >
        <HugeiconsIcon icon={HelpCircleIcon} className="size-3" strokeWidth={1.8} />
      </span>
    </span>
  );
}

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
        {formatFootprintBytes(checkpointBytes)} model +{" "}
        {formatFootprintBytes(companionBytes)} required assets
      </span>
    </>
  );
}

function QuantChip({ label }: { label: string }) {
  return (
    <span className="inline-flex h-[calc(18px*var(--ui-space-scale,1))] max-w-full items-center overflow-hidden rounded-md bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] px-1 font-mono text-ui-9 text-muted-foreground dark:bg-muted">
      {label}
    </span>
  );
}

function isRuntimeLoadedModel(
  loadedModelId: string | undefined,
  activeGgufVariant: string | null | undefined,
  modelId: string,
  variantPolicy: "none" | "required" | "ignore",
  aliasIds: readonly string[] = [],
): boolean {
  if (
    !modelIdsMatchForPicker(loadedModelId, modelId) &&
    !aliasIds.some((id) => modelIdsMatchForPicker(loadedModelId, id))
  )
    return false;
  if (variantPolicy === "ignore") return true;
  const hasActiveGgufVariant = !ggufVariantsMatchForPicker(
    activeGgufVariant,
    null,
  );
  return variantPolicy === "required"
    ? hasActiveGgufVariant
    : !hasActiveGgufVariant;
}

function artifactBudget(gpu: {
  memoryTotalGb: number;
  systemRamAvailableGb: number;
  denseQuantSchemes?: readonly string[];
  quantisedStreaming?: boolean;
  extraOffloadFitTiers?: DeviceBudget["extraOffloadFitTiers"];
}): DeviceBudget {
  return {
    gpuGb: gpu.memoryTotalGb,
    systemRamGb: gpu.systemRamAvailableGb,
    // Judges a pre-quantised row by that checkpoint's size, not the bf16 shards it replaces.
    denseQuantSchemes: gpu.denseQuantSchemes,
    quantisedStreaming: gpu.quantisedStreaming,
    extraOffloadFitTiers: gpu.extraOffloadFitTiers,
  };
}

// Shared row columns; widths are em of the slot's text so they follow the UI font scale.
const META_COLUMN = {
  // Capped at "UD-Q4_K_XL" (longer quants clip).
  quant: "min-[560px]:max-w-[7.2em]",
  // Each width is the widest set its scope draws; wider makes min-w-min shift later columns.
  badge: "min-w-min min-[560px]:w-[calc(26px*var(--ui-space-scale,1))]",
  // One glyph plus the disk mark (26 + 4 + 14).
  badgeMid: "min-w-min min-[560px]:w-[calc(44px*var(--ui-space-scale,1))]",
  badgeDevice: "min-w-min min-[560px]:w-[calc(26px*var(--ui-space-scale,1))]",
  // Hub: the disk mark beside a glyph (26 + 4 + 14).
  badgeWide: "min-w-min min-[560px]:w-[calc(44px*var(--ui-space-scale,1))]",
  vram: "min-w-min min-[560px]:w-[calc(18px*var(--ui-space-scale,1))]",
  // Device rows reserve the slot so the quant column stays aligned; 4.4em fits "0.35B" at text-ui-10.
  param: "min-w-min min-[560px]:w-[4.4em]",
  paramWide: "min-w-min min-[560px]:w-[5.2em]",
  // formatBytes writes no space ("536MB"), so 3.2em suffices.
  size: "min-w-min min-[560px]:w-[3.2em]",
  format: "min-[560px]:w-[calc(14px*var(--ui-space-scale,1))]",
} as const;

const DEVICE_META_GAP = "gap-[max(6px,calc(6px*var(--ui-space-scale,1)))]";

const downloadedRowButtonClassName =
  "bg-transparent pr-1 hover:bg-transparent focus-visible:bg-transparent dark:bg-transparent dark:hover:bg-transparent dark:focus-visible:bg-transparent";
// Not focus-within: the dots menu returns focus on close, leaving the row lit.
const downloadedRowShellClassName = (
  selected: boolean,
  hasMemoryBar = false,
) =>
  cn(
    "group flex items-center transition-colors hover:bg-sidebar-accent has-[:focus-visible]:bg-sidebar-accent has-[[data-state=open]]:bg-sidebar-accent",
    hasMemoryBar ? "rounded-2xl" : "rounded-full",
    selected && "bg-sidebar-accent",
  );

// One gutter for every row so columns never shift by a button.
const ROW_ACTIONS_CLASS =
  "mr-0.5 flex w-[calc(38px*var(--ui-space-scale,1))] shrink-0 items-center justify-end -space-x-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100 has-[[data-state=open]]:opacity-100 [@media(hover:none)]:opacity-100";

// A border snaps to whole pixels, so every row matches.
const PINNED_DROP_CUE_BASE =
  "before:pointer-events-none before:absolute before:inset-x-2 before:z-10 before:h-0 before:border-t-[1.5px] before:border-primary before:content-['']";
const PINNED_DROP_CUE: Record<PinnedDropEdge, string> = {
  top: `${PINNED_DROP_CUE_BASE} before:top-0`,
  bottom: `${PINNED_DROP_CUE_BASE} before:bottom-0`,
};

// Partial rows cannot load, so the menu is their only affordance and stays visible.
const ROW_ACTIONS_PINNED_CLASS = cn(ROW_ACTIONS_CLASS, "opacity-100");

// Same box as ModelLoadSettingsAction, so heading and row buttons align.
const HEADING_ACTION_CLASS =
  "flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground/80 transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";

function ConnectedGroupHeading({
  icon,
  label,
  collapsed,
  onToggle,
  onConfigure,
  configureLabel,
}: {
  icon: ReactNode;
  label: string;
  collapsed: boolean;
  onToggle: () => void;
  onConfigure?: () => void;
  configureLabel?: string;
}) {
  return (
    // Matches ListLabel's pt-3 and gap-1.5 so headings align across tabs.
    <div className="group/heading flex items-center justify-between gap-1 px-2.5 pb-1 pt-3">
      <span className="flex min-w-0 items-center gap-1.5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
        {icon}
        <span className="min-w-0 truncate">{label}</span>
      </span>
      <div className="-mr-2 flex shrink-0 items-center -space-x-0.5">
        {onConfigure ? (
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                onClick={onConfigure}
                aria-label={configureLabel ?? "Connection settings"}
                className={cn(
                  HEADING_ACTION_CLASS,
                  "opacity-0 group-hover/heading:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100",
                )}
              >
                <HugeiconsIcon
                  icon={Settings02Icon}
                  strokeWidth={1.75}
                  className="size-3"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="tooltip-compact">
              {configureLabel ?? "Connection settings"}
            </TooltipContent>
          </Tooltip>
        ) : null}
        <button
          type="button"
          onClick={onToggle}
          aria-label={collapsed ? "Expand section" : "Collapse section"}
          className={HEADING_ACTION_CLASS}
        >
          {collapsed ? (
            <ChevronRightIcon className="size-3" />
          ) : (
            <ChevronDownIcon className="size-3" />
          )}
        </button>
      </div>
    </div>
  );
}

function ModelRow({
  label,
  meta,
  selected,
  loaded = false,
  onClick,
  vramStatus,
  vramEst,
  vramBudget,
  gpuGb,
  tooltipText,
  hubUrl,
  optionProps,
  onArrowDownIntoChildren,
  capabilities,
  hideOwner,
  downloaded,
  partial,
  partialResumable,
  showVision,
  quantChip,
  tags,
  alignMeta,
  showSize,
  memory,
  className,
}: {
  label: string;
  meta?: string | null;
  selected?: boolean;
  /** Override badge state when authoritative runtime state is available. */
  loaded?: boolean;
  onClick: () => void;
  vramStatus?: GgufFitClass | VramFitStatus | null;
  vramEst?: number;
  vramBudget?: CuratedBudget;
  gpuGb?: number;
  tooltipText?: ReactNode;
  /** Hugging Face address shown on hover for Hub rows. */
  hubUrl?: string;
  optionProps?: ModelRowOptionProps;
  onArrowDownIntoChildren?: () => boolean;
  /** Capability override (HF rows have tags); falls back to name detection. */
  capabilities?: ModelCapabilities;
  /** Hide the "owner/" prefix (e.g. Recommended, where all are unsloth). */
  hideOwner?: boolean;
  downloaded?: boolean;
  /** Incomplete snapshot; mutually exclusive with `downloaded`. */
  partial?: boolean;
  /** Whether the partial can resume; undefined reads as no so the mark never promises a resume. */
  partialResumable?: boolean;
  showVision?: boolean;
  quantChip?: string | null;
  /** Chips for the artifact format and, when variants differ only by it, resolution. */
  tags?: string[];
  /** Column layout (see META_COLUMN): "device" reserves the quant chip, "hub" the download and VRAM badges. */
  alignMeta?: "device" | "hub";
  /** Hold the size column open (MLX and Safetensors filters, one download per repo). */
  showSize?: boolean;
  memory?: ModelMemorySource;
  className?: string;
}) {
  const exceeds = isOverBudget(vramStatus);
  const showVramTooltip =
    vramEst != null && vramEst > 0 && gpuGb != null && gpuGb > 0;
  const vramTooltipText =
    showVramTooltip && vramStatus
      ? exceeds
        // "memory", not "VRAM": a `partial` GGUF splits across VRAM and RAM.
        ? vramBudget
          ? curatedBudgetText(vramEst, gpuGb, vramBudget)
          : `Needs ~${vramEst}GB memory (GPU: ${gpuGb}GB)`
        : vramStatus === "tight" || vramStatus === "marginal"
          ? `~${vramEst}GB VRAM (tight fit on ${gpuGb}GB)`
          : `~${vramEst}GB VRAM`
      : null;

  const { owner, name } = splitRepoLabel(label);
  // Hide our own owner: the list is nearly all unsloth/.
  const showOwner = !!owner && !hideOwner && !isUnslothOwner(owner);
  const parsed = parseMetaTokens(meta);
  const paramLabel = parsed.param ?? extractParamLabel(name) ?? null;
  const caps = capabilities ?? detectCapabilities({ id: label });
  const capabilityScope = useContext(CapabilityScope);
  const capabilityBadges = visibleCapabilityBadges(caps, capabilityScope);
  const showCaps = capabilityBadges.length > 0;
  const aligned = alignMeta !== undefined;
  // Reserve only what this picker's scope can draw.
  const badgeColumn =
    capabilityScope === null || capabilityScope.length > 1
      ? alignMeta === "device"
        ? META_COLUMN.badgeDevice
        : META_COLUMN.badgeWide
      : capabilityScope.length === 1
        ? META_COLUMN.badgeMid
        : META_COLUMN.badge;
  const formatDot = parsed.formats[0]
    ? {
        tone: parsed.formats[0].tone,
        label: parsed.formats.map((f) => f.label).join(" · "),
      }
    : null;
  const leading = formatDot ? <FormatTag {...formatDot} /> : null;

  // Only the selected row charts memory, so the list stays scannable.
  const memorySegments = useModelMemory(selected ? memory : undefined, gpuGb);
  const showMemoryBar = memorySegments.status !== "unknown";

  const nameRef = useRef<HTMLSpanElement>(null);
  const nameHoverTimer = useRef<number | undefined>(undefined);
  const [tooltipOpen, setTooltipOpen] = useState(false);
  // Touch has no hover: a tap on the name opens it and a tap elsewhere closes it.
  const [tappedOpen, setTappedOpen] = useState(false);
  useEffect(() => () => window.clearTimeout(nameHoverTimer.current), []);
  useEffect(() => {
    if (!tappedOpen) return;
    const release = (event: Event) => {
      const target = event.target as Element | null;
      if (nameRef.current?.contains(target)) return;
      if (target?.closest?.('[data-slot="tooltip-content"]')) return;
      setTappedOpen(false);
      setTooltipOpen(false);
    };
    // touchstart covers WebViews without pointer events.
    document.addEventListener("pointerdown", release, true);
    document.addEventListener("touchstart", release, { capture: true, passive: true });
    return () => {
      document.removeEventListener("pointerdown", release, true);
      document.removeEventListener("touchstart", release, true);
    };
  }, [tappedOpen]);
  const onNameEnter = (event: React.PointerEvent) => {
    if (event.pointerType === "touch") return;
    window.clearTimeout(nameHoverTimer.current);
    nameHoverTimer.current = window.setTimeout(() => setTooltipOpen(true), 700);
  };
  const onNameLeave = () => {
    window.clearTimeout(nameHoverTimer.current);
    setTappedOpen(false);
    setTooltipOpen(false);
  };
  const onNamePointerLeave = (event: React.PointerEvent) => {
    if (event.pointerType === "touch") return;
    if (nameRef.current?.closest("button")?.matches(":focus-visible")) {
      window.clearTimeout(nameHoverTimer.current);
      return;
    }
    onNameLeave();
  };
  // Read at pointerdown: the trigger closes an open tooltip before the click lands.
  const openAtTap = useRef<boolean | null>(null);
  const onNameDown = (event: React.PointerEvent) => {
    if (event.pointerType === "touch") openAtTap.current = tooltipOpen;
  };
  const hasTooltip = Boolean(vramTooltipText || tooltipText || hubUrl || formatDot);
  // Like hover on iOS: the first tap shows the details, the next picks the row.
  const onNameClick = (event: React.MouseEvent) => {
    const wasOpen = openAtTap.current ?? tooltipOpen;
    openAtTap.current = null;
    if (!hasTooltip || !isTouchClick(event)) return;
    if (wasOpen) {
      onNameLeave();
      return;
    }
    event.stopPropagation();
    setTappedOpen(true);
    setTooltipOpen(true);
  };
  const onTooltipOpenChange = (next: boolean) => {
    if (!next) {
      onNameLeave();
      return;
    }
    const focused = document.activeElement;
    if (
      focused?.matches(":focus-visible") &&
      nameRef.current &&
      focused.contains(nameRef.current)
    ) {
      setTooltipOpen(true);
    }
  };

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
        // 5.5px puts the dot (centred in a 14px target) at 10px, level with section labels.
        "group/row flex w-full flex-col items-stretch py-1.5 pl-[calc(5.5px*var(--ui-space-scale,1))] pr-2 text-left text-sm transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        showMemoryBar ? "rounded-2xl" : "rounded-full",
        selected && "bg-sidebar-accent",
        className,
      )}
    >
      <span
        className={cn(
          "flex w-full items-center",
          alignMeta === "device" ? DEVICE_META_GAP : "gap-1",
          exceeds &&
            !selected &&
            "opacity-60 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100",
        )}
      >
        <span className="flex min-w-0 flex-1 items-baseline">
          {aligned || leading ? (
            <span
              className={cn(
                "mr-1 flex shrink-0 items-center self-center",
                META_COLUMN.format,
              )}
            >
              {leading}
            </span>
          ) : null}
          {showOwner ? (
            <span className="inline-flex min-w-0 max-w-[45%] shrink items-baseline text-ui-13 text-muted-foreground/80">
              <span className="truncate">{owner}</span>
              <span className="shrink-0 text-muted-foreground/80">/</span>
            </span>
          ) : null}
          <span className="min-w-0 flex-1 truncate">
            <span
              ref={nameRef}
              onPointerEnter={onNameEnter}
              onPointerDown={onNameDown}
              onPointerLeave={onNamePointerLeave}
              onClick={onNameClick}
            >
              {name}
            </span>
          </span>
          {alignMeta === "device" && partial ? (
            <span className="ml-[max(6px,calc(6px*var(--ui-space-scale,1)))] flex shrink-0 items-center self-center">
              <PartialBadge resumable={partialResumable} />
            </span>
          ) : null}
          {aligned && loaded && (
            <DotTag
              tone="success"
              label="Loaded"
              className="ml-2 h-[calc(18px*var(--ui-space-scale,1))] shrink-0 gap-1 self-center rounded-md px-1.5"
              dotClassName="size-[calc(5px*var(--ui-space-scale,1))]"
            />
          )}
          {alignMeta !== "device" && quantChip ? (
            <span className="ml-2 shrink-0 rounded-md bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] px-1.5 py-px font-mono text-ui-10 text-muted-foreground dark:bg-muted">
              {quantChip}
            </span>
          ) : null}
          {tags && tags.length > 0 ? (
            <span className="ml-1.5 flex shrink-0 items-center gap-1 self-center">
              {tags.map((tag) => (
                <QuantChip key={tag} label={tag} />
              ))}
            </span>
          ) : null}
        </span>
        <span
          className={cn(
            "ml-auto flex shrink-0 items-center",
            alignMeta === "device" ? DEVICE_META_GAP : aligned ? "gap-1" : "gap-1.5",
          )}
        >
          {alignMeta === "device" && quantChip ? (
            <span
              className={cn(
                "flex shrink-0 items-center justify-end text-ui-9",
                META_COLUMN.quant,
              )}
            >
              <QuantChip label={quantChip} />
            </span>
          ) : null}
          {aligned ? (
            <span
              className={cn(
                "flex shrink-0 items-center gap-1 text-ui-10",
                alignMeta === "device" ? "justify-start" : "justify-end",
                badgeColumn,
              )}
            >
              {showCaps && <CapabilityIcons caps={caps} />}
              {showVision && <VisionBadge />}
              {partial && alignMeta !== "device" ? (
                <PartialBadge resumable={partialResumable} />
              ) : null}
              {downloaded && !partial && !loaded ? <DownloadedBadge /> : null}
            </span>
          ) : (
            <>
              {showCaps && <CapabilityIcons caps={caps} />}
              {showVision && <VisionBadge />}
              {loaded && (
                <DotTag
                  tone="success"
                  label="Loaded"
                  className="h-[calc(18px*var(--ui-space-scale,1))] gap-1 rounded-md px-1.5"
                  dotClassName="size-[calc(5px*var(--ui-space-scale,1))]"
                />
              )}
              {partial ? <PartialBadge resumable={partialResumable} /> : null}
              {downloaded && !partial && !loaded ? <DownloadedBadge /> : null}
            </>
          )}
          {alignMeta === "hub" ? (
            <span
              className={cn(
                "flex shrink-0 items-center justify-end text-ui-9",
                META_COLUMN.vram,
              )}
            >
              <VramBadge status={vramStatus} revealOnHover={!selected} />
            </span>
          ) : (
            <VramBadge status={vramStatus} revealOnHover={!selected} />
          )}
          {aligned ? (
            <span
              className={cn(
                "flex shrink-0 items-center text-ui-10",
                alignMeta === "hub"
                  ? cn("justify-end", META_COLUMN.paramWide)
                  : cn("justify-start", META_COLUMN.param),
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
          {/* GGUF repos hold several quants, so their size shows only once expanded. */}
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
      </span>
      {showMemoryBar ? (
        <ModelMemoryBar segments={memorySegments} compact={true} />
      ) : null}
    </button>
  );

  const hubUrlLine = hubUrl ? (
    <span className="block mt-1 text-ui-10 text-muted-foreground break-all">
      {hubUrl}
    </span>
  ) : null;

  // Keyboard focus never reaches the dot's hover, so the row tooltip carries the format too.
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
      <Tooltip open={tooltipOpen} onOpenChange={onTooltipOpenChange}>
        <TooltipTrigger asChild={true}>{content}</TooltipTrigger>
        <TooltipContent
          side="right"
          className="tooltip-compact max-w-[calc(15rem*var(--ui-space-scale,1))] break-words"
        >
          {tooltipBody}
        </TooltipContent>
      </Tooltip>
    );
  }
  return content;
}


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
    (candidate.pending_drafter_filename === undefined ||
      candidate.pending_drafter_filename === null ||
      typeof candidate.pending_drafter_filename === "string") &&
    (candidate.pending_drafter_size_bytes === undefined ||
      (typeof candidate.pending_drafter_size_bytes === "number" &&
        Number.isFinite(candidate.pending_drafter_size_bytes) &&
        candidate.pending_drafter_size_bytes >= 0)) &&
    // Absent on an older backend, which groups the repo as one, so it must never reject the row.
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
        dependencies_resolved?: unknown;
      }
    | null
    | undefined,
): {
  variants: GgufVariantDetail[];
  defaultVariant: string | null;
  hasVision: boolean | undefined;
  contextLength: number | null;
  resolvedLocally: boolean;
  dependenciesResolved: boolean;
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
    hasVision: normalizeGgufVisionCapability(res?.has_vision),
    contextLength:
      typeof contextLength === "number" &&
      Number.isFinite(contextLength) &&
      contextLength >= 0
        ? contextLength
        : null,
    // The backend's verdict resolves existence-first; an older server leaves the prefix test.
    resolvedLocally: res?.resolved_locally === true,
    dependenciesResolved: res?.dependencies_resolved === true,
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

/** The collapsed row never mounts the expander, so this is its only quant source. */
interface SoleDownloadedQuant {
  variant: GgufVariantDetail;
  hasVision: boolean | undefined;
}

/** The repo's one complete cached quant, or null when its dependencies cannot be verified. */
async function readSoleQuant(
  target: SoleQuantTarget,
  hfToken?: string,
): Promise<SoleDownloadedQuant | null> {
  try {
    const res = await listGgufVariantsCached(target.repoId, hfToken, {
      localOnly: true,
      localPath: target.localSource,
      includeCacheLocations: target.includeCacheLocations,
    });
    const normalized = normalizeGgufVariantsResponse(res);
    const variant = verifiedSoleHubVariant(
      normalized.variants,
      normalized.resolvedLocally,
      normalized.dependenciesResolved,
    );
    return variant ? { variant, hasVision: normalized.hasVision } : null;
  } catch {
    return null;
  }
}

function soleQuantNeedsExpander(
  target: SoleQuantTarget,
  quant: string,
  hfToken?: string,
): Promise<boolean> {
  if (isHuggingFaceOffline()) return Promise.resolve(false);
  return hubWithdrawsSoleQuant(
    () =>
      listGgufVariantsCached(target.repoId, hfToken, {
        localPath: target.localSource,
        includeCacheLocations: target.includeCacheLocations,
      }),
    quant,
  );
}

const EMPTY_SOLE_QUANT_ENTRIES: ReadonlyMap<
  string,
  SoleQuantEntry<SoleDownloadedQuant>
> = new Map();
// A worker pool, not fixed batches, so one slow repo holds up only itself.
const SOLE_QUANT_WORKERS = 6;

/** On Device repos with exactly one quant on disk, which collapse into one row when
 *  "Show all quantizations" is off. Kept per repo. */
function useSoleDownloadedQuants(
  repos: readonly CachedGgufRepo[],
  { enabled, hfToken }: { enabled: boolean; hfToken?: string },
): {
  quants: ReadonlyMap<string, SoleDownloadedQuant>;
  pending: ReadonlySet<string>;
} {
  const repoIds = useMemo(() => repos.map((repo) => repo.repo_id), [repos]);
  const variantsVersion = useGgufVariantsCacheVersions(repoIds);
  const targets = useMemo(() => {
    const versions = variantsVersion.split(",");
    return repos.map((repo, index) => {
      const localSource = repo.load_id || repo.cache_path || null;
      const fingerprint = soleQuantFingerprint(repo);
      return {
        repoId: repo.repo_id,
        localSource,
        includeCacheLocations: !mediaPageForTask(repo.task),
        fingerprint,
        key: soleQuantKey(versions[index], localSource, fingerprint),
      };
    });
  }, [repos, variantsVersion]);

  const [entries, setEntries] = useState<
    ReadonlyMap<string, SoleQuantEntry<SoleDownloadedQuant>>
  >(EMPTY_SOLE_QUANT_ENTRIES);
  const { quants, pending, stale } = useMemo(
    () => partitionSoleQuants(targets, entries, { enabled }),
    [targets, entries, enabled],
  );

  // A change outside this tab moves the row's bytes without touching this cache, so drop it.
  const fingerprintsRef = useRef(new Map<string, string>());
  useEffect(() => {
    for (const repoId of takeDriftedRepos(targets, fingerprintsRef.current)) {
      invalidateGgufVariantsCache(repoId);
    }
  }, [targets]);

  // Read at call time so a token change does not strand the reader.
  const hfTokenRef = useRef(hfToken);
  hfTokenRef.current = hfToken;
  const mountedRef = useRef(true);
  useEffect(() => {
    // Set on setup too: StrictMode replays effects.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const hubProbeLimitRef = useRef(createTaskLimiter(SOLE_QUANT_WORKERS));
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
        if (!quant) return;
        // A later Hub answer may still send the row back to its expander.
        void hubProbeLimitRef.current(() =>
          soleQuantNeedsExpander(
            target,
            quant.variant.quant,
            hfTokenRef.current,
          ),
        ).then((withdraw) => {
          if (!withdraw || !mountedRef.current) return;
          setEntries((prev) => {
            if (prev.get(target.repoId)?.key !== target.key) return prev;
            const next = new Map(prev);
            next.set(target.repoId, { key: target.key, quant: null });
            return next;
          });
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
  pipelineTag,
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
  diffusionLoad = false,
  hostPooledMemory = false,
  gpuCount,
  loadedQuants,
}: {
  repoId: string;
  loadedQuants?: readonly string[];
  pipelineTag?: string | null;
  /** Images / Video: the diffusion backend places GGUFs, so the llama.cpp budget does not apply. */
  diffusionLoad?: boolean;
  /** The load device's memory is a window into host RAM (Apple Silicon, ROCm APU). */
  hostPooledMemory?: boolean;
  /** How many GPUs gpuGb is the sum of, for the loader's per-card VRAM reserve. */
  gpuCount?: number;
  /** Snapshot the cached listing pinned this repo to, if any. */
  loadId?: string | null;
  cachePath?: string | null;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  hfToken?: string;
  parentOptionKey?: string;
  onNavigatePastStart?: () => void;
  onNavigatePastEnd?: () => void;
  onConfigure?: (id: string, meta: ModelSelectorChangeMeta) => void;
  sourceOverride?: ModelSelectorChangeMeta["source"];
  variantActions?: {
    onUpdate?: (quant: string, expectedBytes: number) => Promise<void> | void;
    updateTitle?: string;
    renderUpdateDescription?: (quant: string) => ReactNode;
    getUpdateSuccessMessage?: (quant: string) => string;
    updateDisabled?: boolean;
    onDelete?: (quant: string, cachePath?: string | null) => Promise<void> | void;
    deleteTitle?: string;
    renderDeleteDescription?: (quant: string) => ReactNode;
    getDeleteSuccessMessage?: (quant: string) => string;
    deleteDisabled?: boolean;
  };
  /** On Device rows honor the All quantizations setting; browse lists always show every quant. */
  onDevice?: boolean;
  /** Only managed cached-Hub rows can pin quants; local-path expanders leave this false. */
  allowPin?: boolean;
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
  const [hasVision, setHasVision] = useState<boolean | undefined>(undefined);
  const [nativeContext, setNativeContext] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  // Not derivable from loadId/cachePath, which a downloaded hub model also carries.
  const [resolvedLocally, setResolvedLocally] = useState(false);
  const localSource = loadId || cachePath || null;
  const showAllQuantizations = useChatRuntimeStore(
    (s) => s.showAllQuantizations,
  );

  useEffect(() => {
    let canceled = false;
    // Abort on collapse: a stalled request holds a per-host connection and starves downloads.
    const controller = new AbortController();
    queueMicrotask(() => {
      if (canceled) return;
      setLoading(true);
      setError(null);
      // Reset so the previous row's locality is not applied to this one.
      setResolvedLocally(false);
    });

    let cachedResponse:
      | Awaited<ReturnType<typeof listGgufVariants>>
      | undefined;
    const applyResponse = (
      res: Awaited<ReturnType<typeof listGgufVariants>>,
    ) => {
      if (canceled) return;
      const normalized = normalizeGgufVariantsResponse(res);
      setVariants(normalized.variants);
      setDefaultVariant(normalized.defaultVariant);
      setHasVision(normalized.hasVision);
      if (normalized.hasVision !== undefined) {
        onHasVision?.(normalized.hasVision);
      }
      setNativeContext(normalized.contextLength);
      setResolvedLocally(normalized.resolvedLocally);
    };

    loadPickerGgufVariants(
      (localOnly) =>
        listGgufVariants(repoId, hfToken, {
          ...(localSource ? { localPath: localSource } : {}),
          localOnly,
          includeCacheLocations: !mediaPageForTask(pipelineTag),
          signal: controller.signal,
        }),
      {
        onDevice,
        canDiscoverRemote: () => !isHuggingFaceOffline(),
        signal: controller.signal,
      },
      (res) => {
        cachedResponse = res;
        applyResponse(res);
        if (!canceled) setLoading(false);
      },
    )
      .then((res) => {
        if (res !== cachedResponse) applyResponse(res);
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
  }, [repoId, localSource, refreshKey, hfToken, pipelineTag, onDevice]);

  const isLocalPath = /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(
    repoId,
  );
  // Repo-level mmproj presence is not proof for this quant; only absence is authoritative.
  const variantVisionHint = hasVision === false ? false : undefined;
  // The prefix test misses marker-less relative dirs, so ask the listing.
  const checkpointIsLocal = isLocalPath || resolvedLocally;

  const handleVariantClick = useCallback(
    // `filename` is required: the diffusion pages gate their GGUF branch on meta.ggufFilename.
    (
      quant: string,
      filename: string,
      downloaded?: boolean,
      sizeBytes?: number,
      downloadPresentation?: ModelSelectorChangeMeta["downloadPresentation"],
      contextLength: number | null = nativeContext,
    ) => {
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
        downloadPresentation,
        contextLength: isAvailable ? contextLength : undefined,
        isGguf: true,
        isVision: variantVisionHint,
        pipelineTag,
      });
    },
    [
      repoId,
      loadId,
      isLocalPath,
      onSelect,
      sourceOverride,
      nativeContext,
      pipelineTag,
      variantVisionHint,
    ],
  );

  // The saved VRAM Budget, which is what the loader admits against.
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  const anyBudgetGb = (gpuGb ?? 0) > 0 || (systemRamGb ?? 0) > 0;

  const getGgufFit = useCallback(
    (sizeBytes: number): GgufFitClass => {
      // A known zero budget (e.g. Vulkan) means OOM, distinct from not probed yet.
      if (!anyBudgetGb) return budgetKnown ? "oom" : "fits";
      if (diffusionLoad) {
        return classifyMediaGgufFit(
          sizeBytes,
          gpuGb ?? 0,
          mediaRamBudgetGb(systemRamGb ?? 0, hostPooledMemory),
        );
      }
      return classifyGgufFit(sizeBytes, {
        gpuGb: gpuGb ?? 0,
        systemRamGb: systemRamGb ?? 0,
        budgetFraction,
        gpuCount,
      });
    },
    [
      budgetKnown,
      anyBudgetGb,
      gpuGb,
      systemRamGb,
      budgetFraction,
      diffusionLoad,
      hostPooledMemory,
      gpuCount,
    ],
  );

  // One verdict per variant from the download footprint (incl. companions), so badge, order and star agree.
  const getVariantFit = useCallback(
    (variant: GgufVariantDetail): GgufFitClass =>
      getGgufFit(ggufVariantFitSizeBytes(variant)),
    [getGgufFit],
  );

  const variantGroups = useMemo(
    () => groupGgufVariantsForPicker(variants ?? []),
    [variants],
  );
  const preferredByGroup = useMemo(
    () => preferredGgufVariantByGroup(variantGroups, defaultVariant),
    [variantGroups, defaultVariant],
  );

  const effectiveRecommendedByGroup = useMemo(() => {
    const recommended = new Map<string, string>();
    for (const group of variantGroups) {
      const preferred = preferredByGroup.get(group.key) ?? null;
      if (!anyBudgetGb && !budgetKnown) {
        if (preferred) recommended.set(group.key, preferred.quant);
        continue;
      }
      const pick =
        recommendedQuantForDevice(group.variants, getGgufFit, preferred) ??
        preferred;
      if (pick) recommended.set(group.key, pick.quant);
    }
    return recommended;
  }, [variantGroups, preferredByGroup, anyBudgetGb, budgetKnown, getGgufFit]);
  // Keyed by presentation group while footprints bucket by dependency_key, so ask via the variant.
  const recommendedQuantForVariant = useMemo(() => {
    const byVariant = new Map<GgufVariantDetail, string>();
    for (const group of variantGroups) {
      const recommended = effectiveRecommendedByGroup.get(group.key);
      if (recommended === undefined) continue;
      for (const variant of group.variants) byVariant.set(variant, recommended);
    }
    return byVariant;
  }, [variantGroups, effectiveRecommendedByGroup]);

  const sortedVariants = useMemo(() => {
    if (!variants) return variants;
    // Tier: 0 = downloaded+fits, 1 = downloaded+tight, 2 = fits, 3 = tight, 4 = OOM
    const tierOf = (v: GgufVariantDetail) => {
      const f = getVariantFit(v);
      if (f === "oom") return 4;
      const base = f === "fits" ? 0 : 1;
      return v.downloaded ? base : base + 2;
    };
    return variantGroups.flatMap((group) => {
      const recommended = effectiveRecommendedByGroup.get(group.key);
      return [...group.variants].sort((a, b) => {
        const aTier = tierOf(a);
        const bTier = tierOf(b);
        if (aTier !== bTier) return aTier - bTier;

        const aIsRec = a.quant === recommended;
        const bIsRec = b.quant === recommended;
        if (aIsRec !== bIsRec) return aIsRec ? -1 : 1;

        // fits: largest first; tight/OOM: smallest first (closest to fitting).
        const fitsInGpu = aTier === 0 || aTier === 2;
        return fitsInGpu
          ? b.size_bytes - a.size_bytes
          : a.size_bytes - b.size_bytes;
      });
    });
  }, [variants, variantGroups, effectiveRecommendedByGroup, getVariantFit]);

  const displayVariants = useMemo(() => {
    if (!sortedVariants) return sortedVariants;
    return visibleGgufVariants(sortedVariants, {
      onDevice,
      showAll: showAllQuantizations,
    });
  }, [sortedVariants, showAllQuantizations, onDevice]);
  const displayVariantGroups = useMemo(
    () => groupGgufVariantsForPicker(displayVariants ?? []),
    [displayVariants],
  );
  const hideH3PrunedBuild = useMemo(
    () => h3PickerHasOnlyPrunedBuilds(displayVariants ?? []),
    [displayVariants],
  );

  // A diffusion GGUF's companion set is per family, not repo-wide; group by dependency_key so totals are right.
  const footprintVariants = useMemo(() => {
    const byKey = new Map<string, GgufVariantDetail>();
    for (const variant of displayVariants ?? []) {
      // An unkeyed repo (older backend) collapses to one group.
      const key = variant.dependency_key ?? "";
      const current = byKey.get(key);
      if (current === undefined) {
        byKey.set(key, variant);
        continue;
      }
      // Asked per variant, since two families in one repo can share quant names.
      const recommended = recommendedQuantForVariant.get(variant);
      if (
        recommended !== undefined &&
        current.quant !== recommended &&
        variant.quant === recommended
      ) {
        byKey.set(key, variant);
      }
    }
    return Array.from(byKey.values());
  }, [displayVariants, recommendedQuantForVariant]);
  const [companionBytesByKey, setCompanionBytesByKey] = useState<
    Map<string, number>
  >(() => new Map());
  useEffect(() => {
    let cancelled = false;
    setCompanionBytesByKey(new Map());
    // Local paths are resolved too: the remote base is the larger half.
    if (!resolveDownloadFootprint) {
      return () => {
        cancelled = true;
      };
    }
    for (const footprintVariant of footprintVariants) {
      const dependencyKey = footprintVariant.dependency_key ?? "";
      const expectedBytes = ggufVariantExpectedBytes(footprintVariant);
      void resolveDownloadFootprint(repoId, {
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
          // A checkpoint on disk is not in required_bytes, so subtract nothing for it.
          const checkpoint = checkpointIsLocal
            ? 0
            : footprint.checkpointBytes > 0
              ? footprint.checkpointBytes
              : expectedBytes;
          const companion = footprint.requiredBytes - checkpoint;
          if (Number.isFinite(companion) && companion > 0) {
            // A fresh Map per resolution, since groups resolve independently and React compares by identity.
            setCompanionBytesByKey((previous) => {
              const next = new Map(previous);
              next.set(dependencyKey, companion);
              return next;
            });
          }
        })
        .catch(() => {
          // Keep the checkpoint size when the companion footprint is unavailable.
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
      {!onDevice && !displayVariantGroups.some((group) => group.title) && (
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
      {!onDevice &&
        hasVision &&
        displayVariantGroups.some((group) => group.title) && (
          <div className="px-2 pt-1">
            <span className="flex items-center gap-0.5 text-ui-9 font-medium text-indigo-700 dark:text-indigo-300">
              <HugeiconsIcon
                icon={ViewIcon}
                className="size-3"
                strokeWidth={1.8}
              />
              Vision
            </span>
          </div>
        )}
      {displayVariants.map((v) => {
        const group = displayVariantGroups.find((candidate) =>
          candidate.variants.some((variant) => variant.filename === v.filename),
        );
        const showGroupHeading =
          group?.title != null && group.variants[0]?.filename === v.filename;
        // Matching on quant alone works only because an H3 key is unique per file.
        const isRecommended =
          group != null &&
          effectiveRecommendedByGroup.get(group.key) === v.quant;
        const fit = getVariantFit(v);
        const oom = fit === "oom";
        const expectedBytes = ggufVariantExpectedBytes(v);
        const variantContext = ggufVariantContextLength(v, nativeContext);
        // This row's own dependency group, never the listing's.
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
                pendingDrafterPresentation(v),
                variantContext,
              )
            }
            className={cn(
              "flex min-w-0 flex-1 items-center justify-between gap-2 rounded-full py-1 pl-2 pr-1.5 text-left text-sm transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              unusableLocal &&
                "cursor-default opacity-50 hover:bg-transparent dark:hover:bg-transparent",
            )}
          >
            <span className="min-w-0 flex-1 truncate font-mono text-xs">
              <span className={cn(oom && "!text-gray-500 dark:!text-gray-400")}>
                {ggufVariantPickerLabel(v, {
                  h3Grouped: group?.title != null,
                  hideH3PrunedBuild,
                })}
              </span>
              {unusableLocal ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">
                  incomplete
                </span>
              ) : loadedQuants?.some((q) => ggufVariantsMatchForPicker(q, v.quant)) ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-green-600/90 dark:text-green-400/80">
                  loaded
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
              ) : v.partial === true ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">
                  partial
                </span>
              ) : isRecommended ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-primary">
                  recommended
                </span>
              ) : null}
            </span>
            <span className="flex items-center gap-1.5 shrink-0">
              <VramBadge
                status={
                  diffusionRefuses(fit, diffusionLoad, hostPooledMemory)
                    ? "exceeds"
                    : fit
                }
              />
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
        return [
          showGroupHeading && group?.title ? (
            <div key={`${v.filename}:group`} className="px-2 pb-1 pt-2">
              <div className="text-xs font-semibold text-foreground">
                {group.title}
              </div>
              {group.description && (
                <div className="mt-0.5 text-ui-10 leading-snug text-muted-foreground">
                  {group.description}
                </div>
              )}
            </div>
          ) : null,
          <div key={v.filename} className="flex items-center">
            {/* Nested button triggers are not accessible. */}
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
                    contextLength: variantContext,
                    isGguf: true,
                    isVision: variantVisionHint,
                  })
                }
              />
            )}
            {(v.downloaded || v.partial === true) &&
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
                    allowPin && v.downloaded
                      ? {
                          pinned: pinnedKeys.includes(pinKey(repoId, v.quant)),
                          pinLabel: "Pin",
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
                          impact: {
                            repoId,
                            variant: v.quant,
                            cachePath: v.cache_ref ?? v.cache_path,
                          },
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
                            await onDeleteVariant(v.quant, v.cache_ref ?? v.cache_path);
                            if (isChatGgufTask(pipelineTag)) {
                              await reconcileGgufPinsAfterDelete(repoId, hfToken);
                            } else if (pinnedKeys.includes(pinKey(repoId, v.quant))) {
                              togglePinnedQuant(repoId, v.quant);
                            }
                            setRefreshKey((key) => key + 1);
                          },
                        }
                      : undefined
                  }
                />
              )}
          </div>,
        ];
      })}
    </div>
  );
}


function hasGgufSuffix(id: string): boolean {
  return /-GGUF(?:$|-)/i.test(id);
}

function isGgufRepo(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || hasGgufSuffix(id);
}


// Unknown task (null) passes only with no filter.
function taskMatchesFilter(
  repoTask: string | null | undefined,
  filter: HfTaskFilter,
): boolean {
  if (!filter) return true;
  const wanted = Array.isArray(filter) ? filter : [filter];
  return repoTask != null && (wanted as readonly string[]).includes(repoTask);
}

// Owned by the Images page, never chat-loadable.
export const IMAGE_GEN_TASKS = [
  "text-to-image",
  "image-to-image",
  "image-text-to-image",
] as const;

// Owned by the Video page; includes image-to-video (LTX-2) and image-text-to-video (MiniMax-H3).
export const VIDEO_GEN_TASKS = [
  "text-to-video",
  "image-to-video",
  "image-text-to-video",
] as const;

/** Tasks whose GGUFs the diffusion backend places; Audio GGUFs go to llama.cpp or whisper. */
const DIFFUSION_TASKS: ReadonlySet<string> = new Set([
  ...IMAGE_GEN_TASKS,
  ...VIDEO_GEN_TASKS,
]);

// Audio pipeline tasks: owned by the Audio page. TTS, music and separation picks load there; ASR picks map
// to the dictation sidecar.
export const AUDIO_GEN_TASKS = [
  "text-to-speech",
  "automatic-speech-recognition",
  "text-to-audio",
  "audio-to-audio",
] as const;

// Diffusion GGUF archs the Images backend cannot assemble yet; they would 400 on load.
const UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported";

const MEDIA_PAGE_TASKS: readonly string[] = [
  ...IMAGE_GEN_TASKS,
  ...VIDEO_GEN_TASKS,
  ...AUDIO_GEN_TASKS,
];

function mediaPageForTask(
  task: string | null | undefined,
): "images" | "video" | "audio" | null {
  if (!task || !MEDIA_PAGE_TASKS.includes(task)) return null;
  if ((VIDEO_GEN_TASKS as readonly string[]).includes(task)) return "video";
  if ((AUDIO_GEN_TASKS as readonly string[]).includes(task)) return "audio";
  return "images";
}

// Editing checkpoints need an input image, so hide them by id (mirrors _EDIT_KEYWORDS); keep
// the task since FLUX.2-klein carries it too.
const IMAGE_EDIT_KEYWORDS = ["edit", "kontext", "inpaint", "layered"] as const;
// Mirrors the backend's supported edit families.
const SUPPORTED_EDIT_KEYWORDS = [
  "qwen-image-edit",
  "kontext",
  "qwen-image-layered",
  "qwen_image_layered",
  "qwenimagelayered",
] as const;
// Whole segment match, not substring. Mirrors _token_in_needle.
function idHasSegment(id: string, keyword: string): boolean {
  return new RegExp(`(?:^|[-_./\\\\])${keyword}(?:$|[-_./\\\\])`).test(id);
}
function isImageEditModel(repoId: string | null | undefined): boolean {
  if (!repoId) return false;
  const id = repoId.toLowerCase();
  if (SUPPORTED_EDIT_KEYWORDS.some((kw) => idHasSegment(id, kw))) return false;
  return IMAGE_EDIT_KEYWORDS.some((kw) => idHasSegment(id, kw));
}

function passesTaskGate(
  repoTask: string | null | undefined,
  repoId: string | null | undefined,
  filter: HfTaskFilter,
  catalog?: CatalogGroup[],
  activeCatalogArtifactIds?: ReadonlySet<string>,
  localModel?: { opaque?: boolean },
): boolean {
  if (filter) {
    const exactArtifact =
      repoId && catalog ? artifactForRepoId(repoId, catalog) : null;
    return (
      (taskMatchesFilter(repoTask, filter) ||
        curatedAudioInventoryMatches({
          isActiveCatalogArtifact: Boolean(
            repoId &&
              activeCatalogArtifactIds?.has(repoId.trim().toLowerCase()),
          ),
          catalogScope: exactArtifact?.group.scope,
          catalogTask: exactArtifact?.group.task,
          pickerTask: filter,
        })) &&
      !isImageEditModel(repoId)
    ) || localModel?.opaque === true;
  }
  // Unfiltered (chat): diffusion models stay listed and route to their page.
  return repoTask !== UNSUPPORTED_DIFFUSION_TASK;
}

let _cachedGgufCache: CachedGgufRepo[] = [];
let _cachedModelsCache: CachedModelRepo[] = [];
let _lmStudioCache: LocalModelInfo[] = [];
let _localDirCache: LocalModelInfo[] = [];
let _customFolderCache: LocalModelInfo[] = [];
let _scanFoldersCache: ScanFolderInfo[] = [];

/** True when any loadable on-device model is known (partials do not count). */
export function hasDownloadedModels(): boolean {
  return (
    _cachedGgufCache.some((c) => !c.partial) ||
    _cachedModelsCache.some((c) => !c.partial) ||
    _lmStudioCache.length > 0 ||
    _localDirCache.length > 0 ||
    _customFolderCache.length > 0
  );
}

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


type RecommendedSortKey = "trendingScore" | "lastModified";

const RECOMMENDED_SORT_OPTIONS: HubOption<RecommendedSortKey>[] = [
  { value: "trendingScore", label: "Trending" },
  { value: "lastModified", label: "Recent" },
];

type LocalSortKey = "recent" | "downloaded" | "size" | "name";

const LOCAL_SORT_OPTIONS: HubOption<LocalSortKey>[] = [
  { value: "recent", label: "Recent" },
  { value: "size", label: "Size" },
  { value: "name", label: "Name" },
  { value: "downloaded", label: "Downloaded" },
];

const FORMAT_FILTER_LABELS: Record<FormatFilter, string> = {
  all: "All",
  gguf: "GGUF",
  mlx: "MLX",
  safetensors: "Safetensors",
  npu: "NPU",
};

const FORMAT_FILTER_DOTS: Partial<Record<FormatFilter, string>> = {
  gguf: "bg-format-gguf",
  mlx: "bg-format-mlx",
  safetensors: "bg-format-checkpoint",
  npu: "bg-format-npu",
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
const FORMAT_FILTER_OPTIONS_WITHOUT_NPU = FORMAT_FILTER_OPTIONS.filter(
  (option) => option.value !== "npu",
);

type ConnectedSortKey = "provider" | "name";

const CONNECTED_SORT_OPTIONS: HubOption<ConnectedSortKey>[] = [
  { value: "provider", label: "Connection" },
  { value: "name", label: "Name" },
];

// No Audio: connected models never take audio attachments.
type ConnectedModalityFilter = "all" | "vision" | "imageGen";

const CONNECTED_MODALITY_LABELS: Record<ConnectedModalityFilter, string> = {
  all: "All",
  vision: "Vision",
  imageGen: "Image gen",
};

const CONNECTED_MODALITY_OPTIONS: HubOption<ConnectedModalityFilter>[] = (
  Object.keys(CONNECTED_MODALITY_LABELS) as ConnectedModalityFilter[]
).map((value) => ({ value, label: CONNECTED_MODALITY_LABELS[value] }));

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

/** They carry no size, so "size" falls back to name. */
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
    return name(a).localeCompare(name(b));
  });
}

function localModelIsGguf(m: LocalModelInfo): boolean {
  return (
    m.model_format === "gguf" ||
    isGgufRepo(m.id) ||
    isGgufRepo(m.display_name) ||
    m.path.toLowerCase().endsWith(".gguf")
  );
}

function localPathTooltip(
  name: string,
  path: string,
  // An H3 repo holds both partitions in one directory, so the path cannot say which.
  detail?: string,
): ReactNode {
  return (
    <>
      <span className="block break-words">{name}</span>
      {detail ? <span className="mt-0.5 block break-words">{detail}</span> : null}
      <span className="block mt-1 text-ui-10 text-muted-foreground break-all">
        {path}
      </span>
    </>
  );
}

function localModelMeta(
  isGguf = false,
  pipelineTag?: string | null,
  audioType?: string | null,
  familyOverrideRequired = false,
): ModelSelectorChangeMeta {
  return {
    source: "local",
    isLora: false,
    isDownloaded: true,
    ...(isGguf ? { isGguf: true } : {}),
    pipelineTag: pipelineTag ?? null,
    audioType: audioType ?? null,
    familyOverrideRequired,
  };
}

function localDirectGgufMeta(
  pipelineTag?: string | null,
): ModelSelectorChangeMeta {
  return localModelMeta(true, pipelineTag);
}

function hubRepoUrl(id: string | null | undefined): string | undefined {
  const trimmed = id?.trim();
  return trimmed ? `huggingface.co/${trimmed}` : undefined;
}

/** MLX runs on Mac only, so callers gate visibility on the host. */
function localModelIsMlx(m: LocalModelInfo): boolean {
  return isMlxId(m.id) || isMlxId(m.display_name) || isMlxId(m.model_id ?? "");
}

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

export type ModelPickerRowFilter = (row: {
  id: string;
  task?: string | null;
  audioType?: string | null;
  audioWorkflows?: readonly string[] | null;
}) => boolean;

export function HubModelPicker({
  models,
  additionalOnDeviceModels = [],
  loadedModelIdOverride,
  loraModels = [],
  externalModels = [],
  value,
  onSelect: onSelectProp,
  resolveDownloadFootprint,
  onFoldersChange,
  onBrowseHub,
  onConfigureConnection,
  onModelsChange,
  onConfigure,
  deleteDisabled = false,
  section = "downloaded",
  sectionToggle,
  onEject,
  onEjectAll,
  task,
  catalog,
  communityModelPolicy = "none",
  opaqueKind,
  npu,
  rowFilter,
}: {
  models: ModelOption[];
  /** Task-runtime downloads whose cache layout the shared Hub inventory cannot represent. */
  additionalOnDeviceModels?: ModelOption[];
  loadedModelIdOverride?: string;
  loraModels?: LoraModelOption[];
  externalModels?: ExternalModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  onFoldersChange?: () => void;
  onBrowseHub?: () => void;
  /** Opens the connection's settings for a Connected row's gear. */
  onConfigureConnection?: (providerId: string) => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  onConfigure?: (id: string, meta: ModelSelectorChangeMeta) => void;
  deleteDisabled?: boolean;
  /** Section shown when not searching. Search spans all sections. */
  section?: "downloaded" | "recommended" | "connected";
  sectionToggle?: ReactNode;
  onEject?: (modelId?: string) => void;
  onEjectAll?: () => void;
  /** Restrict results to a pipeline task; undefined = all tasks (the chat default). */
  task?: HfTaskFilter;
  catalog?: CatalogGroup[];
  /** Also surface community models for `task`; opt-in for runtimes that load any publisher. */
  communityModelPolicy?: CommunityModelPolicy;
  opaqueKind?: "diffusers_pipeline" | "diffusers_modular_pipeline";
  npu?: NpuPickerSource;
  rowFilter?: ModelPickerRowFilter;
}) {
  const gpu = useGpuInfo();
  const inferenceGpu = useInferenceGpuInfo();
  // Threaded into every fit call, not only the quant rows.
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  // Not `Boolean(task)`: Audio is task-scoped but runs GGUFs under llama.cpp / whisper.
  const diffusionLoad = useMemo(() => {
    const tasks = task ? (typeof task === "string" ? [task] : task) : [];
    return tasks.some((entry) => DIFFUSION_TASKS.has(entry));
  }, [task]);
  // What the backend holds, not the highlight: an image or video load evicts the chat model.
  const selectedCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const residentCheckpoint = useChatRuntimeStore((s) => s.residentCheckpoint);
  const loadedModels = useChatRuntimeStore((s) => s.loadedModels);
  const isChatPicker = task === undefined;
  const chatLoadedModelId = chatModelLoaded({
    checkpoint: selectedCheckpoint,
    isExternalModel: isExternalModelId(selectedCheckpoint),
    residentCheckpoint,
  })
    ? selectedCheckpoint
    : undefined;
  const loadedModelId = loadedModelIdOverride ?? chatLoadedModelId;
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const loadTimes = useModelLoadTimes(value);
  const [listScrolled, setListScrolled] = useState(false);
  const [listMoreBelow, setListMoreBelow] = useState(false);
  const hfToken = useHfTokenStore((s) => s.token);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  const loadedIdSet = useMemo(
    () => new Set(isChatPicker ? loadedModels.map((m) => m.id.toLowerCase()) : []),
    [isChatPicker, loadedModels],
  );
  const isKeptLoaded = (repoId: string) =>
    loadedIdSet.has(repoId.toLowerCase());
  const loadedQuantsFor = (repoId: string): string[] => {
    const quants = isChatPicker
      ? loadedModels
          .filter((m) => m.quant && modelIdsMatchForPicker(m.id, repoId))
          .map((m) => m.quant as string)
      : [];
    if (activeGgufVariant && modelIdsMatchForPicker(loadedModelId, repoId)) {
      quants.push(activeGgufVariant);
    }
    return [...new Set(quants)];
  };
  const loadedRows = useMemo(() => {
    if (!isChatPicker || section !== "recommended") return [];
    const q = debouncedQuery.trim().toLowerCase();
    return loadedModels.filter((m) => m.id.toLowerCase().includes(q));
  }, [isChatPicker, section, debouncedQuery, loadedModels]);
  const loadedRowIds = useMemo(
    () => new Set(loadedRows.map((m) => m.id.toLowerCase())),
    [loadedRows],
  );
  const online = useOnlineStatus();
  const { phase: hubPhase } = useHubAvailability();
  // Sanitize to anonymous on a malformed token, matching the Hub page.
  const accessToken = hfApiToken(hfToken);
  const [recommendedSort, setRecommendedSort] =
    useState<RecommendedSortKey>("trendingScore");
  const {
    results,
    isLoading,
    isLoadingMore,
    fetchMore,
    scannedCount,
    hasMore,
    error: searchError,
    retry: retrySearch,
  } = useHubModelSearch(debouncedQuery, {
    ownerScope: "unsloth",
    sortBy: recommendedSort,
    sortDirection: "desc",
    pinUnslothFirst: true,
    keepUnsupportedTags: true,
    accessToken,
    // Keep Hub hooks idle off the Recommended tab to preserve offline-local behavior.
    enabled: online && section === "recommended",
  });
  const recommendedSearch = useHubModelSearch("", {
    ownerScope: "unsloth",
    sortBy: recommendedSort,
    sortDirection: "desc",
    pinUnslothFirst: true,
    keepUnsupportedTags: true,
    accessToken,
    enabled: online && section === "recommended",
  });

  // Browse must not refetch per keystroke and search must not be pinned to the empty query.
  const communityDiscoveryEnabled =
    shouldDiscoverCommunityModels(communityModelPolicy) &&
    Boolean(task) &&
    online &&
    section === "recommended";
  const communityRecommendedEnabled =
    shouldRecommendCommunityModels(communityModelPolicy) &&
    communityDiscoveryEnabled;
  const communityQuerySearch = useHubModelSearch(debouncedQuery, {
    task,
    ownerScope: "all",
    sortBy: recommendedSort,
    sortDirection: "desc",
    pinUnslothFirst: false,
    accessToken,
    enabled: communityDiscoveryEnabled && debouncedQuery.trim().length > 0,
  });
  const communityBrowse = useHubModelSearch("", {
    task,
    ownerScope: "all",
    sortBy: recommendedSort,
    sortDirection: "desc",
    pinUnslothFirst: false,
    accessToken,
    enabled: communityRecommendedEnabled && debouncedQuery.trim().length === 0,
  });

  // Absence means no hint, so hasGgufSuffix is the fallback.
  const modelGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const model of models) {
      if (model.isGguf) ids.add(model.id.toLowerCase());
    }
    return ids;
  }, [models]);
  // So a tag-only GGUF still expands variants instead of loading as a checkpoint.
  const resultGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const result of [
      ...results,
      ...recommendedSearch.results,
      ...communityQuerySearch.results,
      ...communityBrowse.results,
    ]) {
      if (result.isGguf) ids.add(result.id.toLowerCase());
    }
    return ids;
  }, [
    results,
    recommendedSearch.results,
    communityQuerySearch.results,
    communityBrowse.results,
  ]);
  const isKnownGgufRepo = useCallback(
    (id: string): boolean => {
      const key = id.toLowerCase();
      return isGgufRepo(id, resultGgufIds.has(key) || modelGgufIds.has(key));
    },
    [modelGgufIds, resultGgufIds],
  );

  const [expandedGguf, setExpandedGguf] = useState<string | null>(null);
  const [visionByRepo, setVisionByRepo] = useState<Record<string, boolean>>({});
  const reportVision = useCallback((repoId: string, hasVision: boolean) => {
    setVisionByRepo((prev) =>
      prev[repoId] === hasVision ? prev : { ...prev, [repoId]: hasVision },
    );
  }, []);
  const expandQuantizations = useChatRuntimeStore((s) => s.expandQuantizations);
  const showAllQuantizations = useChatRuntimeStore(
    (s) => s.showAllQuantizations,
  );
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);
  // In memory only, so both reset on reload.
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
  const toggleGgufExpanded = useCallback(
    // A row held back by its sole-quant probe shows nothing, so a click should open it.
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
  const fineTunedSectionRef = useRef<HTMLDivElement>(null);
  const scrollToFineTuned = useCallback(() => {
    setFineTunedCollapsed(false);
    // Two frames so the expand renders before scrolling.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fineTunedSectionRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    });
  }, []);
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

  // `models` is already narrowed to this Audio mode and platform.
  const activeCatalogArtifactIds = useMemo(
    () =>
      new Set(
        task && catalog
          ? models
              .filter((model) => artifactForRepoId(model.id, catalog) !== null)
              .map((model) => model.id.trim().toLowerCase())
          : [],
      ),
    [catalog, models, task],
  );
  const pickerInventory = useChatPickerInventory({
    enabled: true,
    allowedHiddenModelIds: activeCatalogArtifactIds,
    opaqueKind,
  });
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
  // Ollama rows list alongside custom folders: both are user-managed stores outside ./models,
  // and an Ollama root added as a custom folder is where the rows were expected (#9226).
  // Hermes and oMLX downloads are the same kind of store; a source in no bucket never renders.
  const customFolderModels = useMemo(
    () =>
      pickerInventory.localModels.filter(
        (m) =>
          m.source === "custom" ||
          m.source === "ollama" ||
          m.source === "hermes" ||
          m.source === "omlx",
      ),
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
      // An explicit path avoids racing the `setFolderInput` update.
      const raw = overridePath !== undefined ? overridePath : folderInput;
      const trimmed = raw.trim();
      if (!trimmed || folderLoading) return;
      setFolderError(null);
      setFolderLoading(true);
      // The typed-input panel is closed here, so surface failures via toast.
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

  // Managed downloads pull only changed blobs, so the cached copy stays usable.
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
            // A sibling download blocks this update; say so instead of leaving the copy stale.
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

  // Case-insensitive (the HF cache lowercases ids); complete downloads only, since this set means
  // "can load now".
  const downloadedSet = useMemo(
    () =>
      new Set(
        [...cachedGguf, ...cachedModels]
          .filter((c) => !c.partial)
          .map((c) => c.repo_id.toLowerCase()),
      ),
    [cachedGguf, cachedModels],
  );
  // The row, its unsloth mirror and the vendor repo share files, so match all of them.
  const aliasesOf = useCallback(
    (id: string): string[] => {
      const artifact = catalog && artifactForRepoId(id, catalog)?.artifact;
      return artifact?.upstreamRepoId
        ? [id, artifact.repoId, artifact.upstreamRepoId]
        : [id];
    },
    [catalog],
  );
  const isValueRow = useCallback(
    (id: string) =>
      value === id ||
      aliasesOf(id).some((alias) => modelIdsMatchForPicker(value, alias)),
    [value, aliasesOf],
  );
  const cachedIdFor = useCallback(
    (id: string): string | null =>
      aliasesOf(id).find((alias) => downloadedSet.has(alias.toLowerCase())) ??
      null,
    [aliasesOf, downloadedSet],
  );

  // Torn snapshots, so Hub rows can mark partials. An id with any complete row (another format) is
  // left out, since it loads.
  const partialSet = useMemo(
    () =>
      partialSetFromRows([...cachedGguf, ...cachedModels], (c) => c.repo_id),
    [cachedGguf, cachedModels],
  );
  // A complete alias loads instead, so the row is downloaded, not partial.
  const isPartialRow = useCallback(
    (id: string) =>
      cachedIdFor(id) === null && partialSet.has(id.toLowerCase()),
    [cachedIdFor, partialSet],
  );

  // An id partialSet dropped never draws the mark, so spare entries are harmless.
  const partialResumableSet = useMemo(
    () =>
      new Set(
        [...cachedGguf, ...cachedModels]
          .filter((c) => c.partial === true && c.partial_resumable === true)
          .map((c) => c.repo_id.toLowerCase()),
      ),
    [cachedGguf, cachedModels],
  );

  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const deviceType = usePlatformStore((s) => s.deviceType);
  const isMac = deviceType === "mac";
  const hostClass = useHostClass();
  const denseQuantSchemes = useDenseQuantSchemes();

  // A task-scoped picker wants exactly the tasks the chat classifier calls unsupported.
  const isChatSupported = useCallback(
    (r: HfModelResult) => {
      if (task)
        return (
          taskMatchesFilter(r.pipelineTag, task) && !isImageEditModel(r.id)
        );
      return (
        classifyUnslothSupport({
          modelId: r.id,
          pipelineTag: r.pipelineTag,
          tags: r.tags,
          libraryName: r.libraryName,
          quantMethod: r.quantMethod,
          deviceType,
        }).status !== "unsupported"
      );
    },
    [deviceType, task],
  );

  const isTaskRuntimeSupported = useCallback(
    (result: HfModelResult) => {
      const isStt = Boolean(
        task && taskMatchesFilter("automatic-speech-recognition", task),
      );
      const isTts = Boolean(task && taskMatchesFilter("text-to-speech", task));
      return (
        communityAudioRowIsRunnable({
          isStt,
          isTts,
          isGguf: result.isGguf,
          id: result.id,
          baseModel: result.baseModel,
          tags: result.tags,
          libraryName: result.libraryName,
        }) &&
        macTtsHubRowIsRunnable({
          isMac,
          isTts,
          isGguf: result.isGguf,
          hasRunnableGgufSibling: Boolean(
            catalog &&
              groupForRepoId(result.id, catalog)?.artifacts.some(
                (artifact) => artifact.format === "gguf",
              ),
          ),
        })
      );
    },
    [catalog, isMac, task],
  );

  const taskCatalogSeedIds = useMemo(
    () =>
      task
        ? new Set(models.map((model) => model.id.trim().toLowerCase()))
        : undefined,
    [models, task],
  );

  const recommendedIds = useMemo(() => {
    const all = dedupe([...models.map((model) => model.id), value ?? ""])
      .filter(
        (id) =>
          !isHiddenModelId(id) ||
          allowedHiddenModelIdMatches(taskCatalogSeedIds, id),
      )
      .filter((id) => !downloadedSet.has(id.toLowerCase()))
      // Task pages load single-file GGUF only; curated artifacts stay listed in any format.
      .filter((id) =>
        task
          ? isKnownGgufRepo(id) ||
            Boolean(catalog && artifactForRepoId(id, catalog))
          : !chatOnly || isRecommendableFormat(id, isKnownGgufRepo(id), isMac),
      )
      // Hiding member repos emptied task-scoped pickers, whose `models` are exactly group members.
      .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id));
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
    taskCatalogSeedIds,
  ]);

  const showHfSection = debouncedQuery.trim().length > 0;

  const [downloadedSort, setDownloadedSort] = useState<LocalSortKey>("recent");
  const [customSort, setCustomSort] = useState<LocalSortKey>("recent");
  const [chosenFormatFilter, setFormatFilter] = useState<FormatFilter>("all");
  const npuCatalog = useNpuCatalog(npu);
  const formatFilter: FormatFilter =
    chosenFormatFilter === "npu" && !npuCatalog ? "all" : chosenFormatFilter;
  const [npuOnDeviceCollapsed, setNpuOnDeviceCollapsed] = useState(false);
  const [npuBrowseCollapsed, setNpuBrowseCollapsed] = useState(true);
  // Chat passes no task filter and keeps the full set.
  const capabilityScope = useMemo<readonly (keyof ModelCapabilities)[] | null>(() => {
    const tasks: readonly string[] = task
      ? typeof task === "string"
        ? [task]
        : task
      : [];
    if (tasks.length === 0) return null;
    // Video keeps audio: a soundtrack separates two video models.
    if (VIDEO_GEN_TASKS.some((t) => tasks.includes(t))) return ["audio"];
    if (IMAGE_GEN_TASKS.some((t) => tasks.includes(t))) return [];
    if (AUDIO_GEN_TASKS.some((t) => tasks.includes(t))) return [];
    return null;
  }, [task]);

  const hubRowsShowSize =
    formatFilter === "mlx" || formatFilter === "safetensors";

  const curatedRow = useCallback(
    (id: string) =>
      (catalog && curatedRowLabelFor(id, catalog, hostClass, denseQuantSchemes)) ?? {
        name: id,
        tags: [] as string[],
      },
    [catalog, hostClass, denseQuantSchemes],
  );

  /** Whether the host can run a curated id at all, not whether it has room. Browse rows only. */
  const curatedOfferable = useCallback(
    (id: string) => {
      if (!catalog) return true;
      // Downloaded weights keep their row.
      if (downloadedSet.has(id.toLowerCase())) return true;
      const hit = artifactForRepoId(id, catalog);
      return hit ? curatedArtifactIsOfferable(hit.artifact.repoId, hostClass) : true;
    },
    [catalog, downloadedSet, hostClass],
  );

  // Paint curated rows before any request so task pickers do not sit on a spinner.
  const catalogSeedRows = useMemo<HfModelResult[]>(() => {
    if (!task) return [];
    return dedupe(models.map((model) => model.id))
      .filter((id) => !isMobileVariant(id))
      .filter((id) => !isImageEditModel(id))
      .filter(curatedOfferable)
      .filter((id) => {
        const isG = isKnownGgufRepo(id);
        return taskCatalogFormatMatches(
          formatFilter,
          matchesFormatFilter(id, isG, formatFilter),
        );
      })
      .map((id) => ({
        id,
        downloads: 0,
        likes: 0,
        isGguf: isKnownGgufRepo(id),
        // Size from the catalog, not an id "<n>B" guess (Wan2.2-TI2V-5B is 30 GB).
        curatedSizeBytes: catalog ? curatedSizeBytesFor(id, catalog) : undefined,
        totalParams: catalog ? curatedTotalParamsFor(id, catalog) : undefined,
      }));
  }, [catalog, models, formatFilter, isKnownGgufRepo, task, curatedOfferable]);

  /** Every list judging a row against the device goes through this, so badges and filters agree. */
  const catalogFit = useCallback(
    (id: string, budget: DeviceBudget) =>
      catalog ? curatedArtifactFit(id, catalog, budget) : undefined,
    [catalog],
  );

  const isUnslothOwned = useCallback(
    (id: string) => id.toLowerCase().startsWith("unsloth/"),
    [],
  );

  /** Pipeline tag is the Hub's only signal before download; other serializations dropped by name. */
  const isLoadableCommunityRepo = useCallback(
    (id: string) =>
      !/(^|[-_/.])(onnx|openvino|tflite|coreml)([-_./]|$)/i.test(id),
    [],
  );

  const catalogSeedIds = useMemo(
    () => catalogSeedRows.map((row) => row.id),
    [catalogSeedRows],
  );

  // Recommended suggests GGUF anywhere, plus MLX and safetensors on Mac.
  const hubRowAllowed = (r: HfModelResult) =>
    !rowFilter || rowFilter({ id: r.id, task: r.pipelineTag });
  const recommendedRows = useMemo(() => {
    const catalogSeedIds = new Set(
      catalogSeedRows.map((row) => row.id.toLowerCase()),
    );
    const keepCommon = (r: HfModelResult) => {
      const isCatalogSeed = catalogSeedIds.has(r.id.toLowerCase());
      return (
        !isMobileVariant(r.id) &&
        hubRowAllowed(r) &&
        taskPickerRowMatches({
          isCatalogSeed,
          isHidden: isHiddenModelId(r.id),
          format: formatFilter,
          matchesFormat: matchesFormatFilter(r.id, r.isGguf, formatFilter),
          matchesTask: isChatSupported(r),
          isRecommendable: isRecommendableFormat(r.id, r.isGguf, isMac),
        })
      );
    };
    const keep = (r: HfModelResult) =>
      keepCommon(r) &&
      curatedOfferable(r.id) &&
      (!task ||
        r.isGguf ||
        Boolean(catalog && artifactForRepoId(r.id, catalog)));
    // Community rows have no catalog artifact, so the curated clause would drop them all.
    const keepCommunity = (r: HfModelResult) =>
      keepCommon(r) && isTaskRuntimeSupported(r);
    // Members are not filtered here (see recommendedIds): that dropped them from Hub search too.
    const deviceFiltered = fitOnDeviceOnly;
    const taskScoped = Boolean(task);
    const rowGpu = loadScopedGpu(gpu, taskScoped);
    const rowInferenceGpu = loadScopedGpu(inferenceGpu, taskScoped);
    const pipelineBudget = artifactBudget(rowGpu);
    const fits = (r: HfModelResult) =>
      cachedIdFor(r.id) !== null ||
      // The catalog's verdict where it has one: hfModelFitsDevice counts RAM a card-only load never uses.
      (catalogFit(r.id, pipelineBudget)?.fits ??
        hfModelFitsDevice(r, diffusionLoad || !r.isGguf ? rowGpu : rowInferenceGpu, {
          budgetFraction,
          // Not `&& r.isGguf`: on a task page safetensors rows use the same backend budget.
          mediaLoad: diffusionLoad,
          hostPooledMemory: gpu.loadDeviceSharesHostMemory,
          gpuCount: rowInferenceGpu.deviceCount,
        }));
    const unslothRows = orderRecommendedRows({
      seeds: catalogSeedRows,
      results: recommendedSearch.results,
      keep,
      deviceFiltered,
      fits,
      familyOf: catalog
        ? (id) => groupForRepoId(id, catalog)?.canonicalId.toLowerCase()
        : undefined,
      pinnedFamilies: catalog
        ?.filter((g) => g.pinToTop)
        .map((g) => g.canonicalId.toLowerCase()),
    });
    if (!communityRecommendedEnabled) return unslothRows;
    // Community rows append below everything unsloth publishes, with the same gates.
    const above = new Set(unslothRows.map((r) => r.id.toLowerCase()));
    // A vendor repo whose unsloth mirror is listed above is the same files.
    for (const r of unslothRows) {
      const upstream =
        catalog && artifactForRepoId(r.id, catalog)?.artifact.upstreamRepoId;
      if (upstream) above.add(upstream.toLowerCase());
    }
    const communityRows = communityBrowse.results
      .filter((r) => !r.id.toLowerCase().startsWith("unsloth/"))
      .filter((r) => isLoadableCommunityRepo(r.id))
      .filter((r) => !above.has(r.id.toLowerCase()))
      .filter(keepCommunity)
      .filter((r) => !deviceFiltered || fits(r));
    return [...unslothRows, ...communityRows];
  }, [
    rowFilter,
    budgetFraction,
    diffusionLoad,
    recommendedSearch.results,
    catalogSeedRows,
    cachedIdFor,
    fitOnDeviceOnly,
    formatFilter,
    isMac,
    isTaskRuntimeSupported,
    gpu,
    inferenceGpu,
    isChatSupported,
    task,
    catalog,
    catalogFit,
    curatedOfferable,
    communityRecommendedEnabled,
    communityBrowse.results,
    isLoadableCommunityRepo,
  ]);

  // Listing metadata wins; curated seeds fill rows the listing never returns.
  const recommendedMeta = useMemo(() => {
    const map = new Map<
      string,
      {
        meta: string | null;
        status: GgufFitClass | VramFitStatus | null;
        est: number;
        budget?: CuratedBudget;
      }
    >();
    /** Returns the verdict, not a boolean, so `marginal`/`partial` still badge. */
    const ggufRowFit = (
      sizeBytes: number | undefined,
      budget: typeof inferenceGpu,
    ): GgufFitClass | VramFitStatus | null => {
      const anyBudget =
        budget.memoryTotalGb > 0 || budget.systemRamAvailableGb > 0;
      if (!budget.budgetKnown && !anyBudget) return null;
      if (sizeBytes == null) return null;
      // Probed and genuinely zero (e.g. Vulkan reporting nothing) means nothing fits.
      if (!anyBudget) return "oom";
      // Same diffusion rule as its quant rows, so parent and children agree.
      const fit = diffusionLoad
        ? classifyMediaGgufFit(
            sizeBytes,
            budget.memoryTotalGb,
            mediaRamBudgetGb(
              budget.systemRamAvailableGb,
              gpu.loadDeviceSharesHostMemory,
            ),
          )
        : classifyGgufFit(sizeBytes, {
            gpuGb: budget.memoryTotalGb,
            systemRamGb: budget.systemRamAvailableGb,
            budgetFraction,
            gpuCount: budget.deviceCount,
          });
      if (fit === "fits") return null;
      return diffusionRefuses(fit, diffusionLoad, gpu.loadDeviceSharesHostMemory)
        ? "exceeds"
        : fit;
    };
    // A task load puts the whole pipeline on one device; inferenceGpu is the GGUF backend's inventory.
    const rowGpu = loadScopedGpu(gpu, Boolean(task));
    const pipelineBudget = artifactBudget(rowGpu);
    // Use the inventory of the runtime that places the row: Vulkan may see cards torch cannot.
    const rowInferenceGpu = diffusionLoad
      ? rowGpu
      : loadScopedGpu(inferenceGpu, Boolean(task));
    // Fold in community rows, or they render with no size or VRAM chip.
    for (const r of [
      ...recommendedSearch.results,
      ...catalogSeedRows,
      ...communityBrowse.results,
    ]) {
      if (map.has(r.id)) continue;
      const isG = isKnownGgufRepo(r.id);
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
            isMlxId(r.id) ? "MLX" : "Safetensors",
            r.estimatedSizeBytes ? formatBytes(r.estimatedSizeBytes) : null,
          ]
            .filter(Boolean)
            .join(" · ") || null;
      if (isG) {
        // Repos we cannot size show no badge.
        const params = ggufParams;
        const sizeBytes =
          r.estimatedSizeBytes ??
          (params ? estimateQuantBytes(params) : undefined);
        map.set(r.id, {
          meta,
          // The classifier's own verdict, scoped to the device the load lands on.
          status: ggufRowFit(sizeBytes, rowInferenceGpu),
          // Show the figure the verdict used (weights + activations + KV); the media rule uses raw size.
          est: sizeBytes
            ? Math.round(
                diffusionLoad
                  ? sizeBytes / 1024 ** 3
                  : requiredGgufMemoryGb(sizeBytes),
              )
            : 0,
        });
        continue;
      }
      const curatedFit = catalogFit(r.id, pipelineBudget);
      if (curatedFit !== undefined) {
        map.set(r.id, {
          meta,
          status: curatedFit.fits ? null : "exceeds",
          est: curatedFit.sizeGb ? Math.round(curatedFit.sizeGb) : 0,
          budget: curatedBudget(curatedFit),
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
    budgetFraction,
    diffusionLoad,
    recommendedSearch.results,
    communityBrowse.results,
    catalogSeedRows,
    catalog,
    catalogFit,
    task,
    isKnownGgufRepo,
    gpu,
    inferenceGpu,
  ]);

  // Handed to the page on pick so a task page can classify an uncurated repo.
  const pipelineTagById = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of [
      ...results,
      ...recommendedSearch.results,
      ...communityQuerySearch.results,
      ...communityBrowse.results,
    ]) {
      if (r.pipelineTag && !map.has(r.id)) map.set(r.id, r.pipelineTag);
    }
    return map;
  }, [
    results,
    recommendedSearch.results,
    communityQuerySearch.results,
    communityBrowse.results,
  ]);

  // Same Hub evidence the Audio page judges rows on, so chat routing matches its listing.
  const hubEvidenceById = useMemo(() => {
    const map = new Map<
      string,
      {
        baseModel?: string | null;
        tags?: string[];
        libraryName?: string | null;
        audioType?: string | null;
        taskFromGgufArch?: boolean;
      }
    >();
    for (const r of [
      ...results,
      ...recommendedSearch.results,
      ...communityQuerySearch.results,
      ...communityBrowse.results,
    ]) {
      if (map.has(r.id)) continue;
      map.set(r.id, {
        baseModel: r.baseModel,
        tags: r.tags,
        libraryName: r.libraryName,
      });
    }
    // The backend tags cached Whisper checkpoints as ASR even when the name says nothing.
    for (const c of cachedModels) {
      const existing = map.get(c.repo_id);
      if (existing) {
        map.set(c.repo_id, {
          ...existing,
          audioType: existing.audioType ?? c.audio_type,
        });
        continue;
      }
      map.set(c.repo_id, {
        baseModel: null,
        tags: c.tags,
        libraryName: c.library_name,
        audioType: c.audio_type,
      });
    }
    for (const c of cachedGguf) {
      // Only the audio runtime's header classifier tags a GGUF text-to-audio, so it is runnable.
      const taskFromGgufArch = c.task === "text-to-audio" ? true : undefined;
      const existing = map.get(c.repo_id);
      if (existing) {
        map.set(c.repo_id, {
          ...existing,
          audioType: existing.audioType ?? c.audio_type,
          taskFromGgufArch,
        });
        continue;
      }
      map.set(c.repo_id, { audioType: c.audio_type, taskFromGgufArch });
    }
    return map;
  }, [
    results,
    recommendedSearch.results,
    communityQuerySearch.results,
    communityBrowse.results,
    cachedModels,
    cachedGguf,
  ]);

  // Listings first: real tags outrank curated data.
  const capsById = useMemo(() => {
    const map = new Map<string, ModelCapabilities>();
    for (const r of [
      ...results,
      ...recommendedSearch.results,
      ...communityQuerySearch.results,
      ...communityBrowse.results,
    ]) {
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
        const curated = curatedCapabilitiesFor(row.id, catalog);
        if (!curated) continue;
        const detected = map.get(row.id);
        // Merged, not first-wins: curated entries state things no tag mentions (H3's audio).
        map.set(
          row.id,
          detected
            ? {
                vision: detected.vision || curated.vision,
                reasoning: detected.reasoning || curated.reasoning,
                audio: detected.audio || curated.audio,
                imageGen: detected.imageGen || curated.imageGen,
                videoGen: detected.videoGen || curated.videoGen,
              }
            : curated,
        );
      }
    }
    return map;
  }, [
    results,
    recommendedSearch.results,
    communityQuerySearch.results,
    communityBrowse.results,
    catalog,
    catalogSeedRows,
  ]);

  // Supported diffusion GGUFs stay listed so a pick routes to Images or Video.
  const sortedCachedGguf = useMemo(
    () =>
      sortCachedRepos(
        cachedGguf.filter(
          (c) =>
            passesTaskGate(
              c.task,
              c.repo_id,
              task,
              catalog,
              activeCatalogArtifactIds,
            ) &&
            // A CSM speech GGUF would otherwise list as chat and fail only in llama-server.
            audioPickIsRoutable({
              id: c.repo_id,
              task: c.task,
              audioType: c.audio_type,
              isGguf: true,
              isCurated: artifactForRepoId(c.repo_id, AUDIO_CATALOG) !== null,
              // Codec provenance separates runnable Orpheus from unsupported CSM.
              taskFromGgufArch: true,
            }) &&
            (!rowFilter ||
              rowFilter({
                id: c.repo_id,
                task: c.task,
                audioType: c.audio_type,
                audioWorkflows: c.audio_workflows,
              })),
        ),
        downloadedSort,
        loadTimes,
      ),
    [
      cachedGguf,
      downloadedSort,
      loadTimes,
      task,
      catalog,
      activeCatalogArtifactIds,
      rowFilter,
    ],
  );
  const sortedCachedModels = useMemo(
    () =>
      sortCachedRepos(
        cachedModels.filter(
          (c) =>
            // Partial snapshots are listed (so they can be deleted) but select with isDownloaded: false.
            passesTaskGate(
              c.task,
              c.repo_id,
              task,
              catalog,
              activeCatalogArtifactIds,
              c,
            ) &&
            // Gate on a curated artifact, not a group-key match (base siblings fail the trust gate); unsloth
            // repos must be full pipelines, since from_pretrained fails on single-file repos.
            (!task ||
              (isUnslothRepoId(c.repo_id) && !c.single_file) ||
              ((c.task === "automatic-speech-recognition" ||
                c.task === "text-to-speech") &&
                communityAudioRowIsRunnable({
                  isStt: c.task === "automatic-speech-recognition",
                  isTts: c.task === "text-to-speech",
                  isGguf: false,
                  id: c.repo_id,
                  tags: c.tags,
                  libraryName: c.library_name,
                  audioType: c.audio_type,
                }) &&
                macTtsHubRowIsRunnable({
                  isMac,
                  isTts: c.task === "text-to-speech",
                  isGguf: false,
                  hasRunnableGgufSibling: Boolean(
                    catalog &&
                      groupForRepoId(c.repo_id, catalog)?.artifacts.some(
                        (artifact) => artifact.format === "gguf",
                      ),
                  ),
                  audioType: c.audio_type,
                })) ||
              (catalog
                ? artifactForRepoId(c.repo_id, catalog) !== null
                : false) ||
              // A pinned snapshot admitted only by an explicit family.
              (c.opaque === true && Boolean(c.load_id?.trim()) && c.load_id?.trim() !== c.repo_id.trim())) &&
            (!rowFilter ||
              rowFilter({
                id: c.repo_id,
                task: c.task,
                audioType: c.audio_type,
                audioWorkflows: c.audio_workflows,
              })),
        ),
        downloadedSort,
        loadTimes,
      ),
    [
      cachedModels,
      downloadedSort,
      loadTimes,
      task,
      catalog,
      activeCatalogArtifactIds,
      isMac,
      rowFilter,
    ],
  );
  // Task loads land on one device (lowest visible ordinal), so size against it; chat keeps the sum.
  const expanderGpuGbFrom = (info: typeof inferenceGpu) =>
    info.available
      ? loadScopedGpu(info, Boolean(task)).memoryTotalGb
      : undefined;
  // Images / Video place through torch even on a Vulkan llama.cpp build.
  const expanderBudgetGpu = diffusionLoad ? gpu : inferenceGpu;
  const expanderGpuGb = expanderGpuGbFrom(expanderBudgetGpu);
  const expanderSystemGpuGb = expanderGpuGbFrom(gpu);
  // Same scoping as the capacity, so the per-card reserve is not charged host-wide.
  const expanderScopedGpu = loadScopedGpu(expanderBudgetGpu, Boolean(task));
  const expanderGpuCount = expanderScopedGpu.deviceCount;
  const expanderRamGb = expanderScopedGpu.systemRamAvailableGb;

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
            filesystemRowsSupportedForTask(task, m.task) &&
            // A local CSM file is just as undecodable; routing it to Audio evicts the chat model first.
            audioPickIsRoutable({
              id: m.model_id ?? m.id,
              task: m.task,
              audioType: m.audio_type,
              isGguf: localModelIsGguf(m),
              isCurated: artifactForRepoId(m.model_id ?? m.id, AUDIO_CATALOG) !== null,
              // From the filesystem classifier, so a renamed CSM file cannot pass as Orpheus.
              taskFromGgufArch: true,
            }) &&
            // On Images/Video a chat GGUF must not be offered.
            passesTaskGate(
              m.task,
              m.model_id ?? m.id,
              task,
              catalog,
              activeCatalogArtifactIds,
              m,
            ) &&
            localModelMatchesFormat(m, formatFilter) &&
            matchesLocalQuery(m) &&
            // A task page's row filter applies to local and LM Studio rows too.
            (!rowFilter ||
              rowFilter({
                id: m.model_id ?? m.id,
                task: m.task,
                audioType: m.audio_type,
                audioWorkflows: m.audio_workflows,
              })),
        ),
        downloadedSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      lmStudioModels,
      downloadedSort,
      formatFilter,
      loadTimes,
      localQuery,
      task,
      catalog,
      activeCatalogArtifactIds,
      rowFilter,
    ],
  );
  // Chat-only hides raw checkpoints; task pickers are exempt since the image backend loads local pipelines.
  const sortedLocalDir = useMemo(
    () =>
      sortLocalModels(
        localDirModels.filter(
          (m) =>
            filesystemRowsSupportedForTask(task, m.task) &&
            // Same speech gate as cached GGUF rows.
            audioPickIsRoutable({
              id: m.model_id ?? m.id,
              task: m.task,
              audioType: m.audio_type,
              isGguf: localModelIsGguf(m),
              isCurated: artifactForRepoId(m.model_id ?? m.id, AUDIO_CATALOG) !== null,
              // Filesystem classifier task, so a renamed CSM file cannot pass as Orpheus.
              taskFromGgufArch: true,
            }) &&
            passesTaskGate(
              m.task,
              m.model_id ?? m.id,
              task,
              catalog,
              activeCatalogArtifactIds,
              m,
            ) &&
            (!chatOnly ||
              Boolean(task) ||
              localModelIsGguf(m) ||
              (isMac && localModelIsMlx(m))) &&
            localModelMatchesFormat(m, formatFilter) &&
            matchesLocalQuery(m) &&
            (!rowFilter ||
              rowFilter({
                id: m.model_id ?? m.id,
                task: m.task,
                audioType: m.audio_type,
                audioWorkflows: m.audio_workflows,
              })),
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
      catalog,
      activeCatalogArtifactIds,
      rowFilter,
    ],
  );
  const sortedCustomFolderModels = useMemo(
    () =>
      sortLocalModels(
        customFolderModels.filter(
          (m) =>
            filesystemRowsSupportedForTask(task, m.task) &&
            // Same speech gate as cached GGUF rows.
            audioPickIsRoutable({
              id: m.model_id ?? m.id,
              task: m.task,
              audioType: m.audio_type,
              isGguf: localModelIsGguf(m),
              isCurated: artifactForRepoId(m.model_id ?? m.id, AUDIO_CATALOG) !== null,
              taskFromGgufArch: true,
            }) &&
            passesTaskGate(
              m.task,
              m.model_id ?? m.id,
              task,
              catalog,
              activeCatalogArtifactIds,
              m,
            ) &&
            localModelMatchesFormat(m, formatFilter) &&
            matchesLocalQuery(m) &&
            (!rowFilter ||
              rowFilter({
                id: m.model_id ?? m.id,
                task: m.task,
                audioType: m.audio_type,
                audioWorkflows: m.audio_workflows,
              })),
        ),
        customSort,
        loadTimes,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      customFolderModels,
      customSort,
      formatFilter,
      loadTimes,
      localQuery,
      task,
      catalog,
      activeCatalogArtifactIds,
      rowFilter,
    ],
  );

  // Chat cannot load diffusion models, so the pick routes to the page that runs it.
  const navigateToPage = useNavigate();
  const diffusionTaskById = useMemo(() => {
    const byId = new Map<string, string>();
    const put = (
      id: string | null | undefined,
      t: string | null | undefined,
      exactAudioArtifact = id ? artifactForRepoId(id, AUDIO_CATALOG) : null,
    ) => {
      const task = curatedAudioInventoryTask({
        inventoryTask: t,
        isExactCatalogArtifact: Boolean(exactAudioArtifact),
        catalogScope: exactAudioArtifact?.group.scope,
        catalogTask: exactAudioArtifact?.group.task,
      });
      if (id && task) byId.set(id.toLowerCase(), task);
    };
    for (const c of cachedGguf) put(c.repo_id, c.task);
    for (const c of cachedModels) put(c.repo_id, c.task);
    // Key on both: m.id is a path while m.model_id is its HF-style name.
    const putLocal = (m: LocalModelInfo) => {
      const exactAudioArtifact = m.model_id
        ? artifactForRepoId(m.model_id, AUDIO_CATALOG)
        : null;
      put(m.id, m.task, exactAudioArtifact);
      put(m.model_id, m.task, exactAudioArtifact);
    };
    for (const m of lmStudioModels) putLocal(m);
    for (const m of localDirModels) putLocal(m);
    for (const m of customFolderModels) putLocal(m);
    return byId;
  }, [
    cachedGguf,
    cachedModels,
    lmStudioModels,
    localDirModels,
    customFolderModels,
  ]);

  const onSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      if (!task) {
        const pickedTask = taskForMediaPick(
          meta.pipelineTag,
          diffusionTaskById.get(id.toLowerCase()),
        );
        const page = mediaPageForTask(pickedTask);
        if (
          page === "audio" &&
          !audioPickIsRoutable({
            id,
            task: pickedTask,
            isGguf: Boolean(meta.isGguf || meta.ggufFilename),
            isCurated: artifactForRepoId(id, AUDIO_CATALOG) !== null,
            audioType: meta.audioType,
            isLocalCheckpoint:
              meta.source === "lora" ||
              meta.source === "exported" ||
              meta.source === "local",
            ...(hubEvidenceById.get(id) ?? {}),
          })
        ) {
          // Loading here would evict the chat model for a repo neither surface can run.
          toast.error(
            `${id} is not an audio model Unsloth can run yet. The Audio page lists the families it supports.`,
            { duration: 7000 },
          );
          return;
        }
        if (page) {
          void navigateToPage({
            to: `/${page}`,
            // pickedTask, not meta.pipelineTag: a cached row carries no tag to forward.
            search:
              page === "audio"
                ? audioPickSearch(id, { ...meta, task: pickedTask })
                : diffusionRouteSearch(id, meta),
          });
          return;
        }
      }
      onSelectProp(id, meta);
    },
    [task, diffusionTaskById, hubEvidenceById, navigateToPage, onSelectProp],
  );

  const fineTunedRows = useMemo(() => {
    const needle = normalizeForSearch(debouncedQuery.trim());
    return loraModels
      .filter(
        (m) =>
          // A CSM GGUF export loads nowhere; audioType comes off the checkpoint, so renames are caught.
          !localAudioRowIsUndecodableGguf({
            audioType: m.audioType,
            exportType: m.exportType,
            isDirectGguf: m.isDirectGguf,
          }),
      )
      .filter((m) =>
        nativeAudioCheckpointIsLoadable(m.audioType, m.exportType),
      )
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

  const visibleCachedGguf = useMemo(() => {
    if (!showHfSection)
      return sortedCachedGguf.filter((c) =>
        matchesFormatFilter(c.repo_id, true, formatFilter),
      );
    const q = normalizeForSearch(debouncedQuery.trim());
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

  // Empty-state logic must use this or the picker can go blank.
  const visibleCachedModelRows = chatOnly && !task ? [] : visibleCachedModels;

  const visibleAdditionalOnDeviceModels = useMemo(() => {
    const alreadyListed = new Set(
      [...visibleCachedGguf, ...visibleCachedModels].map((model) =>
        model.repo_id.trim().toLowerCase(),
      ),
    );
    const seen = new Set<string>();
    const needle = normalizeForSearch(debouncedQuery.trim());
    const filtered = additionalOnDeviceModels.filter((model) => {
      const id = model.id.trim().toLowerCase();
      if (!id || alreadyListed.has(id) || seen.has(id)) return false;
      seen.add(id);
      return (
        matchesFormatFilter(model.id, model.isGguf === true, formatFilter) &&
        (!needle ||
          normalizeForSearch(
            `${model.id} ${model.name} ${model.description ?? ""} ${model.descriptionSuffix ?? ""}`,
          ).includes(needle))
      );
    });
    return sortCachedRepos(
      filtered.map((model) => ({
        repo_id: model.id,
        size_bytes: model.deviceSizeBytes ?? 0,
        model,
      })),
      downloadedSort,
      loadTimes,
    ).map(({ model }) => model);
  }, [
    additionalOnDeviceModels,
    visibleCachedGguf,
    visibleCachedModels,
    debouncedQuery,
    downloadedSort,
    formatFilter,
    loadTimes,
  ]);
  const npuModels = npuCatalog?.models ?? null;
  const npuListed =
    npuCatalog !== null && (formatFilter === "all" || formatFilter === "npu");
  const npuOnDeviceRows = useMemo(
    () =>
      npuListed
        ? npuRowsFor(npuModels, { onDevice: true, query: debouncedQuery })
        : [],
    [npuListed, npuModels, debouncedQuery],
  );
  const npuBrowseRows = useMemo(
    () =>
      npuListed
        ? npuRowsFor(npuModels, { onDevice: false, query: debouncedQuery })
        : [],
    [npuListed, npuModels, debouncedQuery],
  );
  const unslothAdditionalOnDeviceModels = useMemo(
    () =>
      visibleAdditionalOnDeviceModels.filter((model) =>
        isUnslothPublisherRepoId(model.id),
      ),
    [visibleAdditionalOnDeviceModels],
  );
  const otherAdditionalOnDeviceModels = useMemo(
    () =>
      visibleAdditionalOnDeviceModels.filter(
        (model) => !isUnslothPublisherRepoId(model.id),
      ),
    [visibleAdditionalOnDeviceModels],
  );

  // Unfiltered list, so typing a query does not re-run resolution.
  const soleQuants = useSoleDownloadedQuants(sortedCachedGguf, {
    enabled: section === "downloaded" && !showAllQuantizations,
    hfToken: hfToken || undefined,
  });

  // GGUF quants pin individually (repo still listed below); non-GGUF repos pin whole.
  const pinnedIds = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const unpinRepo = usePinnedModelsStore((s) => s.unpinRepo);
  const pinnedSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);

  // Separate store: `external::` ids contain the "::" the On Device store uses as a separator.
  const pinnedConnectedIds = usePinnedConnectedModelsStore((s) => s.pinned);
  const togglePinnedConnected = usePinnedConnectedModelsStore(
    (s) => s.togglePinnedConnected,
  );
  const pinnedConnectedSet = useMemo(
    () => new Set(pinnedConnectedIds),
    [pinnedConnectedIds],
  );
  // Base URL matters: a Gemini connection behind an OpenAI-compatible proxy returns no inline images.
  const externalProviders = useExternalProvidersStore((s) => s.providers);
  const externalBaseUrlById = useMemo(
    () =>
      new Map(
        externalProviders.map((provider) => [provider.id, provider.baseUrl]),
      ),
    [externalProviders],
  );
  const externalApiTypeById = useMemo(
    () =>
      new Map(
        externalProviders.map((provider) => [provider.id, provider.apiType]),
      ),
    [externalProviders],
  );
  // The per-model editor must offer the bounds every request is clamped to.
  const externalMaxOutputById = useMemo(
    () =>
      new Map(
        externalProviders.map((provider) => [
          provider.id,
          provider.maxOutputTokens ?? null,
        ]),
      ),
    [externalProviders],
  );
  // Self-hosted endpoints publish no reasoning signal, so the connection flag is the only source.
  const externalReasoningFlagById = useMemo(
    () =>
      new Map(
        externalProviders.map((provider) => [
          provider.id,
          provider.isReasoningModel === true,
        ]),
      ),
    [externalProviders],
  );
  const externalReasoningConfigById = useMemo(
    () =>
      new Map(
        externalProviders.map((provider) => [
          provider.id,
          provider.backendProviderType === "custom" && !provider.decisionsOnly
            ? provider.reasoningConfig
            : undefined,
        ]),
      ),
    [externalProviders],
  );
  // A provider catalogue arrives after first paint and decides most of the marks, so re-read it.
  const catalogVersion = useSyncExternalStore(
    subscribeModelCatalog,
    modelCatalogVersion,
  );
  const [connectedSort, setConnectedSort] =
    useState<ConnectedSortKey>("provider");
  const [connectedModality, setConnectedModality] =
    useState<ConnectedModalityFilter>("all");
  const [collapsedConnectedGroups, setCollapsedConnectedGroups] = useState<
    ReadonlySet<string>
  >(() => new Set());
  const [pinnedConnectedCollapsed, setPinnedConnectedCollapsed] =
    useState(false);
  const toggleConnectedGroup = useCallback((providerId: string) => {
    setCollapsedConnectedGroups((prev) => {
      const next = new Set(prev);
      if (!next.delete(providerId)) next.add(providerId);
      return next;
    });
  }, []);
  const [infoModel, setInfoModel] = useState<{
    model: ExternalModelOption;
    providerModelId: string;
    apiType?: ProviderApiType;
    baseUrl: string | null;
    isReasoningProvider: boolean;
    reasoningConfig?: CustomReasoningConfig;
  } | null>(null);
  const [settingsModel, setSettingsModel] = useState<{
    model: ExternalModelOption;
    providerModelId: string;
    apiType?: ProviderApiType;
    baseUrl: string | null;
    isReasoningProvider: boolean;
    reasoningConfig?: CustomReasoningConfig;
    connectionMaxOutputTokens: number | null;
  } | null>(null);

  // Not navigator.clipboard: undefined in the desktop shell and over plain HTTP.
  const copyConnectedModelId = useCallback(async (providerModelId: string) => {
    if (await copyToClipboard(providerModelId)) {
      toast.success(`Copied ${providerModelId}`);
    } else {
      toast.error("Could not copy the model ID");
    }
  }, []);

  // Per-quant validation is needed because deleting one variant can leave a sibling cached.
  const pinnedQuantCandidates = useMemo(() => {
    // Ignore the text query so a pinned quant stays findable by quant name.
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
    /** Verified downloads by pin key; the value is the copy the listing resolved them in. */
    downloaded: ReadonlyMap<string, string | null>;
    visionByRepo: ReadonlyMap<string, boolean>;
    sizes: ReadonlyMap<string, number>;
  }>({
    validated: false,
    downloaded: new Map(),
    visionByRepo: new Map(),
    sizes: new Map(),
  });
  const prunePinnedQuantValidation = useCallback(
    (repoId: string, quant: string) => {
      const key = pinKey(repoId, quant);
      setPinnedQuantValidation((prev) => {
        if (!prev.downloaded.has(key)) return prev;
        const downloaded = new Map(prev.downloaded);
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
            { preferLocalCache: true, includeCacheLocations: !mediaPageForTask(
              sortedCachedGguf.find((row) => row.repo_id === repoId)?.task,
            ) },
          );
          const normalized = normalizeGgufVariantsResponse(response);
          const downloaded = normalized.variants.filter(
            (variant) => variant.downloaded === true,
          );
          return {
            repoId,
            hasVision: normalized.hasVision,
            downloaded: downloaded.map(
              (variant): [string, string | null] => [
                pinKey(repoId, variant.quant),
                variant.cache_ref ?? variant.cache_path ?? null,
              ],
            ),
            sizes: downloaded.map(
              (variant) =>
                [pinKey(repoId, variant.quant), variant.size_bytes] as const,
            ),
          };
        } catch {
          // Hide unverifiable quants rather than claim a missing file is downloaded.
          return { repoId, hasVision: undefined, downloaded: [], sizes: [] };
        }
      }),
    ).then((groups) => {
      if (!cancelled) {
        setPinnedQuantValidation({
          validated: true,
          downloaded: new Map(groups.flatMap((group) => group.downloaded)),
          visionByRepo: new Map(
            groups.flatMap((group) =>
              group.hasVision === undefined
                ? []
                : [[group.repoId, group.hasVision] as const],
            ),
          ),
          sizes: new Map(groups.flatMap((group) => group.sizes)),
        });
      }
    });

    return () => {
      cancelled = true;
    };
  }, [hfToken, pinnedQuantCandidates, sortedCachedGguf]);
  const downloadedPinnedQuantPaths = useMemo<ReadonlyMap<string, string | null>>(
    () =>
      pinnedQuantValidation.validated
        ? pinnedQuantValidation.downloaded
        : new Map(),
    [pinnedQuantValidation],
  );

  const pinnedQuants = useMemo(() => {
    const q = normalizeForSearch(debouncedQuery.trim());
    return pinnedQuantCandidates.filter(
      (entry) =>
        downloadedPinnedQuantPaths.has(pinKey(entry.repoId, entry.quant)) &&
        (!q ||
          normalizeForSearch(`${entry.repoId} ${entry.quant}`).includes(q)),
    );
  }, [debouncedQuery, downloadedPinnedQuantPaths, pinnedQuantCandidates]);

  const pinnedCachedModelRows = useMemo(
    () =>
      visibleCachedModelRows.filter((c) => pinnedSet.has(pinKey(c.repo_id))),
    [visibleCachedModelRows, pinnedSet],
  );

  const pinnedFineTunedRows = useMemo(
    () => (task ? [] : fineTunedRows.filter((m) => pinnedSet.has(pinKey(m.id)))),
    [task, fineTunedRows, pinnedSet],
  );
  const unpinnedFineTunedRows = useMemo(
    () => fineTunedRows.filter((m) => !pinnedSet.has(pinKey(m.id))),
    [fineTunedRows, pinnedSet],
  );

  const pinnedRows = useMemo(() => {
    const rank = makePinRank(pinnedIds);
    const rows = [
      ...pinnedQuants.map((entry) => ({
        key: pinKey(entry.repoId, entry.quant),
        entry,
        model: null,
        fineTuned: null,
      })),
      ...pinnedCachedModelRows.map((model) => ({
        key: pinKey(model.repo_id),
        entry: null,
        model,
        fineTuned: null,
      })),
      ...pinnedFineTunedRows.map((fineTuned) => ({
        key: pinKey(fineTuned.id),
        entry: null,
        model: null,
        fineTuned,
      })),
    ];
    rows.sort((a, b) => rank(a.key) - rank(b.key));
    return rows;
  }, [pinnedIds, pinnedQuants, pinnedCachedModelRows, pinnedFineTunedRows]);

  // A sole quant that is pinned moves to Pinned instead of showing twice.
  const pinnedSoleQuantRows = useMemo(() => {
    const rows = new Map<
      string,
      { repo: (typeof sortedCachedGguf)[number]; sole: SoleDownloadedQuant }
    >();
    for (const entry of pinnedQuants) {
      const sole = soleQuants.quants.get(entry.repoId);
      const repo = sortedCachedGguf.find((c) => c.repo_id === entry.repoId);
      if (sole && repo && sole.variant.quant === entry.quant) {
        rows.set(pinKey(entry.repoId, entry.quant), { repo, sole });
      }
    }
    return rows;
  }, [pinnedQuants, soleQuants.quants, sortedCachedGguf]);
  const pinnedSoleQuantRepoIds = useMemo(
    () => new Set([...pinnedSoleQuantRows.values()].map(({ repo }) => repo.repo_id)),
    [pinnedSoleQuantRows],
  );

  // Kept-loaded models first, via a stable partition.
  const [keptFirstCachedGguf, keptFirstCachedModelRows] = useMemo(() => {
    const first = <T extends { repo_id: string }>(rows: T[]): T[] =>
      loadedIdSet.size === 0
        ? rows
        : [
            ...rows.filter((r) => loadedIdSet.has(r.repo_id.toLowerCase())),
            ...rows.filter((r) => !loadedIdSet.has(r.repo_id.toLowerCase())),
          ];
    return [first(visibleCachedGguf), first(visibleCachedModelRows)] as const;
  }, [visibleCachedGguf, visibleCachedModelRows, loadedIdSet]);
  const unslothCachedGguf = useMemo(
    () =>
      keptFirstCachedGguf.filter(
        (c) =>
          isUnslothPublisherRepoId(c.repo_id) &&
          !pinnedSoleQuantRepoIds.has(c.repo_id),
      ),
    [keptFirstCachedGguf, pinnedSoleQuantRepoIds],
  );
  const otherCachedGguf = useMemo(
    () =>
      keptFirstCachedGguf.filter(
        (c) =>
          !isUnslothPublisherRepoId(c.repo_id) &&
          !pinnedSoleQuantRepoIds.has(c.repo_id),
      ),
    [keptFirstCachedGguf, pinnedSoleQuantRepoIds],
  );
  const unslothCachedModelRows = useMemo(
    () =>
      keptFirstCachedModelRows.filter(
        (c) =>
          isUnslothPublisherRepoId(c.repo_id) &&
          !pinnedSet.has(pinKey(c.repo_id)),
      ),
    [keptFirstCachedModelRows, pinnedSet],
  );
  const otherCachedModelRows = useMemo(
    () =>
      keptFirstCachedModelRows.filter(
        (c) =>
          !isUnslothPublisherRepoId(c.repo_id) &&
          !pinnedSet.has(pinKey(c.repo_id)),
      ),
    [keptFirstCachedModelRows, pinnedSet],
  );

  const recommendedParamCountById = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of [...results, ...recommendedSearch.results]) {
      if (r.totalParams) map.set(r.id, r.totalParams);
    }
    return map;
  }, [results, recommendedSearch.results]);

  // Shared so a curated id one list drops cannot return via the other.
  const searchRowFits = useCallback(
    (row: {
      id: string;
      totalParams?: number;
      estimatedSizeBytes?: number;
      curatedSizeBytes?: number;
    }) =>
      catalogFit(row.id, artifactBudget(loadScopedGpu(gpu, Boolean(task))))?.fits ??
      searchRowFitsDevice(
        {
          ...row,
          // A repo no listing returns must still be sizable or `requireKnown` hides it.
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
          // taskScoped picks the single-device budget; this picks the diffusion rule (Images and Video only).
          diffusionLoad,
          budgetFraction,
          hostPooledMemory: gpu.loadDeviceSharesHostMemory,
        },
      ),
    [
      budgetFraction,
      catalog,
      diffusionLoad,
      catalogFit,
      gpu,
      inferenceGpu,
      isKnownGgufRepo,
      recommendedParamCountById,
      task,
    ],
  );

  const filteredRecommendedIds = useMemo(() => {
    if (!showHfSection) return [];
    const q = normalizeForSearch(debouncedQuery.trim());
    return (
      // Seeds included, or a curated pick vanishes from search once downloaded.
      searchableRecommendedIds(catalogSeedIds, recommendedIds)
        .filter((id) =>
          aliasesOf(id).some((alias) => normalizeForSearch(alias).includes(q)),
        )
        .filter((id) =>
          matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
        )
        // Curated defaults obey the fit toggle too.
        .filter(
          (id) =>
            !fitOnDeviceOnly ||
            cachedIdFor(id) !== null ||
            searchRowFits({ id }),
        )
    );
  }, [
    showHfSection,
    debouncedQuery,
    catalogSeedIds,
    recommendedIds,
    aliasesOf,
    formatFilter,
    isKnownGgufRepo,
    fitOnDeviceOnly,
    cachedIdFor,
    searchRowFits,
  ]);

  // Aliases included so a vendor hit is not shown twice.
  const recommendedSet = useMemo(
    () =>
      new Set(
        filteredRecommendedIds.flatMap(aliasesOf).map((id) => id.toLowerCase()),
      ),
    [filteredRecommendedIds, aliasesOf],
  );

  const searchIdsFrom = useCallback(
    (rows: readonly HfModelResult[], owned: (id: string) => boolean) =>
      rows
        .filter(isChatSupported)
        .filter(isTaskRuntimeSupported)
        .filter((r) => !rowFilter || rowFilter({ id: r.id, task: r.pipelineTag }))
        .filter(
          (r) =>
            !fitOnDeviceOnly ||
            cachedIdFor(r.id) !== null ||
            searchRowFits(r),
        )
        .map((result) => result.id)
        .filter((id) => !isHiddenModelId(id))
        .filter(owned)
        // Otherwise search re-lands curated rows the host would refuse at load.
        .filter(curatedOfferable)
        .filter((id) => !recommendedSet.has(id.toLowerCase()))
        .filter(
          (id) =>
            !chatOnly || isRecommendableFormat(id, isKnownGgufRepo(id), isMac),
        )
        .filter((id) => !/-FP8[-.]|FP8-Dynamic/i.test(id))
        .filter((id) =>
          matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
        ),
    [
      recommendedSet,
      chatOnly,
      isKnownGgufRepo,
      isChatSupported,
      isTaskRuntimeSupported,
      formatFilter,
      fitOnDeviceOnly,
      cachedIdFor,
      searchRowFits,
      isMac,
      curatedOfferable,
      rowFilter,
    ],
  );

  const hfIds = useMemo(() => {
    if (!showHfSection || section !== "recommended") return [];
    return searchIdsFrom(results, isUnslothOwned);
  }, [results, showHfSection, section, searchIdsFrom, isUnslothOwned]);

  const communitySearchIds = useMemo(() => {
    if (!communityDiscoveryEnabled || !showHfSection) return [];
    const above = new Set(hfIds.map((id) => id.toLowerCase()));
    const runnable = new Set(
      communityQuerySearch.results
        .filter(isTaskRuntimeSupported)
        .map((result) => result.id.toLowerCase()),
    );
    return searchIdsFrom(
      communityQuerySearch.results,
      (id) =>
        !isUnslothOwned(id) &&
        isLoadableCommunityRepo(id) &&
        runnable.has(id.toLowerCase()),
    ).filter((id) => !above.has(id.toLowerCase()));
  }, [
    communityDiscoveryEnabled,
    showHfSection,
    communityQuerySearch.results,
    hfIds,
    searchIdsFrom,
    isUnslothOwned,
    isLoadableCommunityRepo,
    isTaskRuntimeSupported,
  ]);

  /** One list so rows, keyboard order and the empty state cannot drift apart. */
  const searchRowIds = useMemo(
    () => [...hfIds, ...communitySearchIds],
    [hfIds, communitySearchIds],
  );

  // A pin moves a row into the Pinned group rather than copying it.
  const connectedMatches = useMemo(() => {
    const needle = normalizeForSearch(debouncedQuery.trim());
    return externalModels.filter((model) => {
      if (
        needle &&
        !normalizeForSearch(
          `${model.name} ${model.providerName} ${model.id}`,
        ).includes(needle)
      ) {
        return false;
      }
      if (connectedModality === "all") return true;
      const marks = connectedModelMarks({
        providerType: model.providerType,
        modelId: parseExternalModelId(model.id)?.modelId ?? model.name,
        baseUrl: externalBaseUrlById.get(model.providerId) ?? null,
        apiType: externalApiTypeById.get(model.providerId),
      });
      return connectedModality === "vision"
        ? marks.vision
        : marks.capabilities[connectedModality];
    });
    // The marks read module state a provider sync writes later; the version is the only staleness dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    externalModels,
    debouncedQuery,
    connectedModality,
    externalBaseUrlById,
    externalApiTypeById,
    catalogVersion,
  ]);

  // Pin order, always: the hand-arranged list is not sorted.
  const pinnedConnectedRows = useMemo(() => {
    const rank = makePinRank(pinnedConnectedIds);
    return connectedMatches
      .filter((model) => pinnedConnectedSet.has(model.id))
      .sort((a, b) => rank(a.id) - rank(b.id));
  }, [connectedMatches, pinnedConnectedIds, pinnedConnectedSet]);

  // Drops go through the stores' drag session so they persist once.
  const pinnedDrag = usePinnedRowDrag({
    scope: "on-device",
    order: () => pinnedRows.map((row) => row.key),
    onDrop: (fromKey, drop) => {
      const store = usePinnedModelsStore.getState();
      const anchor = pinDropAnchor(store.pinned, fromKey, drop.key, drop.edge);
      if (!anchor) return;
      store.beginPinnedDrag();
      store.movePinned(fromKey, anchor);
      store.endPinnedDrag(true);
    },
  });
  const pinnedConnectedDrag = usePinnedRowDrag({
    scope: "connected",
    order: () => pinnedConnectedRows.map((model) => model.id),
    onDrop: (fromId, drop) => {
      const store = usePinnedConnectedModelsStore.getState();
      const anchor = pinDropAnchor(store.pinned, fromId, drop.key, drop.edge);
      if (!anchor) return;
      store.beginPinnedConnectedDrag();
      store.movePinnedConnected(fromId, anchor);
      store.endPinnedConnectedDrag(true);
    },
  });

  const connectedGroups = useMemo(() => {
    const byProvider = new Map<
      string,
      {
        providerId: string;
        providerName: string;
        providerType: string;
        models: ExternalModelOption[];
      }
    >();
    for (const model of connectedMatches) {
      if (pinnedConnectedSet.has(model.id)) continue;
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
    const groups = [...byProvider.values()]
      .map((group) => ({
        ...group,
        models: group.models.sort((a, b) => a.name.localeCompare(b.name)),
      }))
      .sort((a, b) => a.providerName.localeCompare(b.providerName));
    if (connectedSort !== "name") return groups;
    // Name sort across connections is one unlabelled group.
    return [
      {
        providerId: "__all__",
        providerName: "",
        providerType: "",
        models: groups
          .flatMap((group) => group.models)
          .sort((a, b) => a.name.localeCompare(b.name)),
      },
    ];
  }, [connectedMatches, pinnedConnectedSet, connectedSort]);
  const hubOptionKeys = useMemo(() => {
    const keys: string[] = loadedRows.map((m) =>
      makeModelOptionKey("loaded", m.checkpoint ?? m.id),
    );

    // The tab lists nothing else, so these are the whole roving order.
    if (section === "connected") {
      if (!pinnedConnectedCollapsed) {
        keys.push(
          ...pinnedConnectedRows.map((model) =>
            makeModelOptionKey("connected", model.id),
          ),
        );
      }
      for (const group of connectedGroups) {
        if (collapsedConnectedGroups.has(group.providerId)) continue;
        keys.push(
          ...group.models.map((model) =>
            makeModelOptionKey("connected", model.id),
          ),
        );
      }
      return keys;
    }

    // Pinned rows render before the cache scan settles, so their keys do not wait for it.
    if (section === "downloaded" && !pinnedCollapsed && pinnedRows.length > 0) {
      keys.push(
        ...pinnedRows.map((row) =>
          row.entry
            ? pinnedSoleQuantRows.has(row.key)
              ? makeModelOptionKey("downloaded-gguf", row.entry.repoId)
              : makeModelOptionKey("pinned-quant", row.key)
            : row.model
              ? makeModelOptionKey("downloaded-model", row.model.repo_id)
              : makeModelOptionKey("lora", row.fineTuned.id),
        ),
      );
    }

    if (
      section === "downloaded" &&
      (cachedReady || unslothAdditionalOnDeviceModels.length > 0) &&
      !downloadedCollapsed &&
      (unslothCachedGguf.length > 0 ||
        unslothCachedModelRows.length > 0 ||
        unslothAdditionalOnDeviceModels.length > 0)
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
      keys.push(
        ...unslothAdditionalOnDeviceModels.map((model) =>
          makeModelOptionKey("additional-on-device", model.id),
        ),
      );
    }

    if (showHfSection && section === "recommended") {
      keys.push(
        ...filteredRecommendedIds.map((id) =>
          makeModelOptionKey("search-recommended", id),
        ),
      );
      keys.push(
        ...searchRowIds.map((id) => makeModelOptionKey("search-hf", id)),
      );
      return keys;
    }

    if (
      section === "downloaded" &&
      (cachedReady || otherAdditionalOnDeviceModels.length > 0) &&
      !otherModelsCollapsed &&
      (otherCachedGguf.length > 0 ||
        otherCachedModelRows.length > 0 ||
        otherAdditionalOnDeviceModels.length > 0)
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
      keys.push(
        ...otherAdditionalOnDeviceModels.map((model) =>
          makeModelOptionKey("additional-on-device", model.id),
        ),
      );
    }

    if (section === "downloaded" && !fineTunedCollapsed) {
      keys.push(
        ...unpinnedFineTunedRows.map((m) => makeModelOptionKey("lora", m.id)),
      );
    }

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
        ...recommendedRows
          .filter((r) => !loadedRowIds.has(r.id.toLowerCase()))
          .map((r) => makeModelOptionKey("recommended", r.id)),
      );
    }

    return keys;
  }, [
    loadedRows,
    loadedRowIds,
    cachedReady,
    chatOnly,
    sortedCustomFolderModels,
    customFoldersCollapsed,
    connectedGroups,
    pinnedConnectedRows,
    pinnedConnectedCollapsed,
    collapsedConnectedGroups,
    pinnedRows,
    pinnedCollapsed,
    pinnedSoleQuantRows,
    downloadedCollapsed,
    unpinnedFineTunedRows,
    fineTunedCollapsed,
    filteredRecommendedIds,
    searchRowIds,
    sortedLmStudio,
    lmStudioCollapsed,
    recommendedRows,
    section,
    showHfSection,
    sortedLocalDir,
    localDirCollapsed,
    unslothCachedGguf,
    unslothCachedModelRows,
    unslothAdditionalOnDeviceModels,
    otherCachedGguf,
    otherCachedModelRows,
    otherAdditionalOnDeviceModels,
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
      {
        est: number;
        status: VramFitStatus | null;
        detail: string | null;
        budget?: CuratedBudget;
      }
    >();
    const pipelineBudget = artifactBudget(loadScopedGpu(gpu, Boolean(task)));
    for (const id of filteredRecommendedIds) {
      if (isKnownGgufRepo(id)) continue;
      const totalParams = recommendedParamCountById.get(id) ?? paramsFromId(id);
      // Searching must not change a row's device verdict.
      const curatedFit = catalogFit(id, pipelineBudget);
      if (catalog && curatedFit !== undefined) {
        const params = totalParams ?? curatedTotalParamsFor(id, catalog);
        map.set(id, {
          est: curatedFit.sizeGb ? Math.round(curatedFit.sizeGb) : 0,
          status: curatedFit.fits ? null : "exceeds",
          detail: params ? formatCompact(params) : null,
          budget: curatedBudget(curatedFit),
        });
        continue;
      }
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
    filteredRecommendedIds,
    recommendedParamCountById,
    isKnownGgufRepo,
    catalog,
    catalogFit,
    task,
    gpu,
  ]);

  const searchHasMore =
    hasMore || (communityDiscoveryEnabled && communityQuerySearch.hasMore);
  const searchIsLoadingMore =
    isLoadingMore || communityQuerySearch.isLoadingMore;
  const fetchSearchMore = useCallback((): boolean | undefined => {
    const unslothRequested = hasMore ? fetchMore() : false;
    const communityRequested =
      communityDiscoveryEnabled && communityQuerySearch.hasMore
        ? communityQuerySearch.fetchMore()
        : false;
    if (unslothRequested || communityRequested) return true;
    return undefined;
  }, [
    hasMore,
    fetchMore,
    communityDiscoveryEnabled,
    communityQuerySearch.hasMore,
    communityQuerySearch.fetchMore,
  ]);
  const { scrollRef, sentinelRef } = useHubInfiniteScroll(
    fetchSearchMore,
    scannedCount + communityQuerySearch.scannedCount,
    {
      enabled: online && searchHasMore,
      isFetching:
        isLoading || communityQuerySearch.isLoading || searchIsLoadingMore,
      resultCount: results.length + communityQuerySearch.results.length,
      resetKey: debouncedQuery,
    },
  );

  const updateListFades = useCallback((el: HTMLDivElement) => {
    const scrolled = el.scrollTop > 0;
    setListScrolled((prev) => (prev === scrolled ? prev : scrolled));
    const moreBelow = el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setListMoreBelow((prev) => (prev === moreBelow ? prev : moreBelow));
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    updateListFades(el);
    const observer = new ResizeObserver(() => updateListFades(el));
    observer.observe(el);
    if (el.firstElementChild) observer.observe(el.firstElementChild);
    return () => observer.disconnect();
  }, [scrollRef, updateListFades]);

  // Re-attach per loaded page so a heavily filtered list keeps paging; fetchMore no-ops in flight.
  const [recommendedSentinel, setRecommendedSentinel] =
    useState<HTMLDivElement | null>(null);
  const recommendedSentinelRef = useCallback((node: HTMLDivElement | null) => {
    setRecommendedSentinel(node);
  }, []);
  const recommendedHasMore =
    recommendedSearch.hasMore ||
    (communityRecommendedEnabled && communityBrowse.hasMore);
  const recommendedIsLoadingMore =
    recommendedSearch.isLoadingMore || communityBrowse.isLoadingMore;
  const fetchRecommendedMore = useCallback(() => {
    if (recommendedSearch.hasMore) {
      recommendedSearch.fetchMore();
      return;
    }
    if (communityRecommendedEnabled && communityBrowse.hasMore) {
      communityBrowse.fetchMore();
    }
  }, [
    recommendedSearch.hasMore,
    recommendedSearch.fetchMore,
    communityRecommendedEnabled,
    communityBrowse.hasMore,
    communityBrowse.fetchMore,
  ]);
  useEffect(() => {
    if (!recommendedSentinel || !recommendedHasMore) return;
    const root = scrollRef.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) fetchRecommendedMore();
      },
      { threshold: 0, root },
    );
    obs.observe(recommendedSentinel);
    return () => obs.disconnect();
  }, [
    recommendedSentinel,
    recommendedHasMore,
    fetchRecommendedMore,
    recommendedSearch.results.length,
    communityBrowse.results.length,
    scrollRef,
  ]);

  const handleModelClick = useCallback(
    (id: string) => {
      if (isKnownGgufRepo(id)) {
        setExpandedGguf((prev) => (prev === id ? null : id));
      } else {
        // A cached vendor copy stands in for its mirror.
        const cached = cachedIdFor(id);
        onSelect(cached ?? id, {
          source: "hub",
          isLora: false,
          isDownloaded: cached !== null,
          pipelineTag: pipelineTagById.get(id) ?? null,
        });
      }
    },
    [onSelect, isKnownGgufRepo, cachedIdFor, pipelineTagById],
  );

  const showDownloaded = section === "downloaded";
  const showCustom = section === "downloaded";
  const showRecommendedSection = !showHfSection && section === "recommended";
  const recommendedEmpty = recommendedEmptyState({
    isLoading: recommendedSearch.isLoading,
    error: recommendedSearch.error,
    hubPhase,
  });
  const searchEmpty = recommendedEmptyState({
    isLoading,
    error: searchError,
    hubPhase,
  });
  const downloadedEmpty =
    pinnedRows.length === 0 &&
    visibleCachedGguf.length === 0 &&
    visibleCachedModelRows.length === 0 &&
    visibleAdditionalOnDeviceModels.length === 0 &&
    npuOnDeviceRows.length === 0 &&
    sortedLmStudio.length === 0 &&
    sortedLocalDir.length === 0 &&
    // Do not show the empty state above a non-empty Fine-tuned section.
    unpinnedFineTunedRows.length === 0;

  // Fixed width matches the Search Hub button.
  const sortTriggerClassName =
    "h-(--picker-control-h) w-(--picker-control-w) shrink-0 justify-between gap-1.5 pl-4 pr-3.5 !border-0 text-xs [&>span]:!text-clip";
  // Keep the option's right padding so the checkmark never overlaps the label.
  const sortMenuContentClassName =
    "!p-1 !rounded-[14px] [&_[role=option]]:!pl-2 [&_[role=option]]:!py-1.5 [&_[role=option]]:!text-xs [&_[role=option]]:!rounded-[10px]";
  // The whole row is the button: label-click forwarding to a Checkbox <button> is unreliable.
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
            className="pointer-events-none size-3.5 rounded-full border-muted-foreground/70 [&_svg]:!size-2.5"
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
  const sortTriggerContent = (label: ReactNode) => (
    <span className="flex items-center gap-1">
      <HugeiconsIcon
        icon={ArrowUpDownIcon}
        strokeWidth={1.75}
        className="size-3.5 shrink-0 text-foreground"
      />
      <span className="truncate">{label}</span>
    </span>
  );
  // On Device rows are already on disk, so the fit filter only applies to the Unsloth listing.
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

  const showConnected = section === "connected";
  const severalLoaded = loadedModels.length > 1;
  const ejectsAll = Boolean(onEjectAll) && severalLoaded;
  const ejectsKept = Boolean(onEject) && severalLoaded;
  // The wider Connected box drops the search inset to keep Search Hub aligned.
  const hasConnected = externalModels.length > 0;
  const hasOtherModels =
    otherCachedGguf.length > 0 ||
    otherCachedModelRows.length > 0 ||
    otherAdditionalOnDeviceModels.length > 0;

  const renderHubModelRow = (id: string, row: ReactNode) => {
    if (!onConfigure) return row;
    if (isKnownGgufRepo(id)) {
      // Empty slot: GGUF repos configure per quant, and badges must line up.
      return (
        <div className={downloadedRowShellClassName(isValueRow(id))}>
          <div className="min-w-0 flex-1">{row}</div>
          <span className={ROW_ACTIONS_CLASS} aria-hidden={true} />
        </div>
      );
    }
    return (
      <div className={downloadedRowShellClassName(isValueRow(id))}>
        <div className="min-w-0 flex-1">{row}</div>
        <span className={ROW_ACTIONS_CLASS}>
          <ModelLoadSettingsAction
            ariaLabel={`Inference settings for ${id}`}
            onConfigure={() =>
              onConfigure(cachedIdFor(id) ?? id, {
                source: "hub",
                isLora: false,
                isGguf: false,
                isDownloaded: cachedIdFor(id) !== null,
                pipelineTag: pipelineTagById.get(id) ?? null,
              })
            }
          />
        </span>
      </div>
    );
  };

  const renderPinnedDragRow = (
    drag: ReturnType<typeof usePinnedRowDrag>,
    key: string,
    row: ReactNode,
  ) => {
    const edge = drag.lineEdge(key);
    return (
      <div
        key={key}
        {...drag.rowProps(key)}
        className={cn("relative", edge && [DROP_CUE_CLASS, PINNED_DROP_CUE[edge]])}
        style={drag.draggingKey === key ? { opacity: 0.4 } : undefined}
      >
        {row}
      </div>
    );
  };

  const ejectMenuItems = (modelId: string) =>
    ejectsKept && isKeptLoaded(modelId)
      ? [
          {
            key: "eject",
            label: "Eject",
            icon: (
              <HugeiconsIcon
                icon={RemoveCircleIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
            ),
            onSelect: () => onEject?.(modelId),
          },
        ]
      : undefined;

  const renderLoadedRow = (entry: (typeof loadedModels)[number]) => {
    // Key by checkpoint: local models load by path and two files can share a name.
    const checkpoint = entry.checkpoint ?? entry.id;
    const optionKey = makeModelOptionKey("loaded", checkpoint);
    const isSelected = modelIdsMatchForPicker(value, checkpoint);
    const quant = entry.quant ?? (isSelected ? activeGgufVariant : null);
    return (
      <div
        key={optionKey}
        className={downloadedRowShellClassName(isSelected)}
      >
        <div className="min-w-0 flex-1">
          <ModelRow
            label={entry.id}
            tooltipText={entry.id}
            hideOwner={isUnslothOwned(entry.id)}
            alignMeta="hub"
            meta={quant ? "GGUF" : extractParamLabel(entry.id)}
            tags={quant ? [ggufQuantChipLabel(quant)] : undefined}
            selected={isSelected}
            loaded={true}
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() =>
              onSelect(checkpoint, {
                source: /^(?:[a-zA-Z]:[\\/]|[\\/]|~)/.test(checkpoint) ? "local" : "hub",
                isLora: false,
                ggufVariant: quant ?? undefined,
                isDownloaded: true,
                isGguf: Boolean(quant),
              })
            }
            vramStatus={null}
            className={downloadedRowButtonClassName}
          />
        </div>
        <span className={ROW_ACTIONS_CLASS}>
          {ejectsKept ? (
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onEject?.(checkpoint);
                  }}
                  aria-label={`Eject ${entry.id}`}
                  className="flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-red-500 dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
                >
                  <HugeiconsIcon
                    icon={RemoveCircleIcon}
                    strokeWidth={1.75}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="top" className="tooltip-compact">
                Eject
              </TooltipContent>
            </Tooltip>
          ) : null}
        </span>
      </div>
    );
  };

  // Through ModelRow so badges and hover gutter do not drift from On Device rows.
  const renderConnectedModelRow = (
    model: ExternalModelOption,
    // Two connections can serve one model id, so a headless row must name its connection.
    headless = false,
  ) => {
    const optionKey = makeModelOptionKey("connected", model.id);
    const isSelected = value === model.id;
    const isPinned = pinnedConnectedSet.has(model.id);
    // `model.name` is rewritten by OpenRouter and `model.id` is the `external::` address.
    const providerModelId =
      parseExternalModelId(model.id)?.modelId ?? model.name;
    const baseUrl = externalBaseUrlById.get(model.providerId) ?? null;
    const marks = connectedModelMarks({
      providerType: model.providerType,
      modelId: providerModelId,
      baseUrl,
      apiType: externalApiTypeById.get(model.providerId),
    });
    return (
      <div
        key={model.id}
        // ml-4 + pl-3.5 = 30px, where a heading's label starts.
        className={cn(downloadedRowShellClassName(isSelected), "ml-4")}
      >
        <div className="min-w-0 flex-1">
          <ModelRow
            label={model.name}
            // The provider's own id, since labels can be rewritten or collide.
            tooltipText={
              headless ? (
                <>
                  {providerModelId}
                  <span className="block text-ui-10 mt-1">
                    {model.providerName}
                  </span>
                </>
              ) : (
                providerModelId
              )
            }
            capabilities={marks.capabilities}
            showVision={marks.vision}
            selected={isSelected}
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() =>
              onSelect(model.id, { source: "external", isLora: false })
            }
            vramStatus={null}
            className={cn(downloadedRowButtonClassName, "pl-3.5")}
          />
        </div>
        <span className={ROW_ACTIONS_CLASS}>
          <ModelLoadSettingsAction
            ariaLabel={`Settings for ${model.name}`}
            tooltip="Model settings"
            onConfigure={() =>
              setSettingsModel({
                model,
                providerModelId,
                apiType: externalApiTypeById.get(model.providerId),
                baseUrl,
                isReasoningProvider:
                  externalReasoningFlagById.get(model.providerId) === true,
                reasoningConfig: externalReasoningConfigById.get(model.providerId),
                connectionMaxOutputTokens:
                  externalMaxOutputById.get(model.providerId) ?? null,
              })
            }
          />
          <ModelRowMenu
            ariaLabel={`More options for ${model.name}`}
            pin={{
              pinned: isPinned,
              pinLabel: "Pin",
              unpinLabel: "Unpin",
              onToggle: () => togglePinnedConnected(model.id),
            }}
            items={[
              {
                key: "info",
                label: "Model info",
                icon: (
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
                ),
                onSelect: () =>
                  setInfoModel({
                    model,
                    providerModelId,
                    apiType: externalApiTypeById.get(model.providerId),
                    baseUrl,
                    isReasoningProvider:
                      externalReasoningFlagById.get(model.providerId) === true,
                    reasoningConfig: externalReasoningConfigById.get(model.providerId),
                  }),
              },
              {
                key: "copy",
                label: "Copy model ID",
                icon: (
                  <HugeiconsIcon
                    icon={Copy01Icon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
                ),
                onSelect: () => void copyConnectedModelId(providerModelId),
              },
            ]}
          />
        </span>
      </div>
    );
  };

  const renderPinnedQuantRow = (entry: { repoId: string; quant: string }) => {
    const soleRow = pinnedSoleQuantRows.get(pinKey(entry.repoId, entry.quant));
    if (soleRow) return renderSoleQuantGgufRow(soleRow.repo, soleRow.sole);
    // So the delete removes the copy the validation listing resolved.
    const pinnedCopyPath = downloadedPinnedQuantPaths.get(
      pinKey(entry.repoId, entry.quant),
    );
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
    // A validated false is authoritative; anything else stays unknown.
    const pinnedVisionHint =
      pinnedQuantValidation.visionByRepo.get(entry.repoId) === false
        ? false
        : undefined;
    const sizeBytes = pinnedQuantValidation.sizes.get(
      pinKey(entry.repoId, entry.quant),
    );
    const hasVision =
      pinnedQuantValidation.visionByRepo.get(entry.repoId) ??
      sortedCachedGguf.find((c) => c.repo_id === entry.repoId)?.has_vision;
    return (
      <div
        key={optionKey}
        className={downloadedRowShellClassName(isSelected, true)}
      >
        <div className="min-w-0 flex-1">
          <ModelRow
            label={entry.repoId}
            tooltipText={`${entry.repoId} (${ggufQuantDetailLabel(
              entry.quant,
            )})`}
            meta={
              sizeBytes ? `GGUF · ${formatBytes(sizeBytes)}` : "GGUF"
            }
            quantChip={ggufQuantChipLabel(entry.quant)}
            showVision={hasVision}
            // Media-task quants load through the media planner, so the KV estimator would measure the wrong runtime.
            memory={
              mediaPageForTask(
                diffusionTaskById.get(entry.repoId.toLowerCase()),
              )
                ? undefined
                : { repoId: entry.repoId, quant: entry.quant }
            }
            gpuGb={expanderGpuGb}
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
                // A GGUF pick; without it pages request a pipeline. No filename: the pin stores a label.
                isGguf: true,
                isVision: pinnedVisionHint,
                pipelineTag:
                  diffusionTaskById.get(entry.repoId.toLowerCase()) ?? null,
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
                  isVision: pinnedVisionHint,
                  pipelineTag:
                    diffusionTaskById.get(entry.repoId.toLowerCase()) ?? null,
                })
              }
            />
          )}
          <ModelRowMenu
            ariaLabel={`More options for ${entry.repoId} ${entry.quant}`}
            cachePath={{ repoId: entry.repoId, variant: entry.quant }}
            pin={{
              pinned: true,
              pinLabel: "Pin",
              unpinLabel: "Unpin",
              onToggle: () => togglePinned(entry.repoId, entry.quant),
            }}
            del={{
              title: "Delete cached model?",
              // Shows why a still-needed companion base disables Delete.
              impact: {
                repoId: entry.repoId,
                variant: entry.quant,
                cachePath: pinnedCopyPath,
              },
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
                  pinnedCopyPath ?? undefined,
                );
                refreshCachedLists();
                if (isChatGgufTask(diffusionTaskById.get(entry.repoId.toLowerCase()))) {
                  await reconcileGgufPinsAfterDelete(entry.repoId, hfToken || undefined);
                } else {
                  togglePinned(entry.repoId, entry.quant);
                }
              },
            }}
          />
        </span>
      </div>
    );
  };

  // One quant on disk with "All quantizations" off: the row carries it as a chip.
  const renderSoleQuantGgufRow = (
    c: (typeof visibleCachedGguf)[number],
    sole: SoleDownloadedQuant,
  ) => {
    const variant = sole.variant;
    const optionKey = makeModelOptionKey("downloaded-gguf", c.repo_id);
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
    // Should never be partial, but if it is, state what is on disk rather than load a torn file.
    const isPartial = c.partial === true;
    const isDownloaded = variant.downloaded === true && !isPartial;
    const selectMeta: ModelSelectorChangeMeta = {
      source: "hub",
      isLora: false,
      // Only for complete snapshots: the Audio route reads a loadId as proof the weights are present.
      loadId: isDownloaded ? c.load_id : undefined,
      ggufVariant: variant.quant,
      ggufFilename: variant.filename,
      isDownloaded,
      expectedBytes,
      isGguf: true,
      // Forward the mmproj verdict conservatively so a known text-only quant still warns.
      isVision: sole.hasVision === false ? false : undefined,
      pipelineTag: c.task ?? null,
    };
    return (
      <div
        key={c.repo_id}
        className={downloadedRowShellClassName(isSelected, true)}
      >
        <div className="min-w-0 flex-1">
          <ModelRow
            label={c.repo_id}
            tooltipText={localPathTooltip(
              c.repo_id,
              c.cache_path,
              ggufQuantDetailLabel(variant.quant),
            )}
            meta={`GGUF · ${formatBytes(variant.size_bytes)}`}
            quantChip={ggufQuantChipLabel(variant.quant)}
            partial={isPartial}
            // Only for llama.cpp loads: diffusion GGUFs run on a different planner and the estimate would be wrong.
            memory={
              mediaPageForTask(c.task)
                ? undefined
                : {
                    repoId: c.repo_id,
                    quant: variant.quant,
                    sizeBytes: variant.size_bytes,
                    loadId: c.load_id,
                  }
            }
            gpuGb={expanderGpuGb}
            showVision={c.has_vision || sole.hasVision}
            selected={isSelected}
            loaded={rowState.loaded || isKeptLoaded(c.repo_id)}
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
            items={ejectMenuItems(c.repo_id)}
            pin={{
              pinned: isPinned,
              pinLabel: "Pin",
              unpinLabel: "Unpin",
              onToggle: () => togglePinned(c.repo_id, variant.quant),
            }}
            del={{
              title: "Delete cached model?",
              impact: {
                repoId: c.repo_id,
                variant: variant.quant,
                cachePath:
                  variant.cache_ref ??
                  variant.cache_path ??
                  (mediaPageForTask(c.task) ? c.cache_path : null),
              },
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
                  variant.cache_ref ??
                    variant.cache_path ??
                    (mediaPageForTask(c.task) ? c.cache_path : undefined) ??
                    undefined,
                );
                if (isChatGgufTask(c.task)) {
                  await reconcileGgufPinsAfterDelete(c.repo_id, hfToken || undefined);
                } else if (isPinned) {
                  togglePinned(c.repo_id, variant.quant);
                }
                prunePinnedQuantValidation(c.repo_id, variant.quant);
                refreshCachedLists();
              },
            }}
          />
        </span>
      </div>
    );
  };

  const renderDownloadedGgufRow = (c: (typeof visibleCachedGguf)[number]) => {
    const optionKey = makeModelOptionKey("downloaded-gguf", c.repo_id);
    const isSelected = value === c.repo_id;
    const soleQuant = soleQuants.quants.get(c.repo_id);
    if (soleQuant) return renderSoleQuantGgufRow(c, soleQuant);
    const loadedQuants = loadedQuantsFor(c.repo_id);
    // Auto-expansion waits for the probe, or every row mounts an expander and a remote listing.
    const expanderOpen = shouldMountVariantExpander({
      expanded: isGgufExpanded(c.repo_id),
      autoExpand: expandQuantizations && !reopenedGguf.has(c.repo_id),
      soleQuantsPending: soleQuants.pending.has(c.repo_id),
    });
    // No clean quant, so nothing in the expander can carry the row's actions.
    const isPartialRepo = c.partial === true;
    return (
      <div key={c.repo_id}>
        <div className={downloadedRowShellClassName(isSelected)}>
          <div className="min-w-0 flex-1">
            <ModelRow
              label={c.repo_id}
              tooltipText={localPathTooltip(c.repo_id, c.cache_path)}
              meta="GGUF"
              quantChip={loadedQuants.map(ggufQuantChipLabel).join(", ") || undefined}
              showVision={c.has_vision ?? visionByRepo[c.repo_id]}
              alignMeta="device"
              partial={isPartialRepo}
              partialResumable={c.partial_resumable}
              selected={isSelected}
              loaded={
                isKeptLoaded(c.repo_id) ||
                isRuntimeLoadedModel(
                  loadedModelId,
                  activeGgufVariant,
                  c.repo_id,
                  "required",
                )
              }
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
          {isPartialRepo ? (
            <span className={ROW_ACTIONS_PINNED_CLASS}>
              <ModelRowMenu
                ariaLabel={`More options for ${c.repo_id}`}
                cachePath={{ repoId: c.repo_id }}
                del={{
                  title: "Delete cached model?",
                  impact: { repoId: c.repo_id },
                  // Repo-wide delete, which may include a complete copy in another format.
                  description: (
                    <>
                      This will remove{" "}
                      <span className="font-medium text-foreground">
                        {c.repo_id}
                      </span>{" "}
                      and everything downloaded under it from disk. You can
                      download it again later.
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
                    if (isChatGgufTask(c.task)) {
                      // Another remembered folder may still hold a pinned quant.
                      await reconcileGgufPinsAfterDelete(c.repo_id, hfToken || undefined);
                    } else {
                      unpinRepo(c.repo_id);
                    }
                  },
                  onDeleted: refreshCachedLists,
                }}
              />
            </span>
          ) : (
            <span aria-hidden="true" className={cn(ROW_ACTIONS_CLASS, "h-6")} />
          )}
        </div>
        {expanderOpen && (
          <GgufVariantExpander
            diffusionLoad={diffusionLoad}
            hostPooledMemory={gpu.loadDeviceSharesHostMemory}
            gpuCount={expanderGpuCount}
            repoId={c.repo_id}
            loadedQuants={loadedQuants}
            pipelineTag={c.task ?? null}
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
            systemRamGb={expanderRamGb || undefined}
            budgetKnown={expanderBudgetGpu.budgetKnown}
            variantActions={{
              onUpdate: (quant, expectedBytes) =>
                updateGgufVariant(c.repo_id, quant, expectedBytes),
              updateDisabled: loadedModelId === c.repo_id,
              onDelete: async (quant, cachePath) => {
                await deleteCachedModel(
                  c.repo_id,
                  quant,
                  hfToken || undefined,
                  cachePath ??
                    (mediaPageForTask(c.task) ? c.cache_path || undefined : undefined),
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
    // A partial pick must not claim downloaded, or the load fails on missing shards.
    const isPartial = c.partial === true;
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
            partial={isPartial}
            partialResumable={c.partial_resumable}
            loaded={
              isKeptLoaded(c.repo_id) ||
              isRuntimeLoadedModel(
                loadedModelId,
                activeGgufVariant,
                c.repo_id,
                "none",
              )
            }
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() =>
              onSelect(c.repo_id, {
                source: "hub",
                isLora: false,
                // The Audio route reads a forwarded loadId as proof the weights are there.
                loadId: isPartial ? undefined : c.load_id,
                isDownloaded: !isPartial,
                pipelineTag: c.task ?? null,
                audioType: c.audio_type ?? null,
                familyOverrideRequired: c.opaque === true,
              })
            }
            vramStatus={null}
            className={downloadedRowButtonClassName}
          />
        </div>
        <span
          className={isPartial ? ROW_ACTIONS_PINNED_CLASS : ROW_ACTIONS_CLASS}
        >
          {onConfigure && (
            <ModelLoadSettingsAction
              ariaLabel={`Inference settings for ${c.repo_id}`}
              onConfigure={() =>
                onConfigure(c.repo_id, {
                  source: "hub",
                  isLora: false,
                  // No load identity for an incomplete snapshot.
                  loadId: isPartial ? undefined : c.load_id,
                  isDownloaded: !isPartial,
                  isGguf: false,
                  pipelineTag: c.task ?? null,
                  audioType: c.audio_type ?? null,
                  familyOverrideRequired: c.opaque === true,
                })
              }
            />
          )}
          <ModelRowMenu
            ariaLabel={`More options for ${c.repo_id}`}
            cachePath={{ repoId: c.repo_id }}
            items={ejectMenuItems(c.repo_id)}
            pin={{
              pinned: pinnedSet.has(pinKey(c.repo_id)),
              pinLabel: "Pin",
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
                // Repo-wide, so quant pins go too.
                unpinRepo(c.repo_id);
              },
              onDeleted: refreshCachedLists,
            }}
          />
        </span>
      </div>
    );
  };

  const renderAdditionalOnDeviceModelRow = (model: ModelOption) => {
    const optionKey = makeModelOptionKey("additional-on-device", model.id);
    const isSelected = value === model.id;
    const pipelineTag = typeof task === "string" ? task : (task?.[0] ?? null);
    // A locally trained checkpoint is identified by its directory, so drop the Hub link.
    const isLocalPath = /^(?:[a-zA-Z]:[\\/]|[\\/]|~)/.test(model.id);
    return (
      <div key={model.id} className={downloadedRowShellClassName(isSelected)}>
        <div className="min-w-0 flex-1">
          <ModelRow
            label={isLocalPath ? model.name : model.id}
            hubUrl={isLocalPath ? undefined : hubRepoUrl(model.id)}
            meta={
              isLocalPath
                ? (model.description ?? "Trained here")
                : `${model.isGguf === true ? "GGUF" : "Safetensors"}${model.deviceSize ? ` · ${model.deviceSize}` : ""}`
            }
            quantChip={model.deviceQuant}
            selected={isSelected}
            loaded={model.deviceLoaded === true}
            capabilities={detectCapabilities({
              id: model.id,
              pipelineTag: pipelineTag ?? undefined,
            })}
            alignMeta="device"
            optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
            onClick={() =>
              onSelect(model.id, {
                source: "hub",
                isLora: false,
                isDownloaded: true,
                isGguf: model.isGguf === true,
                pipelineTag,
                audioType: model.audioType ?? null,
              })
            }
            vramStatus={null}
            className={downloadedRowButtonClassName}
          />
        </div>
        <span aria-hidden="true" className={cn(ROW_ACTIONS_CLASS, "h-6")} />
      </div>
    );
  };

  const renderNpuRow = (model: NpuModel, onDevice: boolean) => {
    if (!npuCatalog) return null;
    const optionKey = makeModelOptionKey(
      onDevice ? "npu-on-device" : "npu",
      model.id,
    );
    const isSelected = value === model.model_path;
    const isLoaded = loadedModelId === model.model_path;
    const downloading = model.id in npuCatalog.downloads;
    const progress = npuCatalog.downloads[model.id];
    const reconnecting = model.id in npuCatalog.reconnecting;
    const details = [
      "NPU",
      npuSizeLabel(model.size_gb),
      npuResumeLabel(model),
    ].filter(Boolean);
    const meta = {
      source: "local",
      isLora: false,
      isDownloaded: true,
    } as const;
    const pick = () => onSelect(model.model_path, meta);
    const row = (
      <ModelRow
        label={model.id}
        meta={
          downloading
            ? `NPU · ${npuDownloadLabel(progress, reconnecting)}`
            : details.join(" · ")
        }
        selected={isSelected}
        loaded={isLoaded}
        downloaded={!onDevice && model.downloaded}
        capabilities={{
          vision: model.supports_vision,
          reasoning: model.supports_reasoning,
          audio: false,
          imageGen: false,
          videoGen: false,
        }}
        alignMeta={onDevice ? "device" : "hub"}
        optionProps={hubModelList.getOptionProps(optionKey, isSelected)}
        onClick={() => {
          if (downloading) return;
          if (model.downloaded) {
            pick();
            return;
          }
          const before = useChatRuntimeStore.getState();
          const chosen = before.params.checkpoint;
          void npuCatalog.download(model).then((done) => {
            const now = useChatRuntimeStore.getState();
            // A model picked while this downloaded is the user's newer choice; keep it.
            if (done && now.params.checkpoint === chosen && !now.loadingModelPick) {
              pick();
            }
          });
        }}
        vramStatus={null}
        className={onDevice ? downloadedRowButtonClassName : undefined}
      />
    );
    const settings =
      onConfigure && model.downloaded && !downloading ? (
        <ModelLoadSettingsAction
          ariaLabel={`Inference settings for ${model.id}`}
          onConfigure={() =>
            onConfigure(model.model_path, {
              ...meta,
              contextLength: model.max_context_length,
            })
          }
        />
      ) : null;
    if (!onDevice && !onConfigure) return <div key={model.id}>{row}</div>;
    return (
      <div key={model.id} className={downloadedRowShellClassName(isSelected)}>
        <div className="min-w-0 flex-1">{row}</div>
        <span
          className={cn(ROW_ACTIONS_CLASS, onDevice && "h-6")}
          aria-hidden={settings || onDevice ? undefined : true}
        >
          {settings}
          {onDevice && (
            <ModelDeleteAction
              ariaLabel={`Delete ${model.id}`}
              title={`Delete ${model.id}?`}
              description="This removes the model's NPU files from this device."
              successMessage={`Deleted ${model.id}`}
              disabled={isLoaded}
              onConfirm={() => npuCatalog.remove(model)}
            />
          )}
        </span>
      </div>
    );
  };

  const npuBrowseForcedOpen = formatFilter === "npu" || showHfSection;
  const npuBrowseFolded = !npuBrowseForcedOpen && npuBrowseCollapsed;
  const showNpuBrowse =
    npuCatalog !== null &&
    npuListed &&
    section === "recommended" &&
    (!showHfSection || npuBrowseRows.length > 0);

  return (
    <CapabilityScope.Provider value={capabilityScope}>
      <div className="relative space-y-2">
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
              className="field-soft h-(--picker-control-h) border-0 pl-8 pr-8 text-sm"
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
                  className="hub-tab-toggle-pill hub-pill-action flex h-(--picker-control-h) w-(--picker-control-w) shrink-0 items-center justify-center gap-[calc(5px*var(--ui-space-scale,1))] rounded-full border-0 text-xs text-foreground transition-colors"
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

        <div
          className={cn(
            "flex flex-wrap items-center gap-2",
            hasConnected ? "-mr-4" : "-mr-2",
          )}
        >
          {sectionToggle}
          {showConnected ? (
            <div className="flex max-w-full min-w-0 flex-wrap items-center gap-2">
              <HubOptionMenu
                value={connectedModality}
                options={CONNECTED_MODALITY_OPTIONS}
                onValueChange={setConnectedModality}
                ariaLabel="Filter by modality"
                align="end"
                className={sortTriggerClassName}
                contentClassName={sortMenuContentClassName}
              />
              <HubOptionMenu
                value={connectedSort}
                options={CONNECTED_SORT_OPTIONS}
                onValueChange={setConnectedSort}
                ariaLabel="Sort connected models"
                align="end"
                className={sortTriggerClassName}
                contentClassName={sortMenuContentClassName}
                triggerContent={sortTriggerContent(
                  CONNECTED_SORT_OPTIONS.find(
                    (option) => option.value === connectedSort,
                  )?.label ?? connectedSort,
                )}
              />
            </div>
          ) : (
            <div className="flex max-w-full min-w-0 flex-wrap items-center gap-2">
              <HubOptionMenu
                value={formatFilter}
                options={
                  npuCatalog
                    ? FORMAT_FILTER_OPTIONS
                    : FORMAT_FILTER_OPTIONS_WITHOUT_NPU
                }
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
            // Padding keeps the focus ring off the clip edges; the right inset lives in index.css.
            "model-list-scroll max-h-[calc(335px*var(--ui-space-scale,1))] overflow-y-auto scroll-py-1.5 pl-0.5",
            listScrolled && "is-scrolled",
            listMoreBelow && "is-bottom-faded",
          )}
          {...hubModelList.listboxProps}
        >
          <div
            className={cn(
              // Keep row actions clear of overlay scrollbars.
              "model-list-gutter",
              showDownloaded ? "pt-0" : "pt-[calc(4px*var(--ui-space-scale,1))]",
              onEject ? "pb-[calc(60px*var(--ui-space-scale,1))]" : "pb-4",
            )}
          >
            {loadedRows.length > 0 ? (
              <div className="pb-1.5">
                {loadedRows.map(renderLoadedRow)}
                <div className="mx-2.5 mt-1.5 border-t border-border/50" />
              </div>
            ) : null}
            {showConnected ? (
              connectedMatches.length === 0 ? (
                <div className="px-2.5 py-2 text-xs leading-relaxed text-muted-foreground">
                  {externalModels.length === 0
                    ? "No models from your connections. Set up in Settings then Connections."
                    : "No models match your search."}
                </div>
              ) : (
                <>
                  {pinnedConnectedRows.length > 0 ? (
                    <div>
                      <ConnectedGroupHeading
                        icon={
                          <HugeiconsIcon icon={PinIcon} className="size-3.5" />
                        }
                        label="Pinned"
                        collapsed={pinnedConnectedCollapsed}
                        onToggle={() =>
                          setPinnedConnectedCollapsed((value) => !value)
                        }
                      />
                      {pinnedConnectedCollapsed
                        ? null
                        : pinnedConnectedRows.map((model) =>
                            renderPinnedDragRow(
                              pinnedConnectedDrag,
                              model.id,
                              renderConnectedModelRow(model, true),
                            ),
                          )}
                    </div>
                  ) : null}
                  {connectedGroups.map((group) => {
                    const headed = group.providerName.length > 0;
                    const collapsed =
                      headed && collapsedConnectedGroups.has(group.providerId);
                    return (
                      <div key={group.providerId}>
                        {headed ? (
                          <ConnectedGroupHeading
                            icon={
                              <ApiProviderLogo
                                providerType={group.providerType}
                                className="size-3.5"
                                title={group.providerName}
                              />
                            }
                            label={group.providerName}
                            collapsed={collapsed}
                            onToggle={() =>
                              toggleConnectedGroup(group.providerId)
                            }
                            onConfigure={
                              onConfigureConnection
                                ? () => onConfigureConnection(group.providerId)
                                : undefined
                            }
                            configureLabel={`${group.providerName} connection settings`}
                          />
                        ) : null}
                        {collapsed
                          ? null
                          : group.models.map((model) =>
                              renderConnectedModelRow(model, !headed),
                            )}
                      </div>
                    );
                  })}
                </>
              )
            ) : (
              <>
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
                        renderPinnedDragRow(
                          pinnedDrag,
                          row.key,
                          row.entry ? (
                            renderPinnedQuantRow(row.entry)
                          ) : row.model ? (
                            renderDownloadedModelRow(row.model)
                          ) : (
                            <FineTunedRows
                              adapters={[row.fineTuned]}
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
                          ),
                        ),
                      )}
                  </>
                ) : null}

                {showDownloaded &&
                (unslothCachedGguf.length > 0 ||
                  unslothCachedModelRows.length > 0 ||
                  unslothAdditionalOnDeviceModels.length > 0) ? (
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
                                  className="shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
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
                                  className="shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
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
                                className="shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
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
                      Unsloth
                    </ListLabel>
                    {!downloadedCollapsed &&
                      unslothCachedGguf.map(renderDownloadedGgufRow)}
                    {!downloadedCollapsed &&
                      unslothCachedModelRows.map(renderDownloadedModelRow)}
                    {!downloadedCollapsed &&
                      unslothAdditionalOnDeviceModels.map(
                        renderAdditionalOnDeviceModelRow,
                      )}
                  </>
                ) : null}

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
                    {!otherModelsCollapsed &&
                      otherAdditionalOnDeviceModels.map(
                        renderAdditionalOnDeviceModelRow,
                      )}
                  </div>
                ) : null}

                {showDownloaded && npuOnDeviceRows.length > 0 ? (
                  <>
                    <ListLabel
                      divider={
                        pinnedRows.length > 0 ||
                        unslothCachedGguf.length > 0 ||
                        unslothCachedModelRows.length > 0 ||
                        unslothAdditionalOnDeviceModels.length > 0 ||
                        hasOtherModels
                      }
                      collapsed={npuOnDeviceCollapsed}
                      onToggle={() => setNpuOnDeviceCollapsed((v) => !v)}
                    >
                      NPU
                    </ListLabel>
                    {!npuOnDeviceCollapsed &&
                      npuOnDeviceRows.map((model) => renderNpuRow(model, true))}
                  </>
                ) : null}

                {section === "downloaded" && !task ? (
                  <>
                    <div
                      ref={fineTunedSectionRef}
                      className="mt-3 flex items-center gap-1 border-t border-border px-2.5 pb-1 pt-3"
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
                          className="-mr-0.5 shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
                        >
                          {fineTunedCollapsed ? (
                            <ChevronRightIcon className="size-3" />
                          ) : (
                            <ChevronDownIcon className="size-3" />
                          )}
                        </button>
                      </div>
                    </div>
                    {!fineTunedCollapsed && unpinnedFineTunedRows.length > 0 && (
                      <FineTunedRows
                        adapters={unpinnedFineTunedRows}
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
                      className="mt-3 flex items-center gap-1 border-t border-border px-2.5 pb-1 pt-3"
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
                          className="shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
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
                          className="shrink-0 rounded p-0.5 text-muted-foreground/80 transition-colors hover:text-foreground"
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
                          className="-mr-0.5 shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
                        >
                          {customFoldersCollapsed ? (
                            <ChevronRightIcon className="size-3" />
                          ) : (
                            <ChevronDownIcon className="size-3" />
                          )}
                        </button>
                      </div>
                    </div>

                    {!customFoldersCollapsed &&
                      scanFolders.map((f) => {
                        const problem = scanFolderStatusCopy(f.status);
                        return (
                          <div
                            key={f.id}
                            className="group flex items-center gap-1.5 px-2.5 py-0.5"
                          >
                            <HugeiconsIcon
                              icon={Folder02Icon}
                              className="size-3 shrink-0 text-muted-foreground/80"
                            />
                            <div className="min-w-0 flex-1">
                              <span
                                className="block truncate font-mono text-ui-10 text-muted-foreground/80"
                                title={f.path}
                              >
                                {f.path}
                              </span>
                              {problem ? (
                                <span
                                  className="block truncate text-ui-10 text-amber-600 dark:text-amber-500"
                                  title={problem.hint}
                                >
                                  {problem.title}
                                </span>
                              ) : null}
                            </div>
                            <button
                              type="button"
                              onClick={() => handleRemoveFolder(f.id)}
                              aria-label={`Remove folder ${f.path}`}
                              className="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:bg-destructive/10 focus-visible:text-destructive"
                            >
                              <HugeiconsIcon
                                icon={Cancel01Icon}
                                className="size-3"
                              />
                            </button>
                          </div>
                        );
                      })}

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
                                className="rounded-full border border-dashed border-border px-2 py-0.5 font-mono text-ui-10 text-muted-foreground/80 transition-colors hover:border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)] hover:bg-accent hover:text-foreground disabled:opacity-40"
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

                    {!customFoldersCollapsed && showFolderInput && (
                      <div className="px-2.5 pb-1 pt-0.5">
                        <div className="flex items-center gap-1">
                          <HugeiconsIcon
                            icon={Folder02Icon}
                            className="size-3 shrink-0 text-muted-foreground/80"
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
                            className="h-6 min-w-0 flex-1 rounded border border-border bg-transparent px-1.5 font-mono text-ui-10 text-foreground outline-none placeholder:text-muted-foreground/80 focus:border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)]"
                            disabled={folderLoading}
                            autoFocus={true}
                          />
                          <button
                            type="button"
                            onClick={() => setShowFolderBrowser(true)}
                            disabled={folderLoading}
                            aria-label="Browse for folder"
                            title="Browse folders on the server"
                            className="flex h-6 shrink-0 items-center justify-center rounded border border-border px-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
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
                            className="h-6 shrink-0 rounded border border-border px-1.5 text-ui-10 text-muted-foreground transition-colors hover:bg-accent disabled:opacity-40"
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
                        // Pass the path explicitly: `folderInput` state has not flushed yet.
                        void handleAddFolder(picked);
                      }}
                    />

                    {!customFoldersCollapsed &&
                      sortedCustomFolderModels.map((m) => {
                        const isGgufFile = m.path
                          .toLowerCase()
                          .endsWith(".gguf");
                        // Honor the backend model_format hint (suffixless GGUF folders).
                        const isGguf = localModelIsGguf(m);
                        // An Ollama manifest names one blob, so it loads directly.
                        const isDirectGguf =
                          isGgufFile || m.source === "ollama";
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
                                    // Direct loads set no active variant.
                                    isDirectGguf
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
                                      onSelect(
                                        m.id,
                                        localDirectGgufMeta(m.task),
                                      );
                                    } else if (isGguf) {
                                      toggleGgufExpanded(m.id);
                                    } else {
                                      onSelect(
                                        m.id,
                                        localModelMeta(false, m.task, m.audio_type, m.opaque === true),
                                      );
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
                                  className="pr-1"
                                />
                              </div>
                              <span className={ROW_ACTIONS_CLASS}>
                                {isDirectGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(
                                        m.id,
                                        localDirectGgufMeta(m.task),
                                      )
                                    }
                                  />
                                )}
                                {!isGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(
                                        m.id,
                                        localModelMeta(false, m.task, m.audio_type, m.opaque === true),
                                      )
                                    }
                                  />
                                )}
                              </span>
                            </div>
                            {isGguf &&
                              !isDirectGguf &&
                              isGgufExpanded(m.id) && (
                                <GgufVariantExpander
                                  diffusionLoad={diffusionLoad}
                                  hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                  gpuCount={expanderGpuCount}
                                  repoId={m.id}
                                  loadedQuants={loadedQuantsFor(m.id)}
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
                                  systemRamGb={expanderRamGb || undefined}
                                  budgetKnown={expanderBudgetGpu.budgetKnown}
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
                        // LM Studio dirs rarely carry a -GGUF suffix, so use the shared helper.
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
                                      onSelect(
                                        m.id,
                                        localDirectGgufMeta(m.task),
                                      );
                                    } else if (isGguf) {
                                      toggleGgufExpanded(m.id);
                                    } else {
                                      onSelect(
                                        m.id,
                                        localModelMeta(false, m.task, m.audio_type, m.opaque === true),
                                      );
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
                                  className="pr-1"
                                />
                              </div>
                              <span className={ROW_ACTIONS_CLASS}>
                                {isGgufFile && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(
                                        m.id,
                                        localDirectGgufMeta(m.task),
                                      )
                                    }
                                  />
                                )}
                                {!isGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(
                                        m.id,
                                        localModelMeta(false, m.task, m.audio_type, m.opaque === true),
                                      )
                                    }
                                  />
                                )}
                              </span>
                            </div>
                            {isGguf && !isGgufFile && isGgufExpanded(m.id) && (
                              <GgufVariantExpander
                                diffusionLoad={diffusionLoad}
                                hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                gpuCount={expanderGpuCount}
                                repoId={m.id}
                                loadedQuants={loadedQuantsFor(m.id)}
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
                                systemRamGb={expanderRamGb || undefined}
                                budgetKnown={expanderBudgetGpu.budgetKnown}
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
                        // The variant scanner returns nothing for a config-less loose file, so expanding would dead-end.
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
                                      onSelect(
                                        m.id,
                                        localDirectGgufMeta(m.task),
                                      );
                                    } else if (isGguf) {
                                      toggleGgufExpanded(m.id);
                                    } else {
                                      onSelect(
                                        m.id,
                                        localModelMeta(false, m.task, m.audio_type, m.opaque === true),
                                      );
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
                                  className="pr-1"
                                />
                              </div>
                              <span className={ROW_ACTIONS_CLASS}>
                                {isGgufFile && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(
                                        m.id,
                                        localDirectGgufMeta(m.task),
                                      )
                                    }
                                  />
                                )}
                                {!isGguf && onConfigure && (
                                  <ModelLoadSettingsAction
                                    ariaLabel={`Inference settings for ${
                                      m.model_id ?? m.display_name
                                    }`}
                                    onConfigure={() =>
                                      onConfigure(
                                        m.id,
                                        localModelMeta(false, m.task, m.audio_type, m.opaque === true),
                                      )
                                    }
                                  />
                                )}
                              </span>
                            </div>
                            {isGguf && !isGgufFile && isGgufExpanded(m.id) && (
                              <GgufVariantExpander
                                diffusionLoad={diffusionLoad}
                                hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                gpuCount={expanderGpuCount}
                                repoId={m.id}
                                loadedQuants={loadedQuantsFor(m.id)}
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
                                systemRamGb={expanderRamGb || undefined}
                                budgetKnown={expanderBudgetGpu.budgetKnown}
                              />
                            )}
                          </div>
                        );
                      })}
                  </>
                ) : null}

                {showNpuBrowse && npuCatalog ? (
                  <>
                    <ListLabel
                      collapsed={npuBrowseFolded}
                      onToggle={
                        npuBrowseForcedOpen
                          ? undefined
                          : () => setNpuBrowseCollapsed((v) => !v)
                      }
                    >
                      NPU
                    </ListLabel>
                    {npuBrowseFolded ? null : (
                      <>
                        {showHfSection ? null : (
                          <NpuSetupNotice catalog={npuCatalog} />
                        )}
                        {npuBrowseRows.map((model) =>
                          renderNpuRow(model, false),
                        )}
                      </>
                    )}
                  </>
                ) : null}

                {showRecommendedSection && formatFilter !== "npu" ? (
                  <>
                    {showNpuBrowse ? (
                      <ListLabel divider={!npuBrowseFolded}>Unsloth</ListLabel>
                    ) : null}
                    {recommendedRows.length === 0 &&
                    recommendedEmpty === "loading" ? (
                      <div className="flex items-center gap-2 px-5 py-3">
                        <Spinner className="size-3 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground">
                          Loading models…
                        </span>
                      </div>
                    ) : recommendedRows.length === 0 &&
                      recommendedEmpty === "failed" ? (
                      <HubFailureHint
                        message={recommendedSearch.error}
                        onRetry={recommendedSearch.retry}
                      />
                    ) : recommendedRows.length === 0 ? (
                      <div className="px-2.5 py-2 text-xs text-muted-foreground">
                        No models found.
                      </div>
                    ) : (
                      recommendedRows.map((r) => {
                        const id = r.id;
                        if (loadedRowIds.has(id.toLowerCase())) return null;
                        const info = recommendedMeta.get(id);
                        const isG = isKnownGgufRepo(id);
                        const optionKey = makeModelOptionKey("recommended", id);
                        return (
                          <div key={id}>
                            {renderHubModelRow(
                              id,
                              <ModelRow
                                label={curatedRow(id).name}
                                tags={curatedRow(id).tags}
                                hubUrl={hubRepoUrl(id)}
                                alignMeta="hub"
                                showSize={hubRowsShowSize}
                                // Without the owner a community row reads as an unsloth upload.
                                hideOwner={isUnslothOwned(id)}
                                downloaded={cachedIdFor(id) !== null}
                                partial={isPartialRow(id)}
                                partialResumable={partialResumableSet.has(
                                  id.toLowerCase(),
                                )}
                                capabilities={capsById.get(id)}
                                meta={
                                  info?.meta ??
                                  (isG ? "GGUF" : extractParamLabel(id))
                                }
                                selected={isValueRow(id)}
                                loaded={isRuntimeLoadedModel(
                                  loadedModelId,
                                  activeGgufVariant,
                                  id,
                                  isG ? "required" : "none",
                                  aliasesOf(id),
                                )}
                                optionProps={hubModelList.getOptionProps(
                                  optionKey,
                                  isValueRow(id),
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
                                vramBudget={info?.budget}
                                gpuGb={isG ? expanderGpuGb : expanderSystemGpuGb}
                                onArrowDownIntoChildren={
                                  expandedGguf === id
                                    ? () => focusFirstChildOption(optionKey)
                                    : undefined
                                }
                              />,
                            )}
                            {expandedGguf === id && (
                              <GgufVariantExpander
                                diffusionLoad={diffusionLoad}
                                hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                gpuCount={expanderGpuCount}
                                repoId={id}
                                loadedQuants={loadedQuantsFor(id)}
                                pipelineTag={pipelineTagById.get(id) ?? null}
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
                                systemRamGb={expanderRamGb || undefined}
                                budgetKnown={expanderBudgetGpu.budgetKnown}
                                variantActions={{
                                  onDelete: async (quant, cachePath) => {
                                    await deleteCachedModel(
                                      id,
                                      quant,
                                      hfToken || undefined,
                                      cachePath ?? undefined,
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
                    {recommendedHasMore && (
                      <>
                        <div ref={recommendedSentinelRef} className="h-px" />
                        {recommendedIsLoadingMore ? (
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
                          {renderHubModelRow(
                            id,
                            <ModelRow
                              label={curatedRow(id).name}
                              tags={curatedRow(id).tags}
                              hubUrl={hubRepoUrl(id)}
                              alignMeta="hub"
                              showSize={hubRowsShowSize}
                              downloaded={cachedIdFor(id) !== null}
                              partial={isPartialRow(id)}
                              partialResumable={partialResumableSet.has(
                                id.toLowerCase(),
                              )}
                              capabilities={capsById.get(id)}
                              meta={
                                isKnownGgufRepo(id)
                                  ? (recommendedMeta.get(id)?.meta ?? "GGUF")
                                  : (vram?.detail ?? extractParamLabel(id))
                              }
                              selected={isValueRow(id)}
                              loaded={isRuntimeLoadedModel(
                                loadedModelId,
                                activeGgufVariant,
                                id,
                                isKnownGgufRepo(id) ? "required" : "none",
                                aliasesOf(id),
                              )}
                              optionProps={hubModelList.getOptionProps(
                                optionKey,
                                isValueRow(id),
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
                              vramBudget={
                                isKnownGgufRepo(id) ? undefined : vram?.budget
                              }
                              gpuGb={
                                isKnownGgufRepo(id)
                                  ? expanderGpuGb
                                  : expanderSystemGpuGb
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
                            />,
                          )}
                          {expandedGguf === id && (
                            <GgufVariantExpander
                              diffusionLoad={diffusionLoad}
                              hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                              gpuCount={expanderGpuCount}
                              repoId={id}
                              loadedQuants={loadedQuantsFor(id)}
                              pipelineTag={pipelineTagById.get(id) ?? null}
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
                              systemRamGb={expanderRamGb || undefined}
                              budgetKnown={expanderBudgetGpu.budgetKnown}
                              variantActions={{
                                onDelete: async (quant, cachePath) => {
                                  await deleteCachedModel(
                                    id,
                                    quant,
                                    hfToken || undefined,
                                    cachePath ?? undefined,
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

                {showHfSection &&
                section === "recommended" &&
                formatFilter !== "npu" ? (
                  <>
                    {searchRowIds.length === 0 && !isLoading ? (
                      filteredRecommendedIds.length > 0 ? null : searchEmpty === "failed" ? (
                        <HubFailureHint message={searchError} onRetry={retrySearch} />
                      ) : (
                        <div className="px-2.5 py-2 text-xs text-muted-foreground">
                          {communityDiscoveryEnabled
                            ? "No matching models."
                            : "No matching Unsloth models."}
                        </div>
                      )
                    ) : (
                      searchRowIds.map((id) => {
                        const vram = vramMap.get(id);
                        const isSearchGguf = isKnownGgufRepo(id);
                        const optionKey = makeModelOptionKey("search-hf", id);
                        return (
                          <div key={id}>
                            {renderHubModelRow(
                              id,
                              <ModelRow
                                label={curatedRow(id).name}
                                tags={curatedRow(id).tags}
                                hubUrl={hubRepoUrl(id)}
                                alignMeta="hub"
                                showSize={hubRowsShowSize}
                                partial={isPartialRow(id)}
                                partialResumable={partialResumableSet.has(
                                  id.toLowerCase(),
                                )}
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
                                selected={isValueRow(id)}
                                loaded={isRuntimeLoadedModel(
                                  loadedModelId,
                                  activeGgufVariant,
                                  id,
                                  isSearchGguf ? "required" : "none",
                                  aliasesOf(id),
                                )}
                                optionProps={hubModelList.getOptionProps(
                                  optionKey,
                                  isValueRow(id),
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
                                  isSearchGguf
                                    ? expanderGpuGb
                                    : expanderSystemGpuGb
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
                              />,
                            )}
                            {expandedGguf === id && (
                              <GgufVariantExpander
                                diffusionLoad={diffusionLoad}
                                hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                gpuCount={expanderGpuCount}
                                repoId={id}
                                loadedQuants={loadedQuantsFor(id)}
                                pipelineTag={pipelineTagById.get(id) ?? null}
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
                                systemRamGb={expanderRamGb || undefined}
                                budgetKnown={expanderBudgetGpu.budgetKnown}
                                variantActions={{
                                  onDelete: async (quant, cachePath) => {
                                    await deleteCachedModel(
                                      id,
                                      quant,
                                      hfToken || undefined,
                                      cachePath ?? undefined,
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
                    {searchIsLoadingMore ? (
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
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-end pr-3.5 pb-[calc(19px*var(--ui-space-scale,1))]">
            <button
              type="button"
              onClick={() => (ejectsAll ? onEjectAll?.() : onEject())}
              className="pointer-events-auto inline-flex items-center justify-center gap-2 rounded-md bg-popover px-3 py-2 text-ui-13 font-medium text-destructive shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] transition-colors hover:bg-[color-mix(in_srgb,var(--foreground)_8%,var(--popover))] dark:bg-sidebar-accent dark:shadow-none dark:hover:bg-[color-mix(in_srgb,var(--foreground)_8%,var(--sidebar-accent))]"
              title={ejectsAll ? "Eject all models" : "Eject model"}
            >
              <HugeiconsIcon icon={RemoveCircleIcon} className="size-3.5" />
              {ejectsAll ? "Eject all" : "Eject model"}
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
      {settingsModel ? (
        <ConnectedModelSettingsDialog
          open={true}
          onOpenChange={(next) => {
            if (!next) setSettingsModel(null);
          }}
          checkpointId={settingsModel.model.id}
          displayName={settingsModel.model.name}
          modelId={settingsModel.providerModelId}
          providerType={settingsModel.model.providerType}
          apiType={settingsModel.apiType}
          baseUrl={settingsModel.baseUrl}
          isReasoningProvider={settingsModel.isReasoningProvider}
          reasoningConfig={settingsModel.reasoningConfig}
          connectionMaxOutputTokens={settingsModel.connectionMaxOutputTokens}
        />
      ) : null}
      {infoModel ? (
        <ConnectedModelInfoDialog
          open={true}
          onOpenChange={(next) => {
            if (!next) setInfoModel(null);
          }}
          modelId={infoModel.providerModelId}
          displayName={infoModel.model.name}
          providerName={infoModel.model.providerName}
          providerType={infoModel.model.providerType}
          apiType={infoModel.apiType}
          baseUrl={infoModel.baseUrl}
          isReasoningProvider={infoModel.isReasoningProvider}
          reasoningConfig={infoModel.reasoningConfig}
        />
      ) : null}
    </CapabilityScope.Provider>
  );
}

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
    /** GPUs memoryTotalGb sums, for the loader's per-card VRAM reserve. */
    deviceCount?: number;
    systemRamAvailableGb: number;
  };
}) {
  const pinnedKeys = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const unpinRepo = usePinnedModelsStore((s) => s.unpinRepo);
  const replacePinned = usePinnedModelsStore((s) => s.replacePinned);
  // A GGUF export's id is its first file, so deleting a variant can change or end it.
  const repinExportedGguf = async (oldId: string) => {
    const folder = oldId.slice(0, Math.max(oldId.lastIndexOf("/"), oldId.lastIndexOf("\\")));
    const { loras } = await listLoras();
    const next = loras.find(
      (lora) =>
        lora.source === "exported" &&
        lora.export_type === "gguf" &&
        (lora.adapter_path.startsWith(`${folder}/`) ||
          lora.adapter_path.startsWith(`${folder}\\`)),
    );
    if (!next) unpinRepo(oldId);
    else replacePinned(pinKey(oldId), pinKey(next.adapter_path));
  };
  return (
    <>
      {adapters.map((adapter) => {
        const isLocal = adapter.source === "local";
        const isTraining = adapter.source === "training";
        const isExported = adapter.source === "exported";
        const isMerged = adapter.exportType === "merged";
        const isGguf = adapter.exportType === "gguf";
        const isLora = !isLocal && !isMerged && !isGguf;
        const isExportedGguf = isExported && isGguf;
        const canDelete = canDeleteLoraModel(adapter);
        const isTrainingFull = isTraining && isMerged;
        const localGgufKind = localGgufKindFor(
          adapter,
          isGgufRepo(adapter.id) || isGgufRepo(adapter.name),
        );
        const isLocalGgufDir = localGgufKind === "variants";
        const isLocalDirectGguf = localGgufKind === "direct";
        // Fine-tuned TTS/STT checkpoints must reach the Audio page, so carry the codec as a pipeline tag.
        const selectionMeta: ModelSelectorChangeMeta = {
          source: isLocal ? "local" : isExported ? "exported" : "lora",
          isLora,
          isDownloaded: true,
          isGguf: isLocalDirectGguf,
          pipelineTag: audioPipelineTagFor(adapter.audioType, true, isLora),
          audioType: adapter.audioType ?? null,
        };
        const canConfigure = !(isLocalGgufDir || isExportedGguf);
        const optionKey = makeModelOptionKey("lora", adapter.id);
        const tag = isLocal
          ? isLocalGgufDir || isLocalDirectGguf
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
          ? isLocalGgufDir || isLocalDirectGguf
            ? "GGUF"
            : "Local"
          : isTrainingFull
            ? "Full finetune"
            : isExported
              ? `${tag} · Exported`
              : tag;
        return (
          <div key={adapter.id}>
            <div
              className={downloadedRowShellClassName(value === adapter.id)}
              // The pill a Pinned drag lifts (use-pinned-row-drag.ts).
              data-pinned-row-face=""
            >
              <div className="min-w-0 flex-1">
                <ModelRow
                  label={adapter.name}
                  meta={
                    adapter.sizeBytes
                      ? `${meta} · ${formatBytes(adapter.sizeBytes)}`
                      : meta
                  }
                  selected={value === adapter.id}
                  loaded={isRuntimeLoadedModel(
                    loadedModelId,
                    activeGgufVariant,
                    adapter.id,
                    isLocalDirectGguf
                      ? "ignore"
                      : isLocalGgufDir || isExportedGguf
                        ? "required"
                        : "none",
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
                  className={downloadedRowButtonClassName}
                />
              </div>
              <span className={ROW_ACTIONS_CLASS}>
                {canConfigure && onConfigure && (
                  <ModelLoadSettingsAction
                    ariaLabel={`Inference settings for ${adapter.name}`}
                    onConfigure={() => onConfigure(adapter.id, selectionMeta)}
                  />
                )}
                <ModelRowMenu
                  ariaLabel={`More options for ${adapter.name}`}
                  pin={{
                    pinned: pinnedKeys.includes(pinKey(adapter.id)),
                    pinLabel: "Pin",
                    unpinLabel: "Unpin",
                    onToggle: () => togglePinned(adapter.id),
                  }}
                  onReveal={
                    isLocal
                      ? undefined
                      : () =>
                          revealFineTunedModel(
                            adapter.id,
                            isExported ? "exported" : "training",
                          )
                  }
                  del={
                    canDelete
                      ? {
                          title: "Delete fine-tuned model?",
                          description: (
                            <>
                              This will remove{" "}
                              <span className="font-medium text-foreground">
                                {adapter.name}
                              </span>{" "}
                              from disk. This cannot be undone.
                            </>
                          ),
                          successMessage: `Deleted ${adapter.name}`,
                          disabled: deleteDisabled,
                          onConfirm: async () => {
                            await deleteFineTunedModel({
                              modelPath: adapter.id,
                              source: isExported ? "exported" : "training",
                              exportType: adapter.exportType,
                            });
                            unpinRepo(adapter.id);
                          },
                          onDeleted: () => onModelsChange?.({ id: adapter.id }),
                        }
                      : undefined
                  }
                />
              </span>
            </div>
            {expandedGguf === adapter.id && (
              <GgufVariantExpander
                repoId={adapter.id}
                loadedQuants={
                  activeGgufVariant && modelIdsMatchForPicker(loadedModelId, adapter.id)
                    ? [activeGgufVariant]
                    : undefined
                }
                onSelect={onSelect}
                onConfigure={onConfigure}
                parentOptionKey={optionKey}
                onNavigatePastStart={() => loraModelList.focusOption(optionKey)}
                onNavigatePastEnd={() =>
                  loraModelList.moveFocus(optionKey, "next")
                }
                gpuGb={gpu.available ? gpu.memoryTotalGb : undefined}
                gpuCount={gpu.deviceCount}
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
                        if (pinnedKeys.includes(pinKey(adapter.id))) {
                          await repinExportedGguf(adapter.id).catch(() => unpinRepo(adapter.id));
                        }
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
