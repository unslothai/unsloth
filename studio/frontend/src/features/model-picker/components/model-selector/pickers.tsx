// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import {
  classifyUnslothSupport,
  downloadManager,
  hfApiToken,
  isHiddenModelId,
  jobKeyOf,
  partialSetFromRows,
  scanFolderStatusCopy,
  useDownloadManagerStore,
  useHfTokenStore,
  useOnlineStatus,
} from "@/features/hub";
import type { HfTaskFilter } from "@/features/hub/hooks/use-hub-model-search";
import {
  useDebouncedValue,
  useGpuInfo,
  useHostClass,
  useInferenceGpuInfo,
} from "@/hooks";
import {
  type ModelMemorySource,
  useModelMemory,
} from "@/hooks/use-model-memory";
import { useVramBudgetFraction } from "@/hooks/use-vram-budget-fraction";
import { diffusionRouteSearch } from "@/lib/diffusion-route-search";
import { type GgufFitClass, requiredGgufMemoryGb } from "@/lib/gguf-fit";
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
  Flag01Icon,
  FlimSlateIcon,
  Folder02Icon,
  HelpCircleIcon,
  InformationCircleIcon,
  Image03Icon,
  PinIcon,
  RemoveCircleIcon,
  Search01Icon,
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
} from "react";
import { useChatPickerInventory } from "../../inventory/use-chat-picker-inventory";
import {
  type CommunityModelPolicy,
  allowedHiddenModelIdMatches,
  audioPipelineTagFor,
  audioPickIsRoutable,
  localAudioRowIsUndecodableGguf,
  communityAudioRowIsRunnable,
  curatedAudioInventoryMatches,
  curatedAudioInventoryTask,
  filesystemRowsSupportedForTask,
  macTtsHubRowIsRunnable,
  nativeAudioCheckpointIsLoadable,
  shouldDiscoverCommunityModels,
  shouldRecommendCommunityModels,
  taskCatalogFormatMatches,
  taskForMediaPick,
  taskPickerRowMatches,
} from "./audio-picker-policy";
import { FolderBrowser } from "./folder-browser";
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
  curatedArtifactFitsDevice,
  curatedCapabilitiesFor,
  curatedRowLabelFor,
  curatedSizeBytesFor,
  curatedTotalParamsFor,
  groupForRepoId,
} from "./model-catalog";
import { curatedArtifactIsOfferable } from "./host-artifact-policy";
import { localGgufKindFor } from "./local-gguf-policy";
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
import { describeVariantListingError } from "./variant-listing-error";
import {
  ggufQuantChipLabel,
  ggufQuantDetailLabel,
  ggufVariantPickerLabel,
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

/** Official publisher namespaces used only for visual On Device grouping. */
function isUnslothPublisherRepoId(repoId: string): boolean {
  return isUnslothOwner(splitRepoLabel(repoId).owner);
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
  /** Draw a divider line above to separate it from the section above (omit on the first section). */
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
  // Decimal (base-1000) units to match Hugging Face's reported sizes; the GPU-fit math stays
  // base-1024. Divide iteratively, not via Math.log, which is off at exact powers of 1000.
  // Divide iteratively rather than via Math.log, which has float error at exact powers of 1000
  // (mislabeling 1 TB as "1000 GB") and could run off the end of units.
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

// Most distinguishing first, since only the first MAX_CAPABILITY_BADGES are drawn: what a
// model GENERATES separates it from the list, reasoning very little.
const CAPABILITY_BADGES: {
  key: keyof ModelCapabilities;
  title: string;
  Glyph: (props: { className: string }) => ReactNode;
}[] = [
  {
    key: "videoGen",
    title: "Generates video",
    Glyph: (props) => (
      <HugeiconsIcon icon={FlimSlateIcon} strokeWidth={1.8} {...props} />
    ),
  },
  {
    key: "imageGen",
    title: "Generates images",
    Glyph: (props) => (
      <HugeiconsIcon icon={Image03Icon} strokeWidth={1.8} {...props} />
    ),
  },
  {
    key: "audio",
    // Direction-neutral, unlike the two above: `audio` covers ASR and classification as well as
    // synthesis, so a Whisper row would claim to generate what it consumes.
    title: "Audio",
    Glyph: (props) => (
      <HugeiconsIcon icon={AudioWave01Icon} strokeWidth={1.8} {...props} />
    ),
  },
];

/** Which capability glyphs are worth drawing in the current picker; null draws them all. A
 *  media picker has already filtered to one kind, so its own kind is not information (Audio
 *  on Video is the exception). Context, not a prop, since it comes from the picker. */
const CapabilityScope = createContext<readonly (keyof ModelCapabilities)[] | null>(
  null,
);

// The row reserves a fixed slot for these (META_COLUMN.badge), so the cap is what keeps every column after it lined up.
const MAX_CAPABILITY_BADGES = 3;

/** The glyphs this row actually draws, so the caller can size the slot and skip an empty one. */
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
      {visibleCapabilityBadges(caps, scope).map(({ key, title, Glyph }) => (
        <span
          key={key}
          title={title}
          aria-label={title}
          className="flex size-[18px] shrink-0 items-center justify-center rounded-md border border-border/60 text-muted-foreground"
        >
          <Glyph className="size-3" />
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
    // h-[18px], the height every other chip in the row band already pins (quant, vision, the disk
    // mark, the Loaded tag). py-px left this one sized by its line box instead, which is the only
    // height here that scales with --ui-font-scale: at 1.0 it stood 1px PROUDER than the quant and
    // vision chips beside it and at 0.8125 it sat 1.8px shorter, so the row only looked level at
    // the one scale where the two happened to cross. A shared height is level at every scale.
    <span className="inline-flex h-[18px] shrink-0 items-center whitespace-nowrap rounded-md border border-border/60 px-1.5 text-ui-10 font-medium text-muted-foreground tabular-nums">
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

/** Format as a coloured dot ahead of the name, named on hover: a word like "Safetensors" is
 *  wide enough to shove the rest of the row around. */
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

/** "Already on disk", shown on Hub rows that are also downloaded: their own download arrow
 *  reads as "click to fetch" on the one row that needs no fetching. */
function DownloadedBadge() {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label="On device"
          className="flex h-[18px] w-[14px] shrink-0 items-center justify-center"
        >
          <span
            aria-hidden="true"
            className="size-[5px] rounded-full bg-status-success"
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        On device
      </TooltipContent>
    </Tooltip>
  );
}

/** A cancelled or interrupted download: some bytes on disk, not enough to load. The Hub marks
 *  these with the same warning dot, and the row it sits on selects with isDownloaded: false so
 *  the click opens the download instead of handing incomplete weights to the runtime. Same box
 *  as DownloadedBadge, since the two are alternatives -- a row is complete or it is not. */
function PartialBadge({ resumable }: { resumable?: boolean }) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          aria-label="Partial download"
          className="flex h-[18px] w-[14px] shrink-0 items-center justify-center"
        >
          <span
            aria-hidden="true"
            className="size-[5px] rounded-full bg-status-warning"
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

/** Over budget but still card-sized. `_select_gpus` scores against FREE VRAM, so whether this
 *  lands on the GPU depends on what else is resident; missing it costs speed, not the load. */
/** Over the VRAM Budget, still smaller than the card. Not conditional: `_vram_usable_mib`
 *  gives `free - reserve`, which on an idle card IS the budget this tier passed, so
 *  `_select_gpus` hands it to --fit every time. Raising the budget is the lever. */
const MIGHT_FIT: FitVerdict = {
  label: "Over budget",
  tone: AMBER,
  hint: "Larger than your VRAM Budget allows, so part of it offloads even on an idle GPU. It is still smaller than the card, so raising the budget can keep it resident.",
};
/** Past the card. Every GGUF over budget lands here however far over, since llama-server never
 *  refuses one on size: `_select_gpus` returns `(None, use_fit=True)` and --fit offloads. */
const OFFLOADS: FitVerdict = {
  label: "Does not fit",
  tone: ORANGE,
  hint: "Model may not fit but still works with offloading. Expect slower inference.",
};
/** checkVramFit's 75-100% band, the ONLY source of `tight` here: a torch estimate that still
 *  fits on the card entirely, with no --fit, so it neither spills nor offloads. */
const DEVICE_TIGHT: FitVerdict = {
  label: "Tight fit",
  tone: AMBER,
  hint: "Uses nearly all your VRAM, with little headroom for anything else.",
};
/** A torch load, which has no --fit to fall back on: the pipeline goes wholly on the device. */
const WONT_FIT: FitVerdict = {
  label: "Does not fit",
  tone: ORANGE,
  // "memory", not "VRAM": this also carries a diffusion refusal on a shared pool, where the two are the same bytes.
  hint: "Needs more memory than this device has. This model will not load.",
};

/** What each fit verdict marks and says, keyed by the Hub's classes so one question has one
 *  vocabulary. `tight` and `exceeds` are the training estimator's words and reach here only
 *  from a torch pipeline or the QLoRA estimate, which have no --fit and never offload. */
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

/** Whether a diffusion `oom` is a REFUSAL rather than an offload. On discrete VRAM an oversized
 *  pipeline still streams from host RAM; on a shared pool the loader refuses up front, since
 *  the MPS high-watermark is disabled and the OS kills the process with no exception.
 *  `hostPooled` is the LOAD DEVICE's answer and folds unified_memory in, since hardware.py
 *  sets shared_memory only on Windows while diffusion_memory.py still refuses on Linux APUs. */
function diffusionRefuses(
  fit: GgufFitClass,
  diffusionLoad: boolean,
  hostPooled: boolean,
): boolean {
  return fit === "oom" && diffusionLoad && hostPooled;
}

/** The RAM a DIFFUSION verdict may add to the GPU budget. Zero on a host pool: offload there
 *  moves bytes inside one pool and frees nothing. llama.cpp differs, since a GGUF really
 *  does spill into host RAM the GPU window does not cover. */
function mediaRamBudgetGb(systemRamGb: number, hostPooled: boolean): number {
  return hostPooled ? 0 : systemRamGb;
}

/** The verdicts that read as over budget, which is what dims a row. `marginal` does not: it is
 *  a full GPU load with little room to spare. */
function isOverBudget(status?: GgufFitClass | VramFitStatus | null): boolean {
  return (
    status === "partial" ||
    status === "ram" ||
    status === "oom" ||
    status === "exceeds"
  );
}

/** VRAM verdict: an info mark that names itself on hover, rather than a shouted pill. */
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
        {/* The mark sits inside the row/variant button, so a tap meant to read the explanation would
            otherwise select the model or start its download. */}
        {/* biome-ignore lint/a11y/useKeyWithClickEvents: the handler suppresses the enclosing
            button, it does not add an interaction of its own */}
        <span
          aria-label={verdict.label}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
          }}
          className={cn(
            "flex size-[18px] shrink-0 items-center justify-center",
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

/** Keep the row's size treatment consistent with every other model; diffusion GGUFs get one
 *  small explanation affordance, since their checkpoint is only part of what is kept on disk. */
export function GgufDownloadFootprint({
  checkpointBytes,
  companionBytes,
}: {
  checkpointBytes: number;
  companionBytes: number;
}) {
  const totalBytes = checkpointBytes + companionBytes;
  // Whole-GB rounding is too lossy for a sum ("2.6 GB + 8.2 GB = 11 GB" looks contradictory),
  // so keep one decimal through GB/TB.
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

// Shared row columns, so meta lines up down the list. Widths are em of the slot's own text, so
// they follow the UI font scale; min-w-min holds a width open rather than clamping.
/** A GPU inventory in the shape the catalog's fit rules take. */
function artifactBudget(gpu: {
  memoryTotalGb: number;
  systemRamAvailableGb: number;
}): DeviceBudget {
  return { gpuGb: gpu.memoryTotalGb, systemRamGb: gpu.systemRamAvailableGb };
}

const META_COLUMN = {
  // Fits "UD-Q4_K_XL"; a hard cap, so longer quants clip.
  quant: "min-[560px]:w-[7.2em]",
  // Each width below is the widest set its scope can draw: anything wider makes min-w-min expand
  // the slot and shift every column after it.
  // The slot holds capability glyphs (18px), the vision badge (24px) and the "on disk" mark (14px),
  // gap-1 between them. Scope draws no glyph: the vision badge alone, or the disk mark alone.
  badge: "min-w-min min-[560px]:w-[24px]",
  // One glyph plus the disk mark (18 + 4 + 14).
  badgeMid: "min-w-min min-[560px]:w-[36px]",
  // On Device draws the vision badge (26px) and, since partials are listed, the partial mark
  // (14px) beside it. 44px is that pair with its gap: reserving only the badge let a row drawing
  // both grow past the slot and carry its quant chip 18px left of every other row.
  badgeDevice: "min-w-min min-[560px]:w-[44px]",
  // Hub draws the disk mark and no vision badge (18+4+14). A second glyph grows it via min-w-min.
  badgeWide: "min-w-min min-[560px]:w-[36px]",
  // The fit mark (Hub rows), one 18px glyph.
  vram: "min-w-min min-[560px]:w-[18px]",
  // Device rows reserve the slot rather than hug the chip. This is the last variable column on the
  // right, so hugging it let each row's meta cluster set its own width: a "1B" row, and more so a
  // row with no param at all, gave its name group the leftover and carried its quant chip that much
  // further right, leaving the quant column ragged down the list. 4.4em is the widest these lists
  // draw, measured at text-ui-10 -- "235B" is 38.4px, a 5-char "0.35B" 40.9px -- so nothing routine
  // trips min-w-min and shifts the row back out of line. The slack now sits in front of the chip,
  // as its gap to the modality mark. Hub keeps its own width for "2779.5B".
  param: "min-w-min min-[560px]:w-[4.4em]",
  paramWide: "min-w-min min-[560px]:w-[5.2em]",
  // formatBytes writes no space ("536MB"), so the widest this holds is 29.5px, not the ~40px a spaced "536 MB" needs.
  size: "min-w-min min-[560px]:w-[3.2em]",
  // The format dot that leads the row; the name lives in its tooltip.
  format: "min-[560px]:w-[14px]",
} as const;

// One gutter for every row, gear or no gear, so the columns never shift by a button; the
// buttons show on hover or while their menu is open.
const ROW_ACTIONS_CLASS =
  "mr-0.5 flex w-[38px] shrink-0 items-center justify-end -space-x-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100 has-[[data-state=open]]:opacity-100 [@media(hover:none)]:opacity-100";

// Partial rows keep their buttons on screen. Everywhere else the gutter hides until the row is
// hovered because the row itself is the action -- click it and the model loads, so the menu is a
// secondary path. A partial cannot be loaded at all: the menu IS the row's only affordance, and
// hiding the one control that reaches a stalled multi-GB download behind a hover reads as the
// download having no controls at all.
const ROW_ACTIONS_PINNED_CLASS = cn(ROW_ACTIONS_CLASS, "opacity-100");

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
  gpuGb?: number;
  tooltipText?: ReactNode;
  /** Hugging Face address for online/Hub rows, surfaced on hover the way local rows show an
   *  on-disk path. Omit to show no address line. */
  hubUrl?: string;
  optionProps?: ModelRowOptionProps;
  onArrowDownIntoChildren?: () => boolean;
  /** Capability override (HF rows have tags); falls back to name detection. */
  capabilities?: ModelCapabilities;
  /** Hide the "owner/" prefix (e.g. Recommended, where all are unsloth). */
  hideOwner?: boolean;
  /** Mark a row already on disk (shown in Recommended instead of being hidden). */
  downloaded?: boolean;
  /** Mark a row whose snapshot is incomplete. Mutually exclusive with `downloaded`: the bytes
   *  are there or they are not, and the caller routes the click to the download either way. */
  partial?: boolean;
  /** Whether that partial continues byte for byte. Undefined reads as "no", which is what keeps
   *  the mark from promising a resume the transport cannot deliver. */
  partialResumable?: boolean;
  /** Show a Vision badge on the name (On Device, read from GGUF metadata). */
  showVision?: boolean;
  /** Grey chip beside the name, for rows that load one specific quant. */
  quantChip?: string | null;
  /** Chips for what used to sit in brackets after the name: the artifact format, and a
   *  resolution when variants differ only by it. */
  tags?: string[];
  /** Column layout (see META_COLUMN): "device" reserves the quant chip, "hub" the download and VRAM badges. */
  alignMeta?: "device" | "hub";
  /** Hold the size column open. Hub rows pass this on the MLX and Safetensors filters, where a
   *  repo is one download with one size. */
  showSize?: boolean;
  /** Identifies the on-disk model whose VRAM split the row should chart; omit for rows that are not downloaded. */
  memory?: ModelMemorySource;
  className?: string;
}) {
  const exceeds = isOverBudget(vramStatus);
  const showVramTooltip =
    vramEst != null && vramEst > 0 && gpuGb != null && gpuGb > 0;
  const vramTooltipText =
    showVramTooltip && vramStatus
      ? exceeds
        // "memory", not "VRAM": a GGUF at `partial` splits across VRAM and RAM and the figure
        // is weights plus activations plus KV, so "Needs ~47GB VRAM" contradicted the verdict.
        ?
          `Needs ~${vramEst}GB memory (GPU: ${gpuGb}GB)`
        : vramStatus === "tight" || vramStatus === "marginal"
          ? `~${vramEst}GB VRAM (tight fit on ${gpuGb}GB)`
          : `~${vramEst}GB VRAM`
      : null;

  const { owner, name } = splitRepoLabel(label);
  // Drop our own owner: the list is nearly all unsloth/, so it is noise, and other owners still
  // showing is what tells the two apart.
  const showOwner = !!owner && !hideOwner && !isUnslothOwner(owner);
  const parsed = parseMetaTokens(meta);
  // Param chip from meta, else derived from the name so GGUF rows show it too.
  const paramLabel = parsed.param ?? extractParamLabel(name) ?? null;
  // Use the passed-in capabilities (tag-aware) or infer from the repo name.
  const caps = capabilities ?? detectCapabilities({ id: label });
  const capabilityScope = useContext(CapabilityScope);
  const capabilityBadges = visibleCapabilityBadges(caps, capabilityScope);
  const showCaps = capabilityBadges.length > 0;
  const aligned = alignMeta !== undefined;
  // Reserve only what this picker's scope can draw: Images and Audio draw no glyph and Video
  // draws one, so holding the chat row's slot open there is dead space.
  const badgeColumn =
    capabilityScope === null || capabilityScope.length > 1
      ? alignMeta === "device"
        ? META_COLUMN.badgeDevice
        : META_COLUMN.badgeWide
      : capabilityScope.length === 1
        ? META_COLUMN.badgeMid
        : META_COLUMN.badge;
  // One dot per row: a second format shares the first's colour anyway, so it rides along in the tooltip.
  const formatDot = parsed.formats[0]
    ? {
        tone: parsed.formats[0].tone,
        label: parsed.formats.map((f) => f.label).join(" · "),
      }
    : null;

  // Only the selected row charts itself: a meter under every row turns a list you scan into a wall of charts.
  const memorySegments = useModelMemory(selected ? memory : undefined, gpuGb);
  const showMemoryBar = memorySegments.status !== "unknown";

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
        // pl-[5.5px]: the dot is centred in a 14px hover target, so 5.5 + (14 - 5) / 2 lands it on
        // 10px, level with the section labels at px-2.5.
        "group/row flex w-full flex-col items-stretch py-1.5 pl-[5.5px] pr-2 text-left text-sm transition-colors hover:bg-[#ececec] focus-visible:bg-[#ececec] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:bg-[var(--sidebar-accent)] dark:focus-visible:bg-[var(--sidebar-accent)]",
        showMemoryBar ? "rounded-2xl" : "rounded-full",
        selected && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
        className,
      )}
    >
      {/* gap-1: the quant chip ends the name group, so what this separates is that chip from the
          first meta mark, on the rhythm the meta columns keep. */}
      <span
        className={cn(
          "flex w-full items-center gap-1",
          // Over budget reads as a dimmed row, which scans; hover restores it. The selected row keeps full weight.
          exceeds &&
            !selected &&
            "opacity-60 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100",
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
          {alignMeta !== "device" && quantChip ? (
            <span className="ml-2 shrink-0 rounded-md bg-black/[0.06] px-1.5 py-px font-mono text-ui-10 text-muted-foreground dark:bg-white/[0.1]">
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
            aligned ? "gap-1" : "gap-1.5",
          )}
        >
          {/* The quant chip sits in the meta cluster, not at the end of the name, so one
              items-center rule lines it up with the vision mark, the parameter chip, the size and
              the row's buttons. Inside the name group it was centred against THAT box instead --
              a baseline box sized by the name's own line height -- so it only agreed with the rest
              of the row for as long as the two boxes happened to share a centre. */}
          {alignMeta === "device" ? (
            <span
              className={cn(
                // justify-end: the slot is sized for the longest quant, so left-aligning ended a
                // "Q8_0" and a "UD-Q4_K_XL" at different x even once the slot itself stopped
                // moving. Flush right is what makes the chips read as one column.
                "flex shrink-0 items-center justify-end text-ui-9",
                META_COLUMN.quant,
              )}
            >
              {quantChip ? <QuantChip label={quantChip} /> : null}
            </span>
          ) : null}
          {/* Capabilities, vision and the Hub lists' "on disk" mark share one
              column; two of them widen the slot rather than overlap. */}
          {aligned ? (
            <span
              className={cn(
                // Right, not centre: slack belongs to the name, not split either side of a glyph.
                "flex shrink-0 items-center justify-end gap-1 text-ui-10",
                badgeColumn,
              )}
            >
              {showCaps && <CapabilityIcons caps={caps} />}
              {showVision && <VisionBadge />}
              {partial ? <PartialBadge resumable={partialResumable} /> : null}
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
                  className="h-[18px] gap-1 rounded-md px-1.5"
                  dotClassName="size-[5px]"
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
                // Device leads the chip, Hub trails it. Both columns are fixed, so the choice is
                // only where the column's slack falls: trailing it put the slack in FRONT of the
                // chip, where it read as part of the gap to the modality mark and grew or shrank
                // with the label -- 6.9px after a "217B", 19px after a "1B". Leading the chip
                // leaves that gap as the cluster's own gap-1, the same 4px the quant chip keeps
                // to the same mark, and the slack falls back toward the size instead.
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
          {/* GGUF repos hold several quants of different sizes, so their rows report one only once expanded. */}
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

  // Optional Hugging Face address line for online/Hub rows, rendered under whichever tooltip shows.
  const hubUrlLine = hubUrl ? (
    <span className="block mt-1 text-ui-10 text-muted-foreground break-all">
      {hubUrl}
    </span>
  ) : null;

  // The dot names its format on hover only, which keyboard focus never reaches, so the row tooltip carries it too.
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
    (candidate.shard_count === undefined ||
      (Number.isSafeInteger(candidate.shard_count) &&
        candidate.shard_count >= 0)) &&
    (candidate.downloaded === undefined ||
      typeof candidate.downloaded === "boolean") &&
    // Carried through so each row can look up its own dependency group's footprint. Absent on an
    // older backend, which groups the repo as one, so it must never reject the row.
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
    // The backend's own verdict, which resolves existence-first: a marker-less relative name that
    // exists on disk is a local model. A server predating the field leaves the prefix test.
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

/** The one quant a repo holds, plus the vision flag read with it. The collapsed row never
 *  mounts the expander, so this is its only source. */
interface SoleDownloadedQuant {
  variant: GgufVariantDetail;
  hasVision: boolean;
}

/** The repo's one complete quant, or null when it holds none, holds several, or could not be
 *  read. Disk-only and client-cached. */
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
    // One file on disk and nothing torn beside it; a partial quant keeps the expander, where it can be resumed.
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
// Reads run a few at a time so a large cache does not fire one request per repo. A worker
// pool, not fixed batches: one slow repo holds up only itself.
const SOLE_QUANT_WORKERS = 6;

/** On Device repos holding exactly one quant on disk, keyed by repo id: with "Show all
 *  quantizations" off those collapse into one pinned-style row. Kept per repo, so one
 *  repo's download or delete leaves the others alone. */
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

  const [entries, setEntries] = useState<
    ReadonlyMap<string, SoleQuantEntry<SoleDownloadedQuant>>
  >(EMPTY_SOLE_QUANT_ENTRIES);
  const { quants, pending, stale } = useMemo(
    () => partitionSoleQuants(targets, entries, { enabled }),
    [targets, entries, enabled],
  );

  // A change outside this tab moves the row's bytes without touching this instance's variants
  // cache, so drop that repo's cached listing.
  const fingerprintsRef = useRef(new Map<string, string>());
  useEffect(() => {
    for (const repoId of takeDriftedRepos(targets, fingerprintsRef.current)) {
      invalidateGgufVariantsCache(repoId);
    }
  }, [targets]);

  // Reads outlive a render, so they run outside it; the token is read at call time, so a change
  // to it does not strand the reader.
  const hfTokenRef = useRef(hfToken);
  hfTokenRef.current = hfToken;
  const mountedRef = useRef(true);
  useEffect(() => {
    // Set on setup, not just cleared on teardown: StrictMode replays effects, and a ref left false
    // would discard every later read.
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
}: {
  repoId: string;
  pipelineTag?: string | null;
  /** True on Images / Video, where a GGUF is placed by the diffusion backend rather than
   *  llama-server, so the llama.cpp budget does not apply. Audio is task-scoped but not this. */
  diffusionLoad?: boolean;
  /** The LOAD DEVICE's memory is a window into host RAM (Apple Silicon, a ROCm APU). Only a
   *  diffusion load reads it: see `diffusionRefuses`. */
  hostPooledMemory?: boolean;
  /** How many GPUs gpuGb is the sum of, for the loader's per-card VRAM reserve. */
  gpuCount?: number;
  /** Snapshot the cached listing pinned this repo to, if any. */
  loadId?: string | null;
  /** Cache directory this downloaded row represents, if any. */
  cachePath?: string | null;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  /** HF token threaded into the variant fetch so private/gated repos resolve their GGUF variants. */
  hfToken?: string;
  parentOptionKey?: string;
  onNavigatePastStart?: () => void;
  onNavigatePastEnd?: () => void;
  onConfigure?: (id: string, meta: ModelSelectorChangeMeta) => void;
  sourceOverride?: ModelSelectorChangeMeta["source"];
  /** Update/delete actions for cached variant rows; omitted by browse-only expanders. */
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
  /** On Device rows honor the Show all quantizations setting; browse lists always show every quant. */
  onDevice?: boolean;
  /** Only managed cached-Hub rows can surface quant pins in the Pinned section; local-path
   *  expanders leave this false. */
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
    // Collapsing the row drops the request: a stalled one holds a per-host connection, and enough
    // of them stall download and load too.
    const controller = new AbortController();
    queueMicrotask(() => {
      if (canceled) return;
      setLoading(true);
      setError(null);
      // Belongs to the identifier being listed: carrying it over would apply the previous row's
      // locality to this one's footprint arithmetic.
      setResolvedLocally(false);
    });

    // The row's own directory, so disk contents count against that cache, not the active one. No
    // preferLocalCache: it answers from disk alone.
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

  // Covers Unix absolute, Windows drive, UNC, relative and tilde paths.
  const isLocalPath = /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(
    repoId,
  );
  // The prefix test cannot see a marker-less relative directory like "models/my-image-model",
  // so whether the checkpoint is on disk is asked of the listing, not of the spelling.
  const checkpointIsLocal = isLocalPath || resolvedLocally;

  const handleVariantClick = useCallback(
    // `filename` is required: the diffusion pages gate their GGUF branch on meta.ggufFilename, so
    // a quant label alone made every Images/Video GGUF pick a dead click.
    (
      quant: string,
      filename: string,
      downloaded?: boolean,
      sizeBytes?: number,
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
        contextLength: isAvailable ? nativeContext : undefined,
        isGguf: true,
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
    ],
  );

  // The user's saved VRAM Budget, which is what the loader admits against; the picker used to
  // ignore it, so moving the slider changed the Hub's verdicts only.
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  const anyBudgetGb = (gpuGb ?? 0) > 0 || (systemRamGb ?? 0) > 0;

  const getGgufFit = useCallback(
    (sizeBytes: number): GgufFitClass => {
      // Permissive only when no budget was measured: a known zero Vulkan budget means every
      // non-empty variant is OOM, which classifyGgufFit cannot tell from "not probed yet".
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

  const variantGroups = useMemo(
    () => groupGgufVariantsForPicker(variants ?? []),
    [variants],
  );
  const preferredByGroup = useMemo(
    () => preferredGgufVariantByGroup(variantGroups, defaultVariant),
    [variantGroups, defaultVariant],
  );

  // Each workflow gets its own recommendation: if its preferred variant is OOM use the largest
  // that can run, and if all are OOM the smallest.
  const effectiveRecommendedByGroup = useMemo(() => {
    const recommended = new Map<string, string>();
    for (const group of variantGroups) {
      const preferred = preferredByGroup.get(group.key) ?? null;
      if (!anyBudgetGb && !budgetKnown) {
        if (preferred) recommended.set(group.key, preferred.quant);
        continue;
      }
      if (preferred && getGgufFit(preferred.size_bytes) !== "oom") {
        recommended.set(group.key, preferred.quant);
        continue;
      }
      const fitting = group.variants
        .filter((variant) => getGgufFit(variant.size_bytes) !== "oom")
        .sort((left, right) => right.size_bytes - left.size_bytes);
      if (fitting[0]) {
        recommended.set(group.key, fitting[0].quant);
        continue;
      }
      const smallest = [...group.variants].sort(
        (left, right) => left.size_bytes - right.size_bytes,
      )[0];
      if (smallest) recommended.set(group.key, smallest.quant);
    }
    return recommended;
  }, [variantGroups, preferredByGroup, anyBudgetGb, budgetKnown, getGgufFit]);
  // `effectiveRecommendedByGroup` is keyed by PRESENTATION group while the footprint pass
  // buckets by the backend's dependency_key, so that pass asks through the variant itself.
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
      const f = getGgufFit(v.size_bytes);
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

        // Within the same tier, the workflow's recommendation goes first.
        const aIsRec = a.quant === recommended;
        const bIsRec = b.quant === recommended;
        if (aIsRec !== bIsRec) return aIsRec ? -1 : 1;

        // fits: largest first (best quality that fits); tight/OOM: smallest first (closest to fitting).
        const fitsInGpu = aTier === 0 || aTier === 2;
        return fitsInGpu
          ? b.size_bytes - a.size_bytes
          : a.size_bytes - b.size_bytes;
      });
    });
  }, [variants, variantGroups, effectiveRecommendedByGroup, getGgufFit]);

  // On Device only: with Show all quantizations off, list quants already on disk, torn ones included.
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
  const displayVariantGroups = useMemo(
    () => groupGgufVariantsForPicker(displayVariants ?? []),
    [displayVariants],
  );
  const hideH3PrunedBuild = useMemo(
    () => h3PickerHasOnlyPrunedBuilds(displayVariants ?? []),
    [displayVariants],
  );

  // A diffusion GGUF is not self-contained: the loader also needs a text encoder, VAE, tokenizer
  // and configs, and that companion set is NOT repository-wide (one repo can hold GGUFs of
  // different families). Both are folded into the backend's dependency_key, so grouping by it
  // keeps a non-representative row from advertising a GB-wrong total.
  const footprintVariants = useMemo(() => {
    const byKey = new Map<string, GgufVariantDetail>();
    for (const variant of displayVariants ?? []) {
      // An unkeyed repo (older backend, or no family resolved) collapses to one group, the previous repo-wide behavior.
      const key = variant.dependency_key ?? "";
      const current = byKey.get(key);
      if (current === undefined) {
        byKey.set(key, variant);
        continue;
      }
      // The recommended quant represents its own group when it has one, else the group's first row.
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
    // A local path is resolved too: only the CHECKPOINT is on disk, and its remote base is the
    // larger half, so suppressing the request understated a local row by many gigabytes.
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
          // A checkpoint already on disk is not part of required_bytes, so nothing may be subtracted for
          // it: subtracting drove the total to zero and hid a multi-GB companion set. Only a hub pick
          // carries its checkpoint inside the total.
          // expectedBytes stands in when the planner could not size it.
          const checkpoint = checkpointIsLocal
            ? 0
            : footprint.checkpointBytes > 0
              ? footprint.checkpointBytes
              : expectedBytes;
          const companion = footprint.requiredBytes - checkpoint;
          if (Number.isFinite(companion) && companion > 0) {
            // A fresh Map per resolution: React compares state by identity and the groups resolve
            // independently, so a mutation would drop the rows whose request landed first.
            setCompanionBytesByKey((previous) => {
              const next = new Map(previous);
              next.set(dependencyKey, companion);
              return next;
            });
          }
        })
        .catch(() => {
          // The checkpoint size stays useful when an older backend or a Hub failure cannot provide the
          // companion footprint.
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
      {/* On Device shows the model name above, so the Quantizations heading is redundant; its Vision
          badge is relayed to the name instead. */}
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
        // Its own group's pick. Matching on the quant alone works only because an H3 key is unique per
        // file, which is the backend's rule.
        const isRecommended =
          group != null &&
          effectiveRecommendedByGroup.get(group.key) === v.quant;
        const fit = getGgufFit(v.size_bytes);
        const oom = fit === "oom";
        const expectedBytes = ggufVariantExpectedBytes(v);
        // This row's own dependency group, never the listing's: see the footprintVariants comment above.
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
              <span className={cn(oom && "!text-gray-500 dark:!text-gray-400")}>
                {ggufVariantPickerLabel(v, {
                  h3Grouped: group?.title != null,
                  hideH3PrunedBuild,
                })}
              </span>
              {(v.shard_count ?? 0) > 1 ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-sky-700 dark:text-sky-300">
                  Sharded · {v.shard_count} parts
                </span>
              ) : null}
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
              ) : v.partial === true ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-amber-700 dark:text-amber-300">
                  partial
                </span>
              ) : isRecommended ? (
                <span className="ml-1.5 text-ui-9 font-sans font-medium text-primary/70">
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
                            // Drop the pin too: a pinned row for a deleted file loads something gone.
                            if (pinnedKeys.includes(pinKey(repoId, v.quant))) {
                              togglePinnedQuant(repoId, v.quant);
                            }
                            // Re-fetch this expander's variants so the deleted quant stops showing as
                            // downloaded while other cached quants remain.
                            // repo still has other cached quants.
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


// True when a repo's inferred task is within the picker's task filter; unknown task (null) passes only with no filter.
function taskMatchesFilter(
  repoTask: string | null | undefined,
  filter: HfTaskFilter,
): boolean {
  if (!filter) return true;
  const wanted = Array.isArray(filter) ? filter : [filter];
  return repoTask != null && (wanted as readonly string[]).includes(repoTask);
}

// Image-generation pipeline tasks: owned by the Images page, never chat-loadable. The backend
// reports "text-to-image" for diffusion-arch GGUFs.
export const IMAGE_GEN_TASKS = [
  "text-to-image",
  "image-to-image",
  "image-text-to-image",
] as const;

// Video-generation pipeline tasks: owned by the Video page, never chat-loadable. The backend
// reports "text-to-video" for video-diffusion GGUFs; HF gives the LTX-2 family the image-to-video
// pipeline_tag, and image-text-to-video is MiniMax-H3's. Without them the Video picker misses such
// a model and chat loads it as a language model.
export const VIDEO_GEN_TASKS = [
  "text-to-video",
  "image-to-video",
  "image-text-to-video",
] as const;

/** The tasks whose GGUFs are placed by the DIFFUSION backend, the only reason a picker scores
 *  against `classifyMediaGgufFit`. Audio is task-scoped but excluded: its GGUFs go to
 *  llama.cpp or the whisper sidecars, so scoring them at 70% hid runnable models. */
const DIFFUSION_TASKS: ReadonlySet<string> = new Set([
  ...IMAGE_GEN_TASKS,
  ...VIDEO_GEN_TASKS,
]);

// Speech pipeline tasks: owned by the Audio page. TTS picks load there; ASR picks map to the dictation sidecar.
export const AUDIO_GEN_TASKS = [
  "text-to-speech",
  "automatic-speech-recognition",
] as const;

// Diffusion GGUF archs the Images backend cannot assemble yet. The backend tags them with this
// task so both pickers leave them out; they would 400 on load.
const UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported";

// Generation tasks the Images / Video / Audio pages own. Not chat-loadable, so an on-device pick routes to its page.
const MEDIA_PAGE_TASKS: readonly string[] = [
  ...IMAGE_GEN_TASKS,
  ...VIDEO_GEN_TASKS,
  ...AUDIO_GEN_TASKS,
];

/** The page that runs this task, or null when chat should handle the pick. */
function mediaPageForTask(
  task: string | null | undefined,
): "images" | "video" | "audio" | null {
  if (!task || !MEDIA_PAGE_TASKS.includes(task)) return null;
  if ((VIDEO_GEN_TASKS as readonly string[]).includes(task)) return "video";
  if ((AUDIO_GEN_TASKS as readonly string[]).includes(task)) return "audio";
  return "images";
}

// Editing/inpaint checkpoints are tagged image-to-image but need an input image the
// text-to-image backend rejects, so they are hidden by id (mirrors _EDIT_KEYWORDS). The
// task itself must stay, since FLUX.2-klein carries it too.
const IMAGE_EDIT_KEYWORDS = ["edit", "kontext", "inpaint", "layered"] as const;
// Editing families the backend now SUPPORTS: not hidden despite the edit keyword. Mirrors the
// backend's qwen-image-edit family.
const SUPPORTED_EDIT_KEYWORDS = ["qwen-image-edit", "kontext"] as const;
// Match a keyword as a whole path/name segment, not a raw substring, so "edit" does not hide
// ".../edited/...". Keywords are [a-z-] literals, so no escaping. Mirrors _token_in_needle.
function idHasSegment(id: string, keyword: string): boolean {
  return new RegExp(`(?:^|[-_./\\\\])${keyword}(?:$|[-_./\\\\])`).test(id);
}
function isImageEditModel(repoId: string | null | undefined): boolean {
  if (!repoId) return false;
  const id = repoId.toLowerCase();
  if (SUPPORTED_EDIT_KEYWORDS.some((kw) => idHasSegment(id, kw))) return false;
  return IMAGE_EDIT_KEYWORDS.some((kw) => idHasSegment(id, kw));
}

// Gate an on-device model by the picker's task scope: with a filter keep only matching,
// non-editing tasks; with none drop image-generation models.
function passesTaskGate(
  repoTask: string | null | undefined,
  repoId: string | null | undefined,
  filter: HfTaskFilter,
  catalog?: CatalogGroup[],
  activeCatalogArtifactIds?: ReadonlySet<string>,
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
    );
  }
  // Unfiltered (chat) picker: an on-device diffusion model stays listed and routes to its page
  // on click; only the never-loadable tag is hidden.
  return repoTask !== UNSUPPORTED_DIFFUSION_TASK;
}

// Module-level caches so re-mounting the popover shows results instantly.
let _cachedGgufCache: CachedGgufRepo[] = [];
let _cachedModelsCache: CachedModelRepo[] = [];
let _lmStudioCache: LocalModelInfo[] = [];
let _localDirCache: LocalModelInfo[] = [];
let _customFolderCache: LocalModelInfo[] = [];
let _scanFoldersCache: ScanFolderInfo[] = [];

/** True when any on-device model (downloaded GGUF, cached repo, LM Studio, or
 * custom-folder model) is known. Reads the module caches, which persist across
 * popover mounts, so the selector can default to the On Device tab.
 *
 * Partials do not count. The cached lists carry them so they can be seen and
 * removed, but a machine whose only cached row is a cancelled download has
 * nothing to load, and opening on that tab shows one unusable row instead of
 * the list that would get the user a model. */
export function hasDownloadedModels(): boolean {
  return (
    _cachedGgufCache.some((c) => !c.partial) ||
    _cachedModelsCache.some((c) => !c.partial) ||
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


// Recommended section sort: "recommended" = newly created unsloth GGUF/MLX that fit the
// device; the rest are plain HF sort keys.
type RecommendedSortKey = "recommended" | "trendingScore" | "lastModified";

const RECOMMENDED_SORT_OPTIONS: HubOption<RecommendedSortKey>[] = [
  { value: "recommended", label: "Recommended" },
  { value: "trendingScore", label: "Trending" },
  { value: "lastModified", label: "Recent" },
];

// Sort for the On Device lists: "recent" = last loaded, "downloaded" = file download date.
type LocalSortKey = "recent" | "downloaded" | "size" | "name";

const LOCAL_SORT_OPTIONS: HubOption<LocalSortKey>[] = [
  { value: "recent", label: "Recent" },
  { value: "size", label: "Size" },
  { value: "name", label: "Name" },
  { value: "downloaded", label: "Downloaded" },
];

// Format filter dropdown for the Unsloth listing; the plain labels are reused in the empty-state copy.
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

function localPathTooltip(
  name: string,
  path: string,
  // Which checkpoint, since the path cannot say: an H3 repo holds its keyframe and reference
  // partitions in one directory.
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
): ModelSelectorChangeMeta {
  return {
    source: "local",
    isLora: false,
    isDownloaded: true,
    ...(isGguf ? { isGguf: true } : {}),
    pipelineTag: pipelineTag ?? null,
    audioType: audioType ?? null,
  };
}

function localDirectGgufMeta(
  pipelineTag?: string | null,
): ModelSelectorChangeMeta {
  return localModelMeta(true, pipelineTag);
}

/** Hugging Face address for an online/Hub row, or undefined when the repo id is missing so the
 *  row shows no empty address line. */
function hubRepoUrl(id: string | null | undefined): string | undefined {
  const trimmed = id?.trim();
  return trimmed ? `huggingface.co/${trimmed}` : undefined;
}

/** Whether a local model is an MLX build (name hint). MLX runs on Mac only, so callers gate
 *  visibility on the host being a Mac. */
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
  additionalOnDeviceModels = [],
  loadedModelIdOverride,
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
  communityModelPolicy = "none",
}: {
  models: ModelOption[];
  /** Task-runtime downloads using a cache layout the shared Hub inventory cannot represent (for
   *  example the two-file STT sidecars). */
  additionalOnDeviceModels?: ModelOption[];
  loadedModelIdOverride?: string;
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
  section?: "downloaded" | "recommended" | "connected";
  /** Section toggle rendered under the search bar. */
  sectionToggle?: ReactNode;
  onEject?: () => void;
  /** Restrict results to a pipeline task; undefined = all tasks (the chat default). */
  task?: HfTaskFilter;
  /** Curated catalog for a task-scoped picker: one canonical row per model, formats as the second level. */
  catalog?: CatalogGroup[];
  /** Also surface community models carrying `task`'s pipeline tags, below the unsloth rows.
   *  Opt-in, since the runtime has to load an arbitrary publisher's checkpoint: true of audio. */
  communityModelPolicy?: CommunityModelPolicy;
}) {
  const gpu = useGpuInfo();
  const inferenceGpu = useInferenceGpuInfo();
  // The saved VRAM Budget, threaded into every fit call here. Passing it to the quant rows alone
  // left the parent rows and the "Fits on device" filter on the 0.97 default.
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  // Whether THIS picker's rows load through the diffusion backend. Not `Boolean(task)`: Audio is
  // task-scoped but runs its GGUFs under llama.cpp / whisper.
  const diffusionLoad = useMemo(() => {
    const tasks = task ? (typeof task === "string" ? [task] : task) : [];
    return tasks.some((entry) => DIFFUSION_TASKS.has(entry));
  }, [task]);
  // What the backend actually holds, not the dropdown highlight: an image or video load evicts
  // the chat model and leaves the pick untouched, so rows kept a "Loaded" badge with nothing
  // resident. Same predicate as the header tick.
  const selectedCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const residentCheckpoint = useChatRuntimeStore((s) => s.residentCheckpoint);
  const chatLoadedModelId = chatModelLoaded({
    checkpoint: selectedCheckpoint,
    isExternalModel: isExternalModelId(selectedCheckpoint),
    residentCheckpoint,
  })
    ? selectedCheckpoint
    : undefined;
  const loadedModelId = loadedModelIdOverride ?? chatLoadedModelId;
  // Loaded GGUF quant of the active model; marks the matching pinned row.
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  // Last-loaded timestamps power the "Recent" sort (vs "Downloaded" = file date).
  const loadTimes = useModelLoadTimes(value);
  // Fade the list's top edge once scrolled, and its bottom edge while more rows sit below the fold.
  const [listScrolled, setListScrolled] = useState(false);
  const [listMoreBelow, setListMoreBelow] = useState(false);
  const hfToken = useHfTokenStore((s) => s.token);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query);
  // Shared Hub search stack so the picker and Hub run one implementation. Scoped to unsloth like the old listing.
  const online = useOnlineStatus();
  // Sanitize to anonymous on a malformed token, matching the Hub page.
  const accessToken = hfApiToken(hfToken);
  // Recommended section: a live unsloth listing sorted by the dropdown, the same sort that drives search results.
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
    // Only Recommended renders Hub results, so keep the Hub hooks idle on other tabs and preserve
    // offline-local behavior.
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

  // Two hooks for the same reason the unsloth pair exists: browse must not refetch per
  // keystroke, search must not be pinned to the empty query. pinUnslothFirst is off, since
  // the unsloth rows already sit above.
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
    sortBy: recommendedSortBy,
    sortDirection: "desc",
    pinUnslothFirst: false,
    accessToken,
    enabled: communityDiscoveryEnabled && debouncedQuery.trim().length > 0,
  });
  const communityBrowse = useHubModelSearch("", {
    task,
    ownerScope: "all",
    sortBy: recommendedSortBy,
    sortDirection: "desc",
    pinUnslothFirst: false,
    accessToken,
    enabled: communityRecommendedEnabled && debouncedQuery.trim().length === 0,
  });

  // Lowercased repo ids confirmed GGUF by the store or HF search. Absence means "no hint", so
  // hasGgufSuffix is the fallback rather than conflating unknown with known-not-GGUF.
  const modelGgufIds = useMemo(() => {
    const ids = new Set<string>();
    for (const model of models) {
      if (model.isGguf) ids.add(model.id.toLowerCase());
    }
    return ids;
  }, [models]);
  // Both listings contribute GGUF hints so a tag-only GGUF still expands variants instead of loading as a checkpoint.
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
  // GGUF vision support per repo, reported by the expander once it has read the metadata, so On
  // Device rows can badge the name.
  const [visionByRepo, setVisionByRepo] = useState<Record<string, boolean>>({});
  const reportVision = useCallback((repoId: string, hasVision: boolean) => {
    setVisionByRepo((prev) =>
      prev[repoId] === hasVision ? prev : { ...prev, [repoId]: hasVision },
    );
  }, []);
  // When on, On Device GGUF repos show their quantizations without a click.
  const expandQuantizations = useChatRuntimeStore((s) => s.expandQuantizations);
  // Off: On Device lists only downloaded quants, so a repo holding one collapses into a single row.
  const showAllQuantizations = useChatRuntimeStore(
    (s) => s.showAllQuantizations,
  );
  // Shared with the Hub page: list only models sized within the device budget.
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);
  // Repos the user clicked to collapse while expand-by-default is on, and the ones clicked back
  // open. In memory only, so both reset on reload.
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
  // Toggle a repo's quantizations: flip the collapse set when expand-by-default is on, else
  // drive the single-open expandedGguf state.
  const toggleGgufExpanded = useCallback(
    // `showing` is what the row actually renders, which is not the collapse set alone: a row held
    // back by its sole-quant probe shows nothing, and a click should open it.
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
  // The Fine-tuned section header; the train icon on the Unsloth header scrolls here.
  const fineTunedSectionRef = useRef<HTMLDivElement>(null);
  const scrollToFineTuned = useCallback(() => {
    setFineTunedCollapsed(false);
    // Two frames so the expand renders before the section is scrolled to the top.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fineTunedSectionRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    });
  }, []);
  // The Other models header; the directions icon on the Unsloth header scrolls here.
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
  // The Custom Folders header; the folder icon scrolls here instead of opening the browse popup.
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

  // `models` is already narrowed to this Audio mode and platform, so use only exact artifacts
  // from that contract for the downloaded-task and hidden-sidecar exceptions.
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
  const customFolderModels = useMemo(
    () =>
      pickerInventory.localModels.filter(
        (m) => m.source === "custom" || m.source === "ollama",
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
      // An explicit path lets the folder browser submit in the same tick it calls `setFolderInput`;
      // reading `folderInput` would race the update.
      const raw = overridePath !== undefined ? overridePath : folderInput;
      const trimmed = raw.trim();
      if (!trimmed || folderLoading) return;
      setFolderError(null);
      setFolderLoading(true);
      // From the folder browser's "Use this folder": the typed-input panel is closed, so surface failures via toast.
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

  // Updates run as managed downloads, not a blocking call; the worker pulls only changed blobs,
  // so the cached copy stays usable.
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
            // A sibling variant/snapshot for this repo is already downloading, so this update did not
            // start; say so instead of leaving the cached copy stale.
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
  // Complete downloads only. This set answers "can this id load right now": it decides
  // isDownloaded on a search pick, which is what skips download staging, and it paints the
  // on-disk dot. A partial is on disk but not loadable, so admitting one here would send a
  // torn snapshot straight to the loader from the Hub and Recommended lists.
  const downloadedSet = useMemo(
    () =>
      new Set(
        [...cachedGguf, ...cachedModels]
          .filter((c) => !c.partial)
          .map((c) => c.repo_id.toLowerCase()),
      ),
    [cachedGguf, cachedModels],
  );

  // The torn ones, kept apart so a Hub row can mark a partial rather than show it as complete
  // or as absent. Same split, and the same helper, the Hub page uses. One repo id can hold both a
  // complete GGUF copy and a torn safetensors one, since the cache keys by repo AND format, so an
  // id with any complete row is left out: it loads, and the mark would contradict that.
  const partialSet = useMemo(
    () =>
      partialSetFromRows([...cachedGguf, ...cachedModels], (c) => c.repo_id),
    [cachedGguf, cachedModels],
  );

  // Which of those continue byte for byte, so a Hub row's mark promises what the On Device row's
  // does. An id partialSet dropped never draws the mark, so a spare entry here costs nothing.
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

  // Drop models Unsloth cannot run for chat. A task-scoped picker wants exactly the tasks the
  // chat classifier calls unsupported, so it gates on the task.
  const isChatSupported = useCallback(
    (r: HfModelResult) => {
      // Image/Video tab (task set): only task-matching, non-editing results.
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
      // Task-scoped pages load single-file GGUF only; chat-only keeps runnable formats. A curated
      // artifact stays listed whatever its format, since loadSpecFor knows how to load each.
      .filter((id) =>
        task
          ? isKnownGgufRepo(id) ||
            Boolean(catalog && artifactForRepoId(id, catalog))
          : !chatOnly || isRecommendableFormat(id, isKnownGgufRepo(id), isMac),
      )
      // Member repos would collapse into the canonical group row, but nothing renders those rows yet
      // and a task-scoped picker's `models` is exactly group members, so hiding them emptied it.
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

  // Independent sort for each local section's inline dropdown.
  const [downloadedSort, setDownloadedSort] = useState<LocalSortKey>("recent");
  const [customSort, setCustomSort] = useState<LocalSortKey>("recent");
  // Format filter toggle for the Unsloth listing.
  const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");
  // What this picker's task filter has already established about every row it can show; chat
  // passes none and keeps the full set.
  const capabilityScope = useMemo<readonly (keyof ModelCapabilities)[] | null>(() => {
    const tasks: readonly string[] = task
      ? typeof task === "string"
        ? [task]
        : task
      : [];
    if (tasks.length === 0) return null;
    // Video keeps audio: a soundtrack is the one thing that separates two video models here.
    if (VIDEO_GEN_TASKS.some((t) => tasks.includes(t))) return ["audio"];
    if (IMAGE_GEN_TASKS.some((t) => tasks.includes(t))) return [];
    if (AUDIO_GEN_TASKS.some((t) => tasks.includes(t))) return [];
    return null;
  }, [task]);

  // MLX and Safetensors repos are one download, so a row can name its size.
  const hubRowsShowSize =
    formatFilter === "mlx" || formatFilter === "safetensors";

  // A curated row's name and its chips; ids outside the catalog have neither and show the raw repo id.
  const curatedRow = useCallback(
    (id: string) =>
      (catalog && curatedRowLabelFor(id, catalog, hostClass)) ?? {
        name: id,
        tags: [] as string[],
      },
    [catalog, hostClass],
  );

  /** Whether this host can run a curated id at all, as opposed to whether it has room for it. Browse rows only. */
  const curatedOfferable = useCallback(
    (id: string) => {
      if (!catalog) return true;
      // Downloaded weights keep their row: hiding what is already on disk reads as Unsloth having lost the model.
      if (downloadedSet.has(id.toLowerCase())) return true;
      const hit = artifactForRepoId(id, catalog);
      return hit ? curatedArtifactIsOfferable(hit.artifact.repoId, hostClass) : true;
    },
    [catalog, downloadedSet, hostClass],
  );

  // Paint curated rows before any request, so a task-scoped picker whose models are already in
  // memory does not sit on a spinner.
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
        // Size from the catalog, not an id "<n>B" guess: the guess is missing for most curated ids
        // and wrong for others (Wan2.2-TI2V-5B is 30 GB, not 2).
        curatedSizeBytes: catalog ? curatedSizeBytesFor(id, catalog) : undefined,
        // Same reason the size is curated: a seed the listing does not return has no other source for its param chip.
        totalParams: catalog ? curatedTotalParamsFor(id, catalog) : undefined,
      }));
  }, [catalog, models, formatFilter, isKnownGgufRepo, task, curatedOfferable]);

  /** The catalog's own fit verdict for a curated artifact, or undefined where it has none. Every
   *  list that judges a row against the device goes through this, so a badge and its filters
   *  cannot disagree. */
  const catalogFit = useCallback(
    (id: string, budget: DeviceBudget) =>
      catalog ? curatedArtifactFitsDevice(id, catalog, budget) : undefined,
    [catalog],
  );

  const isUnslothOwned = useCallback(
    (id: string) => id.toLowerCase().startsWith("unsloth/"),
    [],
  );

  /** Pipeline tag is the Hub's only signal, since the real test is the checkpoint's tokenizer and
   *  that needs the download first. Exports in another serialization are dropped by name. */
  const isLoadableCommunityRepo = useCallback(
    (id: string) =>
      !/(^|[-_/.])(onnx|openvino|tflite|coreml)([-_./]|$)/i.test(id),
    [],
  );

  const catalogSeedIds = useMemo(
    () => catalogSeedRows.map((row) => row.id),
    [catalogSeedRows],
  );

  // Recommended suggests GGUF anywhere, plus MLX and safetensors on Mac; the "recommended" sort
  // also drops models too big for the device. Downloaded models stay visible.
  const recommendedRows = useMemo(() => {
    const catalogSeedIds = new Set(
      catalogSeedRows.map((row) => row.id.toLowerCase()),
    );
    const keepCommon = (r: HfModelResult) => {
      const isCatalogSeed = catalogSeedIds.has(r.id.toLowerCase());
      return (
        !isMobileVariant(r.id) &&
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
      // Task pages load single-file GGUF, plus curated artifacts in any format.
      (!task ||
        r.isGguf ||
        Boolean(catalog && artifactForRepoId(r.id, catalog)));
    // A community row has no catalog artifact by definition, so the curated clause above would
    // drop every third-party safetensors checkpoint.
    const keepCommunity = (r: HfModelResult) =>
      keepCommon(r) && isTaskRuntimeSupported(r);
    // Members are not filtered here (see recommendedIds): that dropped them from Hub search too.
    // "recommended" always device-filters; the "Fits on device" tick extends it to other sorts.
    const deviceFiltered = recommendedSort === "recommended" || fitOnDeviceOnly;
    const taskScoped = Boolean(task);
    const rowGpu = loadScopedGpu(gpu, taskScoped);
    const rowInferenceGpu = loadScopedGpu(inferenceGpu, taskScoped);
    const pipelineBudget = artifactBudget(rowGpu);
    const fits = (r: HfModelResult) =>
      // Downloaded models show regardless of fit.
      downloadedSet.has(r.id.toLowerCase()) ||
      // The catalog's own verdict where it has one, so this list and the OOM badge cannot disagree:
      // hfModelFitsDevice counts RAM toward a load that never leaves the card.
      (catalogFit(r.id, pipelineBudget) ??
        hfModelFitsDevice(r, diffusionLoad || !r.isGguf ? rowGpu : rowInferenceGpu, {
          budgetFraction,
          // Not `&& r.isGguf`: on a task page a safetensors row is placed by the same backend, and this
          // rule IS the budget those rows had before the classifiers were merged.
          mediaLoad: diffusionLoad,
          hostPooledMemory: gpu.loadDeviceSharesHostMemory,
          // The scoped inventory's own count, so it always describes rowInferenceGpu's capacity.
          gpuCount: rowInferenceGpu.deviceCount,
        }));
    const unslothRows = orderRecommendedRows({
      seeds: catalogSeedRows,
      results: recommendedSearch.results,
      keep,
      deviceFiltered,
      fits,
    });
    if (!communityRecommendedEnabled) return unslothRows;
    // Appended below everything unsloth publishes, so scrolling past the unsloth uploads continues
    // into the wider Hub. Same keep/fits gates.
    const above = new Set(unslothRows.map((r) => r.id.toLowerCase()));
    const communityRows = communityBrowse.results
      .filter((r) => !r.id.toLowerCase().startsWith("unsloth/"))
      .filter((r) => isLoadableCommunityRepo(r.id))
      .filter((r) => !above.has(r.id.toLowerCase()))
      .filter(keepCommunity)
      .filter((r) => !deviceFiltered || fits(r));
    return [...unslothRows, ...communityRows];
  }, [
    budgetFraction,
    diffusionLoad,
    recommendedSearch.results,
    catalogSeedRows,
    downloadedSet,
    recommendedSort,
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

  // Per-row meta and VRAM badge from the recommended listing's own metadata, with the curated
  // seeds behind it: a listing row wins, and a curated row it never returns keeps its chip.
  const recommendedMeta = useMemo(() => {
    const map = new Map<
      string,
      {
        meta: string | null;
        /** GGUF rows carry the classifier's own verdict; curated torch rows carry "exceeds". */
        status: GgufFitClass | VramFitStatus | null;
        est: number;
      }
    >();
    /** Size-based verdict for a row whose real footprint we know, against the budget that row
     *  actually loads into. Returns the verdict, not a boolean: a boolean collapsed `marginal`
     *  and `partial` into "no badge", so a repo needing offload rendered as a clean fit. */
    const ggufRowFit = (
      sizeBytes: number | undefined,
      budget: typeof inferenceGpu,
    ): GgufFitClass | VramFitStatus | null => {
      const anyBudget =
        budget.memoryTotalGb > 0 || budget.systemRamAvailableGb > 0;
      if (!budget.budgetKnown && !anyBudget) return null;
      if (sizeBytes == null) return null;
      // Probed and genuinely zero (a Vulkan device reporting nothing) means nothing fits.
      if (!anyBudget) return "oom";
      // Images / Video place this GGUF through the diffusion backend, so it takes the same rule its
      // quant rows take; different budgets let a row read as fitting while its children read oom.
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
    // A curated pipeline loads through torch and a task load puts the whole thing on ONE device,
    // so it is judged there. inferenceGpu is the GGUF backend's inventory, a different install.
    const rowGpu = loadScopedGpu(gpu, Boolean(task));
    const pipelineBudget = artifactBudget(rowGpu);
    // The inventory of the runtime that PLACES the row, not of its file format: on a Vulkan chat
    // build inferenceGpu can see a card torch cannot, so the media rule scored against capacity
    // the diffusion loader never gets.
    const rowInferenceGpu = diffusionLoad
      ? rowGpu
      : loadScopedGpu(inferenceGpu, Boolean(task));
    // Community rows come from their own listing; without them folded in here they render with no size or VRAM chip.
    for (const r of [
      ...recommendedSearch.results,
      ...catalogSeedRows,
      ...communityBrowse.results,
    ]) {
      if (map.has(r.id)) continue;
      const isG = isKnownGgufRepo(r.id);
      // GGUF param count comes from the repo name or the GGUF metadata, so even repos with no "<n>B"
      // token show a param chip.
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
        // GGUF fit is size-based: flag OOM when even the smallest quant we can size exceeds the
        // budget. Repos we cannot size show no badge.
        const params = ggufParams;
        const sizeBytes =
          r.estimatedSizeBytes ??
          (params ? estimateQuantBytes(params) : undefined);
        map.set(r.id, {
          meta,
          // The classifier's own verdict, so a GGUF row never borrows the torch-only "exceeds", scoped
          // to the device the load LANDS on: unscoped, a downloaded media parent read as safe while
          // every variant inside it read as oom.
          status: ggufRowFit(sizeBytes, rowInferenceGpu),
          // The figure the verdict was reached with, not the raw file size: classifyGgufFit scores
          // weights plus activations and KV, so a 20 GiB quant needing 24 GiB read "tight fit" beside
          // "~20GB VRAM". The media rule scores the raw size and keeps it.
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
      // A curated pipeline is judged by the catalog, which knows its resident size; the QLoRA
      // estimator reads a diffusion pipeline as a language model it can 4-bit quantize (Wan 2.2
      // TI2V is 30 GB, where 5B params says 5.9).
      const curatedFits = catalogFit(r.id, pipelineBudget);
      if (curatedFits !== undefined) {
        const curatedBytes = catalog
          ? (r.curatedSizeBytes ?? curatedSizeBytesFor(r.id, catalog))
          : undefined;
        map.set(r.id, {
          meta,
          status: curatedFits ? null : "exceeds",
          est: curatedBytes ? Math.round(curatedBytes / 1024 ** 3) : 0,
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

  // Hub pipeline tag per repo id, handed to the page on pick so a task page can classify an uncurated repo.
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

  // The rest of the Hub evidence the Audio page judges a community row on, keyed the same way so
  // a chat pick is routed on what the page would have listed it on.
  const hubEvidenceById = useMemo(() => {
    const map = new Map<
      string,
      {
        baseModel?: string | null;
        tags?: string[];
        libraryName?: string | null;
        audioType?: string | null;
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
    // Downloaded rows too: the backend tags a cached Whisper checkpoint as ASR even when its repo
    // name says nothing, and without those tags the same row was judged on its id alone.
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
      const existing = map.get(c.repo_id);
      if (existing) {
        map.set(c.repo_id, {
          ...existing,
          audioType: existing.audioType ?? c.audio_type,
        });
        continue;
      }
      map.set(c.repo_id, { audioType: c.audio_type });
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

  // Tag-accurate capabilities keyed by repo id, pooled from both HF listings then the catalog
  // for curated ids neither returned. Listings first: real tags outrank curated data.
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
        // Merged, not skipped when the listing already answered: a curated entry states what the model
        // does (H3's audio track, which no repo tag mentions), and first-wins dropped that.
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

  // Ordered by the On Device dropdown. The gate keeps diffusion GGUFs in the Images/Video picker and out of chat.
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
            // A speech GGUF no backend here can decode (CSM) would otherwise be listed as a chat model and
            // fail only in llama-server. Non-audio rows always pass.
            audioPickIsRoutable({
              id: c.repo_id,
              task: c.task,
              audioType: c.audio_type,
              isGguf: true,
              isCurated: artifactForRepoId(c.repo_id, AUDIO_CATALOG) !== null,
              // The task and codec both came from GGUF classification; codec provenance separates runnable
              // Orpheus from unsupported CSM.
              taskFromGgufArch: true,
            }),
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
    ],
  );
  // Cached non-GGUF repos. In chat, passesTaskGate drops diffusers image repos; the Images
  // picker keeps only unsloth-hosted ones this backend can load.
  const sortedCachedModels = useMemo(
    () =>
      sortCachedRepos(
        cachedModels.filter(
          (c) =>
            // Partial snapshots are listed, not loaded. Dropping them here hid a cancelled
            // multi-GB download from the only list that could delete it; the row instead carries
            // a partial mark and selects with isDownloaded: false, so the click opens the
            // download rather than erroring or triggering a silent re-fetch.
            passesTaskGate(
              c.task,
              c.repo_id,
              task,
              catalog,
              activeCatalogArtifactIds,
            ) &&
            // Diffusion pickers: unsloth repos plus any repo the backend can LOAD. Gate on a curated
            // ARTIFACT, not a group-key match: a base sibling matches by key but dead-ends at the trust
            // gate. An unsloth repo must also be a full pipeline, since from_pretrained fails on a
            // single-file checkpoint repo.
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
                : false)),
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
    ],
  );
  // Task-scoped loads put the whole pipeline on ONE device, so quant fit uses the device the
  // load lands on (the lowest visible ordinal), not the multi-GPU sum: sizing against the
  // bigger card OOMs the smaller one. Chat keeps the sum.
  const expanderGpuGbFrom = (info: typeof inferenceGpu) =>
    info.available
      ? loadScopedGpu(info, Boolean(task)).memoryTotalGb
      : undefined;
  // Images / Video place through torch even where llama.cpp is a Vulkan build, so their budget
  // comes from the torch inventory.
  const expanderBudgetGpu = diffusionLoad ? gpu : inferenceGpu;
  const expanderGpuGb = expanderGpuGbFrom(expanderBudgetGpu);
  const expanderSystemGpuGb = expanderGpuGbFrom(gpu);
  // From the SAME scoping decision as the capacity above: loadScopedGpu narrows the count to 1,
  // so the per-card reserve is never charged host-wide against one card.
  const expanderScopedGpu = loadScopedGpu(expanderBudgetGpu, Boolean(task));
  const expanderGpuCount = expanderScopedGpu.deviceCount;
  const expanderRamGb = expanderScopedGpu.systemRamAvailableGb;

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
            filesystemRowsSupportedForTask(task, m.task) &&
            // The same speech gate the cached GGUF rows get: a CSM file found locally is just as
            // undecodable, and routing it to Audio evicts the chat model before the row is refused.
            audioPickIsRoutable({
              id: m.model_id ?? m.id,
              task: m.task,
              audioType: m.audio_type,
              isGguf: localModelIsGguf(m),
              isCurated: artifactForRepoId(m.model_id ?? m.id, AUDIO_CATALOG) !== null,
              // Task and codec came from the filesystem classifier, so a renamed CSM file cannot borrow an
              // Orpheus-looking path.
              taskFromGgufArch: true,
            }) &&
            // The backend tags every local model with its task for exactly this: on the Images/Video pages
            // a chat GGUF must not be offered.
            passesTaskGate(
              m.task,
              m.model_id ?? m.id,
              task,
              catalog,
              activeCatalogArtifactIds,
            ) &&
            localModelMatchesFormat(m, formatFilter) &&
            matchesLocalQuery(m),
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
    ],
  );
  // Local ./models entries. Chat-only Unsloth runs GGUF anywhere and MLX on Mac, so raw
  // checkpoints there are hidden; a task-scoped picker is exempt, since the image backend
  // loads local pipelines.
  const sortedLocalDir = useMemo(
    () =>
      sortLocalModels(
        localDirModels.filter(
          (m) =>
            filesystemRowsSupportedForTask(task, m.task) &&
            // The same speech gate the cached GGUF rows get: a CSM file found locally is just as
            // undecodable, and routing it to Audio evicts the chat model first.
            audioPickIsRoutable({
              id: m.model_id ?? m.id,
              task: m.task,
              audioType: m.audio_type,
              isGguf: localModelIsGguf(m),
              isCurated: artifactForRepoId(m.model_id ?? m.id, AUDIO_CATALOG) !== null,
              // Task and codec came from the filesystem classifier, so a renamed CSM file cannot borrow an
              // Orpheus-looking path.
              taskFromGgufArch: true,
            }) &&
            passesTaskGate(
              m.task,
              m.model_id ?? m.id,
              task,
              catalog,
              activeCatalogArtifactIds,
            ) &&
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
      catalog,
      activeCatalogArtifactIds,
    ],
  );
  const sortedCustomFolderModels = useMemo(
    () =>
      sortLocalModels(
        customFolderModels.filter(
          (m) =>
            filesystemRowsSupportedForTask(task, m.task) &&
            // The same speech gate the cached GGUF rows get: a CSM file found locally is just as
            // undecodable, and routing it to Audio evicts the chat model first.
            audioPickIsRoutable({
              id: m.model_id ?? m.id,
              task: m.task,
              audioType: m.audio_type,
              isGguf: localModelIsGguf(m),
              isCurated: artifactForRepoId(m.model_id ?? m.id, AUDIO_CATALOG) !== null,
              // Task and codec came from the filesystem classifier, so a renamed CSM file cannot borrow an
              // Orpheus-looking path.
              taskFromGgufArch: true,
            }) &&
            passesTaskGate(
              m.task,
              m.model_id ?? m.id,
              task,
              catalog,
              activeCatalogArtifactIds,
            ) &&
            localModelMatchesFormat(m, formatFilter) &&
            matchesLocalQuery(m),
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
    ],
  );

  // Chat cannot load a diffusion model but the Images/Video pages can, so a pick routes to the
  // page that runs it rather than 400ing. Task-scoped pickers select normally.
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
    // Both ids: a local row's click passes m.id (a filesystem path) while m.model_id is its
    // HF-style name, so keying on one alone makes the lookup miss.
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
          // Loading it here would evict the chat model for a repo neither surface can run.
          toast.error(
            `${id} is not a speech model Unsloth can run yet. The Audio page lists the families it supports.`,
            { duration: 7000 },
          );
          return;
        }
        if (page) {
          void navigateToPage({
            to: `/${page}`,
            // `quant` is used verbatim as the gguf filename, so a label like "Q4_K_M" rides ggufQuant
            // instead; dropping it made every non-curated GGUF repo arrive as a bare repo id.
            search:
              page === "audio"
                ? {
                    model: id,
                    quant: meta.ggufFilename ?? undefined,
                    ggufQuant: meta.ggufFilename
                      ? undefined
                      : (meta.ggufVariant ?? undefined),
                    // pickedTask, not meta.pipelineTag: a cached row carries no tag to forward.
                    task: pickedTask ?? undefined,
                    audioType: meta.audioType ?? undefined,
                    loadId: meta.loadId ?? undefined,
                  }
                : diffusionRouteSearch(id, meta),
          });
          return;
        }
      }
      onSelectProp(id, meta);
    },
    [task, diffusionTaskById, hubEvidenceById, navigateToPage, onSelectProp],
  );

  // Fine-tuned models for the On Device section: flat, query-filtered, newest first.
  const fineTunedRows = useMemo(() => {
    const needle = normalizeForSearch(debouncedQuery.trim());
    return loraModels
      .filter(
        (m) =>
          // A CSM export in a GGUF container loads nowhere: llama.cpp has no decoder and the Audio page
          // does not list speech GGUFs. audioType comes off the checkpoint, so a renamed path is
          // still caught.
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

  // While searching, filter Downloaded by the query instead of hiding it, so a downloaded model
  // the user is searching for stays visible.
  const visibleCachedGguf = useMemo(() => {
    if (!showHfSection)
      return sortedCachedGguf.filter((c) =>
        matchesFormatFilter(c.repo_id, true, formatFilter),
      );
    const q = normalizeForSearch(debouncedQuery.trim());
    // Keep the format filter active while searching so the dropdown stays consistent with the no-query branch.
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

  // Non-GGUF cached rows are not shown in chat-only mode, so the empty-state logic must use this
  // or the picker can go blank. A task-scoped picker is exempt.
  // Not visibleCachedModels, or the picker can go blank. A task-scoped picker (Images) is exempt:
  // the image backend loads local diffusers/safetensors pipelines even on chat-only hosts.
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
            `${model.id} ${model.name} ${model.description ?? ""}`,
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

  // Pinned entries surface in their own section above the Unsloth heading: GGUF quants pin
  // individually with their repo still listed below, non-GGUF repos pin whole.
  const pinnedIds = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const unpinRepo = usePinnedModelsStore((s) => s.unpinRepo);
  const pinnedSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);

  // Candidate pins whose repo still exists in the cache; per-quant validation below is needed
  // because deleting one variant can leave a sibling cached.
  const pinnedQuantCandidates = useMemo(() => {
    // The existence check ignores the text query so a pinned quant stays findable by quant name;
    // querying visibleCachedGguf would drop the repo first.
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
          // If the backend cannot verify a quant, hiding the direct-load row is safer than claiming a
          // missing file is downloaded.
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

  // Split downloaded models so non-Unsloth repos get their own "Other models" section above Fine-tuned.
  const unslothCachedGguf = useMemo(
    () => visibleCachedGguf.filter((c) => isUnslothPublisherRepoId(c.repo_id)),
    [visibleCachedGguf],
  );
  const otherCachedGguf = useMemo(
    () => visibleCachedGguf.filter((c) => !isUnslothPublisherRepoId(c.repo_id)),
    [visibleCachedGguf],
  );
  const unslothCachedModelRows = useMemo(
    () =>
      visibleCachedModelRows.filter(
        (c) =>
          isUnslothPublisherRepoId(c.repo_id) &&
          !pinnedSet.has(pinKey(c.repo_id)),
      ),
    [visibleCachedModelRows, pinnedSet],
  );
  const otherCachedModelRows = useMemo(
    () =>
      visibleCachedModelRows.filter(
        (c) =>
          !isUnslothPublisherRepoId(c.repo_id) &&
          !pinnedSet.has(pinKey(c.repo_id)),
      ),
    [visibleCachedModelRows, pinnedSet],
  );

  // Param counts come straight off the unsloth listings the picker already loaded, so the VRAM
  // badges need no extra fetch.
  const recommendedParamCountById = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of [...results, ...recommendedSearch.results]) {
      if (r.totalParams) map.set(r.id, r.totalParams);
    }
    return map;
  }, [results, recommendedSearch.results]);

  // Shared by both search lists so a curated id one drops cannot return via the other as a raw Hub row.
  const searchRowFits = useCallback(
    (row: {
      id: string;
      totalParams?: number;
      estimatedSizeBytes?: number;
      curatedSizeBytes?: number;
    }) =>
      catalogFit(row.id, artifactBudget(loadScopedGpu(gpu, Boolean(task)))) ??
      searchRowFitsDevice(
        {
          ...row,
          // Curated params last, same rule as the curated size: a listing total wins, but a repo no
          // listing returns must still be sizable or `requireKnown` hides it from search.
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
          // Separate from taskScoped: that picks the single-device budget for every task page, this
          // picks the diffusion RULE, which only Images and Video use.
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

  // Recommended models that match the current search query.
  const filteredRecommendedIds = useMemo(() => {
    if (!showHfSection) return [];
    const q = normalizeForSearch(debouncedQuery.trim());
    return (
      // Seeds included: recommendedIds hides downloaded models, which the unfiltered Recommended
      // list still paints, so without them a curated pick vanishes from search once on disk.
      searchableRecommendedIds(catalogSeedIds, recommendedIds)
        .filter((id) => normalizeForSearch(id).includes(q))
        .filter((id) =>
          matchesFormatFilter(id, isKnownGgufRepo(id), formatFilter),
        )
        // Curated defaults obey the fit toggle like the live HF rows, else large defaults resurface in
        // search with the filter on.
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

  // One pipeline for both listings, so community rows clear the same gates; `owned` is the only difference.
  const searchIdsFrom = useCallback(
    (rows: readonly HfModelResult[], owned: (id: string) => boolean) =>
      rows
        .filter(isChatSupported)
        .filter(isTaskRuntimeSupported)
        .filter(
          (r) =>
            !fitOnDeviceOnly ||
            downloadedSet.has(r.id.toLowerCase()) ||
            searchRowFits(r),
        )
        .map((result) => result.id)
        .filter((id) => !isHiddenModelId(id))
        .filter(owned)
        // Search reaches the live Hub, so without this a query re-lands the exact curated row the seed
        // and Recommended filters just dropped, clickable and still refused at load.
        .filter(curatedOfferable)
        .filter((id) => !recommendedSet.has(id))
        // Chat-only keeps runnable formats: GGUF anywhere, plus MLX/safetensors on Mac, matching the
        // empty Recommended view.
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
      downloadedSet,
      searchRowFits,
      isMac,
      curatedOfferable,
    ],
  );

  const hfIds = useMemo(() => {
    // Only the Unsloth tab searches the HF listing.
    if (!showHfSection || section !== "recommended") return [];
    return searchIdsFrom(results, isUnslothOwned);
  }, [results, showHfSection, section, searchIdsFrom, isUnslothOwned]);

  // Community search hits, listed after the unsloth ones and deduped against them.
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

  /** Unsloth first, then community: one list so rows, keyboard order and the empty state cannot drift apart. */
  const searchRowIds = useMemo(
    () => [...hfIds, ...communitySearchIds],
    [hfIds, communitySearchIds],
  );

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

    // Unsloth-tab search keys (curated matches + HF unsloth results).
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

    // Other (non-Unsloth) downloaded rows sit just above Fine-tuned.
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
      { est: number; status: VramFitStatus | null; detail: string | null }
    >();
    const pipelineBudget = artifactBudget(loadScopedGpu(gpu, Boolean(task)));
    for (const id of filteredRecommendedIds) {
      // GGUF fit is size-based and badged elsewhere; skip the qlora estimate.
      if (isKnownGgufRepo(id)) continue;
      const totalParams = recommendedParamCountById.get(id) ?? paramsFromId(id);
      // Same verdict the unfiltered list gives this row: searching for a model must not change what
      // it says about the device.
      const curatedFits = catalogFit(id, pipelineBudget);
      if (catalog && curatedFits !== undefined) {
        const curatedBytes = curatedSizeBytesFor(id, catalog);
        // The catalog is the only source of a count for a curated repo the listing never returns and
        // whose id spells no "<n>B".
        const params = totalParams ?? curatedTotalParamsFor(id, catalog);
        map.set(id, {
          est: curatedBytes ? Math.round(curatedBytes / 1024 ** 3) : 0,
          status: curatedFits ? null : "exceeds",
          detail: params ? formatCompact(params) : null,
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

  // Sentinel and IntersectionObserver for recommended infinite scroll. Re-running per loaded
  // page re-attaches the observer so a heavily filtered list keeps paging; fetchMore no-ops
  // while a page is in flight.
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

  /** Handle clicking a model row — GGUF repos expand, others load directly. */
  const handleModelClick = useCallback(
    (id: string) => {
      if (isKnownGgufRepo(id)) {
        setExpandedGguf((prev) => (prev === id ? null : id));
      } else {
        // Cached repos load now; uncached ones download via the Hub manager.
        onSelect(id, {
          source: "hub",
          isLora: false,
          isDownloaded: downloadedSet.has(id.toLowerCase()),
          pipelineTag: pipelineTagById.get(id) ?? null,
        });
      }
    },
    [onSelect, isKnownGgufRepo, downloadedSet, pipelineTagById],
  );

  // On Device owns the downloaded and custom-folder models; the Unsloth tab searches the HF
  // listing. Both filter locally by the query.
  const showDownloaded = section === "downloaded";
  const showCustom = section === "downloaded";
  const showRecommendedSection = !showHfSection && section === "recommended";
  const downloadedEmpty =
    pinnedRows.length === 0 &&
    visibleCachedGguf.length === 0 &&
    visibleCachedModelRows.length === 0 &&
    visibleAdditionalOnDeviceModels.length === 0 &&
    sortedLmStudio.length === 0 &&
    sortedLocalDir.length === 0 &&
    // Fine-tuned models are on-device too: do not show the empty state above a non-empty Fine-tuned section.
    fineTunedRows.length === 0;

  // Sort dropdown inline right of the section toggle; options depend on the tab and stay visible
  // while searching. Fixed width matches the Search Hub button.
  const sortTriggerClassName =
    "h-(--picker-control-h) w-(--picker-control-w) shrink-0 justify-between pr-2.5 !border-0 text-xs [&>span]:!text-clip";
  // Tighter menu matching the trigger; keep the option's right padding so the checkmark never overlaps the label.
  const sortMenuContentClassName =
    "!p-1 !rounded-[14px] [&_[role=option]]:!pl-2 [&_[role=option]]:!py-1.5 [&_[role=option]]:!text-xs [&_[role=option]]:!rounded-[10px]";
  // Device-fit toggle inside the sort menu. The whole row is the button: a Checkbox renders as a
  // <button> and label-click forwarding is unreliable, so the Checkbox is presentational.
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
  // On Device rows are already on disk, so the device-fit filter only applies to the Unsloth listing.
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
  // The Connected layout uses a wider box, so it drops the search inset to keep Search Hub on the last dropdown's edge.
  const hasConnected = externalModels.length > 0;
  // The Other models section and its shortcut only show with non-Unsloth downloads.
  const hasOtherModels =
    otherCachedGguf.length > 0 ||
    otherCachedModelRows.length > 0 ||
    otherAdditionalOnDeviceModels.length > 0;

  const downloadedRowButtonClassName =
    "bg-transparent pr-1 hover:bg-transparent focus-visible:bg-transparent dark:bg-transparent dark:hover:bg-transparent dark:focus-visible:bg-transparent";
  // Not focus-within: the dots menu returns focus to its trigger on close, so the row stayed lit
  // after the pointer left. A row carrying a memory bar is two lines tall and the shell paints
  // the background, so the radius relaxes with it or the row renders as a stadium.
  const downloadedRowShellClassName = (
    selected: boolean,
    hasMemoryBar = false,
  ) =>
    cn(
      "group flex items-center transition-colors hover:bg-[#ececec] has-[:focus-visible]:bg-[#ececec] has-[[data-state=open]]:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)] dark:has-[:focus-visible]:bg-[var(--sidebar-accent)] dark:has-[[data-state=open]]:bg-[var(--sidebar-accent)]",
      hasMemoryBar ? "rounded-2xl" : "rounded-full",
      selected && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
    );

  // A pinned quant: repo name with the quant as a grey chip, loaded in one click.
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
      <div
        key={optionKey}
        className={downloadedRowShellClassName(isSelected, true)}
      >
        {/* Through ModelRow, so a pinned quant lands in the same columns as the rows below it. */}
        <div className="min-w-0 flex-1">
          <ModelRow
            label={entry.repoId}
            tooltipText={`${entry.repoId} (${ggufQuantDetailLabel(
              entry.quant,
            )})`}
            meta="GGUF"
            quantChip={ggufQuantChipLabel(entry.quant)}
            // Same runtime gate the sole-quant row applies: a pinned quant can belong to an image, video
            // or audio task, which load through the media planner, so the KV estimator would measure
            // the wrong runtime and fall back to the file size.
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
                // The row loads one quant, so it is a GGUF pick like the expander's; without this the pages
                // asked for a pipeline, which a GGUF repo rejects. No filename: the pin stores a label.
                isGguf: true,
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
              pinLabel: "Pin to top",
              unpinLabel: "Unpin",
              onToggle: () => togglePinned(entry.repoId, entry.quant),
            }}
            del={{
              title: "Delete cached model?",
              // Same preview the Hub On Device row asks for, so a companion base an installed image model
              // still needs shows the reason and a disabled Delete.
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

  // One quant on disk with "Show all quantizations" off: the expander would list just that
  // quant, so the row carries it as a chip and loads it in one click.
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
    // A repo only flags partial once no quant is clean, so a sole-quant row should never BE one.
    // Carried anyway: if that ever stops holding, the row states what is on disk instead of
    // handing a torn file to the loader.
    const isPartial = c.partial === true;
    const selectMeta: ModelSelectorChangeMeta = {
      source: "hub",
      isLora: false,
      // Only for a complete snapshot, as the variant select already does. A loadId names a
      // revision on disk, and the Audio route carries no isDownloaded field, so a forwarded
      // one is read there as proof the weights are present.
      loadId: isPartial ? undefined : c.load_id,
      ggufVariant: variant.quant,
      ggufFilename: variant.filename,
      isDownloaded: !isPartial,
      expectedBytes,
      isGguf: true,
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
            // No verdict to pass, so the mark takes its cautious wording:
            // /api/models/gguf-variants builds models.models.GgufVariantDetail, which carries no
            // partial_resumable. A sole-quant row is never partial anyway -- readSoleQuant only
            // picks a clean quant -- so this mark is defensive to begin with.
            // Only for models the llama.cpp path actually loads. The Images and
            // Video pickers deliberately keep diffusion GGUFs listed, and those
            // run on the diffusion planner with different runtime buffers, on a
            // single torch device rather than the aggregate inference pool. The
            // KV estimator has nothing to say about them, and when it returns
            // unsized the bar falls back to the file size and draws a
            // weights-only verdict anyway -- a confident number about the wrong
            // runtime, which is the failure this bar exists to avoid.
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
    // Auto-expansion waits for the probe: expanding every row first would mount an expander, and
    // its remote listing, for repos about to collapse.
    const expanderOpen = shouldMountVariantExpander({
      expanded: isGgufExpanded(c.repo_id),
      autoExpand: expandQuantizations && !reopenedGguf.has(c.repo_id),
      soleQuantsPending: soleQuants.pending.has(c.repo_id),
    });
    // No quant of this repo is clean, so nothing inside the expander can carry the row's actions.
    const isPartialRepo = c.partial === true;
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
              partial={isPartialRepo}
              partialResumable={c.partial_resumable}
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
          {/* A complete repo keeps its actions on the quant rows inside the expander -- delete
              targets one quant, not the repo -- so this row only reserves the gutter, to keep the
              tags lined up. A partial repo has no complete quant to carry them: expanding it just
              to reach a menu is a step with nothing at the end of it, and before this the torn
              bytes could be seen but never removed. */}
          {isPartialRepo ? (
            <span className={ROW_ACTIONS_PINNED_CLASS}>
              <ModelRowMenu
                ariaLabel={`More options for ${c.repo_id}`}
                cachePath={{ repoId: c.repo_id }}
                del={{
                  title: "Delete cached model?",
                  impact: { repoId: c.repo_id },
                  // Repo-wide, like every other repo-level delete: no variant is passed, and
                  // one repo id can also hold a complete copy in another format. Saying
                  // "the partial download" would name a smaller scope than the one that runs.
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
                    // Every quant goes with the repo, so every quant pin goes too.
                    unpinRepo(c.repo_id);
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
    // Some bytes on disk, not enough to load. Claiming it is downloaded skips straight to a load
    // that fails on the missing shards, so the pick reports what is actually there and the
    // download flow picks it up from the same place the Hub would.
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
                // Dropped on a torn snapshot: the Audio route has no isDownloaded field and
                // reads a forwarded loadId as proof the weights are there, so a TTS pick
                // routed with one skips the download it needs.
                loadId: isPartial ? undefined : c.load_id,
                isDownloaded: !isPartial,
                pipelineTag: c.task ?? null,
                audioType: c.audio_type ?? null,
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
                  // Run spreads this meta straight back into a select, so it carries the
                  // row's rule: no load identity for a snapshot that is not all there.
                  // The config page keys its settings off the repo id, not this field.
                  loadId: isPartial ? undefined : c.load_id,
                  isDownloaded: !isPartial,
                  isGguf: false,
                  pipelineTag: c.task ?? null,
                  audioType: c.audio_type ?? null,
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
                // Repo-wide, so the quant pins go too: one id can hold a GGUF copy as well,
                // and this delete takes that with it.
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
    // A checkpoint trained here is identified by its directory, not a repo id, so show the name and drop the Hub link.
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

  return (
    <CapabilityScope.Provider value={capabilityScope}>
      <div className="relative space-y-2">
        {/* A small right inset shortens the search bar so Search Hub lands on the last dropdown's right edge. */}
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

        {/* Keep the left-packed controls on one line while they fit, then wrap whole groups before
            their intrinsic widths cross the picker edge. */}
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
            // The list sits within the menu padding so gaps match; scroll-py and symmetric px keep the
            // focus ring off the overflow clip edges during keyboard nav.
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
              // On Device pulls the heading block tight to the controls; Recommended keeps more top room
              // above its first row.
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
                    {/* Wider than the On Device section labels: nothing divides these groups but the gap. */}
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

                {/* Empty On Device: a search miss versus nothing downloaded yet. Hidden when custom folders
                    below still have matches. */}
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

                {/* Pinned quants and models sit above the Unsloth heading, filtered by the query like the
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
                      {/* Rows drop the unsloth/ prefix; the heading carries it for the group. */}
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

                {/* Other models: non-Unsloth downloads, shown only when such models exist. */}
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

                {/* Fine-tuned models: always shown on On Device so the train shortcut has a target. Hidden
                    under a task filter. */}
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
                      scanFolders.map((f) => {
                        const problem = scanFolderStatusCopy(f.status);
                        return (
                          <div
                            key={f.id}
                            className="group flex items-center gap-1.5 px-2.5 py-0.5"
                          >
                            <HugeiconsIcon
                              icon={Folder02Icon}
                              className="size-3 shrink-0 text-muted-foreground/40"
                            />
                            <div className="min-w-0 flex-1">
                              <span
                                className="block truncate font-mono text-ui-10 text-muted-foreground/70"
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
                              className="shrink-0 rounded p-1 text-foreground/70 transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:bg-destructive/10 focus-visible:text-destructive"
                            >
                              <HugeiconsIcon
                                icon={Cancel01Icon}
                                className="size-3"
                              />
                            </button>
                          </div>
                        );
                      })}

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
                        // Pass the path explicitly: `folderInput` state has not flushed when "Use this folder" submits.
                        void handleAddFolder(picked);
                      }}
                    />

                    {/* Models from custom folders */}
                    {!customFoldersCollapsed &&
                      sortedCustomFolderModels.map((m) => {
                        const isGgufFile = m.path
                          .toLowerCase()
                          .endsWith(".gguf");
                        // Honor the backend model_format hint (suffixless GGUF folders) as well as name/path, so the
                        // row classifies and loads through the same GGUF path as the filter.
                        const isGguf = localModelIsGguf(m);
                        // Single .gguf files load directly; GGUF repos and directories expand to pick a variant. An
                        // Ollama manifest reference names one blob, so it is direct too.
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
                                    // Direct loads set no active variant, so requiring one never reads as loaded.
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
                                        localModelMeta(false, m.task, m.audio_type),
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
                                        localModelMeta(false, m.task, m.audio_type),
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
                        // LM Studio dirs are GGUF but rarely carry a -GGUF suffix, so use the shared helper for row,
                        // filter and load path to agree.
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
                                        localModelMeta(false, m.task, m.audio_type),
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
                                        localModelMeta(false, m.task, m.audio_type),
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
                        // A loose ./models/*.gguf loads directly; a GGUF repo dir expands. The variant scanner returns
                        // nothing for a config-less loose file, so expanding it would dead-end.
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
                                        localModelMeta(false, m.task, m.audio_type),
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
                                        localModelMeta(false, m.task, m.audio_type),
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
                              label={curatedRow(id).name}
                              tags={curatedRow(id).tags}
                              hubUrl={hubRepoUrl(id)}
                              alignMeta="hub"
                              showSize={hubRowsShowSize}
                              // A community row without its owner reads as an unsloth upload, and two
                              // publishers would collide.
                              hideOwner={isUnslothOwned(id)}
                              downloaded={downloadedSet.has(id.toLowerCase())}
                              partial={partialSet.has(id.toLowerCase())}
                              partialResumable={partialResumableSet.has(
                                id.toLowerCase(),
                              )}
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
                              gpuGb={isG ? expanderGpuGb : expanderSystemGpuGb}
                              onArrowDownIntoChildren={
                                expandedGguf === id
                                  ? () => focusFirstChildOption(optionKey)
                                  : undefined
                              }
                            />
                            {expandedGguf === id && (
                              <GgufVariantExpander
                                diffusionLoad={diffusionLoad}
                                hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                gpuCount={expanderGpuCount}
                                repoId={id}
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
                    {recommendedHasMore && (
                      <>
                        <div ref={recommendedSentinelRef} className="h-px" />
                        {/* Only while a page is in flight; on hasMore it sat under a usable list. */}
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
                          <ModelRow
                            label={curatedRow(id).name}
                            tags={curatedRow(id).tags}
                            hubUrl={hubRepoUrl(id)}
                            alignMeta="hub"
                            showSize={hubRowsShowSize}
                            downloaded={downloadedSet.has(id.toLowerCase())}
                            partial={partialSet.has(id.toLowerCase())}
                            partialResumable={partialResumableSet.has(
                              id.toLowerCase(),
                            )}
                            capabilities={capsById.get(id)}
                            // Same meta the unfiltered Recommended row shows, so a model keeps its size
                            // chip when reached by typing.
                            // because it was reached by typing.
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
                          />
                          {expandedGguf === id && (
                            <GgufVariantExpander
                              diffusionLoad={diffusionLoad}
                              hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                              gpuCount={expanderGpuCount}
                              repoId={id}
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
                    {searchRowIds.length === 0 && !isLoading ? (
                      filteredRecommendedIds.length === 0 ? (
                        <div className="px-2.5 py-2 text-xs text-muted-foreground">
                          {communityDiscoveryEnabled
                            ? "No matching models."
                            : "No matching Unsloth models."}
                        </div>
                      ) : null
                    ) : (
                      searchRowIds.map((id) => {
                        const vram = vramMap.get(id);
                        const isSearchGguf = isKnownGgufRepo(id);
                        const optionKey = makeModelOptionKey("search-hf", id);
                        return (
                          <div key={id}>
                            <ModelRow
                              label={curatedRow(id).name}
                              tags={curatedRow(id).tags}
                              hubUrl={hubRepoUrl(id)}
                              alignMeta="hub"
                              showSize={hubRowsShowSize}
                              // Typed results are Hub rows like any other, so a repo left
                              // half-downloaded is marked here too. Without it the row reads
                              // as never fetched while the click resumes a download.
                              partial={partialSet.has(id.toLowerCase())}
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
                            />
                            {expandedGguf === id && (
                              <GgufVariantExpander
                                diffusionLoad={diffusionLoad}
                                hostPooledMemory={gpu.loadDeviceSharesHostMemory}
                                gpuCount={expanderGpuCount}
                                repoId={id}
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
    </CapabilityScope.Provider>
  );
}

/** Fine-tuned model rows for the On Device tab's section; plugs into that section's roving list
 *  and shared GGUF-expand state. */
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
        // A checkpoint that fine-tunes a TTS/STT model has to reach the Audio page, since
        // chat/completions cannot serve it. onSelect routes on the pipeline tag, so carry the
        // detected codec as one.
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
