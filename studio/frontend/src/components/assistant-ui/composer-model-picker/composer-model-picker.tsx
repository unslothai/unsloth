// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelMemoryBarFor } from "@/components/model-memory-bar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { audioPickSearch } from "@/features/audio/route-search";
import { useIsAccountOwner } from "@/features/auth";
import {
  type CachedGgufRepo,
  type CachedModelRepo,
  type ChatModelSummary,
  type GgufVariantDetail,
  deleteFineTunedModel,
  revealFineTunedModel,
  useChatActive,
} from "@/features/chat";
import { ApiProviderLogo } from "@/features/chat/api-provider-logo";
import {
  isExternalModelId,
  newConnectionRequest,
  parseExternalModelId,
} from "@/features/chat/external-providers";
import { DROP_CUE_CLASS } from "@/features/chat/hooks/use-sidebar-drag";
import {} from "@/features/chat/model-catalog";
import { providerLogoPath } from "@/features/chat/provider-logo-path";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import {
  type Capability,
  CapabilityGlyph,
  DELETABLE_LOCAL_SOURCES,
  HubFailureHint,
  OwnerAvatar,
  TrainIcon,
  TransportConflictDialog,
  deleteCachedModel,
  deleteLocalModel,
  detectHubCapabilities,
  downloadManager,
  formatBytes,
  formatParamCount,
  hfApiToken,
  isHiddenModelId,
  jobKeyOf,
  listGgufVariants,
  matchesRecommendedBrowse,
  resolveOwnerProviderLogo,
  revealLocalModel,
  useDownloadManagerStore,
  useHfTokenStore,
  useHubInfiniteScroll,
  useHubModelSearch,
  useOnlineStatus,
} from "@/features/hub";
import { hasMlxCardTag, isMlxId } from "@/features/hub/lib/format-filters";
import { isStandaloneGgufPath } from "@/features/hub/lib/model-identity";
import {
  AUDIO_CATALOG,
  ConnectedModelInfoDialog,
  ConnectedModelSettingsDialog,
  type LoraModelOption,
  type ModelConfigFocus,
  ModelConfigPage,
  ModelLoadSettingsAction,
  type ModelOption,
  type ModelPickTarget,
  ModelRowMenu,
  type ModelSelectorChangeMeta,
  type PickerFitStatus,
  VRAM_VERDICT,
  artifactForRepoId,
  audioPickIsRoutable,
  audioPipelineTagFor,
  chatGgufFit,
  curatedAudioInventoryTask,
  describeVariantListingError,
  ggufVariantPickerLabel,
  isMediaCatalogRepo,
  isOverBudget,
  isRecommendableFormat,
  loadStampId,
  loadedAt,
  loraOptionLabel,
  macMlxSttRowIsRunnable,
  mediaPageForTask,
  missingExternalModel,
  modelConfigInstanceKey,
  modelConfigTarget,
  modelConfigTargetIsResident,
  modelDisplayName,
  nativeAudioCheckpointIsLoadable,
  paramsFromId,
  pickerOrderedGgufVariants,
  pinDropAnchor,
  searchRowFitsDevice,
  taskForMediaPick,
  useChatPickerInventory,
  useModelLoadTimes,
  usePinnedConnectedModelsStore,
  usePinnedModelsStore,
  usePinnedRowDrag,
} from "@/features/model-picker";
import { reconcileGgufPinsAfterDelete } from "@/features/model-picker/components/model-selector/reconcile-gguf-pins";
import {
  type NpuModel,
  NpuSetupNotice,
  npuDownloadLabel,
  npuSizeLabel,
  useNpuCatalog,
  useNpuStatus,
} from "@/features/npu";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useGpuInfo, useInferenceGpuInfo } from "@/hooks/use-gpu-info";
import { useVramBudgetFraction } from "@/hooks/use-vram-budget-fraction";
import {
  ChevronDownStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { diffusionRouteSearch } from "@/lib/diffusion-route-search";
import { Folder02Icon } from "@/lib/folder-icons";
import { classifyVariantFitOnCard } from "@/lib/gguf-fit";
import { extractParamLabel } from "@/lib/model-size";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  BotIcon,
  CloudIcon,
  DashboardCircleIcon,
  Download01Icon,
  FilterMailIcon,
  Flag01Icon,
  FlashIcon,
  InformationCircleIcon,
  LaptopIcon,
  Navigation05Icon,
  RemoveCircleIcon,
  Search01Icon,
  Settings01Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import {
  type CSSProperties,
  Fragment,
  type InputHTMLAttributes,
  type ReactNode,
  type Ref,
  memo,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { useAnimatedListHeight } from "./animated-height";
import { useComposerModelPickerSource } from "./context";
import { useCustomFolders } from "./custom-folders";
import { EffortSlider } from "./effort-slider";
import { ListScrollbar } from "./list-scrollbar";
import {
  CONNECTED_TAB,
  DEFAULT_PROVIDERS,
  DEVICE_SORT_DESC,
  DEVICE_SORT_LABEL,
  DEVICE_SORT_ORDER,
  DEVICE_TAB,
  type DeviceSort,
  FAVORITES_TAB,
  FORMAT_FILTER_LABEL,
  type FormatFilter,
  type FormatTone,
  type ListSection,
  PICKER_EDGE,
  PICKER_GAP,
  PICKER_WIDTH,
  type PickerRow,
  RECOMMENDED_TAB,
  SECTION_LABEL,
  chooseQuant,
  dedupeRows,
  discoverUsesFallback,
  downloadedQuants,
  ejectTargetId,
  filterRows,
  findChosenRow,
  isChatTask,
  isTestFixtureRepo,
  localSection,
  matchesFormat,
  orderBySection,
  pickerAlignOffset,
  pickerDropOffset,
  pickerSide,
  pinCoversId,
  readBounded,
  repoSection,
  sectionsHaveRows,
  sortConnectedRows,
  sortDeviceRows,
} from "./picker-model";
import { useComposerReasoning } from "./use-composer-reasoning";

interface Row extends PickerRow {
  loadedId?: string;
  makerName?: string;
  gguf?: CachedGgufRepo;
  cached?: CachedModelRepo;
  lora?: LoraModelOption;
  hub?: ModelOption;
  npu?: NpuModel;
  loadedQuant?: string | null;
  partial?: boolean;
  resumable?: boolean;
  /** Runs on the Audio page: a pick opens it there instead of loading into Chat. */
  media?: "audio";
}

function cachedMeta(row: Row): ModelSelectorChangeMeta {
  return {
    source: "hub",
    isLora: false,
    loadId: row.partial ? undefined : (row.cached?.load_id ?? undefined),
    isDownloaded: !row.partial,
    isGguf: false,
    pipelineTag: row.cached?.task ?? null,
  };
}

const EXT_PREFIX = "ext:";
const QUANT_READ_WORKERS = 6;
const QUANT_FLUSH_MS = 100;
const MAKER_PREFIX = "maker:";
const MAKER_TAB_MIN_MODELS = 3;
const EXPANDED_KEY = "unsloth_composer_picker_expanded";
const TAB_KEY = "unsloth_composer_picker_tab";
const TAB_ORDER_KEY = "unsloth_composer_picker_tab_order";

function readTabOrder(): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(TAB_ORDER_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((v) => typeof v === "string") : [];
  } catch {
    return [];
  }
}

function readExpanded(): boolean {
  try {
    return localStorage.getItem(EXPANDED_KEY) !== "false";
  } catch {
    return true;
  }
}

function writeExpanded(expanded: boolean): void {
  try {
    localStorage.setItem(EXPANDED_KEY, String(expanded));
  } catch {
    // Storage blocked.
  }
}

function readTab(): string | null {
  try {
    return localStorage.getItem(TAB_KEY);
  } catch {
    return null;
  }
}

function writeTab(tab: string): void {
  try {
    localStorage.setItem(TAB_KEY, tab);
  } catch {
    // Storage blocked.
  }
}

const HUB_ID = /^([\w.-]+)\/([\w.-]+)$/;
const LOOKS_GGUF = /gguf/i;
// Local paths: an `mlx` folder anywhere on the path counts, besides the shared leaf-name rule.
const MLX_PATH_SEGMENT = /(^|[-_/])mlx([-_/]|$)/i;

function loraFormat(lora: LoraModelOption): FormatTone {
  if (
    lora.isDirectGguf ||
    lora.isGguf ||
    lora.exportType === "gguf" ||
    LOOKS_GGUF.test(lora.id)
  ) {
    return "gguf";
  }
  if (lora.source !== "local" && lora.exportType !== "merged") {
    return "adapter";
  }
  return MLX_PATH_SEGMENT.test(lora.id) || isMlxId(lora.id)
    ? "mlx"
    : "checkpoint";
}

function loraDetail(lora: LoraModelOption): string {
  if (lora.source === "local") {
    return lora.baseModel || "Local";
  }
  if (lora.source === "exported") {
    return "Exported";
  }
  return lora.exportType === "merged" ? "Fine-tuned" : "LoRA";
}

function loraNeedsQuant(lora: LoraModelOption): boolean {
  return (
    !lora.isDirectGguf &&
    (lora.isGguf === true ||
      lora.exportType === "gguf" ||
      LOOKS_GGUF.test(`${lora.id} ${lora.name}`))
  );
}

function loraMeta(lora: LoraModelOption): ModelSelectorChangeMeta {
  const isLocal = lora.source === "local";
  const isLora =
    !isLocal && lora.exportType !== "merged" && lora.exportType !== "gguf";
  return {
    source: isLocal
      ? "local"
      : lora.source === "exported"
        ? "exported"
        : "lora",
    isLora,
    isDownloaded: true,
    isGguf: lora.isDirectGguf === true,
    ...(lora.audioType
      ? {
          pipelineTag: audioPipelineTagFor(lora.audioType, true, isLora),
          audioType: lora.audioType,
        }
      : {}),
  };
}

const LOCAL_PATH = /^(?:[a-zA-Z]:[\\/]|[\\/]|~)/;

/** A resident with no on-disk row; a path load picks as local. */
function residentMeta(row: Row): ModelSelectorChangeMeta {
  return {
    source: LOCAL_PATH.test(row.id) ? "local" : "hub",
    isLora: false,
    isDownloaded: true,
    isGguf: Boolean(row.loadedQuant),
    ...(row.loadedQuant ? { ggufVariant: row.loadedQuant } : {}),
  };
}

function quantBytes(variant: GgufVariantDetail): number {
  const bytes = variant.download_size_bytes;
  return typeof bytes === "number" && bytes > 0 ? bytes : variant.size_bytes;
}

function rowSource(row: Row): ModelSelectorChangeMeta["source"] {
  if (row.kind !== "localGguf") {
    return "hub";
  }
  return row.lora?.source === "exported" ? "exported" : "local";
}

function ggufMeta(
  row: Row,
  variant: GgufVariantDetail,
  onDisk: boolean,
  isGguf = true,
  hasVision: boolean | undefined = row.gguf?.has_vision,
): ModelSelectorChangeMeta {
  return {
    source: rowSource(row),
    isLora: false,
    loadId: onDisk ? (row.gguf?.load_id ?? undefined) : undefined,
    ggufVariant: variant.quant,
    ggufFilename: variant.filename,
    isDownloaded: onDisk,
    expectedBytes: quantBytes(variant),
    isGguf,
    // Only a known text-only verdict: Chat warns that earlier images may be ignored.
    isVision: hasVision === false ? false : undefined,
    pipelineTag: row.gguf?.task ?? null,
  };
}

interface QuantTarget {
  id: string;
  localPath: string | null;
  local: boolean;
  remote: boolean;
  /** Skip the cache-first listing: for a repo with a quant on disk it lists only that one. */
  everyQuant?: boolean;
}

function quantTarget(row: Row): QuantTarget | null {
  if (row.kind === "gguf" && row.gguf) {
    return {
      id: row.id,
      localPath: row.gguf.load_id || row.gguf.cache_path || null,
      local: false,
      // A partial repo has no clean quant on disk: list them all, so its own can resume.
      remote: row.partial === true,
    };
  }
  if (row.kind === "localGguf") {
    return { id: row.id, localPath: null, local: true, remote: false };
  }
  // MLX precision folders load only with their variant named, downloaded or not.
  if (row.kind === "hub" && (row.hub?.isGguf || row.format === "mlx")) {
    return { id: row.id, localPath: null, local: false, remote: true };
  }
  if (row.kind === "cached" && row.format === "mlx") {
    return {
      id: row.id,
      localPath: null,
      local: false,
      remote: true,
      everyQuant: true,
    };
  }
  return null;
}

async function rowQuants(
  target: QuantTarget,
  signal?: AbortSignal,
): Promise<{
  quants: GgufVariantDetail[];
  /** Interrupted downloads: not loadable, but resumable and deletable. */
  partials: GgufVariantDetail[];
  defaultVariant: string | null;
  mlx: boolean;
  hasVision: boolean | undefined;
}> {
  // The user's token, as search sends it: a gated or private repo lists nothing anonymously.
  const token = useHfTokenStore.getState().token || undefined;
  const res = await listGgufVariants(target.id, token, {
    localOnly: !target.remote,
    preferLocalCache: target.remote && !target.everyQuant,
    localPath: target.localPath,
    includeCacheLocations: true,
    signal,
  });
  // A partial quant cannot load, but a download offer must keep it so the download resumes.
  const listed = Array.isArray(res?.variants) ? res.variants : [];
  const variants = listed.filter((v) => target.remote || v.partial !== true);
  return {
    partials: listed.filter((v) => v.partial === true),
    quants:
      target.remote || target.local ? variants : downloadedQuants(variants),
    defaultVariant:
      typeof res?.default_variant === "string" && res.default_variant
        ? res.default_variant
        : null,
    mlx: res?.variant_format === "mlx",
    hasVision:
      typeof res?.has_vision === "boolean" ? res.has_vision : undefined,
  };
}

function hubParts(
  id: string | undefined,
): { owner: string; name: string } | null {
  const match = id?.trim().match(HUB_ID);
  return match ? { owner: match[1], name: match[2] } : null;
}

type Maker = NonNullable<ReturnType<typeof resolveOwnerProviderLogo>>;

function rowMaker(
  row?: Pick<Row, "kind"> & { id?: string; lora?: LoraModelOption },
): Maker | null {
  if (!row || row.kind === "external") {
    return null;
  }
  const parts = hubParts(row.id) ?? hubParts(row.lora?.baseModel);
  return parts ? resolveOwnerProviderLogo(parts.owner, parts.name) : null;
}

// Logo files pad their art differently: scale each so its drawn part fills the box.
const LOGO_FIT = new Map<string, string>();
const FIT_PX = 64;

function useLogoFit(src: string): string | undefined {
  const [fit, setFit] = useState(() => LOGO_FIT.get(src));
  useEffect(() => {
    if (LOGO_FIT.has(src)) {
      setFit(LOGO_FIT.get(src));
      return;
    }
    let live = true;
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = FIT_PX;
      canvas.height = FIT_PX;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        return;
      }
      const k =
        FIT_PX / Math.max(image.naturalWidth || 1, image.naturalHeight || 1);
      const w = (image.naturalWidth || FIT_PX) * k;
      const h = (image.naturalHeight || FIT_PX) * k;
      ctx.drawImage(image, (FIT_PX - w) / 2, (FIT_PX - h) / 2, w, h);
      const { data } = ctx.getImageData(0, 0, FIT_PX, FIT_PX);
      let left = FIT_PX;
      let right = -1;
      let top = FIT_PX;
      let bottom = -1;
      for (let y = 0; y < FIT_PX; y++) {
        for (let x = 0; x < FIT_PX; x++) {
          if (data[(y * FIT_PX + x) * 4 + 3] > 24) {
            left = Math.min(left, x);
            right = Math.max(right, x);
            top = Math.min(top, y);
            bottom = Math.max(bottom, y);
          }
        }
      }
      let transform = "";
      if (right >= left) {
        const scale = Math.min(
          1.4,
          FIT_PX / Math.max(right - left + 1, bottom - top + 1),
        );
        const dx = FIT_PX / 2 - (left + right + 1) / 2;
        const dy = FIT_PX / 2 - (top + bottom + 1) / 2;
        if (scale > 1.03 || Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          transform = `scale(${scale.toFixed(3)}) translate(${((dx / FIT_PX) * 100).toFixed(1)}%, ${((dy / FIT_PX) * 100).toFixed(1)}%)`;
        }
      }
      LOGO_FIT.set(src, transform);
      if (live) {
        setFit(transform);
      }
    };
    image.src = src;
    return () => {
      live = false;
    };
  }, [src]);
  return fit || undefined;
}

// Fitted marks still differ in weight: spiky ones scale to the solid marks' ink coverage (4x measure).
const OPTICAL_SCALE: Readonly<Record<string, number>> = {
  qwen: 1.05,
  anthropic: 1.06,
};

// ~IQ1_M, the smallest quant Unsloth uploads.
const SMALLEST_QUANT_BYTES_PER_PARAM = 0.3;

function opticalStyle(id: string | undefined): CSSProperties | undefined {
  const k = id ? OPTICAL_SCALE[id] : undefined;
  return k ? { scale: String(k) } : undefined;
}

function providerLogo(providerType: string, name?: string): ReactNode {
  return (
    <span className="flex shrink-0" style={opticalStyle(providerType)}>
      <ApiProviderLogo
        providerType={providerType}
        name={name}
        className="size-[calc(15px*var(--ui-space-scale,1))]"
      />
    </span>
  );
}

function MakerMark({
  provider,
  className,
}: { provider: Maker; className: string }) {
  const base = import.meta.env.BASE_URL;
  const src =
    provider.treatment === "original" && provider.id === "google"
      ? `${base}provider-logos/google-g.svg`
      : `${base}${provider.logoPath.slice(1)}`;
  const fit = useLogoFit(src);
  const k = OPTICAL_SCALE[provider.id];
  const transform = k ? `scale(${k}) ${fit ?? ""}`.trim() : fit;
  if (provider.treatment !== "original") {
    const url = `url(${src})`;
    return (
      <span
        aria-hidden={true}
        className={cn(className, "bg-current")}
        style={{
          maskImage: url,
          WebkitMaskImage: url,
          maskSize: "contain",
          WebkitMaskSize: "contain",
          maskRepeat: "no-repeat",
          WebkitMaskRepeat: "no-repeat",
          maskPosition: "center",
          WebkitMaskPosition: "center",
          transform,
        }}
      />
    );
  }
  return (
    <img
      src={src}
      alt=""
      aria-hidden={true}
      className={cn(className, "object-contain")}
      style={{ transform }}
    />
  );
}

function RowMark({
  row,
}: {
  row?: Pick<Row, "kind" | "providerType" | "providerName"> & {
    id?: string;
    lora?: LoraModelOption;
  };
}) {
  const size = "size-[calc(14px*var(--ui-space-scale,1))] shrink-0";
  const parts =
    row && row.kind !== "external"
      ? (hubParts(row.id) ?? hubParts(row.lora?.baseModel))
      : null;
  const provider = rowMaker(row);
  if (provider) {
    return <MakerMark provider={provider} className={size} />;
  }
  if (parts) {
    // A block box: an inline one sat on the text baseline, about 2px above centre.
    return (
      <span className="flex shrink-0">
        <OwnerAvatar
          owner={parts.owner}
          repoName={parts.name}
          size="xs"
          remote={false}
          className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 rounded-[3px]"
        />
      </span>
    );
  }
  if (row?.kind === "external") {
    return row.providerType ? (
      <span className="flex shrink-0" style={opticalStyle(row.providerType)}>
        <ApiProviderLogo
          providerType={row.providerType}
          name={row.providerName}
          className={size}
        />
      </span>
    ) : (
      <HugeiconsIcon icon={CloudIcon} strokeWidth={1.75} className={size} />
    );
  }
  return (
    <HugeiconsIcon
      icon={row?.kind === "hub" ? Download01Icon : LaptopIcon}
      strokeWidth={1.75}
      className={size}
    />
  );
}

interface SourceTab {
  value: string;
  label: string;
  icon: ReactNode;
  /** A coloured logo: faded, not greyed, while not shown. */
  brand?: boolean;
  /** Maker and connection tabs: draggable, past the pinned four. */
  movable?: boolean;
  logo?: string;
  /** A connection whose logo another tab also wears. */
  cloudBadge?: boolean;
}

function TabButton({
  tab,
  shown,
  onSelect,
  dragging,
  onDragTab,
  drop,
  onDropTarget,
  onDropTab,
}: {
  tab: SourceTab;
  shown: boolean;
  onSelect: () => void;
  dragging: string | null;
  onDragTab: (value: string | null) => void;
  drop: TabDrop | null;
  onDropTarget: (drop: TabDrop | null) => void;
  onDropTab: (from: string, to: TabDrop) => void;
}) {
  const accepts = Boolean(tab.movable && dragging && dragging !== tab.value);
  const edge = drop?.value === tab.value ? drop.edge : null;
  return (
    <Hint label={tab.label}>
      <button
        type="button"
        role="tab"
        aria-selected={shown}
        aria-label={tab.label}
        onClick={onSelect}
        draggable={tab.movable}
        onDragStart={(event) => {
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", tab.value);
          onDragTab(tab.value);
        }}
        onDragEnd={() => {
          onDragTab(null);
          onDropTarget(null);
        }}
        onDragOver={(event) => {
          if (!accepts) {
            return;
          }
          event.preventDefault();
          event.dataTransfer.dropEffect = "move";
          const box = event.currentTarget.getBoundingClientRect();
          onDropTarget({
            value: tab.value,
            edge: event.clientX < box.left + box.width / 2 ? "before" : "after",
          });
        }}
        onDragLeave={() => {
          if (edge) {
            onDropTarget(null);
          }
        }}
        onDrop={(event) => {
          if (accepts && dragging && edge) {
            event.preventDefault();
            onDropTab(dragging, { value: tab.value, edge });
          }
          onDragTab(null);
          onDropTarget(null);
        }}
        className={cn(
          // Sized to the icon: outline glyphs run narrower than logos, so fixed boxes spaced them unevenly.
          "group/tab relative grid h-[calc(32px*var(--ui-space-scale,1))] shrink-0 place-items-center px-[calc(7.5px*var(--ui-space-scale,1))] transition-colors hover:text-foreground",
          shown ? "text-foreground" : "text-muted-foreground",
          dragging === tab.value && "opacity-40",
          edge && TAB_DROP_CUE[edge],
        )}
      >
        {tab.brand ? (
          <span
            className={cn(
              "grid place-items-center transition-opacity",
              !shown && "opacity-55 group-hover/tab:opacity-100",
            )}
          >
            {tab.icon}
            {tab.cloudBadge ? (
              <HugeiconsIcon
                icon={CloudIcon}
                strokeWidth={1.5}
                className="absolute right-[calc(3.25px*var(--ui-space-scale,1))] bottom-[calc(6px*var(--ui-space-scale,1))] size-[calc(10px*var(--ui-space-scale,1))] text-muted-foreground [&_path]:fill-popover dark:[&_path]:fill-card"
              />
            ) : null}
          </span>
        ) : (
          tab.icon
        )}
        <span
          aria-hidden={true}
          className={cn(
            "composer-tab-line absolute bottom-0 left-1/2 h-[calc(2px*var(--ui-space-scale,1))] w-[calc(16px*var(--ui-space-scale,1))] -translate-x-1/2 rounded-full transition-opacity duration-150",
            shown ? "opacity-100" : "opacity-0",
          )}
        />
      </button>
    </Hint>
  );
}

function FitBadge({
  status,
  revealOnHover = false,
  compact = false,
}: {
  status: PickerFitStatus | null | undefined;
  revealOnHover?: boolean;
  compact?: boolean;
}) {
  const verdict = status ? VRAM_VERDICT[status] : null;
  if (!verdict) {
    return null;
  }
  return (
    <Tooltip delayDuration={0} disableHoverableContent={true}>
      <TooltipTrigger asChild={true}>
        <span
          role="img"
          aria-label={verdict.label}
          className={cn(
            "grid shrink-0 place-items-center",
            compact
              ? "size-[calc(12px*var(--ui-space-scale,1))]"
              : "size-[calc(20px*var(--ui-space-scale,1))]",
            verdict.tone,
            revealOnHover &&
              "opacity-0 transition-opacity group-hover/row:opacity-100 group-data-[active=true]/row:opacity-100",
          )}
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.8}
            className="size-[calc(12px*var(--ui-space-scale,1))]"
          />
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {verdict.hint}
      </TooltipContent>
    </Tooltip>
  );
}

// The hover circle is drawn apart from the hit box, so the narrow ⋮ box keeps even glyph gaps.
const ACTION =
  "relative flex h-[calc(20px*var(--ui-space-scale,1))] shrink-0 items-center justify-center rounded-none text-muted-foreground transition-colors hover:bg-transparent hover:text-foreground dark:hover:bg-transparent before:pointer-events-none before:absolute before:top-1/2 before:left-1/2 before:size-[calc(20px*var(--ui-space-scale,1))] before:-translate-x-1/2 before:-translate-y-1/2 before:rounded-full before:transition-colors before:content-[''] hover:before:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:hover:before:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] data-[state=open]:before:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:data-[state=open]:before:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] [&>svg]:relative";
const ACTION_ROUND = `${ACTION} w-[calc(20px*var(--ui-space-scale,1))]`;
const FAVORITE_STAR =
  "text-[rgb(0_0_0_/_0.42)] hover:text-[rgb(0_0_0_/_0.62)] dark:text-[rgb(255_255_255_/_0.5)] dark:hover:text-[rgb(255_255_255_/_0.72)]";
// Lifts the ⋮ ink 0.3px so the hover circle centres on it.
const ACTION_DOTS = `${ACTION} w-[calc(15px*var(--ui-space-scale,1))] [&>svg]:-translate-y-[0.3px]`;
const FAVORITE_DROP_CUE_BASE =
  "before:pointer-events-none before:absolute before:inset-x-2 before:z-10 before:h-0 before:border-t-[1.5px] before:border-primary before:content-['']";
const FAVORITE_DROP_CUE = {
  top: `${FAVORITE_DROP_CUE_BASE} before:top-0`,
  bottom: `${FAVORITE_DROP_CUE_BASE} before:bottom-0`,
} as const;
// The favorites' drop line, stood upright between tabs.
const TAB_DROP_CUE_BASE =
  "before:pointer-events-none before:absolute before:inset-y-[calc(6px*var(--ui-space-scale,1))] before:z-10 before:w-0 before:border-l-[1.5px] before:border-primary before:content-['']";
const TAB_DROP_CUE = {
  before: `${TAB_DROP_CUE_BASE} before:left-0`,
  after: `${TAB_DROP_CUE_BASE} before:right-0`,
} as const;
type TabDrop = { value: string; edge: keyof typeof TAB_DROP_CUE };

const CURRENT_ROW_BG =
  "bg-[color-mix(in_oklab,var(--sidebar-accent),black_2.5%)] hover:bg-[color-mix(in_oklab,var(--sidebar-accent),black_2.5%)] dark:bg-[color-mix(in_oklab,var(--sidebar-accent),white_5%)] dark:hover:bg-[color-mix(in_oklab,var(--sidebar-accent),white_5%)]";

const DIM_CLASS =
  "opacity-55 transition-opacity group-hover/row:opacity-100 dark:opacity-50";

function QuantPill({ quant }: { quant: string }) {
  return (
    <span className="inline-flex h-[calc(18px*var(--ui-space-scale,1))] max-w-[calc(100px*var(--ui-space-scale,1))] shrink-0 items-center truncate rounded-full bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] px-1.5 font-mono text-ui-9 text-muted-foreground dark:bg-muted">
      {quant}
    </span>
  );
}

function ParamChip({ label }: { label: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-[calc(18px*var(--ui-space-scale,1))] shrink-0 items-center whitespace-nowrap rounded-md border border-[rgb(0_0_0_/_0.16)] px-1.5 text-ui-10 dark:border-[rgb(255_255_255_/_0.2)] font-medium text-muted-foreground",
        label.includes(".") ? "tracking-[-0.02em]" : "tracking-[0.02em]",
      )}
    >
      {label}
    </span>
  );
}

const JUMP_LABEL: Partial<Record<ListSection, string>> = {
  other: "Go to other models",
  fineTuned: "Go to trained models",
  custom: "Go to custom folders",
};

function AmberTag({ label, hint }: { label: string; hint: string }) {
  return (
    <Hint label={hint}>
      <span className="shrink-0 text-ui-11 font-medium text-amber-700 dark:text-amber-300">
        {label}
      </span>
    </Hint>
  );
}

const STATUS_TONE = {
  loaded:
    "bg-[color-mix(in_srgb,var(--status-success)_16%,transparent)] text-status-success",
  green:
    "bg-emerald-500/10 text-emerald-700 dark:bg-emerald-400/15 dark:text-emerald-300",
  amber:
    "bg-amber-500/12 text-amber-800 dark:bg-amber-400/15 dark:text-amber-300",
  accent: "bg-primary/10 text-primary dark:bg-primary/15",
} as const;

function StatusTag({
  tone,
  label,
  hint,
  lead,
}: {
  tone: keyof typeof STATUS_TONE;
  label: string;
  hint?: string;
  lead?: ReactNode;
}) {
  const tag = (
    <span
      className={cn(
        "inline-flex h-[calc(16px*var(--ui-space-scale,1))] shrink-0 items-center gap-1 rounded-full px-[calc(7px*var(--ui-space-scale,1))] font-sans text-ui-10 font-medium leading-none",
        STATUS_TONE[tone],
      )}
    >
      {lead}
      {label}
    </span>
  );
  return hint ? <Hint label={hint}>{tag}</Hint> : tag;
}

function QuantStatus({
  variant,
  loaded,
  recommended,
  onDevice,
}: {
  variant: GgufVariantDetail;
  loaded: boolean;
  recommended: boolean;
  /** Off Discover every quant is on disk, and a dot marks the loaded one. */
  onDevice: boolean;
}) {
  const tags: ReactNode[] = [];
  if (onDevice && loaded) {
    tags.push(
      <Hint key="loaded" label="Loaded">
        <span className="grid size-[calc(14px*var(--ui-space-scale,1))] shrink-0 place-items-center">
          <span className="size-[calc(6px*var(--ui-space-scale,1))] rounded-full bg-status-success" />
        </span>
      </Hint>,
    );
  } else if (onDevice && variant.downloaded) {
    // No tag: everything here is downloaded.
  } else if (loaded) {
    tags.push(
      <StatusTag
        key="loaded"
        tone="loaded"
        label="Loaded"
        lead={
          <span className="size-[calc(5px*var(--ui-space-scale,1))] rounded-full bg-current" />
        }
      />,
    );
  } else if (variant.downloaded) {
    tags.push(<StatusTag key="downloaded" tone="green" label="Downloaded" />);
  } else if (variant.partial === true) {
    tags.push(
      <StatusTag
        key="partial"
        tone="amber"
        label="Partial"
        hint={PARTIAL_HINT.variant}
      />,
    );
  } else if (recommended) {
    tags.push(
      <StatusTag key="recommended" tone="accent" label="Recommended" />,
    );
  }
  if ((loaded || variant.downloaded) && variant.update_available) {
    tags.push(
      <StatusTag key="update" tone="amber" label="Update" hint={UPDATE_HINT} />,
    );
  }
  return tags.length > 0 ? <>{tags}</> : null;
}

const PARTIAL_HINT = {
  resumable: "Download incomplete. Picking it resumes the download.",
  restart: "Download incomplete. Picking it downloads it again.",
  variant: "Download incomplete. Picking it finishes the download.",
};
const UPDATE_HINT =
  "A newer upload is on Hugging Face. Update it from the menu.";

function Hint({
  label,
  side = "top",
  children,
}: {
  label: string;
  side?: "top" | "left";
  children: ReactNode;
}) {
  return (
    <Tooltip delayDuration={0} disableHoverableContent={true}>
      <TooltipTrigger asChild={true}>{children}</TooltipTrigger>
      <TooltipContent side={side} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

function LoadedDot({ label = "Loaded" }: { label?: string }) {
  return (
    <Hint label={label}>
      <span
        role="img"
        aria-label={label}
        className="-mx-0.5 grid size-[calc(18px*var(--ui-space-scale,1))] shrink-0 place-items-center"
      >
        <span className="size-[calc(6px*var(--ui-space-scale,1))] rounded-full bg-status-success" />
      </span>
    </Hint>
  );
}

function SwitchRow({
  label,
  checked,
  disabled,
  hint,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  hint?: string;
  onChange: (next: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="flex h-[calc(32px*var(--ui-space-scale,1))] items-center gap-2 rounded-full px-3 text-ui-14 transition-colors hover:bg-sidebar-accent">
      <label
        htmlFor={id}
        className="flex flex-1 cursor-pointer items-center gap-2 self-stretch"
      >
        {label}
        {hint ? (
          <span className="text-ui-12 text-muted-foreground">{hint}</span>
        ) : null}
      </label>
      <Switch
        id={id}
        size="sm"
        checked={checked}
        disabled={disabled}
        onCheckedChange={onChange}
      />
    </div>
  );
}

const MENU_ITEM =
  "min-h-0 gap-2 rounded-[calc(8px*var(--ui-space-scale,1))] py-[calc(5px*var(--ui-space-scale,1))] pr-8 pl-2.5 text-ui-12 [&_[data-slot=dropdown-menu-radio-item-indicator]]:right-2.5 [&_[data-slot=dropdown-menu-radio-item-indicator]_svg]:size-3.5";
const MENU_LABEL =
  "px-2.5 pt-[calc(6px*var(--ui-space-scale,1))] pb-[calc(3px*var(--ui-space-scale,1))] text-ui-10";
const FORMAT_DOT: Partial<Record<FormatFilter, string>> = {
  gguf: "bg-format-gguf",
  mlx: "bg-format-mlx",
  safetensors: "bg-format-checkpoint",
  npu: "bg-format-npu",
};

// Modalities in, then out, then what the model does; Multilingual and Conversational are not modalities.
const CAPABILITY_ORDER: readonly Capability["key"][] = [
  "vision",
  "audio",
  "reasoning",
  "tools",
  "code",
  "diffusion",
  "embedding",
];
const SHOWN_CAPABILITIES: ReadonlySet<Capability["key"]> = new Set(
  CAPABILITY_ORDER,
);

const FORMAT_NAME: Record<FormatTone, string> = {
  gguf: "GGUF",
  mlx: "MLX",
  checkpoint: "Safetensors",
  adapter: "LoRA adapter",
};

function ModelFacts({ facts }: { facts: readonly [string, ReactNode][] }) {
  return (
    <dl className="flex flex-col gap-1 px-2.5 pt-1.5 pb-2 text-ui-12">
      {facts.map(([label, value]) => (
        <div key={label} className="flex items-baseline justify-between gap-3">
          <dt className="shrink-0 text-muted-foreground">{label}</dt>
          <dd className="flex min-w-0 flex-wrap justify-end gap-x-2.5 gap-y-1 text-right text-foreground">
            {typeof value === "string" ? (
              <span className="min-w-0 truncate">{value}</span>
            ) : (
              value
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

// Scaled to the Vision eye's visual weight (sqrt of ink width x height, 16.7 of 24).
const GLYPH_SCALE: Partial<Record<Capability["key"], string>> = {
  conversational: "scale-[0.86]",
  audio: "scale-[0.95]",
  tools: "scale-[0.88]",
  embedding: "scale-[0.88]",
  diffusion: "scale-[0.88]",
  multilingual: "scale-[0.91]",
  reasoning: "scale-[0.86]",
};
function glyphScale(key: Capability["key"], label: string): string | undefined {
  return key === "diffusion" && label === "Video gen"
    ? "scale-[0.88]"
    : GLYPH_SCALE[key];
}

function modelFacts(
  row: Row,
  onDisk: readonly GgufVariantDetail[],
  known: ChatModelSummary | undefined,
): [string, ReactNode][] {
  const facts: [string, ReactNode][] = [];
  const format =
    row.kind === "npu" ? "NPU" : row.format ? FORMAT_NAME[row.format] : null;
  const withFormat = (text: string) => (format ? `${text} ${format}` : text);
  if (onDisk.length === 1) {
    facts.push(["Quant", withFormat(onDisk[0].quant)]);
  } else if (onDisk.length > 1) {
    facts.push(["Quants", `${withFormat(String(onDisk.length))} on disk`]);
  } else if (format) {
    facts.push(["Format", format]);
  }
  const size = onDisk.length === 1 ? quantBytes(onDisk[0]) : row.sizeBytes;
  if (size && size > 0) {
    facts.push(["Disk space", formatBytes(size)]);
  }
  const params = extractParamLabel(row.lora?.baseModel ?? row.id);
  if (params) {
    facts.push(["Parameters", params]);
  }
  const task = row.gguf?.task ?? row.cached?.task ?? undefined;
  const id = row.lora?.baseModel ? `${row.lora.baseModel} ${row.id}` : row.id;
  // The backend's own flags (projector, audio and video towers) beat the name, which only guesses.
  const vision =
    row.npu?.supports_vision ??
    row.gguf?.has_vision ??
    (known?.isVision ? true : undefined);
  const found = detectHubCapabilities(undefined, task, id).filter(
    (c) =>
      SHOWN_CAPABILITIES.has(c.key) &&
      !(c.key === "vision" && vision === false),
  );
  const has = (key: Capability["key"]) => found.some((c) => c.key === key);
  if (vision === true && !has("vision")) {
    found.push({ key: "vision", label: "Vision" });
  }
  if ((known?.hasAudioInput || known?.isAudio) && !has("audio")) {
    found.push({ key: "audio", label: "Audio" });
  }
  if (row.npu?.supports_reasoning && !has("reasoning")) {
    found.push({ key: "reasoning", label: "Reasoning" });
  }
  found.sort(
    (a, b) => CAPABILITY_ORDER.indexOf(a.key) - CAPABILITY_ORDER.indexOf(b.key),
  );
  const mark = (key: string, glyph: ReactNode, label: string) => (
    <span key={key} className="inline-flex items-center gap-1">
      {glyph}
      {label}
    </span>
  );
  const hub = (key: Capability["key"], label: string) => (
    <CapabilityGlyph
      capability={{ key, label }}
      className={glyphScale(key, label)}
    />
  );
  const pill = (c: Capability) =>
    mark(
      c.key,
      <CapabilityGlyph capability={c} className={glyphScale(c.key, c.label)} />,
      c.label,
    );
  const inputs = new Set<Capability["key"]>(["vision", "audio"]);
  const supports: ReactNode[] = [
    mark("text", hub("conversational", "Text"), "Text"),
    ...found.filter((c) => inputs.has(c.key)).map(pill),
    ...(known?.hasVideoInput
      ? [mark("video", hub("diffusion", "Video gen"), "Video")]
      : []),
    ...(known?.speaks
      ? [mark("speech", hub("audio", "Speech"), "Speech")]
      : []),
    ...found.filter((c) => !inputs.has(c.key)).map(pill),
  ];
  facts.push(["Supports", supports]);
  if (row.partial) {
    facts.push(["Download", "Incomplete"]);
  } else if (row.loaded) {
    facts.push([
      "Status",
      row.loadedQuant ? `Loaded (${row.loadedQuant})` : "Loaded",
    ]);
  }
  return facts;
}

const SECTION_GLYPH = "size-[calc(12px*var(--ui-space-scale,1))]";
const SECTION_ICON: Partial<Record<ListSection, ReactNode>> = {
  other: (
    <HugeiconsIcon
      icon={Flag01Icon}
      strokeWidth={1.75}
      className={SECTION_GLYPH}
    />
  ),
  fineTuned: (
    <HugeiconsIcon
      icon={TrainIcon}
      strokeWidth={1.75}
      className={SECTION_GLYPH}
    />
  ),
  custom: (
    <HugeiconsIcon
      icon={Folder02Icon}
      strokeWidth={1.75}
      className={SECTION_GLYPH}
    />
  ),
};

function SectionHeader({
  section,
  label,
  icon,
  divider,
  collapsed,
  onToggle,
  actions,
  trailing,
}: {
  section: string;
  label: string;
  icon?: ReactNode;
  divider: boolean;
  collapsed: boolean;
  onToggle: () => void;
  actions?: ReactNode;
  trailing?: ReactNode;
}) {
  return (
    <div
      data-section={section}
      className={cn(
        "flex shrink-0 flex-col",
        divider && "mt-[calc(11px*var(--ui-space-scale,1))]",
      )}
    >
      <div className="flex h-[calc(26px*var(--ui-space-scale,1))] items-center gap-1 pr-[calc(6px*var(--ui-space-scale,1))] pl-3">
        <button
          type="button"
          tabIndex={-1}
          onClick={onToggle}
          aria-expanded={!collapsed}
          className="flex min-w-0 items-center gap-1.5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground"
        >
          {icon}
          <span className="truncate">{label}</span>
        </button>
        {actions ? (
          <span className="flex shrink-0 items-center">{actions}</span>
        ) : null}
        {trailing ? (
          <span className="ml-auto flex shrink-0 items-center">{trailing}</span>
        ) : null}
        <Hint label={collapsed ? "Expand" : "Collapse"}>
          <button
            type="button"
            onClick={onToggle}
            aria-label={`${collapsed ? "Expand" : "Collapse"} ${label}`}
            className={cn(
              trailing ? null : "ml-auto",
              "grid size-[calc(20px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground",
            )}
          >
            <HugeiconsIcon
              icon={
                collapsed ? ChevronRightStandardIcon : ChevronDownStandardIcon
              }
              strokeWidth={1.75}
              className="size-3"
            />
          </button>
        </Hint>
      </div>
    </div>
  );
}

const OPTION_ID = "composer-model-option";
const NO_QUANTS: GgufVariantDetail[] = [];

/** Re-renders a row only when `deps` change, which must cover everything its markup and handlers read. */
const MemoRow = memo(
  function MemoRow({ render }: { render: () => ReactNode; deps: unknown[] }) {
    return render();
  },
  (prev, next) =>
    prev.deps.length === next.deps.length &&
    prev.deps.every((dep, i) => Object.is(dep, next.deps[i])),
);
const SEARCH_DEBOUNCE_MS = 250;
type NavKey = "ArrowDown" | "ArrowUp" | "Enter";

/** Holds the draft, so a keystroke re-renders this field and not every row. */
function SearchField({
  value,
  onCommit,
  onNavKey,
  inputRef,
  ...rest
}: {
  value: string;
  onCommit: (next: string) => void;
  onNavKey: (key: NavKey) => void;
  inputRef: Ref<HTMLInputElement>;
} & Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "value" | "onChange" | "onKeyDown"
>) {
  const [draft, setDraft] = useState(value);
  const committed = useRef(value);
  // A reset from outside (a new tab, a pick) replaces the draft; the echo of our own commit does not.
  if (value !== committed.current) {
    committed.current = value;
    if (draft !== value) {
      setDraft(value);
    }
  }
  const commit = (next: string) => {
    committed.current = next;
    onCommit(next);
  };
  const commitLatest = useRef(commit);
  commitLatest.current = commit;
  useEffect(() => {
    if (draft === committed.current) {
      return;
    }
    const id = window.setTimeout(
      () => commitLatest.current(draft),
      SEARCH_DEBOUNCE_MS,
    );
    return () => window.clearTimeout(id);
  }, [draft]);
  return (
    <input
      ref={inputRef}
      {...rest}
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => {
        const key = event.key;
        if (event.nativeEvent.isComposing || event.keyCode === 229) {
          return;
        }
        if (key !== "ArrowDown" && key !== "ArrowUp" && key !== "Enter") {
          return;
        }
        event.preventDefault();
        if (draft !== committed.current) {
          commit(draft);
          window.setTimeout(() => onNavKey(key), 0);
          return;
        }
        onNavKey(key);
      }}
    />
  );
}

export function ComposerModelPicker({
  side = "top",
  align = "end",
}: { side?: "top" | "bottom"; align?: "start" | "end" }) {
  const rawSource = useComposerModelPickerSource();
  // Chat cannot load an audio model, so its pick opens the page that runs it; a ref, as routing
  // reads inventory this render has not built.
  const routeMediaPick = useRef<
    (id: string, meta: ModelSelectorChangeMeta) => boolean
  >(() => false);
  const source = useMemo(
    () =>
      rawSource
        ? {
            ...rawSource,
            onPick: (id: string, meta: ModelSelectorChangeMeta) => {
              if (!routeMediaPick.current(id, meta)) {
                rawSource.onPick(id, meta);
              }
            },
          }
        : null,
    [rawSource],
  );
  const navigate = useNavigate();
  const reasoning = useComposerReasoning();
  const pane = source?.selection;
  const runtimeCheckpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const runtimeVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const runtimeLoading = useChatRuntimeStore((s) => s.modelLoading);
  const checkpoint = pane ? pane.id : runtimeCheckpoint;
  const activeVariant = pane ? pane.ggufVariant : runtimeVariant;
  const modelLoading = pane ? pane.loading === true : runtimeLoading;
  const loadedModels = useChatRuntimeStore((s) => s.loadedModels);
  const chatModels = useChatRuntimeStore((s) => s.models);
  const pinned = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const unpinRepo = usePinnedModelsStore((s) => s.unpinRepo);
  const pinnedConnected = usePinnedConnectedModelsStore((s) => s.pinned);
  const togglePinnedConnected = usePinnedConnectedModelsStore(
    (s) => s.togglePinnedConnected,
  );

  const [localOpen, setLocalOpen] = useState(false);
  const open = source ? source.open : localOpen;
  const setOpen = (next: boolean) => {
    if (source) {
      source.onOpenChange(next);
    } else {
      setLocalOpen(next);
    }
  };
  const [configTarget, setConfigTarget] = useState<ModelPickTarget | null>(
    null,
  );
  const [configFocus, setConfigFocus] = useState<ModelConfigFocus | null>(null);
  const [configStandalone, setConfigStandalone] = useState(false);
  const [connectedSettings, setConnectedSettings] = useState<{
    row: Row;
    view: "settings" | "info";
  } | null>(null);
  // Portaled outside the panel, so they would outlive Chat going off-route; the picker closes then too.
  const pageActive = useChatActive();
  useEffect(() => {
    if (!pageActive) {
      setConnectedSettings(null);
    }
  }, [pageActive]);
  const providers = useExternalProvidersStore((s) => s.providers);
  const [updateConflictKey, setUpdateConflictKey] = useState<string | null>(
    null,
  );
  const updateConflict = useDownloadManagerStore((state) =>
    updateConflictKey
      ? (state.conflicts[updateConflictKey]?.info ?? null)
      : null,
  );
  const openConfig = (target: ModelPickTarget) => {
    setConfigTarget(target);
    setConfigFocus(null);
    setConfigStandalone(false);
  };
  const [expanded, setExpandedState] = useState(readExpanded);
  const setExpanded = useCallback((next: boolean) => {
    setExpandedState(next);
    writeExpanded(next);
  }, []);
  const [tab, setTabState] = useState<string | null>(null);
  const [lastTab, setLastTab] = useState(readTab);
  const setTab = useCallback((next: string) => {
    setTabState(next);
    setLastTab(next);
    writeTab(next);
  }, []);
  const [onRecommended, setOnRecommended] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [highlight, setHighlightState] = useState(false);
  const highlightRef = useRef(false);
  const setHighlight = useCallback((next: boolean) => {
    highlightRef.current = next;
    setHighlightState(next);
  }, []);
  const [resolving, setResolving] = useState<string | null>(null);
  const [deviceQuants, setDeviceQuants] = useState<
    ReadonlyMap<string, GgufVariantDetail[]>
  >(() => new Map());
  const [devicePartials, setDevicePartials] = useState<
    ReadonlyMap<string, GgufVariantDetail[]>
  >(() => new Map());
  const [foldedQuants, setFoldedQuants] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [offer, setOffer] = useState<{
    key: string;
    quants: GgufVariantDetail[] | null;
    defaultVariant: string | null;
    failed: boolean;
    mlx?: boolean;
    hasVision?: boolean;
    error?: string;
    target?: QuantTarget;
  } | null>(null);
  const offerAbort = useRef<AbortController | null>(null);
  const [quants, setQuants] = useState<{
    key: string;
    list: GgufVariantDetail[];
  } | null>(null);
  const [quantsOpen, setQuantsOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  // State as well as the ref: the panel mounts after this render, and the glide must attach then.
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const [tabStrip, setTabStrip] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!tabStrip) {
      return;
    }
    const update = () => {
      const max = tabStrip.scrollWidth - tabStrip.clientWidth;
      tabStrip.toggleAttribute("data-fade-left", tabStrip.scrollLeft > 1);
      tabStrip.toggleAttribute(
        "data-fade-right",
        tabStrip.scrollLeft < max - 1,
      );
    };
    update();
    tabStrip.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(tabStrip);
    return () => {
      tabStrip.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [tabStrip]);

  const inventory = useChatPickerInventory({ enabled: open });
  const [npuStatus, setNpuStatus] = useNpuStatus(open);
  const npuCatalog = useNpuCatalog(
    npuStatus?.supported === true
      ? { status: npuStatus, onStatusChange: setNpuStatus }
      : undefined,
  );
  const npuModels = npuCatalog?.models ?? null;
  const online = useOnlineStatus();
  const hfToken = useHfTokenStore((s) => s.token);
  const gpuInfo = useGpuInfo(open);
  const quantFormats = gpuInfo.checkpointQuantFormats;
  const isMac = usePlatformStore((s) => s.deviceType === "mac");
  // The field already waits SEARCH_DEBOUNCE_MS; together they keep the Hub's 300 ms.
  const hubQuery = useDebouncedValue(
    onRecommended ? query.trim() : "",
    300 - SEARCH_DEBOUNCE_MS,
  );
  const hubSearch = useHubModelSearch(hubQuery, {
    ownerScope: hubQuery ? "all" : "unsloth",
    sortBy: "fresh",
    sortDirection: "desc",
    pinUnslothFirst: true,
    keepUnsupportedTags: true,
    hideEmptyUploads: true,
    accessToken: hfApiToken(hfToken),
    enabled: open && online && onRecommended,
  });
  const hubResults = hubSearch.results;
  const inferenceGpu = useInferenceGpuInfo(open);
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  const keepModelsLoaded = useChatRuntimeStore((s) => s.keepModelsLoaded);
  const expandQuantizations = useChatRuntimeStore((s) => s.expandQuantizations);
  const fitOnDeviceOnly = useChatRuntimeStore((s) => s.fitOnDeviceOnly);
  const setFitOnDeviceOnly = useChatRuntimeStore((s) => s.setFitOnDeviceOnly);
  const showMemoryBar = useChatRuntimeStore((s) => s.showMemoryBar);
  const loadFreesResident =
    !keepModelsLoaded &&
    typeof checkpoint === "string" &&
    checkpoint !== "" &&
    !isExternalModelId(checkpoint);
  const fitOf = useCallback(
    (sizeBytes: number | undefined) =>
      chatGgufFit(sizeBytes, inferenceGpu, {
        budgetFraction,
        loadFreesResident,
      }),
    [inferenceGpu, budgetFraction, loadFreesResident],
  );
  // The header picker's quant budget: no verdict means it fits.
  const quantBudget = useMemo(
    () => ({
      fitOf: (sizeBytes: number) => fitOf(sizeBytes) ?? "fits",
      anyBudget:
        inferenceGpu.memoryTotalGb > 0 || inferenceGpu.systemRamAvailableGb > 0,
      budgetKnown: inferenceGpu.budgetKnown,
      cardGb: 0,
    }),
    [fitOf, inferenceGpu],
  );
  const [listedFits, setListedFits] = useState<
    ReadonlyMap<string, ReturnType<typeof fitOf>>
  >(new Map());
  const hubFit = useMemo(() => {
    const map = new Map<string, ReturnType<typeof fitOf>>();
    for (const r of hubResults) {
      if (!r.isGguf) {
        continue;
      }
      const id = r.id.toLowerCase();
      // Once its quants are listed, a repo with one that runs comfortably is not flagged.
      if (listedFits.has(id)) {
        map.set(id, listedFits.get(id) ?? null);
        continue;
      }
      // Judged by its smallest quant: a repo is flagged only when nothing in it could run. The
      // search's size is the unquantized weights (~2 bytes a parameter), so it is scaled down too.
      const params = r.totalParams ?? paramsFromId(r.id);
      map.set(
        id,
        fitOf(
          params
            ? params * SMALLEST_QUANT_BYTES_PER_PARAM
            : r.estimatedSizeBytes &&
                (r.estimatedSizeBytes / 2) * SMALLEST_QUANT_BYTES_PER_PARAM,
        ),
      );
    }
    return map;
  }, [hubResults, fitOf, listedFits]);
  const noteListedFit = useCallback(
    (repoId: string, list: GgufVariantDetail[]) => {
      const id = repoId.toLowerCase();
      // As its quant rows read: one with no warning (it fits, or runs from RAM) clears the repo.
      const runs = list.some((v) => {
        const fit = classifyVariantFitOnCard(quantBudget.fitOf, v, 0);
        return fit === "fits" || fit === "ram";
      });
      setListedFits((now) =>
        new Map(now).set(
          id,
          runs ? null : (now.get(id) ?? hubFit.get(id) ?? null),
        ),
      );
    },
    [quantBudget, hubFit],
  );
  // Non-GGUF rows have no quant to size: the header picker's checkpoint rule judges them.
  const hubCheckpointFits = useMemo(() => {
    const map = new Map<string, boolean>();
    for (const r of hubResults) {
      if (r.isGguf) {
        continue;
      }
      map.set(
        r.id.toLowerCase(),
        searchRowFitsDevice(
          {
            id: r.id,
            totalParams: r.totalParams,
            estimatedSizeBytes: r.estimatedSizeBytes,
            quantMethod: r.quantMethod,
          },
          {
            isGguf: false,
            gpu: gpuInfo,
            inferenceGpu,
            taskScoped: false,
            budgetFraction,
          },
        ),
      );
    }
    return map;
  }, [hubResults, gpuInfo, inferenceGpu, budgetFraction]);
  const hubParams = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of hubResults) {
      const label = r.totalParams
        ? formatParamCount(r.totalParams)
        : extractParamLabel(r.id);
      if (label) {
        map.set(r.id.toLowerCase(), label);
      }
    }
    return map;
  }, [hubResults]);
  const noGpu = !(inferenceGpu.memoryTotalGb > 0);
  const { scrollRef: hubScrollRef, sentinelRef: hubSentinelRef } =
    useHubInfiniteScroll(hubSearch.fetchMore, hubSearch.scannedCount, {
      enabled: open && online && onRecommended && hubSearch.hasMore,
      isFetching: hubSearch.isLoading || hubSearch.isLoadingMore,
      resultCount: hubResults.length,
      resetKey: hubQuery,
    });
  const attachList = useCallback(
    (node: HTMLDivElement | null) => {
      listRef.current = node;
      hubScrollRef.current = node;
      setListEl(node);
    },
    [hubScrollRef],
  );
  const recommended = useMemo<ModelOption[]>(() => {
    const fromHub = hubResults
      .filter(
        (r) =>
          (matchesRecommendedBrowse(r, quantFormats, !hubQuery) ||
            // A Mac also runs MLX and plain safetensors; the browse gates still apply.
            (isMac &&
              isRecommendableFormat(r.id, r.isGguf, true, r.quantMethod) &&
              matchesRecommendedBrowse(
                { ...r, isGguf: true },
                quantFormats,
                !hubQuery,
              ))) &&
          isChatTask(r.pipelineTag) &&
          !isMediaCatalogRepo(r.id) &&
          !isHiddenModelId(r.id) &&
          !isTestFixtureRepo(r.id),
      )
      .map((r) => ({ id: r.id, name: r.id, isGguf: r.isGguf }));
    if (
      fromHub.length > 0 ||
      !discoverUsesFallback(hubQuery, hubResults.length)
    ) {
      return fromHub;
    }
    return (source?.models ?? []).filter((m) => !isHiddenModelId(m.id));
  }, [hubResults, quantFormats, hubQuery, isMac, source?.models]);
  // Hub results marked MLX only by their card, which the ModelOption rows drop.
  const mlxTaggedHub = useMemo(
    () =>
      new Set(
        hubResults
          .filter((r) => hasMlxCardTag(r))
          .map((r) => r.id.toLowerCase()),
      ),
    [hubResults],
  );
  const loraModels = source?.loraModels;
  const externalModels = source?.externalModels;

  // The inventory often reports an audio GGUF as text-generation; the audio catalog's task wins.
  const mediaTaskById = useMemo(() => {
    const byId = new Map<string, string>();
    const put = (
      ids: readonly (string | null | undefined)[],
      inventoryTask: string | null | undefined,
      artifactId: string | null | undefined,
    ) => {
      const artifact = artifactId
        ? artifactForRepoId(artifactId, AUDIO_CATALOG)
        : null;
      const task = curatedAudioInventoryTask({
        inventoryTask,
        isExactCatalogArtifact: Boolean(artifact),
        catalogScope: artifact?.group.scope,
        catalogTask: artifact?.group.task,
      });
      if (task) {
        for (const id of ids) {
          if (id) {
            byId.set(id.toLowerCase(), task);
          }
        }
      }
    };
    for (const repo of [...inventory.cachedGguf, ...inventory.cachedModels]) {
      put([repo.repo_id], repo.task, repo.repo_id);
    }
    // A local row's id is its path; model_id is its HF-style name.
    for (const m of inventory.localModels) {
      put([m.id, m.model_id], m.task, m.model_id);
    }
    return byId;
  }, [inventory.cachedGguf, inventory.cachedModels, inventory.localModels]);
  // Compare and the API page load what they pick, so they list no audio models.
  const listsAudio = !pane;

  const rows = useMemo<Row[]>(() => {
    const device: Row[] = [];
    const isAudio = (id: string, task: string | null | undefined) =>
      listsAudio &&
      mediaPageForTask(
        taskForMediaPick(task, mediaTaskById.get(id.toLowerCase())),
      ) === "audio";
    for (const repo of inventory.cachedGguf) {
      const audio = isAudio(repo.repo_id, repo.task);
      if (
        !audio &&
        (!isChatTask(repo.task) || isMediaCatalogRepo(repo.repo_id))
      ) {
        continue;
      }
      device.push({
        media: audio ? "audio" : undefined,
        key: `gguf:${repo.repo_id}`,
        id: repo.repo_id,
        label: modelDisplayName(repo.repo_id),
        detail: repo.size_bytes > 0 ? formatBytes(repo.size_bytes) : undefined,
        tab: DEVICE_TAB,
        kind: "gguf",
        section: repoSection(repo.repo_id),
        format: "gguf",
        aliases: [repo.load_id ?? "", repo.cache_path],
        gguf: repo,
        sizeBytes: repo.size_bytes,
        updatedAt: repo.last_modified,
        partial: repo.partial === true,
        resumable: repo.partial_resumable === true,
      });
    }
    for (const repo of inventory.cachedModels) {
      const audio = !repo.companion && isAudio(repo.repo_id, repo.task);
      if (
        !audio &&
        (repo.companion ||
          !isChatTask(repo.task) ||
          isMediaCatalogRepo(repo.repo_id))
      ) {
        continue;
      }
      device.push({
        media: audio ? "audio" : undefined,
        key: `cached:${repo.repo_id}`,
        id: repo.repo_id,
        label: modelDisplayName(repo.repo_id),
        detail: repo.size_bytes > 0 ? formatBytes(repo.size_bytes) : undefined,
        tab: DEVICE_TAB,
        kind: "cached",
        section: repoSection(repo.repo_id),
        // The card's mlx tag or library counts too: many MLX uploads lack it in the name.
        format: isMlxId(
          repo.repo_id,
          hasMlxCardTag({ tags: repo.tags, libraryName: repo.library_name }),
        )
          ? "mlx"
          : "checkpoint",
        aliases: [repo.load_id ?? "", repo.cache_path ?? ""],
        cached: repo,
        sizeBytes: repo.size_bytes,
        updatedAt: repo.last_modified,
        partial: repo.partial === true,
        resumable: repo.partial_resumable === true,
      });
    }
    for (const lora of loraModels ?? []) {
      // Trained native-audio checkpoints load only as merged exports; on-disk full ones always load.
      if (
        (lora.audioType && !listsAudio) ||
        (lora.source !== "local" &&
          !nativeAudioCheckpointIsLoadable(lora.audioType, lora.exportType))
      ) {
        continue;
      }
      device.push({
        media:
          mediaPageForTask(loraMeta(lora).pipelineTag) === "audio"
            ? "audio"
            : undefined,
        key: `local:${lora.id}`,
        id: lora.id,
        label: loraOptionLabel(lora),
        detail: loraDetail(lora),
        tab: DEVICE_TAB,
        kind: loraNeedsQuant(lora) ? "localGguf" : "local",
        section: localSection(lora.source, lora.baseModel),
        format: loraFormat(lora),
        sizeBytes: lora.sizeBytes ?? undefined,
        updatedAt: lora.updatedAt,
        lora,
      });
    }
    for (const model of npuModels ?? []) {
      if (!model.downloaded) {
        continue;
      }
      device.push({
        key: `npu:${model.id}`,
        id: model.model_path,
        label: model.id,
        tab: DEVICE_TAB,
        kind: "npu",
        section: "npu",
        npu: model,
        sizeBytes: model.size_gb != null ? model.size_gb * 1e9 : undefined,
      });
    }
    // Local models load by path and two files can share a name, so the checkpoint is the identity.
    const servedAs = (m: (typeof loadedModels)[number]) => m.checkpoint ?? m.id;
    const loaded = new Map<string, (typeof loadedModels)[number]>();
    for (const m of loadedModels) {
      loaded.set(m.id.toLowerCase(), m);
    }
    for (const m of loadedModels) {
      loaded.set(servedAs(m).toLowerCase(), m);
    }
    const marked = dedupeRows(device).map((row) => {
      const hit =
        loaded.get(row.id.toLowerCase()) ??
        (row.aliases ?? [])
          .map((a) => loaded.get(a.toLowerCase()))
          .find(Boolean);
      return hit
        ? {
            ...row,
            loaded: true,
            loadedQuant: hit.quant ?? null,
            loadedId: servedAs(hit),
          }
        : row;
    });
    for (const m of loadedModels) {
      const checkpoint = servedAs(m);
      if (marked.some((row) => row.loaded && row.loadedId === checkpoint)) {
        continue;
      }
      marked.push({
        key: `loaded:${checkpoint}`,
        id: checkpoint,
        label: modelDisplayName(m.id),
        tab: DEVICE_TAB,
        kind: "loaded",
        section: repoSection(m.id),
        format: m.quant ? "gguf" : "checkpoint",
        aliases: checkpoint === m.id ? undefined : [m.id],
        loaded: true,
        loadedQuant: m.quant ?? null,
        loadedId: checkpoint,
      });
    }
    marked.sort((a, b) => Number(!!b.loaded) - Number(!!a.loaded));
    for (const row of marked) {
      const maker = rowMaker(row);
      if (maker) {
        row.groups = [`${MAKER_PREFIX}${maker.id}`];
        row.makerName = maker.name;
      }
    }
    const loadedIds = new Set(
      marked.filter((row) => row.loaded).map((row) => row.id.toLowerCase()),
    );
    // A partial copy still offers its download in Discover.
    const onDisk = new Set(
      marked.filter((row) => !row.partial).map((row) => row.id.toLowerCase()),
    );
    const hub: Row[] = (recommended ?? [])
      .filter((m) => !isMediaCatalogRepo(m.id))
      .map((m) => ({
        key: `hub:${m.id}`,
        id: m.id,
        label: modelDisplayName(m.id),
        tab: RECOMMENDED_TAB,
        kind: "hub",
        section: "recommended",
        format: m.isGguf
          ? "gguf"
          : isMlxId(m.id, mlxTaggedHub.has(m.id.toLowerCase()))
            ? "mlx"
            : "checkpoint",
        onDevice: onDisk.has(m.id.toLowerCase()),
        loaded: loadedIds.has(m.id.toLowerCase()),
        hub: m,
      }));
    const external: Row[] = (externalModels ?? []).map((m) => ({
      key: m.id,
      id: m.id,
      label: m.name,
      tab: `${EXT_PREFIX}${m.providerId}`,
      groups: [CONNECTED_TAB],
      kind: "external",
      providerType: m.providerType,
      providerName: m.providerName,
    }));
    const npuOffers: Row[] = (npuModels ?? [])
      .filter((model) => !model.downloaded)
      .map((model) => ({
        key: `npu-offer:${model.id}`,
        id: model.model_path,
        label: model.id,
        tab: RECOMMENDED_TAB,
        kind: "npu",
        section: "npu",
        npu: model,
      }));
    return [...marked, ...npuOffers, ...hub, ...external];
  }, [
    inventory.cachedGguf,
    inventory.cachedModels,
    npuModels,
    loraModels,
    loadedModels,
    recommended,
    mlxTaggedHub,
    externalModels,
    mediaTaskById,
    listsAudio,
  ]);
  routeMediaPick.current = (id, meta) => {
    if (!listsAudio) {
      return false;
    }
    const pickedTask = taskForMediaPick(
      meta.pipelineTag,
      mediaTaskById.get(id.toLowerCase()),
    );
    const page = mediaPageForTask(pickedTask);
    if (!page) {
      return false;
    }
    const isGguf = Boolean(meta.isGguf || meta.ggufFilename);
    const gguf = inventory.cachedGguf.find((c) => c.repo_id === id);
    const cached = inventory.cachedModels.find((c) => c.repo_id === id);
    if (
      page === "audio" &&
      !audioPickIsRoutable({
        id,
        task: pickedTask,
        isGguf,
        isCurated: artifactForRepoId(id, AUDIO_CATALOG) !== null,
        audioType: meta.audioType ?? gguf?.audio_type ?? cached?.audio_type,
        isLocalCheckpoint:
          meta.source === "lora" ||
          meta.source === "exported" ||
          meta.source === "local",
        // Only the audio runtime's header classifier tags a GGUF text-to-audio.
        taskFromGgufArch: gguf?.task === "text-to-audio" ? true : undefined,
        tags: cached?.tags,
        libraryName: cached?.library_name,
      }) &&
      !macMlxSttRowIsRunnable({
        isMac,
        isStt: pickedTask === "automatic-speech-recognition",
        isGguf,
        libraryName: cached?.library_name,
      })
    ) {
      toast.error(
        `${id} is not an audio model Unsloth can run yet. The Audio page lists the families it supports.`,
        { duration: 7000 },
      );
      setPending(null);
      return true;
    }
    setPending(null);
    setOpen(false);
    void navigate({
      to: `/${page}`,
      search:
        page === "audio"
          ? audioPickSearch(id, { ...meta, task: pickedTask })
          : diffusionRouteSearch(id, meta),
    });
    return true;
  };

  // biome-ignore lint/correctness/useExhaustiveDependencies: a new inventory is the trigger, not an input.
  const quantsEpoch = useMemo(
    () => ({ open }),
    [open, inventory.cachedGguf, loraModels],
  );
  const quantsReadAt = useRef(new Map<string, object>());
  const quantsEpochRef = useRef(quantsEpoch);
  quantsEpochRef.current = quantsEpoch;
  useEffect(() => {
    if (!open) {
      return;
    }
    const queue = rows.filter(
      (row) =>
        (row.kind === "gguf" || row.kind === "localGguf") &&
        !row.partial &&
        quantsReadAt.current.get(row.key) !== quantsEpoch,
    );
    if (queue.length === 0) {
      return;
    }
    let live = true;
    const epoch = quantsEpoch;
    const pending: (readonly [
      string,
      GgufVariantDetail[] | null,
      GgufVariantDetail[] | null,
    ])[] = [];
    let timer: ReturnType<typeof setTimeout> | null = null;
    const read = async (row: Row) => {
      const target = quantTarget(row);
      if (!target) {
        return [row.key, [] as GgufVariantDetail[], NO_QUANTS] as const;
      }
      try {
        const { quants: list, partials } = await rowQuants(target);
        return [row.key, list, partials] as const;
      } catch {
        return [row.key, null, null] as const;
      }
    };
    // Reads that finish after a rows change still land: only a new inventory makes them stale.
    const flush = () => {
      timer = null;
      const found = pending.splice(0);
      if (found.length === 0 || quantsEpochRef.current !== epoch) {
        return;
      }
      for (const [key] of found) {
        quantsReadAt.current.set(key, epoch);
      }
      setDeviceQuants((prev) => {
        const next = new Map(prev);
        for (const [key, list] of found) {
          if (list) {
            next.set(key, list);
          } else if (!next.has(key)) {
            next.set(key, []);
          }
        }
        return next;
      });
      setDevicePartials((prev) => {
        const next = new Map(prev);
        for (const [key, , partials] of found) {
          if (partials?.length) {
            next.set(key, partials);
          } else if (partials) {
            next.delete(key);
          }
        }
        return next;
      });
    };
    // Bounded, as the header picker's reader is: hundreds of repos at once queue up the backend.
    void readBounded(
      queue,
      QUANT_READ_WORKERS,
      read,
      (found) => {
        pending.push(found);
        timer ??= setTimeout(flush, QUANT_FLUSH_MS);
      },
      () => live,
    ).then(() => {
      if (timer) {
        clearTimeout(timer);
      }
      flush();
    });
    return () => {
      live = false;
    };
  }, [open, rows, quantsEpoch]);

  const isFavorite = useCallback(
    (row: Row) =>
      row.kind === "external"
        ? pinnedConnected.includes(row.id)
        : pinCoversId(pinned, row.id),
    [pinned, pinnedConnected],
  );

  const connectionsOn = source?.connections !== false;
  const isOwner = useIsAccountOwner();
  const makerTabs = useMemo<SourceTab[]>(() => {
    const byMaker = new Map<string, { maker: Maker; ids: Set<string> }>();
    for (const row of rows) {
      const maker =
        row.groups && row.kind !== "external" ? rowMaker(row) : null;
      if (!maker) {
        continue;
      }
      const entry = byMaker.get(maker.id) ?? { maker, ids: new Set() };
      entry.ids.add(row.id.toLowerCase());
      byMaker.set(maker.id, entry);
    }
    const out: SourceTab[] = [];
    for (const { maker, ids } of byMaker.values()) {
      if (ids.size < MAKER_TAB_MIN_MODELS) {
        continue;
      }
      out.push({
        value: `${MAKER_PREFIX}${maker.id}`,
        label: maker.name,
        brand: true,
        movable: true,
        logo: maker.logoPath,
        // Solid marks read larger than the outline glyphs at one size, so tabs draw them at 15px.
        icon: (
          <MakerMark
            provider={maker}
            className="size-[calc(15px*var(--ui-space-scale,1))]"
          />
        ),
      });
    }
    return out;
  }, [rows]);

  const [tabOrder, setTabOrder] = useState(readTabOrder);
  const [draggingTab, setDraggingTab] = useState<string | null>(null);
  const [tabDrop, setTabDrop] = useState<TabDrop | null>(null);
  const tabs = useMemo<SourceTab[]>(() => {
    // Scaled (not resized, which would move the tab gaps) to a little under the logos' 15px of ink.
    const glyph = (icon: typeof StarIcon, scale: string) => (
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className={cn("size-4", scale)}
      />
    );
    const out: SourceTab[] = [
      {
        value: FAVORITES_TAB,
        label: "Favorites",
        icon: glyph(StarIcon, "scale-[1.1]"),
      },
      {
        value: RECOMMENDED_TAB,
        label: "Discover models",
        icon: glyph(Navigation05Icon, "scale-[1.1]"),
      },
      {
        value: DEVICE_TAB,
        label: "On Device",
        icon: glyph(Download01Icon, "scale-[1.21]"),
      },
      ...(connectionsOn || (externalModels?.length ?? 0) > 0
        ? [
            {
              value: CONNECTED_TAB,
              label: "Connected models",
              icon: glyph(CloudIcon, "scale-[1.2]"),
            },
          ]
        : []),
      ...makerTabs,
    ];
    const seen = new Set<string>();
    for (const m of externalModels ?? []) {
      if (seen.has(m.providerId)) {
        continue;
      }
      seen.add(m.providerId);
      out.push({
        value: `${EXT_PREFIX}${m.providerId}`,
        label: m.providerName,
        brand: true,
        movable: true,
        logo: providerLogoPath(m.providerType),
        icon: providerLogo(m.providerType, m.providerName),
      });
    }
    const logos = new Map<string, number>();
    for (const t of out) {
      if (t.logo) {
        logos.set(t.logo, (logos.get(t.logo) ?? 0) + 1);
      }
    }
    for (const t of out) {
      if (
        t.value.startsWith(EXT_PREFIX) &&
        t.logo &&
        (logos.get(t.logo) ?? 0) > 1
      ) {
        t.cloudBadge = true;
        // The maker's own mark, so the pair match in size and differ only by the cloud.
        const twin = out.find(
          (m) => m.value.startsWith(MAKER_PREFIX) && m.logo === t.logo,
        );
        if (twin) {
          t.icon = twin.icon;
        }
      }
    }
    const rank = new Map(tabOrder.map((value, index) => [value, index]));
    const pinned = out.filter((t) => !t.movable);
    const movable = out
      .filter((t) => t.movable)
      .map((t, index) => ({ t, index }))
      .sort(
        (a, b) =>
          (rank.get(a.t.value) ?? Number.MAX_SAFE_INTEGER) -
            (rank.get(b.t.value) ?? Number.MAX_SAFE_INTEGER) ||
          a.index - b.index,
      )
      .map(({ t }) => t);
    return [...pinned, ...movable];
  }, [externalModels, makerTabs, connectionsOn, tabOrder]);
  const moveTab = (from: string, to: TabDrop) => {
    const order = tabs.filter((t) => t.movable).map((t) => t.value);
    const at = order.indexOf(from);
    if (at < 0 || !order.includes(to.value)) {
      return;
    }
    order.splice(at, 1);
    const target = order.indexOf(to.value) + (to.edge === "after" ? 1 : 0);
    order.splice(target, 0, from);
    setTabOrder(order);
    try {
      localStorage.setItem(TAB_ORDER_KEY, JSON.stringify(order));
    } catch {
      // Unavailable storage: the order holds for this session.
    }
  };

  const chosen = findChosenRow(rows, checkpoint);
  const chosenId = chosen?.id;
  // A pick names the chip at once; the checkpoint follows only once its load lands.
  const [pendingPick, setPending] = useState<Row | null>(null);
  const pending = pane ? null : pendingPick;
  useEffect(() => {
    if (!pending) {
      return;
    }
    if (chosen?.id === pending.id) {
      setPending(null);
      return;
    }
    if (modelLoading) {
      return;
    }
    const timer = setTimeout(() => setPending(null), 1500);
    return () => clearTimeout(timer);
  }, [pending, chosen?.id, modelLoading]);
  const shown = pending ?? chosen;
  const chosenName =
    shown?.label ??
    (checkpoint
      ? (missingExternalModel(
          checkpoint,
          externalModels ?? [],
          source?.externalConnections ?? [],
        )?.modelName ?? modelDisplayName(checkpoint))
      : null);
  const chosenMark: Parameters<typeof RowMark>[0]["row"] =
    shown ??
    (isExternalModelId(checkpoint)
      ? { kind: "external" }
      : checkpoint
        ? { kind: "loaded", id: checkpoint }
        : undefined);
  const hasModels =
    !inventory.cachedReady ||
    (externalModels?.length ?? 0) > 0 ||
    rows.some((row) => row.tab === DEVICE_TAB);
  const viewed =
    tab ??
    (hasModels
      ? ((lastTab && tabs.some((t) => t.value === lastTab) ? lastTab : null) ??
        (chosen?.tab === RECOMMENDED_TAB ? DEVICE_TAB : chosen?.tab) ??
        DEVICE_TAB)
      : RECOMMENDED_TAB);
  useEffect(() => {
    setOnRecommended(viewed === RECOMMENDED_TAB);
  }, [viewed]);
  const favoriteKeyOf = useCallback(
    (row: Row): string | null => {
      if (row.kind === "external") {
        return pinnedConnected.includes(row.id) ? row.id : null;
      }
      const base = row.id.toLowerCase();
      return (
        pinned.find((key) => {
          const k = key.toLowerCase();
          return k === base || k.startsWith(`${base}::`);
        }) ?? null
      );
    },
    [pinned, pinnedConnected],
  );
  const [deviceSort, setDeviceSort] = useState<DeviceSort>("recent");
  const [deviceSortDesc, setDeviceSortDesc] = useState(DEVICE_SORT_DESC.recent);
  const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");
  // Stamped here too: the header picker that used to record chat loads is gone from Chat.
  const loadTimes = useModelLoadTimes(loadStampId(checkpoint, source?.loaded));
  const foundRows = useMemo(() => {
    // Discover rows are the Hub's answer to this very query ("google" finds unsloth/gemma-*).
    const hubAnswered =
      viewed === RECOMMENDED_TAB &&
      hubQuery !== "" &&
      hubQuery === query.trim() &&
      hubResults.length > 0;
    const found = filterRows(
      rows,
      viewed,
      query,
      isFavorite,
      hubAnswered ? (row) => row.kind === "hub" : undefined,
    );
    if (viewed === RECOMMENDED_TAB) {
      return found.filter((row) => {
        if (!matchesFormat(row, formatFilter)) {
          return false;
        }
        if (
          !fitOnDeviceOnly ||
          row.kind !== "hub" ||
          row.onDevice ||
          row.id === chosenId
        ) {
          return true;
        }
        const checkpointFits = hubCheckpointFits.get(row.id.toLowerCase());
        if (checkpointFits !== undefined) {
          return checkpointFits;
        }
        const fit = hubFit.get(row.id.toLowerCase());
        return !isOverBudget(fit) || (fit === "ram" && noGpu);
      });
    }
    if (viewed === CONNECTED_TAB || viewed.startsWith(EXT_PREFIX)) {
      return sortConnectedRows(found, isFavorite);
    }
    const lastLoaded = (row: Row) => loadedAt(loadTimes, row.id);
    const favoritesFirst = (list: Row[]) =>
      [...list].sort((a, b) => Number(isFavorite(b)) - Number(isFavorite(a)));
    // A maker's tab reads like On Device's default: favorites, then last loaded or downloaded.
    if (viewed.startsWith(MAKER_PREFIX)) {
      return favoritesFirst(sortDeviceRows(found, "recent", lastLoaded));
    }
    if (viewed !== DEVICE_TAB) {
      return found;
    }
    const sorted = sortDeviceRows(
      found.filter((row) => matchesFormat(row, formatFilter)),
      deviceSort,
      lastLoaded,
      deviceSortDesc,
    );
    return deviceSort === "recent" && deviceSortDesc === DEVICE_SORT_DESC.recent
      ? favoritesFirst(sorted)
      : sorted;
  }, [
    rows,
    viewed,
    query,
    hubQuery,
    hubResults.length,
    isFavorite,
    formatFilter,
    deviceSort,
    deviceSortDesc,
    loadTimes,
    fitOnDeviceOnly,
    hubFit,
    hubCheckpointFits,
    noGpu,
    chosenId,
  ]);
  const [collapsedSections, setCollapsedSections] = useState<
    ReadonlySet<string>
  >(() => new Set());
  const sectionKey = (section: string) => `${viewed}:${section}`;
  const toggleSection = (section: string) => {
    const key = sectionKey(section);
    setCollapsedSections((prev) => {
      const next = new Set(prev);
      if (!next.delete(key)) {
        next.add(key);
      }
      return next;
    });
  };
  const jumpTo = (section: ListSection) => {
    setCollapsedSections((prev) => {
      const next = new Set(prev);
      next.delete(sectionKey(section));
      return next;
    });
    // Two frames, so an expanded section has rendered before it scrolls into view.
    requestAnimationFrame(() =>
      requestAnimationFrame(() =>
        listRef.current
          ?.querySelector(`[data-section="${section}"]`)
          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
      ),
    );
  };
  const listSections = useMemo<ListSection[]>(() => {
    if (viewed === DEVICE_TAB) {
      const present = new Set(foundRows.map((row) => row.section));
      present.add("fineTuned");
      present.add("custom");
      return orderBySection(
        [...present]
          .filter((section): section is ListSection => Boolean(section))
          .map((section) => ({ section })),
      ).map(({ section }) => section);
    }
    if (viewed === RECOMMENDED_TAB && npuCatalog) {
      const npuShown =
        foundRows.some((row) => row.section === "npu") ||
        (!npuCatalog.ready && !query);
      return npuShown ? ["npu", "recommended"] : [];
    }
    return [];
  }, [foundRows, viewed, query, npuCatalog]);
  const shownRows = useMemo(() => {
    const found = foundRows;
    if (listSections.length > 0) {
      return orderBySection(found).filter(
        (row) =>
          !(row.section && collapsedSections.has(`${viewed}:${row.section}`)),
      );
    }
    if (viewed === CONNECTED_TAB) {
      return found.filter(
        (row) => !collapsedSections.has(`${viewed}:${row.tab}`),
      );
    }
    if (viewed !== FAVORITES_TAB) {
      return found;
    }
    const rank = (row: Row) => {
      const key = favoriteKeyOf(row);
      return row.kind === "external"
        ? pinned.length + pinnedConnected.indexOf(key ?? "")
        : pinned.indexOf(key ?? "");
    };
    return [...found].sort((a, b) => rank(a) - rank(b));
  }, [
    viewed,
    favoriteKeyOf,
    pinned,
    pinnedConnected,
    listSections,
    collapsedSections,
    foundRows,
  ]);
  const customFolders = useCustomFolders({
    enabled: open && viewed === DEVICE_TAB,
    onFoldersChange: source?.onFoldersChange,
  });
  const favoriteDragOrder = (connected: boolean) =>
    shownRows
      .filter((row) => (row.kind === "external") === connected)
      .map(favoriteKeyOf)
      .filter((key): key is string => key !== null);
  const localFavoriteDrag = usePinnedRowDrag({
    scope: "composer-favorites",
    order: () => favoriteDragOrder(false),
    onDrop: (fromKey, drop) => {
      const store = usePinnedModelsStore.getState();
      const anchor = pinDropAnchor(store.pinned, fromKey, drop.key, drop.edge);
      if (!anchor) {
        return;
      }
      store.beginPinnedDrag();
      store.movePinned(fromKey, anchor);
      store.endPinnedDrag(true);
    },
  });
  const connectedFavoriteDrag = usePinnedRowDrag({
    scope: "composer-favorites-connected",
    order: () => favoriteDragOrder(true),
    onDrop: (fromId, drop) => {
      const store = usePinnedConnectedModelsStore.getState();
      const anchor = pinDropAnchor(store.pinned, fromId, drop.key, drop.edge);
      if (!anchor) {
        return;
      }
      store.beginPinnedConnectedDrag();
      store.movePinnedConnected(fromId, anchor);
      store.endPinnedConnectedDrag(true);
    },
  });
  const isExpanded = expanded || !chosenName;
  useAnimatedListHeight(listEl);

  const chosenTarget =
    // The Quantization row switches GGUF quants; an MLX precision is picked from its row.
    chosen && chosen.kind !== "hub" && chosen.kind !== "cached"
      ? quantTarget(chosen)
      : null;
  const chosenTargetRef = useRef(chosenTarget);
  chosenTargetRef.current = chosenTarget;
  const chosenQuantKey = chosenTarget ? chosen?.key : undefined;
  useEffect(() => {
    const target = chosenTargetRef.current;
    if (!(open && target && chosenQuantKey)) {
      return;
    }
    let live = true;
    rowQuants(target)
      .then(({ quants: list }) => {
        if (live) {
          setQuants({ key: chosenQuantKey, list });
        }
      })
      .catch(() => {
        if (live) {
          setQuants(null);
        }
      });
    return () => {
      live = false;
    };
  }, [open, chosenQuantKey]);

  const resetView = useCallback(() => {
    offerAbort.current?.abort();
    offerAbort.current = null;
    setTabState(null);
    setOffer(null);
    setQuery("");
    setQuantsOpen(false);
  }, []);

  // Before the inventory lands there is no row: trust residency, since an image load can release the model.
  const chosenLoaded = pane
    ? pane.loaded === true && Boolean(checkpoint)
    : Boolean(
        chosen?.loaded ||
          (!chosen &&
            checkpoint &&
            !isExternalModelId(checkpoint) &&
            source?.loaded !== false),
      );
  // A Compare pane ejects nothing, and the API page only once its load has settled.
  const canEject = !pane || pane.loaded === true;
  const eject = (row: Row) => {
    source?.eject(ejectTargetId(row, chosen?.id));
  };

  const handOff = (row: Row) => {
    openModelCard(row);
  };

  // A Discover repo on disk loads its cached copy; without isDownloaded it stages a download.
  const wholeHubMeta = (row: Row): ModelSelectorChangeMeta => {
    const id = row.id.toLowerCase();
    const device = row.onDevice
      ? rows.find(
          (r) => r.kind === "cached" && !r.partial && r.id.toLowerCase() === id,
        )
      : undefined;
    return device
      ? cachedMeta(device)
      : {
          source: "hub",
          isLora: false,
          isGguf: false,
          isDownloaded: Boolean(row.onDevice),
        };
  };

  const pickSeq = useRef(0);
  const supersede = () => {
    pickSeq.current += 1;
    return pickSeq.current;
  };

  const pickQuantRow = async (row: Row, target: QuantTarget, seq: number) => {
    if (!source) {
      return;
    }
    const {
      quants: list,
      defaultVariant,
      mlx,
      hasVision,
    } = await rowQuants(target);
    if (pickSeq.current !== seq) {
      return;
    }
    const fetched = list.filter((v) => v.downloaded === true);
    const quant = chooseQuant(
      target.remote && fetched.length > 0 ? fetched : list,
      [row.loadedQuant, defaultVariant],
    );
    // A plain MLX snapshot has no precision folders: it loads as one checkpoint.
    if (row.kind === "cached" && !(mlx && quant)) {
      if (row.id !== chosen?.id) {
        setPending(row);
      }
      source.onPick(row.id, cachedMeta(row));
      return;
    }
    if (!quant) {
      handOff(row);
      return;
    }
    if (row.id !== chosen?.id) {
      setPending(row);
    }
    source.onPick(
      row.id,
      ggufMeta(
        row,
        quant,
        target.local || quant.downloaded === true,
        !mlx,
        hasVision ?? row.gguf?.has_vision,
      ),
    );
  };

  const toggleOffer = async (
    row: Row,
    target: QuantTarget,
    seq: number,
    retry = false,
  ) => {
    offerAbort.current?.abort();
    offerAbort.current = null;
    if (offer?.key === row.key && !retry) {
      setOffer(null);
      return;
    }
    const abort = new AbortController();
    offerAbort.current = abort;
    setOffer({
      key: row.key,
      quants: null,
      defaultVariant: null,
      failed: false,
      target,
    });
    const mlxRepo = row.kind === "hub" && !row.hub?.isGguf;
    const pickWhole = () => {
      if (pickSeq.current !== seq || !source) {
        return;
      }
      setOffer(null);
      if (row.id !== chosen?.id) {
        setPending(row);
      }
      source.onPick(row.id, wholeHubMeta(row));
      resetView();
    };
    try {
      const {
        quants: list,
        defaultVariant,
        mlx,
        hasVision,
      } = await rowQuants(target, abort.signal);
      if (abort.signal.aborted) {
        return;
      }
      if (!mlx && list.length > 0) {
        noteListedFit(row.id, list);
      }
      if (mlxRepo && !(mlx && list.length > 0)) {
        pickWhole();
        return;
      }
      if (list.length === 1) {
        if (pickSeq.current === seq) {
          setOffer(null);
          pickOfferQuant(row, list[0], !mlx, hasVision);
        }
        return;
      }
      setOffer((now) =>
        now?.key === row.key
          ? {
              key: row.key,
              quants: list,
              defaultVariant,
              failed: false,
              mlx,
              hasVision,
              target,
            }
          : now,
      );
    } catch (err) {
      if (abort.signal.aborted) {
        return;
      }
      if (mlxRepo) {
        pickWhole();
        return;
      }
      setOffer((now) =>
        now?.key === row.key
          ? {
              ...now,
              quants: [],
              failed: true,
              error: describeVariantListingError(err),
            }
          : now,
      );
    } finally {
      if (offerAbort.current === abort) {
        offerAbort.current = null;
      }
    }
  };
  useEffect(() => () => offerAbort.current?.abort(), []);
  // A listing still in flight at close or tab change must not pick (and download) its lone quant.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `open` and `viewed` are the triggers.
  useEffect(() => {
    offerAbort.current?.abort();
    offerAbort.current = null;
    setOffer((now) => (now && now.quants === null ? null : now));
  }, [open, viewed]);

  const pickOfferQuant = (
    row: Row,
    variant: GgufVariantDetail,
    isGguf = true,
    hasVision?: boolean,
  ) => {
    if (!source) {
      return;
    }
    if (row.id !== chosen?.id) {
      setPending(row);
    }
    supersede();
    source.onPick(
      row.id,
      ggufMeta(
        row,
        variant,
        variant.downloaded === true,
        isGguf,
        hasVision ?? row.gguf?.has_vision,
      ),
    );
    resetView();
  };

  const pickNpu = (row: Row, model: NpuModel, seq: number) => {
    if (!(source && npuCatalog) || model.id in npuCatalog.downloads) {
      return;
    }
    const meta = {
      source: "local",
      isLora: false,
      isDownloaded: true,
    } as const;
    if (model.downloaded) {
      if (row.id !== chosen?.id) {
        setPending(row);
      }
      source.onPick(model.model_path, meta);
      resetView();
      return;
    }
    const before = useChatRuntimeStore.getState().params.checkpoint;
    void npuCatalog.download(model).then((done) => {
      const now = useChatRuntimeStore.getState();
      // A model picked while this downloaded is the user's newer choice; keep it.
      if (
        done &&
        pickSeq.current === seq &&
        now.params.checkpoint === before &&
        !now.loadingModelPick
      ) {
        source.onPick(model.model_path, meta);
      }
    });
  };

  // On Device only: Discover's own offer already lists a partial quant.
  const partialsOf = (row: Row) =>
    row.tab === DEVICE_TAB && !row.partial && viewed !== RECOMMENDED_TAB
      ? (devicePartials.get(row.key) ?? NO_QUANTS)
      : NO_QUANTS;
  const variantCount = (row: Row) =>
    (deviceQuants.get(row.key)?.length ?? 0) + partialsOf(row).length;

  const pick = async (row: Row) => {
    if (!source) {
      return;
    }
    const seq = supersede();
    if (row.npu) {
      pickNpu(row, row.npu, seq);
      return;
    }
    const target = quantTarget(row);
    const offerAll =
      target !== null &&
      row.kind === "gguf" &&
      useChatRuntimeStore.getState().showAllQuantizations;
    // Discover always offers its quants, even with one on disk: that one is ticked, not loaded unasked.
    if (
      target &&
      // A cached MLX snapshot resolves in pickQuantRow, which loads it whole when it has no precisions.
      row.kind !== "cached" &&
      (offerAll || (target.remote && (row.kind === "hub" || !row.onDevice)))
    ) {
      await toggleOffer(
        row,
        // Every quant: a cached partial listed alone would be auto-picked, resuming its download.
        offerAll || row.onDevice || row.kind === "hub"
          ? { ...target, remote: true, everyQuant: true }
          : target,
        seq,
      );
      return;
    }
    if (variantCount(row) > 1) {
      setFoldedQuants((prev) => {
        const next = new Set(prev);
        if (!next.delete(row.key)) {
          next.add(row.key);
        }
        return next;
      });
      return;
    }
    if (target) {
      setResolving(row.key);
      try {
        await pickQuantRow(row, target, seq);
      } catch {
        if (pickSeq.current !== seq) {
          return;
        }
        if (row.kind === "cached") {
          source.onPick(row.id, cachedMeta(row));
          resetView();
        } else {
          handOff(row);
        }
        return;
      } finally {
        setResolving((now) => (now === row.key ? null : now));
      }
      if (pickSeq.current === seq) {
        resetView();
      }
      return;
    }
    if (row.id !== chosen?.id) {
      setPending(row);
    }
    if (row.kind === "external") {
      source.onPick(row.id, { source: "external", isLora: false });
    } else if (row.kind === "local" && row.lora) {
      source.onPick(row.id, loraMeta(row.lora));
    } else if (row.kind === "cached") {
      source.onPick(row.id, cachedMeta(row));
    } else if (row.kind === "loaded") {
      source.onPick(row.id, residentMeta(row));
    } else if (row.kind === "hub") {
      source.onPick(row.id, wholeHubMeta(row));
    }
    resetView();
  };

  const toggleFavorite = async (row: Row) => {
    if (row.kind === "external") {
      togglePinnedConnected(row.id);
      return;
    }
    if (isFavorite(row)) {
      unpinRepo(row.id);
      return;
    }
    const target = row.kind === "gguf" ? quantTarget(row) : null;
    if (!target || variantCount(row) > 1) {
      togglePinned(row.id);
      return;
    }
    const quant =
      row.loadedQuant ??
      (row.id === chosen?.id ? activeVariant : null) ??
      (await rowQuants(target)
        .then(
          ({ quants: list, defaultVariant }) =>
            chooseQuant(list, [defaultVariant])?.quant,
        )
        .catch(() => undefined));
    togglePinned(row.id, quant ?? undefined);
  };

  const pickQuant = (variant: GgufVariantDetail) => {
    setQuantsOpen(false);
    if (source && chosen) {
      supersede();
      source.onPick(chosen.id, ggufMeta(chosen, variant, true));
    }
  };

  const configResident =
    configTarget !== null &&
    modelConfigTargetIsResident({
      target: configTarget,
      selectedId: checkpoint,
      activeGgufVariant: activeVariant,
      loaded: source?.loaded ?? false,
    });

  const triggerRef = useRef<HTMLButtonElement>(null);
  const [openSide, setOpenSide] = useState(side);
  const [alignOffset, setAlignOffset] = useState(0);
  const [panel, setPanel] = useState<HTMLDivElement | null>(null);
  const [dropOffset, setDropOffset] = useState(PICKER_GAP);
  const dropsDown = open && openSide === "bottom";
  useEffect(() => {
    if (!(dropsDown && panel)) {
      setDropOffset(PICKER_GAP);
      return;
    }
    const place = () => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (rect) {
        setDropOffset(
          pickerDropOffset(rect.bottom, panel.offsetHeight, window.innerHeight),
        );
      }
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(panel);
    window.addEventListener("resize", place);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", place);
    };
  }, [dropsDown, panel]);
  const onOpenChange = (next: boolean) => {
    setOpen(next);
  };
  // During render, so an opening paints fresh however it was opened.
  const [seenOpen, setSeenOpen] = useState(open);
  if (open !== seenOpen) {
    setSeenOpen(open);
    setConfigTarget(null);
    if (open) {
      // Here, not in onOpenChange: the shortcut, tour and the Hub's Run open it from outside.
      const trigger = triggerRef.current;
      const rect = trigger?.getBoundingClientRect() ?? null;
      setOpenSide(pickerSide(side, rect, window.innerHeight));
      const scale = trigger
        ? Number.parseFloat(
            getComputedStyle(trigger).getPropertyValue("--ui-space-scale"),
          ) || 1
        : 1;
      setAlignOffset(
        pickerAlignOffset(align, rect, PICKER_WIDTH * scale, window.innerWidth),
      );
      resetView();
      setActive(0);
      highlightRef.current = false;
      setHighlightState(false);
    }
  }
  // After the reset above, so a request that opened the picker lands on its settings.
  const configRequest = open ? (source?.configRequest ?? null) : null;
  const [adoptedRequestId, setAdoptedRequestId] = useState<string | null>(null);
  if (configRequest && adoptedRequestId !== configRequest.requestId) {
    setAdoptedRequestId(configRequest.requestId);
    setConfigTarget(
      modelConfigTarget(
        configRequest.id,
        configRequest.meta,
        configRequest.displayName,
      ),
    );
    setConfigFocus(configRequest.focus ?? null);
    setConfigStandalone(configRequest.standalone === true);
  }
  const onConfigRequestAdopted = source?.onConfigRequestAdopted;
  useEffect(() => {
    if (configRequest && adoptedRequestId === configRequest.requestId) {
      onConfigRequestAdopted?.(configRequest.requestId);
    }
  }, [configRequest, adoptedRequestId, onConfigRequestAdopted]);

  // A new search or tab is a new list, so the highlight starts over as on opening; before the
  // effect below, which then seats `active` inside it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `query` and `viewed` are the triggers.
  useEffect(() => {
    setHighlight(false);
  }, [query, viewed, setHighlight]);

  useEffect(() => {
    if (!isExpanded) {
      return;
    }
    // Rows paging in under the pointer must not move the highlight.
    if (highlightRef.current) {
      return;
    }
    // Discover is a browse list: it opens at its top, not scrolled to the chosen model.
    const at =
      viewed === RECOMMENDED_TAB
        ? -1
        : shownRows.findIndex((row) => row.id === chosen?.id);
    setActive(at >= 0 ? at : 0);
  }, [isExpanded, shownRows, chosen?.id, viewed]);

  useEffect(() => {
    if (!isExpanded) {
      return;
    }
    listRef.current
      ?.querySelector<HTMLElement>(`[data-row-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [active, isExpanded]);

  const navNeighbour = (from: Element | null | undefined, dir: 1 | -1) => {
    const items = Array.from(
      listRef.current?.querySelectorAll<HTMLElement>(
        "[data-row-index], [data-quant-option]",
      ) ?? [],
    );
    const at = from ? items.indexOf(from as HTMLElement) : -1;
    return at < 0 ? undefined : items[at + dir];
  };
  const navFrom = (from: HTMLElement, dir: 1 | -1) => {
    const next = navNeighbour(from, dir);
    if (!next) {
      return;
    }
    const rowIndex = next.dataset.rowIndex;
    if (rowIndex === undefined) {
      next.focus();
      return;
    }
    setActive(Number(rowIndex));
    setHighlight(true);
    (searchRef.current ?? next).focus();
  };

  // Through a ref: a key pressed before the search commits runs after the re-render.
  const navKeyRef = useRef<(key: NavKey) => void>(() => undefined);
  navKeyRef.current = (key) => {
    if (key === "ArrowDown" || key === "ArrowUp") {
      if (highlightRef.current) {
        const quant = navNeighbour(
          listRef.current?.querySelector(`[data-row-index="${active}"]`),
          key === "ArrowDown" ? 1 : -1,
        );
        if (quant && quant.dataset.rowIndex === undefined) {
          quant.focus();
          return;
        }
        setActive((i) =>
          key === "ArrowDown"
            ? Math.min(shownRows.length - 1, i + 1)
            : Math.max(0, i - 1),
        );
      }
      setHighlight(true);
    } else {
      const row = shownRows[active];
      if (row) {
        pick(row);
      }
    }
  };
  const onSearchNavKey = useCallback(
    (key: NavKey) => navKeyRef.current(key),
    [],
  );

  const expand = () => {
    setExpanded(true);
    requestAnimationFrame(() => searchRef.current?.focus());
  };

  const quantChoices =
    quants && chosen && quants.key === chosen.key
      ? pickerOrderedGgufVariants(quants.list, null, quantBudget).ordered
      : [];
  const showQuantRow = quantChoices.length > 1;
  const showEffort = !pane && reasoning.isEffort && reasoning.levels.length > 1;
  const showThinkingSwitch =
    !pane && reasoning.supportsReasoning && !showEffort;
  const showFastSwitch = !pane && reasoning.supportsFast && !showEffort;
  const showPreserve = !pane && reasoning.supportsPreserve;
  const reasoningChip = pane ? null : reasoning.chipLabel;
  const hasFoot =
    showQuantRow ||
    showThinkingSwitch ||
    showPreserve ||
    showEffort ||
    showFastSwitch;

  const connectFrom = (providerType?: string) => {
    setOpen(false);
    useSettingsDialogStore
      .getState()
      .openConnectionSettings(newConnectionRequest(providerType));
  };
  const emptyCopy = (): {
    title: string;
    hint?: string;
    actions?: {
      label: string;
      icon?: ReactNode;
      primary?: boolean;
      row: 0 | 1;
      run: () => void;
    }[];
  } => {
    if (
      viewed === CONNECTED_TAB &&
      !query &&
      (externalModels?.length ?? 0) === 0
    ) {
      return {
        title: "No connected models",
        hint: "Connect a provider to use its models here.",
        actions: [
          {
            label: "Add connection",
            primary: true,
            row: 0,
            run: () => connectFrom(),
          },
          // Agents settings are the owner's, so a managed account is not offered them.
          ...(isOwner
            ? [
                {
                  label: "Connect coding agent",
                  icon: (
                    <HugeiconsIcon
                      icon={BotIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  ),
                  row: 0 as const,
                  run: () => {
                    setOpen(false);
                    useSettingsDialogStore.getState().openDialog("agents");
                  },
                },
              ]
            : []),
          ...DEFAULT_PROVIDERS.map((provider) => ({
            label: provider.label,
            icon: providerLogo(provider.type),
            row: 1 as const,
            run: () => connectFrom(provider.connect),
          })),
        ],
      };
    }
    if (query) {
      return { title: "No models found", hint: `Nothing matches “${query}”.` };
    }
    if (viewed === FAVORITES_TAB) {
      return {
        title: "No favorites yet",
        hint: "Star a model to keep it here.",
      };
    }
    if (viewed === DEVICE_TAB) {
      return {
        title: "No models on this device",
        hint: "Download models from Model hub",
        actions: [
          {
            label: "Discover models",
            icon: (
              <HugeiconsIcon
                icon={DashboardCircleIcon}
                strokeWidth={1.75}
                className="size-4"
              />
            ),
            primary: true,
            row: 0,
            run: () => {
              setOpen(false);
              void navigate({ to: "/hub", search: { tab: "discover" } });
            },
          },
        ],
      };
    }
    return { title: "No models" };
  };

  const chipLabel = chosenName ?? pane?.placeholder ?? "Select model";
  const loadedIsMlx = useChatRuntimeStore((s) => s.loadedIsMlx);
  const formatDetail =
    activeVariant && chosenLoaded && !pending
      ? `${loadedIsMlx ? "MLX" : "GGUF"} · ${activeVariant}`
      : null;
  const loadedDetail =
    (!pane || pane.loaded) && loadedModels.length > 1
      ? `${loadedModels.length} models loaded`
      : null;
  const chipDetail =
    [formatDetail, loadedDetail].filter(Boolean).join(" · ") || null;
  const ejectsAll = loadedModels.length > 1;

  const settingsMeta = async (
    row: Row,
  ): Promise<ModelSelectorChangeMeta | null> => {
    const target = quantTarget(row);
    if (target) {
      const {
        quants: list,
        defaultVariant,
        mlx,
        hasVision,
      } = await rowQuants(target);
      const quant = chooseQuant(list, [
        row.loadedQuant,
        row.id === chosen?.id ? activeVariant : null,
        defaultVariant,
      ]);
      if (quant && (mlx || row.kind !== "cached")) {
        return ggufMeta(
          row,
          quant,
          target.local || quant.downloaded === true,
          !mlx,
          hasVision ?? row.gguf?.has_vision,
        );
      }
      if (row.kind !== "cached") {
        return null;
      }
    }
    if (row.kind === "local" && row.lora) {
      return loraMeta(row.lora);
    }
    if (row.npu?.downloaded) {
      return {
        source: "local",
        isLora: false,
        isDownloaded: true,
        contextLength: row.npu.max_context_length,
      };
    }
    if (row.kind === "cached") {
      return cachedMeta(row);
    }
    if (row.kind === "loaded") {
      // A standalone .gguf keys its settings by the bare path, as residentModelConfigTarget does.
      return isStandaloneGgufPath(row.id)
        ? { ...residentMeta(row), ggufVariant: undefined, isGguf: true }
        : residentMeta(row);
    }
    if (row.kind === "hub") {
      return {
        source: "hub",
        isLora: false,
        isDownloaded: Boolean(row.onDevice),
        isGguf: false,
      };
    }
    return null;
  };

  const openModelCard = (row: Row) => {
    setOpen(false);
    void navigate({
      to: "/hub",
      search: {
        tab: row.tab === DEVICE_TAB || row.onDevice ? "downloaded" : "discover",
        model: row.id,
      },
    });
  };

  const openSettings = async (row: Row) => {
    const meta = await settingsMeta(row).catch(() => null);
    if (!(source && meta)) {
      handOff(row);
      return;
    }
    openConfig(modelConfigTarget(row.id, meta, row.label));
  };

  const startUpdate = async (
    repoId: string,
    variant: GgufVariantDetail,
  ): Promise<void> => {
    const outcome = await downloadManager.requestStart({
      kind: "model",
      repoId,
      variant: variant.quant,
      expectedBytes: quantBytes(variant),
    });
    if (outcome === "conflict") {
      setUpdateConflictKey(jobKeyOf("model", repoId, variant.quant));
    } else if (outcome === "busy") {
      toast.info("A download for this model is already in progress", {
        description: "Try updating again once it finishes.",
      });
    } else if (outcome === "error") {
      throw new Error("Failed to start update");
    }
  };

  const updateFor = (row: Row, variant: GgufVariantDetail | undefined) =>
    variant?.update_available && row.gguf
      ? {
          title: "Update cached model?",
          description: (
            <>
              This will update{" "}
              <span className="font-medium text-foreground">
                {row.id} ({variant.quant})
              </span>
              .
            </>
          ),
          repoId: row.id,
          variant: variant.quant,
          disabled: source?.deleteDisabled,
          onConfirm: () => startUpdate(row.id, variant),
          onUpdated: () => {
            void inventory.refreshInventory();
          },
        }
      : undefined;

  const deleteQuantFor = (row: Row, variant: GgufVariantDetail) => {
    const inMemory =
      Boolean(row.loaded) &&
      variant.quant.toLowerCase() === row.loadedQuant?.toLowerCase();
    if (row.lora?.source === "exported" && row.lora.exportType === "gguf") {
      return {
        title: "Delete exported GGUF variant?",
        description: (
          <>
            This will remove{" "}
            <span className="font-medium text-foreground">
              {row.label} ({variant.quant})
            </span>{" "}
            from disk. This cannot be undone.
          </>
        ),
        successMessage: `Deleted ${row.label} ${variant.quant}`,
        disabled: inMemory || source?.deleteDisabled,
        onConfirm: async () => {
          await deleteFineTunedModel({
            modelPath: row.id,
            source: "exported",
            exportType: "gguf",
            ggufVariant: variant.quant,
          });
        },
        onDeleted: () =>
          source?.onModelsChange({ id: row.id, ggufVariant: variant.quant }),
      };
    }
    const repo = row.gguf;
    if (!repo) {
      return undefined;
    }
    return {
      title: "Delete cached model?",
      impact: {
        repoId: repo.repo_id,
        variant: variant.quant,
        cachePath: variant.cache_ref ?? variant.cache_path,
      },
      description: (
        <>
          This will remove{" "}
          <span className="font-medium text-foreground">
            {repo.repo_id} ({variant.quant})
          </span>{" "}
          from disk. You can re-download it later.
        </>
      ),
      successMessage: `Deleted ${repo.repo_id} ${variant.quant}`,
      disabled: inMemory || source?.deleteDisabled,
      onConfirm: async () => {
        await deleteCachedModel(
          repo.repo_id,
          variant.quant,
          useHfTokenStore.getState().token || undefined,
          variant.cache_ref ?? variant.cache_path ?? undefined,
        );
      },
      onDeleted: () => {
        void reconcileGgufPinsAfterDelete(
          repo.repo_id,
          useHfTokenStore.getState().token || undefined,
        ).finally(() => inventory.refreshInventory());
      },
    };
  };

  const openConnectedSettings = (
    row: Row,
    view: "settings" | "info" = "settings",
  ) => {
    // The dialog sits outside the panel, which would take its clicks as a dismiss.
    setOpen(false);
    setConnectedSettings({ row, view });
  };
  // Not navigator.clipboard: undefined in the desktop shell and over plain HTTP.
  const copyModelId = async (row: Row) => {
    const modelId = parseExternalModelId(row.id)?.modelId ?? row.label;
    if (await copyToClipboard(modelId)) {
      toast.success(`Copied ${modelId}`);
    } else {
      toast.error("Could not copy the model ID");
    }
  };

  // Folder models carry their store and path only in the inventory listing.
  const localInfo = useMemo(
    () => new Map(inventory.localModels.map((m) => [m.id, m])),
    [inventory.localModels],
  );
  const fineTunedSource = (row: Row): "training" | "exported" | null =>
    row.lora?.source === "training" || row.lora?.source === "exported"
      ? row.lora.source
      : null;

  const revealFor = (row: Row): (() => Promise<void>) | undefined => {
    const trained = fineTunedSource(row);
    if (trained) {
      return () => revealFineTunedModel(row.id, trained);
    }
    const local =
      row.lora?.source === "local" ? localInfo.get(row.id) : undefined;
    return local ? () => revealLocalModel(local.path, local.id) : undefined;
  };

  const deleteFor = (row: Row) => {
    const lora = row.lora;
    const trained = fineTunedSource(row);
    if (
      lora &&
      trained &&
      !(trained === "exported" && lora.exportType === "gguf")
    ) {
      return {
        title: "Delete fine-tuned model?",
        description: (
          <>
            This will remove{" "}
            <span className="font-medium text-foreground">{row.label}</span>{" "}
            from disk. This cannot be undone.
          </>
        ),
        successMessage: `Deleted ${row.label}`,
        disabled: row.loaded || source?.deleteDisabled,
        onConfirm: async () => {
          await deleteFineTunedModel({
            modelPath: row.id,
            source: trained,
            exportType: lora.exportType,
          });
          unpinRepo(row.id);
        },
        onDeleted: () => source?.onModelsChange({ id: row.id }),
      };
    }
    const local = lora?.source === "local" ? localInfo.get(row.id) : undefined;
    if (local && DELETABLE_LOCAL_SOURCES.has(local.source)) {
      return {
        title: "Delete local model?",
        description: (
          <>
            This will permanently remove{" "}
            <span className="font-medium text-foreground">{row.label}</span>{" "}
            from{" "}
            <span className="break-all font-mono text-xs">{local.path}</span>.
          </>
        ),
        successMessage: `Deleted ${row.label}`,
        disabled: row.loaded || source?.deleteDisabled,
        onConfirm: () => deleteLocalModel(local.path, local.id),
        onDeleted: () => {
          void inventory.refreshInventory();
          source?.onFoldersChange();
        },
      };
    }
    const npu = row.npu;
    if (npu?.downloaded && npuCatalog) {
      return {
        title: `Delete ${row.label}?`,
        description: "This removes the model's NPU files from this device.",
        successMessage: `Deleted ${row.label}`,
        disabled: row.loaded || source?.deleteDisabled,
        onConfirm: () => npuCatalog.remove(npu),
      };
    }
    const repo = row.gguf ?? row.cached;
    if (!repo) {
      return undefined;
    }
    return {
      title: "Delete cached model?",
      impact: { repoId: repo.repo_id },
      description: (
        <>
          This will remove{" "}
          <span className="font-medium text-foreground">{repo.repo_id}</span>
          {row.gguf ? " and every quant of it" : ""} from disk. You can
          re-download it later.
        </>
      ),
      successMessage: `Deleted ${repo.repo_id}`,
      disabled: row.loaded || source?.deleteDisabled,
      onConfirm: async () => {
        await deleteCachedModel(
          repo.repo_id,
          undefined,
          undefined,
          repo.cache_path || undefined,
        );
        unpinRepo(repo.repo_id);
      },
      onDeleted: () => {
        void inventory.refreshInventory();
      },
    };
  };

  const renderRow = (row: Row, index: number) => {
    const isChosen = row.id === chosen?.id;
    const favorite = isFavorite(row);
    const npuOffer = row.kind === "npu" && !row.npu?.downloaded;
    const npuProgress =
      row.npu && npuCatalog && row.npu.id in npuCatalog.downloads
        ? npuDownloadLabel(
            npuCatalog.downloads[row.npu.id],
            row.npu.id in npuCatalog.reconnecting,
          )
        : null;
    const quiet = (row.kind === "hub" && !row.onDevice) || npuOffer;
    const offering = row.kind === "hub" && quiet && Boolean(row.hub?.isGguf);
    const fit = offering ? hubFit.get(row.id.toLowerCase()) : null;
    // No GPU: every load runs from RAM, so `ram` is no overage.
    const dimmed = isOverBudget(fit) && !(fit === "ram" && noGpu) && !isChosen;
    const offerOpen = offer?.key === row.key;
    const paramLabel =
      row.kind === "hub" ? (hubParams.get(row.id.toLowerCase()) ?? null) : null;
    const onDisk = deviceQuants.get(row.key) ?? NO_QUANTS;
    const partials = partialsOf(row);
    const manyQuants = onDisk.length + partials.length > 1;
    const wholeQuant = onDisk.length === 1 ? onDisk[0].quant : null;
    const soleQuant = manyQuants ? null : wholeQuant;
    // Known without a quant listing, for Reset run settings and the gear's saved state.
    const savedMeta =
      row.kind === "gguf" && onDisk.length === 1
        ? ggufMeta(row, onDisk[0], true, true, row.gguf?.has_vision)
        : row.kind === "cached" && onDisk.length === 0 && !row.partial
          ? cachedMeta(row)
          : row.kind === "local" && row.lora
            ? loraMeta(row.lora)
            : null;
    const savedTarget = savedMeta
      ? modelConfigTarget(row.id, savedMeta, row.label)
      : undefined;
    // `foldedQuants` holds the rows toggled away from the "Expand quantizations" setting.
    const quantsOpen =
      manyQuants && expandQuantizations !== foldedQuants.has(row.key);
    // Discover GGUFs star by quant; a multi-quant device row stars the repo.
    const canStar = !(
      quiet ||
      (row.kind === "hub" && Boolean(row.hub?.isGguf))
    );
    const actionClass =
      "opacity-0 transition-opacity group-hover/row:opacity-100 group-data-[active=true]/row:opacity-100 group-data-[current=true]/row:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100";
    const rowActive = highlight && index === active;
    const rowCurrent = isChosen || Boolean(row.loaded);
    const rowEl = (
      <MemoRow
        key={row.key}
        deps={[
          row,
          index,
          isChosen,
          favorite,
          rowActive,
          offerOpen,
          onDisk,
          resolving === row.key,
          npuProgress,
          fit,
          noGpu,
          quantsOpen,
          viewed,
          source,
          chosen,
          activeVariant,
          localInfo,
          chatModels,
          row.npu ? npuCatalog : null,
          paramLabel,
        ]}
        render={() => (
          <div
            // biome-ignore lint/a11y/useSemanticElements: rows carry their own buttons, which <option> cannot.
            role="option"
            aria-expanded={
              offering || (row.kind === "hub" && row.hub?.isGguf) || offerOpen
                ? offerOpen
                : undefined
            }
            tabIndex={-1}
            id={`${OPTION_ID}-${index}`}
            aria-selected={isChosen}
            data-row-index={index}
            data-active={rowActive ? "true" : undefined}
            data-current={rowCurrent ? "true" : undefined}
            // Shared with the header picker's rows: the Playwright drivers select by these.
            data-model-picker-option=""
            data-model-picker-active-option={rowActive ? "true" : undefined}
            onMouseMove={() => {
              setActive(index);
              setHighlight(true);
            }}
            onClick={() => pick(row)}
            onKeyDown={(event) => {
              // Enter on a row's own button (eject, star, menu) is that button's.
              if (event.key === "Enter") {
                if (event.target === event.currentTarget) {
                  pick(row);
                }
                return;
              }
              if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                navFrom(
                  event.currentTarget,
                  event.key === "ArrowDown" ? 1 : -1,
                );
              }
            }}
            className={cn(
              "group/row flex h-[calc(32px*var(--ui-space-scale,1))] shrink-0 cursor-pointer items-center gap-2 rounded-full pr-[calc(6px*var(--ui-space-scale,1))] pl-3 text-ui-14 transition-colors",
              rowCurrent ? CURRENT_ROW_BG : rowActive && "bg-sidebar-accent",
            )}
          >
            {resolving === row.key || npuProgress ? (
              <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0" />
            ) : (
              <span
                className={cn(
                  "flex shrink-0 text-foreground",
                  dimmed && DIM_CLASS,
                )}
              >
                <RowMark row={row} />
              </span>
            )}
            <span className="flex min-w-0 flex-1 items-center gap-1.5">
              <span className={cn("truncate", dimmed && DIM_CLASS)}>
                {row.label}
              </span>
              {manyQuants ? (
                <HugeiconsIcon
                  icon={ChevronRightStandardIcon}
                  strokeWidth={1.75}
                  className={cn(
                    "size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground transition-transform duration-150",
                    quantsOpen && "rotate-90",
                  )}
                />
              ) : null}
              {row.loaded ? (
                <LoadedDot />
              ) : isChosen && row.kind === "external" ? (
                <LoadedDot label="In use" />
              ) : row.kind === "hub" && row.onDevice ? (
                <Hint label="On device">
                  <span
                    role="img"
                    aria-label="On device"
                    className="grid size-[calc(14px*var(--ui-space-scale,1))] shrink-0 place-items-center"
                  >
                    <span className="size-[calc(5px*var(--ui-space-scale,1))] rounded-full bg-status-success" />
                  </span>
                </Hint>
              ) : null}
            </span>
            {row.media ? (
              <StatusTag
                tone="accent"
                label="Audio"
                hint="Opens on the Audio page"
              />
            ) : null}
            {row.partial ? (
              <AmberTag
                label="Partial"
                hint={
                  row.resumable ? PARTIAL_HINT.resumable : PARTIAL_HINT.restart
                }
              />
            ) : onDisk.some((v) => v.update_available) ? (
              <AmberTag label="Update" hint={UPDATE_HINT} />
            ) : null}
            {soleQuant ? <QuantPill quant={soleQuant} /> : null}
            {row.kind === "hub" ? (
              <span className="-ml-[calc(2.75px*var(--ui-space-scale,1))] flex shrink-0 items-center gap-[calc(5px*var(--ui-space-scale,1))]">
                {offering ? (
                  <FitBadge
                    status={fit}
                    revealOnHover={!isChosen}
                    compact={true}
                  />
                ) : null}
                {paramLabel ? (
                  <span className={cn("flex", dimmed && DIM_CLASS)}>
                    <ParamChip label={paramLabel} />
                  </span>
                ) : null}
              </span>
            ) : null}
            {npuProgress || (npuOffer && row.npu?.size_gb != null) ? (
              <span className="shrink-0 text-ui-12 tabular-nums text-muted-foreground">
                {npuProgress ?? npuSizeLabel(row.npu?.size_gb ?? null)}
              </span>
            ) : row.kind === "npu" ? (
              <QuantPill quant="NPU" />
            ) : null}
            {isChosen && !row.loaded && row.kind !== "external" ? (
              <HugeiconsIcon
                icon={Tick02Icon}
                strokeWidth={2}
                className="size-4 shrink-0 text-muted-foreground"
              />
            ) : null}
            <span
              role="toolbar"
              aria-label={`${row.label} actions`}
              className="-ml-[calc(3px*var(--ui-space-scale,1))] flex shrink-0 items-center"
              // Menus portal out but bubble through React: keep their clicks off the row.
              onClick={(event) => event.stopPropagation()}
              onKeyDown={(event) => event.stopPropagation()}
            >
              {row.kind !== "external" && !npuOffer ? (
                <>
                  <ModelRowMenu
                    ariaLabel={`More options for ${row.label}`}
                    cachePath={
                      row.gguf || row.cached ? { repoId: row.id } : undefined
                    }
                    onReveal={revealFor(row)}
                    details={
                      row.tab === DEVICE_TAB ? (
                        <ModelFacts
                          facts={modelFacts(
                            row,
                            onDisk,
                            chatModels.find(
                              (m) =>
                                m.id.toLowerCase() ===
                                (row.loadedId ?? row.id).toLowerCase(),
                            ),
                          )}
                        />
                      ) : undefined
                    }
                    update={wholeQuant ? updateFor(row, onDisk[0]) : undefined}
                    pin={
                      canStar
                        ? {
                            pinned: favorite,
                            pinLabel: "Favorite",
                            unpinLabel: "Remove favorite",
                            onToggle: () => toggleFavorite(row),
                          }
                        : undefined
                    }
                    items={[
                      ...(row.loaded && canEject
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
                              onSelect: () => eject(row),
                              destructive: true,
                            },
                          ]
                        : []),
                      {
                        key: "run-settings",
                        label: "Run settings",
                        icon: (
                          <HugeiconsIcon
                            icon={Settings01Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                        ),
                        onSelect: () => openSettings(row),
                      },
                      ...(hubParts(row.id)
                        ? [
                            {
                              key: "model-card",
                              label: "Open model card",
                              icon: (
                                <HugeiconsIcon
                                  icon={DashboardCircleIcon}
                                  strokeWidth={1.75}
                                  className="size-icon"
                                />
                              ),
                              onSelect: () => openModelCard(row),
                            },
                          ]
                        : []),
                    ]}
                    del={deleteFor(row)}
                    runSettings={savedTarget}
                    buttonClassName={cn(ACTION_DOTS, actionClass)}
                    iconClassName="composer-row-dots"
                    contentClassName="composer-row-menu"
                  />
                  <ModelLoadSettingsAction
                    ariaLabel={`Run settings for ${row.label}`}
                    tooltip="Load settings"
                    savedFor={savedTarget}
                    onConfigure={() => openSettings(row)}
                    className={cn(ACTION_ROUND, actionClass)}
                    iconClassName="composer-row-icon"
                  />
                </>
              ) : null}
              {row.kind === "external" ? (
                <>
                  <ModelRowMenu
                    ariaLabel={`More options for ${row.label}`}
                    items={[
                      {
                        key: "model-settings",
                        label: "Model settings",
                        icon: (
                          <HugeiconsIcon
                            icon={Settings01Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                        ),
                        onSelect: () => openConnectedSettings(row),
                      },
                      {
                        key: "model-info",
                        label: "Model info",
                        icon: (
                          <HugeiconsIcon
                            icon={InformationCircleIcon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                        ),
                        onSelect: () => openConnectedSettings(row, "info"),
                      },
                      {
                        key: "copy-id",
                        label: "Copy model ID",
                        icon: (
                          <HugeiconsIcon
                            icon={Copy01Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                        ),
                        onSelect: () => void copyModelId(row),
                      },
                    ]}
                    buttonClassName={cn(ACTION_DOTS, actionClass)}
                    iconClassName="composer-row-dots"
                    contentClassName="composer-row-menu"
                  />
                  <ModelLoadSettingsAction
                    ariaLabel={`Settings for ${row.label}`}
                    tooltip="Model settings"
                    onConfigure={() => openConnectedSettings(row)}
                    className={cn(ACTION_ROUND, actionClass)}
                    iconClassName="composer-row-icon"
                  />
                </>
              ) : null}
              {canStar ? (
                <Hint label={favorite ? "Remove favorite" : "Favorite"}>
                  <button
                    type="button"
                    aria-label={
                      favorite
                        ? `Remove ${row.label} from favorites`
                        : `Favorite ${row.label}`
                    }
                    aria-pressed={favorite}
                    onClick={() => toggleFavorite(row)}
                    className={cn(
                      ACTION_ROUND,
                      favorite ? FAVORITE_STAR : actionClass,
                    )}
                  >
                    <HugeiconsIcon
                      icon={StarIcon}
                      strokeWidth={1.75}
                      className={cn(
                        "composer-row-icon",
                        favorite && "[&_path]:fill-current",
                      )}
                    />
                  </button>
                </Hint>
              ) : null}
            </span>
          </div>
        )}
      />
    );
    const memoryQuant = isChosen
      ? (row.loadedQuant ?? activeVariant ?? wholeQuant)
      : null;
    const memoryVariant = memoryQuant
      ? onDisk.find((v) => v.quant.toLowerCase() === memoryQuant.toLowerCase())
      : undefined;
    const memoryBar =
      showMemoryBar && row.kind === "gguf" && memoryQuant ? (
        // Draws nothing where VRAM is not a ceiling (Apple, shared iGPU memory), so no wrapper.
        <ModelMemoryBarFor
          key={`memory:${row.key}`}
          repoId={row.id}
          quant={memoryQuant}
          sizeBytes={memoryVariant ? quantBytes(memoryVariant) : null}
          loadId={row.gguf?.load_id ?? null}
          gpuGb={inferenceGpu.memoryTotalGb}
          compact={true}
          className="mt-0.5 mr-3 mb-1.5 ml-[calc(40px*var(--ui-space-scale,1))] w-auto shrink-0"
        />
      ) : null;
    const quantList = offerOpen ? (
      renderOffer(row)
    ) : quantsOpen || partials.length > 0 ? (
      <Fragment key={`quants:${row.key}`}>
        {quantsOpen ? renderDeviceQuants(row, onDisk) : null}
        {quantsOpen || !manyQuants
          ? partials.length > 0
            ? renderPartialQuants(row, partials)
            : null
          : null}
      </Fragment>
    ) : null;
    const favoriteKey =
      viewed === FAVORITES_TAB && !query ? favoriteKeyOf(row) : null;
    if (favoriteKey) {
      const drag =
        row.kind === "external" ? connectedFavoriteDrag : localFavoriteDrag;
      const edge = drag.lineEdge(favoriteKey);
      return (
        <div
          key={row.key}
          {...drag.rowProps(favoriteKey)}
          className={cn(
            "relative shrink-0",
            edge && [DROP_CUE_CLASS, FAVORITE_DROP_CUE[edge]],
          )}
          style={
            drag.draggingKey === favoriteKey ? { opacity: 0.4 } : undefined
          }
        >
          {rowEl}
          {memoryBar}
          {quantList}
        </div>
      );
    }
    if (!(quantList || memoryBar)) {
      return rowEl;
    }
    return (
      <Fragment key={row.key}>
        {rowEl}
        {memoryBar}
        {quantList}
      </Fragment>
    );
  };

  const renderQuantActions = (
    row: Row,
    variant: GgufVariantDetail,
    onDisk: boolean,
    isGguf = true,
    hasVision: boolean | undefined = row.gguf?.has_vision,
  ) => {
    const exportedGguf =
      row.lora?.source === "exported" && row.lora.exportType === "gguf";
    const key = `${row.id}::${variant.quant}`;
    const starred = pinned.includes(key);
    const button = ACTION_ROUND;
    const reveal =
      "opacity-0 transition-opacity group-hover/row:opacity-100 group-data-[current=true]/row:opacity-100 focus-visible:opacity-100";
    const quantConfigTarget = modelConfigTarget(
      row.id,
      ggufMeta(row, variant, onDisk, isGguf, hasVision),
      row.label,
    );
    return (
      <span className="flex shrink-0 items-center">
        {(onDisk && (row.gguf || exportedGguf)) ||
        (variant.partial === true && row.gguf && row.tab === DEVICE_TAB) ? (
          <ModelRowMenu
            ariaLabel={`More options for ${row.label} ${variant.quant}`}
            cachePath={
              row.gguf ? { repoId: row.id, variant: variant.quant } : undefined
            }
            update={updateFor(row, variant)}
            del={deleteQuantFor(row, variant)}
            runSettings={quantConfigTarget}
            buttonClassName={cn(ACTION_DOTS, reveal)}
            iconClassName="composer-row-dots"
            contentClassName="composer-row-menu"
          />
        ) : null}
        <ModelLoadSettingsAction
          ariaLabel={`Run settings for ${row.label} ${variant.quant}`}
          tooltip="Load settings"
          savedFor={quantConfigTarget}
          onConfigure={() => openConfig(quantConfigTarget)}
          className={cn(button, reveal)}
          iconClassName="composer-row-icon"
        />
        {onDisk ? (
          <Hint label={starred ? "Remove favorite" : "Favorite"}>
            <button
              type="button"
              aria-label={
                starred
                  ? `Remove ${row.label} ${variant.quant} from favorites`
                  : `Favorite ${row.label} ${variant.quant}`
              }
              aria-pressed={starred}
              onClick={() => togglePinned(row.id, variant.quant)}
              className={cn(button, starred ? FAVORITE_STAR : reveal)}
            >
              <HugeiconsIcon
                icon={StarIcon}
                strokeWidth={1.75}
                className={cn(
                  "composer-row-icon",
                  starred && "[&_path]:fill-current",
                )}
              />
            </button>
          </Hint>
        ) : (
          <span
            aria-hidden={true}
            className="w-[calc(20px*var(--ui-space-scale,1))] shrink-0"
          />
        )}
      </span>
    );
  };

  const renderQuantRow = (
    variant: GgufVariantDetail,
    {
      loaded,
      recommended,
      onPick,
      actions,
      onDevice = false,
    }: {
      loaded: boolean;
      recommended: boolean;
      onPick: () => void;
      actions: ReactNode;
      onDevice?: boolean;
    },
  ) => {
    const fit = classifyVariantFitOnCard(quantBudget.fitOf, variant, 0);
    return (
      <div
        key={variant.filename || variant.quant}
        data-current={loaded ? "true" : undefined}
        className={cn(
          "group/row flex h-[calc(26px*var(--ui-space-scale,1))] shrink-0 items-center rounded-full pr-[calc(6px*var(--ui-space-scale,1))] transition-colors hover:bg-sidebar-accent has-[[data-quant-option]:focus-visible]:bg-sidebar-accent",
          loaded && CURRENT_ROW_BG,
        )}
      >
        <button
          type="button"
          data-model-picker-option=""
          data-quant-option=""
          onClick={onPick}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              navFrom(event.currentTarget, event.key === "ArrowDown" ? 1 : -1);
            }
          }}
          className="flex h-full min-w-0 flex-1 items-center gap-2 pr-2 pl-[calc(40px*var(--ui-space-scale,1))] text-left"
        >
          <span className="flex min-w-0 flex-1 items-center gap-1.5">
            <span
              className={cn(
                "truncate font-mono text-xs",
                !onDevice &&
                  fit === "oom" &&
                  "!text-gray-500 dark:!text-gray-400",
              )}
            >
              {ggufVariantPickerLabel(variant)}
            </span>
            <QuantStatus
              variant={variant}
              loaded={loaded}
              recommended={recommended}
              onDevice={onDevice}
            />
          </span>
          <span className="flex shrink-0 items-center gap-1.5">
            <FitBadge status={fit} />
            <span className="font-mono text-ui-10 tabular-nums text-muted-foreground">
              {formatBytes(quantBytes(variant))}
            </span>
          </span>
        </button>
        {actions}
      </div>
    );
  };

  /** An interrupted quant beside loadable ones: picking it resumes the download, its menu deletes it. */
  const renderPartialQuants = (row: Row, list: GgufVariantDetail[]) => (
    <div className="flex shrink-0 flex-col">
      {list.map((variant) =>
        renderQuantRow(variant, {
          loaded: false,
          recommended: false,
          onPick: () => pickOfferQuant(row, variant),
          actions: renderQuantActions(row, variant, false),
          onDevice: true,
        }),
      )}
    </div>
  );

  const renderDeviceQuants = (row: Row, list: GgufVariantDetail[]) => {
    const loadedQuant = (
      row.loadedQuant ?? (row.id === chosen?.id ? activeVariant : null)
    )?.toLowerCase();
    const { ordered, recommended } = pickerOrderedGgufVariants(
      list,
      null,
      quantBudget,
    );
    return (
      <div className="flex shrink-0 flex-col">
        {ordered.map((variant) =>
          renderQuantRow(variant, {
            loaded:
              Boolean(row.loaded) &&
              variant.quant.toLowerCase() === loadedQuant,
            recommended: recommended.has(variant.quant),
            onPick: () => {
              if (!source) {
                return;
              }
              if (row.id !== chosen?.id) {
                setPending(row);
              }
              supersede();
              source.onPick(row.id, ggufMeta(row, variant, true));
              resetView();
            },
            actions: renderQuantActions(row, variant, true),
            onDevice: true,
          }),
        )}
      </div>
    );
  };

  const renderOffer = (row: Row) => {
    if (!offer) {
      return null;
    }
    if (offer.quants === null) {
      return (
        <div className="flex h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center gap-2 pl-[calc(40px*var(--ui-space-scale,1))] text-ui-13 text-muted-foreground">
          <Spinner className="size-[calc(14px*var(--ui-space-scale,1))]" />
          Loading quantizations
        </div>
      );
    }
    if (offer.failed || offer.quants.length === 0) {
      const target = offer.target;
      return (
        <div className="flex min-h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center gap-2 pr-[calc(6px*var(--ui-space-scale,1))] pl-[calc(40px*var(--ui-space-scale,1))] text-ui-13 text-muted-foreground">
          <span className="min-w-0 flex-1">
            {offer.error ?? "Couldn't list quantizations"}
          </span>
          {offer.failed && target ? (
            <Button
              variant="ghost"
              size="xs"
              onClick={() => void toggleOffer(row, target, supersede(), true)}
              className="shrink-0"
            >
              Retry
            </Button>
          ) : null}
        </div>
      );
    }
    const { ordered, recommended } = pickerOrderedGgufVariants(
      offer.quants,
      offer.defaultVariant,
      quantBudget,
    );
    const loadedQuant = row.loaded
      ? (
          row.loadedQuant ?? (row.id === chosen?.id ? activeVariant : null)
        )?.toLowerCase()
      : null;
    return (
      <div className="flex shrink-0 flex-col">
        {ordered.map((variant) =>
          renderQuantRow(variant, {
            loaded: variant.quant.toLowerCase() === loadedQuant,
            recommended: recommended.has(variant.quant),
            onPick: () =>
              pickOfferQuant(row, variant, !offer.mlx, offer.hasVision),
            actions: renderQuantActions(
              row,
              variant,
              variant.downloaded === true,
              !offer.mlx,
              offer.hasVision ?? row.gguf?.has_vision,
            ),
          }),
        )}
      </div>
    );
  };

  const renderConnectedSettings = (row: Row, view: "settings" | "info") => {
    const parsed = parseExternalModelId(row.id);
    const provider = providers.find((p) => p.id === parsed?.providerId);
    const reasoningConfig =
      provider?.backendProviderType === "custom" && !provider.decisionsOnly
        ? provider.reasoningConfig
        : undefined;
    if (view === "info") {
      return (
        <ConnectedModelInfoDialog
          open={true}
          onOpenChange={(next) => {
            if (!next) {
              setConnectedSettings(null);
            }
          }}
          modelId={parsed?.modelId ?? row.label}
          displayName={row.label}
          providerName={row.providerName ?? provider?.name ?? ""}
          providerType={row.providerType ?? provider?.providerType ?? ""}
          apiType={provider?.apiType}
          baseUrl={provider?.baseUrl ?? null}
          isReasoningProvider={provider?.isReasoningModel === true}
          reasoningConfig={reasoningConfig}
        />
      );
    }
    return (
      <ConnectedModelSettingsDialog
        open={true}
        onOpenChange={(next) => {
          if (!next) {
            setConnectedSettings(null);
          }
        }}
        checkpointId={row.id}
        displayName={row.label}
        modelId={parsed?.modelId ?? row.label}
        providerType={row.providerType ?? provider?.providerType ?? ""}
        apiType={provider?.apiType}
        baseUrl={provider?.baseUrl ?? null}
        isReasoningProvider={provider?.isReasoningModel === true}
        reasoningConfig={reasoningConfig}
        connectionMaxOutputTokens={provider?.maxOutputTokens ?? null}
      />
    );
  };

  // Until the device inventory or the Hub answers, an empty list is unknown, not "no models".
  const listLoading =
    viewed === RECOMMENDED_TAB
      ? online && (hubSearch.isLoading || query.trim() !== hubQuery)
      : !(
          viewed.startsWith(EXT_PREFIX) ||
          viewed === CONNECTED_TAB ||
          inventory.cachedReady
        );
  const renderEmpty = () => {
    if (listLoading) {
      return (
        <div
          key="empty"
          aria-live="polite"
          className="flex items-center justify-center gap-2 px-4 py-8 text-ui-13 text-muted-foreground"
        >
          <Spinner className="size-[calc(14px*var(--ui-space-scale,1))]" />
          Loading models
        </div>
      );
    }
    const copy = emptyCopy();
    return (
      <div
        key="empty"
        className="flex flex-col items-center gap-1 px-4 py-8 text-center"
      >
        <span className="text-ui-14 font-medium text-foreground">
          {copy.title}
        </span>
        {copy.hint ? (
          <span className="text-ui-12 text-muted-foreground">{copy.hint}</span>
        ) : null}
        {copy.actions ? (
          <div className="mt-2 flex flex-col items-center gap-2">
            {[0, 1]
              .map((row) => copy.actions?.filter((a) => a.row === row) ?? [])
              .map((group) =>
                group.length > 0 ? (
                  <div
                    key={group[0].label}
                    className="flex flex-wrap justify-center gap-2"
                  >
                    {group.map((action) => (
                      <Button
                        key={action.label}
                        variant={action.primary ? "dark" : "outline"}
                        size="sm"
                        onClick={action.run}
                      >
                        {action.icon}
                        {action.label}
                      </Button>
                    ))}
                  </div>
                ) : null,
              )}
          </div>
        ) : null}
      </div>
    );
  };

  const renderList = () => {
    const hubHint =
      viewed === RECOMMENDED_TAB &&
      hubResults.length === 0 &&
      (!online || hubSearch.error) ? (
        <HubFailureHint
          key="hub-failure"
          message={hubSearch.error}
          onRetry={hubSearch.retry}
        />
      ) : null;
    if (hubHint) {
      return [hubHint, ...[renderSections()].flat()];
    }
    return renderSections();
  };

  const renderJumps = (below: readonly ListSection[]) => {
    const targets = below.filter((s) => JUMP_LABEL[s] && SECTION_ICON[s]);
    if (targets.length === 0) {
      return undefined;
    }
    return targets.map((target) => (
      <Hint key={target} label={JUMP_LABEL[target] ?? ""}>
        <button
          type="button"
          tabIndex={-1}
          onClick={() => jumpTo(target)}
          aria-label={JUMP_LABEL[target]}
          className="grid size-[calc(20px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
        >
          {SECTION_ICON[target]}
        </button>
      </Hint>
    ));
  };

  const renderConnectedGroups = () => {
    const out: ReactNode[] = [];
    const indexOf = new Map(shownRows.map((row, i) => [row.key, i]));
    let group: string | null = null;
    for (const row of foundRows) {
      if (row.tab !== group) {
        group = row.tab;
        const providerId = row.tab.slice(EXT_PREFIX.length);
        const name = row.providerName ?? "Connection";
        out.push(
          <SectionHeader
            key={`section:${row.tab}`}
            section={row.tab}
            label={name}
            divider={out.length > 0}
            collapsed={collapsedSections.has(sectionKey(row.tab))}
            onToggle={() => toggleSection(row.tab)}
            trailing={
              <Hint label="Connection settings">
                <button
                  type="button"
                  tabIndex={-1}
                  aria-label={`Open ${name} connection settings`}
                  onClick={() => {
                    setOpen(false);
                    useSettingsDialogStore
                      .getState()
                      .openConnectionSettings(providerId);
                  }}
                  className="grid size-[calc(20px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={Settings01Icon}
                    strokeWidth={1.75}
                    className="size-3"
                  />
                </button>
              </Hint>
            }
          />,
        );
      }
      const index = indexOf.get(row.key);
      if (index !== undefined) {
        out.push(renderRow(row, index));
      }
    }
    return out;
  };

  const renderSections = () => {
    if (viewed === CONNECTED_TAB && foundRows.length > 0) {
      return renderConnectedGroups();
    }
    if (listSections.length === 0) {
      return shownRows.length === 0 ? renderEmpty() : shownRows.map(renderRow);
    }
    const indexOf = new Map(shownRows.map((row, i) => [row.key, i]));
    const out: ReactNode[] = [];
    const anyRows = sectionsHaveRows(listSections, foundRows);
    if (!anyRows) {
      out.push(renderEmpty());
    }
    listSections.forEach((section, i) => {
      const collapsed = collapsedSections.has(sectionKey(section));
      out.push(
        <SectionHeader
          key={`section:${section}`}
          section={section}
          label={SECTION_LABEL[section]}
          icon={SECTION_ICON[section]}
          divider={i > 0 || !anyRows}
          collapsed={collapsed}
          onToggle={() => toggleSection(section)}
          actions={section === "custom" ? customFolders.actions : undefined}
          trailing={i === 0 ? renderJumps(listSections.slice(1)) : undefined}
        />,
      );
      if (collapsed) {
        return;
      }
      if (section === "custom") {
        out.push(
          <Fragment key="custom-folders">{customFolders.body}</Fragment>,
        );
      }
      if (
        section === "npu" &&
        viewed === RECOMMENDED_TAB &&
        npuCatalog &&
        !npuCatalog.ready
      ) {
        out.push(<NpuSetupNotice key="npu-setup" catalog={npuCatalog} />);
      }
      for (const row of shownRows) {
        if (row.section === section) {
          out.push(renderRow(row, indexOf.get(row.key) ?? 0));
        }
      }
    });
    return out;
  };

  const formatGroup = (
    <>
      <DropdownMenuLabel className={MENU_LABEL}>Format</DropdownMenuLabel>
      <DropdownMenuRadioGroup
        value={formatFilter}
        onValueChange={(next) => setFormatFilter(next as FormatFilter)}
      >
        {(Object.keys(FORMAT_FILTER_LABEL) as FormatFilter[])
          .filter((key) => key !== "npu" || npuCatalog !== null)
          .map((key) => (
            <DropdownMenuRadioItem
              key={key}
              value={key}
              className={cn(MENU_ITEM, "gap-1.5")}
            >
              {FORMAT_DOT[key] ? (
                <span
                  className={cn(
                    "size-[calc(5px*var(--ui-space-scale,1))] shrink-0 rounded-full",
                    FORMAT_DOT[key],
                  )}
                />
              ) : null}
              {FORMAT_FILTER_LABEL[key]}
            </DropdownMenuRadioItem>
          ))}
      </DropdownMenuRadioGroup>
    </>
  );
  const viewedTab = tabs.find((t) => t.value === viewed);
  // Only when the tab changes, so a sideways scroll of the user's own is left alone.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `viewed` is the trigger; the DOM holds the selection.
  useEffect(() => {
    tabStrip
      ?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [tabStrip, viewed]);
  const connectionId = viewed.startsWith(EXT_PREFIX)
    ? viewed.slice(EXT_PREFIX.length)
    : null;

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      {chipDetail ? (
        <Tooltip delayDuration={400}>
          <TooltipTrigger asChild={true}>
            <PopoverTrigger asChild={true}>
              <button
                ref={triggerRef}
                type="button"
                className={cn(
                  "unsloth-thinking-pill composer-model-chip group/chip",
                  !(chosenName || modelLoading || pending) &&
                    "composer-model-chip-empty",
                )}
                data-pill-label={chipLabel}
                data-tour={pane ? undefined : "chat-model-selector"}
                aria-label={`Model: ${chipLabel}${reasoningChip ? `, ${reasoningChip}` : ""}`}
              >
                {modelLoading || pending ? (
                  <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0" />
                ) : chosenName && chosenLoaded ? (
                  <Hint label="Eject model">
                    {/* biome-ignore lint/a11y/useKeyWithClickEvents: mouse-only shortcut; keyboard and screen readers eject from the panel. */}
                    <span
                      aria-hidden={true}
                      data-eject-hit={true}
                      onPointerDown={(event) => event.stopPropagation()}
                      onClick={(event) => {
                        event.stopPropagation();
                        if (chosen) {
                          eject(chosen);
                        } else if (checkpoint) {
                          source?.eject();
                        }
                      }}
                      className="-m-0.5 grid size-[calc(18px*var(--ui-space-scale,1))] shrink-0 cursor-pointer place-items-center rounded-full transition-colors hover:bg-[rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] [@media(hover:none)]:pointer-events-none"
                    >
                      <span className="col-start-1 row-start-1 grid place-items-center group-hover/chip:invisible">
                        <RowMark row={chosenMark} />
                      </span>
                      <HugeiconsIcon
                        icon={RemoveCircleIcon}
                        strokeWidth={1.75}
                        className="invisible col-start-1 row-start-1 size-[calc(14px*var(--ui-space-scale,1))] composer-eject-tone group-hover/chip:visible"
                      />
                    </span>
                  </Hint>
                ) : chosenName ? (
                  <RowMark row={chosenMark} />
                ) : null}
                <span className="unsloth-thinking-label composer-model-chip-name truncate">
                  {chipLabel}
                </span>
                {(chosenName || modelLoading || pending) && !pane ? null : (
                  <HugeiconsIcon
                    icon={ChevronDownStandardIcon}
                    strokeWidth={1.75}
                    className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
                  />
                )}
                {reasoningChip ? (
                  <span className="unsloth-thinking-label composer-model-chip-detail shrink-0">
                    {reasoningChip}
                  </span>
                ) : null}
              </button>
            </PopoverTrigger>
          </TooltipTrigger>
          <TooltipContent side="top" className="tooltip-compact">
            {chipDetail}
          </TooltipContent>
        </Tooltip>
      ) : (
        <PopoverTrigger asChild={true}>
          <button
            ref={triggerRef}
            type="button"
            className={cn(
              "unsloth-thinking-pill composer-model-chip group/chip",
              !(chosenName || modelLoading || pending) &&
                "composer-model-chip-empty",
            )}
            data-pill-label={chipLabel}
            data-tour={pane ? undefined : "chat-model-selector"}
            aria-label={`Model: ${chipLabel}${reasoningChip ? `, ${reasoningChip}` : ""}`}
          >
            {modelLoading || pending ? (
              <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0" />
            ) : chosenName && chosenLoaded ? (
              <Hint label="Eject model">
                {/* biome-ignore lint/a11y/useKeyWithClickEvents: mouse-only shortcut; keyboard and screen readers eject from the panel. */}
                <span
                  aria-hidden={true}
                  data-eject-hit={true}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    if (chosen) {
                      eject(chosen);
                    } else if (checkpoint) {
                      source?.eject();
                    }
                  }}
                  className="-m-0.5 grid size-[calc(18px*var(--ui-space-scale,1))] shrink-0 cursor-pointer place-items-center rounded-full transition-colors hover:bg-[rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] [@media(hover:none)]:pointer-events-none"
                >
                  <span className="col-start-1 row-start-1 grid place-items-center group-hover/chip:invisible">
                    <RowMark row={chosenMark} />
                  </span>
                  <HugeiconsIcon
                    icon={RemoveCircleIcon}
                    strokeWidth={1.75}
                    className="invisible col-start-1 row-start-1 size-[calc(14px*var(--ui-space-scale,1))] composer-eject-tone group-hover/chip:visible"
                  />
                </span>
              </Hint>
            ) : chosenName ? (
              <RowMark row={chosenMark} />
            ) : null}
            <span className="unsloth-thinking-label composer-model-chip-name truncate">
              {chipLabel}
            </span>
            {(chosenName || modelLoading || pending) && !pane ? null : (
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={1.75}
                className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
              />
            )}
            {reasoningChip ? (
              <span className="unsloth-thinking-label composer-model-chip-detail shrink-0">
                {reasoningChip}
              </span>
            ) : null}
          </button>
        </PopoverTrigger>
      )}
      {/* No collision flip: a height change must not move the panel across the composer. */}
      <PopoverContent
        ref={setPanel}
        side={openSide}
        avoidCollisions={false}
        collisionPadding={PICKER_EDGE}
        align={align}
        alignOffset={alignOffset}
        sideOffset={dropsDown ? dropOffset : PICKER_GAP}
        data-composer-model-picker=""
        data-tour="chat-model-selector-popover"
        data-list-last={
          isExpanded && !hasFoot && !configTarget ? "" : undefined
        }
        data-run-settings={configTarget ? "" : undefined}
        aria-label={
          configTarget
            ? `Run settings for ${configTarget.displayName}`
            : undefined
        }
        className={cn(
          "unsloth-plus-menu composer-model-picker relative w-[calc(420px*var(--ui-space-scale,1))] gap-0 overflow-hidden p-2",
          (isExpanded || configTarget) &&
            "h-[calc(460px*var(--ui-space-scale,1))]",
          configTarget &&
            "has-[[role=switch][aria-checked=true][aria-label*=advanced]]:h-[calc(640px*var(--ui-space-scale,1))]",
          // Opening down, it keeps its height and rides up over the composer instead (pickerDropOffset).
          openSide === "bottom"
            ? "max-h-[min(calc(760px*var(--ui-space-scale,1)),calc(100dvh-16px))]"
            : "max-h-[min(calc(760px*var(--ui-space-scale,1)),var(--radix-popover-content-available-height))]",
        )}
        onOpenAutoFocus={(event) => {
          if (isExpanded) {
            requestAnimationFrame(() => searchRef.current?.focus());
          } else {
            event.preventDefault();
          }
        }}
        onEscapeKeyDown={(event) => {
          if (quantsOpen) {
            event.preventDefault();
            setQuantsOpen(false);
          }
        }}
      >
        {configTarget ? (
          <>
            <div
              ref={settingsRef}
              className="composer-run-settings min-h-0 overflow-x-hidden overflow-y-auto overscroll-none pr-[var(--run-settings-pad-r)] pl-[var(--run-settings-pad-l)] [--run-settings-pad-l:calc(22px*var(--ui-space-scale,1))] [--run-settings-pad-r:calc(22px*var(--ui-space-scale,1))]"
            >
              <ModelConfigPage
                key={modelConfigInstanceKey(
                  configTarget.configId ?? configTarget.id,
                  configTarget.ggufVariant,
                  configResident ? (source?.activeModelConfig ?? null) : null,
                )}
                target={configTarget}
                onBack={
                  configStandalone ? undefined : () => setConfigTarget(null)
                }
                focusField={configFocus ?? undefined}
                onRun={(config, isDiffusion) => {
                  supersede();
                  source?.onPick(configTarget.configId ?? configTarget.id, {
                    ...configTarget.meta,
                    config,
                    isDiffusion,
                    forceReload: true,
                  });
                  onOpenChange(false);
                }}
                loadedConfig={
                  configResident ? (source?.activeModelConfig ?? null) : null
                }
                loadedContextLength={
                  configResident ? (source?.loadedContextLength ?? null) : null
                }
                initialConfig={
                  pane?.config &&
                  modelConfigTargetIsResident({
                    target: configTarget,
                    selectedId: pane.id,
                    activeGgufVariant: pane.ggufVariant,
                    loaded: true,
                  })
                    ? pane.config
                    : null
                }
              />
            </div>
            <ListScrollbar
              target={settingsRef}
              edge={true}
              stopAt="[data-pinned-footer]"
            />
          </>
        ) : isExpanded ? (
          <>
            <div className="relative flex shrink-0 items-center gap-1">
              <div
                ref={setTabStrip}
                role="tablist"
                aria-label="Model sources"
                className="composer-tab-strip flex min-w-0 flex-1 items-center gap-1 overflow-x-auto"
                onWheel={(event) => {
                  // A mouse wheel scrolls the tabs sideways; a trackpad already does.
                  if (Math.abs(event.deltaY) > Math.abs(event.deltaX)) {
                    event.currentTarget.scrollLeft += event.deltaY;
                  }
                }}
              >
                {tabs.map((t) => (
                  <TabButton
                    key={t.value}
                    tab={t}
                    shown={viewed === t.value}
                    onSelect={() => setTab(t.value)}
                    dragging={draggingTab}
                    onDragTab={setDraggingTab}
                    drop={tabDrop}
                    onDropTarget={setTabDrop}
                    onDropTab={moveTab}
                  />
                ))}
              </div>
              {(!pane || pane.loaded) &&
              (chosenLoaded || loadedModels.length > 0) &&
              !pending &&
              !modelLoading ? (
                <Button
                  variant="destructive"
                  size="xs"
                  aria-label={
                    ejectsAll
                      ? "Eject all models"
                      : `Eject ${
                          chosenLoaded
                            ? (chosenName ?? "model")
                            : modelDisplayName(loadedModels[0]?.id ?? "model")
                        }`
                  }
                  onClick={() => {
                    if (ejectsAll) {
                      source?.ejectAll();
                    } else if (!chosenLoaded && loadedModels[0]) {
                      source?.eject(
                        loadedModels[0].checkpoint ?? loadedModels[0].id,
                      );
                    } else if (chosen) {
                      eject(chosen);
                    } else {
                      source?.eject();
                    }
                  }}
                  className="ml-auto h-[calc(26px*var(--ui-space-scale,1))]"
                >
                  <HugeiconsIcon
                    icon={RemoveCircleIcon}
                    strokeWidth={1.75}
                    className="size-[calc(14px*var(--ui-space-scale,1))]"
                  />
                  {ejectsAll ? "Eject all" : "Eject"}
                </Button>
              ) : null}
              {chosenName ? (
                <Hint label="Collapse">
                  <button
                    type="button"
                    aria-label="Collapse the model list"
                    aria-expanded={true}
                    onClick={() => setExpanded(false)}
                    className="grid size-[calc(26px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={ChevronDownStandardIcon}
                      strokeWidth={1.5}
                      className="size-4 translate-x-[-0.4px] translate-y-[-0.4px] rotate-180"
                    />
                  </button>
                </Hint>
              ) : null}
            </div>
            <div className="mt-2 flex h-[calc(32px*var(--ui-space-scale,1))] shrink-0 items-center gap-2 rounded-full bg-muted pr-2 pl-3">
              <HugeiconsIcon
                icon={Search01Icon}
                strokeWidth={1.75}
                className="size-4 shrink-0 text-muted-foreground"
              />
              <SearchField
                inputRef={searchRef}
                value={query}
                onCommit={setQuery}
                onNavKey={onSearchNavKey}
                placeholder={
                  viewed === RECOMMENDED_TAB
                    ? "Search models to download"
                    : viewed === FAVORITES_TAB
                      ? "Search favorited models"
                      : viewed === DEVICE_TAB
                        ? "Search on device models"
                        : viewed === CONNECTED_TAB
                          ? "Search connected models"
                          : `Search ${viewedTab?.label ?? ""} models`
                }
                aria-label="Search models"
                aria-controls="composer-model-picker-list"
                aria-activedescendant={
                  highlight && active >= 0 && active < shownRows.length
                    ? `${OPTION_ID}-${active}`
                    : undefined
                }
                data-model-picker-search-input=""
                className="min-w-0 flex-1 bg-transparent text-ui-14 outline-none placeholder:text-muted-foreground"
              />
              {viewed === DEVICE_TAB ? (
                <NonModalDropdownMenu
                  align="end"
                  className="composer-filter-menu w-auto min-w-[calc(140px*var(--ui-space-scale,1))]"
                  trigger={(ref) => (
                    <button
                      ref={ref}
                      type="button"
                      aria-label={`Sort and filter: ${DEVICE_SORT_LABEL[deviceSort]} ${deviceSortDesc ? "descending" : "ascending"}, ${FORMAT_FILTER_LABEL[formatFilter]}`}
                      className="relative grid size-[calc(24px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground data-[state=open]:bg-sidebar-accent"
                    >
                      <HugeiconsIcon
                        icon={FilterMailIcon}
                        strokeWidth={1.75}
                        // Lifts the glyph's ink 0.5px to centre.
                        className="size-4 -translate-y-[0.5px]"
                      />
                      {formatFilter !== "all" ? (
                        <span className="absolute top-0.5 right-0.5 size-1.5 rounded-full bg-primary" />
                      ) : null}
                    </button>
                  )}
                >
                  <DropdownMenuLabel className={MENU_LABEL}>
                    Sort
                  </DropdownMenuLabel>
                  {/* As Library's columns: choosing the active sort again flips it. */}
                  <DropdownMenuRadioGroup value={deviceSort}>
                    {(Object.keys(DEVICE_SORT_LABEL) as DeviceSort[]).map(
                      (key) => {
                        const Arrow = deviceSortDesc
                          ? ArrowDownIcon
                          : ArrowUpIcon;
                        const order = DEVICE_SORT_ORDER[key];
                        const hint =
                          key === deviceSort
                            ? deviceSortDesc
                              ? order.desc
                              : order.asc
                            : DEVICE_SORT_DESC[key]
                              ? order.desc
                              : order.asc;
                        return (
                          <Hint key={key} label={hint} side="left">
                            <DropdownMenuRadioItem
                              value={key}
                              className={cn(
                                MENU_ITEM,
                                "[&_[data-slot=dropdown-menu-radio-item-indicator]]:hidden",
                              )}
                              onSelect={(event) => {
                                if (key === deviceSort) {
                                  event.preventDefault();
                                  setDeviceSortDesc((desc) => !desc);
                                  return;
                                }
                                setDeviceSort(key);
                                setDeviceSortDesc(DEVICE_SORT_DESC[key]);
                              }}
                            >
                              {DEVICE_SORT_LABEL[key]}
                              {key === deviceSort ? (
                                <span className="absolute right-[calc(5px*var(--ui-space-scale,1))] grid size-[calc(24px*var(--ui-space-scale,1))] place-items-center rounded-full transition-colors hover:bg-[rgb(0_0_0_/_calc(0.08*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]">
                                  <Arrow className="size-3.5" strokeWidth={2} />
                                  <span className="sr-only">
                                    {deviceSortDesc
                                      ? "descending"
                                      : "ascending"}
                                  </span>
                                </span>
                              ) : null}
                            </DropdownMenuRadioItem>
                          </Hint>
                        );
                      },
                    )}
                  </DropdownMenuRadioGroup>
                  <DropdownMenuSeparator />
                  {formatGroup}
                </NonModalDropdownMenu>
              ) : null}
              {source && viewed === CONNECTED_TAB ? (
                <Hint label="Connections">
                  <button
                    type="button"
                    aria-label="Open connection settings"
                    onClick={() => {
                      setOpen(false);
                      useSettingsDialogStore
                        .getState()
                        .openDialog("connections");
                    }}
                    className="grid size-[calc(24px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={CloudIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  </button>
                </Hint>
              ) : null}
              {viewed === RECOMMENDED_TAB ? (
                <NonModalDropdownMenu
                  align="end"
                  className="composer-filter-menu w-auto min-w-[calc(140px*var(--ui-space-scale,1))]"
                  trigger={(ref) => (
                    <button
                      ref={ref}
                      type="button"
                      aria-label={`Filter: ${FORMAT_FILTER_LABEL[formatFilter]}${fitOnDeviceOnly ? ", only models that fit" : ""}`}
                      className="relative grid size-[calc(24px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground data-[state=open]:bg-sidebar-accent"
                    >
                      <HugeiconsIcon
                        icon={FilterMailIcon}
                        strokeWidth={1.75}
                        className="size-4 -translate-y-[0.5px]"
                      />
                      {fitOnDeviceOnly || formatFilter !== "all" ? (
                        <span className="absolute top-0.5 right-0.5 size-1.5 rounded-full bg-primary" />
                      ) : null}
                    </button>
                  )}
                >
                  {formatGroup}
                  <DropdownMenuSeparator />
                  <Hint
                    label="Hides models larger than this device's memory budget. Downloaded models stay visible."
                    side="left"
                  >
                    <DropdownMenuItem
                      role="menuitemcheckbox"
                      aria-checked={fitOnDeviceOnly}
                      onSelect={(event) => {
                        event.preventDefault();
                        setFitOnDeviceOnly(!fitOnDeviceOnly);
                      }}
                      className={cn(MENU_ITEM, "gap-1.5 pr-2.5 pl-1.5")}
                    >
                      <span
                        aria-hidden={true}
                        className={cn(
                          "grid size-[calc(11px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full border",
                          fitOnDeviceOnly
                            ? "border-primary bg-primary !text-primary-foreground"
                            : "border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)]",
                        )}
                      >
                        {fitOnDeviceOnly ? (
                          <HugeiconsIcon
                            icon={Tick02Icon}
                            strokeWidth={2.5}
                            // !, on every element: the menu item's hover rule recolours each descendant.
                            className="size-2 !text-primary-foreground [&_*]:!text-primary-foreground"
                          />
                        ) : null}
                      </span>
                      Only show models that fit
                    </DropdownMenuItem>
                  </Hint>
                </NonModalDropdownMenu>
              ) : null}
              {source && connectionId ? (
                <Hint label="Connection settings">
                  <button
                    type="button"
                    aria-label={`Open ${viewedTab?.label ?? ""} connection settings`}
                    onClick={() => {
                      setOpen(false);
                      useSettingsDialogStore
                        .getState()
                        .openConnectionSettings(connectionId);
                    }}
                    className="grid size-[calc(24px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={CloudIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  </button>
                </Hint>
              ) : source &&
                viewed !== DEVICE_TAB &&
                viewed !== RECOMMENDED_TAB &&
                viewed !== CONNECTED_TAB ? (
                <Hint label="Model hub">
                  <button
                    type="button"
                    aria-label="Open the Model hub"
                    onClick={() => {
                      setOpen(false);
                      void navigate({
                        to: "/hub",
                        search: { tab: "downloaded" },
                      });
                    }}
                    className="grid size-[calc(24px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={DashboardCircleIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  </button>
                </Hint>
              ) : null}
            </div>
            <div
              ref={attachList}
              id="composer-model-picker-list"
              data-model-picker-list=""
              onMouseLeave={() => setHighlight(false)}
              // biome-ignore lint/a11y/useSemanticElements: a <select> cannot hold each row's buttons.
              role="listbox"
              tabIndex={-1}
              aria-label={viewedTab?.label}
              className="composer-model-list flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto pt-[calc(9px*var(--ui-space-scale,1))] pb-1.5"
            >
              {renderList()}
              {viewed === RECOMMENDED_TAB && hubSearch.hasMore ? (
                <div
                  ref={hubSentinelRef}
                  className="flex h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center justify-center"
                >
                  {hubSearch.isLoadingMore ? (
                    <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] text-muted-foreground" />
                  ) : null}
                </div>
              ) : null}
            </div>
            <ListScrollbar target={listRef} />
          </>
        ) : (
          <div className="group/collapsed flex h-[calc(32px*var(--ui-space-scale,1))] w-full shrink-0 items-center">
            <button
              type="button"
              aria-expanded={false}
              aria-label={`${chosenName}: show all models`}
              onClick={expand}
              // 1px up, matching the gap above the first control below the line.
              className="flex min-w-0 flex-1 items-center gap-2.5 self-stretch pb-[calc(2px*var(--ui-space-scale,1))] pl-3 text-left text-ui-14"
            >
              {pending || modelLoading ? (
                <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0" />
              ) : (
                <RowMark row={chosenMark} />
              )}
              <span className="truncate">{chosenName}</span>
              {!pending && chosenLoaded ? (
                <Hint label="Loaded">
                  <span className="size-[calc(6px*var(--ui-space-scale,1))] shrink-0 rounded-full bg-status-success" />
                </Hint>
              ) : null}
            </button>
            {!pending && chosenLoaded && !modelLoading ? (
              <Button
                variant="destructive"
                size="xs"
                aria-label={`Eject ${chosenName}`}
                onClick={() => (chosen ? eject(chosen) : source?.eject())}
                className="h-[calc(26px*var(--ui-space-scale,1))]"
              >
                <HugeiconsIcon
                  icon={RemoveCircleIcon}
                  strokeWidth={1.75}
                  className="size-[calc(14px*var(--ui-space-scale,1))]"
                />
                Eject
              </Button>
            ) : null}
            <button
              type="button"
              tabIndex={-1}
              aria-hidden={true}
              onClick={expand}
              className="ml-1 grid size-[calc(26px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors group-hover/collapsed:text-foreground hover:bg-sidebar-accent"
            >
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={1.5}
                className="size-4 translate-x-[-0.4px] translate-y-[0.4px]"
              />
            </button>
          </div>
        )}
        {hasFoot && !configTarget ? (
          <div
            className={cn(
              "flex shrink-0 flex-col gap-0.5",
              isExpanded
                ? "border-t border-border pt-[calc(5px*var(--ui-space-scale,1))]"
                : "mt-[calc(6px*var(--ui-space-scale,1))] border-t border-border pt-[calc(4px*var(--ui-space-scale,1))]",
            )}
          >
            {showQuantRow ? (
              <Popover open={quantsOpen} onOpenChange={setQuantsOpen}>
                <PopoverTrigger asChild={true}>
                  <button
                    type="button"
                    className="flex h-[calc(32px*var(--ui-space-scale,1))] items-center gap-2 rounded-full px-3 text-left text-ui-14 transition-colors hover:bg-sidebar-accent data-[state=open]:bg-sidebar-accent"
                  >
                    <span className="flex-1">Quantization</span>
                    <span className="max-w-[calc(120px*var(--ui-space-scale,1))] truncate text-muted-foreground">
                      {activeVariant ?? quantChoices[0]?.quant}
                    </span>
                    <HugeiconsIcon
                      icon={ChevronDownStandardIcon}
                      strokeWidth={1.5}
                      className="size-4 -rotate-90 text-muted-foreground"
                    />
                  </button>
                </PopoverTrigger>
                <PopoverContent
                  side="right"
                  align="start"
                  sideOffset={12}
                  className="unsloth-plus-menu w-[calc(240px*var(--ui-space-scale,1))] gap-0.5 p-2"
                >
                  {quantChoices.map((variant) => (
                    <button
                      key={variant.quant}
                      type="button"
                      onClick={() => pickQuant(variant)}
                      className="flex h-[calc(28px*var(--ui-space-scale,1))] items-center gap-2 rounded-full px-3 text-left transition-colors hover:bg-sidebar-accent"
                    >
                      <span className="flex min-w-0 flex-1 items-center gap-1.5">
                        <span className="truncate font-mono text-xs">
                          {ggufVariantPickerLabel(variant)}
                        </span>
                        {variant.update_available ? (
                          <StatusTag
                            tone="amber"
                            label="Update"
                            hint={UPDATE_HINT}
                          />
                        ) : null}
                      </span>
                      <span className="font-mono text-ui-10 tabular-nums text-muted-foreground">
                        {formatBytes(quantBytes(variant))}
                      </span>
                      <HugeiconsIcon
                        icon={Tick02Icon}
                        strokeWidth={2}
                        className={cn(
                          "size-4",
                          variant.quant.toLowerCase() !==
                            (activeVariant ?? "").toLowerCase() && "opacity-0",
                        )}
                      />
                    </button>
                  ))}
                </PopoverContent>
              </Popover>
            ) : null}
            {showThinkingSwitch ? (
              <SwitchRow
                label="Thinking"
                hint={
                  reasoning.lockedOn
                    ? "Always on"
                    : reasoning.disabled
                      ? "Load a model first"
                      : undefined
                }
                checked={reasoning.enabled}
                disabled={reasoning.lockedOn || reasoning.disabled}
                onChange={reasoning.setEnabled}
              />
            ) : null}
            {showPreserve ? (
              <SwitchRow
                label="Preserve thinking"
                checked={reasoning.preserve}
                disabled={reasoning.disabled}
                onChange={reasoning.setPreserve}
              />
            ) : null}
            {showFastSwitch ? (
              <SwitchRow
                label="Fast mode"
                checked={reasoning.fast}
                onChange={reasoning.setFast}
              />
            ) : null}
            {showEffort ? (
              <div className="px-3 pt-0.5 pb-1.5">
                <EffortSlider
                  title="Effort"
                  accessory={
                    reasoning.supportsFast ? (
                      <Hint label="Fast mode">
                        <button
                          type="button"
                          role="switch"
                          aria-checked={reasoning.fast}
                          aria-label="Fast mode"
                          onClick={() => reasoning.setFast(!reasoning.fast)}
                          className={cn(
                            "grid size-[calc(32px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full transition-colors hover:bg-sidebar-accent",
                            reasoning.fast
                              ? "text-primary"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          <HugeiconsIcon
                            icon={FlashIcon}
                            strokeWidth={1.75}
                            className={cn(
                              "size-4",
                              reasoning.fast && "[&_path]:fill-current",
                            )}
                          />
                        </button>
                      </Hint>
                    ) : undefined
                  }
                  levels={reasoning.levels}
                  value={reasoning.value}
                  onChange={reasoning.setValue}
                  disabled={reasoning.disabled}
                />
              </div>
            ) : null}
          </div>
        ) : null}
      </PopoverContent>
      {connectedSettings && pageActive
        ? renderConnectedSettings(connectedSettings.row, connectedSettings.view)
        : null}
      <TransportConflictDialog
        conflict={updateConflict}
        onCancel={() => {
          if (updateConflictKey) {
            downloadManager.cancelConflict(updateConflictKey);
          }
          setUpdateConflictKey(null);
        }}
        onKeepTransport={() => {
          if (updateConflictKey) {
            downloadManager.resumeConflict(updateConflictKey);
          }
          setUpdateConflictKey(null);
        }}
        onSwitchTransport={() => {
          if (updateConflictKey) {
            downloadManager.restartConflict(updateConflictKey);
          }
          setUpdateConflictKey(null);
        }}
      />
    </Popover>
  );
}
