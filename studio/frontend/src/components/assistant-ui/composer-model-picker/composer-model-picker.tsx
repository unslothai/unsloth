// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantDetail,
} from "@/features/chat";
import { ApiProviderLogo } from "@/features/chat/api-provider-logo";
import { isExternalModelId } from "@/features/chat/external-providers";
import { DROP_CUE_CLASS } from "@/features/chat/hooks/use-sidebar-drag";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  OwnerAvatar,
  deleteCachedModel,
  formatBytes,
  hfApiToken,
  listGgufVariants,
  matchesRecommendedBrowse,
  resolveOwnerProviderLogo,
  useHfTokenStore,
  useHubInfiniteScroll,
  useHubModelSearch,
  useOnlineStatus,
} from "@/features/hub";
import {
  type LoraModelOption,
  ModelLoadSettingsAction,
  type ModelOption,
  ModelRowMenu,
  type ModelSelectorChangeMeta,
  type PickerFitStatus,
  VRAM_VERDICT,
  chatGgufFit,
  estimateQuantBytes,
  isMediaCatalogRepo,
  isOverBudget,
  loraOptionLabel,
  modelDisplayName,
  paramsFromId,
  pinDropAnchor,
  useChatPickerInventory,
  usePinnedConnectedModelsStore,
  usePinnedModelsStore,
  usePinnedRowDrag,
} from "@/features/model-picker";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useGpuInfo, useInferenceGpuInfo } from "@/hooks/use-gpu-info";
import { useVramBudgetFraction } from "@/hooks/use-vram-budget-fraction";
import {
  ChevronDownStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  CloudIcon,
  DashboardCircleIcon,
  Download01Icon,
  FlashIcon,
  InformationCircleIcon,
  LaptopIcon,
  Navigation05Icon,
  RemoveCircleIcon,
  Search01Icon,
  Settings02Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  Fragment,
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { useComposerModelPickerSource } from "./context";
import { EffortSlider } from "./effort-slider";
import { ListScrollbar } from "./list-scrollbar";
import {
  DEVICE_TAB,
  FAVORITES_TAB,
  type FormatTone,
  type PickerRow,
  RECOMMENDED_TAB,
  chooseQuant,
  dedupeRows,
  downloadedQuants,
  filterRows,
  findChosenRow,
  isChatTask,
  pinCoversId,
} from "./picker-model";
import { useComposerReasoning } from "./use-composer-reasoning";

interface Row extends PickerRow {
  loadedId?: string;
  makerName?: string;
  gguf?: CachedGgufRepo;
  cached?: CachedModelRepo;
  lora?: LoraModelOption;
  hub?: ModelOption;
  loadedQuant?: string | null;
}

const EXT_PREFIX = "ext:";
const MAKER_PREFIX = "maker:";
const EXPANDED_KEY = "unsloth_composer_picker_expanded";
const TAB_KEY = "unsloth_composer_picker_tab";

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
const MLX_ID = /(^|[-_/])mlx([-_/]|$)/i;
const LOOKS_GGUF = /gguf/i;
const isMlxId = (id: string) => MLX_ID.test(id);

function loraFormat(lora: LoraModelOption): FormatTone {
  if (
    lora.isDirectGguf ||
    lora.exportType === "gguf" ||
    LOOKS_GGUF.test(lora.id)
  ) {
    return "gguf";
  }
  if (lora.source !== "local" && lora.exportType !== "merged") {
    return "adapter";
  }
  return isMlxId(lora.id) ? "mlx" : "checkpoint";
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
    (lora.exportType === "gguf" || LOOKS_GGUF.test(`${lora.id} ${lora.name}`))
  );
}

function loraMeta(lora: LoraModelOption): ModelSelectorChangeMeta {
  const isLocal = lora.source === "local";
  return {
    source: isLocal
      ? "local"
      : lora.source === "exported"
        ? "exported"
        : "lora",
    isLora:
      !isLocal && lora.exportType !== "merged" && lora.exportType !== "gguf",
    isDownloaded: true,
    isGguf: lora.isDirectGguf === true,
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
): ModelSelectorChangeMeta {
  return {
    source: rowSource(row),
    isLora: false,
    loadId: onDisk ? (row.gguf?.load_id ?? undefined) : undefined,
    ggufVariant: variant.quant,
    ggufFilename: variant.filename,
    isDownloaded: onDisk,
    expectedBytes: quantBytes(variant),
    isGguf: true,
    pipelineTag: row.gguf?.task ?? null,
  };
}

interface QuantTarget {
  id: string;
  localPath: string | null;
  local: boolean;
  remote: boolean;
}

function quantTarget(row: Row): QuantTarget | null {
  if (row.kind === "gguf" && row.gguf) {
    return {
      id: row.id,
      localPath: row.gguf.load_id || row.gguf.cache_path || null,
      local: false,
      remote: false,
    };
  }
  if (row.kind === "localGguf") {
    return { id: row.id, localPath: null, local: true, remote: false };
  }
  if (row.kind === "hub" && row.hub?.isGguf) {
    return { id: row.id, localPath: null, local: false, remote: true };
  }
  return null;
}

async function rowQuants(target: QuantTarget): Promise<{
  quants: GgufVariantDetail[];
  defaultVariant: string | null;
}> {
  const res = await listGgufVariants(target.id, undefined, {
    localOnly: !target.remote,
    preferLocalCache: target.remote,
    localPath: target.localPath,
    includeCacheLocations: true,
  });
  const variants = (Array.isArray(res?.variants) ? res.variants : []).filter(
    (v) => v.partial !== true,
  );
  return {
    quants:
      target.remote || target.local ? variants : downloadedQuants(variants),
    defaultVariant:
      typeof res?.default_variant === "string" && res.default_variant
        ? res.default_variant
        : null,
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

function MakerMark({
  provider,
  className,
}: { provider: Maker; className: string }) {
  const base = import.meta.env.BASE_URL;
  const src =
    provider.treatment === "original" && provider.id === "google"
      ? `${base}provider-logos/google-g.svg`
      : `${base}${provider.logoPath.slice(1)}`;
  const transform = useLogoFit(src);
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
  row?: Pick<Row, "kind" | "providerType"> & {
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
    return (
      <OwnerAvatar
        owner={parts.owner}
        repoName={parts.name}
        size="xs"
        remote={false}
        className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 rounded-[3px]"
      />
    );
  }
  if (row?.kind === "external") {
    return row.providerType ? (
      <ApiProviderLogo providerType={row.providerType} className={size} />
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
}

function TabButton({
  tab,
  shown,
  onSelect,
}: {
  tab: SourceTab;
  shown: boolean;
  onSelect: () => void;
}) {
  return (
    <Hint label={tab.label}>
      <button
        type="button"
        role="tab"
        aria-selected={shown}
        aria-label={tab.label}
        onClick={onSelect}
        className={cn(
          "relative grid size-[calc(32px*var(--ui-space-scale,1))] shrink-0 place-items-center transition-colors hover:text-foreground",
          shown ? "text-foreground" : "text-muted-foreground",
        )}
      >
        {tab.icon}
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
}: {
  status: PickerFitStatus | null | undefined;
  revealOnHover?: boolean;
}) {
  const verdict = status ? VRAM_VERDICT[status] : null;
  if (!verdict) {
    return null;
  }
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <span
          role="img"
          aria-label={verdict.label}
          className={cn(
            "grid size-[calc(20px*var(--ui-space-scale,1))] shrink-0 place-items-center",
            verdict.tone,
            revealOnHover &&
              "opacity-0 transition-opacity group-hover/row:opacity-100 group-data-[active=true]/row:opacity-100",
          )}
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.8}
            className="size-[calc(14px*var(--ui-space-scale,1))]"
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
const ACTION_DOTS = `${ACTION} w-[calc(15px*var(--ui-space-scale,1))]`;

const FAVORITE_DROP_CUE_BASE =
  "before:pointer-events-none before:absolute before:inset-x-2 before:z-10 before:h-0 before:border-t-[1.5px] before:border-primary before:content-['']";
const FAVORITE_DROP_CUE = {
  top: `${FAVORITE_DROP_CUE_BASE} before:top-0`,
  bottom: `${FAVORITE_DROP_CUE_BASE} before:bottom-0`,
} as const;

const DIM_CLASS =
  "opacity-55 transition-opacity group-hover/row:opacity-100 dark:opacity-50";

function QuantPill({ quant }: { quant: string }) {
  return (
    <span className="inline-flex h-[calc(18px*var(--ui-space-scale,1))] max-w-[calc(100px*var(--ui-space-scale,1))] shrink-0 items-center truncate rounded-full bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] px-1.5 font-mono text-ui-9 text-muted-foreground dark:bg-muted">
      {quant}
    </span>
  );
}

/** The Studio tooltip, in place of the browser's own `title`. */
function Hint({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>{children}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

function LoadedDot({ label, onEject }: { label: string; onEject: () => void }) {
  return (
    <Hint label="Loaded. Click to eject">
      <button
        type="button"
        aria-label={`Eject ${label}`}
        onClick={(event) => {
          event.stopPropagation();
          onEject();
        }}
        className="group/eject -mx-0.5 grid size-[calc(18px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
      >
        <span className="col-start-1 row-start-1 size-[calc(6px*var(--ui-space-scale,1))] rounded-full bg-status-success transition-opacity group-hover/eject:opacity-0 group-focus-visible/eject:opacity-0" />
        <HugeiconsIcon
          icon={RemoveCircleIcon}
          strokeWidth={1.75}
          className="composer-eject-icon col-start-1 row-start-1 text-red-500 opacity-0 transition-opacity group-hover/eject:opacity-100 group-focus-visible/eject:opacity-100"
        />
      </button>
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

export function ComposerModelPicker({
  side = "top",
}: { side?: "top" | "bottom" }) {
  const source = useComposerModelPickerSource();
  const navigate = useNavigate();
  const reasoning = useComposerReasoning();
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const activeVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const loadedModels = useChatRuntimeStore((s) => s.loadedModels);
  const pinned = usePinnedModelsStore((s) => s.pinned);
  const togglePinned = usePinnedModelsStore((s) => s.togglePinned);
  const unpinRepo = usePinnedModelsStore((s) => s.unpinRepo);
  const pinnedConnected = usePinnedConnectedModelsStore((s) => s.pinned);
  const togglePinnedConnected = usePinnedConnectedModelsStore(
    (s) => s.togglePinnedConnected,
  );

  const [open, setOpen] = useState(false);
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
  const [foldedQuants, setFoldedQuants] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [offer, setOffer] = useState<{
    key: string;
    quants: GgufVariantDetail[] | null;
    defaultVariant: string | null;
    failed: boolean;
  } | null>(null);
  const [quants, setQuants] = useState<{
    key: string;
    list: GgufVariantDetail[];
  } | null>(null);
  const [quantsOpen, setQuantsOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const inventory = useChatPickerInventory({ enabled: open });
  const online = useOnlineStatus();
  const hfToken = useHfTokenStore((s) => s.token);
  const quantFormats = useGpuInfo().checkpointQuantFormats;
  const hubQuery = useDebouncedValue(onRecommended ? query.trim() : "");
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
  const inferenceGpu = useInferenceGpuInfo();
  const budgetFraction = useVramBudgetFraction() ?? undefined;
  const keepModelsLoaded = useChatRuntimeStore((s) => s.keepModelsLoaded);
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
  const hubFit = useMemo(() => {
    const map = new Map<string, ReturnType<typeof fitOf>>();
    for (const r of hubResults) {
      if (!r.isGguf) {
        continue;
      }
      const params = r.totalParams ?? paramsFromId(r.id);
      map.set(
        r.id.toLowerCase(),
        fitOf(
          r.estimatedSizeBytes ??
            (params ? estimateQuantBytes(params) : undefined),
        ),
      );
    }
    return map;
  }, [hubResults, fitOf]);
  const noGpu = !(inferenceGpu.memoryTotalGb > 0);
  const { scrollRef: hubScrollRef, sentinelRef: hubSentinelRef } =
    useHubInfiniteScroll(hubSearch.fetchMore, hubSearch.scannedCount, {
      enabled: open && online && onRecommended && hubSearch.hasMore,
      isFetching: hubSearch.isLoading || hubSearch.isLoadingMore,
      resultCount: hubResults.length,
      resetKey: hubQuery,
    });
  const recommended = useMemo<ModelOption[]>(() => {
    const fromHub = hubResults
      .filter(
        (r) =>
          matchesRecommendedBrowse(r, quantFormats, !hubQuery) &&
          isChatTask(r.pipelineTag) &&
          !isMediaCatalogRepo(r.id),
      )
      .map((r) => ({ id: r.id, name: r.id, isGguf: r.isGguf }));
    return fromHub.length > 0 ? fromHub : [...(source?.models ?? [])];
  }, [hubResults, quantFormats, hubQuery, source?.models]);
  const loraModels = source?.loraModels;
  const externalModels = source?.externalModels;

  const rows = useMemo<Row[]>(() => {
    const device: Row[] = [];
    for (const repo of inventory.cachedGguf) {
      if (repo.partial || !isChatTask(repo.task)) {
        continue;
      }
      device.push({
        key: `gguf:${repo.repo_id}`,
        id: repo.repo_id,
        label: modelDisplayName(repo.repo_id),
        detail: repo.size_bytes > 0 ? formatBytes(repo.size_bytes) : undefined,
        tab: DEVICE_TAB,
        kind: "gguf",
        format: "gguf",
        aliases: [repo.load_id ?? "", repo.cache_path],
        gguf: repo,
      });
    }
    for (const repo of inventory.cachedModels) {
      if (repo.partial || repo.companion || !isChatTask(repo.task)) {
        continue;
      }
      device.push({
        key: `cached:${repo.repo_id}`,
        id: repo.repo_id,
        label: modelDisplayName(repo.repo_id),
        detail: repo.size_bytes > 0 ? formatBytes(repo.size_bytes) : undefined,
        tab: DEVICE_TAB,
        kind: "cached",
        format: isMlxId(repo.repo_id) ? "mlx" : "checkpoint",
        aliases: [repo.load_id ?? "", repo.cache_path ?? ""],
        cached: repo,
      });
    }
    for (const lora of loraModels ?? []) {
      if (lora.audioType) {
        continue;
      }
      device.push({
        key: `local:${lora.id}`,
        id: lora.id,
        label: loraOptionLabel(lora),
        detail: loraDetail(lora),
        tab: DEVICE_TAB,
        kind: loraNeedsQuant(lora) ? "localGguf" : "local",
        format: loraFormat(lora),
        lora,
      });
    }
    const loaded = new Map(loadedModels.map((m) => [m.id.toLowerCase(), m]));
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
            loadedId: hit.id,
          }
        : row;
    });
    for (const m of loadedModels) {
      if (
        marked.some(
          (row) =>
            row.loaded && (row.id === m.id || row.aliases?.includes(m.id)),
        )
      ) {
        continue;
      }
      marked.push({
        key: `loaded:${m.id}`,
        id: m.id,
        label: modelDisplayName(m.id),
        tab: DEVICE_TAB,
        kind: "loaded",
        format: m.quant ? "gguf" : "checkpoint",
        aliases: m.checkpoint ? [m.checkpoint] : undefined,
        loaded: true,
        loadedQuant: m.quant ?? null,
        loadedId: m.id,
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
    const onDisk = new Set(marked.map((row) => row.id.toLowerCase()));
    const hub: Row[] = (recommended ?? [])
      .filter((m) => !isMediaCatalogRepo(m.id))
      .map((m) => ({
        key: `hub:${m.id}`,
        id: m.id,
        label: modelDisplayName(m.id),
        tab: RECOMMENDED_TAB,
        kind: "hub",
        format: m.isGguf ? "gguf" : isMlxId(m.id) ? "mlx" : "checkpoint",
        onDevice: onDisk.has(m.id.toLowerCase()),
        loaded: loadedIds.has(m.id.toLowerCase()),
        hub: m,
      }));
    const external: Row[] = (externalModels ?? []).map((m) => ({
      key: m.id,
      id: m.id,
      label: m.name,
      tab: `${EXT_PREFIX}${m.providerId}`,
      kind: "external",
      providerType: m.providerType,
      providerName: m.providerName,
    }));
    return [...marked, ...hub, ...external];
  }, [
    inventory.cachedGguf,
    inventory.cachedModels,
    loraModels,
    loadedModels,
    recommended,
    externalModels,
  ]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const wanted = rows.filter(
      (row) =>
        (row.kind === "gguf" || row.kind === "localGguf") &&
        !deviceQuants.has(row.key),
    );
    if (wanted.length === 0) {
      return;
    }
    let live = true;
    void Promise.all(
      wanted.map(async (row) => {
        const target = quantTarget(row);
        if (!target) {
          return [row.key, [] as GgufVariantDetail[]] as const;
        }
        try {
          return [row.key, (await rowQuants(target)).quants] as const;
        } catch {
          return [row.key, [] as GgufVariantDetail[]] as const;
        }
      }),
    ).then((found) => {
      if (!live) {
        return;
      }
      setDeviceQuants((prev) => {
        const next = new Map(prev);
        for (const [key, list] of found) {
          next.set(key, list);
        }
        return next;
      });
    });
    return () => {
      live = false;
    };
  }, [open, rows, deviceQuants]);

  const isFavorite = useCallback(
    (row: Row) =>
      row.kind === "external"
        ? pinnedConnected.includes(row.id)
        : pinCoversId(pinned, row.id),
    [pinned, pinnedConnected],
  );

  const makerTabs = useMemo<SourceTab[]>(() => {
    const out: SourceTab[] = [];
    const seen = new Set<string>();
    for (const row of rows) {
      const maker = row.groups ? rowMaker(row) : null;
      if (!maker || seen.has(maker.id)) {
        continue;
      }
      seen.add(maker.id);
      out.push({
        value: `${MAKER_PREFIX}${maker.id}`,
        label: maker.name,
        icon: <MakerMark provider={maker} className="size-4" />,
      });
    }
    return out;
  }, [rows]);

  const tabs = useMemo<SourceTab[]>(() => {
    const glyph = (icon: typeof StarIcon) => (
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
    );
    const out: SourceTab[] = [
      { value: FAVORITES_TAB, label: "Favorites", icon: glyph(StarIcon) },
      {
        value: RECOMMENDED_TAB,
        label: "Discover models",
        icon: glyph(Navigation05Icon),
      },
      {
        value: DEVICE_TAB,
        label: "On Device",
        // The tray glyph draws small in its box.
        icon: (
          <HugeiconsIcon
            icon={Download01Icon}
            strokeWidth={1.75}
            className="size-4 scale-[1.15]"
          />
        ),
      },
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
        icon: (
          <ApiProviderLogo providerType={m.providerType} className="size-4" />
        ),
      });
    }
    return out;
  }, [externalModels, makerTabs]);

  const chosen = findChosenRow(rows, checkpoint);
  // A pick names the chip at once; the checkpoint follows only once its load lands.
  const [pending, setPending] = useState<Row | null>(null);
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
    shown?.label ?? (checkpoint ? modelDisplayName(checkpoint) : null);
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
  const shownRows = useMemo(() => {
    const found = filterRows(rows, viewed, query, isFavorite);
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
  }, [rows, viewed, query, isFavorite, favoriteKeyOf, pinned, pinnedConnected]);
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

  const chosenTarget =
    chosen && chosen.kind !== "hub" ? quantTarget(chosen) : null;
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
    setTabState(null);
    setOffer(null);
    setQuery("");
    setQuantsOpen(false);
  }, []);

  const chosenLoaded = Boolean(
    chosen?.loaded || (!chosen && checkpoint && !isExternalModelId(checkpoint)),
  );
  const eject = (row: Row) => {
    source?.eject(row.loadedId ?? row.id);
  };

  const handOff = () => {
    setOpen(false);
    source?.openFullPicker();
  };

  const pickQuantRow = async (row: Row, target: QuantTarget) => {
    if (!source) {
      return;
    }
    const { quants: list, defaultVariant } = await rowQuants(target);
    const fetched = list.filter((v) => v.downloaded === true);
    const quant = chooseQuant(
      target.remote && fetched.length > 0 ? fetched : list,
      [row.loadedQuant, defaultVariant],
    );
    if (!quant) {
      handOff();
      return;
    }
    if (row.id !== chosen?.id) {
      setPending(row);
    }
    source.onPick(
      row.id,
      ggufMeta(row, quant, target.local || quant.downloaded === true),
    );
  };

  const toggleOffer = async (row: Row, target: QuantTarget) => {
    if (offer?.key === row.key) {
      setOffer(null);
      return;
    }
    setOffer({
      key: row.key,
      quants: null,
      defaultVariant: null,
      failed: false,
    });
    try {
      const { quants: list, defaultVariant } = await rowQuants(target);
      if (list.length === 1) {
        setOffer(null);
        pickOfferQuant(row, list[0]);
        return;
      }
      setOffer((now) =>
        now?.key === row.key
          ? { key: row.key, quants: list, defaultVariant, failed: false }
          : now,
      );
    } catch {
      setOffer((now) =>
        now?.key === row.key ? { ...now, quants: [], failed: true } : now,
      );
    }
  };

  const pickOfferQuant = (row: Row, variant: GgufVariantDetail) => {
    if (!source) {
      return;
    }
    if (row.id !== chosen?.id) {
      setPending(row);
    }
    source.onPick(row.id, ggufMeta(row, variant, variant.downloaded === true));
    resetView();
  };

  const pick = async (row: Row) => {
    if (!source) {
      return;
    }
    const target = quantTarget(row);
    if (target?.remote && !row.onDevice) {
      await toggleOffer(row, target);
      return;
    }
    if ((deviceQuants.get(row.key)?.length ?? 0) > 1) {
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
        await pickQuantRow(row, target);
      } catch {
        handOff();
        return;
      } finally {
        setResolving(null);
      }
      resetView();
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
      source.onPick(row.id, {
        source: "hub",
        isLora: false,
        loadId: row.cached?.load_id ?? undefined,
        isDownloaded: true,
        isGguf: false,
        pipelineTag: row.cached?.task ?? null,
      });
    } else if (row.kind === "loaded") {
      source.onPick(row.id, {
        source: "hub",
        isLora: false,
        isDownloaded: true,
        isGguf: Boolean(row.loadedQuant),
        ...(row.loadedQuant ? { ggufVariant: row.loadedQuant } : {}),
      });
    } else if (row.kind === "hub") {
      source.onPick(row.id, { source: "hub", isLora: false, isGguf: false });
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
    if (!target) {
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
      source.onPick(chosen.id, ggufMeta(chosen, variant, true));
    }
  };

  const onOpenChange = (next: boolean) => {
    setOpen(next);
    if (next) {
      resetView();
      setActive(0);
      setHighlight(false);
    }
  };

  useEffect(() => {
    if (!isExpanded) {
      return;
    }
    // Rows paging in under the pointer must not move the highlight.
    if (highlightRef.current) {
      return;
    }
    const at = shownRows.findIndex((row) => row.id === chosen?.id);
    setActive(at >= 0 ? at : 0);
  }, [isExpanded, shownRows, chosen?.id]);

  useEffect(() => {
    if (!isExpanded) {
      return;
    }
    listRef.current
      ?.querySelector<HTMLElement>(`[data-row-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [active, isExpanded]);

  const onSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (highlightRef.current) {
        setActive((i) =>
          event.key === "ArrowDown"
            ? Math.min(shownRows.length - 1, i + 1)
            : Math.max(0, i - 1),
        );
      }
      setHighlight(true);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const row = shownRows[active];
      if (row) {
        pick(row);
      }
    }
  };

  const expand = () => {
    setExpanded(true);
    requestAnimationFrame(() => searchRef.current?.focus());
  };

  const quantChoices =
    quants && chosen && quants.key === chosen.key ? quants.list : [];
  const showQuantRow = quantChoices.length > 1;
  const showEffort = reasoning.isEffort && reasoning.levels.length > 1;
  const showThinkingSwitch = reasoning.supportsReasoning && !showEffort;
  const hasFoot =
    showQuantRow ||
    showThinkingSwitch ||
    reasoning.supportsPreserve ||
    showEffort;

  const emptyCopy = (): { title: string; hint?: string } => {
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
        hint: "Download one from Recommended.",
      };
    }
    return { title: "No models" };
  };

  const chipLabel = chosenName ?? "Select model";

  const settingsMeta = async (
    row: Row,
  ): Promise<ModelSelectorChangeMeta | null> => {
    const target = quantTarget(row);
    if (target) {
      const { quants: list, defaultVariant } = await rowQuants(target);
      const quant = chooseQuant(list, [
        row.loadedQuant,
        row.id === chosen?.id ? activeVariant : null,
        defaultVariant,
      ]);
      return quant
        ? ggufMeta(row, quant, target.local || quant.downloaded === true)
        : null;
    }
    if (row.kind === "local" && row.lora) {
      return loraMeta(row.lora);
    }
    if (row.kind === "cached") {
      return {
        source: "hub",
        isLora: false,
        loadId: row.cached?.load_id ?? undefined,
        isDownloaded: true,
        isGguf: false,
        pipelineTag: row.cached?.task ?? null,
      };
    }
    if (row.kind === "loaded") {
      return {
        source: "hub",
        isLora: false,
        isDownloaded: true,
        isGguf: Boolean(row.loadedQuant),
        ...(row.loadedQuant ? { ggufVariant: row.loadedQuant } : {}),
      };
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
      handOff();
      return;
    }
    setOpen(false);
    source.openRunSettings(row.id, row.label, meta);
  };

  const deleteFor = (row: Row) => {
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
      disabled: row.loaded,
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
    const quiet = row.kind === "hub" && !row.onDevice;
    const offering = quiet && Boolean(row.hub?.isGguf);
    const fit = offering ? hubFit.get(row.id.toLowerCase()) : null;
    // No GPU: every load runs from RAM, so `ram` is no overage.
    const dimmed = isOverBudget(fit) && !(fit === "ram" && noGpu) && !isChosen;
    const offerOpen = offer?.key === row.key;
    const onDisk = deviceQuants.get(row.key) ?? [];
    const soleQuant = onDisk.length === 1 ? onDisk[0].quant : null;
    const manyQuants = onDisk.length > 1;
    const quantsOpen = manyQuants && !foldedQuants.has(row.key);
    // A GGUF is favorited by its quant, never the whole repo.
    const canStar = !(
      quiet ||
      manyQuants ||
      (row.kind === "hub" && Boolean(row.hub?.isGguf))
    );
    const actionClass =
      "opacity-0 transition-opacity group-hover/row:opacity-100 group-data-[active=true]/row:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100";
    const rowEl = (
      <div
        key={row.key}
        // biome-ignore lint/a11y/useSemanticElements: rows carry their own buttons, which <option> cannot.
        role="option"
        aria-expanded={offering ? offerOpen : undefined}
        tabIndex={-1}
        aria-selected={isChosen}
        data-row-index={index}
        data-active={highlight && index === active ? "true" : undefined}
        onMouseMove={() => {
          setActive(index);
          setHighlight(true);
        }}
        onClick={() => pick(row)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            pick(row);
          }
        }}
        className={cn(
          "group/row flex h-[calc(32px*var(--ui-space-scale,1))] shrink-0 cursor-pointer items-center gap-2 rounded-full pr-[calc(6px*var(--ui-space-scale,1))] pl-3 text-ui-14 transition-colors",
          highlight && index === active && "bg-sidebar-accent",
        )}
      >
        {resolving === row.key ? (
          <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0" />
        ) : (
          <span className={cn("text-foreground", dimmed && DIM_CLASS)}>
            <RowMark row={row} />
          </span>
        )}
        <span
          className={cn(
            "flex min-w-0 flex-1 items-center gap-1.5",
            dimmed && DIM_CLASS,
          )}
        >
          <span className="truncate">{row.label}</span>
          {manyQuants ? (
            <HugeiconsIcon
              icon={ChevronRightStandardIcon}
              strokeWidth={1.75}
              className={cn(
                "size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground opacity-0 transition-[transform,opacity] duration-150 group-hover/row:opacity-100 group-data-[active=true]/row:opacity-100",
                quantsOpen && "rotate-90",
              )}
            />
          ) : null}
          {row.loaded ? (
            <LoadedDot label={row.label} onEject={() => eject(row)} />
          ) : null}
        </span>
        {soleQuant ? <QuantPill quant={soleQuant} /> : null}
        {isChosen && !row.loaded ? (
          <HugeiconsIcon
            icon={Tick02Icon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
        ) : null}
        <span
          role="toolbar"
          aria-label={`${row.label} actions`}
          className="flex shrink-0 items-center"
          // Menus portal out but bubble through React: keep their clicks off the row.
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          {offering ? (
            <FitBadge status={fit} revealOnHover={!isChosen} />
          ) : null}
          {row.kind !== "external" ? (
            <>
              <ModelRowMenu
                ariaLabel={`More options for ${row.label}`}
                cachePath={
                  row.gguf || row.cached ? { repoId: row.id } : undefined
                }
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
                  ...(row.loaded
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
                        icon={Settings02Icon}
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
                buttonClassName={cn(ACTION_DOTS, actionClass)}
                iconClassName="composer-row-dots"
              />
              <ModelLoadSettingsAction
                ariaLabel={`Run settings for ${row.label}`}
                tooltip="Load settings"
                onConfigure={() => openSettings(row)}
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
                  favorite ? "text-foreground" : actionClass,
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
    );
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
          {quantsOpen ? renderDeviceQuants(row, onDisk) : null}
        </div>
      );
    }
    if (!(offerOpen || quantsOpen)) {
      return rowEl;
    }
    return (
      <Fragment key={row.key}>
        {rowEl}
        {offerOpen ? renderOffer(row) : renderDeviceQuants(row, onDisk)}
      </Fragment>
    );
  };

  const renderQuantActions = (
    row: Row,
    variant: GgufVariantDetail,
    onDisk: boolean,
  ) => {
    const key = `${row.id}::${variant.quant}`;
    const starred = pinned.includes(key);
    const button = ACTION_ROUND;
    const reveal =
      "opacity-0 transition-opacity group-hover/row:opacity-100 focus-visible:opacity-100";
    return (
      <span className="flex shrink-0 items-center">
        <ModelLoadSettingsAction
          ariaLabel={`Run settings for ${row.label} ${variant.quant}`}
          tooltip="Load settings"
          onConfigure={() => {
            if (!source) {
              return;
            }
            setOpen(false);
            source.openRunSettings(
              row.id,
              row.label,
              ggufMeta(row, variant, onDisk),
            );
          }}
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
              className={cn(button, starred ? "text-foreground" : reveal)}
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
        ) : null}
      </span>
    );
  };

  const renderDeviceQuants = (row: Row, list: GgufVariantDetail[]) => {
    const loadedQuant = (
      row.loadedQuant ?? (row.id === chosen?.id ? activeVariant : null)
    )?.toLowerCase();
    return (
      <div className="flex shrink-0 flex-col gap-0.5">
        {list.map((variant) => {
          const inMemory =
            Boolean(row.loaded) && variant.quant.toLowerCase() === loadedQuant;
          return (
            <div
              key={variant.quant}
              className="group/row flex h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center rounded-full pr-[calc(6px*var(--ui-space-scale,1))] transition-colors hover:bg-sidebar-accent"
            >
              <button
                type="button"
                onClick={() => {
                  if (!source) {
                    return;
                  }
                  if (row.id !== chosen?.id) {
                    setPending(row);
                  }
                  source.onPick(row.id, ggufMeta(row, variant, true));
                  resetView();
                }}
                className="flex h-full min-w-0 flex-1 items-center gap-2 pr-2 pl-[calc(40px*var(--ui-space-scale,1))] text-left"
              >
                <QuantPill quant={variant.quant} />
                {inMemory ? (
                  <span className="size-[calc(6px*var(--ui-space-scale,1))] shrink-0 rounded-full bg-status-success" />
                ) : null}
                <span className="flex-1" />
                <span className="shrink-0 text-ui-12 tabular-nums text-muted-foreground">
                  {formatBytes(quantBytes(variant))}
                </span>
              </button>
              {renderQuantActions(row, variant, true)}
            </div>
          );
        })}
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
      return (
        <div className="flex h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center pl-[calc(40px*var(--ui-space-scale,1))] text-ui-13 text-muted-foreground">
          Couldn't list quantizations
        </div>
      );
    }
    const recommendedQuant = offer.defaultVariant?.toLowerCase();
    return (
      <div className="flex shrink-0 flex-col gap-0.5">
        {offer.quants.map((variant) => {
          const variantFit = fitOf(quantBytes(variant));
          const variantDimmed =
            isOverBudget(variantFit) && !(variantFit === "ram" && noGpu);
          return (
            <div
              key={variant.quant}
              className="group/row flex h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center rounded-full pr-[calc(6px*var(--ui-space-scale,1))] transition-colors hover:bg-sidebar-accent"
            >
              <button
                type="button"
                onClick={() => pickOfferQuant(row, variant)}
                className="flex h-full min-w-0 flex-1 items-center gap-2 pr-2 pl-[calc(40px*var(--ui-space-scale,1))] text-left text-ui-13"
              >
                <span
                  className={cn(
                    "flex min-w-0 items-center gap-2",
                    variantDimmed && DIM_CLASS,
                  )}
                >
                  <QuantPill quant={variant.quant} />
                  {variant.quant.toLowerCase() === recommendedQuant ? (
                    <span className="shrink-0 text-ui-11 text-muted-foreground">
                      Recommended
                    </span>
                  ) : null}
                </span>
                <span className="flex-1" />
                {variant.downloaded ? (
                  <HugeiconsIcon
                    icon={Tick02Icon}
                    strokeWidth={2}
                    className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
                  />
                ) : null}
                <FitBadge status={variantFit} />
                <span
                  className={cn(
                    "shrink-0 text-ui-12 tabular-nums text-muted-foreground",
                    variantDimmed && DIM_CLASS,
                  )}
                >
                  {formatBytes(quantBytes(variant))}
                </span>
              </button>
              {renderQuantActions(row, variant, variant.downloaded === true)}
            </div>
          );
        })}
      </div>
    );
  };

  const renderList = () => {
    if (shownRows.length === 0) {
      const copy = emptyCopy();
      return (
        <div className="flex flex-col items-center gap-1 px-4 py-8 text-center">
          <span className="text-ui-14 font-medium text-foreground">
            {copy.title}
          </span>
          {copy.hint ? (
            <span className="text-ui-12 text-muted-foreground">
              {copy.hint}
            </span>
          ) : null}
        </div>
      );
    }
    return shownRows.map(renderRow);
  };

  const viewedTab = tabs.find((t) => t.value === viewed);
  const connectionId = viewed.startsWith(EXT_PREFIX)
    ? viewed.slice(EXT_PREFIX.length)
    : null;

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          className={cn(
            "unsloth-thinking-pill composer-model-chip group/chip",
            !(chosenName || modelLoading || pending) &&
              "composer-model-chip-empty",
          )}
          data-pill-label={chipLabel}
          aria-label={`Model: ${chipLabel}${reasoning.chipLabel ? `, ${reasoning.chipLabel}` : ""}`}
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
                    source?.eject(checkpoint);
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
                  className="invisible col-start-1 row-start-1 size-[calc(14px*var(--ui-space-scale,1))] text-red-500 group-hover/chip:visible"
                />
              </span>
            </Hint>
          ) : chosenName ? (
            <RowMark row={chosenMark} />
          ) : null}
          <span className="unsloth-thinking-label composer-model-chip-name truncate">
            {chipLabel}
          </span>
          {chosenName || modelLoading || pending ? null : (
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
            />
          )}
          {reasoning.chipLabel ? (
            <span className="unsloth-thinking-label composer-model-chip-detail shrink-0">
              {reasoning.chipLabel}
            </span>
          ) : null}
        </button>
      </PopoverTrigger>
      <PopoverContent
        side={side}
        align="end"
        sideOffset={8}
        data-composer-model-picker=""
        data-list-last={isExpanded && !hasFoot ? "" : undefined}
        className="unsloth-plus-menu composer-model-picker relative w-[calc(404px*var(--ui-space-scale,1))] gap-0 overflow-hidden p-2"
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
        {isExpanded ? (
          <>
            <div className="flex shrink-0 items-center gap-1">
              <div
                role="tablist"
                aria-label="Model sources"
                className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none]"
              >
                {tabs.map((t) => (
                  <TabButton
                    key={t.value}
                    tab={t}
                    shown={viewed === t.value}
                    onSelect={() => setTab(t.value)}
                  />
                ))}
              </div>
              {chosenName ? (
                <Hint label="Collapse">
                  <button
                    type="button"
                    aria-label="Collapse the model list"
                    aria-expanded={true}
                    onClick={() => setExpanded(false)}
                    className="ml-auto grid size-[calc(32px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={ChevronDownStandardIcon}
                      strokeWidth={1.5}
                      className="size-4 rotate-180"
                    />
                  </button>
                </Hint>
              ) : null}
              {/* Outside the scrolling tabs, so it holds the row's right end. */}
              {chosenLoaded && !pending && !modelLoading ? (
                <Hint label="Eject model">
                  <button
                    type="button"
                    aria-label={`Eject ${chosenName ?? "model"}`}
                    onClick={() =>
                      chosen
                        ? eject(chosen)
                        : source?.eject(checkpoint ?? undefined)
                    }
                    className="mr-[calc(8px*var(--ui-space-scale,1))] grid size-[calc(24px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-red-500 transition-colors hover:bg-sidebar-accent"
                  >
                    <HugeiconsIcon
                      icon={RemoveCircleIcon}
                      strokeWidth={1.75}
                      className="size-4"
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
              <input
                ref={searchRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={onSearchKeyDown}
                placeholder={
                  viewed === RECOMMENDED_TAB
                    ? "Search models to download"
                    : viewed === FAVORITES_TAB
                      ? "Search favorited models"
                      : viewed === DEVICE_TAB
                        ? "Search on device models"
                        : `Search ${viewedTab?.label ?? ""} models`
                }
                aria-label="Search models"
                aria-controls="composer-model-picker-list"
                className="min-w-0 flex-1 bg-transparent text-ui-14 outline-none placeholder:text-muted-foreground"
              />
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
              ) : source ? (
                <Hint label="Model hub">
                  <button
                    type="button"
                    aria-label="Open the Model hub"
                    onClick={() => {
                      setOpen(false);
                      void navigate({
                        to: "/hub",
                        search: {
                          tab:
                            viewed === RECOMMENDED_TAB
                              ? "discover"
                              : "downloaded",
                        },
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
              ref={(node) => {
                listRef.current = node;
                hubScrollRef.current = node;
              }}
              id="composer-model-picker-list"
              onMouseLeave={() => setHighlight(false)}
              // biome-ignore lint/a11y/useSemanticElements: a <select> cannot hold each row's buttons.
              role="listbox"
              tabIndex={-1}
              aria-label={viewedTab?.label}
              className="composer-model-list flex max-h-[min(calc(360px*var(--ui-space-scale,1)),45vh)] min-h-0 flex-col gap-0.5 overflow-y-auto pt-1.5 pb-1.5"
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
          <div className="flex h-[calc(34px*var(--ui-space-scale,1))] w-full shrink-0 items-center rounded-full bg-sidebar-accent pr-1">
            <button
              type="button"
              aria-expanded={false}
              aria-label={`${chosenName}: show all models`}
              onClick={expand}
              className="flex min-w-0 flex-1 items-center gap-2.5 self-stretch rounded-l-full pl-3 text-left text-ui-14"
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
              <Hint label="Eject">
                <button
                  type="button"
                  aria-label={`Eject ${chosenName}`}
                  onClick={() =>
                    chosen
                      ? eject(chosen)
                      : source?.eject(checkpoint ?? undefined)
                  }
                  className="flex h-[calc(28px*var(--ui-space-scale,1))] shrink-0 items-center gap-1.5 rounded-full px-2.5 text-ui-12 text-muted-foreground transition-colors hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
                >
                  <HugeiconsIcon
                    icon={RemoveCircleIcon}
                    strokeWidth={1.75}
                    className="composer-row-icon"
                  />
                  Eject
                </button>
              </Hint>
            ) : null}
            <button
              type="button"
              tabIndex={-1}
              aria-hidden={true}
              onClick={expand}
              className="grid size-[calc(28px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground"
            >
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={1.5}
                className="size-4"
              />
            </button>
          </div>
        )}
        {hasFoot ? (
          <div className="flex shrink-0 flex-col gap-0.5 border-t border-border pt-1">
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
                      className="flex h-[calc(32px*var(--ui-space-scale,1))] items-center gap-2 rounded-full px-3 text-left text-ui-14 transition-colors hover:bg-sidebar-accent"
                    >
                      <span className="flex-1 truncate">{variant.quant}</span>
                      <span className="text-ui-12 tabular-nums text-muted-foreground">
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
                hint={reasoning.lockedOn ? "Always on" : undefined}
                checked={reasoning.enabled}
                disabled={reasoning.lockedOn || reasoning.disabled}
                onChange={reasoning.setEnabled}
              />
            ) : null}
            {reasoning.supportsPreserve ? (
              <SwitchRow
                label="Preserve thinking"
                checked={reasoning.preserve}
                disabled={reasoning.disabled}
                onChange={reasoning.setPreserve}
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
    </Popover>
  );
}
