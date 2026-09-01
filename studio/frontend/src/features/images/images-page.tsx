// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowLeftRightIcon,
  ArrowUpDownIcon,
  ArrowReloadHorizontalIcon,
  Delete02Icon,
  Download01Icon,
  FlimSlateIcon,
  Image03Icon,
  ImageAdd02Icon,
  InformationCircleIcon,
  PinIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";

import { ImageDropzone } from "@/components/image-dropzone";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { useSidebar } from "@/components/ui/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InfoHint } from "@/components/ui/info-hint";
import { useDiffusionGpuChoices } from "@/hooks/use-gpu-info";
import { usePersistedChoice } from "@/hooks/use-persisted-choice";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { IMAGE_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type { HostClass } from "@/features/model-picker/components/model-selector/host-artifact-policy";
import {
  IMAGE_CATALOG,
  catalogToModelOptions,
  loadSpecFor,
} from "@/features/model-picker/components/model-selector/model-catalog";
import { useHostClass } from "@/hooks/use-host-class";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { GalleryItemMenu } from "@/components/gallery-item-menu";
import { MediaPageLink } from "@/components/media-page-link";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  type NewRecordProbeBaseline,
  applyPin,
  fetchNextPage,
  fetchWhileStable,
  hasUnknownRecord,
  mergeGenerated,
  newRecordProbeBaseline,
  nextSelectedId,
  pinnedOrder,
  removeGalleryItem,
  restorePinOrder,
  serializeById,
  sortGalleryItems,
  subscribeGalleryChanged,
} from "@/lib/gallery-flags";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useImageWorkflowStore } from "./stores/image-workflow-store";
import { WORKFLOW_TABS } from "./workflows";
import { ParamSlider } from "@/features/chat";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import {
  type ImageGenerationPresetParams,
  MediaGenerationPresetControl,
  useMediaGenerationPresets,
} from "@/features/generation-presets";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { formatBytes, formatEta } from "@/features/hub/lib/format";
import { ChevronDown } from "lucide-react";
import { NegativePromptField } from "@/components/negative-prompt-field";
import { cn } from "@/lib/utils";
import { isTauri } from "@/lib/api-base";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import {
  downloadFile,
  downloadUrl,
  isDownloadCancelled,
} from "@/lib/native-files";
import { resolveDiffusionGgufFilename } from "@/lib/diffusion-gguf-filename";
import { createPickGuard, runGgufRepoPick } from "@/lib/diffusion-gguf-pick";
import { diffusionRoutePick } from "@/lib/diffusion-route-pick";
import {
  PRECISION_REFUSAL_TITLE,
  denseTextEncoderBuildLabel,
  denseTransformerBuildLabel,
  isNativeEngineStatus,
  formatResolvedValue,
  isPrecisionRefusal,
  memoryRecipeValue,
  resolvedBadge,
  resolvedSeedKey,
  resolvedSelectValue,
} from "@/lib/resolved-precision";
import {
  routedGgufFilename,
  routedGgufLabel,
} from "@/lib/diffusion-route-search";
import { toast } from "@/lib/toast";
import { subscribeModelEjected } from "@/lib/model-lifecycle-events";
import { DEFAULT_GEN, defaultsFor } from "./image-generation-defaults";

import {
  type ControlNetSpecInput,
  type DiffusionControlNetInfo,
  type DiffusionGenerateProgress,
  type DiffusionGenerateResponse,
  type DiffusionLoadProgress,
  type DiffusionLoadRequest,
  type DiffusionLoraInfo,
  type DiffusionStatus,
  type GalleryImage,
  type LoraSpecInput,
  GenerateResponseLostError,
  cancelDiffusionGeneration,
  deleteGalleryImage,
  fetchGalleryBlob,
  fetchGalleryObjectUrl,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  getGallery,
  getGenerateProgress,
  listDiffusionControlNets,
  listDiffusionLoras,
  setGalleryImageFlags,
  getDiffusionDownloadPlan,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";
import {
  shouldContinueGenerating,
  shouldReportGenerateError,
} from "./lib/generation-stop";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useStagedDownload } from "@/features/hub/download-manager";
import { DiffusionTrainPanel } from "./train/diffusion-train-panel";
import {
  TrainBaseSelector,
  type TrainFamilyOption,
} from "./train/train-base-selector";

// Curated models come from the shared catalog, one group per model with its artifacts as data and
// the load kind per artifact from loadSpecFor. Built per render, since a host that can only run
// the native engine is not offered pipeline rows.
function useImageModels(host: HostClass): ModelOption[] {
  return useMemo(() => catalogToModelOptions(IMAGE_CATALOG, host), [host]);
}

// Workflow tabs. `requires` is the backend workflow id the model must support; null = always.
// Each conditioned workflow names the images it consumed for the restore toast, since a
// recipe keeps the scalar settings but not the uploads. txt2img restores completely.
const CONDITIONED_WORKFLOW_INPUTS: Record<string, string> = {
  img2img: "the source image",
  inpaint: "the source image and mask",
  upscale: "the source image",
  edit: "the source image",
  reference: "the source and reference images",
  controlnet: "the control image",
};

// Common aspect ratios (landscape; Flip mirrors to portrait). Picking one locks W:H; the
// sliders set the size.
const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};
const ASPECT_OPTIONS = ["custom", ...Object.keys(ASPECT_RATIOS)];
// Names read faster than bare ratios; Flip covers the portrait side of each.
const ASPECT_LABELS: Record<string, string> = {
  "1:1": "Square",
  "3:2": "Photo",
  "4:3": "Landscape",
  "16:9": "Widescreen",
  "21:9": "Ultrawide",
};

// Friendly labels for ControlNet control types; unknown types fall back to a capitalized "(map)" label.
const CONTROL_TYPE_LABELS: Record<string, string> = {
  passthrough: "Passthrough (already a map)",
  canny: "Canny (trace edges)",
  depth: "Depth (map)",
  pose: "Pose (map)",
};

// Z-Image accepts 256-2048, in multiples of 16. Snap any value into range.
const MIN_DIM = 256;
const MAX_DIM = 2048;
// Convenient drag range for the Runs slider; the number box accepts higher typed values on purpose.
const RUNS_SLIDER_MAX = 128;
// Offered sizes; a locked ratio can derive one off-list, so the current value is added in.
const DIM_OPTIONS = [
  256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280,
  1408, 1536, 1664, 1792, 1920, 2048,
];

function snapDim(value: number): number {
  if (!Number.isFinite(value)) return 1024;
  return Math.min(MAX_DIM, Math.max(MIN_DIM, Math.round(value / 16) * 16));
}

/** Compact size control: type a value, or pick one of the usual sizes from the menu. */
function DimensionSelect({
  icon,
  label,
  value,
  open,
  onOpenChange,
  onChange,
}: {
  icon: IconSvgElement;
  label: string;
  value: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChange: (value: number) => void;
}) {
  // Typing is held in a draft so a half-entered number is not snapped mid-keystroke.
  const [draft, setDraft] = useState(String(value));
  const [lastValue, setLastValue] = useState(value);
  if (value !== lastValue) {
    setLastValue(value);
    setDraft(String(value));
  }
  const commit = () => {
    const typed = Number(draft);
    const next = snapDim(Number.isFinite(typed) && typed > 0 ? typed : value);
    setDraft(String(next));
    setLastValue(next);
    if (next !== value) onChange(next);
  };
  const pick = (n: number) => {
    setDraft(String(n));
    setLastValue(n);
    onChange(n);
  };
  return (
    <div className="flex h-9 flex-1 items-center gap-2 rounded-full border border-border bg-background px-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-white/[0.06] dark:focus-within:bg-white/[0.12]">
      <HugeiconsIcon
        icon={icon}
        strokeWidth={1.75}
        className="size-4 shrink-0 text-muted-foreground"
      />
      <input
        aria-label={label}
        inputMode="numeric"
        value={draft}
        onChange={(e) => setDraft(e.target.value.replace(/[^0-9]/g, ""))}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          }
        }}
        className="w-full min-w-0 bg-transparent text-sm tabular-nums outline-none"
      />
      <DropdownMenu open={open} onOpenChange={onOpenChange}>
        <DropdownMenuTrigger
          aria-label={`${label} presets`}
          className="-mr-1 shrink-0 cursor-pointer rounded-full p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ChevronDown className="size-4" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="max-h-72 overflow-y-auto">
          {DIM_OPTIONS.map((n) => (
            <DropdownMenuItem key={n} onSelect={() => pick(n)}>
              <span className="tabular-nums">{n}</span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

// Hidden until the row is hovered or focused. The ratio key is compared long:short, so it survives orientation.
function matchAspect(width: number, height: number): { key: string; portrait: boolean } {
  const target = Math.max(width, height) / Math.min(width, height);
  const found = Object.entries(ASPECT_RATIOS).find(
    ([, [a, b]]) => Math.abs(target - a / b) < 0.01,
  );
  return { key: found ? found[0] : "custom", portrait: height > width };
}

// Module cache of the backend-persisted gallery, so a tab switch re-renders instantly; object
// URLs are revoked only on delete. The 192 MB blob budget never evicts a visible image.
// 192 MB is ~100-200 images, far more than a viewport holds.
const IMAGE_BLOB_BUDGET_BYTES = 192 * 1024 * 1024;

const galleryCache: {
  images: GalleryImage[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  srcById: BlobUrlCache;
  // Ids with a fetch in flight, so concurrent ensureSrc calls do not double-fetch and leak a duplicate object URL.
  inflight: Set<string>;
  // Ids deleted while their PNG was still downloading: a fetch landing afterwards must throw its blob away.
  deleted: Set<string>;
} = {
  images: [],
  hasMore: false,
  selectedId: null,
  quant: null,
  srcById: new BlobUrlCache(IMAGE_BLOB_BUDGET_BYTES),
  inflight: new Set(),
  deleted: new Set(),
};

// Images loaded per infinite-scroll page.
const PAGE_SIZE = 50;

// Passes a window resync may make before giving up: each extra pass only happens when
// pagination moved while it was fetching.
const RESYNC_MAX_ATTEMPTS = 3;

// Export filename. Batch siblings share seed + timestamp, so they get a "_<n>" suffix.
// For example Unsloth_20260624-143005_123.png.
type ImageExportFormat = "png" | "jpeg" | "webp";

function exportFilename(image: GalleryImage, format: ImageExportFormat = "png"): string {
  const d = new Date(image.created_at * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp =
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  const suffix = image.batch_index > 0 ? `_${image.batch_index}` : "";
  const ext = format === "jpeg" ? "jpg" : format;
  return `Unsloth_${stamp}_${image.seed}${suffix}.${ext}`;
}

// PNG saves the stored bytes verbatim, keeping the embedded recipe; JPEG / WebP re-encode
// client-side, JPEG flattened onto white.
async function reencodeImage(
  src: string,
  format: Exclude<ImageExportFormat, "png">,
): Promise<Blob> {
  const el = new Image();
  el.decoding = "async";
  el.src = src;
  await el.decode();
  const canvas = document.createElement("canvas");
  canvas.width = el.naturalWidth;
  canvas.height = el.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("canvas 2d context unavailable");
  }
  if (format === "jpeg") {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  ctx.drawImage(el, 0, 0);
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, `image/${format}`, 0.95),
  );
  if (!blob) {
    throw new Error(`could not encode ${format}`);
  }
  if (blob.type !== `image/${format}`) {
    // WebKit can silently return PNG bytes when an encoder is unavailable, so treat that as a
    // failed conversion and use a matching .png name.
    throw new Error(`${format} encoding is unavailable`);
  }
  return blob;
}

async function downloadImage(
  src: string,
  image: GalleryImage,
  format: ImageExportFormat = "png",
) {
  let outputFormat = format;
  let outputBlob: Blob | null = null;

  if (format !== "png") {
    try {
      outputBlob = await reencodeImage(src, format);
    } catch {
      // Conversion failed, so preserve the original PNG instead.
      outputFormat = "png";
      outputBlob = null;
    }
  }

  const filename = exportFilename(image, outputFormat);
  try {
    if (outputBlob) {
      await downloadFile(outputBlob, filename, outputBlob.type);
    } else if (isTauri) {
      // WebKit can display the cached object URL but fail to fetch it again, so re-fetch the
      // authenticated original for the native save.
      const originalBlob = await fetchGalleryBlob(image.url);
      await downloadFile(originalBlob, filename, originalBlob.type);
    } else {
      await downloadUrl(src, filename);
    }
    if (isTauri) {
      toast.success("Image saved", { description: filename });
    }
  } catch (error) {
    if (isDownloadCancelled(error)) {
      return;
    }
    toast.error("Could not save image", {
      description: error instanceof Error ? error.message : undefined,
    });
  }
}

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

// Bar label for an in-flight generation: step count plus an ETA once known.
function genStepLabel(p: DiffusionGenerateProgress): string {
  // Text encoding happens before the first scheduler tick, so step 0 means "working, not denoising yet".
  if (p.step === 0) return "Preparing (text encoding + warmup)…";
  const base = `Step ${p.step}/${p.total_steps}`;
  const eta = p.eta_seconds != null ? formatEta(p.eta_seconds) : "";
  return eta ? `${base} · ~${eta}` : base;
}

// Settling a generation whose POST response was lost: the backend keeps denoising, so poll until it goes idle.
const SETTLE_POLL_MS = 1000;
const SETTLE_MAX_MS = 6 * 60 * 60 * 1000; // hard cap; a native-CPU batch can run for hours
const SETTLE_MAX_FAILS = 5; // consecutive progress failures before calling the backend gone

/** Wait out a generation that outlived its POST. Idle progress alone is ambiguous, so success
 *  needs evidence: progress seen active, or a gallery record that is new since the POST.
 *  Throws past SETTLE_MAX_MS or if the backend stays unreachable, so a wedge surfaces. */
async function settleLostGeneration(
  isCurrent: () => boolean,
  baseline: NewRecordProbeBaseline,
): Promise<void> {
  const start = Date.now();
  let fails = 0;
  let sawActive = false;
  while (Date.now() - start < SETTLE_MAX_MS) {
    await new Promise((r) => setTimeout(r, SETTLE_POLL_MS));
    if (!isCurrent()) return;
    let idle = false;
    try {
      const p = await getGenerateProgress();
      fails = 0;
      if (p.active) sawActive = true;
      else idle = true;
    } catch {
      fails += 1;
      if (fails >= SETTLE_MAX_FAILS) throw new Error("Lost connection to the image server.");
    }
    if (!idle) continue;
    if (sawActive) return;
    // Idle on the very first look: the run may have finished or never started, so a gallery
    // record we had not seen is the proof.
    try {
      const sawNew = await hasUnknownRecord(
        baseline,
        async (offset) => {
          const p = await getGallery(offset, PAGE_SIZE);
          return { items: p.images, hasMore: p.has_more };
        },
        PAGE_SIZE,
      );
      if (sawNew) return;
    } catch {
      fails += 1;
      if (fails >= SETTLE_MAX_FAILS) throw new Error("Lost connection to the image server.");
      continue;
    }
    throw new Error("The image generation request did not reach the server.");
  }
  // Out of budget with the run still active: returning would report success and start the next
  // run against a busy backend.
  throw new Error("Timed out waiting for the image generation to finish.");
}

// The chat tab model-load toast styling, reused verbatim so the diffusion load toast is identical.
const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

// Render the chat ModelLoadDescription for a progress poll. The base repo downloads alongside
// the GGUF, so the total exceeds the quant size.
function loadToastDescription(p: DiffusionLoadProgress) {
  // "Downloading" only when bytes actually remain: a cached model must not claim a download.
  const downloading = p.bytes_total > 0 && p.bytes_downloaded < p.bytes_total * 0.999;
  const title = downloading
    ? "Downloading model requirements…"
    : p.phase === "finalizing"
      ? "Loading to GPU…"
      : "Starting model…";
  const hasTotal = p.bytes_total > 0;
  return (
    <ModelLoadDescription
      title={title}
      message={
        downloading
          ? "Downloading the files required to load this model."
          : "Loading the model."
      }
      progressPercent={hasTotal ? p.fraction * 100 : null}
      progressLabel={
        hasTotal
          ? `${formatBytes(p.bytes_downloaded)} of ${formatBytes(p.bytes_total)}`
          : p.bytes_downloaded > 0
            ? `${formatBytes(p.bytes_downloaded)} downloaded`
            : null
      }
    />
  );
}

// Toast args mirroring chat; `id` updates in place. `onCancel` adds chat's Cancel, the one
// control that reaches a load in flight: the selector's eject is hidden for exactly the
// span a first load runs, which left a multi-gigabyte pull with no way out.
function loadToastArgs(
  p: DiffusionLoadProgress,
  id?: string | number,
  onCancel?: () => void,
) {
  return {
    ...(id != null ? { id } : {}),
    description: loadToastDescription(p),
    duration: Infinity,
    closeButton: true,
    ...(onCancel ? { cancel: { label: "Cancel", onClick: onCancel } } : {}),
    classNames: LOAD_TOAST_CLASSNAMES,
  };
}

const IDLE_PROGRESS: DiffusionLoadProgress = {
  phase: null,
  bytes_downloaded: 0,
  bytes_total: 0,
  fraction: 0,
  error: null,
};

// One row: label, track, value. The Images sliders are Chat ParamSlider, so both pages share one control.
function SliderField({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <ParamSlider
      inline={true}
      label={label}
      info={hint}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={onChange}
    />
  );
}

// Matches the field-label style used across Unsloth.
function Field({
  label,
  hint,
  className,
  children,
}: {
  label: string;
  hint?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-center gap-1">
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
        {hint && <InfoHint>{hint}</InfoHint>}
      </div>
      {children}
    </div>
  );
}

// The badge for one Advanced control: "Auto: X" when the backend decided, "FP8 -> OFF" in a
// warning tone when an EXPLICIT request was declined. That case used to render nothing while the
// dropdown still showed the request, so a Q4_K_M GGUF could advertise FP8 it never ran.
function ResolvedBadge({
  status,
  controlKey,
}: {
  status: DiffusionStatus | null;
  controlKey: string;
}) {
  const info = resolvedBadge(controlKey, status?.resolved?.[controlKey]);
  if (!info) return null;
  const badge = (
    <span
      className={cn(
        "shrink-0 rounded-sm px-1 py-px text-ui-9 font-medium uppercase tracking-wider",
        info.tone === "warn"
          ? "bg-destructive/15 text-destructive"
          : "bg-muted text-muted-foreground",
      )}
    >
      {info.label}
    </span>
  );
  if (!info.tooltip) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{badge}</TooltipTrigger>
      <TooltipContent>{info.tooltip}</TooltipContent>
    </Tooltip>
  );
}

// A compact labeled Select row for the Advanced Options panel.
function AdvancedSelect({
  label,
  hint,
  badge,
  desc,
  value,
  onValueChange,
  options,
}: {
  label: string;
  hint?: ReactNode;
  // An optional inline badge next to the label (e.g. the "Auto: X" resolved-value pill).
  badge?: ReactNode;
  // A short always-visible description under the row.
  desc?: string;
  value: string;
  onValueChange: (v: string) => void;
  options: Array<[string, string]>;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <span className="flex shrink-0 items-center gap-1 whitespace-nowrap text-xs font-medium text-muted-foreground">
          {label}
          {hint && <InfoHint>{hint}</InfoHint>}
          {badge}
        </span>
        <Select value={value} onValueChange={onValueChange}>
          <SelectTrigger className="h-8 w-[160px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map(([v, l]) => (
              <SelectItem key={v} value={v} className="text-xs">
                {l}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {desc && <p className="text-ui-11 leading-snug text-muted-foreground/70">{desc}</p>}
    </div>
  );
}

// Brush-based mask editor for inpainting, exporting a grayscale PNG mask at NATIVE resolution
// (white = repaint). `brushPct` sizes the brush against the shorter side.
function MaskCanvas({
  image,
  brushPct,
  resetKey,
  onMaskChange,
}: {
  image: string;
  brushPct: number;
  resetKey: number;
  onMaskChange: (dataUrl: string | null) => void;
}) {
  const dispRef = useRef<HTMLCanvasElement | null>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const dims = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const [ready, setReady] = useState(false);

  // (Re)initialise both canvases when the image changes or Clear is pressed: size to native pixels, reset to all-black.
  useEffect(() => {
    setReady(false);
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      dims.current = { w, h };
      const disp = dispRef.current;
      const mask = maskRef.current ?? document.createElement("canvas");
      maskRef.current = mask;
      if (!disp) return;
      disp.width = w;
      disp.height = h;
      mask.width = w;
      mask.height = h;
      const mctx = mask.getContext("2d");
      const dctx = disp.getContext("2d");
      if (!mctx || !dctx) return;
      mctx.fillStyle = "#000";
      mctx.fillRect(0, 0, w, h);
      dctx.clearRect(0, 0, w, h);
      setReady(true);
      onMaskChange(null);
    };
    img.src = image;
  }, [image, resetKey, onMaskChange]);

  const radius = useCallback(() => {
    const base = Math.min(dims.current.w, dims.current.h) || 1024;
    return Math.max(2, (brushPct / 100) * base);
  }, [brushPct]);

  const toNatural = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const disp = dispRef.current;
    if (!disp) return { x: 0, y: 0 };
    const r = disp.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * dims.current.w,
      y: ((e.clientY - r.top) / r.height) * dims.current.h,
    };
  };

  const stroke = (from: { x: number; y: number } | null, to: { x: number; y: number }) => {
    const disp = dispRef.current;
    const mask = maskRef.current;
    if (!disp || !mask) return;
    const r = radius();
    const layers: Array<[CanvasRenderingContext2D | null, string]> = [
      [disp.getContext("2d"), "rgba(244,114,114,0.55)"],
      [mask.getContext("2d"), "#ffffff"],
    ];
    for (const [ctx, style] of layers) {
      if (!ctx) continue;
      ctx.strokeStyle = style;
      ctx.fillStyle = style;
      ctx.lineWidth = r * 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.arc(to.x, to.y, r, 0, Math.PI * 2);
      ctx.fill();
      if (from) {
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!ready) return;
    drawing.current = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // setPointerCapture can throw for synthetic events; safe to ignore.
    }
    const p = toNatural(e);
    last.current = p;
    stroke(null, p);
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const p = toNatural(e);
    stroke(last.current, p);
    last.current = p;
  };
  const onUp = () => {
    if (!drawing.current) return;
    drawing.current = false;
    last.current = null;
    const mask = maskRef.current;
    if (mask) onMaskChange(mask.toDataURL("image/png"));
  };

  return (
    <div className="relative overflow-hidden rounded-[10px] border border-border bg-muted/30">
      <img
        src={image}
        alt="Inpaint source"
        className="block w-full select-none"
        draggable={false}
      />
      <canvas
        ref={dispRef}
        data-testid="mask-canvas"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
      />
    </div>
  );
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Which sides to grow when outpainting.
type ExtendSides = { left: boolean; right: boolean; top: boolean; bottom: boolean };

// Redraw an image/canvas at (w, h). Clamps an outpaint source to a size the browser can back
// and the backend can decode.
function scaleToCanvas(source: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const dst = document.createElement("canvas");
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext("2d");
  if (!dctx) throw new Error("Could not scale the extended canvas");
  dctx.drawImage(source, 0, 0, w, h);
  return dst;
}

// Build the (image, mask) pair for outpaint on the inpaint backend: grow the canvas per
// dimension on the selected sides, edge-bleed the original in, and mask the new bands white.
async function buildOutpaint(
  src: string,
  sides: ExtendSides,
  pct: number,
): Promise<{ image: string; mask: string }> {
  const source = await loadImage(src);
  // Scale the SOURCE so the grown canvas fits MAX_SIDE before allocating: growing all four
  // sides by 100% multiplies the area by 9, and an oversized canvas no-ops every drawImage.
  const MAX_SIDE = 4096;
  const grow = (a: boolean, b: boolean) => 1 + (a ? pct / 100 : 0) + (b ? pct / 100 : 0);
  const fit = Math.min(
    1,
    MAX_SIDE /
      Math.max(
        source.naturalWidth * grow(sides.left, sides.right),
        source.naturalHeight * grow(sides.top, sides.bottom),
      ),
  );
  const w = fit < 1 ? Math.max(1, Math.floor(source.naturalWidth * fit)) : source.naturalWidth;
  const h = fit < 1 ? Math.max(1, Math.floor(source.naturalHeight * fit)) : source.naturalHeight;
  const img: CanvasImageSource = fit < 1 ? scaleToCanvas(source, w, h) : source;
  const px = Math.round((pct / 100) * w);
  const py = Math.round((pct / 100) * h);
  const l = sides.left ? px : 0;
  const r = sides.right ? px : 0;
  const t = sides.top ? py : 0;
  const b = sides.bottom ? py : 0;
  const nw = w + l + r;
  const nh = h + t + b;

  const ic = document.createElement("canvas");
  ic.width = nw;
  ic.height = nh;
  const ictx = ic.getContext("2d");
  if (!ictx) throw new Error("Could not build the extended canvas");
  ictx.drawImage(img, l, t, w, h); // original, centred by the chosen offsets
  // Edge-bleed: stretch the 1px border strips into each new band (and corners).
  if (l) ictx.drawImage(img, 0, 0, 1, h, 0, t, l, h);
  if (r) ictx.drawImage(img, w - 1, 0, 1, h, l + w, t, r, h);
  if (t) ictx.drawImage(img, 0, 0, w, 1, l, 0, w, t);
  if (b) ictx.drawImage(img, 0, h - 1, w, 1, l, t + h, w, b);
  if (l && t) ictx.drawImage(img, 0, 0, 1, 1, 0, 0, l, t);
  if (r && t) ictx.drawImage(img, w - 1, 0, 1, 1, l + w, 0, r, t);
  if (l && b) ictx.drawImage(img, 0, h - 1, 1, 1, 0, t + h, l, b);
  if (r && b) ictx.drawImage(img, w - 1, h - 1, 1, 1, l + w, t + h, r, b);

  const overlap = Math.round(Math.min(w, h) * 0.02);
  const ol = l ? overlap : 0;
  const or = r ? overlap : 0;
  const ot = t ? overlap : 0;
  const ob = b ? overlap : 0;
  const mc = document.createElement("canvas");
  mc.width = nw;
  mc.height = nh;
  const mctx = mc.getContext("2d");
  if (!mctx) throw new Error("Could not build the extend mask");
  mctx.fillStyle = "#ffffff"; // repaint everything...
  mctx.fillRect(0, 0, nw, nh);
  mctx.fillStyle = "#000000"; // ...except the kept original (inset by the seam overlap).
  mctx.fillRect(l + ol, t + ot, w - ol - or, h - ot - ob);

  // The pre-scale fits MAX_SIDE, but per-side rounding can overshoot the backend's 4096px limit; trim the slack.
  const longest = Math.max(nw, nh);
  if (longest > MAX_SIDE) {
    const scale = MAX_SIDE / longest;
    const sw = Math.max(1, Math.round(nw * scale));
    const sh = Math.max(1, Math.round(nh * scale));
    return {
      image: scaleToCanvas(ic, sw, sh).toDataURL("image/png"),
      mask: scaleToCanvas(mc, sw, sh).toDataURL("image/png"),
    };
  }

  return { image: ic.toDataURL("image/png"), mask: mc.toDataURL("image/png") };
}

// One labeled row in the recipe popover.
function RecipeRow({
  label,
  value,
  wrap,
  mono,
}: {
  label: string;
  value: string;
  wrap?: boolean;
  mono?: boolean;
}) {
  return (
    <div className={cn("grid grid-cols-[72px_1fr] gap-2", wrap ? "items-start" : "items-center")}>
      <span className="text-muted-foreground">{label}</span>
      <span
        className={cn(
          "min-w-0 text-foreground",
          wrap ? "whitespace-pre-wrap break-words" : "truncate",
          mono && "font-mono",
        )}
      >
        {value}
      </span>
    </div>
  );
}

// The full generation recipe for an image, with a one-click "restore to inputs".
function RecipePopover({
  image,
  onRestore,
  active,
}: {
  image: GalleryImage;
  onRestore: (image: GalleryImage) => void;
  active: boolean;
}) {
  // Controlled + force-closed off-tab: PopoverContent portals to body, so the inert page wrapper cannot contain it.
  const [open, setOpen] = useState(false);
  // Also clear the flag when leaving the tab so it does not reopen on return.
  useEffect(() => {
    if (!active) setOpen(false);
  }, [active]);
  return (
    <Popover open={active && open} onOpenChange={(o) => setOpen(active && o)}>
      <PopoverTrigger asChild>
        <Button size="sm" variant="ghost" className="gap-1.5">
          <HugeiconsIcon icon={InformationCircleIcon} className="size-4" />
          Recipe
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" side="top" className="w-80 p-0">
        <div className="border-b border-border/60 px-4 py-2.5">
          <p className="text-sm font-semibold">Generation settings</p>
          <p className="text-ui-11 text-muted-foreground">{formatTimestamp(image.created_at)}</p>
        </div>
        <div className="flex flex-col gap-2 px-4 py-3 text-xs">
          <RecipeRow label="Prompt" value={image.prompt} wrap />
          {image.negative_prompt ? (
            <RecipeRow label="Negative" value={image.negative_prompt} wrap />
          ) : null}
          {image.model ? <RecipeRow label="Model" value={image.model} /> : null}
          {/* The load-time build, so the recipe still names the pipeline once the model is unloaded: the
              repo id alone does not say which quant ran. */}
          {image.gguf_filename ? <RecipeRow label="File" value={image.gguf_filename} mono /> : null}
          {image.transformer_quant ? (
            <RecipeRow label="Quant" value={image.transformer_quant} />
          ) : null}
          {/* The ENGAGED text-encoder precision and memory placement: the encoder is often the largest
              resident component, and the memory mode decides whether torchao modes could run at all. */}
          {image.text_encoder_quant ? (
            <RecipeRow label="TE quant" value={image.text_encoder_quant} />
          ) : null}
          {/* Either field is placement information. The native engine reports no memory_mode while still
              recording an active offload, so gating on it hid the offload on exactly the configuration
              this row was extended for. Absent stays absent: "auto" would claim a planner that never ran. */}
          {image.memory_mode ||
          (image.offload_policy && image.offload_policy !== "none") ? (
            <RecipeRow
              label="Memory"
              value={memoryRecipeValue(image.memory_mode, image.offload_policy)}
            />
          ) : null}
          {image.baked_loras?.length ? (
            <RecipeRow label="Baked" value={image.baked_loras.join(", ")} wrap />
          ) : null}
          <RecipeRow label="Size" value={`${image.width} × ${image.height}`} />
          <RecipeRow label="Steps" value={String(image.steps)} />
          <RecipeRow label="Guidance" value={String(image.guidance)} />
          <RecipeRow label="Seed" value={String(image.seed)} mono />
        </div>
        <div className="border-t border-border/60 px-3 py-2.5">
          <Button size="sm" className="w-full gap-1.5" onClick={() => onRestore(image)}>
            <HugeiconsIcon icon={ArrowReloadHorizontalIcon} className="size-4" />
            Restore these settings
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// One "what actually ran" line in the loaded-build summary below.
function BuildRow({ label, value, badge }: { label: string; value: string; badge?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="flex shrink-0 items-center gap-1 whitespace-nowrap text-muted-foreground">
        {label}
        {badge}
      </span>
      <span className="min-w-0 truncate text-foreground">{value}</span>
    </div>
  );
}

/** What the LOADED model is actually running, read from status and never from the request:
 *  transformer and text-encoder precision, memory mode with its offload behaviour, and the
 *  attention backend. The Advanced selects say what was ASKED for; this says what happened. */
function LoadedBuildSummary({ status }: { status: DiffusionStatus | null }) {
  if (!status?.loaded) return null;
  const offload = status.offload_policy ?? "none";
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border/60 px-2.5 py-2 text-ui-11">
      <div className="flex items-center gap-1 pb-0.5 text-xs font-medium text-muted-foreground">
        Loaded build
        <InfoHint>
          What the loaded model is actually running, reported by the backend. A control whose
          requested value could not be used shows it next to that control, with the reason.
        </InfoHint>
      </div>
      <BuildRow
        label="Transformer"
        value={
          status.transformer_quant
            ? formatResolvedValue("transformer_quant", status.transformer_quant)
            // No dense quant ran, so the row reports what the checkpoint itself carries.
            : denseTransformerBuildLabel(status)
        }
        badge={<ResolvedBadge status={status} controlKey="transformer_quant" />}
      />
      <BuildRow
        label="Text encoder"
        value={
          status.text_encoder_quant
            ? formatResolvedValue("text_encoder_quant", status.text_encoder_quant)
            // No runtime TE quant engaged, which on the native engine is not the same as bf16.
            : denseTextEncoderBuildLabel(status)
        }
        badge={<ResolvedBadge status={status} controlKey="text_encoder_quant" />}
      />
      <BuildRow
        label="Memory"
        value={
          offload === "none"
            ? `${status.memory_mode ?? "auto"} · resident`
            : `${status.memory_mode ?? "auto"} · ${offload} offload`
        }
      />
      <BuildRow
        label="Attention"
        value={
          status.attention_backend
            ? formatResolvedValue("attention_backend", status.attention_backend)
            // The sd.cpp engine reports no backend because its attention comes from native flags
            // rather than the diffusers dispatcher, so "Native SDPA" is wrong on the CPU image path.
            :
              isNativeEngineStatus(status)
              ? "sd.cpp built-in"
              : "Native SDPA"
        }
      />
    </div>
  );
}

/** Report a failed load. A refused precision is a long actionable sentence, so it becomes a
 *  toast description under a short title. Nothing is left half-loaded either way. */
function reportLoadFailure(message: string | null | undefined, fallback: string): void {
  const text = (message || "").trim();
  if (text && isPrecisionRefusal(text)) {
    toast.error(PRECISION_REFUSAL_TITLE, { description: text });
    return;
  }
  toast.error(text || fallback);
}

type Busy = "loading" | "unloading" | "generating" | null;

// What a pick optimistically replaced, so a load that never takes can put it all back. The
// quant label and the recipe move together at pick time, so they roll back together.
type PickRevert = {
  prev: string | null;
  steps: number;
  guidance: number;
  commitRecipeClaim?: () => void;
  releaseRecipeClaim?: () => void;
  // What the pick applied. A field the user changed after that is theirs, not ours to put back.
  appliedSteps?: number;
  appliedGuidance?: number;
};

// The Advanced controls a load sends, with "auto" sentinels resolved to omitted. A staged
// download pins one at pick time.
type LoadAdvanced = Pick<
  DiffusionLoadRequest,
  | "cpu_offload"
  | "speed_mode"
  | "transformer_quant"
  | "attention_backend"
  | "memory_mode"
  | "transformer_cache"
  | "loras"
  | "gpu_ids"
>;

export function ImagesPage({
  active = true,
  onInitialReady,
}: {
  active?: boolean;
  onInitialReady?: () => void;
}) {
  const initialReadySent = useRef(false);
  const { isMobile, pinned } = useSidebar();
  const hostClass = useHostClass();
  const imageModels = useImageModels(hostClass);
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(
    "Cinematic wide shot of a whimsical Alice in Wonderland tea party in an overgrown Victorian garden. Exactly three figures at a long white lace-draped table: a tall eccentric gentleman in an oversized emerald velvet top hat pouring tea from a silver pot mid-motion; a young woman in a pale blue Victorian dress seated left, holding a porcelain teacup with both hands, looking up and laughing; an older woman in deep burgundy seated right in profile, reaching for a tiered cake stand. Detailed embroidered fabrics, realistic skin texture, natural expressions. The table holds mismatched porcelain, antique silverware, towering pastel cakes, and wildflowers. Giant red-capped mushrooms rise behind the table, with ancient trees overhead and golden sunlight streaming through leaves. Shot on 85mm, f/2.8, focus on the gentleman, soft background falloff. Photorealistic, saturated storybook color, warm amber and deep green palette.",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [negativeOpen, setNegativeOpen] = useState(false);
  const [widthOpen, setWidthOpen] = useState(false);
  const [heightOpen, setHeightOpen] = useState(false);
  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  // width/height are the source of truth; `aspect` locks their proportion and `portrait` tracks
  // orientation, so Flip keeps the lock.
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [aspect, setAspect] = useState("1:1");
  const [portrait, setPortrait] = useState(false);
  // Z-Image-Turbo official defaults: 9 steps (= 8 DiT forwards), guidance 0 (distilled, CFG-free).
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  // Whether the user has taken the recipe since the pick still waiting for its status: a preset
  // selected while the model downloaded is newer than that pick.
  const pickRecipeSuperseded = useRef<(() => boolean) | null>(null);
  // Put back everything a pick optimistically applied. Setters are stable, so this never re-renders on its own.
  const revertPick = useCallback((r: PickRevert) => {
    setQuant(r.prev);
    setPendingModelDefaults(null);
    // Equality alone cannot tell "nobody touched this" from "the user chose the same number": a
    // preset selected after the pick owns these fields.
    if (!pickRecipeSuperseded.current?.()) {
      setSteps((cur) => (cur === r.appliedSteps ? r.steps : cur));
      setGuidance((cur) => (cur === r.appliedGuidance ? r.guidance : cur));
    }
    pickRecipeSuperseded.current = null;
    r.releaseRecipeClaim?.();
    r.releaseRecipeClaim = undefined;
  }, []);
  // The recipe a pick optimistically claimed until status confirms it or a failed load reverts
  // it; without this the Default preset reads as "modified" for the whole download.
  const [pendingModelDefaults, setPendingModelDefaults] = useState<{
    steps: number;
    guidance: number;
  } | null>(null);
  const [seed, setSeed] = useState("");
  // Batch size = images per forward pass (VRAM-heavy); count = sequential loops.
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);
  // Active workflow tab: create = text-to-image, transform = img2img, inpaint = mask-guided
  // redraw. Workflow and page mode live in a store so the sidebar submenu can drive them.
  const workflow = useImageWorkflowStore((s) => s.workflow);
  const setWorkflow = useImageWorkflowStore((s) => s.setWorkflow);
  const supported = useImageWorkflowStore((s) => s.supported);
  const setSupported = useImageWorkflowStore((s) => s.setSupported);
  // Transform (img2img) / Inpaint inputs: the source image as a data URL, and the denoise strength.
  const [initImage, setInitImage] = useState<string | null>(null);
  const [strength, setStrength] = useState(0.6);
  // Inpaint mask (grayscale PNG data URL, white = repaint), brush size as a percent of the
  // shorter side, and a clear key.
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [brushPct, setBrushPct] = useState(8);
  const [maskResetKey, setMaskResetKey] = useState(0);
  // Extend (outpaint): how far to grow each dimension and which sides; reuses the inpaint backend at generate time.
  const [extendPct, setExtendPct] = useState(25);
  const [extendSides, setExtendSides] = useState<ExtendSides>({
    left: true,
    right: true,
    top: true,
    bottom: true,
  });
  // Upscale (hires fix): the enlargement factor and the low denoise strength that re-details the result.
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [upscaleStrength, setUpscaleStrength] = useState(0.35);
  // Reference (FLUX.2): up to 3 ADDITIONAL reference images beyond the primary one.
  const [referenceImages, setReferenceImages] = useState<string[]>([]);
  // LoRA adapters selected for the next generation (id + weight), plus the list the picker offers.
  const [loras, setLoras] = useState<LoraSpecInput[]>([]);
  const [availableLoras, setAvailableLoras] = useState<DiffusionLoraInfo[]>([]);
  // Page mode: "create" is the generation workspace, "train" the LoRA training workspace.
  const pageMode = useImageWorkflowStore((s) => s.pageMode);
  const setPageMode = useImageWorkflowStore((s) => s.setPageMode);
  // Train family + base live here so the top bar can pick them, replacing the generation model selector on Train.
  const [trainFamilies, setTrainFamilies] = useState<TrainFamilyOption[]>([]);
  const [trainFamilyName, setTrainFamilyName] = useState("flux.1");
  const [trainBaseChoice, setTrainBaseChoice] = useState("");
  // Bumped when a training run completes, so the LoRA discovery effect rescans without a model reload.
  const [loraRefreshKey, setLoraRefreshKey] = useState(0);
  // ControlNet for the next generation: model id, control image, how to derive the map, and the strength.
  const [controlnetId, setControlnetId] = useState<string>("");
  const [controlImage, setControlImage] = useState<string | null>(null);
  // Free-form: a union ControlNet advertises depth/pose alongside "canny", so the picker is
  // built from its own control_types.
  const [controlType, setControlType] = useState<string>("passthrough");
  const [controlStrength, setControlStrength] = useState(0.7);
  const [availableControlNets, setAvailableControlNets] = useState<DiffusionControlNetInfo[]>([]);
  // Advanced options live in a right-docked panel, closed by default; the open state is remembered across visits.
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_images_advanced_open",
  );
  // Advanced (load-time) options; "auto"/"off"/"none" map to the backend defaults. Changing one
  // while loaded shows "Reapply".
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [transformerQuant, setTransformerQuant] = useState<
    "none" | "auto" | "int8" | "fp8" | "nvfp4" | "mxfp8"
  >("auto");
  const [attentionBackend, setAttentionBackend] = useState<"auto" | "native" | "cudnn" | "flash3" | "sage">(
    "auto",
  );
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  // "auto", or the physical index to pin this load to; offered only on a multi-card CUDA/ROCm
  // host. Persisted, unlike the selects around it: status carries the device a pipeline is on
  // but not which card, so a refresh would reset it to Auto. A stale id is dropped on send.
  const [selectedGpu, setSelectedGpu] = usePersistedChoice(
    "unsloth_image_gpu_choice",
    "auto",
  );
  const gpuChoices = useDiffusionGpuChoices();
  const [transformerCache, setTransformerCache] = useState<"auto" | "off" | "fbcache">("auto");
  const [cpuOffload, setCpuOffload] = useState(false);
  // The last load descriptor, so "Reapply" can reload the same model with new advanced options without re-picking it.
  const lastLoad = useRef<{ repoId: string; kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null>(
    null,
  );
  // Render-safe mirror of whether a page-initiated load supplied a complete Reapply target.
  const [canReapply, setCanReapply] = useState(false);
  // Repo id whose defaults were already seeded from a discovered resident model, so we seed
  // once and never clobber a manual edit.
  const seededResident = useRef<string | null>(null);

  const [busy, setBusy] = useState<Busy>(null);
  // {done, total} while a multi-run generation is in flight (null = idle).
  const [genDone, setGenDone] = useState<number | null>(null);
  // Live per-step progress (step / total + ETA) polled during generation.
  const [genStep, setGenStep] = useState<DiffusionGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // visibilitychange handler active while a generation poll runs: background tabs clamp
  // setInterval, so returning fires one immediate poll.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  // Controlled so the body-portaled overlays force-close while this page is mounted but off-tab.
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [aspectOpen, setAspectOpen] = useState(false);
  // Records come from the backend (durable); srcById maps each id to its object URL or data URL.
  const [images, setImages] = useState<GalleryImage[]>(() => galleryCache.images);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    galleryCache.srcById.toRecord(),
  );
  // Guards a "load more" so a fast scroll cannot fire several at once.
  const loadingMore = useRef(false);
  // The gallery strip, used as the IntersectionObserver root so a tile PNG is fetched as it nears view.
  const stripRef = useRef<HTMLDivElement | null>(null);
  // Ids currently intersecting the strip; the blob cache never evicts these.
  const visibleIds = useRef<Set<string>>(new Set());
  // False once the page truly unmounts. Tab switches keep it mounted, so a batch keeps generating off-tab.
  const isMounted = useRef(true);
  // Set by Stop for one handleGenerate call: the backend cancel only reaches the denoise running
  // RIGHT NOW, so a count > 1 request would start its next run straight after.
  const cancelRequested = useRef(false);
  // True only once the backend answered {cancelled: true}. Anything else means the run was NOT
  // stopped, so an error it raises afterwards is a real failure.
  // Anything else -- a POST that never landed, or {cancelled: false} because the run was already past
  // its last cancellation check -- means it was NOT stopped.
  const cancelAcked = useRef(false);
  // Bumped once per handleGenerate call, so a cancel can tell its own run from a later one.
  const runToken = useRef(0);
  // The run token owning the Stop on the wire, or null: without it each extra click posts again
  // and a late duplicate stops whichever generation is running by then. A token, not a flag,
  // because the clear is asynchronous.
  const cancelInFlight = useRef<number | null>(null);
  // Aborts the Stop still on the wire: a pending cancel outlives its run while waiting on a 401
  // refresh-and-replay, and the replay would target whatever is generating by then.
  const cancelAbort = useRef<AbortController | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place.
  const loadToastId = useRef<string | number | null>(null);
  // Last load-progress signature shown, so a tick that moved nothing skips the toast.
  const lastLoadSig = useRef<string | null>(null);
  // The quant to restore if the optimistic swap fails; `{ prev }` distinguishes "revert to
  // null" from "nothing pending". A pick also applies its step/guidance recipe, so the
  // rollback carries that too, or a cancelled Turbo pick leaves a 4-step recipe behind.
  const quantRevert = useRef<PickRevert | null>(null);
  // Which quantRevert entry the live staged download belongs to: staging does not set `busy`, so
  // a second pick can overwrite quantRevert while the first plan resolves.
  const stagedQuantRevert = useRef<PickRevert | null>(null);
  // Bumped per Hub pick, so a plan that resolves after a newer pick can tell it has been superseded.
  const pickSeq = useRef(0);
  // The Reapply target to restore if the optimistic swap fails: handleLoad overwrites
  // lastLoad.current at load start. Mirrors quantRevert.
  const lastLoadRevert = useRef<{ prev: typeof lastLoad.current } | null>(null);
  // A trained adapter awaiting deployment: applied once the base is loaded and LoRA-capable for its family.
  const pendingDeploy = useRef<{ loraId: string; family: string } | null>(null);
  // Which pick owns the page: resolving and staging do not set `busy`, so a pick can land on an
  // awaiting one. Lazy state, not a ref, since a ref cannot be written during render.
  const [pickGuard] = useState(createPickGuard);

  const imagePresetParams = useMemo<ImageGenerationPresetParams>(
    () => ({
      negativePrompt,
      width,
      height,
      steps,
      guidance,
      batchSize,
      runs: count,
    }),
    [batchSize, count, guidance, height, negativePrompt, steps, width],
  );
  const imageDefaultRecipe = useMemo<ImageGenerationPresetParams>(() => {
    const recommended =
      pendingModelDefaults ??
      defaultsFor(status?.base_repo ?? status?.repo_id ?? "");
    return {
      negativePrompt: "",
      width: 1024,
      height: 1024,
      steps: recommended.steps,
      guidance: recommended.guidance,
      batchSize: 1,
      runs: 1,
    };
  }, [pendingModelDefaults, status?.base_repo, status?.repo_id]);
  const applyImagePresetParams = useCallback((params: ImageGenerationPresetParams) => {
    setNegativePrompt(params.negativePrompt);
    // Same rule restoreSettings follows: a negative prompt in effect has to be visible, or the
    // user generates against a setting the collapsed field is hiding.
    if (params.negativePrompt) setNegativeOpen(true);
    setWidth(params.width);
    setHeight(params.height);
    const matched = matchAspect(params.width, params.height);
    setAspect(matched.key);
    setPortrait(matched.portrait);
    setSteps(params.steps);
    setGuidance(params.guidance);
    setBatchSize(params.batchSize);
    setCount(params.runs);
    return params;
  }, []);
  const imagePresets = useMediaGenerationPresets({
    kind: "image",
    defaultParams: imageDefaultRecipe,
    currentParams: imagePresetParams,
    applyParams: applyImagePresetParams,
  });
  const claimImageRecipe = imagePresets.claimRecipe;
  const imageFormClaimId = imagePresets.formClaimId;
  const applyImageModelDefaults = useCallback(
    (repoId: string) => {
      const revert = quantRevert.current;
      if (revert && !revert.releaseRecipeClaim) {
        const claim = claimImageRecipe();
        revert.commitRecipeClaim = claim.commit;
        revert.releaseRecipeClaim = claim.release;
      }
      // Baselined per pick, including one that inherits an earlier pick's rollback: the question is
      // whether the user takes the form after THIS pick.
      const claimedAt = imageFormClaimId();
      pickRecipeSuperseded.current = () => imageFormClaimId() !== claimedAt;
      const recommended = defaultsFor(repoId);
      setPendingModelDefaults(recommended);
      setSteps(recommended.steps);
      setGuidance(recommended.guidance);
      if (revert) {
        revert.appliedSteps = recommended.steps;
        revert.appliedGuidance = recommended.guidance;
      }
    },
    [claimImageRecipe, imageFormClaimId],
  );

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);

  // The load toast is built by handleLoad and the progress poll, both defined above
  // handleCancelLoad, so the action goes through a ref to keep a stable onClick.
  const cancelLoadRef = useRef<() => void>(() => {});
  const cancelLoadFromToast = useCallback(() => cancelLoadRef.current(), []);
  // Bumped by every cancel / eject (see dropResidentState): requests already awaiting a response
  // compare against it and discard their own result.
  const cancelSeq = useRef(0);
  // Bumped by every load start. The compensating unload below carries no identity, so it must
  // not fire once a newer load owns the page.
  const loadSeq = useRef(0);
  // The load in flight, as a promise that settles only once handleLoad has run to the end,
  // compensating unload included. begin_load REFUSES a second load while one is live, so a
  // model picked in that window would be rejected while the cancelled one kept going.
  const pendingStart = useRef<Promise<unknown> | null>(null);

  // Set by restoreLoadTracking: handleLoad's compensating unload failed, so the load it was
  // cancelling is STILL running. handleUnload reads it to report that the eject did nothing.
  const loadTrackingRestored = useRef(false);

  // Client-side state that only means anything while a model is resident: the replacement
  // load's tracking and the Reapply target. Shared with the indicator eject.
  const dropResidentState = useCallback(() => {
    // Cancel, not release: a resolving pick or a staged download would load back what was just
    // ejected. Here rather than in handleUnload, so the loaded-models card is covered too.
    pickGuard.cancel();
    // Everything in flight is now stale. Clearing the timer stops the NEXT poll tick but not a
    // request awaiting its response, and those still apply terminal state; the counter is what
    // they compare against.
    cancelSeq.current += 1;
    // A cancelled deploy must not resurface: this ref outlives the load and would silently mix a
    // discarded adapter into an unrelated model's output.
    pendingDeploy.current = null;
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    // Leaving this set would let Reapply reload the model that was just freed.
    lastLoad.current = null;
    setCanReapply(false);
    // Stopping the poll also stops its "cancelled or evicted" branch, which is what hands back a
    // pick that never became resident, so do it here exactly as that branch would.
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [dismissLoadToast, pickGuard, revertPick]);

  // Mirror to the module cache so a tab switch re-renders instantly.
  useEffect(() => {
    galleryCache.images = images;
    galleryCache.hasMore = hasMore;
    galleryCache.selectedId = selectedId;
    galleryCache.quant = quant;
  }, [images, hasMore, selectedId, quant]);

  // Refresh the LoRA picker when the loaded family changes, since a LoRA is family-specific.
  // Not on first load or unload (a restore can precede the load), so track it in a ref.
  const loraCapable = Boolean(status?.loaded && status?.supports_lora);
  const prevLoraFamilyRef = useRef<string | null | undefined>(undefined);
  // Whether the load in flight baked the LoRA selection into the build (see handleLoad).
  const bakedLorasOnLoad = useRef(false);
  useEffect(() => {
    if (!loraCapable) {
      // Options are gone with the model, but keep the selection: it may have just been restored.
      setAvailableLoras([]);
      return;
    }
    const fam = status?.family ?? null;
    const prev = prevLoraFamilyRef.current;
    if (prev != null && prev !== fam) {
      setLoras([]);
    }
    prevLoraFamilyRef.current = fam;
    // A just-deployed adapter: apply it now that the base is loaded and LoRA-capable, if the family matches.
    const deploy = pendingDeploy.current;
    if (deploy) {
      pendingDeploy.current = null;
      if (!deploy.family || deploy.family === fam) {
        setLoras([{ id: deploy.loraId, weight: 1 }]);
      } else {
        toast.error(
          `The trained adapter is for ${deploy.family}, but the loaded model is ` +
            `${fam ?? "a different family"}, so it was not applied.`,
        );
      }
    }
    let cancelled = false;
    listDiffusionLoras(status?.family ?? undefined)
      .then((list) => {
        if (!cancelled) setAvailableLoras(list);
      })
      .catch(() => {
        // Clear only the OPTIONS on a failed catalog refresh: this free-text picker holds selections
        // that are valid without being in the catalog.
        if (!cancelled) setAvailableLoras([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loraCapable, status?.family, loraRefreshKey]);

  // A torchao int8/fp8 build takes adapters ONLY at load time, so drop the selection once per
  // resident build and say why, rather than 400 on the next Generate.
  const residentBuildKey = `${status?.repo_id ?? ""}|${String(
    status?.resolved?.transformer_quant?.value ?? "",
  )}`;
  const checkedBuildForBake = useRef<string | null>(null);
  useEffect(() => {
    if (!loraCapable || checkedBuildForBake.current === residentBuildKey) return;
    checkedBuildForBake.current = residentBuildKey;
    const engaged = status?.resolved?.transformer_quant?.value;
    if (engaged !== "int8" && engaged !== "fp8") return;
    if (bakedLorasOnLoad.current || loras.length === 0) return;
    setLoras([]);
    toast.info("LoRA selection cleared", {
      description:
        "This quantized load bakes adapters in at load time. Pick them, then load again.",
    });
  }, [loraCapable, residentBuildKey, status?.resolved, loras]);

  // Refresh the ControlNet options when the loaded family changes, and clear a stale selection.
  const controlnetCapable = Boolean(status?.loaded && status?.supports_controlnet);
  useEffect(() => {
    if (!controlnetCapable) {
      setAvailableControlNets([]);
      setControlnetId("");
      setControlImage(null);
      return;
    }
    let cancelled = false;
    listDiffusionControlNets(status?.family ?? undefined)
      .then((list) => {
        if (cancelled) return;
        setAvailableControlNets(list);
        setControlnetId((prev) => (list.some((c) => c.id === prev) ? prev : ""));
      })
      .catch(() => {
        if (!cancelled) setAvailableControlNets([]);
      });
    return () => {
      cancelled = true;
    };
  }, [controlnetCapable, status?.family]);

  // The control types offered for the selected ControlNet: a union model advertises several.
  const controlTypeOptions = useMemo(() => {
    const cn = availableControlNets.find((c) => c.id === controlnetId);
    const types = cn?.control_types?.length ? cn.control_types : ["passthrough", "canny"];
    return types;
  }, [availableControlNets, controlnetId]);

  // Keep controlType valid for the selected model: snap to the first (prefer passthrough) when the choice is gone.
  useEffect(() => {
    if (!controlTypeOptions.includes(controlType)) {
      setControlType(
        controlTypeOptions.includes("passthrough") ? "passthrough" : controlTypeOptions[0],
      );
    }
  }, [controlTypeOptions, controlType]);

  const selected = useMemo(
    () => images.find((i) => i.id === selectedId) ?? images[0] ?? null,
    [images, selectedId],
  );
  const selectedSrc = selected ? srcById[selected.id] : undefined;

  // Fetch (once) the object URL for a record's PNG; cached across remounts.
  const ensureSrc = useCallback(async (image: GalleryImage) => {
    if (galleryCache.srcById.has(image.id) || galleryCache.inflight.has(image.id)) return;
    galleryCache.inflight.add(image.id);
    try {
      const { url, bytes } = await fetchGalleryObjectUrl(image.url);
      if (galleryCache.deleted.has(image.id)) {
        URL.revokeObjectURL(url);
        return;
      }
      galleryCache.srcById.set(image.id, url, bytes);
      // Evict the coldest off-screen images this one pushed over budget; on-screen and open tiles are protected.
      const evicted = galleryCache.srcById.prune(
        new Set([image.id, ...visibleIds.current, galleryCache.selectedId ?? ""]),
      );
      setSrcById((prev) => {
        const next = { ...prev, [image.id]: url };
        for (const id of evicted) delete next[id];
        return next;
      });
    } catch {
      // Leave it without a src; the tile shows a placeholder.
    } finally {
      galleryCache.inflight.delete(image.id);
    }
  }, []);

  // Bumped by every LOCAL change to the strip. A resync started before one holds a snapshot the
  // listing cannot reconcile with what the user just did.
  const stripEpoch = useRef(0);
  // Bumped by the window growing from the server. Not a conflict: the resync merely sized itself
  // against a smaller window, so it refetches.
  const pageEpoch = useRef(0);
  // Only the newest resync may apply: two restores in a row would otherwise let the older snapshot land last.
  const resyncSeq = useRef(0);
  // Shelf mutations in flight. The epoch is an EDGE, so a page starting after the bump and
  // landing before the row is dropped sees it hold still.
  const pendingShelfMutations = useRef(0);

  const loadGallery = useCallback(async () => {
    try {
      // Fenced: this page renders from the module cache while the load runs, so its tiles are
      // actionable and a pre-pin snapshot would undo the action.
      const page = await fetchWhileStable(
        () => stripEpoch.current,
        () => getGallery(0, PAGE_SIZE),
      );
      if (!page) return;
      pageEpoch.current += 1;
      galleryCache.images = page.images;
      galleryCache.hasMore = page.has_more;
      setImages(page.images);
      setHasMore(page.has_more);
      // No visibility signal without IntersectionObserver (jsdom / old webview), so keep the eager fetch there.
      if (typeof IntersectionObserver === "undefined") {
        page.images.forEach((image) => void ensureSrc(image));
      }
    } catch {
      // Best-effort: a failed gallery load should not block the page.
    }
  }, [ensureSrc]);

  // Load the next older page. offset = how many are loaded so far; a new image sorts to the front on the backend too.
  const loadMore = useCallback(async () => {
    if (loadingMore.current || !galleryCache.hasMore) return;
    loadingMore.current = true;
    try {
      // Guarded on all three counters: an archive landing across this GET shortens the shelf, and
      // the record that shifts over the page boundary is returned by no page at all.
      const result = await fetchNextPage(
        () => galleryCache.images.length,
        () => stripEpoch.current,
        () => pendingShelfMutations.current,
        (offset) => getGallery(offset, PAGE_SIZE),
      );
      if (!result) return;
      const page = result.page;
      pageEpoch.current += 1;
      setImages((prev) => {
        const seen = new Set(prev.map((i) => i.id));
        const next = [...prev, ...page.images.filter((i) => !seen.has(i.id))];
        galleryCache.images = next;
        return next;
      });
      galleryCache.hasMore = page.has_more;
      setHasMore(page.has_more);
      if (typeof IntersectionObserver === "undefined") {
        page.images.forEach((image) => void ensureSrc(image));
      }
    } catch {
      // transient; the user can scroll again to retry
    } finally {
      loadingMore.current = false;
    }
  }, [ensureSrc]);

  // A gallery page holds PAGE_SIZE multi-megabyte PNGs and an object URL lives until the page
  // closes, so fetch a tile as it nears the strip edge instead. Mirrors Video.
  useEffect(() => {
    const root = stripRef.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.imageId;
          if (!id) continue;
          // Visibility is also the cache recency and protection signal: an on-screen tile is never evicted.
          if (!entry.isIntersecting) {
            visibleIds.current.delete(id);
            continue;
          }
          visibleIds.current.add(id);
          galleryCache.srcById.touch(id);
          const image = images.find((i) => i.id === id);
          if (image) void ensureSrc(image);
        }
      },
      // rootMargin applies to the ROOT box only, so the root must be the scrolling strip; the
      // sideways margin fetches a few tiles early.
      { root, rootMargin: "0px 600px" },
    );
    for (const tile of root.querySelectorAll("[data-image-id]")) io.observe(tile);
    return () => io.disconnect();
  }, [images, ensureSrc]);

  // The preview is what the user looks at, so the selected image is fetched whether or not its tile is on screen.
  useEffect(() => {
    if (!selected) return;
    void (async () => {
      await ensureSrc(selected);
    })();
  }, [selected, ensureSrc]);

  // Drop an image from the strip. `discardBlob` is for a real delete: the bytes are gone, so
  // the object URL is revoked and any in-flight fetch discards. An archived image keeps both.
  const dropFromStrip = useCallback((id: string, discardBlob: boolean) => {
    if (discardBlob) {
      galleryCache.srcById.delete(id); // revokes the URL with the entry
      galleryCache.deleted.add(id);
      setSrcById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
    visibleIds.current.delete(id);
    stripEpoch.current += 1;
    // Read the list from the cache rather than nesting a setSelectedId inside a setImages
    // updater, which would run a side effect during dispatch.
    const at = galleryCache.images.findIndex((i) => i.id === id);
    const next = removeGalleryItem(galleryCache.images, id);
    galleryCache.images = next;
    setImages(next);
    setSelectedId((cur) => nextSelectedId(next, id, cur, at));
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      // Held for the whole round trip: the server shortens the shelf when it processes this, and a
      // page read inside that window sees the shortened list at a consistent offset.
      stripEpoch.current += 1;
      pendingShelfMutations.current += 1;
      try {
        await deleteGalleryImage(id);
      } catch (err) {
        pendingShelfMutations.current -= 1;
        toast.error(err instanceof Error ? err.message : "Failed to delete image");
        return;
      }
      dropFromStrip(id, true);
      pendingShelfMutations.current -= 1;
    },
    [dropFromStrip],
  );

  /** Refetch the loaded window from offset 0. Unpinning can drop an image past the end of the
   *  window and promote an unloaded one into it, which the local reorder cannot know about. */
  const resyncWindow = useCallback(
    async (count: number, stillFresh?: () => boolean) => {
      const ticket = (resyncSeq.current += 1);
      for (let attempt = 0; attempt < RESYNC_MAX_ATTEMPTS; attempt += 1) {
        const paged = pageEpoch.current;
        // Sized against the live window, so a page appended while this ran is covered rather than cut
        // off the bottom of the strip.
        const wanted = Math.max(count, galleryCache.images.length, PAGE_SIZE);
        const collected: GalleryImage[] = [];
        let more = false;
        while (collected.length < wanted) {
          // The REMAINDER, not a whole page: a window of 51 would otherwise ask for 100 and read 49
          // recipes off disk for a one-row shortfall.
          const page = await getGallery(
            collected.length,
            Math.min(PAGE_SIZE, wanted - collected.length),
          );
          collected.push(...page.images);
          more = page.has_more;
          if (!page.has_more || page.images.length === 0) break;
        }
        // Checked here, not by the caller: by the time this returns the window is already applied, so
        // a stale snapshot has to be dropped first.
        if (stillFresh && !stillFresh()) return;
        if (resyncSeq.current !== ticket) return;
        // Pagination moved under this pass. That is only server data, so cover it with another pass
        // instead of giving up, which is what left an unpin's promoted image missing.
        if (pageEpoch.current !== paged) continue;
        galleryCache.images = collected;
        galleryCache.hasMore = more;
        setImages(collected);
        setHasMore(more);
        if (typeof IntersectionObserver === "undefined") {
          collected.forEach((image) => void ensureSrc(image));
        }
        return;
      }
    },
    [ensureSrc],
  );

  // This page stays mounted across route changes, so an archive restore would not reach the
  // strip until a reload. Resync the loaded window: loadGallery would cut it to page one.
  useEffect(
    () =>
      subscribeGalleryChanged("images", () => {
        // Bumped FIRST: a restore changes the shelf, so reads already in flight must be discarded, or
        // they pass their own checks and land on the new window.
        stripEpoch.current += 1;
        // Fenced like the unpin resync: a generation or a new page landing while this GET runs would
        // be overwritten by a snapshot taken before it.
        const epoch = stripEpoch.current;
        void resyncWindow(
          galleryCache.images.length,
          () => stripEpoch.current === epoch,
        ).catch(() => void loadGallery());
      }),
    [loadGallery, resyncWindow],
  );

  // The pin state each id was last CLICKED into, so a failing request can tell whether it is
  // still the current intent; without it a slow failure rolls back a later success.
  const pinAttempt = useRef(new Map<string, number>());
  const pinSeq = useRef(0);

  const handleTogglePin = useCallback(
    async (id: string, pinned: boolean) => {
      const loadedCount = galleryCache.images.length;
      // The pinned order BEFORE the click, so a failed unpin can put the image back where it was
      // instead of at the front.
      const orderBefore = pinnedOrder(galleryCache.images);
      // A per-attempt token, not the target boolean: pin, unpin, pin stores true twice, so the first
      // attempt's failure would roll back the third attempt's pin.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      const epoch = stripEpoch.current;
      // Optimistic: the reorder should land on the click, not a round trip later.
      setImages((prev) => {
        const next = applyPin(prev, id, pinned);
        galleryCache.images = next;
        return next;
      });
      try {
        // One queue for the whole gallery: the server stamps `pinned_at` when it runs the PATCH, so
        // two requests in flight can be stamped in either order. One at a time follows the clicks.
        await serializeById("image-pin", () => setGalleryImageFlags(id, { pinned }));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to pin image");
        // Put the old order back rather than leave the strip lying about server state, but only while
        // this is still what the user last asked for.
        if (pinAttempt.current.get(id) === attempt) {
          pinAttempt.current.delete(id);
          stripEpoch.current += 1;
          setImages((prev) => {
            // A failed pin goes back to unpinned; a failed unpin has to be restored to its old position
            // among the pins, which applyPin cannot do.
            const next = pinned
              ? applyPin(prev, id, false)
              : restorePinOrder(prev, id, orderBefore);
            galleryCache.images = next;
            return next;
          });
        }
        return;
      }
      if (pinAttempt.current.get(id) !== attempt) return; // superseded by a later click
      pinAttempt.current.delete(id);
      // Pinning keeps the same set in the window, so only unpinning can open a gap.
      if (!pinned && loadedCount > 0) {
        try {
          // Fenced: a pin clicked while this GET is in flight would be overwritten by a snapshot taken before it.
          await resyncWindow(loadedCount, () => stripEpoch.current === epoch);
        } catch {
          // Best-effort: the strip is still usable, just possibly short one image until a reload.
        }
      }
    },
    [resyncWindow],
  );

  const handleArchive = useCallback(
    async (id: string) => {
      // Held for the whole round trip: the server shortens the shelf when it processes this, so a
      // page read inside that window still sees a consistent offset.
      stripEpoch.current += 1;
      pendingShelfMutations.current += 1;
      try {
        await setGalleryImageFlags(id, { archived: true });
      } catch (err) {
        pendingShelfMutations.current -= 1;
        toast.error(err instanceof Error ? err.message : "Failed to archive image");
        return;
      }
      dropFromStrip(id, false);
      pendingShelfMutations.current -= 1;
      const toastId = toast(
        <button
          type="button"
          onClick={() => {
            toast.dismiss(toastId);
            useSettingsDialogStore.getState().openArchivedMedia("images");
          }}
          className="w-full cursor-pointer text-left"
        >
          You can view archived images in Settings
        </button>,
        { closeButton: true },
      );
    },
    [dropFromStrip],
  );

  // Load an image's recipe back into the form inputs.
  const restoreSettings = useCallback((image: GalleryImage) => {
    setPrompt(image.prompt);
    // Negative prompt only applies when guidance>0; do not restore a hidden value.
    const restoredNegative = image.guidance > 0 ? (image.negative_prompt ?? "") : "";
    setNegativePrompt(restoredNegative);
    if (restoredNegative) setNegativeOpen(true);
    setSteps(image.steps);
    setGuidance(image.guidance);
    // Restore from the BASE batch seed, not this image's derived seed, or a replay with batch_size
    // advances it again.
    setSeed(String(image.batch_seed ?? image.seed));
    setWidth(image.width);
    setHeight(image.height);
    // The batch shared one base seed, so a batch_index>0 image only reproduces by replaying the whole batch.
    setBatchSize(image.batch_size ?? 1);
    const m = matchAspect(image.width, image.height);
    setAspect(m.key);
    setPortrait(m.portrait);
    // Restore LoRA adapters from the recipe ("id:weight"), splitting on the LAST colon so an id
    // containing ':' survives. A recipe with no LoRAs clears the selection.
    const restoredLoras: LoraSpecInput[] = [];
    for (const entry of image.loras ?? []) {
      const idx = entry.lastIndexOf(":");
      if (idx <= 0) continue;
      const id = entry.slice(0, idx);
      const weight = Number(entry.slice(idx + 1));
      if (id && Number.isFinite(weight)) restoredLoras.push({ id, weight });
    }
    setLoras(restoredLoras);
    // The conditioned workflows' scalar settings ARE persisted, so restore them even though the form returns to Create.
    if (typeof image.strength === "number") {
      if (image.workflow === "upscale") setUpscaleStrength(image.strength);
      else setStrength(image.strength);
    }
    if (typeof image.upscale === "number") setUpscaleFactor(image.upscale);
    // None of the conditioning images are persisted, so a restore must clear the Transform /
    // Inpaint / Edit uploads and return to Create.
    setWorkflow("create");
    setInitImage(null);
    setMaskImage(null);
    setReferenceImages([]);
    // The control image is not persisted, so clear any stale ControlNet selection.
    setControlnetId("");
    setControlImage(null);
    // Say so, rather than letting a conditioned image restore as a plain Create that generates something unrelated.
    const conditioned = CONDITIONED_WORKFLOW_INPUTS[image.workflow ?? ""];
    if (conditioned) {
      toast.success(`Settings restored. Add ${conditioned} again to reproduce this image.`);
    } else {
      toast.success("Settings restored to inputs");
    }
  }, [setWorkflow]);

  // A locked ratio keeps the paired dimension in step; "custom" frees both, Flip swaps W/H. ratioHW is h/w for [a,b].
  const ratioHW = (a: number, b: number) => (portrait ? a / b : b / a);
  const changeAspect = (key: string) => {
    setAspect(key);
    if (key === "custom") return;
    const [a, b] = ASPECT_RATIOS[key];
    setHeight(snapDim(width * ratioHW(a, b)));
  };
  const changeWidth = (v: number) => {
    setWidth(v);
    if (aspect === "custom") return;
    const [a, b] = ASPECT_RATIOS[aspect];
    setHeight(snapDim(v * ratioHW(a, b)));
  };
  const changeHeight = (v: number) => {
    setHeight(v);
    if (aspect === "custom") return;
    const [a, b] = ASPECT_RATIOS[aspect];
    setWidth(snapDim(v / ratioHW(a, b)));
  };
  const flipDimensions = () => {
    setWidth(height);
    setHeight(width);
    setPortrait((p) => !p);
  };

  // A status read started before an eject can answer after the one that followed it, and this
  // page has no periodic poll to correct it, so only the newest ticket may write.
  const statusTicket = useRef(0);
  const setStatusIfNewest = useCallback(
    (ticket: number, next: DiffusionStatus) => {
      if (ticket === statusTicket.current) setStatus(next);
    },
    [],
  );

  const refreshStatus = useCallback(async () => {
    const ticket = ++statusTicket.current;
    try {
      setStatusIfNewest(ticket, await getDiffusionStatus());
    } catch {
      // Status is best-effort; a failed poll should not surface an error toast.
    }
  }, [setStatusIfNewest]);

  // Track mount so a long generate run stops issuing GPU work only on a true unmount; the page
  // stays mounted across tab switches, so a batch keeps generating off-tab.
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  // Re-sync model status when the tab becomes active: the model may have been evicted while off-tab.
  useEffect(() => {
    if (!active) return;
    if (initialReadySent.current) {
      void refreshStatus();
      return;
    }
    let cancelled = false;
    void (async () => {
      await Promise.all([
        refreshStatus(),
        (async () => {
          await loadGallery();
          const initialSelection =
            galleryCache.images.find(
              (image) => image.id === galleryCache.selectedId,
            ) ?? galleryCache.images[0];
          if (initialSelection) await ensureSrc(initialSelection);
        })(),
      ]);
      if (cancelled || initialReadySent.current) return;
      initialReadySent.current = true;
      onInitialReady?.();
    })();
    return () => {
      cancelled = true;
    };
  }, [active, ensureSrc, loadGallery, onInitialReady, refreshStatus]);

  // Ejected from the loaded models indicator, which does not run handleUnload: without this the
  // controls keep offering to generate on a freed runtime. So: handleUnload minus the unload.
  useEffect(
    () =>
      subscribeModelEjected("image", () => {
        dropResidentState();
        // That eject cancelled the replacement load, and its progress poll is the only thing that
        // clears `busy`, which dropResidentState just stopped; leaving it set locks the page.
        // Narrowed to "loading" so a generation is left alone, and held until the start settles.
        const pending = pendingStart.current;
        if (pending) {
          setBusy((prev) => (prev === "loading" ? "unloading" : prev));
          void pending
            .catch(() => {})
            .finally(() => setBusy((prev) => (prev === "unloading" ? null : prev)));
        } else {
          setBusy((prev) => (prev === "loading" ? null : prev));
        }
        setQuant(null);
        void refreshStatus();
      }),
    [refreshStatus, dropResidentState],
  );

  // Collapse the body-ported popovers when leaving the tab: the open flag stays set, so
  // returning would pop them back open.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
    setAspectOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error", updating the persistent toast in place.
  const pollLoadProgress = useCallback(async () => {
    // This tick's cancellation fence: clearing pollTimer stops the next tick, not the awaits below.
    const seq = cancelSeq.current;
    try {
      const p = await getDiffusionLoadProgress();
      if (seq !== cancelSeq.current) return;
      if (p.phase === "ready") {
        dismissLoadToast();
        const ticket = ++statusTicket.current;
        const loaded = await getDiffusionStatus();
        if (seq !== cancelSeq.current) {
          // Cancelled while this read was in flight, so it describes a pipeline being torn down. Drop
          // it and refresh NOTHING: the unload's own response is authoritative.
          return;
        }
        setStatusIfNewest(ticket, loaded);
        toast.success("Model loaded");
        setBusy(null);
        // Load succeeded: the optimistic quant is now the real one, so drop the pending revert.
        quantRevert.current?.commitRecipeClaim?.();
        quantRevert.current = null;
        // Status owns the recipe from here, so the pick's claim on Default expires with it.
        setPendingModelDefaults(null);
        // lastLoad.current already holds the now-resident pick, so drop its revert too.
        lastLoadRevert.current = null;
        return;
      }
      if (p.phase === "error") {
        dismissLoadToast();
        reportLoadFailure(p.error, "Failed to load model");
        setBusy(null);
        // A load that failed AFTER starting leaves the previous pipeline loaded, so roll the
        // optimistic quant label back.
        if (quantRevert.current) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        // Same rollback for the Reapply target: the previous pipeline is still resident.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.prev != null);
          lastLoadRevert.current = null;
        }
        // A failed load may have freed a previously-loaded model, so resync to the real backend state.
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // No load in flight and nothing loaded: the load was cancelled or evicted. Terminal, else this
        // loop spins forever.
        dismissLoadToast();
        setBusy(null);
        // Same optimistic-quant rollback as the error path: the swap did not take.
        if (quantRevert.current) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        // Restore the Reapply target too, so it never lingers on the failed pick.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.prev != null);
          lastLoadRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      // Include bytes_total: the estimate lands as a 0->real jump while phase and bytes_downloaded hold.
      const sig = `${p.phase}:${p.bytes_downloaded}:${p.bytes_total}`;
      if (loadToastId.current != null && sig !== lastLoadSig.current) {
        lastLoadSig.current = sig;
        toast(null, loadToastArgs(p, loadToastId.current, cancelLoadFromToast));
      }
    } catch {
      // Transient poll failure: keep trying.
    }
    if (seq !== cancelSeq.current) return;
    pollTimer.current = setTimeout(() => void pollLoadProgress(), 1000);
  }, [dismissLoadToast, refreshStatus, cancelLoadFromToast]);

  // Put back what a teardown removed when the load it was tearing down is still running: the
  // unload failed, so the poll and toast were stopped for nothing. refreshStatus cannot do
  // this, since a first load is not resident yet.
  const restoreLoadTracking = useCallback(() => {
    loadTrackingRestored.current = true;
    setBusy("loading");
    lastLoadSig.current = null;
    loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, undefined, cancelLoadFromToast));
    void pollLoadProgress();
  }, [pollLoadProgress, cancelLoadFromToast]);

  // Re-enter the per-step poll for a generation in flight that this page did not start.
  // generate-progress carries no terminal record, so refresh the gallery on completion.
  const resumeGeneratePoll = useCallback(() => {
    if (genPollTimer.current) clearInterval(genPollTimer.current);
    if (genVisibilityListener.current)
      document.removeEventListener("visibilitychange", genVisibilityListener.current);
    let pollInFlight = false;
    const pollGenerateOnce = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const p = await getGenerateProgress();
        if (!p.active) {
          if (genPollTimer.current) clearInterval(genPollTimer.current);
          genPollTimer.current = null;
          if (genVisibilityListener.current) {
            document.removeEventListener("visibilitychange", genVisibilityListener.current);
            genVisibilityListener.current = null;
          }
          if (!isMounted.current) return;
          setBusy(null);
          setGenStep(null);
          // Re-fetch the first page to merge images the finished run saved, and resync status.
          void loadGallery();
          void refreshStatus();
          return;
        }
        setGenStep((prev) => {
          if (prev && prev.step === p.step && prev.eta_seconds === p.eta_seconds) return prev;
          return p;
        });
      } catch {
        // transient; keep polling
      } finally {
        pollInFlight = false;
      }
    };
    genVisibilityListener.current = () => {
      if (document.visibilityState === "visible") void pollGenerateOnce();
    };
    document.addEventListener("visibilitychange", genVisibilityListener.current);
    genPollTimer.current = setInterval(() => void pollGenerateOnce(), 300);
  }, [loadGallery, refreshStatus]);

  useEffect(() => {
    void (async () => {
      await refreshStatus();
      // A load runs on the backend as a daemon thread that survives navigation, so resume tracking one still in flight.
      try {
        const p = await getDiffusionLoadProgress();
        if (p.phase === "downloading" || p.phase === "finalizing") {
          setBusy("loading");
          dismissLoadToast();
          lastLoadSig.current = null;
          loadToastId.current = toast(null, loadToastArgs(p, undefined, cancelLoadFromToast));
          void pollLoadProgress();
        }
      } catch {
        // Resume is best-effort; a failed probe just leaves the idle view.
      }
      // Resume tracking a generation started elsewhere so the page shows the in-flight run. Mirrors the video page.
      try {
        const g = await getGenerateProgress();
        if (g.active) {
          setBusy("generating");
          setGenStep(g);
          resumeGeneratePoll();
        }
      } catch {
        // Resume is best-effort; a failed probe just leaves the idle view.
      }
    })();
    // Stop polling if the page unmounts mid-load/generate, and dismiss the load toast: its poll loop is gone.
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      if (genVisibilityListener.current) {
        document.removeEventListener("visibilitychange", genVisibilityListener.current);
        genVisibilityListener.current = null;
      }
      dismissLoadToast();
    };
  }, [refreshStatus, dismissLoadToast, pollLoadProgress, resumeGeneratePoll, cancelLoadFromToast]);

  // Seed the sliders from a resident model's recipe when the page finds one it did not load,
  // else a resident flux.1-dev generates garbage at 9 steps. Guarded by a per-repo ref.
  const residentSeeded = useRef(false);
  useEffect(() => {
    const repoId = status?.loaded ? status.repo_id : null;
    if (!repoId) return;
    if (lastLoad.current) return;
    if (seededResident.current === repoId) return;
    seededResident.current = repoId;
    // Wire Reapply to the resident model too. Only a full pipeline is reloadable by repo id
    // alone; a resident GGUF carries no checkpoint filename, so the target stays null and the
    // button hidden. Set before the recipe decision below, which is a separate question.
    if (status?.model_kind === "pipeline") {
      lastLoad.current = { repoId, kind: "pipeline" };
    }
    // A stored recipe is the user's own choice, so it outranks the resident model's defaults on the first seed.
    if (!residentSeeded.current) {
      residentSeeded.current = true;
      if (imagePresets.storedRecipe) return;
    }
    // Seed from base_repo (the resolved diffusers base, holding the family), not repo_id: a GGUF
    // resident has no family substring. Status is the authority for a resident model.
    const d = defaultsFor(status?.base_repo ?? repoId);
    setPendingModelDefaults(null);
    setSteps(d.steps);
    setGuidance(d.guidance);
  }, [imagePresets.storedRecipe, status?.loaded, status?.repo_id, status?.base_repo, status?.model_kind]);

  // Reseed the Advanced selects from the LOADED build, so a declined request snaps to what
  // engaged and Precision never advertises a scheme the model is not running. Keyed on the
  // LOAD-TIME half of the record: the backend rewrites the speed/attention entries at
  // GENERATION time, and the whole record threw away a Precision picked but not yet loaded.
  const resolvedKey = status?.loaded ? resolvedSeedKey(status.resolved) : null;
  useEffect(() => {
    const record = status?.loaded ? status.resolved : null;
    if (!record) return;
    const quant = resolvedSelectValue(record.transformer_quant, (v) =>
      // The engaged value spells "no quant" as "off"; the select's option for it is "none".
      (["auto", "none", "int8", "fp8", "nvfp4", "mxfp8"] as const).find(
        (o) => o === v || (o === "none" && v === "off"),
      ) ?? null,
    );
    if (quant) setTransformerQuant(quant);
    const memory = resolvedSelectValue(record.memory_mode, (v) =>
      (["auto", "fast", "balanced", "low_vram"] as const).find((o) => o === v) ?? null,
    );
    if (memory) setMemoryMode(memory);
    const attention = resolvedSelectValue(record.attention_backend, (v) =>
      // The engaged value uses the dispatcher's own name; map it back to the option.
      (["auto", "native", "cudnn", "flash3", "sage"] as const).find(
        (o) => o === v || `_native_${o}` === v,
      ) ?? null,
    );
    if (attention) setAttentionBackend(attention);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- resolvedKey stands for the record
  }, [resolvedKey]);

  const bakedLorasFor = useCallback(
    (repoId: string): LoraSpecInput[] => {
      const sameTarget = repoId === (lastLoad.current?.repoId ?? status?.repo_id ?? null);
      if (!sameTarget) return [];
      return loras
        .map((l) => ({ id: l.id.trim(), weight: l.weight }))
        .filter((l) => l.id && l.weight > 0);
    },
    [loras, status?.repo_id],
  );

  // One snapshot of every Advanced control a load sends, so a staged pick can pin the values it planned against.
  const currentLoadAdvanced = useCallback(
    (repoId: string): LoadAdvanced => {
      const baked = bakedLorasFor(repoId);
      return {
        cpu_offload: cpuOffload,
        speed_mode: speedMode === "auto" ? undefined : speedMode,
        transformer_quant: transformerQuant === "auto" ? undefined : transformerQuant,
        attention_backend: attentionBackend === "auto" ? undefined : attentionBackend,
        memory_mode: memoryMode === "auto" ? undefined : memoryMode,
        transformer_cache: transformerCache === "auto" ? undefined : transformerCache,
        loras: baked.length > 0 ? baked : undefined,
        // Dropped when the chosen card is gone, so a stale pick loads automatically instead of 400ing.
        gpu_ids:
          selectedGpu !== "auto" &&
          gpuChoices.some((d) => String(d.index) === selectedGpu)
            ? [Number(selectedGpu)]
            : undefined,
      };
    },
    [
      bakedLorasFor,
      cpuOffload,
      speedMode,
      transformerQuant,
      attentionBackend,
      memoryMode,
      transformerCache,
      selectedGpu,
      gpuChoices,
    ],
  );

  const handleLoad = useCallback(
    // Resolves true when the background load STARTED (callers may revert optimistic picker state on false).
    async (
      repoId: string,
      opts: {
        kind: "gguf" | "single_file" | "pipeline";
        filename?: string;
      },
      // The Advanced values this load must use when pinned earlier: a staged download plans its file
      // set at pick time and loads minutes later, so live state could outrun the staged files.
      pinned?: LoadAdvanced,
    ): Promise<boolean> => {
      // Cancel any prior poll loop so two cannot run at once.
      if (pollTimer.current) clearTimeout(pollTimer.current);
      // Read BEFORE the start request goes out: a Cancel pressed while it is in flight sends an
      // unload that can reach the backend first, find no load registered, and stop nothing.
      const startSeq = cancelSeq.current;
      const startLoad = ++loadSeq.current;
      // Published now and settled in the finally below, so a cancel waits for the WHOLE path.
      let settleLoad: () => void = () => {};
      const inFlight = new Promise<void>((resolve) => {
        settleLoad = resolve;
      });
      pendingStart.current = inFlight;
      // Every exit below goes through this: it settles the promise a cancel is waiting on and releases the ref.
      const settle = (started: boolean): boolean => {
        settleLoad();
        if (pendingStart.current === inFlight) pendingStart.current = null;
        return started;
      };
      setBusy("loading");
      // Show the chat-style toast immediately; the poll updates it by id.
      dismissLoadToast();
      lastLoadSig.current = null;
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, undefined, cancelLoadFromToast));
      // Remember what was loaded so "Reapply" can reload it. Snapshot the prior target first: a load
      // that fails to START leaves the previous model resident.
      const prevLastLoad = lastLoad.current;
      // A torchao int8/fp8 transformer takes adapters only at LOAD time and /images/generate then
      // rejects a new set, so a reload must keep the selection. Ignored by bf16 / bnb-4bit.
      const advanced = pinned ?? currentLoadAdvanced(repoId);
      const bakeLoras = advanced.loras ?? [];
      // Whether THIS load carries the selection into the build, so a quantized load that did not can drop it.
      bakedLorasOnLoad.current = bakeLoras.length > 0;
      lastLoad.current = { repoId, kind: opts.kind, filename: opts.filename };
      setCanReapply(true);
      // Carry the prior target so the async poll can restore it if the background load fails after starting.
      lastLoadRevert.current = { prev: prevLastLoad };
      try {
        // Returns immediately; the load runs in the background and we poll. The backend infers the
        // family and base repo from the id, and the saved HF token covers gated bases.
        const startRequest = loadDiffusionModel({
          model_path: repoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          cpu_offload: advanced.cpu_offload,
          speed_mode: advanced.speed_mode,
          // GGUF picks only: the dense fast path replaces a GGUF transformer, and every other kind runs
          // its checkpoint's own precision. The control is hidden there but the state persists across
          // picks, so a stale scheme would reach a load that can only decline it.
          transformer_quant: opts.kind === "gguf" ? advanced.transformer_quant : undefined,
          attention_backend: advanced.attention_backend,
          memory_mode: advanced.memory_mode,
          transformer_cache: advanced.transformer_cache,
          loras: bakeLoras.length > 0 ? bakeLoras : undefined,
          gpu_ids: advanced.gpu_ids,
        });
        await startRequest;
      } catch (err) {
        lastLoad.current = prevLastLoad;
        setCanReapply(prevLastLoad != null);
        lastLoadRevert.current = null;
        dismissLoadToast();
        reportLoadFailure(err instanceof Error ? err.message : "", "Failed to start load");
        setBusy(null);
        void refreshStatus();
        return settle(false);
      }
      if (startSeq !== cancelSeq.current) {
        // Cancelled during the start request: the unload it sent may have landed before this load
        // registered, leaving it running with no toast and no Cancel. The load exists as of this
        // line, so unload once more, unless a NEWER load has taken the page.
        if (startLoad === loadSeq.current) {
          try {
            await unloadDiffusionModel();
          } catch {
            // This request is the ONLY one that can still stop the load the first unload missed, so a
            // failure here is not best-effort: put the tracking back exactly as a failed cancel does.
            restoreLoadTracking();
            return settle(false);
          }
        }
        void refreshStatus();
        return settle(false);
      }
      void pollLoadProgress();
      return settle(true);
    },
    [pollLoadProgress, refreshStatus, dismissLoadToast, currentLoadAdvanced, cancelLoadFromToast],
  );

  // Set or clear the Transform/Inpaint source image; always drop the painted mask, which is
  // sized to the previous source.
  const handleInitChange = useCallback((dataUrl: string | null) => {
    setInitImage(dataUrl);
    setMaskImage(null);
    setMaskResetKey((k) => k + 1);
  }, []);

  // Downloads go through the Hub download manager like every other model, so the load finds a
  // warm cache. In a ref, so the callback is not a render dep.
  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string };
    // The Advanced values the plan was built from: staging does not set `busy`, so the user can
    // change precision or LoRAs while the download runs.
    advanced: LoadAdvanced;
    // The pick that staged it: a download outlives its pick, so it must not evict a newer one when it lands.
    token: number;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  handleLoadRef.current = handleLoad;
  // Set when a staged download finished while this page was hidden: both diffusion pages stay
  // mounted and a load evicts whatever holds the GPU. The pick fires on return.
  const stagedLoadDeferred = useRef(false);
  // Both deferred paths run the load minutes after the pick was reported started, so both need
  // the same rollback: a deferred load can still be REFUSED, and staging polls nothing.
  // `owned` is read BEFORE the call, so a newer pick's label is left alone.
  const runStagedLoad = useCallback(
    (pending: NonNullable<typeof pendingStagedLoad.current>) => {
      if (pendingStagedLoad.current === pending) pendingStagedLoad.current = null;
      if (!pickGuard.isLatest(pending.token)) return;
      const owned = stagedQuantRevert.current;
      void handleLoadRef.current(pending.repoId, pending.opts, pending.advanced).then((started) => {
        if (started) return;
        if (quantRevert.current && quantRevert.current === owned) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        if (stagedQuantRevert.current === owned) stagedQuantRevert.current = null;
      });
    },
    [pickGuard, revertPick],
  );
  const { stage } = useStagedDownload({
    scopeId: "diffusion",
    onReady: () => {
      if (!active) {
        stagedLoadDeferred.current = true;
        return;
      }
      const pending = pendingStagedLoad.current;
      if (pending) runStagedLoad(pending);
    },
    onCancelled: () => {
      // The selected model is only an intent until every dependency is ready: a cancelled companion
      // must not leave that intent behind for a late completion to load.
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      // Staging starts no load, so the optimistic label must come back here or the selector keeps
      // describing the resident model with a quant that never loaded. Only for this job's pick.
      if (quantRevert.current && quantRevert.current === stagedQuantRevert.current) {
        revertPick(quantRevert.current);
        quantRevert.current = null;
      }
      stagedQuantRevert.current = null;
    },
  });

  useEffect(() => {
    if (!active || !stagedLoadDeferred.current) return;
    stagedLoadDeferred.current = false;
    const pending = pendingStagedLoad.current;
    if (pending) runStagedLoad(pending);
  }, [active, runStagedLoad]);

  // Ask for a cache-aware plan on every Hub pick: the picker knows only whether the checkpoint
  // is cached, and image models can need a separate text encoder/VAE repo. True = accepted.
  const requestDownloadPlan = useCallback(
    (
      repoId: string,
      opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string },
      advanced: LoadAdvanced,
    ) =>
      getDiffusionDownloadPlan({
        model_path: repoId,
        gguf_filename: opts.filename,
        model_kind: opts.kind,
        // The same token and Advanced values handleLoad sends, so the plan describes the load that
        // will run: without the token a gated base plans no companion, and without the controls it
        // stages shards the load never opens.
        hf_token: hfApiToken(getHfToken()),
        cpu_offload: advanced.cpu_offload,
        speed_mode: advanced.speed_mode,
        // Non-GGUF loads ignore this control; the plan must describe the same request as handleLoad.
        transformer_quant: opts.kind === "gguf" ? advanced.transformer_quant : undefined,
        memory_mode: advanced.memory_mode,
        // The backend prefetch decision reads the adapter selection too: a baked LoRA always runs the
        // dense build path, and omitting it staged too little.
        loras: advanced.loras,
        // The plan route preflights precision and sizes the file set against the card the load will
        // use, so a selection the load carries has to reach the plan.
        gpu_ids: advanced.gpu_ids,
      }),
    [],
  );

  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string },
      source: ModelSelectorChangeMeta["source"] = "hub",
      token?: number,
    ): Promise<boolean> => {
      // Staging never sets `busy`, so a second pick passes the guard while this plan is in flight,
      // and plans resolve in response order rather than pick order. Bumped before the non-hub
      // return too, so a local pick invalidates an in-flight hub plan.
      // Plans resolve in response order, not pick order. Bumped before the non-hub return too: a local
      // pick must invalidate an in-flight hub plan.
      const pick = ++pickSeq.current;
      // The previous pick's staged intent dies with it: a pick that stages nothing never calls
      // stage(), so the queue keeps the older job and its onReady loads the abandoned model.
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      stagedQuantRevert.current = null;
      const owns = () => token === undefined || pickGuard.holds(token);
      if (!owns()) return true;
      if (source !== "hub") return handleLoadRef.current(repoId, opts);
      // ONE snapshot for the plan and the load it fires: the download runs for minutes without setting `busy`.
      const advanced = currentLoadAdvanced(repoId);
      // Read before the await: a pick made while the plan resolves replaces quantRevert, and this
      // job must not revert it.
      const ownRevert = quantRevert.current;
      // Read inside the try, acted on outside it: refusing from in there would fall through to the
      // load if the refusal itself threw.
      let incompatible: string | null = null;
      try {
        const plan = await requestDownloadPlan(repoId, opts, advanced);
        // Superseded. Report started so this pick's `.then` leaves the newer label alone.
        if (pick !== pickSeq.current || !owns()) return true;
        incompatible = plan.incompatible_reason ?? null;
        if (!incompatible && plan.entries.length > 0) {
          pendingStagedLoad.current = {
            repoId,
            opts,
            advanced,
            token: token ?? pickGuard.claim(),
          };
          stagedQuantRevert.current = ownRevert;
          stage(
            plan.entries.map((e) => ({
              repoId: e.repo_id,
              files: e.files,
              bytes: e.bytes,
              ggufFilename: e.gguf_filename,
              // The entry carrying the picked checkpoint file, so the panel can label it without guessing:
              // filenames cannot tell the two apart, and repo identity is not enough when a checkpoint
              // shares its repo with cached companions. The backend's answer wins, since a gated pipeline
              // is staged from an ungated MIRROR. Nullish coalescing, not `or`: a planner
              // answering false is still an answer.
              checkpoint:
                e.checkpoint ??
                (opts.filename
                  ? e.files.includes(opts.filename)
                  : e.repo_id === repoId),
            })),
          );
          return true;
        }
      } catch {
        // No plan (older backend, metadata hiccup): fall back to the load's own download.
      }
      // Re-checked: a plan that REJECTED after a newer pick would otherwise reach the fallback load.
      if (pick !== pickSeq.current || !owns()) return true;
      if (incompatible) {
        toast.error(incompatible);
        return false;
      }
      return handleLoadRef.current(repoId, opts, advanced);
    },
    [stage, currentLoadAdvanced, requestDownloadPlan],
  );

  const resolveDownloadFootprint = useCallback(
    async (repoId: string, meta: ModelSelectorChangeMeta) => {
      if (!meta.ggufFilename) return null;
      const plan = await requestDownloadPlan(
        repoId,
        { kind: "gguf", filename: meta.ggufFilename },
        currentLoadAdvanced(repoId),
      );
      const requiredBytes = plan.required_bytes ?? 0;
      if (requiredBytes <= 0) return null;
      return {
        requiredBytes,
        checkpointBytes:
          plan.checkpoint_bytes ?? meta.expectedBytes ?? 0,
      };
    },
    [currentLoadAdvanced, requestDownloadPlan, pickGuard, stage],
  );

  // A GGUF pick can arrive with only a repo id. The backend rejects a gguf load with no filename
  // and a pipeline load of a GGUF repo, so name the file from the listing first.
  const loadGgufRepoPick = useCallback(
    async (
      repoId: string,
      quantHint: string | null,
      source: ModelSelectorChangeMeta["source"] = "hub",
      localPath?: string | null,
    ): Promise<boolean> => {
      // Claimed here so every entry point is covered; the next pick's claim makes this one inert.
      const token = pickGuard.claim();
      const isCurrent = () => isMounted.current && pickGuard.holds(token);
      const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
      return runGgufRepoPick({
        isCurrent,
        resolve: () =>
          resolveDiffusionGgufFilename(repoId, {
            quant: quantHint,
            localPath,
            hfToken: hfApiToken(getHfToken()),
          }),
        // Still ambiguous (several quants, or the listing failed): only the expander can say which.
        onAmbiguous: () =>
          toast.error("Pick a quantization for this model to load it"),
        // Optimistic label, reverted if the load never starts, like the curated GGUF branch below.
        onResolved: (filename) => {
          quantRevert.current = revert;
          setQuant(quantHint ?? filename);
          applyImageModelDefaults(repoId);
        },
        onNotStarted: () => {
          if (quantRevert.current === revert) {
            revertPick(revert);
            quantRevert.current = null;
          }
        },
        load: (filename) =>
          loadOrStage(repoId, { kind: "gguf", filename }, source, token),
      });
    },
    [applyImageModelDefaults, loadOrStage, pickGuard, quant, revertPick],
  );

  // A hidden page owns nothing: both stay mounted, so a resolution started here must not load after the user switched.
  useEffect(() => {
    if (!active) pickGuard.release();
  }, [active, pickGuard]);

  // A diffusion model picked from the chat picker arrives as ?model= on this route. This route's
  // own match, never `strict: false`: that resolves to the ROOT match, whose search is
  // whatever route is live, and /hub names its selection with the same param.
  const routeSearch = useSearch({ from: "/images", shouldThrow: false });
  const navigateSelf = useNavigate();
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    // A hidden page owns no query: both diffusion pages stay mounted.
    if (!active) return;
    if (!imagePresets.hydrated) return;
    const wanted = routeSearch?.model;
    // Key on the model AND the quant, and release the marker once the query is gone: this page
    // stays mounted, so a marker outliving the query made re-picking the same file a dead click.
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    // `quant` is used verbatim as a filename, so a label there is resolved instead. The two
    // fields, not the object: `routeSearch` is rebuilt every render.
    const routed = { quant: routeSearch?.quant, ggufQuant: routeSearch?.ggufQuant };
    const routedFilename = routedGgufFilename(routed);
    const routedLabel = routedGgufLabel(routed);
    const key = `${wanted}|${routeSearch?.quant ?? ""}|${routeSearch?.ggufQuant ?? ""}`;
    if (handledRouteModel.current === key) return;
    handledRouteModel.current = key;
    // This arrival owns the page like a direct pick, so a download staged by an earlier one cannot land on top.
    const token = pickGuard.claim();
    void navigateSelf({ to: "/images", search: {}, replace: true });
    // A label means a GGUF repo whatever the catalog says, and is not loadable, so resolve it
    // rather than routing it as a filename.
    if (routedLabel) {
      // Deferred, not inline: resolution is a request, and the load it fires owns the state a direct pick sets.
      void Promise.resolve().then(() =>
        loadGgufRepoPick(wanted, routedLabel, "hub"),
      );
      return;
    }
    // Same catalog lookup a direct pick makes: the chat picker can only forward a GGUF filename.
    const pick = diffusionRoutePick(
      wanted,
      routedFilename ?? undefined,
      loadSpecFor(wanted, IMAGE_CATALOG),
    );
    // A curated GGUF artifact resolves to kind "gguf" with no filename: the catalog lists the repo, not its files.
    if (pick.opts.kind === "gguf" && !pick.opts.filename) {
      void Promise.resolve().then(() => loadGgufRepoPick(pick.repoId, null, "hub"));
      return;
    }
    // Match every direct picker branch: the routed intent owns both the visible build label and
    // the Default recipe, and a load that never becomes resident rolls both back.
    const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
    quantRevert.current = revert;
    setQuant(pick.opts.kind === "pipeline" ? null : (pick.opts.filename ?? null));
    applyImageModelDefaults(wanted);
    void loadOrStage(pick.repoId, pick.opts, "hub", token).then((started) => {
      if (!started && pickGuard.holds(token) && quantRevert.current === revert) {
        revertPick(revert);
        quantRevert.current = null;
      }
    });
  }, [
    active,
    applyImageModelDefaults,
    imagePresets.hydrated,
    routeSearch?.model,
    routeSearch?.quant,
    routeSearch?.ggufQuant,
    loadOrStage,
    loadGgufRepoPick,
    navigateSelf,
    pickGuard,
    quant,
    revertPick,
  ]);

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    const l = lastLoad.current;
    if (l) void handleLoad(l.repoId, { kind: l.kind, filename: l.filename });
  }, [handleLoad]);

  // Every pick supersedes the one before it, whichever route it takes: a staged download
  // outlives its pick, and the direct-local branches call handleLoad rather than loadOrStage,
  // so clearing only inside loadOrStage left the old job free to load the abandoned model. A
  // pick rejected after beginPick() has already retired its predecessor, so restore here.
  const abandonPick = useCallback(() => {
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [revertPick]);

  const beginPick = useCallback(() => {
    pickSeq.current += 1;
    pendingStagedLoad.current = null;
    stagedLoadDeferred.current = false;
    stagedQuantRevert.current = null;
  }, []);

  // The chat picker emits (modelId, quant + filename) for a GGUF, or just (modelId) for a curated safetensors pick.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight: the backend rejects a second load with a 409.
      if (busy !== null) return;
      beginPick();
      // This pick owns the page now, so one still awaiting a listing or a plan drops out. Before any
      // branch, since staging never sets `busy`.
      const token = pickGuard.claim();
      // Curated non-GGUF model: load as a full pipeline or single-file safetensors.
      const spec = loadSpecFor(id, IMAGE_CATALOG);
      if (spec && spec.kind !== "gguf") {
      // Carried forward when one is already pending: a superseded staged pick left its optimistic
      // quant and recipe in state, so snapshotting now would record THAT and restore a model that
      // never loaded. Leaving the old entry would also let that download revert this pick.
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(null);
        applyImageModelDefaults(id);
        void loadOrStage(
          id,
          { kind: spec.kind, filename: spec.filename },
          meta.source,
          token,
        ).then((started) => {
            if (!started && pickGuard.holds(token)) {
              revertPick(revert);
              quantRevert.current = null;
            }
          });
        return;
      }
      // GGUF quant pick from the variant expander. Optimistic for instant feedback, but reverted if
      // the load fails to START or later in the poll: the old pipeline stays loaded either way.
      if (meta.ggufVariant && meta.ggufFilename) {
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(meta.ggufVariant);
        applyImageModelDefaults(id);
        void loadOrStage(
          id,
          { kind: "gguf", filename: meta.ggufFilename },
          meta.source,
          token,
        ).then((started) => {
          // `quantRevert` is one slot, so only the pick that set the label may take it back.
          if (!started && pickGuard.holds(token)) {
            revertPick(revert);
            quantRevert.current = null;
          }
        });
        return;
      }
      // A direct local .gguf pick has no variant/filename; load it by splitting the path into (parent dir, basename).
      if (meta.isGguf) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        if (!filename.toLowerCase().endsWith(".gguf")) {
          // A repo id or local directory, not a file: the listing names its .gguf and the label picks
          // between siblings, and a local pick passes its directory so the listing reads that path.
          void loadGgufRepoPick(
            id,
            meta.ggufVariant ?? null,
            meta.source,
            meta.source === "local" ? id : null,
          );
          return;
        }
        // A direct pick carries no curated variant label, so surface the filename or the selector
        // keeps advertising the old quant. Optimistic, reverted if the load never starts.
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(filename);
        applyImageModelDefaults(id);
        void handleLoad(dir, { kind: "gguf", filename }).then((started) => {
          if (!started) {
            revertPick(revert);
            quantRevert.current = null;
          }
        });
        return;
      }
      // A direct local single-file .safetensors pick must load via from_single_file: the pipeline
      // route rejects a bare file, and only after evicting the resident model.
      if (meta.source === "local" && id.toLowerCase().endsWith(".safetensors")) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(filename);
        applyImageModelDefaults(id);
        void handleLoad(dir, { kind: "single_file", filename }).then((started) => {
          if (!started) {
            revertPick(revert);
            quantRevert.current = null;
          }
        });
        return;
      }
      // A GGUF repo with no filename: these used to fall through to the pipeline branch below, which
      // the backend rejects for a single-file GGUF repo.
      if (spec?.kind === "gguf" || meta.ggufVariant) {
        // An artifact that names its file short-circuits the listing; otherwise the label is the hint.
        void loadGgufRepoPick(
          id,
          spec?.filename ?? meta.ggufVariant ?? null,
          meta.source,
          meta.source === "local" ? id : null,
        );
        return;
      }
      // Otherwise treat it as a full diffusers repo. The backend gates loads to unsloth/* repos or on-device paths.
      if (meta.source !== "local" && !id.toLowerCase().startsWith("unsloth/")) {
        toast.error("Only unsloth or on-device image models can be loaded here");
        abandonPick();
        return;
      }
      // Optimistically clear the quant label, revert it if the load never starts.
      const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
      quantRevert.current = revert;
      setQuant(null);
      applyImageModelDefaults(id);
      void loadOrStage(id, { kind: "pipeline" }, meta.source, token).then((started) => {
        if (!started && pickGuard.holds(token)) {
          revertPick(revert);
          quantRevert.current = null;
        }
      });
    },
    [
      abandonPick,
      applyImageModelDefaults,
      beginPick,
      busy,
      handleLoad,
      loadGgufRepoPick,
      loadOrStage,
      pickGuard,
      quant,
      revertPick,
    ],
  );

  // Deploy a freshly-trained adapter from the Train tab: switch to Create, load the base, and
  // queue the adapter for the LoRA discovery effect.
  const handleDeployAdapter = useCallback(
    (args: { baseRepo: string; family: string; catalogPath: string; trigger: string }) => {
      if (busy !== null) {
        toast.error("Finish the current model load before deploying the adapter.");
        return;
      }
      // The picker keys a local adapter by its filename stem (see diffusion_lora scan).
      const base = args.catalogPath.replace(/\\/g, "/").split("/").pop() ?? "";
      const stem = base.replace(/\.(safetensors|gguf)$/i, "");
      if (!stem) {
        toast.error("Could not resolve the trained adapter's name.");
        return;
      }
      // The deploy owns the page now: a resolving pick or a staged download would load over the base it is about to.
      pickGuard.cancel();
      pendingDeploy.current = { loraId: stem, family: args.family };
      if (args.trigger.trim()) setPrompt(args.trigger.trim());
      setPageMode("create");
      const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
      quantRevert.current = revert;
      setQuant(null);
      applyImageModelDefaults(args.baseRepo);
      void handleLoad(args.baseRepo, { kind: "pipeline" }).then((started) => {
        if (!started) {
          pendingDeploy.current = null;
          if (quantRevert.current === revert) {
            revertPick(revert);
            quantRevert.current = null;
          }
        }
      });
    },
    [applyImageModelDefaults, busy, handleLoad, pickGuard, quant, revertPick, setPageMode],
  );

  // Resolves true when the backend accepted the unload; handleCancelLoad reports the cancel only then.
  const handleUnload = useCallback(async (): Promise<boolean> => {
    // Ejecting cancels any in-flight replacement load, so tear down its client-side tracking too
    // or the toast leaks forever.
    dropResidentState();
    loadTrackingRestored.current = false;
    setBusy("unloading");
    try {
      setStatusIfNewest(++statusTicket.current, await unloadDiffusionModel());
      setQuant(null);
      // Hold the page until any load start still in flight has run to its END, compensating unload
      // and all. Without the fence an eject landing before the start registered returned success
      // and cleared busy, so the next pick was refused while the older load carried on.
      // That older handler, seeing the newer loadSeq, skips its compensating unload and returns without
      // restarting its poll, leaving a multi-gigabyte load running with no toast and no cancel control.
      const pending = pendingStart.current;
      if (pending) {
        try {
          await pending;
        } catch {
          // Its own handler reports the failure; this only waits for the window to close.
        }
      }
      // The wait above can end with the tracking RESTORED: handleLoad's compensating unload failed,
      // so the load is still running and this eject stopped nothing.
      return !loadTrackingRestored.current;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
      return false;
    } finally {
      // Not an unconditional clear: a restore during the wait put the page back to "loading"
      // deliberately, and wiping it hides the Cancel controls over a load still running.
      setBusy((prev) => (prev === "unloading" ? null : prev));
    }
  }, [refreshStatus, dropResidentState]);

  // Cancelling a load IS the unload: it sets the load's cancel event, bumps the load token so
  // the worker can never commit, and drops the load marker. What it leaves is only cache, so
  // loading the same model again resumes instead of restarting.
  const handleCancelLoad = useCallback(async () => {
    const wasLoading = busy === "loading";
    if (await handleUnload()) {
      // handleUnload holds the page for the whole pending-start path, so by here the window for the
      // backend's "a load is already in progress" refusal is shut.
      toast.info("Stopped loading the model", {
        description: "Anything already downloaded stays cached, so loading it again resumes.",
      });
      return;
    }
    // Already restored inside handleUnload, so the toast and poll are up: a second restore would
    // raise a duplicate toast and a second poll loop.
    if (!wasLoading || loadTrackingRestored.current) return;
    // The unload failed, so the load is still running and its tracking was torn down for nothing.
    restoreLoadTracking();
  }, [busy, handleUnload, restoreLoadTracking]);

  useEffect(() => {
    cancelLoadRef.current = () => void handleCancelLoad();
  }, [handleCancelLoad]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    const isTransform = workflow === "transform";
    const isInpaint = workflow === "inpaint";
    const isExtend = workflow === "extend";
    const isUpscale = workflow === "upscale";
    const isReference = workflow === "reference";
    const isEdit = workflow === "edit";
    const usesInit = isTransform || isInpaint || isExtend || isUpscale || isReference || isEdit;
    const tabLabel = isInpaint
      ? "Inpaint"
      : isExtend
        ? "Extend"
        : isUpscale
          ? "Upscale"
          : isReference
            ? "Reference"
            : isEdit
              ? "Edit"
              : "Transform";
    if (usesInit && !initImage) {
      toast.error(`Upload a source image for ${tabLabel}`);
      return;
    }
    if (isInpaint && !maskImage) {
      toast.error("Paint a mask over the region to regenerate");
      return;
    }
    if (isExtend && !(extendSides.left || extendSides.right || extendSides.top || extendSides.bottom)) {
      toast.error("Pick at least one side to extend");
      return;
    }

    // Resolve the conditioning image/mask/strength up front. Extend is built here by padding and
    // masking, then sent through inpaint.
    let condInit: string | undefined;
    let condMask: string | undefined;
    let condStrength: number | undefined;
    let condUpscale: number | undefined;
    let condRefImages: string[] | undefined;
    try {
      if (isTransform) {
        condInit = initImage ?? undefined;
        condStrength = strength;
      } else if (isInpaint) {
        condInit = initImage ?? undefined;
        condMask = maskImage ?? undefined;
        condStrength = strength;
      } else if (isExtend) {
        const built = await buildOutpaint(initImage!, extendSides, extendPct);
        condInit = built.image;
        condMask = built.mask;
        condStrength = 1; // the new border is blank canvas: redraw it fully
      } else if (isUpscale) {
        // Hires fix: the backend enlarges the source and re-denoises at this low strength, gaining
        // detail without changing content.
        condInit = initImage ?? undefined;
        condUpscale = upscaleFactor;
        condStrength = upscaleStrength;
      } else if (isReference) {
        // FLUX.2 reference conditioning: send the primary plus extra references. A fresh image is
        // generated at the slider size.
        condInit = initImage ?? undefined;
        const extras = referenceImages.filter(Boolean);
        if (extras.length) condRefImages = extras;
      } else if (isEdit) {
        // Instruction editing: send the source image; the prompt IS the instruction. No mask, no strength.
        condInit = initImage ?? undefined;
      }
    } catch {
      toast.error("Could not prepare the source image");
      return;
    }
    // Resolve a base seed up front, so each sequential image gets a distinct, reproducible seed (base + i).
    let baseSeed: number;
    if (seed.trim()) {
      const n = Number(seed);
      if (!Number.isInteger(n) || n < 0 || n > Number.MAX_SAFE_INTEGER) {
        toast.error("Seed must be a non-negative integer");
        return;
      }
      baseSeed = n;
    } else {
      baseSeed = Math.floor(Math.random() * 2 ** 32);
    }

    // Snap custom dims to the model's grid so a half-typed value cannot 422.
    const w = snapDim(width);
    const h = snapDim(height);

    // A large run count is legitimate, so no upper cap: floor at 1 and ignore non-numeric input.
    const runs = Number.isFinite(count) && count >= 1 ? Math.floor(count) : 1;
    if (runs !== count) setCount(runs);

    // An explicit seed near the 2**53-1 backend cap can overflow once the per-run and in-batch
    // offsets are added, 422ing a later run after earlier images generated. Fail before GPU work.
    if (baseSeed > Number.MAX_SAFE_INTEGER - (runs * batchSize - 1)) {
      toast.error("Seed too large for this run count and batch size; use a smaller seed");
      return;
    }

    setBusy("generating");
    setGenDone(0);
    setGenStep(null);
    // Fresh run: a Stop from the PREVIOUS run must not cancel this one.
    cancelRequested.current = false;
    cancelAcked.current = false;
    // Per-run, like the two above: a cancel POST still outstanding from the PREVIOUS run would
    // leave the guard set and swallow this run's own Stop.
    cancelInFlight.current = null;
    // Any Stop still pending belongs to the run that just ended, so it must not reach the server now.
    cancelAbort.current?.abort();
    cancelAbort.current = null;
    runToken.current += 1;
    // Poll the backend's per-step progress across the whole run. A named poll body also serves the
    // visibilitychange listener, so a throttled tab catches up when visible.
    let pollInFlight = false;
    const pollGenerateOnce = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const p = await getGenerateProgress();
        // Skip the state update (and re-render) when nothing the bar shows moved.
        setGenStep((prev) => {
          if (!p.active) return null;
          if (prev && prev.step === p.step && prev.eta_seconds === p.eta_seconds) return prev;
          return p;
        });
      } catch {
        // transient; keep polling
      } finally {
        pollInFlight = false;
      }
    };
    if (genVisibilityListener.current)
      document.removeEventListener("visibilitychange", genVisibilityListener.current);
    genVisibilityListener.current = () => {
      if (document.visibilityState === "visible") void pollGenerateOnce();
    };
    document.addEventListener("visibilitychange", genVisibilityListener.current);
    genPollTimer.current = setInterval(() => void pollGenerateOnce(), 300);
    // Every gallery id this page has seen, captured BEFORE the first POST and grown as records
    // arrive: settleLostGeneration proves a lost POST landed by finding a record outside it.
    const knownIds = new Set(galleryCache.images.map((image) => image.id));
    try {
      for (let i = 0; i < runs; i++) {
        // Stop issuing more GPU generations once the page unmounted or Stop was pressed: the backend
        // cancel only reaches the denoise in flight, so a count > 1 request would run on.
        if (
          !shouldContinueGenerating({
            mounted: isMounted.current,
            stopRequested: cancelRequested.current,
          })
        )
          break;
        // Frozen BEFORE the POST, because both halves must describe the same moment: `knownIds` is
        // from before the request, so deriving the window half in the catch mixes the two and the
        // newest historical row could read as proof the generation landed.
        const probeBaseline = newRecordProbeBaseline(
          galleryCache.images,
          galleryCache.hasMore,
          knownIds,
        );
        let res: DiffusionGenerateResponse;
        try {
          res = await generateDiffusionImage({
            prompt: prompt.trim(),
            // Only send a negative prompt when guidance uses it, so the recipe does not record one the model ignored.
            negative_prompt: guidance > 0 ? negativePrompt.trim() || undefined : undefined,
            width: w,
            height: h,
            steps,
            guidance,
            // Offset runs by the batch size: the native engine seeds image j at seed+j, so a +1 offset
            // would regenerate batch-mates.
            seed: baseSeed + i * batchSize,
            batch_size: batchSize,
            // Transform/Inpaint/Extend send the source image (and mask) with a strength; the backend
            // derives output size from the image.
            init_image: condInit,
            mask_image: condMask,
            strength: condStrength,
            upscale: condUpscale,
            reference_images: condRefImages,
            // Drop empty and zero-weight rows and trim hand-typed repo ids, so the recipe records only
            // adapters that applied. Gated on loraCapable, since a restore can leave adapters in state.
            loras: (() => {
              if (!loraCapable) return undefined;
              const active = loras
                .map((l) => ({ id: l.id.trim(), weight: l.weight }))
                .filter((l) => l.id && l.weight > 0);
              return active.length ? active : undefined;
            })(),
            // ControlNet: sent only when a model + control image are chosen; v1 conditions plain text-to-image only.
            controlnet:
              controlnetCapable && controlnetId && controlImage && workflow === "create"
                ? {
                    id: controlnetId,
                    image: controlImage,
                    control_type: controlType,
                    strength: controlStrength,
                  }
                : undefined,
          });
        } catch (err) {
          // The POST response was lost while the backend kept generating (the secure-mode tunnel caps
          // the response near 100s). Retrying would duplicate the work, so wait the run out.
          if (!(err instanceof GenerateResponseLostError)) throw err;
          // A record outside the baseline proves the request reached the backend. Taken per attempt, so
          // it reflects what the client could see when THIS post went out.
          await settleLostGeneration(() => isMounted.current, probeBaseline);
          if (!isMounted.current) break;
          await loadGallery();
          // loadGallery refreshes the module cache synchronously, so this run's records are folded in
          // before the next run.
          galleryCache.images.forEach((image) => knownIds.add(image.id));
          setGenDone(i + 1);
          continue;
        }
        if (!isMounted.current) break;
        // Merge this run's records and load their blobs. Sorted, not prepended: a new image is
        // unpinned, so the server puts it after the pinned group.
        stripEpoch.current += 1;
        // Deduplicated: a resync in flight can fetch the saved record first, and prepending it again
        // duplicates a React key and inflates the next page's offset.
        setImages((prev) => mergeGenerated(prev, res.images));
        res.images.forEach((image) => knownIds.add(image.id));
        if (res.images[0]) setSelectedId(res.images[0].id);
        res.images.forEach((image) => void ensureSrc(image));
        setGenDone(i + 1);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Image generation failed";
      // The user's own Stop comes back as the backend's cancelled sentinel (409), so it is not
      // toasted. Only a Stop the backend confirmed explains an error away: a POST that never
      // landed, or {cancelled: false}, means whatever it raised is a real failure.
      if (
        shouldReportGenerateError({
          message: msg,
          stopRequested: cancelRequested.current && cancelAcked.current,
        })
      )
        toast.error(msg);
    } finally {
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      genPollTimer.current = null;
      if (genVisibilityListener.current) {
        document.removeEventListener("visibilitychange", genVisibilityListener.current);
        genVisibilityListener.current = null;
      }
      cancelRequested.current = false;
      // Refresh on EVERY exit, not just the successful one, and AWAIT it before Generate comes back:
      // a generation can change server-side status and a cancelled native run can leave no model
      // at all, so re-enabling first would offer a button that 409s.
      // Speed=Auto compiles on the 3rd LoRA-free run and supports_lora flips false.
      if (isMounted.current) await refreshStatus();
      setBusy(null);
      setGenDone(null);
      setGenStep(null);
    }
  }, [prompt, negativePrompt, width, height, steps, guidance, seed, batchSize, count, workflow, initImage, maskImage, strength, extendPct, extendSides, upscaleFactor, upscaleStrength, referenceImages, loras, loraCapable, controlnetCapable, controlnetId, controlImage, controlType, controlStrength, ensureSrc, loadGallery, refreshStatus]);

  // Stop the in-flight generation. Latch FIRST, so a multi-run request stops even if the POST
  // races the run that is already finishing.
  const handleCancelGenerate = useCallback(async () => {
    cancelRequested.current = true;
    // One Stop on the wire at a time. The button stays enabled so the click still latches, but a
    // second POST would target whatever is active when IT arrives.
    const token = runToken.current;
    if (cancelInFlight.current === token) return;
    cancelInFlight.current = token;
    const abort = new AbortController();
    cancelAbort.current = abort;
    try {
      const { cancelled } = await cancelDiffusionGeneration(abort.signal);
      cancelAcked.current = Boolean(cancelled);
    } catch {
      // An abort means the next run dropped this one on purpose, so there is nothing to report.
      // Otherwise the request never landed and the denoise runs on, so the click is not handled.
      if (!abort.signal.aborted) {
        cancelAcked.current = false;
        toast.error("Could not reach the server to stop this generation; it is still running");
      }
    } finally {
      // Only if they are still ours: a slow cancel from an earlier run must not release the guard a later run set.
      if (cancelInFlight.current === token) cancelInFlight.current = null;
      if (cancelAbort.current === abort) cancelAbort.current = null;
    }
  }, []);

  // Publish what the loaded model can do, so the sidebar submenu dims the rest. null while
  // nothing is loaded, which leaves every workflow open to set up first.
  useEffect(() => {
    if (!status?.loaded) {
      setSupported(null);
      return;
    }
    const wf = status.workflows ?? [];
    setSupported(
      WORKFLOW_TABS.filter((t) =>
        wf.includes(t.requires === null ? "txt2img" : t.requires),
      ).map((t) => t.id),
    );
  }, [status?.loaded, status?.workflows, setSupported]);

  // Keep the active workflow valid for the loaded model: snap to the first supported one when capabilities change.
  useEffect(() => {
    // Skipped in Train, which has no workflows and must not be snapped back to Create.
    if (supported === null || pageMode !== "create") return;
    if (!supported.includes(workflow) && supported[0]) {
      setWorkflow(supported[0]);
    }
  }, [supported, workflow, setWorkflow, pageMode]);

  const activeWorkflowTab =
    WORKFLOW_TABS.find((t) => t.id === workflow) ?? WORKFLOW_TABS[0];

  // The Advanced (load-time) tuning controls, rendered in the right-docked panel below.
  const advancedControls = (
    <>
      <AdvancedSelect
        label="Speed"
        hint="Auto picks per model: GGUF compiles at load; a dense model keeps the first two images exact and eager, then compiles from the 3rd (~2x from there). eager = fused kernels, no compile. default/max add torch.compile (max also TF32 + fused QKV)."
        badge={<ResolvedBadge status={status} controlKey="speed_mode" />}
        value={speedMode}
        onValueChange={(v) => setSpeedMode(v as typeof speedMode)}
        options={[
          ["auto", "Auto"],
          ["off", "Off (bit-exact)"],
          ["eager", "Eager"],
          ["default", "Default (compile)"],
          ["max", "Max"],
        ]}
      />
      {/* The dense transformer_quant fast path engages only on the GGUF kind, so gate the control to
          GGUF (or nothing loaded) and otherwise say why it is unavailable. */}
      {!status?.loaded || status.model_kind === "gguf" ? (
        <AdvancedSelect
          label="Precision"
          hint="How the model computes. Auto picks the fastest precision the hardware supports (at least INT8 on a capable GPU; FP8 on data-center cards) by loading the FULL base model and quantising its transformer onto low-precision tensor cores, and falls back to running the GGUF as-is when the device, VRAM or disk can't take it. Off always runs the GGUF as-is."
          badge={<ResolvedBadge status={status} controlKey="transformer_quant" />}
          value={transformerQuant}
          onValueChange={(v) => setTransformerQuant(v as typeof transformerQuant)}
          options={[
            ["auto", "Auto (fastest for GPU)"],
            ["none", "Off (run the GGUF)"],
            ["fp8", "FP8"],
            ["int8", "INT8"],
            ["nvfp4", "NVFP4 (Blackwell)"],
            ["mxfp8", "MXFP8 (Blackwell)"],
          ]}
        />
      ) : (
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
            Precision
          </span>
          <span className="text-xs text-muted-foreground/60">GGUF models only</span>
        </div>
      )}
      <AdvancedSelect
        label="Attention"
        hint="Attention kernel. Auto upgrades to cuDNN fused attention on NVIDIA when a speed profile is active. sage is INT8 attention: fast (10-40%) but can black-frame some families (Qwen, Wan), so it never engages automatically."
        badge={<ResolvedBadge status={status} controlKey="attention_backend" />}
        value={attentionBackend}
        onValueChange={(v) => setAttentionBackend(v as typeof attentionBackend)}
        options={[
          ["auto", "Auto"],
          ["native", "Native SDPA"],
          ["cudnn", "cuDNN"],
          ["flash3", "FlashAttention 3"],
          ["sage", "SageAttention (INT8)"],
        ]}
      />
      <AdvancedSelect
        label="Memory"
        hint="auto measures free VRAM. fast keeps everything resident. balanced streams the transformer. low_vram offloads every component (lowest VRAM, slower)."
        badge={<ResolvedBadge status={status} controlKey="memory_mode" />}
        value={memoryMode}
        onValueChange={(v) => setMemoryMode(v as typeof memoryMode)}
        options={[
          ["auto", "Auto"],
          ["fast", "Fast (resident)"],
          ["balanced", "Balanced"],
          ["low_vram", "Low VRAM"],
        ]}
      />
      {gpuChoices.length > 0 && (
        <AdvancedSelect
          label="GPU"
          hint="Which card this model loads on. Auto uses whichever device torch is pointing at, which on a mixed box is not necessarily the largest. An image model is never split across cards, so this is one choice, not a pool."
          value={selectedGpu}
          onValueChange={setSelectedGpu}
          options={[
            ["auto", "Auto"],
            ...gpuChoices.map(
              (d) =>
                [
                  String(d.index),
                  `GPU ${d.index}${d.memoryTotalGb ? ` · ${Math.round(d.memoryTotalGb)} GiB` : ""}`,
                ] as [string, string],
            ),
          ]}
        />
      )}
      <AdvancedSelect
        label="Step cache"
        hint="First-Block-Cache reuses the transformer tail across steps for many-step models (~1.4x). Auto turns it on at 20+ steps and off for few-step distilled models, re-checked per image."
        badge={<ResolvedBadge status={status} controlKey="transformer_cache" />}
        value={transformerCache}
        onValueChange={(v) => setTransformerCache(v as typeof transformerCache)}
        options={[
          ["auto", "Auto"],
          ["off", "Off"],
          ["fbcache", "First-Block-Cache"],
        ]}
      />
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
          CPU offload
          <InfoHint>Offload to CPU to fit low-VRAM cards (slower). Overridden by Memory mode when that is not Auto.</InfoHint>
          <ResolvedBadge status={status} controlKey="cpu_offload" />
        </span>
        <Switch checked={cpuOffload} onCheckedChange={setCpuOffload} />
      </div>
      <LoadedBuildSummary status={status} />
      {/* A resident full pipeline is reloadable by repo id alone, so it keeps Reapply even before a
          user-initiated load; GGUF/single_file residents hide the button. */}
      {status?.loaded && (canReapply || status?.model_kind === "pipeline") && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button
              variant="secondary"
              size="sm"
              disabled={busy !== null}
              onClick={handleReapply}
            >
              <HugeiconsIcon icon={ArrowReloadHorizontalIcon} className="mr-2 size-3.5" />
              Reapply to loaded model
            </Button>
          </TooltipTrigger>
          <TooltipContent>Reload the current model with these advanced options</TooltipContent>
        </Tooltip>
      )}
    </>
  );

  return (
    // The chat-style layout gives this page no outer top inset, so clear the custom titlebar here as chat does.
    // 34px on win/linux, 0 under macOS's native one.
    <div className="diffusion-surface @container flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]">
      {/* Keep the tabs centered over the preview at every width: the model rail holds at 408px when
          space permits and shrinks only to preserve the controls. */}
      <div className="pointer-events-none relative z-40 grid h-[48px] shrink-0 grid-cols-[minmax(0,408px)_minmax(13rem,1fr)]">
        <div
          className={cn(
            "pointer-events-none flex h-full min-w-0 items-start overflow-hidden @[50rem]:border-r @[50rem]:border-border/60",
            isMobile
              ? "pl-12"
              : !pinned && isTauri
                ? "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
                : "pl-[var(--studio-media-header-left-inset,1.5rem)]",
          )}
        >
          <div className="pointer-events-auto flex min-w-0 max-w-full items-center gap-2 overflow-hidden pt-[var(--studio-chat-header-padding-top,11px)]">
            {pageMode === "train" ? (
              <TrainBaseSelector
                families={trainFamilies}
                familyName={trainFamilyName}
                base={trainBaseChoice}
                onSelect={(family, repo) => {
                  setTrainFamilyName(family);
                  setTrainBaseChoice(repo);
                }}
              />
            ) : (
              <ModelSelector
                models={imageModels}
                value={status?.loaded ? status.repo_id ?? undefined : undefined}
                activeGgufVariant={quant}
                onValueChange={handleModelSelect}
                resolveDownloadFootprint={resolveDownloadFootprint}
                onEject={status?.loaded ? handleUnload : undefined}
                variant="ghost"
                className="!h-[34px] max-w-full gap-1 overflow-hidden pl-3 pr-1 @[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"
                triggerLabelClassName="text-ui-14 @[68rem]:text-ui-16"
                task={IMAGE_GEN_TASKS}
                catalog={IMAGE_CATALOG}
                placeholder="Select image model"
                open={active && selectorOpen}
                onOpenChange={(o) => setSelectorOpen(active && o)}
              />
            )}
            {pageMode !== "train" && busy === "loading" && (
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    aria-label="Cancel load"
                    className="!h-[34px] rounded-full text-xs"
                    onClick={() => void handleCancelLoad()}
                  >
                    Cancel load
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Stop loading this model</TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
        <div className="grid h-full min-w-0 grid-cols-[1fr_auto_auto] gap-2 @[50rem]:grid-cols-[1fr_auto_1fr] @[50rem]:gap-0">
          <div className="pointer-events-auto col-start-2 justify-self-center pt-[var(--studio-chat-header-padding-top,11px)]">
            <PillTabs
              ariaLabel="Page mode"
              value={pageMode}
              onValueChange={(v) => setPageMode(v as "create" | "train")}
              fit={true}
              className="h-[34px] [&>button]:h-[34px] [&>button]:px-3 @[68rem]:[&>button]:px-11"
              tabs={[
                { value: "create", label: "Create", icon: <HugeiconsIcon icon={SparklesIcon} className="size-3.5" /> },
                { value: "train", label: "Train", icon: <HugeiconsIcon icon={TestTubeOutlineIcon} className="size-3.5" /> },
              ]}
            />
          </div>
          <div className="pointer-events-none col-start-3 flex min-w-0 items-start justify-end pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
            <div className="pointer-events-auto flex min-w-0 items-center gap-2">
              <MediaPageLink
                to="/video"
                label="Video"
                icon={FlimSlateIcon}
                labelClassName="hidden @[50rem]:inline"
                arrowClassName="hidden @[50rem]:block"
              />
            </div>
          </div>
        </div>
      </div>
      {/* Train mode: the full-page training workspace. Unmounted in Create mode so its polling stops. */}
      {pageMode === "train" ? (
        <DiffusionTrainPanel
          active={active && pageMode === "train"}
          loadedFamily={status?.family ?? null}
          loadedBaseRepo={
            // Prefer base_repo (the full diffusers pipeline) over repo_id: for a GGUF load repo_id is a
            // checkpoint path, not a trainable base.
            status?.base_repo ?? status?.repo_id ?? null
          }
          onTrainingComplete={() => setLoraRefreshKey((k) => k + 1)}
          onDeploy={handleDeployAdapter}
          familyName={trainFamilyName}
          onFamilyNameChange={setTrainFamilyName}
          baseChoice={trainBaseChoice}
          onBaseChoiceChange={setTrainBaseChoice}
          onFamiliesChange={setTrainFamilies}
        />
      ) : (
      /* Settings column + preview canvas. Structural borders stay edge-to-edge; spacing belongs inside each pane.
         The same 50rem page-container breakpoint drives this body and the header above. */
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto overflow-x-hidden @[50rem]:flex-row @[50rem]:overflow-hidden">
        <div className="flex w-full shrink-0 flex-col border-b border-border/60 @[50rem]:w-[408px] @[50rem]:overflow-hidden @[50rem]:border-r @[50rem]:border-b-0">
          {/* pl-0.5 keeps focus rings off the scroll container's edge. */}
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar panel-scroll-fade-action flex min-h-0 flex-1 flex-col gap-4 px-10 pt-9 pb-6 @[50rem]:overflow-y-auto",
              settingsFadeClass,
            )}
          >
            {/* The sidebar submenu is the switcher, so name the active workflow over its controls. */}
            {/* Icon rides the heading; the line below runs the full width. Same shape on the Video page. */}
            <div className="mb-2 flex items-start justify-between gap-3">
              <div className="min-w-0 grid gap-1.5">
                <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none text-foreground">
                  {/* Same icon the sidebar submenu uses for this workflow. */}
                  <HugeiconsIcon
                    icon={activeWorkflowTab.icon}
                    className="size-[18px] shrink-0"
                  />
                  {activeWorkflowTab.heading ?? activeWorkflowTab.label}
                </h2>
                <p className="text-xs leading-snug text-muted-foreground">
                  {activeWorkflowTab.hint}
                </p>
              </div>

              {workflow === "create" && (
                <MediaGenerationPresetControl
                  kind="image"
                  presets={imagePresets.presets}
                  activePreset={imagePresets.activePreset}
                  ready={imagePresets.presetsReady}
                  hasUnsavedChanges={imagePresets.hasUnsavedChanges}
                  onSelect={imagePresets.selectPreset}
                  onSave={imagePresets.savePreset}
                  onDelete={imagePresets.deletePreset}
                />
              )}
            </div>

            {workflow === "transform" && (
              <>
                <Field
                  label="Source image"
                  hint="The image to transform. Generation redraws it guided by your prompt; the Strength below controls how far."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} />
                </Field>
                <SliderField
                  label="Strength"
                  hint="How much to redraw the source. Low keeps the original composition; high reimagines it from the prompt."
                  value={strength}
                  min={0.1}
                  max={1}
                  step={0.05}
                  onChange={setStrength}
                />
              </>
            )}

            {workflow === "inpaint" && (
              <>
                {!initImage ? (
                  <Field
                    label="Source image"
                    hint="The image to edit. After uploading, paint over the area you want to regenerate; the rest is kept."
                  >
                    <ImageDropzone value={null} onChange={handleInitChange} />
                  </Field>
                ) : (
                  <>
                    <Field
                      label="Mask"
                      hint="Brush over the region to regenerate (shown in red). Those pixels are repainted from your prompt; everything else is preserved."
                    >
                      <MaskCanvas
                        image={initImage}
                        brushPct={brushPct}
                        resetKey={maskResetKey}
                        onMaskChange={setMaskImage}
                      />
                    </Field>
                    <SliderField
                      label="Brush size"
                      hint="Brush radius as a percent of the image's shorter side."
                      value={brushPct}
                      min={2}
                      max={25}
                      step={1}
                      onChange={setBrushPct}
                    />
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="flex-1"
                        onClick={() => {
                          setMaskImage(null);
                          setMaskResetKey((k) => k + 1);
                        }}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                        Clear mask
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="flex-1"
                        onClick={() => handleInitChange(null)}
                      >
                        <HugeiconsIcon icon={ImageAdd02Icon} className="size-3.5" />
                        Replace image
                      </Button>
                    </div>
                    <SliderField
                      label="Strength"
                      hint="How much to redraw the masked region. Low blends with the source; high fully reimagines it from the prompt."
                      value={strength}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onChange={setStrength}
                    />
                  </>
                )}
              </>
            )}

            {workflow === "extend" && (
              <>
                <Field
                  label="Source image"
                  hint="The image to outpaint. The canvas grows on the selected sides and the new area is filled from your prompt; the original is kept."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} />
                </Field>
                <SliderField
                  label="Expand by"
                  hint="How far to grow each selected side, as a percent of the image's size."
                  value={extendPct}
                  min={10}
                  max={100}
                  step={5}
                  onChange={setExtendPct}
                />
                <Field label="Sides" hint="Which edges to extend.">
                  <div className="grid grid-cols-2 gap-1.5">
                    {(
                      [
                        ["top", "Top"],
                        ["bottom", "Bottom"],
                        ["left", "Left"],
                        ["right", "Right"],
                      ] as Array<[keyof ExtendSides, string]>
                    ).map(([key, label]) => {
                      const on = extendSides[key];
                      return (
                        <button
                          key={key}
                          type="button"
                          aria-pressed={on}
                          onClick={() => setExtendSides((s) => ({ ...s, [key]: !s[key] }))}
                          className={cn(
                            // No border in either mode: the fill alone marks the state, and a ring would not survive
                            // mouse focus anyway.
                            "rounded-lg px-2 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                            on
                              ? "bg-primary/15 text-foreground hover:bg-primary/20 dark:bg-primary/25 dark:hover:bg-primary/30"
                              : "bg-muted text-muted-foreground hover:bg-muted/70 hover:text-foreground dark:bg-white/[0.06] dark:hover:bg-white/[0.1]",
                          )}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </Field>
              </>
            )}

            {workflow === "upscale" && (
              <>
                <Field
                  label="Source image"
                  hint="The image to upscale. It is enlarged by the factor below, then re-detailed at higher resolution guided by your prompt; keep the prompt describing the same content."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} />
                </Field>
                <SliderField
                  label="Scale"
                  hint="How much larger to make the image. The output size is the source size times this factor (capped and rounded to a multiple of 16)."
                  value={upscaleFactor}
                  min={1.5}
                  max={4}
                  step={0.5}
                  onChange={setUpscaleFactor}
                />
                <SliderField
                  label="Detail strength"
                  hint="How much new detail to add while upscaling. Low keeps the image faithful to the source; high adds more (and may drift). 0.35 is a good hires-fix default."
                  value={upscaleStrength}
                  min={0.1}
                  max={0.6}
                  step={0.05}
                  onChange={setUpscaleStrength}
                />
              </>
            )}

            {workflow === "reference" && (
              <>
                <Field
                  label="Reference image"
                  hint="A reference the model draws on (subject, style, or composition) while generating a NEW image from your prompt at the size below. Unlike Transform, it is not a redraw of this image, so there is no strength."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} />
                </Field>
                {referenceImages.map((img, i) => (
                  <Field
                    key={i}
                    label={`Reference ${i + 2}`}
                    hint="An extra reference combined with the others (e.g. one for the subject, one for the style)."
                  >
                    <div className="space-y-1.5">
                      <ImageDropzone
                        value={img}
                        onChange={(v) =>
                          // Keep the slot in place (empty string when cleared) so other slots do not renumber mid-edit;
                          // empty slots are dropped at send time.
                          setReferenceImages((prev) =>
                            prev.map((p, j) => (j === i ? (v ?? "") : p)),
                          )
                        }
                      />
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="w-full"
                        onClick={() => setReferenceImages((prev) => prev.filter((_, j) => j !== i))}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                        Remove reference {i + 2}
                      </Button>
                    </div>
                  </Field>
                ))}
                {referenceImages.length < 3 && (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="w-full"
                    disabled={!initImage}
                    onClick={() => setReferenceImages((prev) => [...prev, ""])}
                  >
                    <HugeiconsIcon icon={ImageAdd02Icon} className="size-3.5" />
                    Add another reference
                  </Button>
                )}
              </>
            )}

            {workflow === "edit" && (
              <Field
                label="Source image"
                hint="The image to edit. Describe the change in the prompt below (e.g. 'make it night', 'add a red hat', 'change the background to a beach')."
              >
                <ImageDropzone value={initImage} onChange={handleInitChange} />
              </Field>
            )}

            <Field label={workflow === "edit" ? "Instruction" : "Prompt"}>
              <Textarea
                rows={4}
                placeholder={
                  workflow === "edit" ? "Describe the edit, e.g. make the sky sunset orange" : undefined
                }
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </Field>
            <NegativePromptField
              value={negativePrompt}
              onChange={setNegativePrompt}
              open={negativeOpen}
              onOpenChange={setNegativeOpen}
              hint="What to steer the image away from. Only used when guidance is above 0."
            />
            {/* LoRA adapters: shown whenever the loaded model + quant can apply them. Each carries a 0-2 weight. */}
            {loraCapable && (
              <Field
                label="LoRAs"
                hint="Style or character adapters applied on top of the model. Enter a Hugging Face repo id (or pick a suggestion) and set the strength (1.0 = full effect, 0 disables). Stack several."
              >
                <div className="space-y-2">
                  {availableLoras.length > 0 && (
                    <datalist id="diffusion-lora-suggestions">
                      {availableLoras.map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.display_name}
                        </option>
                      ))}
                    </datalist>
                  )}
                  {loras.map((sel, i) => (
                    <div
                      // Key on the index, not sel.id: sel.id is the editable repo-id Input's value, so keying on it
                      // remounts the row and drops focus on the first character typed. The list is
                      // index-addressed and rows are removed explicitly.
                      key={i}
                      className="space-y-1.5 rounded-lg border border-border bg-muted/30 p-2"
                    >
                      <div className="flex items-center gap-2">
                        <Input
                          value={sel.id}
                          list={availableLoras.length > 0 ? "diffusion-lora-suggestions" : undefined}
                          placeholder="owner/name or owner/name:file.safetensors"
                          spellCheck={false}
                          autoCapitalize="none"
                          autoCorrect="off"
                          className="h-8 flex-1 text-xs"
                          onChange={(e) =>
                            setLoras((prev) =>
                              prev.map((p, j) => (j === i ? { ...p, id: e.target.value } : p)),
                            )
                          }
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="size-8 shrink-0"
                          aria-label={`Remove LoRA ${i + 1}`}
                          onClick={() => setLoras((prev) => prev.filter((_, j) => j !== i))}
                        >
                          <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                        </Button>
                      </div>
                      <SliderField
                        label="Weight"
                        value={sel.weight}
                        min={0}
                        max={2}
                        step={0.05}
                        onChange={(v) =>
                          setLoras((prev) => prev.map((p, j) => (j === i ? { ...p, weight: v } : p)))
                        }
                      />
                    </div>
                  ))}
                  {loras.length < 8 && (
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      className="w-full"
                      onClick={() => {
                        // Prefill with the first unused suggestion when a curated catalog exists, else an empty row.
                        const taken = new Set(loras.map((l) => l.id));
                        const next = availableLoras.find((a) => !taken.has(a.id));
                        setLoras((prev) => [
                          ...prev,
                          next ? { id: next.id, weight: next.weight_default || 1 } : { id: "", weight: 1 },
                        ]);
                      }}
                    >
                      <HugeiconsIcon icon={ImageAdd02Icon} className="size-3.5" />
                      Add LoRA
                    </Button>
                  )}
                </div>
              </Field>
            )}
            {/* ControlNet: shown when the model supports it, one is discoverable, and txt2img is active. */}
            {controlnetCapable && availableControlNets.length > 0 && workflow === "create" && (
              <Field
                label="ControlNet"
                hint="Condition the image on a control map (edges / depth / pose). Union models cover many types. Use 'Canny' to trace edges from your image, or 'Passthrough' if it is already a control map."
              >
                <div className="space-y-2 rounded-lg border border-border bg-muted/30 p-2">
                  <Select value={controlnetId || undefined} onValueChange={setControlnetId}>
                    <SelectTrigger className="h-8 w-full text-xs">
                      <SelectValue placeholder="Select a ControlNet" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableControlNets.map((c) => (
                        <SelectItem key={c.id} value={c.id}>
                          {c.display_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {controlnetId && (
                    <>
                      <ImageDropzone value={controlImage} onChange={setControlImage} />
                      <div className="flex items-center gap-2">
                        <span className="shrink-0 text-xs text-muted-foreground">Control type</span>
                        <Select value={controlType} onValueChange={setControlType}>
                          <SelectTrigger className="h-8 flex-1 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {controlTypeOptions.map((t) => (
                              <SelectItem key={t} value={t}>
                                {CONTROL_TYPE_LABELS[t] ??
                                  `${t.charAt(0).toUpperCase()}${t.slice(1)} (map)`}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <SliderField
                        label="Strength"
                        value={controlStrength}
                        min={0}
                        max={2}
                        step={0.05}
                        onChange={setControlStrength}
                      />
                    </>
                  )}
                </div>
              </Field>
            )}
            <Field
              label="Aspect ratio"
              hint="Pick a ratio to lock the proportions, then set the size below. Flip swaps width and height."
            >
              <div className="flex items-center gap-2">
                <Select
                  value={aspect}
                  onValueChange={changeAspect}
                  open={active && aspectOpen}
                  onOpenChange={(o) => setAspectOpen(active && o)}
                >
                  <SelectTrigger className="flex-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {ASPECT_OPTIONS.map((key) => (
                      <SelectItem key={key} value={key}>
                        {key === "custom"
                          ? "Custom"
                          : `${ASPECT_LABELS[key]} (${key})`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <Button
                      type="button"
                      variant="secondary"
                      size="icon"
                      aria-label="Flip width and height"
                      onClick={flipDimensions}
                    >
                      {/* Arrows turn with the orientation, showing which way it flips. */}
                      <HugeiconsIcon
                        icon={ArrowLeftRightIcon}
                        className={cn(
                          "size-4 transition-transform duration-200",
                          portrait && "rotate-90",
                        )}
                      />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {portrait ? "Switch to landscape" : "Switch to portrait"}
                  </TooltipContent>
                </Tooltip>
              </div>
            </Field>
            <Field
              label="Resolution"
              hint={
                // Image-conditioned workflows size from the source, so "this is the output size" is wrong
                // there: Transform caps the source by this box, the rest ignore it.
                workflow === "transform"
                  ? "Caps the output size. The source image is scaled down to fit inside this box, keeping its aspect ratio, so the result may be smaller than the values shown."
                  : workflow === "inpaint" ||
                      workflow === "extend" ||
                      workflow === "upscale" ||
                      workflow === "edit"
                    ? "Not used by this workflow: the output size comes from the source image. Upload a smaller image to generate at a smaller size."
                    : "Width and height in pixels. Sizes run from 256 to 2048 in steps of 16. Z-Image is trained around 1 megapixel, so much larger sizes can look worse."
              }
            >
              <div className="flex items-center gap-2">
                <DimensionSelect
                  icon={ArrowLeftRightIcon}
                  label="Width"
                  value={width}
                  open={active && widthOpen}
                  onOpenChange={(o) => setWidthOpen(active && o)}
                  onChange={changeWidth}
                />
                <DimensionSelect
                  icon={ArrowUpDownIcon}
                  label="Height"
                  value={height}
                  open={active && heightOpen}
                  onOpenChange={(o) => setHeightOpen(active && o)}
                  onChange={changeHeight}
                />
              </div>
            </Field>

            {/* First of the one-line sliders, so it takes a bigger break than the gap gives. */}
            <div className="pt-2">
              <SliderField
                label="Steps"
                hint="9 is the recommended setting for Z-Image-Turbo. More steps rarely help."
                value={steps}
                min={1}
                max={50}
                step={1}
                onChange={setSteps}
              />
            </div>
            <SliderField
              label="Guidance"
              hint="Keep this at 0 for Z-Image-Turbo. Higher values make its output worse. Other models use guidance."
              value={guidance}
              min={0}
              max={15}
              step={0.5}
              onChange={setGuidance}
            />
            <SliderField
              label="Batch size"
              hint="How many images to make at once. Faster than running them one by one, but uses more VRAM. They share a seed but each one is different."
              value={batchSize}
              min={1}
              max={32}
              step={1}
              onChange={setBatchSize}
            />
            <SliderField
              label="Runs"
              hint="How many times to repeat the generation, one after another. Each run uses the next seed, so the images differ and can be reproduced."
              value={count}
              min={1}
              max={RUNS_SLIDER_MAX}
              step={1}
              onChange={setCount}
            />
            {/* A slider row ends flush with its track, so the label below needs room. */}
            <Field
              label="Seed"
              hint="Leave empty for a fresh random seed each run."
              className="pt-2"
            >
              <Input
                placeholder="Random if empty"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </Field>

            <AdvancedDisclosure open={advancedOpen} onOpenChange={setAdvancedOpen}>
              {advancedControls}
            </AdvancedDisclosure>

          </div>
          {/* The scroll mask provides the fade; leave the footer unpainted to avoid dark-mode banding. */}
          <div className="relative z-10 flex shrink-0 justify-center px-10 pt-0.5 pb-4">
            {busy === "generating" ? (
              /* Replaces Generate while a run is in flight. Every workflow funnels through the same
                 handler, so one control stops all of them. */
              <Button
                className="relative z-10 h-11 px-8 hover:bg-muted dark:hover:bg-muted"
                variant="outline"
                onClick={handleCancelGenerate}
              >
                <Spinner className="mr-2 size-4" />
                {genDone != null && count > 1 ? `Stop (${genDone}/${count})` : "Stop"}
              </Button>
            ) : (
              <Button
                className="relative z-10 h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                onClick={handleGenerate}
                disabled={busy !== null || !status?.loaded}
              >
                Generate
              </Button>
            )}
          </div>
        </div>

        <div className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden @[50rem]:min-h-0">
          <div className="hover-scrollbar relative flex flex-1 items-center justify-center overflow-auto p-6 px-10 @[50rem]:pt-[60px]">
            {selected && selectedSrc ? (
              <>
                <img
                  src={selectedSrc}
                  alt={selected.prompt}
                  className="max-h-full max-w-full object-contain shadow-sm"
                />
                {/* Actions grouped in one glass toolbar so they stay legible over any image. Size and seed
                    live in the Recipe popover. */}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-xl bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur">
                  <RecipePopover image={selected} onRestore={restoreSettings} active={active} />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild={true}>
                      <Button size="sm" variant="ghost" className="gap-1.5">
                        <HugeiconsIcon icon={Download01Icon} className="size-4" />
                        Download
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => void downloadImage(selectedSrc, selected, "png")}
                      >
                        PNG (original, keeps recipe)
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => void downloadImage(selectedSrc, selected, "jpeg")}
                      >
                        JPEG (smaller)
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => void downloadImage(selectedSrc, selected, "webp")}
                      >
                        WebP
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <GalleryItemMenu
                    noun="image"
                    active={active}
                    pinned={Boolean(selected.pinned)}
                    archived={Boolean(selected.archived)}
                    onTogglePin={() =>
                      void handleTogglePin(selected.id, !selected.pinned)
                    }
                    onToggleArchive={() => void handleArchive(selected.id)}
                    onDelete={() => void handleDelete(selected.id)}
                  />
                </div>
              </>
            ) : selected ? (
              // The selected record's blob is still loading; spin in place.
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Spinner className="size-8" />
                <p className="text-sm">Loading…</p>
              </div>
            ) : busy === "generating" ? null : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                {/* Same icon as the Images nav item. */}
                <HugeiconsIcon icon={Image03Icon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a diffusion model to load"}
                </p>
              </div>
            )}

            {/* Live generation progress: a per-step bar with ETA, centered when there is nothing else to show. */}
            {busy === "generating" && (
              <div
                className={cn(
                  "pointer-events-none absolute flex justify-center px-4",
                  selectedSrc ? "inset-x-0 bottom-4" : "inset-0 items-center",
                )}
              >
                <div className="w-72 max-w-full rounded-xl bg-background/85 p-3 shadow-lg ring-1 ring-border backdrop-blur">
                  <ModelLoadDescription
                    // Drop the chat min-height: this floating card has no layout to stabilise.
                    className="min-h-0"
                    title={
                      genDone != null && count > 1
                        ? `Run ${genDone + 1}/${count}`
                        : null
                    }
                    message="Starting…"
                    progressPercent={genStep ? genStep.fraction * 100 : null}
                    progressLabel={genStep ? genStepLabel(genStep) : null}
                  />
                </div>
              </div>
            )}
          </div>

          {(images.length > 0 || busy === "generating") && (
            <div
              ref={stripRef}
              // The rule spans the pane; only the thumbnail contents receive the 40px gutter.
              className="hover-scrollbar flex shrink-0 gap-2 overflow-x-auto border-t border-foreground/10 px-10 py-3"
              onScroll={(e) => {
                // Near the right edge: pull the next older page (infinite scroll).
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past images stay browsable while
                  the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center rounded-lg bg-muted/50 ring-2 ring-primary/30">
                  <Spinner className="size-5 text-muted-foreground" />
                </div>
              )}
              {/* The tile is a wrapper, not a button: the actions menu must be the select button's SIBLING,
                  since a button inside a button is invalid. data-image-id rides the wrapper so the
                  observer still sees it. */}
              {images.map((image) => (
                <div
                  key={image.id}
                  data-image-id={image.id}
                  className="group relative size-16 shrink-0"
                >
                  <button
                    type="button"
                    onClick={() => setSelectedId(image.id)}
                    className="relative size-full overflow-hidden rounded-[10px] bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {srcById[image.id] ? (
                      <img
                        src={srcById[image.id]}
                        alt={image.prompt}
                        className="size-full object-cover"
                      />
                    ) : (
                      <span className="flex size-full items-center justify-center">
                        <Spinner className="size-4 text-muted-foreground" />
                      </span>
                    )}
                    {/* Selection marker on a non-focusable overlay, so the button's focus state cannot mask it. */}
                    {image.id === selected?.id && (
                      <span className="pointer-events-none absolute inset-0 rounded-[10px] border border-border bg-white/35 dark:border-white/25 dark:bg-white/20" />
                    )}
                  </button>
                  {/* Pin marker, bottom-left so it never sits under the menu. */}
                  {image.pinned && (
                    <span className="pointer-events-none absolute bottom-0.5 left-0.5 rounded-full bg-background/80 p-0.5 text-foreground shadow-sm ring-1 ring-border backdrop-blur">
                      <HugeiconsIcon icon={PinIcon} className="size-3" />
                    </span>
                  )}
                  <div className="absolute right-0.5 top-0.5">
                    <GalleryItemMenu
                      variant="overlay"
                      noun="image"
                      active={active}
                      pinned={Boolean(image.pinned)}
                      archived={Boolean(image.archived)}
                      onTogglePin={() => void handleTogglePin(image.id, !image.pinned)}
                      onToggleArchive={() => void handleArchive(image.id)}
                      onDelete={() => void handleDelete(image.id)}
                    />
                  </div>
                </div>
              ))}
              {/* Tail spinner while older pages stream in on scroll. */}
              {hasMore && (
                <div className="flex size-16 shrink-0 items-center justify-center">
                  <Spinner className="size-4 text-muted-foreground" />
                </div>
              )}
            </div>
          )}
        </div>

      </div>
      )}
    </div>
  );
}
