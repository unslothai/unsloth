// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowDown01Icon,
  ArrowExpand01Icon,
  ArrowLeftRightIcon,
  ArrowReloadHorizontalIcon,
  CheckmarkCircle02Icon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  Image03Icon,
  ImageAdd02Icon,
  ImageUpload01Icon,
  InformationCircleIcon,
  MagicWand01Icon,
  LayoutAlignRightIcon,
  PaintBrush02Icon,
  PencilEdit02Icon,
  Settings02Icon,
  ZoomInAreaIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
// Raw Radix trigger: the app's adds a click-to-toggle, wrong inside a menu row.
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";

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
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InfoHint } from "@/components/ui/info-hint";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { IMAGE_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import {
  IMAGE_CATALOG,
  catalogToModelOptions,
  loadSpecFor,
} from "@/features/model-picker/components/model-selector/model-catalog";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { ParamSlider } from "@/features/chat";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { formatBytes, formatEta } from "@/features/hub/lib/format";
import { cn } from "@/lib/utils";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { diffusionRoutePick } from "@/lib/diffusion-route-pick";
import { toast } from "@/lib/toast";

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
  deleteGalleryImage,
  fetchGalleryObjectUrl,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  getGallery,
  getGenerateProgress,
  listDiffusionControlNets,
  listDiffusionLoras,
  getDiffusionDownloadPlan,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useStagedDownload } from "@/features/hub/download-manager";
import { DiffusionTrainPanel } from "./train/diffusion-train-panel";
import {
  TrainBaseSelector,
  type TrainFamilyOption,
} from "./train/train-base-selector";

// Curated models come from the shared catalog: one canonical group per model with its artifacts (GGUF / FP8 / bnb-4bit / BF16) as data, and the load kind per artifact via loadSpecFor.
// The picker renders groups with a format second level and routes bare clicks to the best artifact for the device.
const MODELS: ModelOption[] = catalogToModelOptions(IMAGE_CATALOG);

// Workflow tabs. `requires` is the backend workflow id (status.workflows) the loaded model must support for the tab to enable; null = always available.
type WorkflowId = "create" | "transform" | "inpaint" | "extend" | "upscale" | "reference" | "edit";

const WORKFLOW_TABS: Array<{
  id: WorkflowId;
  label: string;
  requires: string | null;
  icon: typeof PencilEdit02Icon;
  hint?: string;
}> = [
  {
    id: "create",
    label: "Create",
    requires: null,
    icon: PencilEdit02Icon,
    hint: "Generate a new image from a prompt",
  },
  {
    id: "transform",
    label: "Transform",
    icon: MagicWand01Icon,
    requires: "img2img",
    hint: "Redraw an image from your prompt",
  },
  {
    id: "inpaint",
    label: "Inpaint",
    icon: PaintBrush02Icon,
    requires: "inpaint",
    hint: "Regenerate a painted region",
  },
  {
    id: "extend",
    label: "Extend",
    icon: ArrowExpand01Icon,
    requires: "outpaint",
    hint: "Grow the canvas and fill the edges",
  },
  {
    id: "upscale",
    label: "Upscale",
    icon: ZoomInAreaIcon,
    requires: "upscale",
    hint: "Enlarge and re-detail an image",
  },
  {
    id: "reference",
    label: "Reference",
    icon: ImageUpload01Icon,
    requires: "reference",
    hint: "Generate guided by a reference image",
  },
  {
    id: "edit",
    label: "Edit",
    icon: Edit03Icon,
    requires: "edit",
    hint: "Change an image with an instruction",
  },
];

// Per-model generation defaults (steps + guidance), matched by repo-id substring, most specific first: distilled turbo/schnell want few steps and little guidance, full "dev" models more steps and real CFG.
// The images each conditioned workflow consumed, named for the restore toast: a recipe keeps the scalar settings but not the uploads, so restoring one lands on Create and the user must know which input to add back.
// Keys are the backend own workflow strings (diffusion.py); txt2img is absent because it restores completely.
const CONDITIONED_WORKFLOW_INPUTS: Record<string, string> = {
  img2img: "the source image",
  inpaint: "the source image and mask",
  upscale: "the source image",
  edit: "the source image",
  reference: "the source and reference images",
  controlnet: "the control image",
};

// Generation defaults when the model is unrecognised: the distilled few-step / no-CFG shape. Also seeds the sliders.
const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  { match: "z-image-turbo", steps: 9, guidance: 0 },
  // Krea 2 Raw is the undistilled base (also inference-loadable): 52 steps at guidance 3.5, so it must precede the distilled "krea-2" key below.
  { match: "krea-2-raw", steps: 52, guidance: 3.5 },
  // Krea 2 Turbo is distilled (TDM): 8 steps, no CFG. "krea-2" covers Turbo and any other krea id but Raw, matched above.
  { match: "krea-2", steps: 8, guidance: 0 },
  { match: "flux.1-schnell", steps: 4, guidance: 0 },
  // Kontext (editing) before the generic flux.1: ~28 steps, lower guidance (~2.5).
  { match: "kontext", steps: 28, guidance: 2.5 },
  // Krea FLUX.1-dev finetune runs its card recipe (28 steps, guidance 4.5); before the generic flux.1 key, and it never hits the krea-2 keys.
  { match: "flux.1-krea", steps: 28, guidance: 4.5 },
  { match: "flux.1", steps: 28, guidance: 3.5 },
  { match: "flux.2-klein", steps: 4, guidance: 0 },
  // FLUX.2-dev is the full (non-distilled) model: more steps + real guidance, unlike klein.
  { match: "flux.2-dev", steps: 28, guidance: 4 },
  { match: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", steps: 20, guidance: 4 },
  // Ideogram 4 model-card settings (48 steps, guidance 7). At exactly these the backend keeps the pipeline recommended tapered guidance schedule instead of a flat constant.
  { match: "ideogram", steps: 48, guidance: 7 },
  // Lumina Image 2.0 model-card recipe (the backend adds cfg_trunc_ratio itself).
  { match: "lumina", steps: 50, guidance: 4 },
  // HunyuanImage 2.1: 50 steps; guidance feeds distilled_guidance_scale (default 3.25), real CFG runs in the repo guider components.
  { match: "hunyuanimage", steps: 50, guidance: 3.25 },
  // HiDream-I1: Full runs 50 steps at guidance 5; the Dev/Fast distillations are guidance-free. Specific keys before the generic "hidream".
  { match: "hidream-i1-dev", steps: 28, guidance: 0 },
  { match: "hidream-i1-fast", steps: 16, guidance: 0 },
  { match: "hidream", steps: 50, guidance: 5 },
  // SDXL: Turbo is distilled (few steps, no CFG), base wants ~30 steps and real CFG (~7); "sdxl-turbo" must precede the generic "sdxl".
  { match: "sdxl-turbo", steps: 3, guidance: 0 },
  { match: "stable-diffusion-xl", steps: 30, guidance: 7 },
  { match: "sdxl", steps: 30, guidance: 7 },
];

function defaultsFor(repoId: string): { steps: number; guidance: number } {
  const id = repoId.toLowerCase();
  // The fallback is only hit for an unrecognised on-device image GGUF; a curated entry covers every model in MODELS.
  return MODEL_DEFAULTS.find((d) => id.includes(d.match)) ?? DEFAULT_GEN;
}

// Common aspect ratios (landscape; Flip gives the portrait mirror). Picking one locks the W:H proportion; the sliders set the size.
const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};
const ASPECT_OPTIONS = ["custom", ...Object.keys(ASPECT_RATIOS)];

// Friendly labels for ControlNet control types: "canny" traces edges from a source image, every other type is an already-made map. Unknown types fall back to a capitalized "(map)" label.
const CONTROL_TYPE_LABELS: Record<string, string> = {
  passthrough: "Passthrough (already a map)",
  canny: "Canny (trace edges)",
  depth: "Depth (map)",
  pose: "Pose (map)",
};

// Z-Image accepts 256–2048, in multiples of 16. Snap any value into range.
const MIN_DIM = 256;
const MAX_DIM = 2048;
// Convenient drag range for the Runs slider. The number box accepts higher typed values on purpose (set it large to generate all night); the loop only floors at 1 and ignores non-numeric input.
const RUNS_SLIDER_MAX = 128;
function snapDim(value: number): number {
  if (!Number.isFinite(value)) return 1024;
  return Math.min(MAX_DIM, Math.max(MIN_DIM, Math.round(value / 16) * 16));
}

// The ratio key (compared by long:short, so it survives orientation) matching width/height, plus whether the orientation is portrait.
function matchAspect(width: number, height: number): { key: string; portrait: boolean } {
  const target = Math.max(width, height) / Math.min(width, height);
  const found = Object.entries(ASPECT_RATIOS).find(
    ([, [a, b]]) => Math.abs(target - a / b) < 0.01,
  );
  return { key: found ? found[0] : "custom", portrait: height > width };
}

// Module cache of the backend-persisted gallery, so a tab switch re-renders instantly. Object URLs are revoked only on delete (not unmount), so they stay valid across remounts.
// Blob budget for cached gallery PNGs. A 1024x1024 PNG is ~1-2 MB and the page stays mounted after its first visit, so an unbounded cache grew all session as the user scrolled.
// 192 MB is ~100-200 images: far more than any viewport holds, while capping what the webview can be made to pin. On-screen and open-in-the-viewer images are never evicted.
const IMAGE_BLOB_BUDGET_BYTES = 192 * 1024 * 1024;

const galleryCache: {
  images: GalleryImage[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  srcById: BlobUrlCache;
  // Ids with a fetch in flight, so concurrent ensureSrc calls do not double-fetch (and leak the duplicate object URL).
  inflight: Set<string>;
  // Ids deleted while their PNG was still downloading: the delete revoked whatever URL existed, so a fetch landing afterwards must throw its blob away rather than cache one no card can release.
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

// Export filename, e.g. Unsloth_20260624-143005_123.png. Batch siblings share the seed + timestamp, so they get a "_<n>" suffix past the first one.
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

function saveBlobUrl(href: string, filename: string) {
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  link.click();
}

// PNG saves the stored bytes verbatim (keeping the embedded recipe metadata); JPEG / WebP re-encode client-side from the already-fetched object URL, and JPEG is flattened onto white (no alpha).
async function downloadImage(
  src: string,
  image: GalleryImage,
  format: ImageExportFormat = "png",
) {
  if (format === "png") {
    saveBlobUrl(src, exportFilename(image, format));
    return;
  }
  try {
    const el = new Image();
    el.decoding = "async";
    el.src = src;
    await el.decode();
    const canvas = document.createElement("canvas");
    canvas.width = el.naturalWidth;
    canvas.height = el.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2d context unavailable");
    if (format === "jpeg") {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.drawImage(el, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, `image/${format}`, 0.95),
    );
    if (!blob) throw new Error(`could not encode ${format}`);
    const url = URL.createObjectURL(blob);
    try {
      saveBlobUrl(url, exportFilename(image, format));
    } finally {
      // Give the click a tick to start before revoking.
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    }
  } catch {
    // Conversion failed (decode/encode); fall back to the original PNG bytes.
    saveBlobUrl(src, exportFilename(image, "png"));
  }
}

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

// Bar label for an in-flight generation: step count plus an ETA once known (formatEta returns "" for non-positive, so the last step shows just the step).
function genStepLabel(p: DiffusionGenerateProgress): string {
  // Text encoding (and any first-run warmup) happens before the first scheduler tick, so step 0 means "working, not denoising yet" rather than "Step 0/N".
  if (p.step === 0) return "Preparing (text encoding + warmup)…";
  const base = `Step ${p.step}/${p.total_steps}`;
  const eta = p.eta_seconds != null ? formatEta(p.eta_seconds) : "";
  return eta ? `${base} · ~${eta}` : base;
}

// Settling a generation whose POST response was lost (a tunnel timeout, say): the backend keeps denoising, so poll until it goes idle rather than failing the run.
const SETTLE_POLL_MS = 1000;
const SETTLE_MAX_MS = 6 * 60 * 60 * 1000; // hard cap; a native-CPU batch can run for hours
const SETTLE_MAX_FAILS = 5; // consecutive progress failures before calling the backend gone

/** Wait for the generation that outlived its POST, and confirm one existed: a rejected fetch does not say whether the request reached the backend, so idle progress is ambiguous. The wait needs evidence (progress seen active, or a gallery record that was not there when the POST went out) and reports a failed submission otherwise. Throws if the backend stays unreachable, or past SETTLE_MAX_MS, so a real outage or a wedged generation surfaces instead of counting the run as done. */
async function settleLostGeneration(
  isCurrent: () => boolean,
  knownIds: ReadonlySet<string>,
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
    // Idle on the very first look: the run may already have finished (a batch shorter than one poll still persists its records first) or never started. A gallery record we had not seen is the proof.
    try {
      const page = await getGallery(0, 1);
      if (page.images.some((image) => !knownIds.has(image.id))) return;
    } catch {
      fails += 1;
      if (fails >= SETTLE_MAX_FAILS) throw new Error("Lost connection to the image server.");
      continue;
    }
    throw new Error("The image generation request did not reach the server.");
  }
  // Out of budget with the run still active. Returning here would report success and let the next run start against a busy backend, so surface it as a failure.
  throw new Error("Timed out waiting for the image generation to finish.");
}

// The chat tab model-load toast styling, reused verbatim so the diffusion load toast is visually identical.
const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

// Render the chat ModelLoadDescription for a progress poll. The base repo (text-encoder/VAE) downloads alongside the GGUF, so the total exceeds the picked quant size.
function loadToastDescription(p: DiffusionLoadProgress) {
  // "Downloading" only when bytes actually remain: a cached model (or the pre-estimate window, total still 0) must not claim a download.
  const downloading = p.bytes_total > 0 && p.bytes_downloaded < p.bytes_total * 0.999;
  const title = downloading
    ? "Downloading model…"
    : p.phase === "finalizing"
      ? "Loading to GPU…"
      : "Starting model…";
  const hasTotal = p.bytes_total > 0;
  return (
    <ModelLoadDescription
      title={title}
      message="Loading the model. This may include downloading its base model."
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

// Toast args mirroring chat: persistent, closeable, content in `description`. Pass `id` to update the existing toast in place instead of stacking a new one.
function loadToastArgs(p: DiffusionLoadProgress, id?: string | number) {
  return {
    ...(id != null ? { id } : {}),
    description: loadToastDescription(p),
    duration: Infinity,
    closeButton: true,
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

// Mirrors the Train page SliderRow (studio/sections/params-section.tsx): label + standard Slider + number input. The Images sliders are Chat ParamSlider, so both pages share one control.
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

// Matches the field-label style used across Studio (export/chat settings).
function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1">
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
        {hint && <InfoHint>{hint}</InfoHint>}
      </div>
      {children}
    </div>
  );
}

// The engaged value of a resolved Advanced control, formatted for its "Auto: X" badge: short scheme/mode tokens uppercase (INT8, FP8, FBCACHE), `_native_cudnn` as cuDNN, cpu_offload as On/Off.
function formatResolvedValue(key: string, value: string | boolean | null): string {
  if (key === "cpu_offload") return value ? "On" : "Off";
  if (value === null || value === "") return "Off";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === "_native_cudnn" || value.toLowerCase() === "cudnn") return "cuDNN";
  // Deferred speed auto: the dense pipe stays exact/eager and compiles on the 3rd image (the tooltip carries the full reason).
  if (value === "deferred") return "On from 3rd image";
  return value.toUpperCase();
}

// The "Auto: X" badge for one Advanced control, rendered only when the backend resolved that control (source === "auto"); an explicit choice renders nothing and the reason is a hover tooltip.
function ResolvedBadge({
  status,
  controlKey,
}: {
  status: DiffusionStatus | null;
  controlKey: string;
}) {
  const resolved = status?.resolved?.[controlKey];
  if (!resolved || resolved.source !== "auto") return null;
  const badge = (
    <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-ui-9 font-medium uppercase tracking-wider text-muted-foreground">
      Auto: {formatResolvedValue(controlKey, resolved.value)}
    </span>
  );
  if (!resolved.reason) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{badge}</TooltipTrigger>
      <TooltipContent>{resolved.reason}</TooltipContent>
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
  // A short always-visible description under the row, for controls whose label alone does not convey what they do (the hint tooltip carries the full detail).
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

// Source-image picker for the Transform (img2img) workflow: click or drag-drop an image, read it to a data URL the generate request sends as init_image. Shows a thumbnail with a Clear button once set.
function ImageDropzone({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (dataUrl: string | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const readFile = useCallback(
    (file: File | undefined | null) => {
      if (!file || !file.type.startsWith("image/")) {
        if (file) toast.error("Please choose an image file");
        return;
      }
      const reader = new FileReader();
      reader.onload = () => onChange(typeof reader.result === "string" ? reader.result : null);
      reader.onerror = () => toast.error("Could not read the image");
      reader.readAsDataURL(file);
    },
    [onChange],
  );

  if (value) {
    return (
      <div className="relative overflow-hidden rounded-[10px] border border-border">
        <img src={value} alt="Source" className="max-h-44 w-full object-contain bg-muted/30" />
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              aria-label="Remove source image"
              className="absolute right-1.5 top-1.5 size-7"
              onClick={() => {
                onChange(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Remove</TooltipContent>
        </Tooltip>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        readFile(e.dataTransfer.files?.[0]);
      }}
      className={cn(
        "flex h-28 w-full flex-col items-center justify-center gap-1 rounded-xl border border-dashed text-xs transition-colors",
        dragging
          ? "border-primary/60 bg-primary/5 text-foreground"
          : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
      )}
    >
      <HugeiconsIcon icon={ImageAdd02Icon} className="size-5" />
      <span>Click or drop an image</span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => readFile(e.target.files?.[0])}
      />
    </button>
  );
}

// A brush-based mask editor for inpainting: the source image with a paintable overlay, exporting a grayscale PNG mask at the image NATIVE resolution (diffusers convention: white = repaint, black = keep).
// Strokes draw to the visible tinted overlay and an offscreen mask canvas in lockstep, so the exported mask matches what the user sees. `brushPct` sizes the brush as a fraction of the shorter side.
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

  // (Re)initialise both canvases whenever the image changes or Clear is pressed: size them to the image native pixels and reset the mask to all-black (keep all).
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

// Redraw an image/canvas at (w, h). Used to clamp an outpaint source (and the built pair) to a size the browser can back and the backend can decode.
function scaleToCanvas(source: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const dst = document.createElement("canvas");
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext("2d");
  if (!dctx) throw new Error("Could not scale the extended canvas");
  dctx.drawImage(source, 0, 0, w, h);
  return dst;
}

// Build the (image, mask) pair for outpaint by reusing the inpaint backend: grow the canvas by `pct` per dimension on the selected sides and edge-bleed the original pixels into the new bands, so the VAE encodes plausible content.
// Mask the new bands white (= repaint), with a small overlap into the original on each grown side so the seam blends.
async function buildOutpaint(
  src: string,
  sides: ExtendSides,
  pct: number,
): Promise<{ image: string; mask: string }> {
  const source = await loadImage(src);
  // Scale the SOURCE so the grown canvas fits MAX_SIDE, before allocating anything. Growing all four sides by 100% multiplies the source area by 9, and a canvas past the browser limit is
  // "unusable -- drawing commands will not work" (MDN): every drawImage silently no-ops, so Extend posted a fully transparent init_image + mask instead of the photo. Measured Chromium cap is
  // 268,435,456 px of area; iOS caps a canvas at 4096x4096. Pre-scaling also keeps the two full-size allocations (image + mask) under a gigabyte. The backend rounds to /16, so exact dims are not required.
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

  // The source pre-scale sizes the canvases to fit MAX_SIDE, but the per-side rounding can overshoot by a pixel or two, which the backend 4096px-per-side decode limit would 400. Trim that slack here.
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
  // Controlled + force-closed off-tab: PopoverContent portals to body, so the hidden/inert page wrapper cannot contain it when the page is kept mounted.
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

type Busy = "loading" | "unloading" | "generating" | null;

// The Advanced controls a load sends, with the "auto" sentinels already resolved to omitted. A staged download pins one at pick time so the load matches the file set planned and downloaded for it.
type LoadAdvanced = Pick<
  DiffusionLoadRequest,
  | "cpu_offload"
  | "speed_mode"
  | "transformer_quant"
  | "attention_backend"
  | "memory_mode"
  | "transformer_cache"
  | "loras"
>;

export function ImagesPage({ active = true }: { active?: boolean }) {
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(
    "a tiny ginger sloth coding in a sunlit treehouse, photorealistic",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  // width/height are the source of truth; `aspect` locks their proportion ("custom" = free) and `portrait` tracks orientation, so Flip keeps the lock.
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [aspect, setAspect] = useState("1:1");
  const [portrait, setPortrait] = useState(false);
  // Z-Image-Turbo official defaults: 9 steps (= 8 DiT forwards), guidance 0 (distilled CFG-free; a negative prompt is ignored at this guidance).
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const [seed, setSeed] = useState("");
  // Batch size = images per forward pass (VRAM-heavy); count = sequential loops.
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);
  // Active workflow tab: "create" = text-to-image, "transform" = img2img, "inpaint" = mask-guided redraw. More tabs slot in here.
  const [workflow, setWorkflow] = useState<WorkflowId>("create");
  // Transform (img2img) / Inpaint inputs: the uploaded source image as a data URL, and the denoise strength (low = keep source, high = reimagine).
  const [initImage, setInitImage] = useState<string | null>(null);
  const [strength, setStrength] = useState(0.6);
  // Inpaint mask (grayscale PNG data URL, white = repaint), the brush size as a percent of the image shorter side, and a key bumped to clear the painted mask.
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [brushPct, setBrushPct] = useState(8);
  const [maskResetKey, setMaskResetKey] = useState(0);
  // Extend (outpaint): how far to grow each dimension and which sides. Reuses the inpaint backend by building a padded image + border mask at generate time.
  const [extendPct, setExtendPct] = useState(25);
  const [extendSides, setExtendSides] = useState<ExtendSides>({
    left: true,
    right: true,
    top: true,
    bottom: true,
  });
  // Upscale (hires fix): the enlargement factor and the low denoise strength used to re-detail the enlarged image. The backend caps the factor and rounds the target size.
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [upscaleStrength, setUpscaleStrength] = useState(0.35);
  // Reference (FLUX.2): up to 3 ADDITIONAL reference images beyond the primary one, combined by the model (subject + style, character + scene).
  const [referenceImages, setReferenceImages] = useState<string[]>([]);
  // LoRA adapters selected for the next generation (id + weight), plus the list the picker offers. Applied at generate time; refreshed per loaded family.
  const [loras, setLoras] = useState<LoraSpecInput[]>([]);
  const [availableLoras, setAvailableLoras] = useState<DiffusionLoraInfo[]>([]);
  // Page mode: "create" is the generation workspace, "train" the full-page LoRA training workspace. Independent of the loaded generation model.
  const [pageMode, setPageMode] = useState<"create" | "train">("create");
  // Train family + base live here so the top bar can pick them: on Train it replaces the generation model selector, which only offers untrainable GGUFs.
  const [trainFamilies, setTrainFamilies] = useState<TrainFamilyOption[]>([]);
  const [trainFamilyName, setTrainFamilyName] = useState("flux.1");
  const [trainBaseChoice, setTrainBaseChoice] = useState("");
  // Bumped when a training run completes, forcing the LoRA discovery effect to rescan so a freshly-trained adapter appears without a model reload.
  const [loraRefreshKey, setLoraRefreshKey] = useState(0);
  // ControlNet for the next generation: model id, control image (data URL), how to derive the control map, and the conditioning strength. Options refresh per family; applied only when a model + control image are set.
  const [controlnetId, setControlnetId] = useState<string>("");
  const [controlImage, setControlImage] = useState<string | null>(null);
  // Free-form: a union ControlNet advertises depth/pose/etc alongside the preprocessing "canny", and the backend maps the exact control_type to the union control_mode. The picker is built from the model own control_types.
  const [controlType, setControlType] = useState<string>("passthrough");
  const [controlStrength, setControlStrength] = useState(0.7);
  const [availableControlNets, setAvailableControlNets] = useState<DiffusionControlNetInfo[]>([]);
  // Advanced options live in a right-docked panel (like Chat settings panel), closed by default; one fixed top-bar toggle opens/closes it so the icon never moves.
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Advanced (load-time) options; "auto"/"off"/"none" map to the backend defaults. They apply when a model loads, and changing them while one is loaded shows a "Reapply" button.
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [transformerQuant, setTransformerQuant] = useState<
    "none" | "auto" | "int8" | "fp8" | "nvfp4" | "mxfp8"
  >("auto");
  const [attentionBackend, setAttentionBackend] = useState<"auto" | "native" | "cudnn" | "flash3" | "sage">(
    "auto",
  );
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  const [transformerCache, setTransformerCache] = useState<"auto" | "off" | "fbcache">("auto");
  const [cpuOffload, setCpuOffload] = useState(false);
  // The last load descriptor, so "Reapply" can reload the same model with new advanced options without the user re-picking it.
  const lastLoad = useRef<{ repoId: string; kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null>(
    null,
  );
  // Render-safe mirror of "lastLoad.current was set by a user-initiated load": a resident GGUF/single_file discovered by refresh carries no checkpoint filename in status, so lastLoad stays null and Reapply would be dead.
  // Set only from event handlers; the resident-pipeline case is derived from status at render time (mirrors the video page).
  const [canReapply, setCanReapply] = useState(false);
  // Repo id whose defaults were already seeded from a discovered resident model, so we seed once per resident model and never clobber a later manual edit.
  const seededResident = useRef<string | null>(null);

  const [busy, setBusy] = useState<Busy>(null);
  // {done, total} while a multi-run generation is in flight (null = idle). The total is just `count`, so it is not stored separately.
  const [genDone, setGenDone] = useState<number | null>(null);
  // Live per-step progress (step / total + ETA) polled during generation.
  const [genStep, setGenStep] = useState<DiffusionGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // visibilitychange handler active while a generation poll runs: background tabs clamp setInterval to >=1s (and can suspend it after ~5 min), so returning to the tab fires one immediate poll.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  // Controlled so the body-portaled overlays (model selector, aspect-ratio dropdown) force-close while this page is mounted but off-tab; a hidden/inert parent cannot contain a body portal.
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [aspectOpen, setAspectOpen] = useState(false);
  // Records come from the backend (durable); srcById maps each id to its object URL (loaded images) or data URL (the one just generated).
  const [images, setImages] = useState<GalleryImage[]>(() => galleryCache.images);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    galleryCache.srcById.toRecord(),
  );
  // Guards a "load more" so a fast scroll can't fire several at once.
  const loadingMore = useRef(false);
  // The gallery strip, used as the IntersectionObserver root so a tile PNG is fetched as it nears view instead of every tile of every page up front.
  const stripRef = useRef<HTMLDivElement | null>(null);
  // Ids currently intersecting the strip. The blob cache never evicts these, so pruning cannot pull an image out from under a visible tile.
  const visibleIds = useRef<Set<string>>(new Set());
  // False once the page truly unmounts (app close / chat-only eject). Tab switches keep it mounted, so a batch keeps generating off-tab; the multi-run loop only stops on a real unmount.
  const isMounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place (chat-style).
  const loadToastId = useRef<string | number | null>(null);
  // Last load-progress signature shown, so a tick that moved nothing skips the toast.
  const lastLoadSig = useRef<string | null>(null);
  // The quant to restore if the optimistic swap fails. A same-repo quant change sets `quant` immediately for picker feedback; if the load then fails AFTER starting the old pipeline stays loaded,
  // so the poll rolls the label back rather than advertise the failed quant. `{ prev }` distinguishes "revert to null" from "nothing pending".
  const quantRevert = useRef<{ prev: string | null } | null>(null);
  // The Reapply target to restore if the optimistic swap fails: handleLoad overwrites lastLoad.current with the pending pick at load start, and a load that fails AFTER starting leaves the previous pipeline resident.
  // The poll then rolls lastLoad back to what is actually resident. Mirrors quantRevert; `{ prev }` distinguishes "restore null" from "nothing pending".
  const lastLoadRevert = useRef<{ prev: typeof lastLoad.current } | null>(null);
  // A trained adapter awaiting deployment: after Deploy loads the base, the LoRA discovery effect applies this once the model is loaded + LoRA-capable for the matching family.
  const pendingDeploy = useRef<{ loraId: string; family: string } | null>(null);

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);

  // Mirror to the module cache so a tab switch re-renders instantly.
  useEffect(() => {
    galleryCache.images = images;
    galleryCache.hasMore = hasMore;
    galleryCache.selectedId = selectedId;
    galleryCache.quant = quant;
  }, [images, hasMore, selectedId, quant]);

  // Refresh the LoRA picker suggestions when the loaded family changes. A LoRA is trained for a specific base family, so a real model SWAP invalidates the selection -- clear it then.
  // But NOT on the first load or an unload: a user can restore a saved recipe before the model finishes loading, and clearing on that load->capable transition would drop the restored adapters, so track the previous family in a ref.
  // We also do NOT filter the selection against the catalog: a valid pick can be a free-text HF repo id absent from the (often empty) curated list.
  const loraCapable = Boolean(status?.loaded && status?.supports_lora);
  const prevLoraFamilyRef = useRef<string | null | undefined>(undefined);
  // Whether the load in flight baked the LoRA selection into the build (see handleLoad).
  const bakedLorasOnLoad = useRef(false);
  useEffect(() => {
    if (!loraCapable) {
      // Options are gone with the model, but keep the selection: it may have just been restored while the model is (re)loading. It is only SENT when loraCapable, and a real family swap below clears it.
      setAvailableLoras([]);
      return;
    }
    const fam = status?.family ?? null;
    const prev = prevLoraFamilyRef.current;
    if (prev != null && prev !== fam) {
      setLoras([]);
    }
    prevLoraFamilyRef.current = fam;
    // A just-deployed adapter: now that the base is loaded + LoRA-capable, apply it (after the family-swap clear above). Only when the family matches what it was trained for; otherwise warn.
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
        // Clear only the OPTIONS on a failed catalog refresh: unlike the catalog-only picker below, this free-text picker holds selections (bare HF repo ids) valid without being in the catalog, so a transient failure must not wipe them.
        // Stale cross-family selections are already cleared by the family-swap check above, and hidden LoRAs are never sent (handleGenerate is gated on loraCapable).
        if (!cancelled) setAvailableLoras([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loraCapable, status?.family, loraRefreshKey]);

  // A torchao int8/fp8 build takes adapters ONLY at load time, baked in before quantize_ and compile. Switching artifact within one family keeps the selection (the family-swap clear sees no change) while the new load did not bake it,
  // so the next Generate would 400 against a picker still showing the adapter as active. Drop it once per resident build and say why.
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

  // Refresh the ControlNet picker options when the loaded model (family) changes, and clear a stale selection the new model cannot use.
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

  // The control types offered for the selected ControlNet: a union model advertises several (canny/depth/pose/passthrough), a plain model its own. Falls back to the preprocessing pair when nothing is selected.
  // to the preprocessing pair when nothing is selected.
  const controlTypeOptions = useMemo(() => {
    const cn = availableControlNets.find((c) => c.id === controlnetId);
    const types = cn?.control_types?.length ? cn.control_types : ["passthrough", "canny"];
    return types;
  }, [availableControlNets, controlnetId]);

  // Keep controlType valid for the selected model: if the current choice is not among its advertised types, snap to the first (prefer passthrough when offered).
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
      // Evict the coldest off-screen images this one pushed over budget. On-screen tiles and the image open in the viewer are protected, so eviction is never visible; an evicted tile re-fetches when scrolled back to.
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

  const loadGallery = useCallback(async () => {
    try {
      const page = await getGallery(0, PAGE_SIZE);
      galleryCache.images = page.images;
      galleryCache.hasMore = page.has_more;
      setImages(page.images);
      setHasMore(page.has_more);
      // No visibility signal without IntersectionObserver (jsdom / an old webview), so keep the eager fetch there rather than render a strip of permanent spinners.
      if (typeof IntersectionObserver === "undefined") {
        page.images.forEach((image) => void ensureSrc(image));
      }
    } catch {
      // Best-effort: a failed gallery load shouldn't block the page.
    }
  }, [ensureSrc]);

  // Load the next older page. offset = how many are loaded so far; a newly generated image sorts to the front on the backend too, so the offset keeps pointing at the same older images.
  const loadMore = useCallback(async () => {
    if (loadingMore.current || !galleryCache.hasMore) return;
    loadingMore.current = true;
    try {
      const page = await getGallery(galleryCache.images.length, PAGE_SIZE);
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

  // A gallery page holds PAGE_SIZE multi-megabyte PNGs and an object URL lives until the page closes, so fetching every record of every page up front grew memory without bound for tiles the user may never look at.
  // Fetch a tile as it nears the strip edge instead; it already shows a spinner until its src lands. Observed from here rather than through a ref on each tile,
  // and re-run per page so "load more" tiles are picked up and removed ones go with the observer. Mirrors the Video page strip.
  useEffect(() => {
    const root = stripRef.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.imageId;
          if (!id) continue;
          // Visibility is also the cache recency and protection signal: an on-screen tile is never evicted, and leaving the viewport makes it a candidate again.
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
      // rootMargin applies to the ROOT box only, so the root must be the horizontally scrolling strip: a tile past its right edge is clipped, and a viewport-root margin would never reach it. The sideways margin starts a fetch a few tiles early.
      { root, rootMargin: "0px 600px" },
    );
    for (const tile of root.querySelectorAll("[data-image-id]")) io.observe(tile);
    return () => io.disconnect();
  }, [images, ensureSrc]);

  // The preview is what the user actually looks at, so the selected image is fetched whether or not its tile is on screen (a restored selection, a fresh generation, or a pick just before the strip scrolls it away).
  useEffect(() => {
    if (!selected) return;
    void (async () => {
      await ensureSrc(selected);
    })();
  }, [selected, ensureSrc]);

  useEffect(() => {
    void loadGallery();
  }, [loadGallery]);

  const handleDelete = useCallback(async (id: string) => {
    try {
      await deleteGalleryImage(id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete image");
      return;
    }
    galleryCache.srcById.delete(id); // revokes the URL with the entry
    visibleIds.current.delete(id);
    // A fetch still in flight for this id must discard its blob rather than cache it.
    galleryCache.deleted.add(id);
    setSrcById((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setImages((prev) => prev.filter((i) => i.id !== id));
    setSelectedId((cur) => (cur === id ? null : cur));
  }, []);

  // Load an image's recipe back into the form inputs.
  const restoreSettings = useCallback((image: GalleryImage) => {
    setPrompt(image.prompt);
    // Negative prompt only applies when guidance>0; don't restore a hidden value.
    setNegativePrompt(image.guidance > 0 ? (image.negative_prompt ?? "") : "");
    setSteps(image.steps);
    setGuidance(image.guidance);
    // Restore from the BASE batch seed, not this image own derived seed (base + index), or replaying with batch_size would advance again and reproduce a different image. Diffusers shares one seed; older records fall back to seed.
    setSeed(String(image.batch_seed ?? image.seed));
    setWidth(image.width);
    setHeight(image.height);
    // The batch shared one base seed, so an image with batch_index>0 only reproduces by replaying the whole batch: restore the batch size too (older recipes default to 1).
    setBatchSize(image.batch_size ?? 1);
    const m = matchAspect(image.width, image.height);
    setAspect(m.key);
    setPortrait(m.portrait);
    // Restore selected LoRA adapters from the recipe ("id:weight" strings); split on the LAST colon so an id containing ':' is preserved, and skip unparseable entries.
    // A recipe with no LoRAs clears the selection, so the restore reproduces the image faithfully rather than leaking a stale form selection.
    const restoredLoras: LoraSpecInput[] = [];
    for (const entry of image.loras ?? []) {
      const idx = entry.lastIndexOf(":");
      if (idx <= 0) continue;
      const id = entry.slice(0, idx);
      const weight = Number(entry.slice(idx + 1));
      if (id && Number.isFinite(weight)) restoredLoras.push({ id, weight });
    }
    setLoras(restoredLoras);
    // The conditioned workflows scalar settings ARE persisted, so restore them even though the form returns to Create: they are what the user re-applies after adding the image back.
    if (typeof image.strength === "number") {
      if (image.workflow === "upscale") setUpscaleStrength(image.strength);
      else setStrength(image.strength);
    }
    if (typeof image.upscale === "number") setUpscaleFactor(image.upscale);
    // None of the conditioning images are persisted in a recipe, so a restore must not leave the form pointing at whatever is loaded in the Transform / Inpaint / Edit tabs: the next Generate would condition on an unrelated image.
    // Clear them all and return to Create (the workflow-validity effect snaps back if the loaded model is edit-only).
    setWorkflow("create");
    setInitImage(null);
    setMaskImage(null);
    setReferenceImages([]);
    // The control image isn't persisted, so clear any stale ControlNet selection.
    setControlnetId("");
    setControlImage(null);
    // Say so, rather than letting a conditioned image restore as a plain Create that quietly generates something unrelated. The scalar settings ARE restored above; only the images have to be supplied again.
    const conditioned = CONDITIONED_WORKFLOW_INPUTS[image.workflow ?? ""];
    if (conditioned) {
      toast.success(`Settings restored. Add ${conditioned} again to reproduce this image.`);
    } else {
      toast.success("Settings restored to inputs");
    }
  }, []);

  // A locked ratio keeps the paired dimension in step while dragging; "custom" frees both, Flip swaps W/H. ratioHW is h/w for ratio [a,b] (long:short): landscape h = w*b/a, portrait h = w*a/b.
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

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getDiffusionStatus());
    } catch {
      // Status is best-effort; a failed poll shouldn't surface an error toast.
    }
  }, []);

  // Track mount so a long generate run stops issuing GPU work only on a true unmount (app close / chat-only eject). The page stays mounted across tab switches (RootLayout keeps it alive like chat), so a batch keeps generating off-tab.
  // The mount-time refreshStatus and timer/toast cleanup live in the load-resume effect below, so this one carries only the mount flag.
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  // Re-sync model status when the tab becomes active again: while off-tab the diffusion model may have been evicted (a chat load claimed the GPU) and the page no longer remounts on return.
  useEffect(() => {
    if (!active) return;
    void (async () => {
      await refreshStatus();
    })();
  }, [active, refreshStatus]);

  // Collapse the body-ported popovers when leaving the tab: their open state is force-closed via `active && open` while off-tab, but the underlying flag stays set, so returning would pop them back open.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
    setAspectOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error", updating the persistent toast in place each tick.
  const pollLoadProgress = useCallback(async () => {
    try {
      const p = await getDiffusionLoadProgress();
      if (p.phase === "ready") {
        dismissLoadToast();
        setStatus(await getDiffusionStatus());
        toast.success("Model loaded");
        setBusy(null);
        // Load succeeded: the optimistic quant is now the real one, so drop the pending revert.
        quantRevert.current = null;
        // lastLoad.current already holds the now-resident pick, so drop its revert too.
        lastLoadRevert.current = null;
        return;
      }
      if (p.phase === "error") {
        dismissLoadToast();
        toast.error(p.error || "Failed to load model");
        setBusy(null);
        // A load that failed AFTER starting leaves the previous pipeline loaded, so roll the optimistic quant label back to what is actually loaded (status carries no quant, so refreshStatus alone cannot correct it).
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        // Same rollback for the Reapply target: the previous pipeline is still resident, so point Reapply back at it rather than the failed pick.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.prev != null);
          lastLoadRevert.current = null;
        }
        // A failed load may have freed a previously-loaded model, so resync to the real backend state (the synchronous failure path does the same).
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // No load in flight and nothing loaded: the load was cancelled or evicted (a chat load took the GPU) and the backend cleared its state. Terminal -- else this loop spins forever and busy sticks on "loading".
        dismissLoadToast();
        setBusy(null);
        // Same optimistic-quant rollback as the error path: the swap did not take.
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        // Restore the Reapply target too (symmetric with the error/quant rollback), so it never lingers on the failed pick after the load is cancelled/evicted.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.prev != null);
          lastLoadRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      // Include bytes_total: the estimate lands as a 0->real jump while phase and bytes_downloaded hold, and the toast shows the percentage off that total.
      const sig = `${p.phase}:${p.bytes_downloaded}:${p.bytes_total}`;
      if (loadToastId.current != null && sig !== lastLoadSig.current) {
        lastLoadSig.current = sig;
        toast(null, loadToastArgs(p, loadToastId.current));
      }
    } catch {
      // Transient poll failure: keep trying.
    }
    pollTimer.current = setTimeout(() => void pollLoadProgress(), 1000);
  }, [dismissLoadToast, refreshStatus]);

  // Re-enter the per-step poll for a generation already in flight on the backend that this page did not start (another client, or a reload mid-generate), instead of showing a stale idle view.
  // generate-progress carries no terminal record, so refresh the gallery on completion to merge any image saved after the mount fetch. Separate from handleGenerate own loop.
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
      // A load runs on the backend as a daemon thread that survives navigation. On (re)mount, resume tracking one still in flight so the page shows progress and updates on completion, not a stale view.
      try {
        const p = await getDiffusionLoadProgress();
        if (p.phase === "downloading" || p.phase === "finalizing") {
          setBusy("loading");
          dismissLoadToast();
          lastLoadSig.current = null;
          loadToastId.current = toast(null, loadToastArgs(p));
          void pollLoadProgress();
        }
      } catch {
        // Resume is best-effort; a failed probe just leaves the idle view.
      }
      // Resume tracking a generation started elsewhere (another client, or before a reload) so the page shows the in-flight run instead of a stale idle view. Mirrors the video page.
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
    // Stop polling if the page unmounts mid-load / mid-generate, and dismiss the load toast: its poll loop is gone, so it would hang forever (duration: Infinity) on whatever page the user navigated to.
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      if (genVisibilityListener.current) {
        document.removeEventListener("visibilitychange", genVisibilityListener.current);
        genVisibilityListener.current = null;
      }
      dismissLoadToast();
    };
  }, [refreshStatus, dismissLoadToast, pollLoadProgress, resumeGeneratePoll]);

  // Seed the generation sliders to a resident model recipe when the page discovers one it did not load itself (left by a prior session or another route); refreshStatus() only sets `status`.
  // Without this the sliders keep the unrecognised-model fallback (few-step / no-CFG), so a resident full model (flux.1-dev) generates at 9 steps / guidance 0 and produces garbage until re-picked.
  // Guarded by lastLoad.current === null (a user-initiated load seeds its own defaults) and a per-repo ref, so a manual edit after the seed is never clobbered.
  useEffect(() => {
    const repoId = status?.loaded ? status.repo_id : null;
    if (!repoId) return;
    if (lastLoad.current) return;
    if (seededResident.current === repoId) return;
    seededResident.current = repoId;
    // Seed from base_repo (the resolved diffusers base, holding the family), not repo_id: a GGUF/single_file/local-path resident has a repo_id with no family substring, so defaultsFor(repoId) would pick the wrong distilled recipe.
    const d = defaultsFor(status?.base_repo ?? repoId);
    setSteps(d.steps);
    setGuidance(d.guidance);
    // Wire "Reapply" to the resident model too, so an advanced-option reload works without re-picking from the dropdown. Only a full pipeline load needs no checkpoint filename;
    // a resident GGUF/single_file carries none in status and the backend rejects such a load (400 before it evicts), so leave lastLoad null for those rather than wire a reload that can never complete.
    const kind = status?.model_kind;
    if (kind === "pipeline") {
      lastLoad.current = { repoId, kind };
    }
  }, [status?.loaded, status?.repo_id, status?.base_repo, status?.model_kind]);

  // The adapter list a load of *repoId* would BAKE into the build, shared by the load itself and by the download plan so the two never disagree.
  // A torchao int8/fp8 transformer can only take adapters before quantize_ + compile, so a baked selection forces the dense build path, which needs different files staged.
  // Only a reload of the SAME target bakes: a fresh pick of another model can still hold a cross-family selection the family-swap effect clears after the load. Reads lastLoad.current, so call it before the load overwrites it.
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

  // One snapshot of every Advanced control a load sends, so a staged pick can pin the values it planned against and the load cannot drift from them. Sentinels ("auto") map to undefined here once, not at each call site.
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
    ],
  );

  const handleLoad = useCallback(
    // Resolves true when the background load STARTED (callers may revert optimistic picker state on false); poll outcomes are handled internally.
    async (
      repoId: string,
      opts: {
        kind: "gguf" | "single_file" | "pipeline";
        filename?: string;
      },
      // The Advanced values this load must use, when they were pinned earlier. A staged download plans its file set from the controls as they stood at pick time and only loads minutes later,
      // so reading the live state here would run a load the staged files may not cover (the rest then pulled inline, outside the manager) or leave gigabytes staged for a build this load no longer uses.
      pinned?: LoadAdvanced,
    ): Promise<boolean> => {
      // Cancel any prior poll loop so two can't run at once.
      if (pollTimer.current) clearTimeout(pollTimer.current);
      setBusy("loading");
      // Show the chat-style toast immediately; the poll updates it by id.
      dismissLoadToast();
      lastLoadSig.current = null;
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS));
      // Remember what was loaded so "Reapply" can reload it with new advanced options. Snapshot the prior target first: a load that fails to START (validation, gated repo, training guard)
      // leaves the previous model resident, so Reapply and the resident-default seeding must keep pointing at it, not the failed pick.
      const prevLastLoad = lastLoad.current;
      // A torchao int8/fp8 transformer (what the default GGUF fast path resolves to on any capable GPU) can only take adapters at LOAD time: they attach on the dense transformer before quantize_ + compile.
      // Generation-time /images/generate then rejects a new adapter set with "reload the model with the adapter selection", so a reload that drops the selection leaves the LoRA picker permanently unusable.
      // Ignored by every other load kind (bf16 / bnb-4bit apply adapters at generation time).
      const advanced = pinned ?? currentLoadAdvanced(repoId);
      const bakeLoras = advanced.loras ?? [];
      // Whether THIS load carries the selection into the build, so a quantized load that did not can drop a selection it can never apply rather than failing the next Generate.
      bakedLorasOnLoad.current = bakeLoras.length > 0;
      lastLoad.current = { repoId, kind: opts.kind, filename: opts.filename };
      setCanReapply(true);
      // Carry the prior target so the async poll can restore it if the background load fails after starting (the previous pipeline stays resident); the sync catch clears it.
      lastLoadRevert.current = { prev: prevLastLoad };
      try {
        // Returns immediately -- the load runs in the background and we poll for it. The backend infers the family + base diffusers repo from the id; forward the saved HF token so gated bases (FLUX dev/klein) download.
        // A pipeline load carries no filename (the repo IS the pipeline); single-file kinds send the GGUF / safetensors filename.
        // Advanced sentinels ("auto"/"off"/"none") map to omitted so the backend uses its defaults.
        await loadDiffusionModel({
          model_path: repoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          cpu_offload: advanced.cpu_offload,
          speed_mode: advanced.speed_mode,
          transformer_quant: advanced.transformer_quant,
          attention_backend: advanced.attention_backend,
          memory_mode: advanced.memory_mode,
          transformer_cache: advanced.transformer_cache,
          loras: bakeLoras.length > 0 ? bakeLoras : undefined,
        });
      } catch (err) {
        lastLoad.current = prevLastLoad;
        setCanReapply(prevLastLoad != null);
        lastLoadRevert.current = null;
        dismissLoadToast();
        toast.error(err instanceof Error ? err.message : "Failed to start load");
        setBusy(null);
        void refreshStatus();
        return false;
      }
      void pollLoadProgress();
      return true;
    },
    [pollLoadProgress, refreshStatus, dismissLoadToast, currentLoadAdvanced],
  );

  // Set (or clear) the Transform/Inpaint source image; always drop any painted mask, which is sized to the previous source.
  const handleInitChange = useCallback((dataUrl: string | null) => {
    setInitImage(dataUrl);
    setMaskImage(null);
    setMaskResetKey((k) => k + 1);
  }, []);

  // Downloads go through the Hub download manager like every other model, sharing its panel, progress, cancel/resume, disk preflight and manifest verification; the load then finds a warm cache. Held in a ref so the completion callback is not a render dep.
  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string };
    // The Advanced values the plan was built from. Staging does not set `busy`, so the user can change precision, memory mode, speed or the baked LoRAs while the download runs;
    // without this the completed load would use the new values against the old file set, pulling whatever is missing inline (no progress, no disk preflight) or leaving gigabytes staged for nothing.
    advanced: LoadAdvanced;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  handleLoadRef.current = handleLoad;
  // Set when a staged download finished while this page was hidden. Both diffusion pages stay mounted and a load evicts whatever holds the GPU, so a download landing in the background must not take the model out from under the page the user is on.
  // The pick is not dropped: it fires when this page comes back.
  const stagedLoadDeferred = useRef(false);
  const { stage } = useStagedDownload({
    scopeId: "diffusion",
    onReady: () => {
      if (!active) {
        stagedLoadDeferred.current = true;
        return;
      }
      const pending = pendingStagedLoad.current;
      pendingStagedLoad.current = null;
      if (pending) void handleLoadRef.current(pending.repoId, pending.opts, pending.advanced);
    },
  });

  useEffect(() => {
    if (!active || !stagedLoadDeferred.current) return;
    stagedLoadDeferred.current = false;
    const pending = pendingStagedLoad.current;
    pendingStagedLoad.current = null;
    if (pending) void handleLoadRef.current(pending.repoId, pending.opts, pending.advanced);
  }, [active]);

  // Stage a not-yet-downloaded hub pick, else load it directly. Returns true when the pick was accepted either way, so the callers optimistic picker state stands.
  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string },
      isDownloaded?: boolean,
    ): Promise<boolean> => {
      if (isDownloaded !== false) return handleLoadRef.current(repoId, opts);
      // ONE snapshot for the plan and for the load it will fire: the download runs for minutes without setting `busy`, so anything read separately at completion time could have changed.
      const advanced = currentLoadAdvanced(repoId);
      try {
        const plan = await getDiffusionDownloadPlan({
          model_path: repoId,
          gguf_filename: opts.filename,
          model_kind: opts.kind,
          // The same token and Advanced values handleLoad sends, so the plan describes the load that will actually run.
          // Without the token the backend metadata lookup fails on a gated base and quietly plans no companion entry, leaving the load to pull those multi-GB files inline, outside this manager;
          // without the memory and quant controls an explicit Speed/Precision Off or low-VRAM pick stages base transformer/ shards the load never opens.
          hf_token: hfApiToken(getHfToken()),
          cpu_offload: advanced.cpu_offload,
          speed_mode: advanced.speed_mode,
          transformer_quant: advanced.transformer_quant,
          memory_mode: advanced.memory_mode,
          // The backend prefetch decision reads the adapter selection too: a baked LoRA always runs the dense build path. Omitting it here planned a quantized load file set and
          // staged too little for the load that actually ran, which pulled the rest inline outside the download manager. Same list handleLoad bakes.
          loras: advanced.loras,
        });
        if (plan.entries.length > 0) {
          pendingStagedLoad.current = { repoId, opts, advanced };
          stage(
            plan.entries.map((e) => ({
              repoId: e.repo_id,
              files: e.files,
              bytes: e.bytes,
              ggufFilename: e.gguf_filename,
            })),
          );
          return true;
        }
      } catch {
        // No plan (older backend, metadata hiccup): fall back to the load's own download.
      }
      return handleLoadRef.current(repoId, opts);
    },
    [stage, currentLoadAdvanced],
  );

  // A diffusion model picked from the chat picker arrives as ?model= on this route. Load it once, then clear the params so a refresh or a later manual eject does not reload it.
  const routeSearch = useSearch({ strict: false }) as {
    model?: string;
    quant?: string;
  };
  const navigateSelf = useNavigate();
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    // Only the page being shown consumes the query. This hook is loose (`strict: false`) and both diffusion pages stay mounted once visited, so the hidden one saw /video?model= too and raced this one:
    // it navigated back to its own route and tried to load the other page checkpoint as its own kind of model.
    if (!active) return;
    const wanted = routeSearch.model;
    // Key on the model AND the quant, and release the marker once the query is gone. This page stays mounted, so a marker that outlived the query made re-picking the same checkpoint a click that neither loaded nor cleared the URL.
    // It only has to survive the re-renders between consuming a pick and the clear landing.
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    const key = `${wanted}|${routeSearch.quant ?? ""}`;
    if (handledRouteModel.current === key) return;
    handledRouteModel.current = key;
    void navigateSelf({ to: "/images", search: {}, replace: true });
    // Same catalog lookup a direct pick makes: the chat picker can only forward a GGUF filename, so a curated single-file artifact would otherwise be loaded as a pipeline and fail.
    const pick = diffusionRoutePick(
      wanted,
      routeSearch.quant,
      loadSpecFor(wanted, IMAGE_CATALOG),
    );
    void loadOrStage(pick.repoId, pick.opts, false);
  }, [active, routeSearch.model, routeSearch.quant, loadOrStage, navigateSelf]);

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    const l = lastLoad.current;
    if (l) void handleLoad(l.repoId, { kind: l.kind, filename: l.filename });
  }, [handleLoad]);

  // The chat picker emits (modelId, picked quant + its exact filename) for a GGUF, or just (modelId) for a curated non-GGUF safetensors pick; load it and seed the inputs with that model defaults.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight: a replacement load now would tear down the live poll/toast and reset busy, and the backend rejects the second load with a 409.
      if (busy !== null) return;
      // Curated non-GGUF model: load as a full pipeline or single-file safetensors.
      const spec = loadSpecFor(id, IMAGE_CATALOG);
      if (spec && spec.kind !== "gguf") {
        setQuant(null);
        const d = defaultsFor(id);
        setSteps(d.steps);
        setGuidance(d.guidance);
        void loadOrStage(id, { kind: spec.kind, filename: spec.filename }, meta.isDownloaded);
        return;
      }
      // GGUF quant pick from the variant expander. Optimistic for instant picker feedback, but revert if the load fails to START (400/409/network) or LATER in the poll (download/preflight error/eviction):
      // in both cases the old pipeline stays loaded, so do not advertise the failed quant. The poll owns the after-start revert via quantRevert; here we only handle the never-started case.
      if (meta.ggufVariant && meta.ggufFilename) {
        const prevQuant = quant;
        quantRevert.current = { prev: prevQuant };
        setQuant(meta.ggufVariant);
        const dq = defaultsFor(id);
        setSteps(dq.steps);
        setGuidance(dq.guidance);
        void loadOrStage(
          id,
          { kind: "gguf", filename: meta.ggufFilename },
          meta.isDownloaded,
        ).then((started) => {
          if (!started) {
            setQuant(prevQuant);
            quantRevert.current = null;
          }
        });
        return;
      }
      // A direct single-file local .gguf pick has no variant/filename (custom folder / LM Studio). Load it by splitting the path into (parent dir, basename) the backend resolves, instead of silently doing nothing.
      if (meta.isGguf) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        if (!filename.toLowerCase().endsWith(".gguf")) {
          // A repo pick that reached here has no filename to load, and the backend cannot map a quant label back to one. Say so instead of doing nothing.
          toast.error("Pick a quantization for this model to load it");
          return;
        }
        // A direct pick carries no curated variant label; surface the filename so the selector stops advertising the previously loaded quant.
        // Optimistic, reverted if the load fails to start OR fails later in the poll (mirrors the curated branch above; the poll owns the after-start revert via quantRevert).
        const prevQuant = quant;
        quantRevert.current = { prev: prevQuant };
        setQuant(filename);
        const dq2 = defaultsFor(id);
        setSteps(dq2.steps);
        setGuidance(dq2.guidance);
        void handleLoad(dir, { kind: "gguf", filename }).then((started) => {
          if (!started) {
            setQuant(prevQuant);
            quantRevert.current = null;
          }
        });
        return;
      }
      // A direct local single-file .safetensors pick (custom folder / on-device file) must load via from_single_file: the pipeline route rejects a bare file (no model_index.json), and only after evicting the resident model.
      // Split into (parent dir, basename) like the local GGUF branch above.
      if (meta.source === "local" && id.toLowerCase().endsWith(".safetensors")) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        const prevQuant = quant;
        quantRevert.current = { prev: prevQuant };
        setQuant(filename);
        const dsf = defaultsFor(id);
        setSteps(dsf.steps);
        setGuidance(dsf.guidance);
        void handleLoad(dir, { kind: "single_file", filename }).then((started) => {
          if (!started) {
            setQuant(prevQuant);
            quantRevert.current = null;
          }
        });
        return;
      }
      // Otherwise treat it as a full diffusers repo (safetensors / bnb-4bit). The backend infers the family + base repo from the id and gates loads to unsloth/* repos or on-device paths, so only attempt those.
      if (meta.source !== "local" && !id.toLowerCase().startsWith("unsloth/")) {
        toast.error("Only unsloth or on-device image models can be loaded here");
        return;
      }
      // Optimistically clear the quant label, revert it if the load never starts.
      const prevQuant = quant;
      quantRevert.current = { prev: prevQuant };
      setQuant(null);
      const d = defaultsFor(id);
      setSteps(d.steps);
      setGuidance(d.guidance);
      void loadOrStage(id, { kind: "pipeline" }, meta.isDownloaded).then((started) => {
        if (!started) {
          setQuant(prevQuant);
          quantRevert.current = null;
        }
      });
    },
    [busy, handleLoad, loadOrStage, quant],
  );

  // Deploy a freshly-trained adapter from the Train tab: switch to Create, load the base as a pipeline, and queue the adapter so the LoRA discovery effect applies it once the base is loaded + LoRA-capable. Seeds the prompt with the trigger phrase when provided.
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
      pendingDeploy.current = { loraId: stem, family: args.family };
      if (args.trigger.trim()) setPrompt(args.trigger.trim());
      setPageMode("create");
      setQuant(null);
      const d = defaultsFor(args.baseRepo);
      setSteps(d.steps);
      setGuidance(d.guidance);
      void handleLoad(args.baseRepo, { kind: "pipeline" }).then((started) => {
        if (!started) pendingDeploy.current = null;
      });
    },
    [busy, handleLoad],
  );

  const handleUnload = useCallback(async () => {
    // Ejecting cancels any in-flight replacement load on the backend, so tear down its client-side tracking too: the load poll reschedules on phase null and the persistent toast never resolves, so both would leak forever.
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    // Drop the Reapply target with the model (like the Video page): the ejected pick is no longer resident, and leaving it set makes the resident-status seeding effect above skip repair (guarded on lastLoad.current === null)
    // while "Reapply to loaded model" reloads the ejected model instead of whatever another client/session left loaded.
    lastLoad.current = null;
    setCanReapply(false);
    setBusy("unloading");
    try {
      setStatus(await unloadDiffusionModel());
      setQuant(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
    } finally {
      setBusy(null);
    }
  }, [refreshStatus, dismissLoadToast]);

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

    // Resolve the conditioning image/mask/strength for this workflow up front. Extend (outpaint) is built here from the source by padding + masking the new border, then sent through the same inpaint path. txt2img leaves all three undefined.
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
        // Hires fix: the backend enlarges the source by `upscale` and re-denoises at this low strength, so it gains detail without changing the content.
        condInit = initImage ?? undefined;
        condUpscale = upscaleFactor;
        condStrength = upscaleStrength;
      } else if (isReference) {
        // FLUX.2 reference conditioning: send the primary reference + any extra references (combined by the model). A fresh image is generated at the slider size guided by the references + prompt. No mask, no strength.
        condInit = initImage ?? undefined;
        const extras = referenceImages.filter(Boolean);
        if (extras.length) condRefImages = extras;
      } else if (isEdit) {
        // Instruction editing: send the source image; the prompt IS the instruction. No mask, no strength (the edit pipeline fully regenerates).
        condInit = initImage ?? undefined;
      }
    } catch {
      toast.error("Could not prepare the source image");
      return;
    }
    // Resolve a base seed up front. With an explicit seed the run is fully reproducible; with a random one we still pick a concrete base now so each sequential image gets a distinct, reproducible seed (base + i).
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

    // Snap custom dims to the model's grid so a half-typed value can't 422.
    const w = snapDim(width);
    const h = snapDim(height);

    // A large run count (generate all night) is legitimate, so there is no upper cap: floor at 1 and ignore non-numeric input (the number box can yield NaN), which would otherwise make the loop a silent no-op.
    const runs = Number.isFinite(count) && count >= 1 ? Math.floor(count) : 1;
    if (runs !== count) setCount(runs);

    // An explicit seed near the 2**53-1 backend cap can overflow once the per-run offset (base + i*batchSize) and the engine in-batch +j offsets are added, 422ing a later run AFTER earlier images generated.
    // Fail before any GPU work. Subtraction keeps the comparison exact where the sum would round.
    if (baseSeed > Number.MAX_SAFE_INTEGER - (runs * batchSize - 1)) {
      toast.error("Seed too large for this run count and batch size; use a smaller seed");
      return;
    }

    setBusy("generating");
    setGenDone(0);
    setGenStep(null);
    // Poll the backend per-step progress across the whole run (all sequential generations) so the bar tracks live denoising steps.
    // A named poll body (guarded against overlap) also serves the visibilitychange listener: a background tab throttled interval catches up the moment the tab is visible.
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
    // Every gallery id this page has already seen, captured BEFORE the first POST and grown as the run produces records. settleLostGeneration proves a lost POST reached the backend by finding a record outside this set,
    // so the set must not be rebuilt inside the catch: by then run 1 own images are on the page, and run 2 would accept one of them as its proof.
    const knownIds = new Set(galleryCache.images.map((image) => image.id));
    try {
      for (let i = 0; i < runs; i++) {
        // The page truly unmounted mid-run (app close / chat-only eject): stop issuing more GPU generations. A plain tab switch keeps it mounted.
        if (!isMounted.current) break;
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
            // Offset runs by the batch size: the native engine seeds image j of a run at seed+j, so a +1 run offset would regenerate the previous run batch-mates. Unique per image on both engines, reproducible via recipes.
            seed: baseSeed + i * batchSize,
            batch_size: batchSize,
            // Transform/Inpaint/Extend send the source image (+ mask for inpaint/extend) and a denoise strength, resolved above. The backend derives output size from the image, so width/height are advisory.
            init_image: condInit,
            mask_image: condMask,
            strength: condStrength,
            upscale: condUpscale,
            reference_images: condRefImages,
            // Drop empty (no id yet) and zero-weight rows, and trim hand-typed repo ids, so the recipe records only adapters that applied.
            // Gate on loraCapable: a restore can leave adapters in state while the loaded model does not support LoRA (picker hidden), and sending them would fail generation with no visible row to remove.
            loras: (() => {
              if (!loraCapable) return undefined;
              const active = loras
                .map((l) => ({ id: l.id.trim(), weight: l.weight }))
                .filter((l) => l.id && l.weight > 0);
              return active.length ? active : undefined;
            })(),
            // ControlNet: sent only when a model + control image are chosen; v1 conditions plain text-to-image only, so skip it for image-conditioned workflows.
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
          // The POST response was lost while the backend kept generating (secure mode tunnel caps the origin response near 100s, which a native-CPU or high-step run exceeds).
          // Retrying would duplicate the work, so wait the run out and read its images off the gallery, where they are persisted anyway.
          if (!(err instanceof GenerateResponseLostError)) throw err;
          // The ids known before the POST went out: a record outside them proves the request reached the backend when progress reads idle straight away.
          await settleLostGeneration(() => isMounted.current, knownIds);
          if (!isMounted.current) break;
          await loadGallery();
          // loadGallery refreshes the module cache synchronously, so this run settled records are folded in before the next run can mistake one for its own proof.
          galleryCache.images.forEach((image) => knownIds.add(image.id));
          setGenDone(i + 1);
          continue;
        }
        if (!isMounted.current) break;
        // Prepend this run's records (newest first) and load their blobs.
        setImages((prev) => [...res.images, ...prev]);
        res.images.forEach((image) => knownIds.add(image.id));
        if (res.images[0]) setSelectedId(res.images[0].id);
        res.images.forEach((image) => void ensureSrc(image));
        setGenDone(i + 1);
      }
      // A generation can change server-side status: Speed=Auto compiles the transformer on the 3rd LoRA-free run (supports_lora flips to false), so without a refresh the LoRA picker stays enabled and the next LoRA run fails.
      // Cheap status GET; also picks up any other drift.
      if (isMounted.current) void refreshStatus();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Image generation failed");
    } finally {
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      genPollTimer.current = null;
      if (genVisibilityListener.current) {
        document.removeEventListener("visibilitychange", genVisibilityListener.current);
        genVisibilityListener.current = null;
      }
      setBusy(null);
      setGenDone(null);
      setGenStep(null);
    }
  }, [prompt, negativePrompt, width, height, steps, guidance, seed, batchSize, count, workflow, initImage, maskImage, strength, extendPct, extendSides, upscaleFactor, upscaleStrength, referenceImages, loras, loraCapable, controlnetCapable, controlnetId, controlImage, controlType, controlStrength, ensureSrc, loadGallery, refreshStatus]);

  // Keep the active workflow valid for the loaded model: an edit-only model has no Create/Transform tabs, a base model no Edit tab. Snap to the first supported workflow when capabilities change.
  // supported workflow whenever the loaded model's capabilities change.
  useEffect(() => {
    if (!status?.loaded) return;
    const wf = status.workflows ?? [];
    const ok = (id: WorkflowId) => {
      const t = WORKFLOW_TABS.find((x) => x.id === id);
      if (!t) return false;
      return t.requires === null ? wf.includes("txt2img") : wf.includes(t.requires);
    };
    if (!ok(workflow)) {
      const first = WORKFLOW_TABS.find((t) => ok(t.id));
      if (first) setWorkflow(first.id);
    }
  }, [status?.loaded, status?.workflows, workflow]);

  // Create (requires null) needs txt2img once a model is loaded: an edit-only model (workflows: ["edit"]) has no text-to-image mode, so its Create row is disabled and Edit is the only enabled one.
  // With nothing loaded, Create stays available so the user can pick a model.
  const workflowEnabled = (t: (typeof WORKFLOW_TABS)[number]) => {
    const wf = status?.workflows ?? [];
    return t.requires === null
      ? !status?.loaded || wf.includes("txt2img")
      : wf.includes(t.requires);
  };
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
      {/* The dense transformer_quant fast path only engages on the GGUF kind; on a loaded safetensors pipeline / single-file model it is a silent no-op, so gate the control to GGUF (or nothing loaded) and otherwise show why it is unavailable. */}
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
      {/* A resident full pipeline is reloadable by repo id alone (the resident effect wires lastLoad for it), so it keeps Reapply even before any user-initiated load; GGUF/single_file residents have no reload target and hide the button. */}
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
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* Top: the model selector, kept at the chat tab exact position so the shared element matches. Load progress shows in a chat-style toast, not here. */}
      <div className="relative flex h-[48px] shrink-0 items-start justify-between pl-2 pr-2 pt-[11px]">
        <div className="flex items-center gap-2">
          {pageMode === "train" ? (
            <TrainBaseSelector
              families={trainFamilies}
              familyName={trainFamilyName}
              base={trainBaseChoice}
              onSelect={(family, repo) => {
                // Set both together: the panel reseed effect keeps a base that is valid for the new family, so the pick survives the family switch.
                setTrainFamilyName(family);
                setTrainBaseChoice(repo);
              }}
            />
          ) : (
            <ModelSelector
              models={MODELS}
              value={status?.loaded ? status.repo_id ?? undefined : undefined}
              activeGgufVariant={quant}
              onValueChange={handleModelSelect}
              onEject={status?.loaded ? handleUnload : undefined}
              variant="ghost"
              className="!h-[34px]"
              task={IMAGE_GEN_TASKS}
              catalog={IMAGE_CATALOG}
              placeholder="Select image model"
              open={active && selectorOpen}
              onOpenChange={(o) => setSelectorOpen(active && o)}
            />
          )}
        </div>
        {/* Create | Train page-mode switch, centered on the page rather than tied to the selector width (the selector stays leftmost, its position shared with Chat). PillTabs is the app segmented control; icons match the sidebar New chat / Train rows. */}
        <div className="pointer-events-none absolute inset-x-0 top-[11px] flex justify-center">
          <PillTabs
            ariaLabel="Page mode"
            value={pageMode}
            onValueChange={(v) => setPageMode(v as "create" | "train")}
            fit={true}
            className="pointer-events-auto h-[34px] [&>button]:h-[34px] [&>button]:px-10"
            tabs={[
              {
                value: "create",
                label: "Create",
                icon: (
                  <HugeiconsIcon icon={PencilEdit02Icon} className="size-3.5" />
                ),
              },
              {
                value: "train",
                label: "Train",
                icon: (
                  <HugeiconsIcon
                    icon={TestTubeOutlineIcon}
                    className="size-3.5"
                  />
                ),
              },
            ]}
          />
        </div>
        <div className="flex items-center gap-2">
          {/* Single fixed toggle for the right-docked Advanced panel (mirrors Chat settings toggle, same icon in both states so it never moves). Highlighted when open, and hidden while training since it only tunes loads. */}
          {pageMode === "create" && (
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  onClick={() => setAdvancedOpen((o) => !o)}
                  aria-label={advancedOpen ? "Hide advanced options" : "Show advanced options"}
                  aria-pressed={advancedOpen}
                  className={cn(
                    "flex h-[34px] w-[34px] items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    advancedOpen
                      ? "bg-muted text-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <HugeiconsIcon icon={LayoutAlignRightIcon} className="size-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Advanced options</TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
      {/* Train mode: the full-page training workspace. Kept unmounted in Create mode so its polling stops; Create own state (gallery, model, workflow) is untouched. */}
      {pageMode === "train" ? (
        <DiffusionTrainPanel
          active={active && pageMode === "train"}
          loadedFamily={status?.family ?? null}
          loadedBaseRepo={
            // Prefer base_repo (the full diffusers pipeline) over repo_id: for a GGUF or single-file load repo_id is the checkpoint path, not a trainable base.
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
      /* Controls rail + preview canvas. No cards: both sit on the page background, split by a rule, on the Hub centered measure. No bottom padding, so the rule runs the full page height; each pane pads its own content. */
      <div className="mx-auto flex min-h-0 w-full min-w-0 max-w-[1100px] flex-1 overflow-hidden px-5 pt-9 sm:px-8">
        <div className="flex w-[368px] shrink-0 flex-col overflow-hidden border-r border-border/60">
          {/* pl-0.5 keeps focus rings off the scroll container's edge. */}
          <div className="hover-scrollbar flex min-h-0 flex-col gap-4 overflow-y-auto pb-7 pl-0.5 pr-7">
            {/* Seven workflows do not fit a segmented strip in this rail, so: a dropdown. A row stays disabled until the loaded model supports it (status.workflows). */}
            <div className="grid gap-1.5">
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <button
                    type="button"
                    aria-label="Workflow"
                    className="corner-squircle flex h-9 w-full items-center gap-2 rounded-xl bg-foreground/[0.07] px-3 text-left transition-colors hover:bg-foreground/[0.11] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:bg-foreground/[0.12] dark:hover:bg-foreground/[0.16]"
                  >
                    <HugeiconsIcon
                      icon={activeWorkflowTab.icon}
                      className="size-4 shrink-0 text-muted-foreground"
                    />
                    <span className="min-w-0 flex-1 truncate text-xs font-medium text-foreground">
                      {activeWorkflowTab.label}
                    </span>
                    <HugeiconsIcon
                      icon={ArrowDown01Icon}
                      className="size-4 shrink-0 text-muted-foreground"
                    />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start">
                  {WORKFLOW_TABS.map((t) => {
                    const enabled = workflowEnabled(t);
                    return (
                      // Rows stay one line; the description arrives on hover. The trigger wraps the row: a disabled item has pointer-events: none.
                      <Tooltip key={t.id} delayDuration={550}>
                        <TooltipPrimitive.Trigger asChild={true}>
                          <div>
                            <DropdownMenuItem
                              disabled={!enabled}
                              onSelect={() => setWorkflow(t.id)}
                              className="gap-2"
                            >
                              <HugeiconsIcon
                                icon={t.icon}
                                className="size-4 shrink-0 text-muted-foreground"
                              />
                              <span className="min-w-0 flex-1 truncate text-xs font-medium">
                                {t.label}
                              </span>
                              <HugeiconsIcon
                                icon={CheckmarkCircle02Icon}
                                className={cn(
                                  "size-3.5 shrink-0",
                                  workflow === t.id ? "text-foreground" : "invisible",
                                )}
                              />
                            </DropdownMenuItem>
                          </div>
                        </TooltipPrimitive.Trigger>
                        <TooltipContent side="right" sideOffset={6}>
                          {enabled
                            ? t.hint
                            : `Needs a loaded model that supports ${t.label.toLowerCase()}`}
                        </TooltipContent>
                      </Tooltip>
                    );
                  })}
                </DropdownMenuContent>
              </DropdownMenu>
              {/* Description sits under the closed trigger, matching the Field hints below it, so the button stays one line. */}
              <p className="px-0.5 text-ui-11p5 leading-snug text-muted-foreground">
                {activeWorkflowTab.hint}
              </p>
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
                          onClick={() => setExtendSides((s) => ({ ...s, [key]: !s[key] }))}
                          className={cn(
                            "rounded-lg px-2 py-1.5 text-xs font-medium ring-1 transition-colors",
                            on
                              ? "bg-primary/10 text-foreground ring-primary/40"
                              : "text-muted-foreground ring-border hover:text-foreground",
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
                          // Keep the slot in place (empty string when cleared) so other slots do not renumber mid-edit; empty slots are dropped only at send time and removed explicitly via the button below. Stable key={i}.
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
            {/* LoRA adapters: shown whenever the loaded model + quant can apply them. Type a Hugging Face repo id (owner/name, or owner/name:weight-file.safetensors) or pick a discovered adapter; stack multiple, each with a 0-2 weight. The backend owns how they apply; the UI only sends {id, weight}. */}
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
                      key={sel.id || i}
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
                        // Prefill with the first unused suggestion when a curated catalog exists, else an empty row the user fills with a Hugging Face repo id.
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
            {/* ControlNet: shown when the loaded model supports it, a model is discoverable, and the plain text-to-image workflow is active (v1 conditions txt2img only). Pick a model, add a control image, choose how to derive the map, set the strength. */}
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
            {/* A negative prompt only does anything with guidance on, so hide it at guidance 0 (Z-Image-Turbo default) instead of showing a dead field. */}
            {guidance > 0 && (
              <Field
                label="Negative prompt"
                hint="What to steer the image away from. Only used when guidance is above 0."
              >
                <Textarea
                  rows={2}
                  placeholder="What to avoid (optional)"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                />
              </Field>
            )}

            <Field
              label="Aspect ratio"
              hint="Pick a ratio to lock the proportions, then set the size with the sliders. Flip swaps width and height. Sizes run from 256 to 2048 in steps of 16. Z-Image is trained around 1 megapixel, so much larger sizes can look worse."
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
                        {key === "custom" ? "Custom" : key}
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
            <SliderField label="Width" value={width} min={MIN_DIM} max={MAX_DIM} step={16} onChange={changeWidth} />
            <SliderField label="Height" value={height} min={MIN_DIM} max={MAX_DIM} step={16} onChange={changeHeight} />

            <SliderField
              label="Steps"
              hint="9 is the recommended setting for Z-Image-Turbo. More steps rarely help."
              value={steps}
              min={1}
              max={50}
              step={1}
              onChange={setSteps}
            />
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
            <Field label="Seed" hint="Leave empty for a fresh random seed each run.">
              <Input
                placeholder="Random if empty"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </Field>

            <Button onClick={handleGenerate} disabled={busy !== null || !status?.loaded}>
              {busy === "generating" ? <Spinner className="mr-2 size-4" /> : null}
              {busy === "generating" && genDone != null && count > 1
                ? `Generating ${genDone}/${count}…`
                : "Generate"}
            </Button>

          </div>
        </div>

        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden pl-2">
          <div className="hover-scrollbar relative flex flex-1 items-center justify-center overflow-auto p-6">
            {selected && selectedSrc ? (
              <>
                <img
                  src={selectedSrc}
                  alt={selected.prompt}
                  className="max-h-full max-w-full rounded-xl object-contain shadow-sm"
                />
                {/* Actions grouped in one glass toolbar so they stay legible over any image and read as a unit instead of blending into the canvas. Size/seed live in the Recipe popover. */}
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
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label="Delete image"
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => void handleDelete(selected.id)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Delete</TooltipContent>
                  </Tooltip>
                </div>
              </>
            ) : selected ? (
              // The selected record's blob is still loading — spin in place.
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

            {/* Live generation progress: a per-step bar with ETA, centered when there is nothing else to show, tucked at the bottom over an image. */}
            {busy === "generating" && (
              <div
                className={cn(
                  "pointer-events-none absolute flex justify-center px-4",
                  selectedSrc ? "inset-x-0 bottom-4" : "inset-0 items-center",
                )}
              >
                <div className="w-72 max-w-full rounded-xl bg-background/85 p-3 shadow-lg ring-1 ring-border backdrop-blur">
                  <ModelLoadDescription
                    // Drop the chat min-height: this floating card has no layout to stabilise, and it would center the thin bar and leave empty space above it.
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
              className="hover-scrollbar flex shrink-0 gap-2 overflow-x-auto border-t border-foreground/10 p-3"
              onScroll={(e) => {
                // Near the right edge: pull the next older page (infinite scroll).
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past images stay visible and browsable while the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center rounded-lg bg-muted/50 ring-2 ring-primary/30">
                  <Spinner className="size-5 text-muted-foreground" />
                </div>
              )}
              {images.map((image) => (
                <button
                  key={image.id}
                  type="button"
                  data-image-id={image.id}
                  onClick={() => setSelectedId(image.id)}
                  className="relative size-16 shrink-0 overflow-hidden rounded-[10px] bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
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
                  {/* Selection marker on a non-focusable overlay, so the button own focus state can never mask it. */}
                  {image.id === selected?.id && (
                    <span className="pointer-events-none absolute inset-0 rounded-lg border-2 border-primary" />
                  )}
                </button>
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

        {/* Right-docked Advanced panel (mirrors Chat settings panel): closed by default, opened by the single fixed top-bar toggle above (which never moves between states), so the optimisation controls are discoverable without being docked open or buried at the bottom of the left rail. */}
        {advancedOpen && (
          <div className="ml-4 flex w-[300px] shrink-0 flex-col overflow-hidden border-l border-border/60 pl-4">
            <div className="flex h-[52px] shrink-0 items-center border-b border-border/60">
              <span className="flex items-center gap-1.5 text-sm font-semibold text-foreground">
                <HugeiconsIcon icon={Settings02Icon} className="size-4" />
                Advanced
              </span>
            </div>
            <div className="hover-scrollbar flex flex-col gap-3 overflow-y-auto py-4 pr-2">
              <p className="text-xs text-muted-foreground">
                Load-time tuning. Changes apply on the next load; Reapply reloads the current model.
              </p>
              {advancedControls}
            </div>
          </div>
        )}
      </div>
      )}
    </div>
  );
}
