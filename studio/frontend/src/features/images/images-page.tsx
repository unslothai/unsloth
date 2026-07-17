// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowLeftRightIcon,
  ArrowReloadHorizontalIcon,
  Delete02Icon,
  Download01Icon,
  ImageAdd02Icon,
  InformationCircleIcon,
  LayoutAlignRightIcon,
  PencilEdit02Icon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { InfoHint } from "@/components/ui/info-hint";
import { ModelSelector } from "@/components/assistant-ui/model-selector";
import { IMAGE_GEN_TASKS } from "@/components/assistant-ui/model-selector/pickers";
import {
  IMAGE_CATALOG,
  catalogToModelOptions,
  loadSpecFor,
} from "@/components/assistant-ui/model-selector/model-catalog";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/components/assistant-ui/model-selector/types";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { formatBytes, formatEta } from "@/features/hub/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import {
  type ControlNetSpecInput,
  type DiffusionControlNetInfo,
  type DiffusionGenerateProgress,
  type DiffusionLoadProgress,
  type DiffusionLoraInfo,
  type DiffusionStatus,
  type GalleryImage,
  type LoraSpecInput,
  deleteGalleryImage,
  fetchGalleryObjectUrl,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  getGallery,
  getGenerateProgress,
  listDiffusionControlNets,
  listDiffusionLoras,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";
import { DiffusionTrainPanel } from "./train/diffusion-train-panel";

// Curated models come from the shared catalog: one canonical group per model, its artifacts
// (GGUF / FP8 / bnb-4bit / BF16) as data, and the load kind per artifact via loadSpecFor
// (replacing the old SAFETENSORS_MODELS table). The picker renders groups with a format
// second level and routes bare clicks to the best artifact for the device.
const MODELS: ModelOption[] = catalogToModelOptions(IMAGE_CATALOG);

// Workflow tabs. `requires` is the backend workflow id (status.workflows) that must
// be supported by the loaded model for the tab to enable; null = always available.
type WorkflowId = "create" | "transform" | "inpaint" | "extend" | "upscale" | "reference" | "edit";

const WORKFLOW_TABS: Array<{
  id: WorkflowId;
  label: string;
  requires: string | null;
  hint?: string;
}> = [
  { id: "create", label: "Create", requires: null, hint: "Generate a new image from a prompt" },
  {
    id: "transform",
    label: "Transform",
    requires: "img2img",
    hint: "Redraw an uploaded image guided by your prompt (img2img)",
  },
  {
    id: "inpaint",
    label: "Inpaint",
    requires: "inpaint",
    hint: "Paint over a region to regenerate just that area, keeping the rest",
  },
  {
    id: "extend",
    label: "Extend",
    requires: "outpaint",
    hint: "Outpaint: grow the canvas and fill the new edges from your prompt",
  },
  {
    id: "upscale",
    label: "Upscale",
    requires: "upscale",
    hint: "Hires fix: enlarge an uploaded image and re-detail it at higher resolution",
  },
  {
    id: "reference",
    label: "Reference",
    requires: "reference",
    hint: "Generate a new image guided by a reference image + your prompt (FLUX.2)",
  },
  {
    id: "edit",
    label: "Edit",
    requires: "edit",
    hint: "Instruction editing: change an image with a prompt (Qwen-Image-Edit)",
  },
];

// Per-model generation defaults (steps + guidance), matched by repo-id substring, most specific
// first. Distilled "turbo/schnell" models want few steps and little guidance; the full "dev" models
// want more steps and real CFG. Generation defaults when the model is unrecognised: the distilled
// few-step / no-CFG shape. Also seeds the sliders' initial state.
const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  { match: "z-image-turbo", steps: 9, guidance: 0 },
  // Krea 2 Raw is the undistilled base (also inference-loadable): its card runs 52 steps at
  // guidance 3.5, so it must precede the distilled "krea-2" key below or a Raw load would run
  // the 8-step recipe and produce garbage.
  { match: "krea-2-raw", steps: 52, guidance: 3.5 },
  // Krea 2 Turbo is distilled (TDM): 8 steps, no CFG. "krea-2" then covers Turbo (and any
  // other krea id) but Raw, which is matched more specifically above.
  { match: "krea-2", steps: 8, guidance: 0 },
  { match: "flux.1-schnell", steps: 4, guidance: 0 },
  // Kontext (editing) before the generic flux.1: ~28 steps, lower guidance (~2.5).
  { match: "kontext", steps: 28, guidance: 2.5 },
  // Krea's FLUX.1-dev finetune runs its card recipe (28 steps, guidance 4.5); before
  // the generic flux.1 key. It never hits the krea-2 keys above ("krea-2" is not a
  // substring of "flux.1-krea-dev").
  { match: "flux.1-krea", steps: 28, guidance: 4.5 },
  { match: "flux.1", steps: 28, guidance: 3.5 },
  { match: "flux.2-klein", steps: 4, guidance: 0 },
  // FLUX.2-dev is the full (non-distilled) model: more steps + real guidance, unlike klein.
  { match: "flux.2-dev", steps: 28, guidance: 4 },
  { match: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", steps: 20, guidance: 4 },
  // Ideogram 4's model-card settings (48 steps, guidance 7). At exactly these defaults the backend
  // keeps the pipeline's recommended tapered guidance schedule instead of a flat constant.
  { match: "ideogram", steps: 48, guidance: 7 },
  // SDXL: Turbo is distilled (few steps, no CFG); base/full SDXL wants ~30 steps and
  // real CFG (~7). "sdxl-turbo" must precede the generic "sdxl" substring match.
  { match: "sdxl-turbo", steps: 3, guidance: 0 },
  { match: "stable-diffusion-xl", steps: 30, guidance: 7 },
  { match: "sdxl", steps: 30, guidance: 7 },
];

function defaultsFor(repoId: string): { steps: number; guidance: number } {
  const id = repoId.toLowerCase();
  // The fallback is only hit for an unrecognised on-device image GGUF; a curated
  // entry covers every model in MODELS.
  return MODEL_DEFAULTS.find((d) => id.includes(d.match)) ?? DEFAULT_GEN;
}

// Common aspect ratios (landscape; Flip gives the portrait mirror). Picking one
// locks the W:H proportion; the sliders set the size.
const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};
const ASPECT_OPTIONS = ["custom", ...Object.keys(ASPECT_RATIOS)];

// Friendly labels for ControlNet control types. "canny" traces edges from a source image;
// every other type is an already-made map (passthrough/depth/pose/...). Unknown types fall
// back to a capitalized "(map)" label so a new backend type still renders.
const CONTROL_TYPE_LABELS: Record<string, string> = {
  passthrough: "Passthrough (already a map)",
  canny: "Canny (trace edges)",
  depth: "Depth (map)",
  pose: "Pose (map)",
};

// Z-Image accepts 256–2048, in multiples of 16. Snap any value into range.
const MIN_DIM = 256;
const MAX_DIM = 2048;
// Convenient drag range for the Runs slider. The number box accepts higher typed values on purpose
// (set it large to generate all night); the loop only floors at 1 and ignores non-numeric input.
const RUNS_SLIDER_MAX = 128;
function snapDim(value: number): number {
  if (!Number.isFinite(value)) return 1024;
  return Math.min(MAX_DIM, Math.max(MIN_DIM, Math.round(value / 16) * 16));
}

// The ratio key (compared by long:short, so it survives orientation) matching
// width/height, plus whether the current orientation is portrait.
function matchAspect(width: number, height: number): { key: string; portrait: boolean } {
  const target = Math.max(width, height) / Math.min(width, height);
  const found = Object.entries(ASPECT_RATIOS).find(
    ([, [a, b]]) => Math.abs(target - a / b) < 0.01,
  );
  return { key: found ? found[0] : "custom", portrait: height > width };
}

// Module cache of the backend-persisted gallery, so a tab switch re-renders instantly. Object URLs
// are revoked only on delete (not unmount), so they stay valid across remounts.
const galleryCache: {
  images: GalleryImage[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  srcById: Map<string, string>;
  // Ids with a fetch in flight, so concurrent ensureSrc calls don't double-fetch
  // (and leak the duplicate object URL).
  inflight: Set<string>;
} = {
  images: [],
  hasMore: false,
  selectedId: null,
  quant: null,
  srcById: new Map(),
  inflight: new Set(),
};

// Images loaded per infinite-scroll page.
const PAGE_SIZE = 50;

// Export filename, e.g. Unsloth_20260624-143005_123.png. Batch siblings share
// the seed + timestamp, so they get a "_<n>" suffix past the first one.
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

// PNG saves the stored bytes verbatim (keeps the embedded recipe metadata);
// JPEG / WebP re-encode client-side from the already-fetched object URL. JPEG
// has no alpha, so it is flattened onto white first.
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

// Bar label for an in-flight generation: step count plus an ETA once it's known
// (formatEta returns "" for non-positive, so the last step shows just the step).
function genStepLabel(p: DiffusionGenerateProgress): string {
  // Text encoding (and any first-run warmup) happens before the first scheduler tick, so step 0
  // means "working, not denoising yet" -- label it that way instead of sitting on "Step 0/N".
  if (p.step === 0) return "Preparing (text encoding + warmup)…";
  const base = `Step ${p.step}/${p.total_steps}`;
  const eta = p.eta_seconds != null ? formatEta(p.eta_seconds) : "";
  return eta ? `${base} · ~${eta}` : base;
}

// The chat tab's model-load toast styling, reused verbatim so the diffusion
// load toast is visually identical (persistent, progress bar, same chrome).
const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

// Render the chat ModelLoadDescription for a progress poll. The base repo (text-encoder/VAE)
// downloads alongside the GGUF, so the total exceeds the picked quant's size.
function loadToastDescription(p: DiffusionLoadProgress) {
  // "Downloading" only when bytes actually remain to fetch — a cached model (or
  // the pre-estimate window, total still 0) shouldn't claim a download.
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

// Toast args mirroring chat's: persistent, closeable, content in `description`.
// Pass `id` to update the existing toast in place instead of stacking a new one.
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

// Mirrors the Train page's SliderRow (studio/sections/params-section.tsx):
// label + standard Slider + number input, same classes.
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
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
        {label}
        {hint && <InfoHint>{hint}</InfoHint>}
      </span>
      <div className="flex items-center gap-3">
        <Slider
          value={[value]}
          onValueChange={([v]) => onChange(v)}
          min={min}
          max={max}
          step={step}
          className="w-32"
        />
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          // The slider is the primary control, so the native number spinners are redundant --
          // and on this narrow field their arrows overlapped the value. Remove them on every
          // engine: appearance:textfield for Firefox, zeroed webkit inner/outer spin buttons.
          className="w-14 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [appearance:textfield] [&::-webkit-outer-spin-button]:m-0 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:m-0 [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>
    </div>
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

// The engaged value of a resolved Advanced control, formatted for its "Auto: X" badge.
// Short scheme/mode tokens go uppercase (INT8, FP8, FBCACHE); the attention backend the
// backend reports as `_native_cudnn` shows as cuDNN; cpu_offload's boolean shows On/Off.
function formatResolvedValue(key: string, value: string | boolean | null): string {
  if (key === "cpu_offload") return value ? "On" : "Off";
  if (value === null || value === "") return "Off";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === "_native_cudnn" || value.toLowerCase() === "cudnn") return "cuDNN";
  // Deferred speed auto: the dense pipe stays exact/eager and compiles on the
  // 3rd image of the session (the tooltip carries the full reason).
  if (value === "deferred") return "On from 3rd image";
  return value.toUpperCase();
}

// The "Auto: X" badge for one Advanced control: rendered only when the backend resolved
// that control itself (source === "auto"); an explicit user choice renders nothing. The
// reason is surfaced as a hover tooltip. Muted pill matching the panel's other chips.
function ResolvedBadge({
  status,
  controlKey,
}: {
  status: DiffusionStatus | null;
  controlKey: string;
}) {
  const resolved = status?.resolved?.[controlKey];
  if (!resolved || resolved.source !== "auto") return null;
  return (
    <span
      title={resolved.reason || undefined}
      className="shrink-0 rounded-sm bg-muted px-1 py-px text-[9px] font-medium uppercase tracking-wider text-muted-foreground"
    >
      Auto: {formatResolvedValue(controlKey, resolved.value)}
    </span>
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
  // A short always-visible description under the row (the hint tooltip carries the full
  // detail). Used for controls whose label alone does not convey what they do.
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
      {desc && <p className="text-[11px] leading-snug text-muted-foreground/70">{desc}</p>}
    </div>
  );
}

// Source-image picker for the Transform (img2img) workflow: click or drag-drop an
// image, read it to a data URL the generate request sends as init_image. Shows a
// thumbnail preview with a Clear button once an image is set.
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
      <div className="relative overflow-hidden rounded-xl border border-border">
        <img src={value} alt="Source" className="max-h-44 w-full object-contain bg-muted/30" />
        <Button
          type="button"
          variant="secondary"
          size="icon"
          aria-label="Remove source image"
          title="Remove"
          className="absolute right-1.5 top-1.5 size-7"
          onClick={() => {
            onChange(null);
            if (inputRef.current) inputRef.current.value = "";
          }}
        >
          <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
        </Button>
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

// A brush-based mask editor for inpainting. Shows the source image with a paintable overlay
// and exports a grayscale PNG mask at the image's NATIVE resolution (diffusers convention:
// white = repaint, black = keep). Strokes draw to both a visible tinted overlay (feedback)
// and an offscreen mask canvas in lockstep, so the exported mask matches what the user sees.
// `brushPct` sizes the brush as a fraction of the shorter side, so it's resolution-consistent.
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

  // (Re)initialise both canvases whenever the image changes or Clear is pressed:
  // size them to the image's native pixels and reset the mask to all-black (keep all).
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
    <div className="relative overflow-hidden rounded-xl border border-border bg-muted/30">
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

// Build the (image, mask) pair for outpaint by reusing the inpaint backend: grow the canvas
// by `pct` per dimension on the selected sides, edge-bleed the original pixels into the new
// bands (so the VAE encodes plausible content), and mask the new bands white (= repaint) with
// a small overlap into the original on each grown side so the seam blends.
async function buildOutpaint(
  src: string,
  sides: ExtendSides,
  pct: number,
): Promise<{ image: string; mask: string }> {
  const img = await loadImage(src);
  const w = img.naturalWidth;
  const h = img.naturalHeight;
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

  // The grown canvas can exceed the backend's 4096px-per-side decode limit (e.g. a 2048px
  // source at 100% on both sides -> 6144px), which would 400 the load. Scale the built pair
  // down proportionally to fit so Extend still returns an outpaint. The backend rounds to /16,
  // so exact dims aren't required.
  const MAX_SIDE = 4096;
  const longest = Math.max(nw, nh);
  if (longest > MAX_SIDE) {
    const scale = MAX_SIDE / longest;
    const sw = Math.max(1, Math.round(nw * scale));
    const sh = Math.max(1, Math.round(nh * scale));
    const scaleCanvas = (source: HTMLCanvasElement): HTMLCanvasElement => {
      const dst = document.createElement("canvas");
      dst.width = sw;
      dst.height = sh;
      const dctx = dst.getContext("2d");
      if (!dctx) throw new Error("Could not scale the extended canvas");
      dctx.drawImage(source, 0, 0, sw, sh);
      return dst;
    };
    return {
      image: scaleCanvas(ic).toDataURL("image/png"),
      mask: scaleCanvas(mc).toDataURL("image/png"),
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
  // Controlled + force-closed off-tab: PopoverContent portals to body, so the
  // hidden/inert page wrapper can't contain it when the page is kept mounted.
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
          <p className="text-[11px] text-muted-foreground">{formatTimestamp(image.created_at)}</p>
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

export function ImagesPage({ active = true }: { active?: boolean }) {
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(
    "a tiny ginger sloth coding in a sunlit treehouse, photorealistic",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  // width/height are the source of truth; `aspect` locks their proportion
  // ("custom" = free) and `portrait` tracks orientation, so Flip keeps the lock.
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [aspect, setAspect] = useState("1:1");
  const [portrait, setPortrait] = useState(false);
  // Z-Image-Turbo official defaults: 9 steps (= 8 DiT forwards), guidance 0
  // (distilled CFG-free; a negative prompt is ignored at this guidance).
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const [seed, setSeed] = useState("");
  // Batch size = images per forward pass (VRAM-heavy); count = sequential loops.
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);
  // Active workflow tab. "create" = text-to-image; "transform" = img2img; "inpaint" =
  // mask-guided redraw. More tabs (edit/extend/control/enhance) slot in here.
  const [workflow, setWorkflow] = useState<WorkflowId>("create");
  // Transform (img2img) / Inpaint inputs: the uploaded source image as a data URL, and
  // the denoise strength (how far to redraw it: low = keep source, high = reimagine).
  const [initImage, setInitImage] = useState<string | null>(null);
  const [strength, setStrength] = useState(0.6);
  // Inpaint mask (grayscale PNG data URL, white = repaint), the brush size as a percent
  // of the image's shorter side, and a key bumped to clear the painted mask.
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [brushPct, setBrushPct] = useState(8);
  const [maskResetKey, setMaskResetKey] = useState(0);
  // Extend (outpaint): how far to grow each dimension and which sides to grow. Reuses the
  // inpaint backend by building a padded image + border mask at generate time.
  const [extendPct, setExtendPct] = useState(25);
  const [extendSides, setExtendSides] = useState<ExtendSides>({
    left: true,
    right: true,
    top: true,
    bottom: true,
  });
  // Upscale (hires fix): the enlargement factor and the (low) denoise strength used to
  // re-detail the enlarged image. The backend caps the factor and rounds the target size.
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [upscaleStrength, setUpscaleStrength] = useState(0.35);
  // Reference (FLUX.2): up to 3 ADDITIONAL reference images beyond the primary one, combined
  // by the model (subject + style, character + scene).
  const [referenceImages, setReferenceImages] = useState<string[]>([]);
  // LoRA adapters selected for the next generation (id + weight), plus the list the picker
  // offers. Applied at generate time; available adapters are refreshed per loaded family.
  const [loras, setLoras] = useState<LoraSpecInput[]>([]);
  const [availableLoras, setAvailableLoras] = useState<DiffusionLoraInfo[]>([]);
  // Page mode: "create" is the generation workspace; "train" is the full-page LoRA
  // training workspace. Independent of the loaded generation model.
  const [pageMode, setPageMode] = useState<"create" | "train">("create");
  // Bumped when a training run completes, to force the LoRA discovery effect to rescan so
  // a freshly-trained adapter appears in the picker without a model reload.
  const [loraRefreshKey, setLoraRefreshKey] = useState(0);
  // ControlNet for the next generation: the chosen model id, a control image (data URL),
  // how to derive the control map, and the conditioning strength. Available models refresh
  // per loaded family; applied at generate time only when a model + control image are set.
  const [controlnetId, setControlnetId] = useState<string>("");
  const [controlImage, setControlImage] = useState<string | null>(null);
  // Free-form: a union ControlNet advertises depth/pose/etc alongside the preprocessing
  // "canny", and the backend maps the exact control_type to the union control_mode. The
  // picker is built from the selected model's control_types, so it isn't limited to two.
  const [controlType, setControlType] = useState<string>("passthrough");
  const [controlStrength, setControlStrength] = useState(0.7);
  const [availableControlNets, setAvailableControlNets] = useState<DiffusionControlNetInfo[]>([]);
  // Advanced options live in a right-docked panel (like Chat's settings panel). Closed by
  // default; a single fixed toggle in the top bar opens/closes it (the icon never moves).
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Advanced (load-time) options. "auto"/"off"/"none" map to the backend defaults
  // (sent through on load). They apply when a model loads; changing them while a model
  // is loaded shows a "Reapply" button that reloads the same model with the new values.
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
  // The last load descriptor, so "Reapply" can reload the same model with new advanced
  // options without the user re-picking it from the dropdown.
  const lastLoad = useRef<{ repoId: string; kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null>(
    null,
  );
  // Render-safe mirror of "lastLoad.current was set by a user-initiated load": a resident
  // GGUF/single_file model discovered by refresh carries no checkpoint filename in status, so
  // lastLoad stays null and Reapply would be dead. Set only from event handlers; the
  // resident-pipeline case is derived from status at render time (mirrors the video page).
  const [canReapply, setCanReapply] = useState(false);
  // Repo id whose defaults we've already seeded from a discovered resident model, so
  // we seed the sliders once per resident model and never clobber a later manual edit.
  const seededResident = useRef<string | null>(null);

  const [busy, setBusy] = useState<Busy>(null);
  // {done, total} while a multi-run generation is in flight (for the button).
  // Number of runs finished in the current multi-run generation (null = idle).
  // The total is just `count`, so it isn't stored separately.
  const [genDone, setGenDone] = useState<number | null>(null);
  // Live per-step progress (step / total + ETA) polled during generation.
  const [genStep, setGenStep] = useState<DiffusionGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // visibilitychange handler active while a generation poll runs: background tabs clamp
  // setInterval to >=1s (and can suspend it outright after ~5 min), so returning to the
  // tab fires one immediate poll instead of waiting for a throttled tick.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  // Controlled so the body-portaled overlays force-close when this page is mounted
  // but off-tab (a hidden/inert parent can't contain a body portal): the model
  // selector and the aspect-ratio dropdown.
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [aspectOpen, setAspectOpen] = useState(false);
  // Records come from the backend (durable); srcById maps each id to its object
  // URL (loaded images) or data URL (the one just generated).
  const [images, setImages] = useState<GalleryImage[]>(() => galleryCache.images);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    Object.fromEntries(galleryCache.srcById),
  );
  // Guards a "load more" so a fast scroll can't fire several at once.
  const loadingMore = useRef(false);
  // False once the page truly unmounts (app close / chat-only eject). The page now
  // stays mounted across tab switches, so a switch does NOT flip this -- a batch keeps
  // generating off-tab; the multi-run loop only stops on a real unmount.
  const isMounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place (chat-style).
  const loadToastId = useRef<string | number | null>(null);
  // Last load-progress signature shown, so a tick that moved nothing skips the toast.
  const lastLoadSig = useRef<string | null>(null);
  // The quant to restore if the optimistic swap fails. A same-repo quant change sets `quant`
  // immediately for picker feedback; if the load then fails AFTER starting (error/eviction
  // during download) the old pipeline stays loaded, so the poll rolls the label back rather
  // than advertise the failed quant. `{ prev }` distinguishes "revert to null" from "nothing pending".
  const quantRevert = useRef<{ prev: string | null } | null>(null);
  // A trained adapter awaiting deployment: after Deploy loads the base, the LoRA discovery
  // effect applies this once the model is loaded + LoRA-capable for the matching family.
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

  // Refresh the LoRA picker's suggestions when the loaded family changes. A LoRA is trained
  // for a specific base family, so a real model SWAP invalidates the selection -- clear it then.
  // But do NOT clear on the first load or an unload: a user can restore a saved recipe (setting
  // loras) BEFORE the model finishes loading, and clearing on that load->capable transition
  // would drop the restored adapters. We track the previous family in a ref and clear only on a
  // change to a different loaded family. We also do NOT filter the selection against the catalog:
  // a valid pick can be a free-text HF repo id absent from the (often empty) curated list.
  const loraCapable = Boolean(status?.loaded && status?.supports_lora);
  const prevLoraFamilyRef = useRef<string | null | undefined>(undefined);
  useEffect(() => {
    if (!loraCapable) {
      // Options are gone with the model, but keep the selection: it may have just been
      // restored while the model is (re)loading. It is only SENT when loraCapable, and a
      // real family swap below clears it.
      setAvailableLoras([]);
      return;
    }
    const fam = status?.family ?? null;
    const prev = prevLoraFamilyRef.current;
    if (prev != null && prev !== fam) {
      setLoras([]);
    }
    prevLoraFamilyRef.current = fam;
    // A just-deployed adapter: now that the base is loaded + LoRA-capable, apply it (after
    // the family-swap clear above so it isn't wiped). Only when the family matches what it
    // was trained for; otherwise warn instead of silently applying an incompatible adapter.
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
        // Clear only the OPTIONS on a failed catalog refresh. Unlike the catalog-only picker
        // below, this free-text picker holds selections (bare HF repo ids) valid without being
        // in the catalog; a transient refresh failure must not wipe them. Stale cross-family
        // selections are already cleared by the family-swap check above, and hidden LoRAs are
        // never sent (handleGenerate is gated on loraCapable).
        if (!cancelled) setAvailableLoras([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loraCapable, status?.family, loraRefreshKey]);

  // Refresh the ControlNet picker's options when the loaded model (family) changes, and clear
  // a stale selection the new model can't use so an incompatible ControlNet is never sent.
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

  // The control types offered for the selected ControlNet. A union model advertises
  // several (canny/depth/pose/passthrough); a plain model advertises its own. Fall back
  // to the preprocessing pair when nothing is selected.
  const controlTypeOptions = useMemo(() => {
    const cn = availableControlNets.find((c) => c.id === controlnetId);
    const types = cn?.control_types?.length ? cn.control_types : ["passthrough", "canny"];
    return types;
  }, [availableControlNets, controlnetId]);

  // Keep controlType valid for the selected model: if the current choice isn't among the
  // model's advertised types, snap to the first (prefer passthrough when offered).
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
      const url = await fetchGalleryObjectUrl(image.url);
      galleryCache.srcById.set(image.id, url);
      setSrcById((prev) => ({ ...prev, [image.id]: url }));
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
      page.images.forEach((image) => void ensureSrc(image));
    } catch {
      // Best-effort: a failed gallery load shouldn't block the page.
    }
  }, [ensureSrc]);

  // Load the next older page. offset = how many we've loaded so far. A newly
  // generated image gets a newer mtime, so it sorts to the front on the backend
  // too — the offset keeps pointing at the same older images.
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
      page.images.forEach((image) => void ensureSrc(image));
    } catch {
      // transient; the user can scroll again to retry
    } finally {
      loadingMore.current = false;
    }
  }, [ensureSrc]);

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
    const url = galleryCache.srcById.get(id);
    if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
    galleryCache.srcById.delete(id);
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
    setSeed(String(image.seed));
    setWidth(image.width);
    setHeight(image.height);
    // The batch shared one seed, so image batch_index>0 only reproduces by replaying the
    // whole batch: restore the batch size too (older recipes without it default to 1).
    setBatchSize(image.batch_size ?? 1);
    // Restore the LoRA selection. The recipe stores each adapter as an "id:weight" string;
    // the id itself may contain a colon (owner/name:weight-file.safetensors), so split on
    // the LAST colon to recover the weight. A malformed entry falls back to weight 1.
    setLoras(
      (image.loras ?? []).map((s) => {
        const idx = s.lastIndexOf(":");
        const w = idx > 0 ? Number.parseFloat(s.slice(idx + 1)) : NaN;
        return Number.isFinite(w) ? { id: s.slice(0, idx), weight: w } : { id: s, weight: 1 };
      }),
    );
    const m = matchAspect(image.width, image.height);
    setAspect(m.key);
    setPortrait(m.portrait);
    // Restore selected LoRA adapters from the recipe ("id:weight" strings); split on the LAST
    // colon so an id containing ':' is preserved. Unparseable entries are skipped, and a recipe
    // with no LoRAs clears the selection so the restore reproduces the image faithfully rather
    // than leaking a stale form selection.
    const restoredLoras: LoraSpecInput[] = [];
    for (const entry of image.loras ?? []) {
      const idx = entry.lastIndexOf(":");
      if (idx <= 0) continue;
      const id = entry.slice(0, idx);
      const weight = Number(entry.slice(idx + 1));
      if (id && Number.isFinite(weight)) restoredLoras.push({ id, weight });
    }
    setLoras(restoredLoras);
    // The control image isn't persisted, so clear any stale ControlNet selection.
    setControlnetId("");
    setControlImage(null);
    toast.success("Settings restored to inputs");
  }, []);

  // A locked ratio keeps the paired dimension in step while dragging; "custom"
  // frees both; Flip swaps W/H. ratioHW is the h/w for ratio [a,b] (long:short):
  // landscape h = w*b/a, portrait h = w*a/b.
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

  // Track mount so a long generate run stops issuing GPU work only on a true unmount (app
  // close / chat-only eject). The page stays mounted across tab switches (RootLayout keeps it
  // alive like chat), so a switch doesn't break the loop -- a batch keeps generating off-tab.
  // The mount-time refreshStatus and timer/toast cleanup live in the load-resume effect below,
  // so this one carries only the mount flag.
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  // Re-sync model status when the tab becomes active again: while off-tab the
  // diffusion model may have been evicted (e.g. a chat load claimed the GPU),
  // and the page no longer remounts on return to refresh it on its own.
  useEffect(() => {
    if (!active) return;
    void (async () => {
      await refreshStatus();
    })();
  }, [active, refreshStatus]);

  // Collapse the body-ported popovers when leaving the tab. Their open state is controlled
  // and force-closed via `active && open` while off-tab, but the underlying flag stays set, so
  // returning to /images would pop them back open. Reset it so the page returns neutral.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
    setAspectOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error",
  // updating the persistent toast in place each tick.
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
        return;
      }
      if (p.phase === "error") {
        dismissLoadToast();
        toast.error(p.error || "Failed to load model");
        setBusy(null);
        // A load that failed AFTER starting leaves the previous pipeline loaded, so
        // roll the optimistic quant label back to what is actually loaded (status
        // does not carry the quant, so refreshStatus alone can't correct it).
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        // A failed load may have freed a previously-loaded model, so resync to
        // the real backend state (the synchronous failure path does the same).
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // No load in flight and nothing loaded: the load was cancelled or evicted (e.g. a chat
        // load took the GPU) and the backend cleared its state. Terminal -- else this loop
        // spins forever and leaves busy stuck on "loading", deadening the picker and Generate.
        dismissLoadToast();
        setBusy(null);
        // Same optimistic-quant rollback as the error path: the swap did not take.
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      // Include bytes_total: the estimate lands as a 0→real jump while phase and
      // bytes_downloaded hold, and the toast shows the percentage off that total.
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

  // Re-enter the per-step poll for a generation already in flight on the backend that this page
  // did not start (another client, or a reload mid-generate), instead of showing a stale idle
  // view. generate-progress carries no terminal record, so refresh the gallery on completion to
  // merge any image saved after the mount fetch. Separate from handleGenerate's own loop.
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
      // A load runs on the backend as a daemon thread that survives navigation. On (re)mount,
      // resume tracking one still in flight so the page shows progress and updates on
      // completion, instead of a stale view that never polls.
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
      // Resume tracking a generation started elsewhere (another client, or before a reload) so the
      // page shows the in-flight run instead of a stale idle view. Mirrors the video page.
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
    // Stop polling if the page unmounts mid-load / mid-generate, and dismiss the
    // load toast — its poll loop is gone, so it would otherwise hang forever
    // (duration: Infinity) on whatever page the user navigated to.
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

  // Seed the generation sliders to a resident model's recipe when the page discovers one it
  // did not load itself (left loaded by a prior session or another route). refreshStatus() only
  // sets `status`; without this the sliders keep the unrecognised-model fallback (few-step /
  // no-CFG), so a resident full model (e.g. flux.1-dev) generates at 9 steps / guidance 0 and
  // produces garbage until re-picked. Guarded by lastLoad.current === null (a user-initiated
  // load seeds its own defaults) and a per-repo ref, so a manual edit after the seed is never
  // clobbered.
  useEffect(() => {
    const repoId = status?.loaded ? status.repo_id : null;
    if (!repoId) return;
    if (lastLoad.current) return;
    if (seededResident.current === repoId) return;
    seededResident.current = repoId;
    // Seed from base_repo (the resolved diffusers base, holding the family), not repo_id: a
    // GGUF/single_file/local-path resident has a repo_id with no family substring, so
    // defaultsFor(repoId) would fall back to the wrong distilled few-step/no-CFG recipe.
    const d = defaultsFor(status?.base_repo ?? repoId);
    setSteps(d.steps);
    setGuidance(d.guidance);
    // Wire "Reapply" to the resident model too, so an advanced-option reload works without
    // re-picking from the dropdown. Only a full pipeline load needs no checkpoint filename; a
    // resident GGUF/single_file carries no filename in status and the backend rejects such a
    // load without one (400 before it evicts), so leave lastLoad null for those (Reapply stays
    // hidden) rather than wire a reload that can never complete.
    const kind = status?.model_kind;
    if (kind === "pipeline") {
      lastLoad.current = { repoId, kind };
    }
  }, [status?.loaded, status?.repo_id, status?.base_repo, status?.model_kind]);

  const handleLoad = useCallback(
    // Resolves true when the background load STARTED (callers may revert
    // optimistic picker state on false); poll outcomes are handled internally.
    async (
      repoId: string,
      opts: {
        kind: "gguf" | "single_file" | "pipeline";
        filename?: string;
      },
    ): Promise<boolean> => {
      // Cancel any prior poll loop so two can't run at once.
      if (pollTimer.current) clearTimeout(pollTimer.current);
      setBusy("loading");
      // Show the chat-style toast immediately; the poll updates it by id.
      dismissLoadToast();
      lastLoadSig.current = null;
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS));
      // Remember what was loaded so "Reapply" can reload it with new advanced options. Snapshot
      // the prior target first: a load that fails to START (validation, gated repo, training
      // guard) leaves the previous model resident, so Reapply and the resident-default seeding
      // must keep pointing at it, not the failed pick.
      const prevLastLoad = lastLoad.current;
      lastLoad.current = { repoId, kind: opts.kind, filename: opts.filename };
      setCanReapply(true);
      try {
        // Returns immediately -- the load runs in the background; we poll for it. The backend
        // infers the family + base diffusers repo from the id. Forward the saved HF token so
        // gated bases (FLUX dev/klein) download. A pipeline load carries no filename (the repo
        // IS the pipeline); single-file kinds send the GGUF / safetensors filename. Advanced
        // sentinels ("auto"/"off"/"none") map to omitted so the backend uses its defaults.
        await loadDiffusionModel({
          model_path: repoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          cpu_offload: cpuOffload,
          speed_mode: speedMode === "auto" ? undefined : speedMode,
          transformer_quant: transformerQuant === "auto" ? undefined : transformerQuant,
          attention_backend: attentionBackend === "auto" ? undefined : attentionBackend,
          memory_mode: memoryMode === "auto" ? undefined : memoryMode,
          transformer_cache: transformerCache === "auto" ? undefined : transformerCache,
        });
      } catch (err) {
        lastLoad.current = prevLastLoad;
        setCanReapply(prevLastLoad != null);
        dismissLoadToast();
        toast.error(err instanceof Error ? err.message : "Failed to start load");
        setBusy(null);
        void refreshStatus();
        return false;
      }
      void pollLoadProgress();
      return true;
    },
    [
      pollLoadProgress,
      refreshStatus,
      dismissLoadToast,
      cpuOffload,
      speedMode,
      transformerQuant,
      attentionBackend,
      memoryMode,
      transformerCache,
    ],
  );

  // Set (or clear) the Transform/Inpaint source image; always drop any painted mask so it
  // can't be applied to a different image (the mask is sized to the previous source).
  const handleInitChange = useCallback((dataUrl: string | null) => {
    setInitImage(dataUrl);
    setMaskImage(null);
    setMaskResetKey((k) => k + 1);
  }, []);

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    const l = lastLoad.current;
    if (l) void handleLoad(l.repoId, { kind: l.kind, filename: l.filename });
  }, [handleLoad]);

  // The chat picker emits (modelId, picked quant + its exact filename) for a GGUF,
  // or just (modelId) for a curated non-GGUF safetensors pick; load it, and seed the
  // inputs with that model's defaults.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight: starting a
      // replacement load now would tear down the live poll/toast and reset
      // busy, while the backend rejects the second load with a 409.
      if (busy !== null) return;
      // Curated non-GGUF model: load as a full pipeline or single-file safetensors.
      const spec = loadSpecFor(id, IMAGE_CATALOG);
      if (spec && spec.kind !== "gguf") {
        setQuant(null);
        const d = defaultsFor(id);
        setSteps(d.steps);
        setGuidance(d.guidance);
        void handleLoad(id, { kind: spec.kind, filename: spec.filename });
        return;
      }
      // GGUF quant pick from the variant expander. Optimistic for instant picker feedback, but
      // revert if the load fails to START (400/409/network) or LATER in the poll (download/
      // preflight error/eviction) -- in both cases the old pipeline stays loaded, so don't
      // advertise the failed quant. The poll owns the after-start revert via quantRevert; here
      // we only handle the never-started case.
      if (meta.ggufVariant && meta.ggufFilename) {
        const prevQuant = quant;
        quantRevert.current = { prev: prevQuant };
        setQuant(meta.ggufVariant);
        const dq = defaultsFor(id);
        setSteps(dq.steps);
        setGuidance(dq.guidance);
        void handleLoad(id, { kind: "gguf", filename: meta.ggufFilename }).then((started) => {
          if (!started) {
            setQuant(prevQuant);
            quantRevert.current = null;
          }
        });
        return;
      }
      // A direct single-file local .gguf pick has no variant/filename (custom folder /
      // LM Studio). Load it by splitting the path into (parent dir, basename) the backend
      // resolves, instead of silently doing nothing.
      if (meta.isGguf) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        if (!filename.toLowerCase().endsWith(".gguf")) return;
        // A direct pick carries no curated variant label; surface the filename so
        // the selector stops advertising the previously loaded quant. Optimistic,
        // reverted if the load fails to start OR fails later in the poll (mirrors the
        // curated branch above; the poll owns the after-start revert via quantRevert).
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
      // A direct local single-file .safetensors pick (custom folder / on-device file) must
      // load via from_single_file: the pipeline route rejects a bare file (no model_index.json)
      // and only after evicting the resident model. Split into (parent dir, basename) like the
      // local GGUF branch above.
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
      // Otherwise treat it as a full diffusers repo (safetensors / bnb-4bit). The backend
      // infers the family + base repo from the id and gates loads to unsloth/* repos or
      // on-device paths, so only attempt those; other Hub orgs can't be assembled here.
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
      void handleLoad(id, { kind: "pipeline" }).then((started) => {
        if (!started) {
          setQuant(prevQuant);
          quantRevert.current = null;
        }
      });
    },
    [busy, handleLoad, quant],
  );

  // Deploy a freshly-trained adapter from the Train tab: switch to Create, load the base as
  // a pipeline, and queue the adapter so the LoRA discovery effect applies it once the base
  // is loaded + LoRA-capable. Seeds the prompt with the trigger phrase when provided.
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
    // Ejecting cancels any in-flight replacement load on the backend, so tear down its
    // client-side tracking too: the load poll reschedules on phase null and the persistent
    // toast never resolves, so both would otherwise leak forever after the unload.
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
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

    // Resolve the conditioning image/mask/strength for this workflow up front. Extend
    // (outpaint) is built here from the source by padding + masking the new border, then
    // sent through the same inpaint path. txt2img leaves all three undefined.
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
        // Hires fix: the backend enlarges the source by `upscale` and re-denoises it at
        // this low strength so it gains detail without changing the content.
        condInit = initImage ?? undefined;
        condUpscale = upscaleFactor;
        condStrength = upscaleStrength;
      } else if (isReference) {
        // FLUX.2 reference conditioning: send the primary reference + any extra references
        // (combined by the model). The model generates a fresh image at the slider size
        // guided by the references + prompt. No mask, no strength (not a denoise blend).
        condInit = initImage ?? undefined;
        const extras = referenceImages.filter(Boolean);
        if (extras.length) condRefImages = extras;
      } else if (isEdit) {
        // Instruction editing: send the source image; the prompt IS the instruction.
        // No mask, no strength (the edit pipeline fully regenerates from the instruction).
        condInit = initImage ?? undefined;
      }
    } catch {
      toast.error("Could not prepare the source image");
      return;
    }
    // Resolve a base seed up front. With an explicit seed the run is fully
    // reproducible; with a random one we still pick a concrete base now so each
    // sequential image gets a distinct, reproducible seed (base + i).
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

    // A large run count (generate all night) is a legitimate choice, so there's
    // no upper cap; just floor at 1 and ignore non-numeric input (the number box
    // can yield NaN), which would otherwise make the loop a silent no-op.
    const runs = Number.isFinite(count) && count >= 1 ? Math.floor(count) : 1;
    if (runs !== count) setCount(runs);

    // An explicit seed near the 2**53-1 backend cap can overflow once the per-run offset
    // (base + i*batchSize) and the engine's in-batch +j offsets are added, 422ing a later run
    // AFTER earlier images generated. Fail before any GPU work. Subtraction keeps the comparison
    // exact where the sum would round.
    if (baseSeed > Number.MAX_SAFE_INTEGER - (runs * batchSize - 1)) {
      toast.error("Seed too large for this run count and batch size; use a smaller seed");
      return;
    }

    setBusy("generating");
    setGenDone(0);
    setGenStep(null);
    // Poll the backend's per-step progress across the whole run (all sequential generations)
    // so the bar tracks live denoising steps. A named poll body (guarded against overlap) also
    // serves the visibilitychange listener: a background tab's throttled interval catches up the
    // moment the tab is visible.
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
    try {
      for (let i = 0; i < runs; i++) {
        // The page truly unmounted mid-run (app close / chat-only eject): stop
        // issuing more GPU generations. A plain tab switch keeps it mounted.
        if (!isMounted.current) break;
        const res = await generateDiffusionImage({
          prompt: prompt.trim(),
          // Only send a negative prompt when guidance uses it, so the recipe
          // doesn't record one the model ignored.
          negative_prompt: guidance > 0 ? negativePrompt.trim() || undefined : undefined,
          width: w,
          height: h,
          steps,
          guidance,
          // Offset runs by the batch size: the native engine seeds image j of a
          // run at seed+j, so a +1 run offset would regenerate the previous run's
          // batch-mates. Unique per image on both engines, reproducible via recipes.
          seed: baseSeed + i * batchSize,
          batch_size: batchSize,
          // Transform/Inpaint/Extend send the source image (+ mask for inpaint/extend) and
          // a denoise strength, resolved above. The backend derives output size from the
          // image, so width/height are advisory here.
          init_image: condInit,
          mask_image: condMask,
          strength: condStrength,
          upscale: condUpscale,
          reference_images: condRefImages,
          // Drop empty (no id yet) and zero-weight rows, and trim hand-typed repo ids, so the
          // recipe records only adapters that applied. Gate on loraCapable: a restore can leave
          // adapters in state while the loaded model doesn't support LoRA (picker hidden), and
          // sending them would fail generation with no visible row to remove.
          loras: (() => {
            if (!loraCapable) return undefined;
            const active = loras
              .map((l) => ({ id: l.id.trim(), weight: l.weight }))
              .filter((l) => l.id && l.weight > 0);
            return active.length ? active : undefined;
          })(),
          // ControlNet: sent only when a model + control image are chosen; v1 conditions plain
          // text-to-image only, so skip it for image-conditioned workflows.
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
        if (!isMounted.current) break;
        // Prepend this run's records (newest first) and load their blobs.
        setImages((prev) => [...res.images, ...prev]);
        if (res.images[0]) setSelectedId(res.images[0].id);
        res.images.forEach((image) => void ensureSrc(image));
        setGenDone(i + 1);
      }
      // A generation can change server-side status: Speed=Auto compiles the
      // transformer on the 3rd LoRA-free run (supports_lora flips to false), so
      // without a refresh the LoRA picker stays enabled and the next LoRA run
      // fails on the backend. Cheap status GET; also picks up any other drift.
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
  }, [prompt, negativePrompt, width, height, steps, guidance, seed, batchSize, count, workflow, initImage, maskImage, strength, extendPct, extendSides, upscaleFactor, upscaleStrength, referenceImages, loras, loraCapable, controlnetCapable, controlnetId, controlImage, controlType, controlStrength, ensureSrc, refreshStatus]);

  // Keep the active workflow valid for the loaded model: an edit-only model (Qwen-Image-
  // Edit) has no Create/Transform tabs, a base model has no Edit tab. Snap to the first
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
      {/* The dense transformer_quant fast path only engages on the GGUF kind; on a loaded
          safetensors pipeline / single-file model it is a silent no-op, so gate the control
          to GGUF (or nothing loaded) and otherwise show why it is unavailable. */}
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
      {/* A resident full pipeline is reloadable by repo id alone (the resident effect
          wires lastLoad for it), so it keeps Reapply even before any user-initiated
          load; GGUF/single_file residents have no reload target and hide the button. */}
      {status?.loaded && (canReapply || status?.model_kind === "pipeline") && (
        <Button
          variant="secondary"
          size="sm"
          disabled={busy !== null}
          onClick={handleReapply}
          title="Reload the current model with these advanced options"
        >
          <HugeiconsIcon icon={ArrowReloadHorizontalIcon} className="mr-2 size-3.5" />
          Reapply to loaded model
        </Button>
      )}
    </>
  );

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* ── Top: the model selector, kept at the chat tab's exact position so the
          shared element matches. The load progress shows in a chat-style toast,
          not here. ── */}
      <div className="flex h-[48px] shrink-0 items-start justify-between pl-2 pr-2 pt-[11px]">
        <div className="flex items-center gap-2">
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
            open={active && selectorOpen}
            onOpenChange={(o) => setSelectorOpen(active && o)}
          />
          {/* Create | Train page-mode switch, on the left next to the model selector
              (the selector itself stays leftmost: its position is shared with Chat's).
              Create is the generation workspace; Train is the LoRA training workspace. */}
          <Tabs value={pageMode} onValueChange={(v) => setPageMode(v as "create" | "train")}>
            <TabsList className="h-[34px]">
              {/* Same icons as the sidebar's New Chat / Train entries, so the
                  two workspaces read as the same actions everywhere. */}
              {/* TabsTrigger renders children inside a plain inline span (and preflight
                  makes svg display:block), so the icon and label need their own flex
                  row to stay on one line. */}
              <TabsTrigger value="create" className="w-[84px]">
                <span className="flex items-center gap-1.5">
                  <HugeiconsIcon icon={PencilEdit02Icon} className="size-3.5" />
                  Create
                </span>
              </TabsTrigger>
              <TabsTrigger value="train" className="w-[84px]">
                <span className="flex items-center gap-1.5">
                  <HugeiconsIcon icon={TestTubeOutlineIcon} className="size-3.5" />
                  Train
                </span>
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        <div className="flex items-center gap-2">
          {/* Single fixed toggle for the right-docked Advanced panel (mirrors Chat's settings
              toggle, same icon in both states so it never moves). Highlighted when open.
              Only meaningful in Create mode (load-time tuning), so hidden while training. */}
          {pageMode === "create" && (
            <button
              type="button"
              onClick={() => setAdvancedOpen((o) => !o)}
              aria-label={advancedOpen ? "Hide advanced options" : "Show advanced options"}
              aria-pressed={advancedOpen}
              title="Advanced options"
              className={cn(
                "flex h-[34px] w-[34px] items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                advancedOpen
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              <HugeiconsIcon icon={LayoutAlignRightIcon} className="size-4" />
            </button>
          )}
        </div>
      </div>
      {/* Train mode: the full-page training workspace. Kept unmounted in Create mode so its
          polling stops; Create's own state (gallery, model, workflow) is untouched. */}
      {pageMode === "train" ? (
        <DiffusionTrainPanel
          active={active && pageMode === "train"}
          loadedFamily={status?.family ?? null}
          loadedBaseRepo={
            // Prefer base_repo (the full diffusers pipeline) over repo_id: for a GGUF or
            // single-file load repo_id is the checkpoint path, not a trainable base.
            status?.base_repo ?? status?.repo_id ?? null
          }
          onTrainingComplete={() => setLoraRefreshKey((k) => k + 1)}
          onDeploy={handleDeployAdapter}
        />
      ) : (
      /* ── Controls rail + preview canvas. Padding mirrors the other tabs
          (Export, Data Recipes): px-5 / sm:px-9, with a roomy bottom. ── */
      <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden px-5 pb-8 sm:px-9">
        {/* The controls rail. Plain card (the gray surface) with no header —
            the prompt + Generate button make the panel self-explanatory. */}
        <div className="bg-card corner-squircle flex w-[340px] shrink-0 flex-col gap-4 overflow-y-auto rounded-3xl p-5 ring-1 ring-foreground/10">
          {/* Workflow tabs. Create = text-to-image; Transform = img2img (needs a
              source image). A tab is disabled until the loaded model supports it
              (status.workflows), with a tooltip explaining why. More workflows
              (Edit/Extend/Control/Enhance) slot in here as they land. */}
          <div className="flex gap-1 rounded-xl bg-muted/50 p-1">
            {WORKFLOW_TABS.map((t) => {
              const wf = status?.workflows ?? [];
              // Create (requires null) needs txt2img once a model is loaded -- an edit-only
              // model (workflows: ["edit"]) has no text-to-image mode, so its Create tab is
              // disabled and Edit is the only enabled one. With nothing loaded, Create stays
              // available so the user can pick a model.
              const enabled =
                t.requires === null
                  ? !status?.loaded || wf.includes("txt2img")
                  : wf.includes(t.requires);
              const active = workflow === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  disabled={!enabled}
                  title={enabled ? t.hint : `Load a model that supports ${t.label.toLowerCase()}`}
                  onClick={() => setWorkflow(t.id)}
                  className={cn(
                    "flex-1 rounded-lg px-2 py-1 text-xs font-medium transition-colors",
                    active
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground",
                    !enabled && "cursor-not-allowed opacity-40 hover:text-muted-foreground",
                  )}
                >
                  {t.label}
                </button>
              );
            })}
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
                        // Keep the slot in place (empty string when cleared) so other slots
                        // don't renumber mid-edit; empty slots are dropped only at send time
                        // and removed explicitly via the button below. Stable key={i}.
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
          {/* LoRA adapters: shown whenever the loaded model + quant can apply them. Type a
              Hugging Face repo id (owner/name, or owner/name:weight-file.safetensors) or pick
              a discovered adapter from the suggestions. Stack multiple, each with a 0-2 weight.
              The backend owns how they apply (native prompt tags / diffusers set_adapters); the
              UI only sends {id, weight}. */}
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
                    // Stable key={i}: sel.id is the editable repo-id input, so keying on it would
                    // remount the row on the first keystroke and drop input focus. The list is
                    // index-addressed (mutations use j === i) and rows removed explicitly.
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
                      // Prefill with the first unused suggestion when a curated catalog exists,
                      // else an empty row the user fills with a Hugging Face repo id.
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
          {/* ControlNet: shown when the loaded model supports it, a model is discoverable, and
              the plain text-to-image workflow is active (v1 conditions txt2img only). Pick a
              model, add a control image, choose how to derive the map, and set the strength. */}
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
          {/* A negative prompt only does anything with guidance on, so hide it at
              guidance 0 (Z-Image-Turbo's default) instead of showing a dead field. */}
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
              <Button
                type="button"
                variant="secondary"
                size="icon"
                aria-label="Flip width and height"
                title="Flip orientation"
                onClick={flipDimensions}
              >
                <HugeiconsIcon icon={ArrowLeftRightIcon} className="size-4" />
              </Button>
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

        <div className="bg-card corner-squircle relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-3xl ring-1 ring-foreground/10">
          <div className="relative flex flex-1 items-center justify-center overflow-auto p-6">
            {selected && selectedSrc ? (
              <>
                <img
                  src={selectedSrc}
                  alt={selected.prompt}
                  className="max-h-full max-w-full rounded-xl object-contain shadow-sm"
                />
                {/* Actions grouped in one glass toolbar so they stay legible over
                    any image and read as a unit instead of blending into the canvas.
                    Size/seed live in the Recipe popover, so no separate chip here. */}
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
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label="Delete image"
                    title="Delete"
                    className="text-muted-foreground hover:text-destructive"
                    onClick={() => void handleDelete(selected.id)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
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
                <HugeiconsIcon icon={ImageAdd02Icon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a diffusion model to load"}
                </p>
              </div>
            )}

            {/* Live generation progress: a per-step bar with ETA, centered when
                there's nothing else to show, tucked at the bottom over an image. */}
            {busy === "generating" && (
              <div
                className={cn(
                  "pointer-events-none absolute flex justify-center px-4",
                  selectedSrc ? "inset-x-0 bottom-4" : "inset-0 items-center",
                )}
              >
                <div className="w-72 max-w-full rounded-xl bg-background/85 p-3 shadow-lg ring-1 ring-border backdrop-blur">
                  <ModelLoadDescription
                    // Drop the chat min-height: this floating card has no layout to
                    // stabilise, and it would otherwise center the thin bar and
                    // leave empty space above it (most visible with no title).
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
              className="flex shrink-0 gap-2 overflow-x-auto border-t border-foreground/10 p-3"
              onScroll={(e) => {
                // Near the right edge: pull the next older page (infinite scroll).
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past
                  images stay visible and browsable while the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center rounded-lg bg-muted/50 ring-2 ring-primary/30">
                  <Spinner className="size-5 text-muted-foreground" />
                </div>
              )}
              {images.map((image) => (
                <button
                  key={image.id}
                  type="button"
                  onClick={() => setSelectedId(image.id)}
                  className="relative size-16 shrink-0 overflow-hidden rounded-lg bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
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
                  {/* Selection marker on a non-focusable overlay, so the button's
                      own focus state can never mask it. */}
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

        {/* Right-docked Advanced panel (mirrors Chat's settings panel): closed by default,
            opened by the single fixed top-bar toggle above (which never moves between states,
            like Chat's run-settings toggle), so the optimisation controls are discoverable
            without being docked open or buried at the bottom of the left rail. */}
        {advancedOpen && (
          <div className="bg-card corner-squircle flex w-[300px] shrink-0 flex-col overflow-hidden rounded-3xl ring-1 ring-foreground/10">
            <div className="flex h-[52px] shrink-0 items-center border-b border-border/60 px-4">
              <span className="flex items-center gap-1.5 text-sm font-semibold text-foreground">
                <HugeiconsIcon icon={Settings02Icon} className="size-4" />
                Advanced
              </span>
            </div>
            <div className="flex flex-col gap-3 overflow-y-auto p-4">
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
