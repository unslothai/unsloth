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
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
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
import { Textarea } from "@/components/ui/textarea";
import { SectionCard } from "@/components/section-card";
import { InfoHint } from "@/components/ui/info-hint";
import { ModelSelector } from "@/components/assistant-ui/model-selector";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/components/assistant-ui/model-selector/types";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import { formatBytes } from "@/features/hub/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

import {
  type DiffusionLoadProgress,
  type DiffusionStatus,
  type GalleryImage,
  deleteGalleryImage,
  fetchGalleryObjectUrl,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  getGallery,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";

// MVP: a single curated diffusion GGUF. The chat ModelSelector lists/picks its
// quants; the base diffusers repo (VAE/text-encoder) is resolved server-side.
const MODEL = {
  repo_id: "unsloth/Z-Image-Turbo-GGUF",
  label: "Z-Image-Turbo",
  family: "z-image",
  // Hub-canonical filename pattern: z-image-turbo-<QUANT>.gguf
  ggufFor: (quant: string) => `z-image-turbo-${quant}.gguf`,
};

const MODELS: ModelOption[] = [
  { id: MODEL.repo_id, name: MODEL.label, description: "Text-to-image · GGUF", isGguf: true },
];

// Common aspect ratios (landscape-oriented; the Flip button gives the portrait
// mirror). Picking one locks the W:H proportion; the sliders size it, and the
// generation snaps to Z-Image's ~1-megapixel-trained 16px grid.
const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};
const ASPECT_OPTIONS = ["custom", ...Object.keys(ASPECT_RATIOS)];

// Z-Image accepts 256–2048, in multiples of 16. Snap any value into range.
const MIN_DIM = 256;
const MAX_DIM = 2048;
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

// The gallery is persisted on the backend (durable across reloads); this module
// cache only holds the last-fetched records + their object/data URLs so a tab
// switch re-renders instantly without a refetch flash. Object URLs live for the
// app's lifetime (revoked only on delete), so they stay valid across remounts.
const galleryCache: {
  images: GalleryImage[];
  selectedId: string | null;
  quant: string | null;
  srcById: Map<string, string>;
  // Ids with a fetch in flight, so concurrent ensureSrc calls don't double-fetch
  // (and leak the duplicate object URL).
  inflight: Set<string>;
} = { images: [], selectedId: null, quant: null, srcById: new Map(), inflight: new Set() };

// Export filename: app name, a compact sortable timestamp, and the seed. A batch
// shares the seed + timestamp, so a "_<n>" suffix is added past the first image.
// e.g. Unsloth_20260624-143005_123.png / Unsloth_20260624-143005_123_1.png
// (the full recipe is embedded in the PNG regardless).
function exportFilename(image: GalleryImage): string {
  const d = new Date(image.created_at * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp =
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  const suffix = image.batch_index > 0 ? `_${image.batch_index}` : "";
  return `Unsloth_${stamp}_${image.seed}${suffix}.png`;
}

function downloadImage(src: string, image: GalleryImage) {
  const link = document.createElement("a");
  link.href = src;
  link.download = exportFilename(image);
  link.click();
}

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

// The chat tab's model-load toast styling, reused verbatim so the diffusion
// load toast is visually identical (persistent, progress bar, same chrome).
const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

// Render the chat ModelLoadDescription for a progress poll. The base text-
// encoder/VAE repo downloads alongside the GGUF, so the total can exceed the
// picked quant's size — that's the real one-time download.
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
      message={`Loading ${MODEL.label}. This may include downloading its base model.`}
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
          className="w-12 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
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
}: {
  image: GalleryImage;
  onRestore: (image: GalleryImage) => void;
}) {
  return (
    <Popover>
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

export function ImagesPage() {
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
  const [steps, setSteps] = useState(9);
  const [guidance, setGuidance] = useState(0.0);
  const [seed, setSeed] = useState("");
  // Batch size = images per forward pass (VRAM-heavy); count = sequential loops.
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);

  const [busy, setBusy] = useState<Busy>(null);
  // {done, total} while a multi-run generation is in flight (for the button).
  const [genProgress, setGenProgress] = useState<{ done: number; total: number } | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  // Records come from the backend (durable); srcById maps each id to its object
  // URL (loaded images) or data URL (the one just generated).
  const [images, setImages] = useState<GalleryImage[]>(() => galleryCache.images);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    Object.fromEntries(galleryCache.srcById),
  );
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place (chat-style).
  const loadToastId = useRef<string | number | null>(null);

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);

  // Mirror to the module cache so a tab switch re-renders instantly.
  useEffect(() => {
    galleryCache.images = images;
    galleryCache.selectedId = selectedId;
    galleryCache.quant = quant;
  }, [images, selectedId, quant]);

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
      const records = await getGallery();
      galleryCache.images = records;
      setImages(records);
      records.forEach((image) => void ensureSrc(image));
    } catch {
      // Best-effort: a failed gallery load shouldn't block the page.
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
    setNegativePrompt(image.negative_prompt ?? "");
    setSteps(image.steps);
    setGuidance(image.guidance);
    setSeed(String(image.seed));
    setWidth(image.width);
    setHeight(image.height);
    const m = matchAspect(image.width, image.height);
    setAspect(m.key);
    setPortrait(m.portrait);
    toast.success("Settings restored to inputs");
  }, []);

  // Size controls: a locked aspect ratio keeps the paired dimension in step as
  // you drag a slider; "custom" frees both. Flip swaps width and height and
  // keeps the lock (the ratio just applies in the other orientation).
  // h/w multiplier for the locked ratio [a,b] (a:b = long:short): landscape puts
  // the long side on width (h = w*b/a), portrait puts it on height (h = w*a/b).
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

  useEffect(() => {
    void refreshStatus();
    // Stop polling if the page unmounts mid-load.
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, [refreshStatus]);

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
        return;
      }
      if (p.phase === "error") {
        dismissLoadToast();
        toast.error(p.error || "Failed to load model");
        setBusy(null);
        return;
      }
      if (loadToastId.current != null) toast(null, loadToastArgs(p, loadToastId.current));
    } catch {
      // Transient poll failure: keep trying.
    }
    pollTimer.current = setTimeout(() => void pollLoadProgress(), 1000);
  }, [dismissLoadToast]);

  const handleLoad = useCallback(
    async (ggufFilename: string) => {
      // Cancel any prior poll loop so two can't run at once.
      if (pollTimer.current) clearTimeout(pollTimer.current);
      setBusy("loading");
      // Show the chat-style toast immediately; the poll updates it by id.
      dismissLoadToast();
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS));
      try {
        // Returns immediately — the load runs in the background; we poll for it.
        await loadDiffusionModel({
          model_path: MODEL.repo_id,
          gguf_filename: ggufFilename,
          family_override: MODEL.family,
        });
      } catch (err) {
        dismissLoadToast();
        toast.error(err instanceof Error ? err.message : "Failed to start load");
        setBusy(null);
        void refreshStatus();
        return;
      }
      void pollLoadProgress();
    },
    [pollLoadProgress, refreshStatus, dismissLoadToast],
  );

  // The chat picker emits (modelId, picked quant); load its matching GGUF.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      if (!meta.ggufVariant) return; // a non-quant pick; ignore
      if (id !== MODEL.repo_id) {
        toast.error("Only Z-Image-Turbo is supported right now.");
        return;
      }
      setQuant(meta.ggufVariant);
      void handleLoad(MODEL.ggufFor(meta.ggufVariant));
    },
    [handleLoad],
  );

  const handleUnload = useCallback(async () => {
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
  }, [refreshStatus]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
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

    setBusy("generating");
    setGenProgress({ done: 0, total: count });
    try {
      for (let i = 0; i < count; i++) {
        const res = await generateDiffusionImage({
          prompt: prompt.trim(),
          // Only send a negative prompt when guidance uses it, so the recipe
          // doesn't record one the model ignored.
          negative_prompt: guidance > 0 ? negativePrompt.trim() || undefined : undefined,
          width: w,
          height: h,
          steps,
          guidance,
          seed: baseSeed + i,
          batch_size: batchSize,
        });
        // Prepend this run's records (newest first) and load their blobs.
        setImages((prev) => [...res.images, ...prev]);
        if (res.images[0]) setSelectedId(res.images[0].id);
        res.images.forEach((image) => void ensureSrc(image));
        setGenProgress({ done: i + 1, total: count });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Image generation failed");
    } finally {
      setBusy(null);
      setGenProgress(null);
    }
  }, [prompt, negativePrompt, width, height, steps, guidance, seed, batchSize, count, ensureSrc]);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* ── Top: the model selector, kept at the chat tab's exact position so the
          shared element matches. The load progress shows in a chat-style toast,
          not here. ── */}
      <div className="flex h-[48px] shrink-0 items-start pl-2 pr-2 pt-[11px]">
        <ModelSelector
          models={MODELS}
          value={status?.loaded ? status.repo_id ?? undefined : undefined}
          activeGgufVariant={quant}
          onValueChange={handleModelSelect}
          onEject={status?.loaded ? handleUnload : undefined}
          variant="ghost"
          className="!h-[34px]"
          task={["text-to-image", "image-to-image", "image-text-to-image"]}
        />
      </div>

      {/* ── Controls rail + preview canvas. Padding mirrors the other tabs
          (Export, Data Recipes): px-5 / sm:px-9, with a roomy bottom. ── */}
      <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden px-5 pb-8 sm:px-9">
        <SectionCard
          icon={<HugeiconsIcon icon={ImageAdd02Icon} className="size-5" strokeWidth={1.5} />}
          title="Generate"
          description="Prompt and settings"
          accent="indigo"
          className="w-[340px] shrink-0 gap-4 overflow-y-auto"
        >
          <Field label="Prompt">
            <Textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </Field>
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
              <Select value={aspect} onValueChange={changeAspect}>
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
          <SliderField label="Width" value={width} min={256} max={2048} step={16} onChange={changeWidth} />
          <SliderField label="Height" value={height} min={256} max={2048} step={16} onChange={changeHeight} />

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
            max={128}
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
            {busy === "generating" && genProgress && genProgress.total > 1
              ? `Generating ${genProgress.done}/${genProgress.total}…`
              : "Generate"}
          </Button>
        </SectionCard>

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
                  <RecipePopover image={selected} onRestore={restoreSettings} />
                  <Button
                    size="sm"
                    variant="ghost"
                    className="gap-1.5"
                    onClick={() => downloadImage(selectedSrc, selected)}
                  >
                    <HugeiconsIcon icon={Download01Icon} className="size-4" />
                    Download
                  </Button>
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
            ) : busy === "generating" || selected ? (
              // First image generating, or the selected record's blob is still
              // loading — spin in place rather than flashing the empty state.
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Spinner className="size-8" />
                <p className="text-sm">{busy === "generating" ? "Generating…" : "Loading…"}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <HugeiconsIcon icon={ImageAdd02Icon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a model quant to load, then generate."}
                </p>
              </div>
            )}
          </div>

          {(images.length > 0 || busy === "generating") && (
            <div className="flex shrink-0 gap-2 overflow-x-auto border-t border-foreground/10 p-3">
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
                  title={`seed ${image.seed}`}
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
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
