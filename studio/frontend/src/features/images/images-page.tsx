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
import { InfoHint } from "@/components/ui/info-hint";
import { ModelSelector } from "@/components/assistant-ui/model-selector";
import { IMAGE_GEN_TASKS } from "@/components/assistant-ui/model-selector/pickers";
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
  type DiffusionGenerateProgress,
  type DiffusionLoadProgress,
  type DiffusionStatus,
  type GalleryImage,
  deleteGalleryImage,
  fetchGalleryObjectUrl,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  getGallery,
  getGenerateProgress,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";

// Curated diffusion GGUFs the picker recommends. The backend resolves each one's
// pipeline + base diffusers repo from its repo id, so the rail just lists them;
// the chat ModelSelector also surfaces any other on-device image GGUF.
const txt2img = (id: string, name: string): ModelOption => ({
  id,
  name,
  description: "Text-to-image · GGUF",
  isGguf: true,
});
const MODELS: ModelOption[] = [
  txt2img("unsloth/Z-Image-Turbo-GGUF", "Z-Image-Turbo"),
  txt2img("unsloth/Z-Image-GGUF", "Z-Image"),
  txt2img("unsloth/Qwen-Image-2512-GGUF", "Qwen-Image 2512"),
  txt2img("unsloth/Qwen-Image-GGUF", "Qwen-Image"),
  txt2img("unsloth/FLUX.1-schnell-GGUF", "FLUX.1 schnell"),
  txt2img("unsloth/FLUX.1-dev-GGUF", "FLUX.1 dev"),
  txt2img("unsloth/FLUX.2-klein-4B-GGUF", "FLUX.2 klein 4B"),
  txt2img("unsloth/FLUX.2-klein-9B-GGUF", "FLUX.2 klein 9B"),
];

// Per-model generation defaults (steps + guidance), matched by repo-id substring,
// most specific first. Distilled "turbo/schnell" models want few steps and little
// guidance; the full "dev" models want more steps and real CFG.
// Generation defaults when the model is unrecognised: the distilled few-step /
// no-CFG shape. Also seeds the sliders' initial state.
const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  { match: "z-image-turbo", steps: 9, guidance: 0 },
  { match: "flux.1-schnell", steps: 4, guidance: 0 },
  { match: "flux.1", steps: 28, guidance: 3.5 },
  { match: "flux.2-klein", steps: 4, guidance: 0 },
  { match: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", steps: 20, guidance: 4 },
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

// Z-Image accepts 256–2048, in multiples of 16. Snap any value into range.
const MIN_DIM = 256;
const MAX_DIM = 2048;
// Convenient drag range for the Runs slider. The number box accepts higher typed
// values on purpose (set it large to generate all night); the loop only floors at
// 1 and ignores non-numeric input.
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

// Module cache of the backend-persisted gallery, so a tab switch re-renders
// instantly. Object URLs are revoked only on delete (not unmount), so they stay
// valid across remounts.
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

// Bar label for an in-flight generation: step count plus an ETA once it's known
// (formatEta returns "" for non-positive, so the last step shows just the step).
function genStepLabel(p: DiffusionGenerateProgress): string {
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

// Render the chat ModelLoadDescription for a progress poll. The base repo
// (text-encoder/VAE) downloads alongside the GGUF, so the total exceeds the
// picked quant's size.
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
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const [seed, setSeed] = useState("");
  // Batch size = images per forward pass (VRAM-heavy); count = sequential loops.
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);

  const [busy, setBusy] = useState<Busy>(null);
  // {done, total} while a multi-run generation is in flight (for the button).
  // Number of runs finished in the current multi-run generation (null = idle).
  // The total is just `count`, so it isn't stored separately.
  const [genDone, setGenDone] = useState<number | null>(null);
  // Live per-step progress (step / total + ETA) polled during generation.
  const [genStep, setGenStep] = useState<DiffusionGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
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
  // False once the page unmounts, so a multi-run generation loop stops issuing further
  // GPU work (and state updates) after the user navigates away.
  const isMounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place (chat-style).
  const loadToastId = useRef<string | number | null>(null);
  // Last load-progress signature shown, so a tick that moved nothing skips the toast.
  const lastLoadSig = useRef<string | null>(null);

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
    setNegativePrompt(image.negative_prompt ?? "");
    setSteps(image.steps);
    setGuidance(image.guidance);
    setSeed(String(image.seed));
    setWidth(image.width);
    setHeight(image.height);
    // The batch shared one seed, so image batch_index>0 only reproduces by replaying the
    // whole batch: restore the batch size too (older recipes without it default to 1).
    setBatchSize(image.batch_size ?? 1);
    const m = matchAspect(image.width, image.height);
    setAspect(m.key);
    setPortrait(m.portrait);
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

  // Track mount so a long sequential generate run stops issuing GPU work once
  // the user navigates away (the generate loop checks isMounted.current). The
  // mount-time refreshStatus and the timer/toast cleanup live in the load-resume
  // effect below, so this one carries only the mount flag.
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

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
        // A failed load may have freed a previously-loaded model, so resync to
        // the real backend state (the synchronous failure path does the same).
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // No load in flight and nothing loaded: the load was cancelled or
        // evicted (e.g. a chat load took the GPU) and the backend cleared its
        // state. Terminal — otherwise this loop would spin forever and leave
        // busy stuck on "loading", deadening the picker and Generate button.
        dismissLoadToast();
        setBusy(null);
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

  useEffect(() => {
    void (async () => {
      await refreshStatus();
      // A load runs on the backend as a daemon thread that survives navigation.
      // On (re)mount, resume tracking one that's still in flight so the page
      // shows progress and updates on completion, instead of a stale view that
      // never polls (no toast, and no refresh when the load finishes).
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
    })();
    // Stop polling if the page unmounts mid-load / mid-generate, and dismiss the
    // load toast — its poll loop is gone, so it would otherwise hang forever
    // (duration: Infinity) on whatever page the user navigated to.
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      dismissLoadToast();
    };
  }, [refreshStatus, dismissLoadToast, pollLoadProgress]);

  const handleLoad = useCallback(
    async (repoId: string, ggufFilename: string) => {
      // Cancel any prior poll loop so two can't run at once.
      if (pollTimer.current) clearTimeout(pollTimer.current);
      setBusy("loading");
      // Show the chat-style toast immediately; the poll updates it by id.
      dismissLoadToast();
      lastLoadSig.current = null;
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS));
      try {
        // Returns immediately — the load runs in the background; we poll for it.
        // The backend infers the family + base diffusers repo from the repo id.
        // Forward the saved HF token so gated bases (FLUX dev/klein) can download.
        await loadDiffusionModel({
          model_path: repoId,
          gguf_filename: ggufFilename,
          hf_token: hfApiToken(getHfToken()),
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

  // The chat picker emits (modelId, picked quant + its exact filename); load it,
  // and seed the inputs with that model's defaults.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight: starting a
      // replacement load now would tear down the live poll/toast and reset
      // busy, while the backend rejects the second load with a 409.
      if (busy !== null) return;
      if (meta.ggufVariant && meta.ggufFilename) {
        setQuant(meta.ggufVariant);
        const d = defaultsFor(id);
        setSteps(d.steps);
        setGuidance(d.guidance);
        void handleLoad(id, meta.ggufFilename);
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
        const d = defaultsFor(id);
        setSteps(d.steps);
        setGuidance(d.guidance);
        void handleLoad(dir, filename);
      }
    },
    [busy, handleLoad],
  );

  const handleUnload = useCallback(async () => {
    // Ejecting cancels any in-flight replacement load on the backend, so tear
    // down its client-side tracking too: the load poll reschedules on phase
    // null and the persistent toast never resolves, so both would otherwise
    // leak forever after the unload.
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

    setBusy("generating");
    setGenDone(0);
    setGenStep(null);
    // Poll the backend's per-step progress across the whole run (all sequential
    // generations), so the bar tracks the live denoising steps.
    genPollTimer.current = setInterval(async () => {
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
      }
    }, 300);
    try {
      for (let i = 0; i < runs; i++) {
        // The user navigated away mid-run: stop issuing more GPU generations.
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
          seed: baseSeed + i,
          batch_size: batchSize,
        });
        if (!isMounted.current) break;
        // Prepend this run's records (newest first) and load their blobs.
        setImages((prev) => [...res.images, ...prev]);
        if (res.images[0]) setSelectedId(res.images[0].id);
        res.images.forEach((image) => void ensureSrc(image));
        setGenDone(i + 1);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Image generation failed");
    } finally {
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      genPollTimer.current = null;
      setBusy(null);
      setGenDone(null);
      setGenStep(null);
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
          task={IMAGE_GEN_TASKS}
        />
      </div>

      {/* ── Controls rail + preview canvas. Padding mirrors the other tabs
          (Export, Data Recipes): px-5 / sm:px-9, with a roomy bottom. ── */}
      <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden px-5 pb-8 sm:px-9">
        {/* The controls rail. Plain card (the gray surface) with no header —
            the prompt + Generate button make the panel self-explanatory. */}
        <div className="bg-card corner-squircle flex w-[340px] shrink-0 flex-col gap-4 overflow-y-auto rounded-3xl p-5 ring-1 ring-foreground/10">
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
                    : "Select a model quant to load, then generate."}
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
      </div>
    </div>
  );
}
