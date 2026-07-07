// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Delete02Icon,
  Download01Icon,
  InformationCircleIcon,
  LayoutAlignRightIcon,
  Settings02Icon,
  Video01Icon,
  VolumeHighIcon,
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
import { VIDEO_GEN_TASKS } from "@/components/assistant-ui/model-selector/pickers";
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
  type GalleryVideo,
  type VideoGenerateProgress,
  type VideoLoadProgress,
  type VideoStatus,
  cancelVideoGeneration,
  clearVideoGallery,
  deleteGalleryVideo,
  fetchGalleryVideoObjectUrl,
  generateVideo,
  getVideoGallery,
  getVideoGenerateProgress,
  getVideoLoadProgress,
  getVideoStatus,
  loadVideoModel,
  unloadVideoModel,
} from "./api";

// How to load a curated non-GGUF (safetensors) video model. "pipeline" = a full diffusers
// repo (from_pretrained). The backend gates these to unsloth/* repos plus the official
// family base repos. Keyed by repo id so the load handler knows the kind.
type PipelineSpec = { kind: "pipeline"; filename?: string };
const PIPELINE_MODELS: Record<string, PipelineSpec> = {
  "Lightricks/LTX-2": { kind: "pipeline" },
  // Wan2.2 diffusers base repos (no GGUF variant yet): loaded as full pipelines. TI2V-5B
  // is a single-DiT 720p-class model; T2V-A14B is the dual-expert MoE. The backend gates
  // these to the Wan-AI base repos (see _TRUSTED_NON_GGUF_VIDEO_REPOS).
  "Wan-AI/Wan2.2-TI2V-5B-Diffusers": { kind: "pipeline" },
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers": { kind: "pipeline" },
};

// A curated GGUF picker entry: isGguf true expands its .gguf files in the quant expander
// (like the image GGUF repos), and the backend resolves the pipeline + base repo from the id.
const ggufModel = (id: string, name: string): ModelOption => ({
  id,
  name,
  description: "Text-to-video · GGUF",
  isGguf: true,
});

// A curated non-GGUF pipeline entry (isGguf false -> no quant expander, direct load).
const pipelineModel = (id: string, name: string, description: string): ModelOption => ({
  id,
  name,
  description,
  isGguf: false,
});

// Curated text-to-video models the picker recommends. The chat ModelSelector also surfaces
// any other on-device video GGUF (via the VIDEO_GEN_TASKS filter).
const VIDEO_MODELS: ModelOption[] = [
  ggufModel("unsloth/LTX-2.3-GGUF", "LTX 2.3 distilled"),
  pipelineModel("Lightricks/LTX-2", "LTX 2 (base, bf16)", "Text-to-video with audio · Safetensors"),
  pipelineModel(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Wan 2.2 TI2V 5B",
    "Text-to-video 720p · Safetensors",
  ),
  pipelineModel(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan 2.2 T2V A14B (MoE)",
    "Text-to-video, dual-expert · Safetensors",
  ),
];

// Per-model generation defaults (steps + guidance), matched by repo-id substring, most
// specific first. The distilled model wants very few steps and no guidance; the full base
// model wants more steps and real CFG.
const DEFAULT_GEN = { steps: 8, guidance: 1 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  // "distilled" before the generic "ltx": the distilled model runs at 8 steps, guidance 1.
  { match: "distilled", steps: 8, guidance: 1 },
  { match: "ltx", steps: 40, guidance: 4 },
  // Wan2.2 pipelines default to 50 steps at CFG 5.0 (WanPipeline defaults, verified in
  // diffusers 0.39). The backend supplies the fps per family (24 for TI2V-5B, 16 for A14B).
  { match: "wan", steps: 50, guidance: 5 },
];

function defaultsFor(repoId: string): { steps: number; guidance: number } {
  const id = repoId.toLowerCase();
  return MODEL_DEFAULTS.find((d) => id.includes(d.match)) ?? DEFAULT_GEN;
}

// Resolution presets offered before a model is loaded (default first). Once loaded, the
// backend's status.defaults.resolution_presets replaces these.
const FALLBACK_RESOLUTION_PRESETS: Array<[number, number]> = [
  [768, 512],
  [1216, 704],
  [704, 1216],
];

// Fallbacks used to build the duration presets before a model is loaded, so the duration
// select is populated and valid on first paint.
const FALLBACK_FRAME_STEP = 8;
const FALLBACK_FPS = 24;

// Module cache of the backend-persisted gallery, so a tab switch re-renders instantly.
// Object URLs are revoked only on delete (not unmount), so they stay valid across remounts.
const galleryCache: {
  videos: GalleryVideo[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  srcById: Map<string, string>;
  // Ids with a fetch in flight, so concurrent ensureSrc calls don't double-fetch
  // (and leak the duplicate object URL).
  inflight: Set<string>;
} = {
  videos: [],
  hasMore: false,
  selectedId: null,
  quant: null,
  srcById: new Map(),
  inflight: new Set(),
};

// Videos loaded per infinite-scroll page.
const PAGE_SIZE = 50;

// Export filename, e.g. Unsloth_video_20260624-143005_123.mp4.
function exportFilename(video: GalleryVideo): string {
  const d = new Date(video.created_at);
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = Number.isNaN(d.getTime())
    ? "unknown"
    : `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
      `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  return `Unsloth_video_${stamp}_${video.seed}.mp4`;
}

function downloadVideo(src: string, video: GalleryVideo) {
  const link = document.createElement("a");
  link.href = src;
  link.download = exportFilename(video);
  link.click();
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// A terse clip descriptor for the gallery card / player caption: duration + resolution.
function clipMeta(video: GalleryVideo): string {
  const secs = video.duration_s > 0 ? `${video.duration_s.toFixed(1)}s` : `${video.num_frames}f`;
  return `${secs} · ${video.width}×${video.height}`;
}

// Bar label for an in-flight generation: the phase ("Denoising step X/Y" during denoise,
// "Encoding video…" during export) plus an ETA once known.
function genStepLabel(p: VideoGenerateProgress): string {
  if (p.phase === "export") return "Encoding video…";
  const base = p.total > 0 ? `Denoising step ${p.step}/${p.total}` : "Denoising…";
  const eta = p.eta_seconds != null ? formatEta(p.eta_seconds) : "";
  return eta ? `${base} · ~${eta}` : base;
}

// The chat tab's model-load toast styling, reused verbatim so the video load toast is
// visually identical (persistent, progress bar, same chrome).
const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

// The download total for a video load can only be estimated from a companion base repo, so
// the toast shows a byte count rather than a hard percentage until the total is known.
function loadFraction(p: VideoLoadProgress): number | null {
  if (!p.expected_bytes || p.expected_bytes <= 0) return null;
  return Math.min(1, p.downloaded_bytes / p.expected_bytes);
}

function loadToastDescription(p: VideoLoadProgress) {
  const frac = loadFraction(p);
  const downloading =
    p.phase === "downloading" && (frac === null || frac < 0.999);
  const title = downloading
    ? "Downloading model…"
    : p.phase === "finalizing"
      ? "Loading to GPU…"
      : "Starting model…";
  const hasTotal = frac !== null;
  return (
    <ModelLoadDescription
      title={title}
      message="Loading the model. This may include downloading its base model."
      progressPercent={hasTotal ? frac * 100 : null}
      progressLabel={
        hasTotal
          ? `${formatBytes(p.downloaded_bytes)} of ${formatBytes(p.expected_bytes ?? 0)}`
          : p.downloaded_bytes > 0
            ? `${formatBytes(p.downloaded_bytes)} downloaded`
            : null
      }
    />
  );
}

// Toast args mirroring chat's: persistent, closeable, content in `description`. Pass `id`
// to update the existing toast in place instead of stacking a new one.
function loadToastArgs(p: VideoLoadProgress, id?: string | number) {
  return {
    ...(id != null ? { id } : {}),
    description: loadToastDescription(p),
    duration: Infinity,
    closeButton: true,
    classNames: LOAD_TOAST_CLASSNAMES,
  };
}

const IDLE_PROGRESS: VideoLoadProgress = {
  phase: null,
  downloaded_bytes: 0,
  expected_bytes: null,
  error: null,
};

// Mirrors the Train page's SliderRow: label + Slider + number input, same classes as the
// images page's SliderField.
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
// Short scheme/mode tokens go uppercase (FBCACHE); the attention backend the backend reports
// as `_native_cudnn` shows as cuDNN.
function formatResolvedValue(value: string | boolean | null): string {
  if (value === null || value === "") return "Off";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === "_native_cudnn" || value.toLowerCase() === "cudnn") return "cuDNN";
  return value.toUpperCase();
}

// The "Auto: X" badge for one Advanced control: rendered only when the backend resolved that
// control itself (source === "auto"); an explicit user choice renders nothing. The reason is
// surfaced as a hover tooltip. Muted pill matching the panel's other chips. Reuses the same
// markup as the images page's ResolvedBadge.
function ResolvedBadge({
  status,
  controlKey,
}: {
  status: VideoStatus | null;
  controlKey: string;
}) {
  const resolved = status?.resolved?.[controlKey];
  if (!resolved || resolved.source !== "auto") return null;
  return (
    <span
      title={resolved.reason || undefined}
      className="shrink-0 rounded-sm bg-muted px-1 py-px text-[9px] font-medium uppercase tracking-wider text-muted-foreground"
    >
      Auto: {formatResolvedValue(resolved.value)}
    </span>
  );
}

// A compact labeled Select row for the Advanced Options panel.
function AdvancedSelect({
  label,
  hint,
  badge,
  value,
  onValueChange,
  options,
}: {
  label: string;
  hint?: ReactNode;
  badge?: ReactNode;
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
    </div>
  );
}

// One row in the loaded-model status line.
function StatusChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="text-muted-foreground/70">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </span>
  );
}

// The full generation recipe for a clip, with a one-click "restore to inputs".
function RecipePopover({
  video,
  onRestore,
  active,
}: {
  video: GalleryVideo;
  onRestore: (video: GalleryVideo) => void;
  active: boolean;
}) {
  // Controlled + force-closed off-tab: PopoverContent portals to body, so the hidden/inert
  // page wrapper can't contain it when the page is kept mounted.
  const [open, setOpen] = useState(false);
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
          <p className="text-[11px] text-muted-foreground">{formatTimestamp(video.created_at)}</p>
        </div>
        <div className="flex flex-col gap-2 px-4 py-3 text-xs">
          <RecipeRow label="Prompt" value={video.prompt} wrap />
          {video.negative_prompt ? (
            <RecipeRow label="Negative" value={video.negative_prompt} wrap />
          ) : null}
          {video.model ? <RecipeRow label="Model" value={video.model} /> : null}
          <RecipeRow label="Size" value={`${video.width} × ${video.height}`} />
          <RecipeRow label="Frames" value={`${video.num_frames} @ ${video.fps} fps`} />
          <RecipeRow label="Duration" value={`${video.duration_s.toFixed(2)}s`} />
          <RecipeRow label="Steps" value={String(video.steps)} />
          <RecipeRow label="Guidance" value={String(video.guidance)} />
          <RecipeRow label="Seed" value={String(video.seed)} mono />
        </div>
        <div className="border-t border-border/60 px-3 py-2.5">
          <Button size="sm" className="w-full gap-1.5" onClick={() => onRestore(video)}>
            Restore these settings
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

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
    <div className="flex gap-2">
      <span className="w-16 shrink-0 text-muted-foreground">{label}</span>
      <span
        className={cn(
          "min-w-0 flex-1 text-foreground",
          wrap ? "whitespace-pre-wrap break-words" : "truncate",
          mono && "font-mono",
        )}
      >
        {value}
      </span>
    </div>
  );
}

type Busy = "loading" | "unloading" | "generating" | null;

export function VideoPage({ active = true }: { active?: boolean }) {
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(
    "a tiny ginger sloth surfing a wave at sunset, cinematic, smooth motion",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const [seed, setSeed] = useState("");
  // The chosen resolution preset index into the current preset list.
  const [resolutionIdx, setResolutionIdx] = useState(0);
  // The chosen frame count (must lie on the family's temporal lattice: k*frame_step+1).
  const [numFrames, setNumFrames] = useState(FALLBACK_FRAME_STEP * 3 + 1);
  // Advanced options live in a right-docked panel (like Chat's settings panel). Closed by
  // default; a single fixed toggle in the top bar opens/closes it.
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Advanced (load-time) options. "auto"/"off" map to the backend defaults (sent through on
  // load). They apply when a model loads; a "Reapply" button reloads with the new values.
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [attentionBackend, setAttentionBackend] = useState<
    "auto" | "native" | "cudnn" | "flash3" | "sage"
  >("auto");
  const [transformerCache, setTransformerCache] = useState<"off" | "fbcache">("off");
  // The last load descriptor, so "Reapply" can reload the same model with new advanced
  // options without the user re-picking it from the dropdown.
  const lastLoad = useRef<{ repoId: string; kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null>(
    null,
  );
  // Whether this session holds a reapply descriptor (set only by our own loads). On a mount/refresh
  // with a model already resident, status.loaded is true but lastLoad is null, so Reapply would
  // silently do nothing -- hide the button in that case rather than offer a dead control.
  const [canReapply, setCanReapply] = useState(false);

  const [busy, setBusy] = useState<Busy>(null);
  // Live per-step progress (phase / step / total + ETA) polled during generation.
  const [genStep, setGenStep] = useState<VideoGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [status, setStatus] = useState<VideoStatus | null>(null);
  // Controlled so the body-portaled overlays force-close when this page is mounted but
  // off-tab (a hidden/inert parent can't contain a body portal): the model selector.
  const [selectorOpen, setSelectorOpen] = useState(false);
  // Records come from the backend (durable); srcById maps each id to its object URL.
  const [videos, setVideos] = useState<GalleryVideo[]>(() => galleryCache.videos);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    Object.fromEntries(galleryCache.srcById),
  );
  // Guards a "load more" so a fast scroll can't fire several at once.
  const loadingMore = useRef(false);
  // False once the page truly unmounts (app close / chat-only eject). The page stays mounted
  // across tab switches, so a switch does NOT flip this.
  const isMounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place (chat-style).
  const loadToastId = useRef<string | number | null>(null);
  // Last load-progress signature shown, so a tick that moved nothing skips the toast.
  const lastLoadSig = useRef<string | null>(null);
  // The quant to restore if the current optimistic swap fails.
  const quantRevert = useRef<{ prev: string | null } | null>(null);

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);

  // Mirror to the module cache so a tab switch re-renders instantly.
  useEffect(() => {
    galleryCache.videos = videos;
    galleryCache.hasMore = hasMore;
    galleryCache.selectedId = selectedId;
    galleryCache.quant = quant;
  }, [videos, hasMore, selectedId, quant]);

  const selected = useMemo(
    () => videos.find((v) => v.id === selectedId) ?? videos[0] ?? null,
    [videos, selectedId],
  );
  const selectedSrc = selected ? srcById[selected.id] : undefined;

  // The resolution presets + temporal lattice for the currently loaded family, or the
  // fallbacks before anything is loaded.
  const resolutionPresets = useMemo<Array<[number, number]>>(() => {
    const presets = status?.defaults?.resolution_presets;
    if (presets && presets.length > 0) {
      return presets.map((p) => [p[0], p[1]] as [number, number]);
    }
    return FALLBACK_RESOLUTION_PRESETS;
  }, [status?.defaults?.resolution_presets]);

  const frameStep = status?.defaults?.frame_step ?? FALLBACK_FRAME_STEP;
  const fps = status?.defaults?.fps ?? FALLBACK_FPS;

  // Duration presets: valid frame counts (k*frame_step+1) closest to ~1s/2s/3s/5s at the
  // current fps. Deduped so two targets that snap to the same count don't repeat.
  const durationOptions = useMemo<Array<{ frames: number; seconds: number }>>(() => {
    const targets = [1, 2, 3, 5];
    const seen = new Set<number>();
    const out: Array<{ frames: number; seconds: number }> = [];
    for (const t of targets) {
      const desired = t * fps;
      const k = Math.max(1, Math.round((desired - 1) / frameStep));
      const frames = k * frameStep + 1;
      if (seen.has(frames)) continue;
      seen.add(frames);
      out.push({ frames, seconds: frames / fps });
    }
    return out;
  }, [frameStep, fps]);

  // Keep the resolution / frame-count selections valid when the loaded family changes.
  useEffect(() => {
    setResolutionIdx((idx) => (idx < resolutionPresets.length ? idx : 0));
  }, [resolutionPresets.length]);
  const loadedFamily = status?.loaded ? status.family : null;
  const familyDefaultFrames = status?.defaults?.num_frames;
  const prevFamilyRef = useRef<string | null>(null);
  useEffect(() => {
    const familyChanged = loadedFamily !== prevFamilyRef.current;
    prevFamilyRef.current = loadedFamily;
    setNumFrames((cur) => {
      // A newly loaded family brings its own default clip length (121 frames for
      // LTX-2); without this the pre-load fallback (25 frames, still on the new
      // lattice) silently sticks and every default run is a ~1s clip.
      if (familyChanged && loadedFamily && familyDefaultFrames) {
        const best = durationOptions.reduce((a, b) =>
          Math.abs(b.frames - familyDefaultFrames) < Math.abs(a.frames - familyDefaultFrames)
            ? b
            : a,
        );
        return best?.frames ?? cur;
      }
      if (durationOptions.some((o) => o.frames === cur)) return cur;
      // Prefer the ~3s preset (index 2) as a sensible default, else the first.
      return durationOptions[2]?.frames ?? durationOptions[0]?.frames ?? cur;
    });
  }, [durationOptions, loadedFamily, familyDefaultFrames]);

  // Seed steps/guidance from the loaded model's backend-authoritative defaults. On mount with a
  // model already loaded (browser refresh, or a load from another client) only refreshStatus runs
  // -- handleModelSelect never fires -- so the controls otherwise stick at the pre-load DEFAULT_GEN
  // (8/1) and a base checkpoint that wants 40/4 silently generates a degraded clip. Key on the repo
  // id so it fires once per newly-loaded model (a distilled vs base checkpoint of the same family
  // has different defaults); a later user edit is not clobbered because the key only changes when
  // the loaded model changes, and a gallery restore (which keeps the same repo) is left untouched.
  const loadedModelKey = status?.loaded ? status.repo_id : null;
  const defaultSteps = status?.defaults?.steps;
  const defaultGuidance = status?.defaults?.guidance;
  const prevLoadedModelRef = useRef<string | null>(null);
  useEffect(() => {
    const modelChanged = loadedModelKey !== prevLoadedModelRef.current;
    prevLoadedModelRef.current = loadedModelKey;
    if (modelChanged && loadedModelKey && defaultSteps != null && defaultGuidance != null) {
      setSteps(defaultSteps);
      setGuidance(defaultGuidance);
    }
  }, [loadedModelKey, defaultSteps, defaultGuidance]);

  // Fetch (once) the object URL for a record's MP4; cached across remounts. Same
  // auth-protected blob pattern the images gallery uses.
  const ensureSrc = useCallback(async (video: GalleryVideo) => {
    if (galleryCache.srcById.has(video.id) || galleryCache.inflight.has(video.id)) return;
    galleryCache.inflight.add(video.id);
    try {
      const url = await fetchGalleryVideoObjectUrl(video.url);
      galleryCache.srcById.set(video.id, url);
      // The URL is cached above either way; skip the state update after unmount
      // (matches the other async callbacks in this file).
      if (isMounted.current) {
        setSrcById((prev) => ({ ...prev, [video.id]: url }));
      }
    } catch {
      // Leave it without a src; the card shows a placeholder.
    } finally {
      galleryCache.inflight.delete(video.id);
    }
  }, []);

  const loadGallery = useCallback(async () => {
    try {
      const page = await getVideoGallery(0, PAGE_SIZE);
      galleryCache.videos = page.videos;
      galleryCache.hasMore = page.has_more;
      setVideos(page.videos);
      setHasMore(page.has_more);
      page.videos.forEach((video) => void ensureSrc(video));
    } catch {
      // Best-effort: a failed gallery load shouldn't block the page.
    }
  }, [ensureSrc]);

  const loadMore = useCallback(async () => {
    if (loadingMore.current || !galleryCache.hasMore) return;
    loadingMore.current = true;
    try {
      const page = await getVideoGallery(galleryCache.videos.length, PAGE_SIZE);
      setVideos((prev) => {
        const seen = new Set(prev.map((v) => v.id));
        const next = [...prev, ...page.videos.filter((v) => !seen.has(v.id))];
        galleryCache.videos = next;
        return next;
      });
      galleryCache.hasMore = page.has_more;
      setHasMore(page.has_more);
      page.videos.forEach((video) => void ensureSrc(video));
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
      await deleteGalleryVideo(id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete video");
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
    setVideos((prev) => prev.filter((v) => v.id !== id));
    setSelectedId((cur) => (cur === id ? null : cur));
  }, []);

  const handleClearAll = useCallback(async () => {
    try {
      await clearVideoGallery();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to clear gallery");
      return;
    }
    for (const url of galleryCache.srcById.values()) {
      if (url.startsWith("blob:")) URL.revokeObjectURL(url);
    }
    galleryCache.srcById.clear();
    galleryCache.videos = [];
    galleryCache.hasMore = false;
    galleryCache.selectedId = null;
    setSrcById({});
    setVideos([]);
    setHasMore(false);
    setSelectedId(null);
  }, []);

  // Load a clip's recipe back into the form inputs.
  const restoreSettings = useCallback(
    (video: GalleryVideo) => {
      setPrompt(video.prompt);
      setNegativePrompt(video.negative_prompt ?? "");
      setSteps(video.steps);
      setGuidance(video.guidance);
      setSeed(String(video.seed));
      // Snap the resolution to the matching preset when one exists; else leave as is.
      const presetIdx = resolutionPresets.findIndex(
        ([w, h]) => w === video.width && h === video.height,
      );
      if (presetIdx >= 0) setResolutionIdx(presetIdx);
      // Restore the frame count when it lies on the current lattice.
      if (durationOptions.some((o) => o.frames === video.num_frames)) {
        setNumFrames(video.num_frames);
      }
      toast.success("Settings restored to inputs");
    },
    [resolutionPresets, durationOptions],
  );

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getVideoStatus());
    } catch {
      // Status is best-effort; a failed poll shouldn't surface an error toast.
    }
  }, []);

  // Track mount so a long generate stops issuing GPU work when the page is truly unmounted.
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  // Re-sync model status when the tab becomes active again: while off-tab the video model
  // may have been evicted (e.g. a chat/image load claimed the GPU).
  useEffect(() => {
    if (!active) return;
    void (async () => {
      await refreshStatus();
    })();
  }, [active, refreshStatus]);

  // Collapse the body-ported model selector when leaving the tab so returning to /video
  // does not pop it back open unprompted.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error", updating the
  // persistent toast in place each tick.
  const pollLoadProgress = useCallback(async () => {
    try {
      const p = await getVideoLoadProgress();
      if (p.phase === "ready") {
        dismissLoadToast();
        setStatus(await getVideoStatus());
        toast.success("Model loaded");
        setBusy(null);
        quantRevert.current = null;
        return;
      }
      if (p.phase === "error") {
        dismissLoadToast();
        toast.error(p.error || "Failed to load model");
        setBusy(null);
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // No load in flight and nothing loaded: the load was cancelled or evicted and the
        // backend cleared its state. Terminal, else this loop spins forever.
        dismissLoadToast();
        setBusy(null);
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      const sig = `${p.phase}:${p.downloaded_bytes}:${p.expected_bytes ?? 0}`;
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
      // A load runs on the backend as a daemon thread that survives navigation. On (re)mount,
      // resume tracking one that's still in flight so the page shows progress and updates on
      // completion, instead of a stale view that never polls.
      try {
        const p = await getVideoLoadProgress();
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
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      dismissLoadToast();
    };
  }, [refreshStatus, dismissLoadToast, pollLoadProgress]);

  const handleLoad = useCallback(
    // Resolves true when the background load STARTED (callers may revert optimistic picker
    // state on false); poll outcomes are handled internally.
    async (
      repoId: string,
      opts: {
        kind: "gguf" | "single_file" | "pipeline";
        filename?: string;
      },
    ): Promise<boolean> => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      setBusy("loading");
      dismissLoadToast();
      lastLoadSig.current = null;
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS));
      lastLoad.current = { repoId, kind: opts.kind, filename: opts.filename };
      setCanReapply(true);
      try {
        // Returns immediately -- the load runs in the background; we poll for it. The backend
        // infers the family + base diffusers repo from the repo id. Advanced options map
        // sentinels ("auto"/"off") to omitted so the backend uses its defaults.
        await loadVideoModel({
          model_path: repoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          memory_mode: memoryMode === "auto" ? undefined : memoryMode,
          speed_mode: speedMode === "auto" ? undefined : speedMode,
          attention_backend: attentionBackend === "auto" ? undefined : attentionBackend,
          transformer_cache: transformerCache === "off" ? undefined : transformerCache,
        });
      } catch (err) {
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
      memoryMode,
      speedMode,
      attentionBackend,
      transformerCache,
    ],
  );

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    const l = lastLoad.current;
    if (l) void handleLoad(l.repoId, { kind: l.kind, filename: l.filename });
  }, [handleLoad]);

  // The chat picker emits (modelId, picked quant + its exact filename) for a GGUF, or just
  // (modelId) for a curated non-GGUF pipeline pick; load it, and seed the inputs with that
  // model's defaults.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight.
      if (busy !== null) return;
      // Curated non-GGUF model: load as a full pipeline.
      const spec = PIPELINE_MODELS[id];
      if (spec) {
        setQuant(null);
        const d = defaultsFor(id);
        setSteps(d.steps);
        setGuidance(d.guidance);
        void handleLoad(id, { kind: spec.kind, filename: spec.filename });
        return;
      }
      // GGUF quant pick from the variant expander. Optimistic for instant picker feedback,
      // reverted if the load fails to START; the poll owns the after-start revert.
      if (meta.ggufVariant && meta.ggufFilename) {
        const prevQuant = quant;
        quantRevert.current = { prev: prevQuant };
        setQuant(meta.ggufVariant);
        // Include the picked filename: the variant (distilled vs dev) lives there,
        // not in the repo id.
        const dq = defaultsFor(`${id}/${meta.ggufFilename}`);
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
      // A direct single-file local .gguf pick has no variant/filename (custom folder / LM
      // Studio). Load it by splitting the path into (parent dir, basename).
      if (meta.isGguf) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        if (!filename.toLowerCase().endsWith(".gguf")) return;
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
      // Otherwise treat it as a full diffusers repo. The backend gates loads to unsloth/*
      // repos, the family bases, or on-device paths, so only attempt those.
      if (meta.source !== "local" && !id.toLowerCase().startsWith("unsloth/")) {
        toast.error("Only unsloth or on-device video models can be loaded here");
        return;
      }
      setQuant(null);
      const d = defaultsFor(id);
      setSteps(d.steps);
      setGuidance(d.guidance);
      void handleLoad(id, { kind: "pipeline" });
    },
    [busy, handleLoad, quant],
  );

  const handleUnload = useCallback(async () => {
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    lastLoad.current = null;
    setCanReapply(false);
    setBusy("unloading");
    try {
      setStatus(await unloadVideoModel());
      setQuant(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
    } finally {
      setBusy(null);
    }
  }, [refreshStatus, dismissLoadToast]);

  const handleCancelGenerate = useCallback(async () => {
    try {
      await cancelVideoGeneration();
    } catch {
      // The generation may have already finished; the poll/finally clears the UI.
    }
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    // Resolve a base seed up front. With an explicit seed the run is reproducible; with a
    // random one we still pick a concrete seed now so the recipe records it.
    let resolvedSeed: number | undefined;
    if (seed.trim()) {
      const n = Number(seed);
      if (!Number.isInteger(n) || n < 0 || n > Number.MAX_SAFE_INTEGER) {
        toast.error("Seed must be a non-negative integer");
        return;
      }
      resolvedSeed = n;
    } else {
      resolvedSeed = Math.floor(Math.random() * 2 ** 32);
    }

    const preset = resolutionPresets[resolutionIdx] ?? resolutionPresets[0];
    const [w, h] = preset;

    setBusy("generating");
    setGenStep(null);
    // Poll the backend's per-step progress so the bar tracks the live denoising steps and
    // the encode phase.
    genPollTimer.current = setInterval(async () => {
      try {
        const p = await getVideoGenerateProgress();
        setGenStep((prev) => {
          if (!p.active) return null;
          if (
            prev &&
            prev.step === p.step &&
            prev.phase === p.phase &&
            prev.eta_seconds === p.eta_seconds
          )
            return prev;
          return p;
        });
      } catch {
        // transient; keep polling
      }
    }, 300);
    try {
      const res = await generateVideo({
        prompt: prompt.trim(),
        // Only send a negative prompt when guidance uses it, so the recipe doesn't record
        // one the model ignored.
        negative_prompt: guidance > 0 ? negativePrompt.trim() || undefined : undefined,
        width: w,
        height: h,
        num_frames: numFrames,
        fps,
        steps,
        guidance,
        seed: resolvedSeed,
      });
      if (!isMounted.current) return;
      // Prepend the new clip (newest first) and load its blob.
      setVideos((prev) => [res.video, ...prev.filter((v) => v.id !== res.video.id)]);
      setSelectedId(res.video.id);
      void ensureSrc(res.video);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Video generation failed";
      // The user's own Cancel comes back as the backend's 409 sentinel; not an error.
      if (!msg.toLowerCase().includes("cancelled")) toast.error(msg);
    } finally {
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      genPollTimer.current = null;
      setBusy(null);
      setGenStep(null);
    }
  }, [
    prompt,
    negativePrompt,
    guidance,
    seed,
    resolutionPresets,
    resolutionIdx,
    numFrames,
    fps,
    steps,
    ensureSrc,
  ]);

  // The Advanced (load-time) tuning controls, rendered in the right-docked panel below.
  const advancedControls = (
    <>
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
        label="Speed"
        hint="Auto picks per model (GGUF compiles near-losslessly, dense stays eager). eager = fused kernels, no compile. default/max add torch.compile (max also TF32 + fused QKV)."
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
      <AdvancedSelect
        label="Attention"
        hint="Attention kernel. Auto upgrades to cuDNN fused attention on NVIDIA when a speed profile is active. sage is INT8 attention (small quality cost)."
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
        label="Step cache"
        hint="First-Block-Cache reuses the transformer tail across steps for many-step models. Leave off for few-step distilled models."
        badge={<ResolvedBadge status={status} controlKey="transformer_cache" />}
        value={transformerCache}
        onValueChange={(v) => setTransformerCache(v as typeof transformerCache)}
        options={[
          ["off", "Off"],
          ["fbcache", "First-Block-Cache"],
        ]}
      />
      {status?.loaded && canReapply && (
        <Button
          variant="secondary"
          size="sm"
          disabled={busy !== null}
          onClick={handleReapply}
          title="Reload the current model with these advanced options"
        >
          Reapply to loaded model
        </Button>
      )}
    </>
  );

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* ── Top: the model selector, kept at the chat tab's exact position so the shared
          element matches. The load progress shows in a chat-style toast, not here. ── */}
      <div className="flex h-[48px] shrink-0 items-start justify-between pl-2 pr-2 pt-[11px]">
        <div className="flex items-center gap-3">
          <ModelSelector
            models={VIDEO_MODELS}
            value={status?.loaded ? status.repo_id ?? undefined : undefined}
            activeGgufVariant={quant}
            onValueChange={handleModelSelect}
            onEject={status?.loaded ? handleUnload : undefined}
            variant="ghost"
            className="!h-[34px]"
            task={VIDEO_GEN_TASKS}
            open={active && selectorOpen}
            onOpenChange={(o) => setSelectorOpen(active && o)}
          />
          {/* Loaded-model status line: family / kind / offload / speed, like the images page
              surfaces on load. Hidden until a model is resident. */}
          {status?.loaded && (
            <div className="hidden items-center gap-3 text-[11px] md:flex">
              {status.family && <StatusChip label="Family" value={status.family} />}
              {status.model_kind && <StatusChip label="Kind" value={status.model_kind} />}
              {status.offload_policy && (
                <StatusChip label="Offload" value={status.offload_policy} />
              )}
              {status.speed_mode && <StatusChip label="Speed" value={status.speed_mode} />}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Single fixed toggle for the right-docked Advanced panel (mirrors Chat's settings
              toggle, same icon in both states so it never moves). Highlighted when open. */}
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
        </div>
      </div>

      {/* ── Controls rail + preview canvas. Padding mirrors the other tabs. ── */}
      <div className="flex min-h-0 min-w-0 flex-1 gap-4 overflow-hidden px-5 pb-8 sm:px-9">
        {/* The controls rail. Plain card with no header -- the prompt + Generate button make
            the panel self-explanatory. */}
        <div className="bg-card corner-squircle flex w-[340px] shrink-0 flex-col gap-4 overflow-y-auto rounded-3xl p-5 ring-1 ring-foreground/10">
          <Field label="Prompt">
            <Textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </Field>

          {/* A negative prompt only does anything with guidance on, so hide it at guidance 0
              (the distilled model's default) instead of showing a dead field. */}
          {guidance > 0 && (
            <Field
              label="Negative prompt"
              hint="What to steer the video away from. Only used when guidance is above 0."
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
            label="Resolution"
            hint="The frame size. Presets come from the loaded model; portrait presets are marked."
          >
            <Select
              value={String(resolutionIdx)}
              onValueChange={(v) => setResolutionIdx(Number(v))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {resolutionPresets.map(([w, h], i) => (
                  <SelectItem key={`${w}x${h}`} value={String(i)}>
                    {w} × {h}
                    {h > w ? " (portrait)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field
            label="Duration"
            hint="Clip length in seconds at the current frame rate. Valid lengths are set by the model's temporal lattice."
          >
            <Select
              value={String(numFrames)}
              onValueChange={(v) => setNumFrames(Number(v))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {durationOptions.map((o) => (
                  <SelectItem key={o.frames} value={String(o.frames)}>
                    {o.seconds.toFixed(1)}s · {o.frames} frames
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
              Frame rate
              <InfoHint>Playback frame rate, fixed per model.</InfoHint>
            </span>
            <span className="font-mono text-xs font-medium text-foreground">{fps} fps</span>
          </div>

          <SliderField
            label="Steps"
            hint="Denoising steps. Distilled models want very few (8); the full base model wants more (40)."
            value={steps}
            min={1}
            max={100}
            step={1}
            onChange={setSteps}
          />
          <SliderField
            label="Guidance"
            hint="Classifier-free guidance scale. Keep low (1) for the distilled model; the base model uses real guidance (4)."
            value={guidance}
            min={0}
            max={20}
            step={0.5}
            onChange={setGuidance}
          />
          <Field label="Seed" hint="Leave empty for a fresh random seed each run.">
            <Input
              placeholder="Random if empty"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </Field>

          {busy === "generating" ? (
            <Button variant="secondary" onClick={handleCancelGenerate}>
              <Spinner className="mr-2 size-4" />
              Cancel
            </Button>
          ) : (
            <Button onClick={handleGenerate} disabled={busy !== null || !status?.loaded}>
              Generate
            </Button>
          )}
        </div>

        <div className="bg-card corner-squircle relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-3xl ring-1 ring-foreground/10">
          <div className="relative flex flex-1 items-center justify-center overflow-auto p-6">
            {selected && selectedSrc ? (
              <>
                {/* The first video element in the app. autoPlay + loop + muted + playsInline
                    so it plays inline without a gesture; controls let the user scrub/unmute. */}
                <video
                  key={selected.id}
                  src={selectedSrc}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="max-h-full max-w-full rounded-xl object-contain shadow-sm"
                />
                {selected.has_audio && (
                  <div className="absolute left-4 top-4 flex items-center gap-1 rounded-lg bg-background/80 px-2 py-1 text-[11px] font-medium shadow-lg ring-1 ring-border backdrop-blur">
                    <HugeiconsIcon icon={VolumeHighIcon} className="size-3.5" />
                    Audio
                  </div>
                )}
                {/* Actions grouped in one glass toolbar so they stay legible over any clip. */}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-xl bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur">
                  <RecipePopover video={selected} onRestore={restoreSettings} active={active} />
                  <Button
                    size="sm"
                    variant="ghost"
                    className="gap-1.5"
                    onClick={() => downloadVideo(selectedSrc, selected)}
                  >
                    <HugeiconsIcon icon={Download01Icon} className="size-4" />
                    Download
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label="Delete video"
                    title="Delete"
                    className="text-muted-foreground hover:text-destructive"
                    onClick={() => void handleDelete(selected.id)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
                </div>
              </>
            ) : selected ? (
              // The selected record's blob is still loading -- spin in place.
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Spinner className="size-8" />
                <p className="text-sm">Loading…</p>
              </div>
            ) : busy === "generating" ? null : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <HugeiconsIcon icon={Video01Icon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a video model to load"}
                </p>
              </div>
            )}

            {/* Live generation progress: a per-step bar with the phase label + ETA, centered
                when there's nothing else to show, tucked at the bottom over a clip. */}
            {busy === "generating" && (
              <div
                className={cn(
                  "pointer-events-none absolute flex justify-center px-4",
                  selectedSrc ? "inset-x-0 bottom-4" : "inset-0 items-center",
                )}
              >
                <div className="w-72 max-w-full rounded-xl bg-background/85 p-3 shadow-lg ring-1 ring-border backdrop-blur">
                  <ModelLoadDescription
                    className="min-h-0"
                    title={null}
                    message="Starting…"
                    progressPercent={
                      genStep && genStep.total > 0 ? (genStep.step / genStep.total) * 100 : null
                    }
                    progressLabel={genStep ? genStepLabel(genStep) : null}
                  />
                </div>
              </div>
            )}
          </div>

          {(videos.length > 0 || busy === "generating") && (
            <div
              className="flex shrink-0 items-stretch gap-2 overflow-x-auto border-t border-foreground/10 p-3"
              onScroll={(e) => {
                // Near the right edge: pull the next older page (infinite scroll).
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past clips stay
                  visible and browsable while the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center rounded-lg bg-muted/50 ring-2 ring-primary/30">
                  <Spinner className="size-5 text-muted-foreground" />
                </div>
              )}
              {videos.map((video) => (
                <button
                  key={video.id}
                  type="button"
                  onClick={() => setSelectedId(video.id)}
                  title={`${video.prompt}\nseed ${video.seed} · ${clipMeta(video)}`}
                  className="relative flex h-16 w-24 shrink-0 flex-col justify-end overflow-hidden rounded-lg bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {srcById[video.id] ? (
                    // Muted, preload="metadata" so the first frame renders as a poster
                    // without playing every card at once.
                    <video
                      src={srcById[video.id]}
                      muted
                      playsInline
                      preload="metadata"
                      className="absolute inset-0 size-full object-cover"
                    />
                  ) : (
                    <span className="absolute inset-0 flex items-center justify-center">
                      <Spinner className="size-4 text-muted-foreground" />
                    </span>
                  )}
                  {/* A terse caption strip so cards read at a glance. */}
                  <span className="relative z-10 truncate bg-gradient-to-t from-black/70 to-transparent px-1 pb-0.5 pt-2 text-left text-[9px] font-medium text-white">
                    {clipMeta(video)}
                  </span>
                  {/* Selection marker on a non-focusable overlay. */}
                  {video.id === selected?.id && (
                    <span className="pointer-events-none absolute inset-0 z-20 rounded-lg border-2 border-primary" />
                  )}
                </button>
              ))}
              {/* Tail spinner while older pages stream in on scroll. */}
              {hasMore && (
                <div className="flex size-16 shrink-0 items-center justify-center">
                  <Spinner className="size-4 text-muted-foreground" />
                </div>
              )}
              {/* Clear-all, tucked at the end so it never sits under a hover. */}
              {videos.length > 0 && (
                <button
                  type="button"
                  onClick={() => void handleClearAll()}
                  title="Clear all videos"
                  className="flex h-16 w-16 shrink-0 flex-col items-center justify-center gap-1 rounded-lg text-muted-foreground ring-1 ring-border transition-colors hover:text-destructive hover:ring-destructive/40"
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  <span className="text-[9px] font-medium">Clear all</span>
                </button>
              )}
            </div>
          )}
        </div>

        {/* Right-docked Advanced panel (mirrors Chat's settings panel): closed by default,
            opened by the single fixed top-bar toggle above. */}
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
    </div>
  );
}
