// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Delete02Icon,
  Download01Icon,
  FlimSlateIcon,
  Image03Icon,
  InformationCircleIcon,
  VolumeHighIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { MediaPageLink } from "@/components/media-page-link";
import { usePlatformStore } from "@/config/env";
import { useHardwareInfo } from "@/hooks/use-hardware-info";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
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
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InfoHint } from "@/components/ui/info-hint";
import { NegativePromptField } from "@/components/negative-prompt-field";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { VIDEO_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import {
  VIDEO_CATALOG,
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
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useStagedDownload } from "@/features/hub/download-manager";
import { cn } from "@/lib/utils";
import { resolveDiffusionGgufFilename } from "@/lib/diffusion-gguf-filename";
import { createPickGuard, runGgufRepoPick } from "@/lib/diffusion-gguf-pick";
import { diffusionRoutePick } from "@/lib/diffusion-route-pick";
import {
  routedGgufFilename,
  routedGgufLabel,
} from "@/lib/diffusion-route-search";
import { toast } from "@/lib/toast";
import { subscribeModelEjected } from "@/lib/model-lifecycle-events";

import {
  type GalleryVideo,
  type VideoGenerateProgress,
  type VideoLoadProgress,
  type VideoStatus,
  cancelVideoGeneration,
  clearVideoGallery,
  deleteGalleryVideo,
  fetchGalleryVideoExport,
  fetchGalleryVideoSignedUrl,
  generateVideo,
  getVideoGallery,
  getVideoGenerateProgress,
  getVideoLoadProgress,
  getVideoDownloadPlan,
  getVideoStatus,
  loadVideoModel,
  unloadVideoModel,
} from "./api";

// Curated models come from the shared catalog: one canonical group per model with its artifacts as data (HunyuanVideo carries both repacks), and the load kind per artifact via loadSpecFor.
// The picker renders groups with a format second level, which also surfaces LTX-2.3 in Recommended (its HF pipeline_tag is image-to-video).
const VIDEO_MODELS: ModelOption[] = catalogToModelOptions(VIDEO_CATALOG);

// Per-model generation defaults (steps + guidance), matched by repo-id substring, most specific first.
const DEFAULT_GEN = { steps: 8, guidance: 1 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  // "distilled" before the generic "ltx": the distilled model runs at 8 steps, guidance 1.
  { match: "distilled", steps: 8, guidance: 1 },
  { match: "ltx", steps: 40, guidance: 4 },
  // Wan2.2 pipelines default to 50 steps at CFG 5.0 (verified in diffusers 0.39). The backend supplies the fps per family.
  { match: "wan", steps: 50, guidance: 5 },
  // HunyuanVideo-1.5 runs 50 steps; guidance 6 matches the guider the repo ships (there is no pipeline kwarg).
  { match: "hunyuanvideo", steps: 50, guidance: 6 },
];

function defaultsFor(repoId: string): { steps: number; guidance: number } {
  const id = repoId.toLowerCase();
  return MODEL_DEFAULTS.find((d) => id.includes(d.match)) ?? DEFAULT_GEN;
}

// Resolution presets offered before a model is loaded. Once loaded, status.defaults.resolution_presets replaces these.
const FALLBACK_RESOLUTION_PRESETS: Array<[number, number]> = [
  [768, 512],
  [1216, 704],
  [704, 1216],
];

// Fallbacks for the duration presets before a model is loaded, so the select is populated and valid on first paint.
const FALLBACK_FRAME_STEP = 8;
const FALLBACK_FPS = 24;

// Module cache of the backend-persisted gallery, so a tab switch re-renders instantly. The srcById entries are short-lived
// signed links, not object URLs: nothing is pinned in the webview, and the media element streams ranges as it plays.
const galleryCache: {
  videos: GalleryVideo[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  // id -> the signed link and when it was minted. The link is short-lived and its signing secret is per-process, while this
  // cache survives navigation, so an entry has to be re-mintable or playback, seeking and Save would 401 until a reload.
  srcById: Map<string, { url: string; mintedAt: number }>;
  // Ids re-minted once after a media error already, so a clip that is broken for any other reason cannot spin in a mint/error loop.
  refreshed: Set<string>;
  // Ids with a mint in flight, so concurrent ensureSrc calls don't double-request.
  inflight: Set<string>;
  // Ids deleted while their link was still being minted, so a reply landing after the delete is not cached. Clear-all bumps the epoch instead.
  deleted: Set<string>;
  epoch: number;
} = {
  videos: [],
  hasMore: false,
  selectedId: null,
  quant: null,
  srcById: new Map(),
  refreshed: new Set(),
  inflight: new Set(),
  deleted: new Set(),
  epoch: 0,
};

// Re-mint a cached link once it is this old, comfortably inside the backend's own expiry, so a long-lived tab keeps working.
const VIDEO_LINK_REFRESH_MS = 6 * 60 * 60 * 1000;

// Videos loaded per infinite-scroll page.
const PAGE_SIZE = 50;

// Export filename, e.g. Unsloth_video_20260624-143005_123.mp4.
type VideoExportFormat = "mp4" | "webm" | "gif";

function exportFilename(video: GalleryVideo, format: VideoExportFormat = "mp4"): string {
  const d = new Date(video.created_at);
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = Number.isNaN(d.getTime())
    ? "unknown"
    : `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
      `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  return `Unsloth_video_${stamp}_${video.seed}.${format}`;
}

function saveLink(href: string, filename: string) {
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  link.click();
}

// MP4 saves the original file straight from its signed link; WebM / GIF are transcoded by the backend on demand (501 when the codec is absent).
async function downloadVideo(
  src: string,
  video: GalleryVideo,
  format: VideoExportFormat = "mp4",
) {
  if (format === "mp4") {
    saveLink(src, exportFilename(video, format));
    return;
  }
  const blob = await fetchGalleryVideoExport(video.id, format);
  const url = URL.createObjectURL(blob);
  try {
    saveLink(url, exportFilename(video, format));
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  }
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

// Bar label for an in-flight generation: the phase ("Denoising step X/Y", "Encoding video...") plus an ETA once known.
function genStepLabel(p: VideoGenerateProgress): string {
  if (p.phase === "export") return "Encoding video…";
  // Text encoding and the first-step warmup run before the first scheduler tick, so step 0 means "working, not denoising yet" -- up to a minute at 720p.
  if (p.step === 0) return "Preparing (text encoding + warmup)…";
  const base = p.total > 0 ? `Denoising step ${p.step}/${p.total}` : "Denoising…";
  const eta = p.eta_seconds != null ? formatEta(p.eta_seconds) : "";
  return eta ? `${base} · ~${eta}` : base;
}

// The chat tab's model-load toast styling, reused verbatim so the video load toast is visually identical.
const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

// The download total for a video load can only be estimated from a companion base repo, so the toast shows a byte count until the total is known.
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

// Toast args mirroring chat: persistent, closeable, content in `description`. Pass `id` to update in place.
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

// Chat's slider, shared with Create. Signature kept for the call sites below.
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

// Matches the field-label style used across Studio (export/chat settings).
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

// The engaged value of a resolved Advanced control, formatted for its "Auto: X" badge (`_native_cudnn` shows as cuDNN).
function formatResolvedValue(value: string | boolean | null): string {
  if (value === null || value === "") return "Off";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === "_native_cudnn" || value.toLowerCase() === "cudnn") return "cuDNN";
  return value.toUpperCase();
}

// The "Auto: X" badge for one Advanced control: rendered only when the backend resolved it (source === "auto").
// The reason is a hover tooltip. Muted pill matching the panel's other chips, same markup as the images page ResolvedBadge.
function ResolvedBadge({
  status,
  controlKey,
}: {
  status: VideoStatus | null;
  controlKey: string;
}) {
  const resolved = status?.resolved?.[controlKey];
  if (!resolved || resolved.source !== "auto") return null;
  const badge = (
    <span className="shrink-0 rounded-sm bg-muted px-1 py-px text-ui-9 font-medium uppercase tracking-wider text-muted-foreground">
      Auto: {formatResolvedValue(resolved.value)}
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
  // Controlled + force-closed off-tab: PopoverContent portals to body, so the inert page wrapper cannot contain it.
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
          <p className="text-ui-11 text-muted-foreground">{formatTimestamp(video.created_at)}</p>
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

// Centered panel used for both halves of the capability gate below: the wait, and the answer.
function VideoGate({ children }: { children: ReactNode }) {
  return (
    <div className="diffusion-surface flex h-full min-h-0 min-w-0 flex-1 flex-col items-center justify-center gap-3 pt-[var(--studio-content-top-inset,0px)] text-center text-sm text-muted-foreground">
      {children}
    </div>
  );
}

/**
 * Capability gate in front of the generator.
 *
 * The root guard never bounces /video: not on the browser-platform guess, and not on a measured
 * chat-only verdict either, because a CPU-only host or a Mac without MLX is precisely where the
 * explanation below has something to say. Video also has no Apple path in the backend, so the
 * page answers for itself: spin while the answer is out, explain when it is no.
 *
 * The spin is bounded: AppSidebar is mounted on this route and re-reads /api/health while the
 * verdict is unknown, writing it to the same store this reads, so a host slower than
 * fetchDeviceType's bounded wait still lands here rather than spinning for the session.
 */
export function VideoPage({ active = true }: { active?: boolean }) {
  const hardware = useHardwareInfo();
  const capabilitiesUnknown = usePlatformStore((s) => s.capabilitiesUnknown());

  if (capabilitiesUnknown || !hardware.loaded) {
    return (
      <VideoGate>
        <Spinner className="size-5" />
        <span>Checking this machine for video support...</span>
      </VideoGate>
    );
  }

  // Only an authoritative "no" hides the generator; an older backend omits the field, which
  // arrives as null, and must keep the page it has always served.
  if (hardware.videoSupported === false) {
    return (
      <VideoGate>
        <HugeiconsIcon
          icon={FlimSlateIcon}
          className="size-7 shrink-0 text-muted-foreground/70"
        />
        <p className="max-w-sm text-balance">
          {hardware.videoUnsupportedMessage ??
            "Video generation is not supported on this machine."}
        </p>
      </VideoGate>
    );
  }

  return <VideoGenerator active={active} />;
}

function VideoGenerator({ active = true }: { active?: boolean }) {
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(
    "a tiny ginger sloth surfing a wave at sunset, cinematic, smooth motion",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [negativeOpen, setNegativeOpen] = useState(false);
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const [seed, setSeed] = useState("");
  // The chosen resolution preset index into the current preset list.
  const [resolutionIdx, setResolutionIdx] = useState(0);
  // The chosen frame count (must lie on the family's temporal lattice: k*frame_step+1).
  const [numFrames, setNumFrames] = useState(FALLBACK_FRAME_STEP * 3 + 1);
  // Advanced options live in a right-docked panel, closed by default; a single fixed top-bar toggle opens it.
  // Sits inline under Seed; the open state is remembered across visits.
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_video_advanced_open",
  );
  // Advanced (load-time) options; "auto"/"off" map to the backend defaults. "Reapply" reloads with new values.
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [attentionBackend, setAttentionBackend] = useState<
    "auto" | "native" | "cudnn" | "flash3" | "sage"
  >("auto");
  const [transformerCache, setTransformerCache] = useState<"auto" | "off" | "fbcache">("auto");
  const [transformerQuant, setTransformerQuant] = useState<
    "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8"
  >("auto");
  // The last load descriptor, so "Reapply" can reload the same model with new advanced options.
  const lastLoad = useRef<{ repoId: string; kind: "gguf" | "single_file" | "pipeline"; filename?: string } | null>(
    null,
  );
  // Whether this session holds a reapply descriptor: with a model already resident, lastLoad is null, so hide the button rather than offer a dead control.
  const [canReapply, setCanReapply] = useState(false);

  const [busy, setBusy] = useState<Busy>(null);
  // Live per-step progress (phase / step / total + ETA) polled during generation.
  const [genStep, setGenStep] = useState<VideoGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // visibilitychange handler active while a generation poll runs: background tabs clamp setInterval, so returning fires one immediate poll.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<VideoStatus | null>(null);
  // Controlled so the body-portaled model selector force-closes when this page is mounted but off-tab.
  const [selectorOpen, setSelectorOpen] = useState(false);
  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  // Records come from the backend (durable); srcById maps each id to its object URL.
  const [videos, setVideos] = useState<GalleryVideo[]>(() => galleryCache.videos);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  // Autoplay replays per selected clip (3 total plays, then pause). Reset on every selection change.
  const playCountRef = useRef(0);
  useEffect(() => {
    playCountRef.current = 0;
  }, [selectedId]);
  // Pause the preview when this page stops being visible: the keep-alive layout only hides it, and display:none does not pause a media element.
  const previewRef = useRef<HTMLVideoElement | null>(null);
  // The media element's own handlers fire while the page is hidden, so they read `active` through a ref.
  const activeRef = useRef(active);
  useEffect(() => {
    activeRef.current = active;
    if (!active) previewRef.current?.pause();
  }, [active]);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    Object.fromEntries([...galleryCache.srcById].map(([id, e]) => [id, e.url])),
  );
  // Guards a "load more" so a fast scroll can't fire several at once.
  const loadingMore = useRef(false);
  // False once the page truly unmounts. The page stays mounted across tab switches, so a switch does NOT flip this.
  const isMounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The persistent load toast's id, so each poll updates it in place (chat-style).
  const loadToastId = useRef<string | number | null>(null);
  // Last load-progress signature shown, so a tick that moved nothing skips the toast.
  const lastLoadSig = useRef<string | null>(null);
  // The quant to restore if the current optimistic swap fails.
  const quantRevert = useRef<{ prev: string | null } | null>(null);
  // The Reapply target (and its canReapply flag) to restore if the optimistic swap fails: handleLoad overwrites lastLoad at
  // load start, and a load failing AFTER that leaves the previous model resident, so the poll rolls it back.
  const lastLoadRevert = useRef<{ prev: typeof lastLoad.current; canReapply: boolean } | null>(null);
  // Which pick owns the page: resolving and staging are requests that do not set `busy`, so a pick can land on an awaiting
  // one. Lazy state, not a ref: a ref cannot be written during render.
  const [pickGuard] = useState(createPickGuard);

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);

  // Client-side state that only means anything while a model is resident: the
  // in-flight replacement load's tracking, and the Reapply target. Shared with
  // the indicator eject, which frees the runtime without going through the
  // page's own Unload.
  const dropResidentState = useCallback(() => {
    // Cancel, not release: a resolving pick or a staged download would load
    // back what was just ejected. In here rather than only in handleUnload, so
    // an eject driven from the loaded models card is covered by it too.
    pickGuard.cancel();
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    // Leaving this set would let Reapply reload the model that was just freed.
    lastLoad.current = null;
    setCanReapply(false);
  }, [dismissLoadToast, pickGuard]);

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

  // The resolution presets + temporal lattice for the currently loaded family, or the fallbacks before anything is loaded.
  const resolutionPresets = useMemo<Array<[number, number]>>(() => {
    const presets = status?.defaults?.resolution_presets;
    if (presets && presets.length > 0) {
      return presets.map((p) => [p[0], p[1]] as [number, number]);
    }
    return FALLBACK_RESOLUTION_PRESETS;
  }, [status?.defaults?.resolution_presets]);

  const frameStep = status?.defaults?.frame_step ?? FALLBACK_FRAME_STEP;
  const fps = status?.defaults?.fps ?? FALLBACK_FPS;

  // Duration presets: valid frame counts (k*frame_step+1) closest to ~1s/2s/3s/5s at the current fps, deduped.
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
      // A newly loaded family brings its own default clip length; without this the pre-load fallback sticks and every default run is a ~1s clip.
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

  // Seed steps/guidance from the loaded model's backend defaults: on mount with a model already loaded only refreshStatus runs, so the
  // controls would stick at the pre-load DEFAULT_GEN and a base checkpoint wanting 40/4 generates a degraded clip. Keyed on the resolved
  // schedule, not the repo alone: a GGUF repo holds several variants, so another client swapping builds changes the defaults in place.
  const defaultSteps = status?.defaults?.steps;
  const defaultGuidance = status?.defaults?.guidance;
  const loadedModelKey = status?.loaded
    ? `${status.repo_id ?? ""}|${defaultSteps ?? ""}|${defaultGuidance ?? ""}`
    : null;
  const prevLoadedModelRef = useRef<string | null>(null);
  useEffect(() => {
    const modelChanged = loadedModelKey !== prevLoadedModelRef.current;
    prevLoadedModelRef.current = loadedModelKey;
    if (modelChanged && loadedModelKey && defaultSteps != null && defaultGuidance != null) {
      setSteps(defaultSteps);
      setGuidance(defaultGuidance);
    }
  }, [loadedModelKey, defaultSteps, defaultGuidance]);

  // Mint (once) a playable link for a record's MP4, cached across remounts. Unlike the images gallery this does NOT download
  // the file: the link goes straight into the <video> element, which streams ranges, so playback starts and seeking works.
  const ensureSrc = useCallback(async (video: GalleryVideo) => {
    const cached = galleryCache.srcById.get(video.id);
    if (cached && Date.now() - cached.mintedAt < VIDEO_LINK_REFRESH_MS) return;
    if (galleryCache.inflight.has(video.id)) return;
    galleryCache.inflight.add(video.id);
    const epochAtStart = galleryCache.epoch;
    try {
      const url = await fetchGalleryVideoSignedUrl(video.id);
      // The record can be deleted (or the gallery cleared) while the link is being minted; caching it then would strand an entry.
      if (galleryCache.deleted.has(video.id) || galleryCache.epoch !== epochAtStart) return;
      galleryCache.srcById.set(video.id, { url, mintedAt: Date.now() });
      // The URL is cached above either way; skip the state update after unmount (matches the other async callbacks in this file).
      if (isMounted.current) setSrcById((prev) => ({ ...prev, [video.id]: url }));
    } catch {
      // Leave it without a src; the card shows a placeholder.
    } finally {
      galleryCache.inflight.delete(video.id);
    }
  }, []);

  // A media error on a playing clip means its link died early (the server restarted, changing its signing secret). Re-mint once per clip per session.
  const remintSrc = useCallback(
    (video: GalleryVideo) => {
      if (galleryCache.refreshed.has(video.id)) return;
      galleryCache.refreshed.add(video.id);
      galleryCache.srcById.delete(video.id);
      void ensureSrc(video);
    },
    [ensureSrc],
  );

  // A card's poster frame appears once its src lands and each src costs a request, so minting a full page up front would queue PAGE_SIZE
  // requests ahead of the clip being waited on. Mint as a card nears the viewport, observed from here (the tile is a Tooltip trigger), per page.
  const stripRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const root = stripRef.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const id = (entry.target as HTMLElement).dataset.clipId;
          if (!id) continue;
          const clip = videos.find((v) => v.id === id);
          if (clip) void ensureSrc(clip);
        }
      },
      // rootMargin is added to the ROOT box only, so the root has to be the strip itself: a card past its right edge is clipped,
      // and a viewport-root margin would never reach it. The strip scrolls horizontally, so the sideways margin is the one that matters.
      { root, rootMargin: "0px 600px" },
    );
    for (const card of root.querySelectorAll("[data-clip-id]")) io.observe(card);
    return () => io.disconnect();
  }, [videos, ensureSrc]);

  // The preview player is what the user watches, so the selected clip is fetched whether or not its card is on screen.
  useEffect(() => {
    if (!selected) return;
    void (async () => {
      await ensureSrc(selected);
    })();
  }, [selected, ensureSrc]);

  const loadGallery = useCallback(async () => {
    try {
      const page = await getVideoGallery(0, PAGE_SIZE);
      galleryCache.videos = page.videos;
      galleryCache.hasMore = page.has_more;
      setVideos(page.videos);
      setHasMore(page.has_more);
      // No visibility signal without IntersectionObserver (jsdom / old webview), so keep the eager fetch there.
      if (typeof IntersectionObserver === "undefined") {
        page.videos.forEach((video) => void ensureSrc(video));
      }
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
      if (typeof IntersectionObserver === "undefined") {
        page.videos.forEach((video) => void ensureSrc(video));
      }
    } catch {
      // transient; the user can scroll again to retry
    } finally {
      loadingMore.current = false;
    }
  }, [ensureSrc]);

  useEffect(() => {
    void loadGallery();
  }, [loadGallery]);

  // WebM/GIF go through a server-side transcode that can take seconds (and 501s when the codec is missing), so wrap the helper with toasts.
  const handleDownload = useCallback(
    async (src: string, video: GalleryVideo, format: "mp4" | "webm" | "gif") => {
      if (format === "mp4") {
        void downloadVideo(src, video, format);
        return;
      }
      const toastId = toast.loading(`Converting to ${format.toUpperCase()}…`);
      try {
        await downloadVideo(src, video, format);
        toast.dismiss(toastId);
      } catch (err) {
        toast.dismiss(toastId);
        toast.error(
          err instanceof Error ? err.message : `Failed to export ${format}`,
        );
      }
    },
    [],
  );

  const handleDelete = useCallback(async (id: string) => {
    try {
      await deleteGalleryVideo(id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete video");
      return;
    }
    galleryCache.srcById.delete(id);
    galleryCache.refreshed.delete(id);
    // A mint still in flight for this id must throw its link away rather than cache it.
    galleryCache.deleted.add(id);
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
    galleryCache.srcById.clear();
    galleryCache.refreshed.clear();
    // Every mint in flight now belongs to a cleared gallery, so their links are discarded on arrival. The epoch covers unlisted ids too.
    galleryCache.epoch += 1;
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
      const restoredNegative = video.negative_prompt ?? "";
      setNegativePrompt(restoredNegative);
      if (restoredNegative) setNegativeOpen(true);
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

  // Re-sync model status when the tab becomes active again: while off-tab the video model may have been evicted.
  useEffect(() => {
    if (!active) return;
    void (async () => {
      await refreshStatus();
    })();
  }, [active, refreshStatus]);

  // Ejected from the loaded models indicator, which does not run handleUnload:
  // without this the controls keep offering to generate on a freed runtime, and
  // Reapply still points at the model that was just ejected. The runtime is
  // already free, so this is handleUnload without the unload call.
  useEffect(
    () =>
      subscribeModelEjected("video", () => {
        dropResidentState();
        // That eject cancelled the replacement load, and its progress poll is
        // the only thing that clears `busy` -- which dropResidentState has just
        // stopped. Leaving it set locks the page: the picker ignores every
        // choice while busy, and Unload is not offered once the status read
        // comes back empty, so only an app reload recovered. Narrowed to
        // "loading" so a generation in flight is left alone, as handleUnload's
        // own finally does.
        setBusy((prev) => (prev === "loading" ? null : prev));
        setQuant(null);
        void refreshStatus();
      }),
    [refreshStatus, dropResidentState],
  );

  // Collapse the body-ported model selector when leaving the tab so returning to /video does not pop it back open unprompted.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error", updating the persistent toast in place each tick.
  const pollLoadProgress = useCallback(async () => {
    try {
      const p = await getVideoLoadProgress();
      if (p.phase === "ready") {
        dismissLoadToast();
        setStatus(await getVideoStatus());
        toast.success("Model loaded");
        setBusy(null);
        quantRevert.current = null;
        // lastLoad.current already holds the now-resident pick, so drop its revert too.
        lastLoadRevert.current = null;
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
        // Same rollback for the Reapply target: the previous model is still resident, so point Reapply back at it.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.canReapply);
          lastLoadRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // No load in flight and nothing loaded: the load was cancelled or evicted. Terminal, else this loop spins forever.
        dismissLoadToast();
        setBusy(null);
        if (quantRevert.current) {
          setQuant(quantRevert.current.prev);
          quantRevert.current = null;
        }
        // Restore the Reapply target too, so it never lingers on the failed pick after a cancel or eviction.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.canReapply);
          lastLoadRevert.current = null;
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

  // Stops the generation poll and its visibilitychange catch-up listener.
  const stopGenPoll = useCallback(() => {
    if (genPollTimer.current) clearInterval(genPollTimer.current);
    genPollTimer.current = null;
    if (genVisibilityListener.current) {
      document.removeEventListener("visibilitychange", genVisibilityListener.current);
      genVisibilityListener.current = null;
    }
  }, []);

  // Poll the backend's per-step progress so the bar tracks denoising and the encode phase, driving completion off the terminal
  // phase. A named poll body also serves the visibilitychange listener. Shared by handleGenerate and the mount-time resume.
  const startGenPoll = useCallback(() => {
    stopGenPoll();
    let pollInFlight = false;
    const pollGenerateOnce = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const p = await getVideoGenerateProgress();
        if (p.phase === "completed" || p.phase === "failed") {
          stopGenPoll();
          if (!isMounted.current) return;
          setBusy(null);
          setGenStep(null);
          if (p.phase === "completed" && p.video) {
            // Prepend the new clip (newest first) and mint its link.
            const clip = p.video;
            setVideos((prev) => [clip, ...prev.filter((v) => v.id !== clip.id)]);
            setSelectedId(clip.id);
            void ensureSrc(clip);
          } else if (p.phase === "failed") {
            const msg = p.error || "Video generation failed";
            // The user's own Cancel surfaces as the backend's cancelled sentinel; not an error.
            if (!msg.toLowerCase().includes("cancelled")) toast.error(msg);
          }
          return;
        }
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
      } finally {
        pollInFlight = false;
      }
    };
    genVisibilityListener.current = () => {
      if (document.visibilityState === "visible") void pollGenerateOnce();
    };
    document.addEventListener("visibilitychange", genVisibilityListener.current);
    genPollTimer.current = setInterval(() => void pollGenerateOnce(), 300);
  }, [ensureSrc, stopGenPoll]);

  useEffect(() => {
    void (async () => {
      await refreshStatus();
      // A load runs on the backend as a daemon thread that survives navigation, so resume tracking one still in flight.
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
      // A generation also runs on a daemon thread, so a reload mid-denoise must re-enter the same poll loop rather than show an idle page.
      try {
        const g = await getVideoGenerateProgress();
        if (g.active) {
          setBusy("generating");
          setGenStep(g.phase === "queued" ? null : g);
          startGenPoll();
        } else if (g.phase === "completed" && g.video) {
          // The job finished while no page was mounted. The terminal record persists until the next job; merging here covers the race where it completed after the mount fetch.
          const clip = g.video;
          // Deleted this session: the backend clears its terminal record on delete, but a client racing that must not merge a record whose file is gone.
          if (!galleryCache.deleted.has(clip.id)) {
            setVideos((prev) => (prev.some((v) => v.id === clip.id) ? prev : [clip, ...prev]));
            void ensureSrc(clip);
          }
        } else if (g.phase === "failed") {
          // The other terminal phase, kept only until the next job: without this a reload after a failed generation shows an idle page and loses the error.
          const msg = g.error || "Video generation failed";
          if (!msg.toLowerCase().includes("cancelled")) toast.error(msg);
        }
      } catch {
        // Resume is best-effort; a failed probe just leaves the idle view.
      }
    })();
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      stopGenPoll();
      dismissLoadToast();
    };
  }, [refreshStatus, dismissLoadToast, pollLoadProgress, startGenPoll, stopGenPoll, ensureSrc]);

  const handleLoad = useCallback(
    // Resolves true when the background load STARTED (callers may revert optimistic picker state on false).
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
      // Snapshot the prior Reapply target first: a load that fails to START leaves the previous model resident, so Reapply must keep pointing at it.
      const prevLastLoad = lastLoad.current;
      const prevCanReapply = canReapply;
      lastLoad.current = { repoId, kind: opts.kind, filename: opts.filename };
      setCanReapply(true);
      // Carry the prior target so the async poll can restore it if the background load fails after starting.
      lastLoadRevert.current = { prev: prevLastLoad, canReapply: prevCanReapply };
      try {
        // Returns immediately -- the load runs in the background and we poll. The backend infers the family + base repo from the id; the "auto"/"off" sentinels map to omitted.
        await loadVideoModel({
          model_path: repoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          memory_mode: memoryMode === "auto" ? undefined : memoryMode,
          speed_mode: speedMode === "auto" ? undefined : speedMode,
          attention_backend: attentionBackend === "auto" ? undefined : attentionBackend,
          transformer_cache: transformerCache === "auto" ? undefined : transformerCache,
          transformer_quant: transformerQuant === "auto" ? undefined : transformerQuant,
        });
      } catch (err) {
        lastLoad.current = prevLastLoad;
        setCanReapply(prevCanReapply);
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
    [
      pollLoadProgress,
      refreshStatus,
      dismissLoadToast,
      canReapply,
      memoryMode,
      speedMode,
      attentionBackend,
      transformerCache,
      transformerQuant,
    ],
  );

  // Downloads go through the Hub download manager like every other model, sharing its panel, progress, cancel and preflight. Mirrors Images.
  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string };
    // The pick that staged it: a download outlives its pick, so it must not evict a newer one when it lands.
    token: number;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  handleLoadRef.current = handleLoad;
  // A download finishing while this page is hidden must not evict the model the visible page loaded. The pick is held, not dropped.
  const stagedLoadDeferred = useRef(false);
  // A pick made while the download ran already owns the page; only the newest may load. `isLatest`, not `holds`: leaving the
  // page is not a new pick, and the deferred load below is exactly that case.
  const runStagedLoad = useCallback(() => {
    const pending = pendingStagedLoad.current;
    pendingStagedLoad.current = null;
    if (!pending || !pickGuard.isLatest(pending.token)) return;
    void handleLoadRef.current(pending.repoId, pending.opts);
  }, [pickGuard]);
  const { stage } = useStagedDownload({
    scopeId: "diffusion",
    onReady: () => {
      if (!active) {
        stagedLoadDeferred.current = true;
        return;
      }
      runStagedLoad();
    },
  });

  useEffect(() => {
    if (!active || !stagedLoadDeferred.current) return;
    stagedLoadDeferred.current = false;
    runStagedLoad();
  }, [active, runStagedLoad]);

  // Stage a not-yet-downloaded hub pick, else load it directly.
  // `token` lets an awaiting caller drop out: the plan below is a second window for a newer pick to take the page.
  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: { kind: "gguf" | "single_file" | "pipeline"; filename?: string },
      isDownloaded?: boolean,
      token?: number,
    ): Promise<boolean> => {
      const owns = () => token === undefined || pickGuard.holds(token);
      if (isDownloaded !== false) return handleLoadRef.current(repoId, opts);
      try {
        const plan = await getVideoDownloadPlan({
          model_path: repoId,
          gguf_filename: opts.filename,
          model_kind: opts.kind,
          // Same token handleLoad sends: without it the metadata lookup fails on a gated base and the plan drops the companion entry, so the load pulls those files inline.
          hf_token: hfApiToken(getHfToken()),
        });
        // Superseded mid-plan: neither stage nor load, and leave `pendingStagedLoad` to its new owner.
        if (!owns()) return false;
        if (plan.entries.length > 0) {
          pendingStagedLoad.current = { repoId, opts, token: token ?? pickGuard.claim() };
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
      if (!owns()) return false;
      return handleLoadRef.current(repoId, opts);
    },
    [stage, pickGuard],
  );

  // A GGUF pick can arrive with only a repo id (a pinned row, a curated artifact, a local GGUF directory). The backend
  // rejects a gguf load with no filename and a pipeline load of a GGUF repo, so name the file from the listing first.
  const loadGgufRepoPick = useCallback(
    async (
      repoId: string,
      quantHint: string | null,
      isDownloaded?: boolean,
      localPath?: string | null,
    ): Promise<boolean> => {
      // Claimed here so every entry point is covered; the next pick's claim makes this one inert.
      const token = pickGuard.claim();
      const isCurrent = () => isMounted.current && pickGuard.holds(token);
      const prevQuant = quant;
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
          quantRevert.current = { prev: prevQuant };
          setQuant(quantHint ?? filename);
          // Filename-qualified like the expander branch: the LTX variant lives in the checkpoint name, not the repo id.
          const d = defaultsFor(`${repoId}/${filename}`);
          setSteps(d.steps);
          setGuidance(d.guidance);
        },
        onNotStarted: () => {
          setQuant(prevQuant);
          quantRevert.current = null;
        },
        load: (filename) =>
          loadOrStage(repoId, { kind: "gguf", filename }, isDownloaded, token),
      });
    },
    [loadOrStage, pickGuard, quant],
  );

  // A hidden page owns nothing: both stay mounted, so a resolution started here must not load after the user switched.
  useEffect(() => {
    if (!active) pickGuard.release();
  }, [active, pickGuard]);

  // A diffusion model picked from the chat picker arrives as ?model= on this route. Load it once, then clear the params.
  const routeSearch = useSearch({ strict: false }) as {
    model?: string;
    quant?: string;
    ggufQuant?: string;
  };
  const navigateSelf = useNavigate();
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    // Only the page being shown consumes the query: this hook is loose and both diffusion pages stay mounted, so the hidden one
    // saw /images?model= too and raced that page, trying to load an image checkpoint as a video model.
    if (!active) return;
    const wanted = routeSearch.model;
    // Model AND quant, released once the query is gone: this page stays mounted, so a marker that outlived the query made re-picking a dead click.
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    // `quant` is used verbatim as a filename; a label there (a hand-built link, an older producer) is resolved instead.
    // The two fields, not the object: `routeSearch` is rebuilt every render, so it would churn the deps.
    const routed = { quant: routeSearch.quant, ggufQuant: routeSearch.ggufQuant };
    const routedFilename = routedGgufFilename(routed);
    const routedLabel = routedGgufLabel(routed);
    const key = `${wanted}|${routeSearch.quant ?? ""}|${routeSearch.ggufQuant ?? ""}`;
    if (handledRouteModel.current === key) return;
    handledRouteModel.current = key;
    // This arrival owns the page like a direct pick, so a download staged by an earlier one cannot land on top.
    const token = pickGuard.claim();
    void navigateSelf({ to: "/video", search: {}, replace: true });
    // A label means a GGUF repo whatever the catalog says, and is not loadable, so resolve it instead of routing it as a
    // filename.
    if (routedLabel) {
      // Deferred, not inline: resolution is a request, and the load it fires owns the state a direct pick sets.
      void Promise.resolve().then(() =>
        loadGgufRepoPick(wanted, routedLabel, false),
      );
      return;
    }
    // Same catalog lookup a direct pick makes: the chat picker can only forward a GGUF filename, so a curated single-file artifact would load as a pipeline and fail.
    const pick = diffusionRoutePick(
      wanted,
      routedFilename ?? undefined,
      loadSpecFor(wanted, VIDEO_CATALOG),
    );
    // A curated GGUF artifact resolves to kind "gguf" with no filename: the catalog lists the repo, not its files.
    if (pick.opts.kind === "gguf" && !pick.opts.filename) {
      void Promise.resolve().then(() => loadGgufRepoPick(pick.repoId, null, false));
      return;
    }
    void loadOrStage(pick.repoId, pick.opts, false, token);
  }, [
    active,
    routeSearch.model,
    routeSearch.quant,
    routeSearch.ggufQuant,
    loadOrStage,
    loadGgufRepoPick,
    navigateSelf,
    pickGuard,
  ]);

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    const l = lastLoad.current;
    if (l) void handleLoad(l.repoId, { kind: l.kind, filename: l.filename });
  }, [handleLoad]);

  // The chat picker emits (modelId, quant + filename) for a GGUF, or just (modelId) for a curated pipeline pick.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight.
      if (busy !== null) return;
      // This pick owns the page now, so one still awaiting a listing or a plan drops out. Before any branch: staging never
      // sets `busy`, so any pick can land on an awaiting one.
      const token = pickGuard.claim();
      // Curated non-GGUF model: load as a full pipeline.
      const spec = loadSpecFor(id, VIDEO_CATALOG);
      if (spec && spec.kind !== "gguf") {
        setQuant(null);
        // The distilled variant lives in the checkpoint name, not the repo id, so include the filename when seeding defaults.
        // Without it these distilled entries fall through to the generic LTX 40-step/CFG-4 defaults instead of the 8-step schedule.
        const d = defaultsFor(spec.filename ? `${id}/${spec.filename}` : id);
        setSteps(d.steps);
        setGuidance(d.guidance);
        void loadOrStage(
          id,
          { kind: spec.kind, filename: spec.filename },
          meta.isDownloaded,
          token,
        );
        return;
      }
      // GGUF quant pick from the variant expander. Optimistic for picker feedback, reverted if the load fails to START; the poll owns the after-start revert.
      if (meta.ggufVariant && meta.ggufFilename) {
        const prevQuant = quant;
        quantRevert.current = { prev: prevQuant };
        setQuant(meta.ggufVariant);
        // Include the picked filename: the variant (distilled vs dev) lives there, not in the repo id.
        const dq = defaultsFor(`${id}/${meta.ggufFilename}`);
        setSteps(dq.steps);
        setGuidance(dq.guidance);
        void loadOrStage(
          id,
          { kind: "gguf", filename: meta.ggufFilename },
          meta.isDownloaded,
          token,
        ).then((started) => {
          // `quantRevert` is one slot, so only the pick that set the label may take it back.
          if (!started && pickGuard.holds(token)) {
            setQuant(prevQuant);
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
          // A repo id or local directory, not a file. The listing names its .gguf and the label picks between siblings; a
          // local pick passes its directory so the listing reads that path, not a hub repo.
          void loadGgufRepoPick(
            id,
            meta.ggufVariant ?? null,
            meta.isDownloaded,
            meta.source === "local" ? id : null,
          );
          return;
        }
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
      // A direct local .safetensors pick must load via from_single_file: the pipeline route rejects a bare file, and only after evicting the resident model.
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
      // A GGUF repo with no filename: these used to fall through to the pipeline branch below, which the backend rejects
      // for a single-file GGUF repo.
      if (spec?.kind === "gguf" || meta.ggufVariant) {
        // An artifact that names its file short-circuits the listing; otherwise the label is the hint.
        void loadGgufRepoPick(
          id,
          spec?.filename ?? meta.ggufVariant ?? null,
          meta.isDownloaded,
          meta.source === "local" ? id : null,
        );
        return;
      }
      // Otherwise treat it as a full diffusers repo. The backend gates loads to unsloth/* repos, the family bases, or on-device paths.
      if (meta.source !== "local" && !id.toLowerCase().startsWith("unsloth/")) {
        toast.error("Only unsloth or on-device video models can be loaded here");
        return;
      }
      setQuant(null);
      const d = defaultsFor(id);
      setSteps(d.steps);
      setGuidance(d.guidance);
      void loadOrStage(id, { kind: "pipeline" }, meta.isDownloaded, token);
    },
    [busy, handleLoad, loadGgufRepoPick, loadOrStage, pickGuard, quant],
  );

  const handleUnload = useCallback(async () => {
    dropResidentState();
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
  }, [refreshStatus, dropResidentState]);

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
    // Resolve a base seed up front: with a random one we still pick a concrete seed now so the recipe records it.
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
    // The POST only STARTS the job and returns at once (a clip takes minutes, and the secure-mode tunnel caps responses near 100s).
    // A synchronous rejection still surfaces here; everything after acceptance arrives via the poll.
    try {
      await generateVideo({
        prompt: prompt.trim(),
        // Only send a negative prompt when guidance uses it, so the recipe does not record one the model ignored.
        negative_prompt: guidance > 0 ? negativePrompt.trim() || undefined : undefined,
        width: w,
        height: h,
        num_frames: numFrames,
        fps,
        steps,
        guidance,
        seed: resolvedSeed,
      });
    } catch (err) {
      if (!isMounted.current) return;
      toast.error(err instanceof Error ? err.message : "Video generation failed");
      setBusy(null);
      setGenStep(null);
      return;
    }
    // Track live progress + the terminal outcome via the shared poll loop (also used by the mount-time resume).
    startGenPoll();
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
    startGenPoll,
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
        hint="Auto compiles every model at load: a clip takes minutes to denoise, so the one-time compile always pays for itself within a single run. eager = fused kernels, no compile. max adds TF32 + fused QKV."
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
      {/* The dense transformer_quant fast path only engages on a full-pipeline load, so gate the control and otherwise show why it is unavailable. */}
      {!status?.loaded || status.model_kind === "pipeline" ? (
        <AdvancedSelect
          label="Precision"
          hint="How the model computes. Auto picks the fastest precision the hardware supports (at least INT8 on a capable GPU; FP8 on data-center cards) by quantising the transformer onto low-precision tensor cores, and keeps plain bf16 when the device or memory plan can't take it. Off always runs bf16."
          badge={<ResolvedBadge status={status} controlKey="transformer_quant" />}
          value={transformerQuant}
          onValueChange={(v) => setTransformerQuant(v as typeof transformerQuant)}
          options={[
            ["auto", "Auto (fastest for GPU)"],
            ["none", "Off (bf16)"],
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
          <span className="text-xs text-muted-foreground/60">Full-pipeline models only</span>
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
        label="Step cache"
        hint="First-Block-Cache reuses the transformer tail across steps for many-step models. Auto turns it on at 20+ steps and off for few-step distilled models, re-checked per clip."
        badge={<ResolvedBadge status={status} controlKey="transformer_cache" />}
        value={transformerCache}
        onValueChange={(v) => setTransformerCache(v as typeof transformerCache)}
        options={[
          ["auto", "Auto"],
          ["off", "Off"],
          ["fbcache", "First-Block-Cache"],
        ]}
      />
      {status?.loaded && canReapply && (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button
              variant="outline"
              size="sm"
              disabled={busy !== null}
              onClick={handleReapply}
            >
              Reapply to loaded model
            </Button>
          </TooltipTrigger>
          <TooltipContent>Reload the current model with these advanced options</TooltipContent>
        </Tooltip>
      )}
    </>
  );

  return (
    // The chat-style layout gives this page no outer top inset, so clear the custom
    // titlebar here (34px on win/linux, 0 under macOS's native one) as chat does.
    <div className="diffusion-surface flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]">
      {/* Top: the model selector, sitting clear of the sidebar and level with the controls column below. Load progress shows in a toast. */}
      <div className="pointer-events-none relative z-40 flex h-[48px] shrink-0 items-start justify-between pl-[var(--studio-media-header-left-inset,1.5rem)] pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
        <div className="pointer-events-auto flex items-center gap-3">
          <ModelSelector
            models={VIDEO_MODELS}
            value={status?.loaded ? status.repo_id ?? undefined : undefined}
            activeGgufVariant={quant}
            onValueChange={handleModelSelect}
            onEject={status?.loaded ? handleUnload : undefined}
            variant="ghost"
            className="!h-[34px]"
            task={VIDEO_GEN_TASKS}
            catalog={VIDEO_CATALOG}
            placeholder="Select video model"
            open={active && selectorOpen}
            onOpenChange={(o) => setSelectorOpen(active && o)}
          />
          {/* Loaded-model status line: family / kind / offload / speed, as the images page surfaces on load. Hidden until a model is resident. */}
          {status?.loaded && (
            <div className="hidden items-center gap-3 text-ui-11 md:flex">
              {status.family && <StatusChip label="Family" value={status.family} />}
              {status.model_kind && <StatusChip label="Kind" value={status.model_kind} />}
              {status.offload_policy && (
                <StatusChip label="Offload" value={status.offload_policy} />
              )}
              {status.speed_mode && <StatusChip label="Speed" value={status.speed_mode} />}
            </div>
          )}
        </div>
        <div className="pointer-events-auto flex items-center gap-2">
          {/* Images is a separate page, so it sits out here, not in this page's controls. */}
          <MediaPageLink to="/images" label="Images" icon={Image03Icon} />
        </div>
      </div>

      {/* Controls rail + preview canvas, as on the Images tabs: no cards, a rule the full page
          height. Full width, so the preview grows with the window.
          Gutters match Images, so both pages' content starts at the same 40px. */}
      {/* overflow-x-hidden: an unset overflow-x computes to auto beside overflow-y-auto,
          letting a wide row pan the page sideways on a phone. */}
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto overflow-x-hidden pl-2 pr-5 pt-9 sm:pr-8 md:flex-row md:overflow-hidden">
        {/* Widened by the pl-8 so the controls keep their old width. */}
        <div className="relative flex w-full shrink-0 flex-col border-b border-border/60 pl-8 md:w-[400px] md:overflow-hidden md:border-r md:border-b-0">
          {/* pl-0.5 keeps focus rings off the scroll container's edge. */}
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              // pb-20 at every width: the floating Generate button below is absolutely
              // positioned over this rail and stands 72px tall (h-11 + pb-7), so a smaller
              // phone padding puts it on top of the last control.
              "hover-scrollbar panel-scroll-fade flex min-h-0 flex-1 flex-col gap-4 pb-20 pl-0.5 pr-7 md:overflow-y-auto",
              settingsFadeClass,
            )}
          >
          {/* Names the pane, as the Images column does. Same shape there, so
              the two pages stay level. */}
          <div className="mb-2 grid gap-1.5">
            <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none text-foreground">
              {/* The app's Video icon, same as the sidebar row. */}
              <HugeiconsIcon icon={FlimSlateIcon} className="size-[18px] shrink-0" />
              Create videos
            </h2>
            <p className="text-xs leading-snug text-muted-foreground">
              Generate a video from a prompt
            </p>
          </div>

          <Field label="Prompt">
            <Textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </Field>

          <NegativePromptField
            value={negativePrompt}
            onChange={setNegativePrompt}
            open={negativeOpen}
            onOpenChange={setNegativeOpen}
            hint="What to steer the video away from. Only used when guidance is above 0."
          />

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
          {/* Floats over the settings so it needs no bar of its own. */}
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center pb-7 pl-8 pr-7">
            {busy === "generating" ? (
              <Button
                // Opaque hover: this one floats over the settings too.
                className="pointer-events-auto h-11 px-8 hover:bg-muted dark:hover:bg-muted"
                variant="outline"
                onClick={handleCancelGenerate}
              >
                <Spinner className="mr-2 size-4" />
                Cancel
              </Button>
            ) : (
              <Button
                className="btn-float-action pointer-events-auto h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                onClick={handleGenerate}
                disabled={busy !== null || !status?.loaded}
              >
                Generate
              </Button>
            )}
          </div>
        </div>

        <div className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden pl-2 md:min-h-0">
          <div className="hover-scrollbar relative flex flex-1 items-center justify-center overflow-auto p-6">
            {selected && selectedSrc ? (
              <>
                {/* The first video element in the app. autoPlay + muted + playsInline so it plays inline without a gesture; controls let the user scrub. onEnded replays up to 3 total plays, and the counter resets per selection. */}
                <video
                  key={selected.id}
                  ref={previewRef}
                  src={selectedSrc}
                  controls
                  autoPlay
                  muted
                  playsInline
                  onPlay={() => {
                    playCountRef.current += 1;
                  }}
                  onEnded={(e) => {
                    // Not while hidden: a replay here would restart audio on another page.
                    if (activeRef.current && playCountRef.current < 3) {
                      e.currentTarget.currentTime = 0;
                      void e.currentTarget.play();
                    }
                  }}
                  onError={() => remintSrc(selected)}
                  className="max-h-full max-w-full object-contain shadow-sm"
                />
                {selected.has_audio && (
                  <div className="absolute left-4 top-4 flex items-center gap-1 rounded-lg bg-background/80 px-2 py-1 text-ui-11 font-medium shadow-lg ring-1 ring-border backdrop-blur">
                    <HugeiconsIcon icon={VolumeHighIcon} className="size-3.5" />
                    Audio
                  </div>
                )}
                {/* Actions grouped in one glass toolbar so they stay legible over any clip. */}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-xl bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur">
                  <RecipePopover video={selected} onRestore={restoreSettings} active={active} />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild={true}>
                      <Button size="sm" variant="ghost" className="gap-1.5">
                        <HugeiconsIcon icon={Download01Icon} className="size-4" />
                        Download
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => void handleDownload(selectedSrc, selected, "mp4")}
                      >
                        MP4 (original{selected.has_audio ? ", keeps audio" : ""})
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => void handleDownload(selectedSrc, selected, "webm")}
                      >
                        WebM (web embeds)
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => void handleDownload(selectedSrc, selected, "gif")}
                      >
                        GIF (preview, no audio)
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label="Delete video"
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
              // The selected record's link has not landed yet -- spin in place.
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Spinner className="size-8" />
                <p className="text-sm">Loading…</p>
              </div>
            ) : busy === "generating" ? null : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                {/* Same icon as the Video nav item. */}
                <HugeiconsIcon icon={FlimSlateIcon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a video model to load"}
                </p>
              </div>
            )}

            {/* Live generation progress: a per-step bar with the phase label + ETA, centered when there is nothing else to show. */}
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
              ref={stripRef}
              className="hover-scrollbar flex shrink-0 items-stretch gap-2 overflow-x-auto border-t border-foreground/10 p-3"
              onScroll={(e) => {
                // Near the right edge: pull the next older page (infinite scroll).
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past clips stay visible and browsable while the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center rounded-[10px] bg-muted/50 ring-2 ring-primary/30">
                  <Spinner className="size-5 text-muted-foreground" />
                </div>
              )}
              {videos.map((video) => (
                <Tooltip key={video.id}>
                <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  data-clip-id={video.id}
                  onClick={() => setSelectedId(video.id)}
                  className="relative flex h-16 w-24 shrink-0 flex-col justify-end overflow-hidden rounded-[10px] bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {srcById[video.id] ? (
                    // Muted, preload="metadata" so the first frame renders as a poster without playing every card at once.
                    <video
                      src={srcById[video.id]}
                      muted
                      playsInline
                      preload="metadata"
                      onError={() => remintSrc(video)}
                      className="absolute inset-0 size-full object-cover"
                    />
                  ) : (
                    <span className="absolute inset-0 flex items-center justify-center">
                      <Spinner className="size-4 text-muted-foreground" />
                    </span>
                  )}
                  {/* A terse caption strip so cards read at a glance. Left/bottom padding clears the rounded corner and the selection border. */}
                  <span className="relative z-10 truncate bg-gradient-to-t from-black/70 to-transparent px-2 pb-1 pt-2 text-left text-ui-9 font-medium leading-none text-white">
                    {clipMeta(video)}
                  </span>
                  {/* Selection marker on a non-focusable overlay. */}
                  {video.id === selected?.id && (
                    <span className="pointer-events-none absolute inset-0 z-20 rounded-[10px] border border-border bg-white/35 dark:border-white/25 dark:bg-white/20" />
                  )}
                </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  {video.prompt}
                  <span className="mt-0.5 block opacity-70">
                    seed {video.seed} - {clipMeta(video)}
                  </span>
                </TooltipContent>
                </Tooltip>
              ))}
              {/* Tail spinner while older pages stream in on scroll. */}
              {hasMore && (
                <div className="flex size-16 shrink-0 items-center justify-center">
                  <Spinner className="size-4 text-muted-foreground" />
                </div>
              )}
              {/* Clear-all, tucked at the end so it never sits under a hover. */}
              {videos.length > 0 && (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      onClick={() => void handleClearAll()}
                      className="flex h-16 w-16 shrink-0 flex-col items-center justify-center gap-1 rounded-[10px] text-muted-foreground ring-1 ring-border transition-colors hover:text-destructive hover:ring-destructive/40"
                    >
                      <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                      <span className="text-ui-9 font-medium">Clear all</span>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>Clear all videos</TooltipContent>
                </Tooltip>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
