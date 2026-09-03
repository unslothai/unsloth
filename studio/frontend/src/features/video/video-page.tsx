// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Cancel01Icon,
  Delete02Icon,
  Download01Icon,
  FlimSlateIcon,
  ImageCropIcon,
  Image03Icon,
  InformationCircleIcon,
  PinIcon,
  VolumeHighIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { GalleryItemMenu } from "@/components/gallery-item-menu";
import { ImageDropzone } from "@/components/image-dropzone";
import { MediaPageLink } from "@/components/media-page-link";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  applyPin,
  fetchNextPage,
  fetchWhileStable,
  nextSelectedId,
  pinnedOrder,
  removeGalleryItem,
  restorePinOrder,
  serializeById,
  sortGalleryItems,
  subscribeGalleryChanged,
} from "@/lib/gallery-flags";
import { useDiffusionGpuChoices } from "@/hooks/use-gpu-info";
import { useHardwareInfo } from "@/hooks/use-hardware-info";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { usePersistedChoice } from "@/hooks/use-persisted-choice";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { VIDEO_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import type { HostClass } from "@/features/model-picker/components/model-selector/host-artifact-policy";
import {
  VIDEO_CATALOG,
  catalogToModelOptions,
  loadSpecFor,
} from "@/features/model-picker/components/model-selector/model-catalog";
import { useHostClass } from "@/hooks/use-host-class";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { ParamSlider } from "@/features/chat";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import {
  MediaGenerationPresetControl,
  type VideoGenerationPresetParams,
  closestDurationIndex,
  closestResolutionIndex,
  shouldApplyModelDefaults,
  useMediaGenerationPresets,
} from "@/features/generation-presets";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { formatBytes, formatEta } from "@/features/hub/lib/format";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useStagedDownload } from "@/features/hub/download-manager";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { resolveDiffusionGgufFilename } from "@/lib/diffusion-gguf-filename";
import { createPickGuard, runGgufRepoPick } from "@/lib/diffusion-gguf-pick";
import { diffusionRoutePick } from "@/lib/diffusion-route-pick";
import {
  PRECISION_REFUSAL_TITLE,
  denseTextEncoderBuildLabel,
  denseTransformerBuildLabel,
  formatResolvedValue,
  isPrecisionRefusal,
  resolvedBadge,
  resolvedSeedKey,
  resolvedSelectValue,
} from "@/lib/resolved-precision";
import {
  routedGgufFilename,
  routedGgufLabel,
} from "@/lib/diffusion-route-search";
import {
  downloadFile,
  downloadUrlStreaming,
  isDownloadCancelled,
} from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { subscribeModelEjected } from "@/lib/model-lifecycle-events";

import { MATCH_SOURCE_RESOLUTION, matchedCanvas } from "./keyframe-canvas";
import { hasReferenceCapacity } from "./reference-budget";
import {
  applyReferenceImageCrop,
  referenceImageDataUrls,
  stageReferenceImage,
  type StagedReferenceImage,
} from "./reference-image-crop";
import { ReferenceImageEditor } from "./reference-image-editor";
import { type ReferenceMedia, ReferenceMediaPicker } from "./reference-picker";
import {
  defaultReferenceVideoTrim,
  H3_REFERENCE_MAX_SECONDS,
  referenceVideoTrimError,
  referenceVideoTrimFeedback,
} from "./reference-trim";
import {
  type GalleryVideo,
  type VideoGenerateProgress,
  type VideoReferenceVideo,
  type VideoLoadProgress,
  type VideoLoadRequest,
  type VideoStatus,
  cancelVideoGeneration,
  clearVideoGallery,
  deleteGalleryVideo,
  setGalleryVideoFlags,
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
// Host-dependent, so it is built per render rather than once at module load: a Mac is offered
// only the GGUF rows, and an accelerated host gets the speed qualifiers.
function useVideoModels(host: HostClass): ModelOption[] {
  return useMemo(() => catalogToModelOptions(VIDEO_CATALOG, host), [host]);
}

// Per-model generation defaults (steps + guidance), matched by repo-id substring, most specific first.
const DEFAULT_GEN = { steps: 8, guidance: 1 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  { match: "minimax-h3", steps: 30, guidance: 1 },
  { match: "minimax_h3", steps: 30, guidance: 1 },
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
const FALLBACK_FRAME_OFFSET = 1;
const FALLBACK_FPS = 24;
const FALLBACK_DURATION_TARGETS = [1, 2, 3, 5];

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
  /** Clips archived locally: a terminal progress response snapshotted before the archive cannot be
   *  revoked, so the merges below must refuse it or the clip returns to the active strip. */
  archived: Set<string>;
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
  archived: new Set(),
  epoch: 0,
};

// Re-mint a cached link once it is this old, comfortably inside the backend's own expiry, so a long-lived tab keeps working.
const VIDEO_LINK_REFRESH_MS = 6 * 60 * 60 * 1000;

// Videos loaded per infinite-scroll page.
const PAGE_SIZE = 50;

// Passes a window resync may make before giving up. Each extra pass only happens when pagination
// moved while it was fetching, which cannot repeat indefinitely without the user scrolling along.
const RESYNC_MAX_ATTEMPTS = 3;

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

// MP4 streams from its signed link to the chosen path: that link is cross-origin under Tauri,
// where an anchor no longer saves, and a clip is too big to hold in memory on the way past.
// WebM / GIF are transcoded by the backend on demand (501 when the codec is absent).
async function downloadVideo(
  src: string,
  video: GalleryVideo,
  format: VideoExportFormat = "mp4",
) {
  if (format === "mp4") {
    await downloadUrlStreaming(src, exportFilename(video, format));
    return;
  }
  const blob = await fetchGalleryVideoExport(video.id, format);
  await downloadFile(blob, exportFilename(video, format), blob.type);
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// Labels for conditioned MiniMax-H3 tasks. Text-only and older clips need none.
const CONDITIONING_LABELS: Record<string, string> = {
  i2va: "From start frame",
  l2va: "To end frame",
  fl2va: "Start to end frame",
  ref2va: "From references",
};

// Keep the narrow gallery caption to duration and resolution.
function clipMeta(video: GalleryVideo): string {
  const secs = video.duration_s > 0 ? `${video.duration_s.toFixed(1)}s` : `${video.num_frames}f`;
  return `${secs} · ${video.width}×${video.height}`;
}

// Bar label for an in-flight generation, plus an ETA while denoising.
function genStepLabel(p: VideoGenerateProgress): string {
  if (p.phase === "decode") return "Decoding video and audio…";
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
// `onCancel` adds chat's Cancel action, the one control that reaches a load already in flight: the model
// selector's eject is hidden for exactly the span a first load runs (nothing is resident, so it has no
// selection to eject), which left a multi-gigabyte pull with no way out.
function loadToastArgs(
  p: VideoLoadProgress,
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

// Matches the field-label style used across Unsloth (export/chat settings).
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

// The badge for one Advanced control: "Auto: X" when the backend decided it, "NVFP4 -> OFF" in a
// warning tone when an EXPLICIT request was declined. The old rule rendered nothing for the second
// case, which is how a clip could be labelled BF16 while telemetry confirmed NVFP4 (and the other
// way round). Same markup and helpers as the images page ResolvedBadge.
function ResolvedBadge({
  status,
  controlKey,
}: {
  status: VideoStatus | null;
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

// One "what actually ran" line in the loaded-build summary below. Mirrors the images page.
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

/**
 * What the LOADED model is actually running, read from status (never from the request): the DiT and
 * text-encoder precision, the memory mode with its resolved offload behaviour, and the attention
 * backend. The Advanced selects above say what was ASKED for; this says what happened, and any
 * control whose request was declined carries its reason in the badge tooltip.
 */
function LoadedBuildSummary({ status }: { status: VideoStatus | null }) {
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
            : "Native SDPA"
        }
      />
    </div>
  );
}

/**
 * Report a failed load. A refused precision is a long actionable sentence, so it becomes a toast
 * description under a short title rather than one unreadable line. Mirrors the images page.
 */
function reportLoadFailure(message: string | null | undefined, fallback: string): void {
  const text = (message || "").trim();
  if (text && isPrecisionRefusal(text)) {
    toast.error(PRECISION_REFUSAL_TITLE, { description: text });
    return;
  }
  toast.error(text || fallback);
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

function ReferenceVideoTrimStatus({
  label,
  start,
  end,
  sourceDuration,
}: {
  label: string;
  start: number | null;
  end: number | null;
  sourceDuration?: number;
}) {
  const feedback = referenceVideoTrimFeedback(
    label,
    start,
    end,
    sourceDuration,
  );
  return (
    <p
      aria-live="polite"
      className={cn(
        "text-ui-11 leading-snug",
        feedback.invalid ? "text-destructive" : "text-muted-foreground/70",
      )}
    >
      {feedback.message}
    </p>
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
          {CONDITIONING_LABELS[video.conditioning ?? ""] ? (
            <RecipeRow
              label="Source"
              value={CONDITIONING_LABELS[video.conditioning ?? ""]}
            />
          ) : null}
          {/* The load-time build, all ENGAGED values, so a saved clip can never be labelled with a
              precision that did not run. Matches what the images Recipe already shows. */}
          {video.gguf_filename ? (
            <RecipeRow label="File" value={video.gguf_filename} mono />
          ) : null}
          {video.transformer_quant ? (
            <RecipeRow label="Quant" value={video.transformer_quant} />
          ) : null}
          {video.text_encoder_quant ? (
            <RecipeRow label="TE quant" value={video.text_encoder_quant} />
          ) : null}
          {video.memory_mode ? (
            <RecipeRow
              label="Memory"
              value={
                video.offload_policy && video.offload_policy !== "none"
                  ? `${video.memory_mode} (${video.offload_policy} offload)`
                  : video.memory_mode
              }
            />
          ) : null}
          <RecipeRow label="Size" value={`${video.width} × ${video.height}`} />
          <RecipeRow label="Frames" value={`${video.num_frames} @ ${video.fps} fps`} />
          <RecipeRow label="Duration" value={`${video.duration_s.toFixed(2)}s`} />
          <RecipeRow label="Steps" value={String(video.steps)} />
          <RecipeRow label="Guidance" value={String(video.guidance)} />
          {video.flow_shift != null ? (
            <RecipeRow
              label="Shift"
              value={
                video.audio_flow_shift != null
                  ? `${video.flow_shift} video / ${video.audio_flow_shift} audio`
                  : String(video.flow_shift)
              }
            />
          ) : null}
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
type H3Task = NonNullable<VideoLoadRequest["h3_task"]>;
type VideoLoadOptions = {
  kind: "gguf" | "single_file" | "pipeline";
  filename?: string;
  h3Task?: H3Task;
};
/** A pick held back while the user chooses the H3 partition. It carries what the deferred
 *  loadOrStage call would otherwise have been given inline, so the choice only adds `h3Task`:
 *  `source` decides whether the pick is preflighted against the Hub download plan or loaded
 *  straight off disk, and a dialog must not silently change that for the pick it is holding. */
type PendingH3Load = {
  repoId: string;
  opts: VideoLoadOptions;
  source: ModelSelectorChangeMeta["source"];
  token: number;
};

const H3_BF16_REPO = "MiniMaxAI/MiniMax-H3";

/** Whether a pick is the H3 base pipeline, whose denoiser partition the user must choose.
 *  Shared by both entry points: a chat-picker pick arrives as ?model= and reaches loadOrStage
 *  without passing through handleModelSelect, so checking it in one place staged the default
 *  fl2va partition, tens of GB, with no way to ask for References.
 *
 *  An on-device copy counts. The same pipeline added as a directory reaches the generic
 *  local-pipeline branch, which a Hub-id equality test never recognises, so an omitted h3_task
 *  pinned it to fl2va and its transformer_ref partition was unreachable even with the weights
 *  sitting on disk. Matched on the final path segment, the same way a local checkpoint's family
 *  is read off its filename elsewhere. */
function isH3PipelinePick(repoId: string, kind: VideoLoadOptions["kind"]): boolean {
  if (kind !== "pipeline") return false;
  const id = repoId.toLowerCase();
  if (id === H3_BF16_REPO.toLowerCase()) return true;
  const leaf = id.replace(/\\/g, "/").replace(/\/+$/, "").split("/").at(-1) ?? "";
  return leaf === H3_BF16_REPO.split("/")[1].toLowerCase();
}

// What a pick optimistically replaced, so a load that never takes can put all of it back. The quant
// label and the generation recipe move together at pick time, so they have to roll back together too.
type PickRevert = {
  prev: string | null;
  steps: number;
  guidance: number;
  commitRecipeClaim?: () => void;
  releaseRecipeClaim?: () => void;
  // What the pick applied. A field the user changed after that is theirs, not ours to put back.
  appliedSteps?: number;
  appliedGuidance?: number;
  modelSeeded?: boolean;
  familySeeded?: boolean;
};
// Resolved Advanced controls pinned across preflight, staging, and load.
type VideoLoadAdvanced = Pick<
  VideoLoadRequest,
  | "memory_mode"
  | "speed_mode"
  | "attention_backend"
  | "transformer_cache"
  | "transformer_quant"
  | "gpu_ids"
>;

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
 * The root guard never bounces /video: a chat-only host is both where the explanation below has
 * something to say (a CPU-only box) and where video works anyway (Apple Silicon whose only
 * problem is MLX). So the page answers for itself: spin while the answer is out, explain a no.
 *
 * That answer is /api/system/hardware's alone, which settles detection before replying. Waiting
 * on the chat-only verdict too would spin through an MLX self-heal /api/health holds it back for,
 * which cannot change a Metal answer.
 */
export function VideoPage({
  active = true,
  onInitialReady,
}: {
  active?: boolean;
  onInitialReady?: () => void;
}) {
  const hardware = useHardwareInfo();

  useEffect(() => {
    if (active && hardware.loaded && hardware.videoSupported === false) {
      onInitialReady?.();
    }
  }, [active, hardware.loaded, hardware.videoSupported, onInitialReady]);

  if (!hardware.loaded) {
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

  return (
    <VideoGenerator active={active} onInitialReady={onInitialReady} />
  );
}

function VideoGenerator({
  active = true,
  onInitialReady,
}: {
  active?: boolean;
  onInitialReady?: () => void;
}) {
  const initialReadySent = useRef(false);
  const hostClass = useHostClass();
  const videoModels = useVideoModels(hostClass);
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(
    "Ultra-realistic cinematic documentary footage of a quiet Kyoto neighborhood at sunrise. An elderly Japanese man opens his traditional wooden shop while a young woman wearing a simple kimono walks past carrying a small basket. Cherry blossom petals gently fall through the air, bicycles pass by, warm sunlight enters between narrow streets, distant temple bells echo. The camera slowly moves forward like a professional travel documentary, realistic human movements, natural expressions, authentic Japanese architecture, subtle wind movement in clothing and trees, realistic colors, 35mm film photography style.",
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [negativeOpen, setNegativeOpen] = useState(false);
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const modelSeeded = useRef(false);
  const familySeeded = useRef(false);
  // Whether the user has taken the recipe since the pick that is still waiting for its status.
  // Both the seeding effects and the rollback below ask: a preset selected while the model
  // downloaded is newer than that pick, so neither its defaults nor its rollback may land on top.
  const pickRecipeSuperseded = useRef<(() => boolean) | null>(null);
  // Put back everything a pick optimistically applied. Setters are stable, so this never re-renders on its own.
  const revertPick = useCallback((r: PickRevert) => {
    setQuant(r.prev);
    setPendingModelDefaults(null);
    // Equality alone cannot tell "nobody touched this" from "the user chose the same number": a
    // preset selected after the pick owns these fields even where it matches what the pick applied.
    if (!pickRecipeSuperseded.current?.()) {
      setSteps((cur) => (cur === r.appliedSteps ? r.steps : cur));
      setGuidance((cur) => (cur === r.appliedGuidance ? r.guidance : cur));
    }
    if (r.modelSeeded != null) modelSeeded.current = r.modelSeeded;
    if (r.familySeeded != null) familySeeded.current = r.familySeeded;
    pickRecipeSuperseded.current = null;
    r.releaseRecipeClaim?.();
    r.releaseRecipeClaim = undefined;
  }, []);
  // The recipe a pick optimistically claimed, until status confirms it or a failed load reverts
  // it. Without this the Default preset would read as "modified" for the whole download.
  const [pendingModelDefaults, setPendingModelDefaults] = useState<{
    steps: number;
    guidance: number;
  } | null>(null);
  const [seed, setSeed] = useState("");
  // Preset index, or MATCH_SOURCE_RESOLUTION for a keyframe-derived canvas.
  const [resolutionIdx, setResolutionIdx] = useState(0);
  const [resolutionIntent, setResolutionIntent] = useState<[number, number]>(
    FALLBACK_RESOLUTION_PRESETS[0]!,
  );
  // MiniMax-H3 keyframes as data URLs: the frame the clip starts from, the frame it ends on, or both.
  const [firstFrame, setFirstFrame] = useState<string | null>(null);
  const [lastFrame, setLastFrame] = useState<string | null>(null);
  // Natural pixel size of whichever keyframe drives the canvas, for the "match source" preview.
  const [keyframeAspect, setKeyframeAspect] = useState<[number, number] | null>(null);
  // Separate lists preserve Ref2VA's image, video, then audio request order.
  const [referenceImages, setReferenceImages] = useState<StagedReferenceImage[]>([]);
  const [cropPictureIndex, setCropPictureIndex] = useState<number | null>(null);
  const [referenceVideos, setReferenceVideos] = useState<
    Array<{
      video: ReferenceMedia;
      audio: ReferenceMedia | null;
      trimStartSeconds: number | null;
      trimEndSeconds: number | null;
    }>
  >([]);
  const [referenceAudios, setReferenceAudios] = useState<ReferenceMedia[]>([]);
  const [referenceImageSize, setReferenceImageSize] = useState<"match" | "max">("match");
  // Null until the loaded family provides released schedule shifts.
  const [flowShift, setFlowShift] = useState<number | null>(null);
  const [audioFlowShift, setAudioFlowShift] = useState<number | null>(null);
  // The chosen frame count must lie on the family's temporal lattice.
  const [numFrames, setNumFrames] = useState(
    FALLBACK_FRAME_STEP * 3 + FALLBACK_FRAME_OFFSET,
  );
  const [durationIntentSeconds, setDurationIntentSeconds] = useState(
    numFrames / FALLBACK_FPS,
  );
  // Advanced options live in a right-docked panel, closed by default; a single fixed top-bar toggle opens it.
  // Sits inline under Seed; the open state is remembered across visits.
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_video_advanced_open",
  );
  // Advanced (load-time) options; "auto"/"off" map to the backend defaults. "Reapply" reloads with new values.
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  // "auto", or the physical index to pin this load to. Only offered on a multi-card CUDA / ROCm host.
  // Persisted, unlike the selects around it: those are reseeded from the loaded build, and the
  // status carries the device a pipeline is on but not which card, so a refresh would reset this
  // one to Auto while the model stayed put and the next Reapply would move it to the default GPU.
  // A stored id is only a hint; the send path below still drops one whose card is no longer there.
  const [selectedGpu, setSelectedGpu] = usePersistedChoice(
    "unsloth_video_gpu_choice",
    "auto",
  );
  const gpuChoices = useDiffusionGpuChoices();
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [attentionBackend, setAttentionBackend] = useState<
    "auto" | "native" | "cudnn" | "flash3" | "sage"
  >("auto");
  const [transformerCache, setTransformerCache] = useState<"auto" | "off" | "fbcache">("auto");
  const [transformerQuant, setTransformerQuant] = useState<
    "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8"
  >("auto");
  // The last load descriptor, so "Reapply" can reload the same model with new advanced options.
  const lastLoad = useRef<({ repoId: string } & VideoLoadOptions) | null>(null);
  // Render-safe mirror of whether a page-initiated load supplied a complete Reapply target.
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
  const [pendingH3Load, setPendingH3Load] = useState<PendingH3Load | null>(null);
  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  // Records come from the backend (durable); srcById maps each id to its object URL.
  const [videos, setVideos] = useState<GalleryVideo[]>(() => galleryCache.videos);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [clearingGallery, setClearingGallery] = useState(false);
  // The `active` gate below only HIDES the confirm, and Radix does not call onOpenChange for
  // a parent-forced close, so on this persistently mounted page the state would outlive the
  // route change and the confirm would be back on return. Reset it during render.
  if (!active && clearConfirmOpen) setClearConfirmOpen(false);
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
  // A pick also applies its own step/guidance recipe, so the rollback carries those too: a cancelled distilled
  // pick otherwise leaves its low-step, guidance-0 recipe applied to the model that is still resident.
  const quantRevert = useRef<PickRevert | null>(null);
  // Which quantRevert entry the live staged download belongs to. Staging does not set `busy`, so a second pick can overwrite
  // quantRevert while the first plan is still resolving; without this the dying first job reverts the newer pick's label.
  const stagedQuantRevert = useRef<PickRevert | null>(null);
  // Bumped per Hub pick, so a plan that resolves after a newer pick can tell it has been superseded.
  const pickSeq = useRef(0);
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

  // The load toast is built by handleLoad and the progress poll, both of which are defined above
  // handleCancelLoad (it needs handleUnload, which needs them). Route the action through a ref so
  // the toast keeps a stable onClick instead of dragging the whole load graph into its deps.
  const cancelLoadRef = useRef<() => void>(() => {});
  const cancelLoadFromToast = useCallback(() => cancelLoadRef.current(), []);
  // Bumped by every cancel / eject (see dropResidentState). Requests that were already awaiting a
  // response when the cancel landed compare against it and discard their own result.
  const cancelSeq = useRef(0);
  // Bumped by every load start. The compensating unload below carries no identity, so it must not
  // fire once a newer load owns the page -- it would tear that one down instead.
  const loadSeq = useRef(0);
  // The load currently in flight, if any, as a promise that settles only once handleLoad has run
  // to the end -- including the compensating unload it may issue. A cancel that lands before the
  // backend has registered the load has to wait for all of it: begin_load REFUSES a second load
  // while one is live ("a load is already in progress"), so a model picked in that window would be
  // rejected while the cancelled one kept going, and the compensating unload names no load, so one
  // still in flight would tear down whatever the user picked next. Holding busy shuts both.
  const pendingStart = useRef<Promise<unknown> | null>(null);

  // Set by restoreLoadTracking: handleLoad's compensating unload failed, so the load it was
  // cancelling is STILL running and its toast and poll are back up. handleUnload reads it to
  // report that the eject stopped nothing, rather than claiming success over a live load.
  const loadTrackingRestored = useRef(false);

  // Client-side state that only means anything while a model is resident: the
  // in-flight replacement load's tracking, and the Reapply target. Shared with
  // the indicator eject, which frees the runtime without going through the
  // page's own Unload.
  const dropResidentState = useCallback(() => {
    // Cancel, not release: a resolving pick or a staged download would load
    // back what was just ejected. In here rather than only in handleUnload, so
    // an eject driven from the loaded models card is covered by it too.
    pickGuard.cancel();
    // Everything already in flight is now stale. Clearing the timer below stops the NEXT poll
    // tick, but not a poll or a start request currently awaiting its response, and those still
    // apply terminal state when they land. The counter is what they compare against.
    cancelSeq.current += 1;
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    // Leaving this set would let Reapply reload the model that was just freed.
    lastLoad.current = null;
    setCanReapply(false);
    // Stopping the poll above also stops its "the load was cancelled or evicted" branch, which is
    // what hands back a pick that never became resident. Do it here, exactly as that branch would:
    // an unreleased recipe claim leaves hydration parked behind a load that is never coming, and a
    // rollback left behind is one a later pick would inherit in place of its own.
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [dismissLoadToast, pickGuard, revertPick]);

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
  const frameOffset = status?.defaults?.frame_offset ?? FALLBACK_FRAME_OFFSET;
  const fps = status?.defaults?.fps ?? FALLBACK_FPS;
  const durationTargets =
    status?.defaults?.duration_presets ?? FALLBACK_DURATION_TARGETS;

  // Duration presets: valid frame counts closest to ~1s/2s/3s/5s at the current fps, deduped.
  const durationOptions = useMemo<Array<{ frames: number; seconds: number }>>(() => {
    const seen = new Set<number>();
    const out: Array<{ frames: number; seconds: number }> = [];
    for (const t of durationTargets) {
      const desired = t * fps;
      const k = Math.max(1, Math.round((desired - frameOffset) / frameStep));
      const frames = k * frameStep + frameOffset;
      if (seen.has(frames)) continue;
      seen.add(frames);
      out.push({ frames, seconds: frames / fps });
    }
    return out;
  }, [frameStep, frameOffset, fps, durationTargets]);

  // Keep the resolution / frame-count selections valid when the loaded family changes.
  useEffect(() => {
    setResolutionIdx((idx) =>
      idx === MATCH_SOURCE_RESOLUTION || idx < resolutionPresets.length ? idx : 0,
    );
  }, [resolutionPresets.length]);

  // ── keyframes ──────────────────────────────────────────────────────────────
  const supportsKeyframes = status?.supports_keyframes === true;
  // The keyframe the canvas follows: the first when there is one, else the last, matching the backend.
  const canvasKeyframe = firstFrame ?? lastFrame;

  const supportsReferences = status?.supports_references === true;
  const hasReferenceRoom = hasReferenceCapacity(
    referenceImages.length,
    referenceVideos.length,
    referenceAudios.length,
  );
  // Only Diffusers supports the 2048px reference policy.
  const canPickReferenceSize = supportsReferences && status?.engine !== "sd_cpp";

  // Drop conditioning that the newly loaded partition cannot accept.
  useEffect(() => {
    if (status?.loaded && !supportsKeyframes) {
      setFirstFrame(null);
      setLastFrame(null);
    }
  }, [status?.loaded, supportsKeyframes]);
  useEffect(() => {
    if (status?.loaded && !supportsReferences) {
      setReferenceImages([]);
      setReferenceVideos([]);
      setReferenceAudios([]);
      setCropPictureIndex(null);
    }
  }, [status?.loaded, supportsReferences]);
  useEffect(() => {
    if (!canPickReferenceSize) setReferenceImageSize("match");
  }, [canPickReferenceSize]);

  // Measure the keyframe that drives the canvas preview.
  useEffect(() => {
    if (!canvasKeyframe) {
      setKeyframeAspect(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled) setKeyframeAspect([img.naturalWidth, img.naturalHeight]);
    };
    img.onerror = () => {
      if (!cancelled) setKeyframeAspect(null);
    };
    img.src = canvasKeyframe;
    return () => {
      cancelled = true;
    };
  }, [canvasKeyframe]);

  // Resolved "match source" canvas, when valid.
  const matchedResolution = useMemo(
    () =>
      keyframeAspect
        ? matchedCanvas(keyframeAspect[0], keyframeAspect[1], status?.defaults)
        : null,
    [keyframeAspect, status?.defaults],
  );

  const videoPresetParams = useMemo<VideoGenerationPresetParams>(() => {
    const resolution =
      resolutionIdx === MATCH_SOURCE_RESOLUTION && matchedResolution
        ? matchedResolution
        : resolutionIntent;
    return {
      negativePrompt,
      width: resolution[0],
      height: resolution[1],
      durationSeconds: durationIntentSeconds,
      steps,
      guidance,
      flowShift,
      audioFlowShift,
    };
  }, [audioFlowShift, durationIntentSeconds, flowShift, guidance, matchedResolution, negativePrompt, resolutionIdx, resolutionIntent, steps]);
  const defaultSteps = status?.defaults?.steps;
  const defaultGuidance = status?.defaults?.guidance;
  const familyDefaultFrames = status?.defaults?.num_frames;
  const defaultFlowShift = status?.defaults?.flow_shift ?? null;
  const defaultAudioFlowShift = status?.defaults?.audio_flow_shift ?? null;
  const presetRepoId = status?.repo_id ?? "";
  const videoDefaultRecipe = useMemo<VideoGenerationPresetParams>(() => {
    const resolution = resolutionPresets[0] ?? [768, 512];
    const recommended =
      pendingModelDefaults ??
      (defaultSteps != null && defaultGuidance != null
        ? { steps: defaultSteps, guidance: defaultGuidance }
        : defaultsFor(presetRepoId));
    const defaultDuration = familyDefaultFrames
      ? durationOptions[
          closestDurationIndex(durationOptions, familyDefaultFrames / fps)
        ]?.seconds
      : status?.loaded
        ? durationOptions[2]?.seconds ?? durationOptions[0]?.seconds ?? 3
        : durationOptions[0]?.seconds ?? 1;
    return {
      negativePrompt: "",
      width: resolution[0],
      height: resolution[1],
      durationSeconds: defaultDuration,
      steps: recommended.steps,
      guidance: recommended.guidance,
      flowShift: defaultFlowShift,
      audioFlowShift: defaultAudioFlowShift,
    };
  }, [defaultAudioFlowShift, defaultFlowShift, defaultGuidance, defaultSteps, durationOptions, familyDefaultFrames, fps, pendingModelDefaults, presetRepoId, resolutionPresets, status?.loaded]);
  const applyVideoPresetParams = useCallback(
    (params: VideoGenerationPresetParams) => {
      const resolutionIndex = closestResolutionIndex(
        resolutionPresets,
        params.width,
        params.height,
      );
      const durationIndex = closestDurationIndex(durationOptions, params.durationSeconds);
      const durationFrames =
        durationOptions[durationIndex]?.frames ?? durationOptions[0]?.frames ?? numFrames;
      setResolutionIntent([params.width, params.height]);
      setDurationIntentSeconds(params.durationSeconds);
      setNegativePrompt(params.negativePrompt);
      // Same rule restoreSettings follows: a negative prompt that is in effect has to be visible,
      // or the user generates against a setting the collapsed field is hiding.
      if (params.negativePrompt) setNegativeOpen(true);
      setResolutionIdx(resolutionIndex);
      setNumFrames(durationFrames);
      setSteps(params.steps);
      setGuidance(params.guidance);
      setFlowShift(params.flowShift);
      setAudioFlowShift(params.audioFlowShift);
      return params;
    },
    [durationOptions, numFrames, resolutionPresets],
  );
  const normalizeVideoPresetParams = useCallback(
    (params: VideoGenerationPresetParams) => {
      const resolution =
        resolutionPresets[
          closestResolutionIndex(resolutionPresets, params.width, params.height)
        ] ?? resolutionPresets[0] ?? [768, 512];
      const duration =
        durationOptions[
          closestDurationIndex(durationOptions, params.durationSeconds)
        ]?.seconds ?? params.durationSeconds;
      return {
        ...params,
        width: resolution[0],
        height: resolution[1],
        durationSeconds: duration,
      };
    },
    [durationOptions, resolutionPresets],
  );
  const videoPresets = useMediaGenerationPresets({
    kind: "video",
    defaultParams: videoDefaultRecipe,
    currentParams: videoPresetParams,
    applyParams: applyVideoPresetParams,
    normalizeParams: normalizeVideoPresetParams,
  });
  const claimVideoRecipe = videoPresets.claimRecipe;
  const videoFormClaimId = videoPresets.formClaimId;
  const applyVideoModelDefaults = useCallback(
    (repoId: string) => {
      const revert = quantRevert.current;
      if (revert && !revert.releaseRecipeClaim) {
        const claim = claimVideoRecipe();
        revert.commitRecipeClaim = claim.commit;
        revert.releaseRecipeClaim = claim.release;
      }
      // Baselined per pick, including a pick that inherits an earlier one's rollback: the question
      // is whether the user takes the form after THIS pick, not after the one it replaced.
      const claimedAt = videoFormClaimId();
      pickRecipeSuperseded.current = () => videoFormClaimId() !== claimedAt;
      const recommended = defaultsFor(repoId);
      setPendingModelDefaults(recommended);
      setSteps(recommended.steps);
      setGuidance(recommended.guidance);
      if (revert) {
        revert.modelSeeded ??= modelSeeded.current;
        revert.familySeeded ??= familySeeded.current;
        revert.appliedSteps = recommended.steps;
        revert.appliedGuidance = recommended.guidance;
      }
      // This explicit pick owns the first status confirmation. On failure, revertPick restores
      // both markers so a previously saved recipe can still outrank a merely discovered resident.
      modelSeeded.current = true;
      familySeeded.current = true;
    },
    [claimVideoRecipe, videoFormClaimId],
  );

  useEffect(() => {
    setResolutionIdx((current) => {
      if (current === MATCH_SOURCE_RESOLUTION) return current;
      return closestResolutionIndex(
        resolutionPresets,
        resolutionIntent[0],
        resolutionIntent[1],
      );
    });
  }, [resolutionIntent, resolutionPresets]);

  // Select "match source" only after the staged keyframe passes the aspect-ratio check.
  const hadKeyframeRef = useRef(false);
  useEffect(() => {
    const has = canvasKeyframe != null;
    if (has === hadKeyframeRef.current) return;
    if (has && !matchedResolution) return;
    hadKeyframeRef.current = has;
    setResolutionIdx((idx) => {
      if (has) return MATCH_SOURCE_RESOLUTION;
      return idx === MATCH_SOURCE_RESOLUTION
        ? closestResolutionIndex(
            resolutionPresets,
            resolutionIntent[0],
            resolutionIntent[1],
          )
        : idx;
    });
  }, [canvasKeyframe, matchedResolution, resolutionIntent, resolutionPresets]);

  const loadedFamily = status?.loaded ? status.family : null;
  const prevFamilyRef = useRef<string | null>(null);
  useEffect(() => {
    const familyChanged = loadedFamily !== prevFamilyRef.current;
    prevFamilyRef.current = loadedFamily;
    // A newly loaded family brings its own default clip length; without this the pre-load fallback
    // sticks and every default run is a ~1s clip. Intent moves with it, so the recipe a preset is
    // compared against and the frame count generation sends never disagree.
    const applyFamilyDefault = shouldApplyModelDefaults(
      familySeeded.current,
      videoPresets.storedRecipe,
      pickRecipeSuperseded.current?.() ?? false,
    );
    if (familyChanged && loadedFamily) familySeeded.current = true;
    if (familyChanged && loadedFamily && familyDefaultFrames && applyFamilyDefault) {
      const option =
        durationOptions[
          closestDurationIndex(durationOptions, familyDefaultFrames / fps)
        ];
      if (option) {
        setDurationIntentSeconds(option.seconds);
        setNumFrames(option.frames);
        return;
      }
    }
    setNumFrames((cur) => {
      if (!familyChanged && durationOptions.some((o) => o.frames === cur)) return cur;
      return (
        durationOptions[
          closestDurationIndex(durationOptions, durationIntentSeconds)
        ]?.frames ?? cur
      );
    });
  }, [
    durationIntentSeconds,
    durationOptions,
    familyDefaultFrames,
    fps,
    loadedFamily,
    videoPresets.storedRecipe,
  ]);

  // Seed steps/guidance from the loaded model's backend defaults: on mount with a model already loaded only refreshStatus runs, so the
  // controls would stick at the pre-load DEFAULT_GEN and a base checkpoint wanting 40/4 generates a degraded clip. Keyed on the resolved
  // schedule, not the repo alone: a GGUF repo holds several variants, so another client swapping builds changes the defaults in place.
  const loadedModelKey = status?.loaded
    ? `${status.repo_id ?? ""}|${defaultSteps ?? ""}|${defaultGuidance ?? ""}|${defaultFlowShift ?? ""}|${defaultAudioFlowShift ?? ""}`
    : null;
  const prevLoadedModelRef = useRef<string | null>(null);
  useEffect(() => {
    const modelChanged = loadedModelKey !== prevLoadedModelRef.current;
    prevLoadedModelRef.current = loadedModelKey;
    if (modelChanged && loadedModelKey && defaultSteps != null && defaultGuidance != null) {
      // Status is the authority now, so the recipe a pick claimed has served its purpose.
      setPendingModelDefaults(null);
      // A stored recipe is the user's own choice, so it outranks the model's defaults on the first
      // seed. Every later model change still seeds, as picking a model always has.
      const applyDefaults = shouldApplyModelDefaults(
        modelSeeded.current,
        videoPresets.storedRecipe,
        pickRecipeSuperseded.current?.() ?? false,
      );
      // This status IS the pending pick's confirmation, so the question is answered for good. Read
      // after the family effect above, which runs first on the same status and asks the same thing.
      pickRecipeSuperseded.current = null;
      modelSeeded.current = true;
      if (!applyDefaults) return;
      setSteps(defaultSteps);
      setGuidance(defaultGuidance);
      setFlowShift(defaultFlowShift);
      setAudioFlowShift(defaultAudioFlowShift);
    }
  }, [
    defaultAudioFlowShift,
    defaultFlowShift,
    defaultGuidance,
    defaultSteps,
    loadedModelKey,
    videoPresets.storedRecipe,
  ]);

  const canPickAudioFlowShift = status?.defaults?.supports_audio_flow_shift === true;

  // Reseed the Advanced selects from the LOADED build, so they stop being pure local request state.
  // An honored request re-selects itself; a declined one snaps to what actually engaged, so the
  // Precision dropdown can never go on advertising a scheme the loaded DiT is not running. Keyed on
  // the LOAD-TIME half of the record: the backend rewrites the transformer_cache entry at GENERATION
  // time, so serializing the whole record let a step-cache toggle discard a pending Advanced edit.
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

  // Bumped by every LOCAL change to the strip: a pin, an archive, a delete, a merged generation.
  // A resync started before one of those holds a snapshot the server listing cannot reconcile with
  // what the user just did, so it drops it rather than applying a window they have moved off.
  const stripEpoch = useRef(0);
  // Bumped by the window growing from the server instead: a load, an appended page. That is not a
  // conflict, it just means the resync sized itself against a smaller window, so it refetches.
  const pageEpoch = useRef(0);
  // Only the most recently started resync may apply. Two restores in a row start two of them, and
  // the older snapshot arriving last would drop whatever the newer one had already shown.
  const resyncSeq = useRef(0);
  // Shelf mutations in flight. The epoch is an EDGE, so a page starting after the bump and landing
  // before the row is dropped sees it hold still. A page is only trusted while this is zero.
  const pendingShelfMutations = useRef(0);

  const loadGallery = useCallback(async () => {
    try {
      // Fenced: this page renders from the module cache while the load runs, so its tiles are
      // actionable, and a pre-pin snapshot would undo the action with nothing to correct it.
      const page = await fetchWhileStable(
        () => stripEpoch.current,
        () => getVideoGallery(0, PAGE_SIZE),
      );
      if (!page) return;
      pageEpoch.current += 1;
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
      // Guarded on all three counters: an archive landing anywhere across this GET shortens the
      // shelf, and the clip that shifts over the page boundary is returned by no page at all.
      const result = await fetchNextPage(
        () => galleryCache.videos.length,
        () => stripEpoch.current,
        () => pendingShelfMutations.current,
        (offset) => getVideoGallery(offset, PAGE_SIZE),
      );
      if (!result) return;
      const page = result.page;
      pageEpoch.current += 1;
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

  // WebM/GIF go through a server-side transcode that can take seconds (and 501s when the codec is missing), so wrap the helper with toasts.
  const handleDownload = useCallback(
    async (src: string, video: GalleryVideo, format: "mp4" | "webm" | "gif") => {
      const toastId =
        format === "mp4" ? null : toast.loading(`Converting to ${format.toUpperCase()}…`);
      try {
        await downloadVideo(src, video, format);
        if (toastId !== null) toast.dismiss(toastId);
        if (isTauri) {
          toast.success("Video saved", { description: exportFilename(video, format) });
        }
      } catch (err) {
        if (toastId !== null) toast.dismiss(toastId);
        if (isDownloadCancelled(err)) return;
        toast.error("Could not save video", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
    },
    [],
  );

  // Drop a clip from the strip. `discardLink` is for a real delete: the bytes are gone, so the
  // cached link must go and any mint in flight must throw its result away. An archived clip keeps
  // both, since the archived view plays the same file.
  const dropFromStrip = useCallback((id: string, discardLink: boolean) => {
    if (discardLink) {
      galleryCache.srcById.delete(id);
      galleryCache.refreshed.delete(id);
      galleryCache.deleted.add(id);
      setSrcById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
    stripEpoch.current += 1;
    // Read the list from the cache (kept in sync with state every render) rather than nesting a
    // setSelectedId inside a setVideos updater, which would run a side effect during dispatch.
    const at = galleryCache.videos.findIndex((v) => v.id === id);
    const next = removeGalleryItem(galleryCache.videos, id);
    galleryCache.videos = next;
    setVideos(next);
    setSelectedId((cur) => nextSelectedId(next, id, cur, at));
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      // Held for the whole round trip: the server shortens the shelf when it processes this, and a
      // page read inside that window sees the shortened list at an offset nothing contradicts.
      stripEpoch.current += 1;
      pendingShelfMutations.current += 1;
      try {
        await deleteGalleryVideo(id);
      } catch (err) {
        pendingShelfMutations.current -= 1;
        toast.error(err instanceof Error ? err.message : "Failed to delete video");
        return;
      }
      dropFromStrip(id, true);
      pendingShelfMutations.current -= 1;
    },
    [dropFromStrip],
  );

  /**
   * Refetch the loaded window from offset 0.
   *
   * Unpinning can drop a clip past the end of the loaded window and promote a previously unloaded
   * one into it. The local reorder cannot know about the promoted clip, and the next loadMore
   * still pages from the unchanged length, so that clip would be skipped until a reload.
   */
  const resyncWindow = useCallback(
    async (count: number, stillFresh?: () => boolean) => {
      const ticket = (resyncSeq.current += 1);
      for (let attempt = 0; attempt < RESYNC_MAX_ATTEMPTS; attempt += 1) {
        const paged = pageEpoch.current;
        // Sized against the live window, so a page appended while this ran is covered rather than
        // cut back off the bottom of the strip when the snapshot lands.
        const wanted = Math.max(count, galleryCache.videos.length, PAGE_SIZE);
        const collected: GalleryVideo[] = [];
        let more = false;
        while (collected.length < wanted) {
          // The REMAINDER, not a whole page: a window of 51 (a page plus one new generation) would
          // otherwise ask for 100 and grow the strip to match, reading 49 recipes off disk and
          // rendering their tiles for a one-row shortfall.
          const page = await getVideoGallery(
            collected.length,
            Math.min(PAGE_SIZE, wanted - collected.length),
          );
          collected.push(...page.videos);
          more = page.has_more;
          if (!page.has_more || page.videos.length === 0) break;
        }
        // Checked here, not by the caller: by the time this returns the window is already applied,
        // so a stale snapshot has to be dropped before it overwrites a newer local change.
        if (stillFresh && !stillFresh()) return;
        if (resyncSeq.current !== ticket) return;
        // Pagination moved under this pass. That is only server data, so cover it with another
        // pass instead of giving up: giving up is what left an unpin's promoted clip missing.
        if (pageEpoch.current !== paged) continue;
        galleryCache.videos = collected;
        galleryCache.hasMore = more;
        setVideos(collected);
        setHasMore(more);
        if (typeof IntersectionObserver === "undefined") {
          collected.forEach((video) => void ensureSrc(video));
        }
        return;
      }
    },
    [ensureSrc],
  );

  // This page stays mounted across route changes, so a restore from the Settings archive would
  // otherwise not reach the strip until a full reload. Resync the window that is actually loaded:
  // loadGallery would cut it back to the first page and throw away everything scrolled to.
  useEffect(
    () =>
      subscribeGalleryChanged("videos", () => {
        // Bumped FIRST: a restore changes the shelf, so reads already in flight must be discarded.
        // Capturing without advancing let them pass their own checks and land on the new window.
        stripEpoch.current += 1;
        // Fenced like the unpin resync: a generation or a new page landing while this GET runs
        // would otherwise be overwritten by a snapshot taken before it.
        const epoch = stripEpoch.current;
        void resyncWindow(
          galleryCache.videos.length,
          () => stripEpoch.current === epoch,
        ).catch(() => void loadGallery());
      }),
    [loadGallery, resyncWindow],
  );

  // The pin state each id was last CLICKED into, so a failing request can tell whether it is still
  // the current intent. Without it, a slow first click failing after a later click succeeded would
  // roll the strip back onto the state the user has since moved off.
  const pinAttempt = useRef(new Map<string, number>());
  const pinSeq = useRef(0);

  const handleTogglePin = useCallback(
    async (id: string, pinned: boolean) => {
      const loadedCount = galleryCache.videos.length;
      // The pinned order as it stands BEFORE the click, so a failed unpin can put the clip back
      // where it was instead of at the front of the pins.
      const orderBefore = pinnedOrder(galleryCache.videos);
      // A per-attempt token, not the target boolean: pin, unpin, pin stores true twice, so the FIRST
      // attempt's failure would roll back the THIRD attempt's pin and leave the two disagreeing.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      const epoch = stripEpoch.current;
      // Optimistic: the reorder should land on the click, not a round trip later.
      setVideos((prev) => {
        const next = applyPin(prev, id, pinned);
        galleryCache.videos = next;
        return next;
      });
      try {
        // One queue for the whole gallery, not one per clip. The server stamps `pinned_at` when
        // it runs the PATCH and orders pins by that stamp, so two requests in flight together can
        // be stamped in either order and the strip disagrees with the next load. Issuing them one
        // at a time makes the stamps follow the clicks.
        await serializeById("video-pin", () => setGalleryVideoFlags(id, { pinned }));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to pin video");
        // Put the old order back rather than leave the strip lying about server state, but only
        // while this is still what the user last asked for.
        if (pinAttempt.current.get(id) === attempt) {
          pinAttempt.current.delete(id);
          stripEpoch.current += 1;
          setVideos((prev) => {
            // A failed pin simply goes back to unpinned; a failed unpin has to be restored to its
            // old position among the pins, which applyPin cannot do (it means "freshly pinned").
            const next = pinned
              ? applyPin(prev, id, false)
              : restorePinOrder(prev, id, orderBefore);
            galleryCache.videos = next;
            return next;
          });
        }
        return;
      }
      if (pinAttempt.current.get(id) !== attempt) return; // superseded by a later click
      pinAttempt.current.delete(id);
      // Pinning keeps the same set in the window (it only moves an already-loaded clip to the
      // front), so only unpinning can open a gap.
      if (!pinned && loadedCount > 0) {
        try {
          // Fenced: a pin clicked while this GET is in flight would otherwise be overwritten by a
          // snapshot taken before it, leaving the strip unpinned while the server is pinned.
          await resyncWindow(loadedCount, () => stripEpoch.current === epoch);
        } catch {
          // Best-effort: the strip is still usable, just possibly short one clip until a reload.
        }
      }
    },
    [resyncWindow],
  );

  const handleArchive = useCallback(
    async (id: string) => {
      // Held for the whole round trip: the server shortens the shelf when it processes this, and a
      // page read inside that window sees the shortened list at an offset nothing contradicts.
      stripEpoch.current += 1;
      pendingShelfMutations.current += 1;
      try {
        await setGalleryVideoFlags(id, { archived: true });
      } catch (err) {
        pendingShelfMutations.current -= 1;
        toast.error(err instanceof Error ? err.message : "Failed to archive video");
        return;
      }
      galleryCache.archived.add(id);
      dropFromStrip(id, false);
      pendingShelfMutations.current -= 1;
      const toastId = toast(
        <button
          type="button"
          onClick={() => {
            toast.dismiss(toastId);
            useSettingsDialogStore.getState().openArchivedMedia("videos");
          }}
          className="w-full cursor-pointer text-left"
        >
          You can view archived videos in Settings
        </button>,
        { closeButton: true },
      );
    },
    [dropFromStrip],
  );

  const handleClearAll = useCallback(async () => {
    setClearingGallery(true);
    try {
      await clearVideoGallery();
      galleryCache.srcById.clear();
      galleryCache.refreshed.clear();
      // Every mint in flight now belongs to a cleared gallery, so their links are discarded on arrival. The epoch covers unlisted ids too.
      galleryCache.epoch += 1;
      stripEpoch.current += 1;
      galleryCache.videos = [];
      galleryCache.hasMore = false;
      galleryCache.selectedId = null;
      setSrcById({});
      setVideos([]);
      setHasMore(false);
      setSelectedId(null);
      setClearConfirmOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to clear gallery");
    } finally {
      setClearingGallery(false);
    }
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
      if (video.flow_shift != null) setFlowShift(video.flow_shift);
      if (video.audio_flow_shift != null) setAudioFlowShift(video.audio_flow_shift);
      setSeed(String(video.seed));
      // Snap the resolution to the matching preset when one exists; else leave as is.
      const presetIdx = resolutionPresets.findIndex(
        ([w, h]) => w === video.width && h === video.height,
      );
      if (presetIdx >= 0) {
        setResolutionIntent([video.width, video.height]);
        setResolutionIdx(presetIdx);
      }
      // Restore the frame count when it lies on the current lattice.
      if (durationOptions.some((o) => o.frames === video.num_frames)) {
        setDurationIntentSeconds(video.num_frames / fps);
        setNumFrames(video.num_frames);
      }
      toast.success("Settings restored to inputs");
    },
    [resolutionPresets, durationOptions, fps],
  );

  // A status read started before an eject can answer after the one that
  // followed it, and this page has no periodic poll to correct it: the controls
  // would go on offering to generate against a runtime that is already free.
  // So every read takes a ticket and only the newest may write.
  const statusTicket = useRef(0);
  const setStatusIfNewest = useCallback(
    (ticket: number, next: VideoStatus) => {
      if (ticket === statusTicket.current) setStatus(next);
    },
    [],
  );

  // Answers with what it wrote, or null when the read failed or a newer one superseded it,
  // so a caller can act on what the server now says.
  const refreshStatus = useCallback(async (): Promise<VideoStatus | null> => {
    const ticket = ++statusTicket.current;
    try {
      const next = await getVideoStatus();
      setStatusIfNewest(ticket, next);
      return ticket === statusTicket.current ? next : null;
    } catch {
      // Status is best-effort; a failed poll shouldn't surface an error toast.
      return null;
    }
  }, [setStatusIfNewest]);

  // A generation can be refused because the runtime went away under the page: an idle
  // auto-unload frees it server-side and the browser hears nothing, since the eject event
  // is raised by whoever clicked eject. Without a re-read here Generate stays enabled off
  // the stale flag and every retry 409s again, so the refusal is the news that the model
  // is gone. Also clears the state that only means anything while one is resident (the
  // Reapply target, a replacement load's tracking), as the indicator eject does.
  const resyncAfterGenerateRefusal = useCallback(async () => {
    // A model picked while this read is in flight makes the answer stale rather than wrong:
    // /video/status reports committed state, so it says loaded: false for the load that has
    // just started. Acting on it would dismiss that load's toast and poll, and -- while its
    // start request is still out -- the cancel it counts as sends the compensating unload
    // that tears it down. The load counter is the fence handleLoad already bumps.
    const startLoad = loadSeq.current;
    const next = await refreshStatus();
    if (!isMounted.current || next === null || next.loaded) return;
    if (startLoad !== loadSeq.current) return;
    dropResidentState();
    setQuant(null);
  }, [refreshStatus, dropResidentState]);

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
            galleryCache.videos.find(
              (video) => video.id === galleryCache.selectedId,
            ) ?? galleryCache.videos[0];
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
        // ...but not while a load start is still in flight. begin_load refuses a second load
        // while one is registered, so a model picked in that window is rejected while the load
        // this eject was meant to cancel carries on, and handleLoad's compensating unload names
        // no load. Hold busy until the whole path settles, exactly as handleCancelLoad does.
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

  // Collapse the body-ported model selector when leaving the tab so returning to /video does not pop it back open unprompted.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error", updating the persistent toast in place each tick.
  const pollLoadProgress = useCallback(async () => {
    // This tick's cancellation fence: clearing pollTimer stops the next tick, not the awaits below.
    const seq = cancelSeq.current;
    try {
      const p = await getVideoLoadProgress();
      if (seq !== cancelSeq.current) return;
      if (p.phase === "ready") {
        dismissLoadToast();
        const ticket = ++statusTicket.current;
        const loaded = await getVideoStatus();
        if (seq !== cancelSeq.current) {
          // Cancelled while this read was in flight, so it describes a pipeline being torn down.
          // Drop it and refresh NOTHING: the unload's own response is authoritative and already
          // holds the newest ticket, and a status read issued from here would take a newer one
          // still and could re-report the model mid-teardown.
          return;
        }
        setStatusIfNewest(ticket, loaded);
        toast.success("Model loaded");
        setBusy(null);
        quantRevert.current?.commitRecipeClaim?.();
        quantRevert.current = null;
        // lastLoad.current already holds the now-resident pick, so drop its revert too.
        lastLoadRevert.current = null;
        return;
      }
      if (p.phase === "error") {
        dismissLoadToast();
        reportLoadFailure(p.error, "Failed to load model");
        setBusy(null);
        if (quantRevert.current) {
          revertPick(quantRevert.current);
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
          revertPick(quantRevert.current);
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
        toast(null, loadToastArgs(p, loadToastId.current, cancelLoadFromToast));
      }
    } catch {
      // Transient poll failure: keep trying.
    }
    if (seq !== cancelSeq.current) return;
    pollTimer.current = setTimeout(() => void pollLoadProgress(), 1000);
  }, [dismissLoadToast, refreshStatus, cancelLoadFromToast]);

  // Put back what a teardown removed when the load it was tearing down is still running: the
  // unload failed, so the poll and the toast were stopped for nothing. refreshStatus cannot do
  // this -- a first load is not resident yet, so status has nothing to report -- and without it a
  // multi-gigabyte load continues with no progress and no way to cancel it a second time.
  const restoreLoadTracking = useCallback(() => {
    loadTrackingRestored.current = true;
    setBusy("loading");
    lastLoadSig.current = null;
    loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, undefined, cancelLoadFromToast));
    void pollLoadProgress();
  }, [pollLoadProgress, cancelLoadFromToast]);

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
            // Merge the new clip and mint its link. Sorted, not prepended: a new clip is unpinned,
            // so the server puts it after the pinned group.
            const clip = p.video;
            // Refused if archived while this poll was in flight: forgetting the backend record
            // cannot revoke a response already on the wire, and it still says archived: false.
            if (!galleryCache.archived.has(clip.id) && !galleryCache.deleted.has(clip.id)) {
              stripEpoch.current += 1;
              setVideos((prev) =>
                sortGalleryItems([clip, ...prev.filter((v) => v.id !== clip.id)]),
              );
              setSelectedId(clip.id);
              void ensureSrc(clip);
            }
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
          loadToastId.current = toast(null, loadToastArgs(p, undefined, cancelLoadFromToast));
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
          if (!galleryCache.deleted.has(clip.id) && !galleryCache.archived.has(clip.id)) {
            stripEpoch.current += 1;
            setVideos((prev) =>
              prev.some((v) => v.id === clip.id) ? prev : sortGalleryItems([clip, ...prev]),
            );
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
  }, [refreshStatus, dismissLoadToast, pollLoadProgress, startGenPoll, stopGenPoll, ensureSrc, cancelLoadFromToast]);

  // Keep the snapshot helper stable: route effects depend on loadOrStage.
  const loadControlsRef = useRef({
    memoryMode,
    speedMode,
    attentionBackend,
    transformerCache,
    transformerQuant,
    selectedGpu,
    gpuChoices,
  });
  loadControlsRef.current = {
    memoryMode,
    speedMode,
    attentionBackend,
    transformerCache,
    transformerQuant,
    selectedGpu,
    gpuChoices,
  };
  const currentLoadAdvanced = useCallback(
    (kind: "gguf" | "single_file" | "pipeline"): VideoLoadAdvanced => {
      const controls = loadControlsRef.current;
      return {
        memory_mode: controls.memoryMode === "auto" ? undefined : controls.memoryMode,
        speed_mode: controls.speedMode === "auto" ? undefined : controls.speedMode,
        attention_backend:
          controls.attentionBackend === "auto" ? undefined : controls.attentionBackend,
        transformer_cache:
          controls.transformerCache === "auto" ? undefined : controls.transformerCache,
        transformer_quant:
          kind === "pipeline" && controls.transformerQuant !== "auto"
            ? controls.transformerQuant
            : undefined,
        // Dropped when the chosen card is gone (a driver reset, an eGPU unplugged), so a stale pick loads automatically instead of 400ing.
        gpu_ids:
          controls.selectedGpu !== "auto" &&
          controls.gpuChoices.some((d) => String(d.index) === controls.selectedGpu)
            ? [Number(controls.selectedGpu)]
            : undefined,
      };
    },
    [],
  );
  const resolveDownloadFootprint = useCallback(
    async (repoId: string, meta: ModelSelectorChangeMeta) => {
      if (!meta.ggufFilename) return null;
      const advanced = currentLoadAdvanced("gguf");
      const plan = await getVideoDownloadPlan({
        model_path: repoId,
        gguf_filename: meta.ggufFilename,
        model_kind: "gguf",
        hf_token: hfApiToken(getHfToken()),
        transformer_quant: advanced.transformer_quant,
        memory_mode: advanced.memory_mode,
        // The plan sizes its file set against the card the load will use, so it needs the pick.
        gpu_ids: advanced.gpu_ids,
      });
      const requiredBytes = plan.required_bytes ?? 0;
      if (requiredBytes <= 0) return null;
      return {
        requiredBytes,
        checkpointBytes: plan.checkpoint_bytes ?? meta.expectedBytes ?? 0,
      };
    },
    [currentLoadAdvanced],
  );

  const handleLoad = useCallback(
    // Resolves true when the background load STARTED (callers may revert optimistic picker state on false).
    async (
      repoId: string,
      opts: VideoLoadOptions,
      // Staged loads use the controls their preflight validated.
      pinned?: VideoLoadAdvanced,
    ): Promise<boolean> => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      // Read BEFORE the start request goes out: a Cancel pressed while it is in flight sends an
      // unload that can reach the backend first, find no load registered, and succeed without
      // stopping anything.
      const startSeq = cancelSeq.current;
      const startLoad = ++loadSeq.current;
      // Published now and settled in the finally below, so a cancel waits for the WHOLE path.
      let settleLoad: () => void = () => {};
      const inFlight = new Promise<void>((resolve) => {
        settleLoad = resolve;
      });
      pendingStart.current = inFlight;
      // Every exit below goes through this: it settles the promise a cancel is waiting on and
      // releases the ref, so the page cannot stay busy on a load that has already finished.
      const settle = (started: boolean): boolean => {
        settleLoad();
        if (pendingStart.current === inFlight) pendingStart.current = null;
        return started;
      };
      setBusy("loading");
      dismissLoadToast();
      lastLoadSig.current = null;
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, undefined, cancelLoadFromToast));
      // Snapshot the prior Reapply target first: a load that fails to START leaves the previous model resident, so Reapply must keep pointing at it.
      const prevLastLoad = lastLoad.current;
      const prevCanReapply = canReapply;
      const advanced = pinned ?? currentLoadAdvanced(opts.kind);
      // Spread, so the H3 partition rides along: Reapply reloads the same denoiser, not the default one.
      lastLoad.current = { repoId, ...opts };
      setCanReapply(true);
      // Carry the prior target so the async poll can restore it if the background load fails after starting.
      lastLoadRevert.current = { prev: prevLastLoad, canReapply: prevCanReapply };
      try {
        // Returns immediately; the load runs in the background and we poll.
        const startRequest = loadVideoModel({
          model_path: repoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          memory_mode: advanced.memory_mode,
          speed_mode: advanced.speed_mode,
          attention_backend: advanced.attention_backend,
          transformer_cache: advanced.transformer_cache,
          transformer_quant: advanced.transformer_quant,
          // Not an Advanced control: the partition is chosen per pick, so it stays on opts rather
          // than joining the pinned set.
          h3_task: opts.h3Task,
          gpu_ids: advanced.gpu_ids,
        });
        await startRequest;
      } catch (err) {
        lastLoad.current = prevLastLoad;
        setCanReapply(prevCanReapply);
        lastLoadRevert.current = null;
        dismissLoadToast();
        reportLoadFailure(err instanceof Error ? err.message : "", "Failed to start load");
        setBusy(null);
        void refreshStatus();
        return settle(false);
      }
      if (startSeq !== cancelSeq.current) {
        // Cancelled during the start request. The unload it sent may have landed before this load
        // registered, in which case it stopped nothing and the model is loading right now with no
        // toast and no Cancel button. The load exists on the backend as of this line, so unload
        // once more -- that one cannot miss it. Unless a NEWER load has since taken the page:
        // this unload names nothing, so firing it then would cancel that load instead of this
        // one, and the newer start has already superseded this load's token on the backend.
        if (startLoad === loadSeq.current) {
          try {
            await unloadVideoModel();
          } catch {
            // This request is the ONLY one that can still stop the load the first unload missed,
            // so a failure here is not best-effort: the load is running, untracked. Put the
            // tracking back exactly as a failed cancel does, so it stays visible and cancellable.
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
    [
      pollLoadProgress,
      refreshStatus,
      dismissLoadToast,
      cancelLoadFromToast,
      canReapply,
      currentLoadAdvanced,
    ],
  );

  // Downloads go through the Hub download manager like every other model, sharing its panel, progress, cancel and preflight. Mirrors Images.
  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: VideoLoadOptions;
    advanced: VideoLoadAdvanced;
    // The pick that staged it: a download outlives its pick, so it must not evict a newer one when it lands.
    token: number;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  handleLoadRef.current = handleLoad;
  // A download finishing while this page is hidden must not evict the model the visible page loaded. The pick is held, not dropped.
  const stagedLoadDeferred = useRef(false);
  // Both deferred paths run the load minutes after the pick was reported started, so both need
  // the same rollback: onReady when the page is active, and the effect below when the download
  // finished off-tab. The deferred load can still be REFUSED, by a training run or another load
  // claiming the slot while the download ran. Staging started no load, so nothing polls and the
  // poll's own rollback never runs; without this the selector keeps advertising a quant that was
  // never loaded. `owned` is read BEFORE the call, so a newer pick's label is left alone.
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
      // Same rule as the images page: a plan that ends without every dependency on disk must not
      // leave an intent for a late completion or a deferred activation to act on.
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      // No load started, so the poll that owns the after-start rollback never runs: put the
      // optimistic quant label back, or the selector describes the resident model with a
      // quant nothing ever loaded. Only for the pick that staged THIS job: a newer pick owns
      // the label from the moment it is made.
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

  // Stage a not-yet-downloaded hub pick, else load it directly.
  // `token` lets an awaiting caller drop out: the plan below is a second window for a newer pick to take the page.
  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: VideoLoadOptions,
      source: ModelSelectorChangeMeta["source"] = "hub",
      token?: number,
    ): Promise<boolean> => {
      // Every Hub pick needs the plan, not just an undownloaded one: a cached checkpoint can
      // still be missing its base repo's text encoder or VAE, and only the plan can see that.
      // The plan is cache-aware, so a fully cached pick comes back with no entries.
      // Staging never sets `busy`, so a second pick passes handleModelSelect's guard while this
      // plan is still in flight. Plans then resolve in response order, not pick order: without
      // this the older one restages over the newer queue, or loads the model the user left.
      // Bumped before the non-hub return too: a local pick must invalidate an in-flight hub plan.
      const pick = ++pickSeq.current;
      // The previous pick's staged intent dies with it. A pick that stages nothing (fully cached,
      // local, no plan) never calls stage(), so the hook's queue keeps running the older job and
      // its onReady would load the model the user moved away from, evicting this one.
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      stagedQuantRevert.current = null;
      const owns = () => token === undefined || pickGuard.holds(token);
      if (!owns()) return true;
      if (source !== "hub") return handleLoadRef.current(repoId, opts);

      const advanced = currentLoadAdvanced(opts.kind);
      // Read before the await: a pick made while the plan resolves replaces quantRevert, and this job must not revert it.
      const ownRevert = quantRevert.current;
      // Read inside the try, acted on outside it, as on the images page.
      let incompatible: string | null = null;
      try {
        const plan = await getVideoDownloadPlan({
          model_path: repoId,
          gguf_filename: opts.filename,
          model_kind: opts.kind,
          // Same token handleLoad sends: without it the metadata lookup fails on a gated base and the plan drops the companion entry, so the load pulls those files inline.
          hf_token: hfApiToken(getHfToken()),
          // The route preflights the same values used by the eventual load.
          transformer_quant: advanced.transformer_quant,
          memory_mode: advanced.memory_mode,
          // And the partition, for the same reason: the two H3 denoisers are separate downloads,
          // so a plan asked without it stages the default fl2va weights for a References pick.
          h3_task: opts.h3Task,
          // The plan sizes its file set against the card the load will use, so it needs the pick.
          gpu_ids: advanced.gpu_ids,
        });
        // Superseded. Report started so this pick's `.then` leaves the newer label alone.
        if (pick !== pickSeq.current || !owns()) return true;
        // Same selection-time refusal the images page makes: the plan is the last point at which
        // an incompatible pairing can be caught before the download it would waste. No video
        // family declares one today (the check is the FLUX.2 GGUF/base size pairing, and the video
        // planner has no diffusers base to pair against), so this is the shared envelope's half of
        // the contract rather than a live path -- keep it, or a future one lands unguarded.
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
              // The entry carrying the picked checkpoint file, so the panel can label it without
              // guessing: filenames cannot tell the two apart once a checkpoint ships as
              // .safetensors like its companions do. Repo identity alone is not enough, because a
              // checkpoint that shares its repo with the companions and is already cached leaves an
              // entry of companion files only. A pipeline pick has no one file: the repo IS it.
              // The backend's own answer wins: a gated pipeline is staged from an ungated MIRROR,
              // so its entry no longer carries the id we picked and the id test below reads the
              // whole selected model as companion assets. `??`, not `||`: a planner that says false
              // is answering, and the fallback exists only for a backend too old to send the key.
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
    [stage, pickGuard, currentLoadAdvanced],
  );

  // A GGUF pick can arrive with only a repo id (a pinned row, a curated artifact, a local GGUF directory). The backend
  // rejects a gguf load with no filename and a pipeline load of a GGUF repo, so name the file from the listing first.
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
          // Filename-qualified like the expander branch: the LTX variant lives in the checkpoint name, not the repo id.
          applyVideoModelDefaults(`${repoId}/${filename}`);
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
    [applyVideoModelDefaults, loadOrStage, pickGuard, quant, revertPick],
  );

  // A pick that is rejected after beginPick() has already retired the staged pick it replaced, so
  // nothing will load and nothing else will restore the label. Hand the resident state back here or
  // the selector keeps showing the abandoned pick's quant and recipe for good.
  const abandonPick = useCallback(() => {
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [revertPick]);

  // A hidden page owns nothing: both stay mounted, so a resolution started here must not load after the user switched.
  useEffect(() => {
    if (!active) {
      pickGuard.release();
      // Through the SAME ending as pressing Cancel. The pick that opened this dialog already
      // replaced the quant label, steps and guidance and parked their rollback in quantRevert,
      // and the load it was deferring never ran. Clearing the dialog alone leaves the controls
      // describing H3 over whatever model is still resident, and leaves a stale rollback for the
      // next pick to trip over.
      setPendingH3Load((pending) => {
        if (pending) abandonPick();
        return null;
      });
    }
  }, [abandonPick, active, pickGuard]);

  // A diffusion model picked from the chat picker arrives as ?model= on this route. Load it once, then clear the params.
  // This route's own match, never `strict: false`: that resolves to the ROOT match, whose search is whatever route is live, and
  // /hub names its selection with the same param. `active` cannot fence that off, since it lags the matches by a render.
  const routeSearch = useSearch({ from: "/video", shouldThrow: false });
  const navigateSelf = useNavigate();
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    // A hidden page owns no query: both diffusion pages stay mounted.
    if (!active) return;
    if (!videoPresets.hydrated) return;
    const wanted = routeSearch?.model;
    // Model AND quant, released once the query is gone: this page stays mounted, so a marker that outlived the query made re-picking a dead click.
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    // `quant` is used verbatim as a filename; a label there (a hand-built link, an older producer) is resolved instead.
    // The two fields, not the object: `routeSearch` is rebuilt every render, so it would churn the deps.
    const routed = { quant: routeSearch?.quant, ggufQuant: routeSearch?.ggufQuant };
    const routedFilename = routedGgufFilename(routed);
    const routedLabel = routedGgufLabel(routed);
    const key = `${wanted}|${routeSearch?.quant ?? ""}|${routeSearch?.ggufQuant ?? ""}`;
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
        loadGgufRepoPick(wanted, routedLabel, "hub"),
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
      void Promise.resolve().then(() => loadGgufRepoPick(pick.repoId, null, "hub"));
      return;
    }
    // Match every direct picker branch: the routed intent owns both the visible build label and
    // the model-specific Default recipe, and a load that never becomes resident rolls both back.
    const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
    quantRevert.current = revert;
    setQuant(pick.opts.kind === "pipeline" ? null : (pick.opts.filename ?? null));
    applyVideoModelDefaults(
      pick.opts.filename ? `${pick.repoId}/${pick.opts.filename}` : pick.repoId,
    );
    // A routed pick owns the page exactly like a direct one, so it has to offer the same choice.
    if (isH3PipelinePick(pick.repoId, pick.opts.kind)) {
      setPendingH3Load({
        repoId: pick.repoId,
        opts: pick.opts,
        source: "hub",
        token,
      });
      return;
    }
    void loadOrStage(pick.repoId, pick.opts, "hub", token).then((started) => {
      if (!started && pickGuard.holds(token) && quantRevert.current === revert) {
        revertPick(revert);
        quantRevert.current = null;
      }
    });
  }, [
    active,
    applyVideoModelDefaults,
    routeSearch?.model,
    routeSearch?.quant,
    routeSearch?.ggufQuant,
    loadOrStage,
    loadGgufRepoPick,
    navigateSelf,
    pickGuard,
    quant,
    revertPick,
    videoPresets.hydrated,
  ]);


  // The task dialog defers the load out of the branch that snapshotted the rollback, so the two
  // ways out of it carry that branch's two endings: choosing runs the load and reverts if it never
  // starts, cancelling abandons the pick outright.
  const chooseH3Task = useCallback(
    (task: H3Task) => {
      const pending = pendingH3Load;
      setPendingH3Load(null);
      if (!pending || !pickGuard.holds(pending.token)) return;
      const revert = quantRevert.current;
      void loadOrStage(
        pending.repoId,
        { ...pending.opts, h3Task: task },
        pending.source,
        pending.token,
      ).then((started) => {
        // One slot, so only the pick that set the label may take it back.
        if (!started && revert && quantRevert.current === revert && pickGuard.holds(pending.token)) {
          revertPick(revert);
          quantRevert.current = null;
        }
      });
    },
    [loadOrStage, pendingH3Load, pickGuard, revertPick],
  );

  const cancelH3TaskChoice = useCallback(() => {
    setPendingH3Load(null);
    abandonPick();
    pickGuard.cancel();
  }, [abandonPick, pickGuard]);

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    // Status is authoritative when another client replaced the resident model. The ref remains
    // the fallback while this page's own load is committing and status has not caught up yet.
    const l = lastLoad.current;
    if (l) {
      void handleLoad(l.repoId, {
        kind: l.kind,
        filename: l.filename,
        h3Task: l.h3Task,
      });
    }
  }, [handleLoad]);

  // The chat picker emits (modelId, quant + filename) for a GGUF, or just (modelId) for a curated pipeline pick.
  // Every pick supersedes the one before it, whichever route it takes. A staged download outlives
  // its pick, and the direct-local branches call handleLoad rather than loadOrStage, so clearing
  // only inside loadOrStage left the old job's onReady free to load the abandoned model over the
  // one just chosen. Bumping the sequence here also invalidates any plan still in flight.
  const beginPick = useCallback(() => {
    pickSeq.current += 1;
    pendingStagedLoad.current = null;
    stagedLoadDeferred.current = false;
    stagedQuantRevert.current = null;
  }, []);

  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // Ignore picks while a load/generation/unload is in flight.
      if (busy !== null) return;
      beginPick();
      // This pick owns the page now, so one still awaiting a listing or a plan drops out. Before any branch: staging never
      // sets `busy`, so any pick can land on an awaiting one.
      const token = pickGuard.claim();
      // Curated non-GGUF model: load as a full pipeline.
      const spec = loadSpecFor(id, VIDEO_CATALOG);
      if (spec && spec.kind !== "gguf") {
      // Carried forward when one is already pending: a superseded staged pick left its
      // optimistic quant and recipe in state, so snapshotting now would record THAT and
      // restore a model which never loaded. The live entry already holds the resident one.
        // Registers its own rollback like every other branch. Leaving the previous pick's entry in
        // place would let that older staged download, on cancelling, revert to state from before it
        // -- over a selection this pick already replaced -- and leave this one with no rollback.
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(null);
        // The distilled variant lives in the checkpoint name, not the repo id, so include the filename when seeding defaults.
        // Without it these distilled entries fall through to the generic LTX 40-step/CFG-4 defaults instead of the 8-step schedule.
        applyVideoModelDefaults(spec.filename ? `${id}/${spec.filename}` : id);
        if (isH3PipelinePick(id, spec.kind)) {
          setPendingH3Load({
            repoId: id,
            opts: { kind: spec.kind, filename: spec.filename },
            source: meta.source,
            token,
          });
          return;
        }
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
      // GGUF quant pick from the variant expander. Optimistic for picker feedback, reverted if the load fails to START; the poll owns the after-start revert.
      if (meta.ggufVariant && meta.ggufFilename) {
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(meta.ggufVariant);
        // Include the picked filename: the variant (distilled vs dev) lives there, not in the repo id.
        applyVideoModelDefaults(`${id}/${meta.ggufFilename}`);
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
          // A repo id or local directory, not a file. The listing names its .gguf and the label picks between siblings; a
          // local pick passes its directory so the listing reads that path, not a hub repo.
          void loadGgufRepoPick(
            id,
            meta.ggufVariant ?? null,
            meta.source,
            meta.source === "local" ? id : null,
          );
          return;
        }
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(filename);
        applyVideoModelDefaults(id);
        void handleLoad(dir, { kind: "gguf", filename }).then((started) => {
          if (!started) {
            revertPick(revert);
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
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(filename);
        applyVideoModelDefaults(id);
        void handleLoad(dir, { kind: "single_file", filename }).then((started) => {
          if (!started) {
            revertPick(revert);
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
          meta.source,
          meta.source === "local" ? id : null,
        );
        return;
      }
      // Otherwise treat it as a full diffusers repo. The backend gates loads to unsloth/* repos, the family bases, or on-device paths.
      if (meta.source !== "local" && !id.toLowerCase().startsWith("unsloth/")) {
        toast.error("Only unsloth or on-device video models can be loaded here");
        abandonPick();
        return;
      }
      // Its own rollback, like every other branch: leaving the previous pick's entry live lets an
      // older staged download revert over a selection this pick already replaced.
      const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
      quantRevert.current = revert;
      setQuant(null);
      applyVideoModelDefaults(id);
      // The on-device copy of the H3 pipeline lands here rather than in the curated branch, and
      // it needs the same partition question: without it the load silently takes fl2va.
      if (isH3PipelinePick(id, "pipeline")) {
        setPendingH3Load({
          repoId: id,
          opts: { kind: "pipeline" },
          source: meta.source,
          token,
        });
        return;
      }
      void loadOrStage(id, { kind: "pipeline" }, meta.source, token).then((started) => {
        if (!started && pickGuard.holds(token)) {
          revertPick(revert);
          quantRevert.current = null;
        }
      });
    },
    [
      abandonPick,
      applyVideoModelDefaults,
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

  // Resolves true when the backend accepted the unload; handleCancelLoad reports the cancel only then.
  const handleUnload = useCallback(async (): Promise<boolean> => {
    dropResidentState();
    loadTrackingRestored.current = false;
    setBusy("unloading");
    try {
      setStatusIfNewest(++statusTicket.current, await unloadVideoModel());
      setQuant(null);
      // Hold the page until any load start still in flight has run to its END, compensating
      // unload and all. The selector's eject routes straight here, so without the fence an eject
      // landing before the start registered returned success and cleared busy: the user picks
      // another model, the backend refuses it ("a load is already in progress") because the older
      // start won the race, and that older handler -- seeing the newer loadSeq -- skips its
      // compensating unload and returns without restarting its poll, leaving a multi-gigabyte
      // load running with no toast and no cancel control. Same fence handleCancelLoad and the
      // external-eject listener take.
      const pending = pendingStart.current;
      if (pending) {
        try {
          await pending;
        } catch {
          // Its own handler reports the failure; this only waits for the window to close.
        }
      }
      // The wait above can end with the tracking RESTORED: handleLoad's compensating unload
      // failed, so the load is still running. This eject stopped nothing, so do not report
      // success -- the caller would toast "stopped loading" over a live load.
      return !loadTrackingRestored.current;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
      return false;
    } finally {
      // Not an unconditional clear. A restore during the wait above put the page back to
      // "loading" deliberately; wiping it hides the toast's Cancel and the "Cancel load"
      // button and re-enables the picker over a load that is still running.
      setBusy((prev) => (prev === "unloading" ? null : prev));
    }
  }, [refreshStatus, dropResidentState]);

  // Cancelling a load IS the unload: it sets the running load's cancel event, bumps the load token so the
  // worker can never commit, and drops the load marker. What it leaves behind is only cache: bytes already
  // fetched stay in the HF cache, so loading the same model again resumes instead of restarting, and no
  // half-built pipeline survives (the worker's commit is token-gated and unload clears the GPU state).
  const handleCancelLoad = useCallback(async () => {
    const wasLoading = busy === "loading";
    if (await handleUnload()) {
      // handleUnload holds the page for the whole pending-start path before it returns, so by
      // here the window the backend's "a load is already in progress" refusal lives in is shut.
      toast.info("Stopped loading the model", {
        description: "Anything already downloaded stays cached, so loading it again resumes.",
      });
      return;
    }
    // Already restored inside handleUnload (its own compensating-unload failure), so the toast
    // and the poll are up: a second restore would raise a duplicate toast and a second poll loop.
    if (!wasLoading || loadTrackingRestored.current) return;
    // The unload failed, so the load is still running and its tracking was torn down for nothing.
    restoreLoadTracking();
  }, [busy, handleUnload, restoreLoadTracking]);

  useEffect(() => {
    cancelLoadRef.current = () => void handleCancelLoad();
  }, [handleCancelLoad]);

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
    if (supportsReferences && referenceImages.length === 0 && referenceVideos.length === 0) {
      toast.error("Add a reference picture or video for this checkpoint");
      return;
    }
    for (const [index, entry] of referenceVideos.entries()) {
      const start = entry.trimStartSeconds;
      const end = entry.trimEndSeconds;
      const trimError = referenceVideoTrimError(
        `Video ${index + 1}`,
        start,
        end,
        entry.video.durationSeconds,
      );
      if (trimError) {
        toast.error(trimError);
        return;
      }
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

    // Omitting both dimensions delegates "match source" to the backend.
    const matchSource = resolutionIdx === MATCH_SOURCE_RESOLUTION;
    const preset = resolutionPresets[resolutionIdx] ?? resolutionPresets[0];

    setBusy("generating");
    setGenStep(null);
    // The POST only STARTS the job and returns at once (a clip takes minutes, and the secure-mode tunnel caps responses near 100s).
    // A synchronous rejection still surfaces here; everything after acceptance arrives via the poll.
    try {
      await generateVideo({
        prompt: prompt.trim(),
        // Only send a negative prompt when guidance uses it, so the recipe does not record one the model ignored.
        negative_prompt:
          status?.supports_cfg !== false && guidance > 0
            ? negativePrompt.trim() || undefined
            : undefined,
        width: matchSource ? undefined : preset[0],
        height: matchSource ? undefined : preset[1],
        num_frames: numFrames,
        fps,
        steps,
        guidance: status?.supports_cfg !== false ? guidance : undefined,
        seed: resolvedSeed,
        first_frame: supportsKeyframes ? firstFrame ?? undefined : undefined,
        last_frame: supportsKeyframes ? lastFrame ?? undefined : undefined,
        reference_images:
          supportsReferences && referenceImages.length > 0
            ? referenceImageDataUrls(referenceImages)
            : undefined,
        reference_videos:
          supportsReferences && referenceVideos.length > 0
            ? referenceVideos.map(
                (entry): VideoReferenceVideo => ({
                  video: entry.video.dataUrl,
                  audio: entry.audio?.dataUrl,
                  trim_start_seconds: entry.trimStartSeconds ?? undefined,
                  trim_end_seconds: entry.trimEndSeconds ?? undefined,
                }),
              )
            : undefined,
        reference_audios:
          supportsReferences && referenceAudios.length > 0
            ? referenceAudios.map((entry) => entry.dataUrl)
            : undefined,
        reference_image_size: canPickReferenceSize ? referenceImageSize : undefined,
        // Send only overrides of the released schedule.
        flow_shift:
          defaultFlowShift != null && flowShift != null && flowShift !== defaultFlowShift
            ? flowShift
            : undefined,
        audio_flow_shift:
          canPickAudioFlowShift && audioFlowShift != null && audioFlowShift !== defaultAudioFlowShift
            ? audioFlowShift
            : undefined,
      });
    } catch (err) {
      if (!isMounted.current) return;
      toast.error(err instanceof Error ? err.message : "Video generation failed");
      setBusy(null);
      setGenStep(null);
      // The refusal can be "No video model is loaded": re-read rather than leave Generate
      // enabled against a runtime that is already free.
      void resyncAfterGenerateRefusal();
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
    status?.supports_cfg,
    supportsKeyframes,
    firstFrame,
    lastFrame,
    supportsReferences,
    referenceImages,
    referenceVideos,
    referenceAudios,
    canPickReferenceSize,
    referenceImageSize,
    flowShift,
    defaultFlowShift,
    audioFlowShift,
    defaultAudioFlowShift,
    canPickAudioFlowShift,
    startGenPoll,
    resyncAfterGenerateRefusal,
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
      {gpuChoices.length > 0 && (
        <AdvancedSelect
          label="GPU"
          hint="Which card this model loads on. Auto uses whichever device torch is pointing at, which on a mixed box is not necessarily the largest. A video model is never split across cards, so this is one choice, not a pool."
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
      <LoadedBuildSummary status={status} />
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
      <AlertDialog
        open={active && clearConfirmOpen}
        onOpenChange={(open) => {
          if (!clearingGallery) setClearConfirmOpen(open);
        }}
      >
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Clear all videos?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes every generated video from the gallery. This action cannot
              be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={clearingGallery}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={clearingGallery}
              onClick={(event) => {
                event.preventDefault();
                void handleClearAll();
              }}
            >
              {clearingGallery ? "Clearing…" : "Clear all"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <Dialog
        open={pendingH3Load !== null}
        onOpenChange={(open) => {
          if (!open) cancelH3TaskChoice();
        }}
      >
        {/* Squarer than the shared dialog's rounded-4xl: at this width the default reads as a
            lozenge rather than a panel. The option cards step down from it so the nesting holds. */}
        <DialogContent className="max-w-lg rounded-2xl">
          <DialogHeader>
            <DialogTitle>Choose how MiniMax H3 should generate</DialogTitle>
            <DialogDescription>
              MiniMax H3 uses a separate denoiser for reference generation. Choose the mode you
              want to load now. Shared components already on disk are reused.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3 sm:grid-cols-2">
            <Button
              type="button"
              variant="outline"
              className="h-auto items-start justify-start whitespace-normal rounded-xl p-4 text-left"
              onClick={() => chooseH3Task("fl2va")}
            >
              <span className="grid gap-1">
                <span className="font-medium">Text and frames</span>
                <span className="text-ui-11 font-normal leading-snug text-muted-foreground">
                  Generate from text, with optional first and last frame images.
                </span>
              </span>
            </Button>
            <Button
              type="button"
              variant="outline"
              className="h-auto items-start justify-start whitespace-normal rounded-xl p-4 text-left"
              onClick={() => chooseH3Task("ref2va")}
            >
              <span className="grid gap-1">
                <span className="font-medium">References</span>
                <span className="text-ui-11 font-normal leading-snug text-muted-foreground">
                  Generate from reference pictures, videos and audio tracks.
                </span>
              </span>
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      {/* Top: the model selector, sitting clear of the sidebar and level with the controls column below. Load progress shows in a toast. */}
      <div className="@container pointer-events-none relative z-40 flex h-[48px] shrink-0 items-start justify-between pl-[var(--studio-media-header-left-inset,1.5rem)] pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
        {/* min-w-0: without it a long resident model name pushes the Images link off a phone screen. */}
        <div className="pointer-events-auto flex min-w-0 items-center gap-3">
          <ModelSelector
            models={videoModels}
            value={status?.loaded ? status.repo_id ?? undefined : undefined}
            activeGgufVariant={quant}
            onValueChange={handleModelSelect}
            resolveDownloadFootprint={resolveDownloadFootprint}
            onEject={status?.loaded ? handleUnload : undefined}
            variant="ghost"
            className="!h-[34px]"
            task={VIDEO_GEN_TASKS}
            catalog={VIDEO_CATALOG}
            placeholder="Select video model"
            open={active && selectorOpen}
            onOpenChange={(o) => setSelectorOpen(active && o)}
          />
          {/* The load's own cancel, beside the selector rather than inside it: the selector's eject needs a
              resident model, so it is hidden for exactly the span a first load runs. A real button, not the
              trigger's aria-hidden eject hit area, so it is reachable by keyboard and a screen reader. Says
              "load", never "download": the download manager's own Cancel stops a staged pull, a different job. */}
          {busy === "loading" && (
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
          {/* Loaded-model status line: family / kind / offload / speed, as the images page surfaces on load. Hidden until a model is resident. */}
          {status?.loaded && (
            <div className="hidden min-w-0 items-center gap-3 text-ui-11 @min-[720px]:flex">
              {status.family && <StatusChip label="Family" value={status.family} />}
              {status.engine && <StatusChip label="Engine" value={status.engine} />}
              {status.model_kind && <StatusChip label="Kind" value={status.model_kind} />}
              {status.offload_policy && (
                <StatusChip label="Offload" value={status.offload_policy} />
              )}
              {status.speed_mode && <StatusChip label="Speed" value={status.speed_mode} />}
            </div>
          )}
        </div>
        <div className="pointer-events-auto flex shrink-0 items-center gap-2">
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
        <div className="flex w-full shrink-0 flex-col border-b border-border/60 pl-8 md:w-[400px] md:overflow-hidden md:border-r md:border-b-0">
          {/* pl-0.5 keeps focus rings off the scroll container's edge. */}
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar panel-scroll-fade-action flex min-h-0 flex-1 flex-col gap-4 pb-6 pl-0.5 pr-7 md:overflow-y-auto",
              settingsFadeClass,
            )}
          >
          {/* Names the pane, as the Images column does. Same shape there, so
              the two pages stay level. */}
          <div className="mb-2 flex items-start justify-between gap-3">
            <div className="min-w-0 grid gap-1.5">
              <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none text-foreground">
                {/* The app's Video icon, same as the sidebar row. */}
                <HugeiconsIcon icon={FlimSlateIcon} className="size-[18px] shrink-0" />
                Create videos
              </h2>
              <p className="text-xs leading-snug text-muted-foreground">
                {supportsReferences
                  ? "Generate a video from a prompt and reference pictures, videos or audio"
                  : supportsKeyframes
                    ? "Generate a video from a prompt, or from a start and end frame"
                    : "Generate a video from a prompt"}
              </p>
            </div>

            <MediaGenerationPresetControl
              kind="video"
              presets={videoPresets.presets}
              activePreset={videoPresets.activePreset}
              ready={videoPresets.presetsReady}
              hasUnsavedChanges={videoPresets.hasUnsavedChanges}
              onSelect={videoPresets.selectPreset}
              onSave={videoPresets.savePreset}
              onDelete={videoPresets.deletePreset}
            />
          </div>

          <Field label="Prompt">
            <Textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </Field>

          {supportsKeyframes && (
            <div className="grid gap-2">
              <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
                Start and end frame
                <InfoHint>
                  Optional. A start frame animates that picture; an end frame makes the clip land
                  on one; both make it travel between them. Text-to-video is what you get with
                  neither. The start frame is stretched onto the canvas and the end frame is
                  centre-cropped, which is how the model was conditioned.
                </InfoHint>
              </span>
              <div className="grid grid-cols-2 gap-2">
                <div className="grid gap-1.5">
                  <span className="text-ui-11 text-muted-foreground/70">Start frame</span>
                  <ImageDropzone
                    value={firstFrame}
                    onChange={setFirstFrame}
                    label="Click or drop"
                    removeLabel="Remove start frame"
                  />
                </div>
                <div className="grid gap-1.5">
                  <span className="text-ui-11 text-muted-foreground/70">End frame</span>
                  <ImageDropzone
                    value={lastFrame}
                    onChange={setLastFrame}
                    label="Click or drop"
                    removeLabel="Remove end frame"
                  />
                </div>
              </div>
              {canvasKeyframe && !matchedResolution && (
                // Surface the same aspect-ratio rejection before Generate.
                <p className="text-ui-11 leading-snug text-destructive">
                  This picture is too far from square for MiniMax-H3, which was trained between
                  1:4 and 4:1. Crop it, or pick a resolution preset to stretch it onto.
                </p>
              )}
            </div>
          )}

          {supportsReferences && (
            <div className="grid gap-2">
              <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
                References
                <InfoHint>
                  Lock the clip to a character, style, motion, camera move or voice. Name them in
                  the prompt by the tags below -- "use the cat from &lt;Picture 1&gt;, match the
                  shot rhythm of &lt;Video 1&gt;" -- since order is what the model reads them by.
                  At most 9 pictures, 3 videos and 3 audio clips, 12 in all. Audio needs a picture
                  or a video to go with it.
                </InfoHint>
              </span>

              <div className="grid grid-cols-3 gap-2">
                {referenceImages.map((image, index) => (
                  // Index IS the identity here: the tag in the prompt is the position.
                  // biome-ignore lint/suspicious/noArrayIndexKey: position is the reference's name
                  <div key={`picture-${index}`} className="grid gap-1">
                    <span className="text-ui-11 text-muted-foreground/70">
                      Picture {index + 1}
                    </span>
                    <div className="relative h-24 overflow-hidden rounded-[10px] border border-border bg-muted/30">
                      <button
                        type="button"
                        aria-label={`Edit crop for picture ${index + 1}`}
                        className="group h-full w-full overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary"
                        onClick={() => setCropPictureIndex(index)}
                      >
                        <img
                          src={image.dataUrl}
                          alt=""
                          className="h-full w-full object-cover transition-transform group-hover:scale-[1.02]"
                        />
                        <span className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-1 bg-gradient-to-t from-black/80 to-transparent px-2 pb-1.5 pt-6 text-ui-11 font-medium text-white">
                          <HugeiconsIcon icon={ImageCropIcon} className="size-3.5" />
                          Edit crop
                        </span>
                      </button>
                      <Tooltip>
                        <TooltipTrigger asChild={true}>
                          <Button
                            type="button"
                            variant="secondary"
                            size="icon"
                            aria-label={`Remove picture ${index + 1}`}
                            className="absolute right-1.5 top-1.5 size-7 bg-background/85 shadow-sm backdrop-blur-sm"
                            onClick={() =>
                              setReferenceImages((prev) =>
                                prev.filter((_, current) => current !== index),
                              )
                            }
                          >
                            <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Remove picture {index + 1}</TooltipContent>
                      </Tooltip>
                    </div>
                  </div>
                ))}
                {referenceImages.length < 9 && hasReferenceRoom && (
                  <div className="grid gap-1">
                    <span className="text-ui-11 text-muted-foreground/70">
                      Picture {referenceImages.length + 1}
                    </span>
                    <ImageDropzone
                      value={null}
                      onChange={(next) =>
                        next &&
                        setReferenceImages((prev) => [...prev, stageReferenceImage(next)])
                      }
                      label="Add"
                      className="h-24"
                    />
                  </div>
                )}
              </div>

              <div className="grid gap-1.5">
                {referenceVideos.map((entry, index) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: position is the reference's name
                  <div key={`video-${index}`} className="grid gap-1">
                    <span className="text-ui-11 text-muted-foreground/70">Video {index + 1}</span>
                    <ReferenceMediaPicker
                      kind="video"
                      value={entry.video}
                      label={`Video ${index + 1}`}
                      onChange={(next) => {
                        const trim = defaultReferenceVideoTrim(next?.durationSeconds);
                        setReferenceVideos((prev) =>
                          next
                            ? prev.map((item, i) =>
                                i === index
                                  ? {
                                      ...item,
                                      video: next,
                                      trimStartSeconds: trim.start,
                                      trimEndSeconds: trim.end,
                                    }
                                  : item,
                              )
                            : prev.filter((_, i) => i !== index),
                        );
                      }}
                    />
                    <video
                      controls={true}
                      muted={true}
                      preload="metadata"
                      src={entry.video.dataUrl}
                      className="max-h-36 w-full rounded-[10px] bg-black object-contain"
                    />
                    <div className="grid grid-cols-2 gap-2">
                      <div className="grid gap-1 text-ui-11 text-muted-foreground">
                        Trim start (seconds)
                        <Input
                          aria-label={`Video ${index + 1} trim start in seconds`}
                          type="number"
                          min={0}
                          max={entry.video.durationSeconds}
                          step={0.1}
                          value={entry.trimStartSeconds ?? ""}
                          placeholder="0"
                          onChange={(event) =>
                            setReferenceVideos((prev) =>
                              prev.map((item, i) =>
                                i === index
                                  ? {
                                      ...item,
                                      trimStartSeconds:
                                        event.target.value === ""
                                          ? null
                                          : Number(event.target.value),
                                    }
                                  : item,
                              ),
                            )
                          }
                        />
                      </div>
                      <div className="grid gap-1 text-ui-11 text-muted-foreground">
                        Trim end (seconds)
                        <Input
                          aria-label={`Video ${index + 1} trim end in seconds`}
                          type="number"
                          min={0}
                          max={entry.video.durationSeconds}
                          step={0.1}
                          value={entry.trimEndSeconds ?? ""}
                          placeholder={
                            entry.video.durationSeconds !== undefined
                              ? Math.min(
                                  entry.video.durationSeconds,
                                  H3_REFERENCE_MAX_SECONDS,
                                ).toFixed(1)
                              : "15"
                          }
                          onChange={(event) =>
                            setReferenceVideos((prev) =>
                              prev.map((item, i) =>
                                i === index
                                  ? {
                                      ...item,
                                      trimEndSeconds:
                                        event.target.value === ""
                                          ? null
                                          : Number(event.target.value),
                                    }
                                  : item,
                              ),
                            )
                          }
                        />
                      </div>
                    </div>
                    <ReferenceVideoTrimStatus
                      label={`Video ${index + 1}`}
                      start={entry.trimStartSeconds}
                      end={entry.trimEndSeconds}
                      sourceDuration={entry.video.durationSeconds}
                    />
                    <ReferenceMediaPicker
                      kind="audio"
                      compact={true}
                      value={entry.audio}
                      label="Replace its soundtrack (optional)"
                      onChange={(next) =>
                        setReferenceVideos((prev) =>
                          prev.map((item, i) => (i === index ? { ...item, audio: next } : item)),
                        )
                      }
                    />
                  </div>
                ))}
                {referenceVideos.length < 3 && hasReferenceRoom && (
                  <ReferenceMediaPicker
                    kind="video"
                    value={null}
                    label={`Add video ${referenceVideos.length + 1}`}
                    onChange={(next) => {
                      if (!next) return;
                      const trim = defaultReferenceVideoTrim(next.durationSeconds);
                      setReferenceVideos((prev) => [
                        ...prev,
                        {
                          video: next,
                          audio: null,
                          trimStartSeconds: trim.start,
                          trimEndSeconds: trim.end,
                        },
                      ]);
                    }}
                  />
                )}
              </div>

              <div className="grid gap-1.5">
                {referenceAudios.map((audio, index) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: position is the reference's name
                  <div key={`audio-${index}`} className="grid gap-1">
                    <span className="text-ui-11 text-muted-foreground/70">Audio {index + 1}</span>
                    <ReferenceMediaPicker
                      kind="audio"
                      value={audio}
                      label={`Audio ${index + 1}`}
                      onChange={(next) =>
                        setReferenceAudios((prev) =>
                          next
                            ? prev.map((item, i) => (i === index ? next : item))
                            : prev.filter((_, i) => i !== index),
                        )
                      }
                    />
                  </div>
                ))}
                {referenceAudios.length < 3 &&
                  hasReferenceRoom &&
                  (referenceImages.length > 0 || referenceVideos.length > 0) && (
                    <ReferenceMediaPicker
                      kind="audio"
                      value={null}
                      label={`Add audio ${referenceAudios.length + 1}`}
                      onChange={(next) => next && setReferenceAudios((prev) => [...prev, next])}
                    />
                  )}
              </div>

              {referenceImages.length === 0 && referenceVideos.length === 0 && (
                // Ref2VA cannot generate without an image or video reference.
                <p className="text-ui-11 leading-snug text-muted-foreground/70">
                  This checkpoint generates from references. Add a picture or a video, or load a
                  first/last-frame checkpoint for plain text-to-video.
                </p>
              )}

              {canPickReferenceSize && (
                <Field
                  label="Reference detail"
                  hint="How reference pictures are sized. Match keeps them at the clip's own pixel area. Max encodes them at 2048px for stronger identity fidelity, and rides every sampling step, so it can be several times slower."
                >
                  <Select
                    value={referenceImageSize}
                    onValueChange={(v) => setReferenceImageSize(v as "match" | "max")}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="match">Match the clip</SelectItem>
                      <SelectItem value="max">Max (2048px, slower)</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              )}
            </div>
          )}

          {cropPictureIndex !== null && referenceImages[cropPictureIndex] && (
            <ReferenceImageEditor
              key={cropPictureIndex}
              open={true}
              picture={referenceImages[cropPictureIndex]}
              pictureNumber={cropPictureIndex + 1}
              onOpenChange={(open) => {
                if (!open) setCropPictureIndex(null);
              }}
              onApply={(dataUrl, crop) =>
                setReferenceImages((prev) =>
                  applyReferenceImageCrop(prev, cropPictureIndex, dataUrl, crop),
                )
              }
            />
          )}

          {status?.supports_cfg !== false && (
            <NegativePromptField
              value={negativePrompt}
              onChange={setNegativePrompt}
              open={negativeOpen}
              onOpenChange={setNegativeOpen}
              hint="What to steer the video away from. Only used when guidance is above 0."
            />
          )}

          <Field
            label="Resolution"
            hint="The frame size. Presets come from the loaded model; portrait presets are marked. With a keyframe staged, Match source keeps the picture's own shape."
          >
            <Select
              value={String(resolutionIdx)}
              onValueChange={(v) => {
                const index = Number(v);
                const resolution = resolutionPresets[index];
                if (resolution) setResolutionIntent(resolution);
                setResolutionIdx(index);
              }}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {canvasKeyframe && (
                  <SelectItem value={String(MATCH_SOURCE_RESOLUTION)}>
                    Match source
                    {matchedResolution
                      ? ` · ${matchedResolution[0]} × ${matchedResolution[1]}`
                      : ""}
                  </SelectItem>
                )}
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
              onValueChange={(v) => {
                const frames = Number(v);
                setDurationIntentSeconds(frames / fps);
                setNumFrames(frames);
              }}
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
          {status?.supports_cfg !== false && (
            <SliderField
              label="Guidance"
              hint="Classifier-free guidance scale. Keep low (1) for the distilled model; the base model uses real guidance (4)."
              value={guidance}
              min={0}
              max={20}
              step={0.5}
              onChange={setGuidance}
            />
          )}
          {defaultFlowShift != null && (
            <SliderField
              label="Motion shift"
              hint="Sigma shift of the video schedule. Higher spends more of the schedule at high noise, which reads as more motion and less fine detail. MiniMax-H3 ships 12."
              value={flowShift ?? defaultFlowShift}
              min={1}
              max={30}
              step={0.5}
              onChange={setFlowShift}
            />
          )}
          {canPickAudioFlowShift && defaultAudioFlowShift != null && (
            <SliderField
              label="Audio shift"
              hint="Sigma shift of the audio schedule, which MiniMax-H3 runs alongside the video one. Ships at 3."
              value={audioFlowShift ?? defaultAudioFlowShift}
              min={1}
              max={30}
              step={0.5}
              onChange={setAudioFlowShift}
            />
          )}
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
          <div className="relative z-10 flex shrink-0 justify-center pt-0.5 pb-4 pl-8 pr-7">
            {busy === "generating" ? (
              <Button
                // Kept in step with the Images Stop control, which uses the same fill.
                className="relative z-10 h-11 px-8 hover:bg-muted dark:hover:bg-muted"
                variant="outline"
                onClick={handleCancelGenerate}
              >
                <Spinner className="mr-2 size-4" />
                Cancel
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
                  <GalleryItemMenu
                    noun="video"
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
                      genStep?.phase === "denoise" && genStep.total > 0
                        ? (genStep.step / genStep.total) * 100
                        : null
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
              {/* The card is a wrapper, not a button: the actions menu has to be the select
                  button's SIBLING, since a button inside a button is invalid and would swallow
                  its own clicks. data-clip-id rides the wrapper so the observer still sees it. */}
              {videos.map((video) => (
                <div
                  key={video.id}
                  data-clip-id={video.id}
                  className="group relative h-16 w-24 shrink-0"
                >
                <Tooltip>
                <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  onClick={() => setSelectedId(video.id)}
                  className="relative flex size-full flex-col justify-end overflow-hidden rounded-[10px] bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
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
                    {CONDITIONING_LABELS[video.conditioning ?? ""]
                      ? ` - ${CONDITIONING_LABELS[video.conditioning ?? ""]}`
                      : ""}
                  </span>
                </TooltipContent>
                </Tooltip>
                {/* Pin marker, top-left so it clears both the caption and the menu. */}
                {video.pinned && (
                  <span className="pointer-events-none absolute left-0.5 top-0.5 z-30 rounded-full bg-background/80 p-0.5 text-foreground shadow-sm ring-1 ring-border backdrop-blur">
                    <HugeiconsIcon icon={PinIcon} className="size-3" />
                  </span>
                )}
                <div className="absolute right-0.5 top-0.5 z-30">
                  <GalleryItemMenu
                    variant="overlay"
                    noun="video"
                    active={active}
                    pinned={Boolean(video.pinned)}
                    archived={Boolean(video.archived)}
                    onTogglePin={() => void handleTogglePin(video.id, !video.pinned)}
                    onToggleArchive={() => void handleArchive(video.id)}
                    onDelete={() => void handleDelete(video.id)}
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
              {/* Clear-all, tucked at the end so it never sits under a hover. */}
              {videos.length > 0 && (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      onClick={() => setClearConfirmOpen(true)}
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
