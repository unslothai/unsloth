// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { generationFailureLogsAction } from "@/features/settings/lib/view-logs-action";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ArrowExpand01Icon,
  Refresh01Icon,
  Cancel01Icon,
  Delete02Icon,
  Download01Icon,
  FlimSlateIcon,
  ImageCropIcon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { Volume02Icon } from "@/lib/volume-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { GalleryItemMenu, GalleryPinBadge } from "@/components/gallery-item-menu";
import { MediaRailResizeHandle } from "@/components/media-rail-resize-handle";
import { MediaViewer } from "@/components/media-viewer";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { MEDIA_RAIL_ROOT_ATTR, useMediaRailWidth } from "@/hooks/use-media-rail-width";
import { StripDropLine } from "@/components/gallery-strip-reorder";
import { useStripReorder } from "@/hooks/use-strip-reorder";
import { ImageDropzone } from "@/components/image-dropzone";
import { LibraryPageLink } from "@/components/media-page-link";
import { translate, useT } from "@/i18n";
import {
  chatAboutMedia,
  revealInFolder,
  useLibraryFavorites,
  useRevealLabel,
} from "@/features/library";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { videoTourSteps } from "./tour";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  applyPin,
  fetchNextPage,
  fetchWhileStable,
  moveGalleryItem,
  nextSelectedId,
  pinnedOrder,
  removeGalleryItem,
  restorePinOrder,
  serializeById,
  sortGalleryItems,
  subscribeGalleryChanged,
} from "@/lib/gallery-flags";
import {
  dismissExample,
  isExampleDismissed,
  readLastPrompt,
  saveLastPrompt,
} from "@/lib/last-prompt";

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
import { Switch } from "@/components/ui/switch";
import { useSidebar } from "@/components/ui/sidebar";
import { NegativePromptField } from "@/components/negative-prompt-field";
import { usePersistedChoice } from "@/hooks/use-persisted-choice";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import {
  explicitFamily,
  resolvedFamilyOverrideSelection,
  useFamilyOverride,
} from "@/features/model-picker/components/model-selector/family-override";
import { VIDEO_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import {
  type HostClass,
  hostOffersDensePrecision,
} from "@/features/model-picker/components/model-selector/host-artifact-policy";
import {
  VIDEO_CATALOG,
  catalogToModelOptions,
  loadSpecFor,
} from "@/features/model-picker/components/model-selector/model-catalog";
import {
  useDenseQuantSchemes,
  useHostClass,
  useNvfp4Diffusion,
  useNvfp4DiffusionKnown,
} from "@/hooks/use-host-class";
import { nvfp4SelectionFallback, withNvfp4Option } from "@/lib/nvfp4-options";
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
import { generatePhaseLabel, sameGenerateProgress } from "@/lib/media-generate-phase";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useStagedDownload } from "@/features/hub/download-manager";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { useIsMobileShell } from "@/hooks/use-mobile";
import { resolveDiffusionGgufFilename } from "@/lib/diffusion-gguf-filename";
import { createPickGuard, runGgufRepoPick } from "@/lib/diffusion-gguf-pick";
import { diffusionRoutePick } from "@/lib/diffusion-route-pick";
import { useDiffusionPickToast, usePickToastProgress } from "@/lib/use-diffusion-pick-toast";
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
import { diffusionPipelineLoadTarget, diffusionStagingEntries } from "@/lib/diffusion-pipeline-load-target";
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
import { loadGalleryUntil } from "@/lib/gallery-deep-link";
import { subscribeModelEjected } from "@/lib/model-lifecycle-events";
import { BlobUrlCache } from "@/lib/blob-url-cache";

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
  addGalleryVideoToProject,
  moveGalleryVideo,
  setGalleryVideoFlags,
  fetchGalleryVideoExport,
  fetchGalleryVideoSignedUrl,
  fetchGalleryVideoThumbnail,
  generateVideo,
  getVideoGallery,
  getVideoGenerateProgress,
  getVideoLoadProgress,
  getVideoDownloadPlan,
  getVideoStatus,
  loadVideoModel,
  unloadVideoModel,
} from "./api";
import { stopButtonLabel } from "@/features/images/lib/generation-stop";
import { type Playback, fetchWithFreshLink, playWithMutedFallback, readPlayback } from "./viewer";
import { videoThumbnailQueue, withThumbnailRetries } from "./thumbnail-request-queue";

const VIDEO_EXAMPLE_PROMPT =
  "A slow cinematic shot down a quiet Kyoto street at sunrise, cherry blossom petals drifting in the air, a shopkeeper opening a wooden storefront, warm natural light.";

// Host-dependent: a Mac gets only GGUF rows. The load kind per artifact comes from loadSpecFor.
function useVideoModels(
  host: HostClass,
  denseQuantSchemes: readonly string[],
): ModelOption[] {
  return useMemo(
    () => catalogToModelOptions(VIDEO_CATALOG, host, denseQuantSchemes),
    [host, denseQuantSchemes],
  );
}

// Matched by repo-id substring, most specific first.
const DEFAULT_GEN = { steps: 8, guidance: 1 };

const MODEL_DEFAULTS: Array<{ match: string; steps: number; guidance: number }> = [
  { match: "minimax-h3", steps: 30, guidance: 1 },
  { match: "minimax_h3", steps: 30, guidance: 1 },
  { match: "distilled", steps: 8, guidance: 1 },
  { match: "ltx", steps: 40, guidance: 4 },
  { match: "a14b", steps: 20, guidance: 3.5 },
  { match: "wan2.2-14b", steps: 20, guidance: 3.5 },
  // The backend supplies the fps per family.
  { match: "wan", steps: 20, guidance: 5 },
  { match: "hunyuanvideo", steps: 20, guidance: 6 },
];

function defaultsFor(repoId: string): { steps: number; guidance: number } {
  const id = repoId.toLowerCase();
  return MODEL_DEFAULTS.find((d) => id.includes(d.match)) ?? DEFAULT_GEN;
}

function defaultsKeyFor(repoId: string, familyOverride: string): string {
  return defaultsFor(repoId) !== DEFAULT_GEN ? repoId : (explicitFamily(familyOverride) ?? repoId);
}

// Replaced by status.defaults.resolution_presets once loaded.
const FALLBACK_RESOLUTION_PRESETS: Array<[number, number]> = [
  [768, 512],
  [1216, 704],
  [704, 1216],
];

const FALLBACK_FRAME_STEP = 8;
const FALLBACK_FRAME_OFFSET = 1;
const FALLBACK_FPS = 24;
const FALLBACK_DURATION_TARGETS = [1, 2, 3, 5];

// Module cache so a tab switch re-renders instantly.
const galleryCache: {
  videos: GalleryVideo[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  // Signed links expire and their secret is per-process while this cache survives, so re-mint.
  srcById: Map<string, { url: string; mintedAt: number }>;
  thumbnailById: BlobUrlCache;
  thumbnailInflight: Map<string, Promise<boolean>>;
  thumbnailFailed: Set<string>;
  // Re-minted once after a media error, so a broken clip cannot loop.
  refreshed: Set<string>;
  inflight: Set<string>;
  // Deleted mid-mint, so a late reply is not cached. Clear-all bumps the epoch instead.
  deleted: Set<string>;
  /** A terminal progress snapshot taken before the archive must not return the clip to the strip. */
  archived: Set<string>;
  epoch: number;
} = {
  videos: [],
  hasMore: false,
  selectedId: null,
  quant: null,
  srcById: new Map(),
  thumbnailById: new BlobUrlCache(32 * 1024 * 1024),
  thumbnailInflight: new Map(),
  thumbnailFailed: new Set(),
  refreshed: new Set(),
  inflight: new Set(),
  deleted: new Set(),
  archived: new Set(),
  epoch: 0,
};

// Comfortably inside the backend's own expiry.
const VIDEO_LINK_REFRESH_MS = 6 * 60 * 60 * 1000;

const PAGE_SIZE = 50;

const INLINE_PLAYBACK: Playback = { time: 0, playing: false, muted: true, volume: 1 };

// Extra passes only happen when pagination moved mid-fetch.
const RESYNC_MAX_ATTEMPTS = 3;

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

// MP4 streams from its signed link: cross-origin under Tauri anchors do not save, and clips are
// too big for memory. WebM / GIF are transcoded by the backend.
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

const CONDITIONING_LABELS: Record<string, string> = {
  i2va: "From start frame",
  l2va: "To end frame",
  fl2va: "Start to end frame",
  ref2va: "From references",
};

function clipMeta(video: GalleryVideo): string {
  const secs = video.duration_s > 0 ? `${video.duration_s.toFixed(1)}s` : `${video.num_frames}f`;
  return `${secs} · ${video.width}×${video.height}`;
}

function genStepLabel(p: VideoGenerateProgress, hasAudio: boolean): string {
  return generatePhaseLabel(p, { hasAudio, formatEta });
}

const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

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

// `onCancel` is the only control that reaches a load in flight; the eject is hidden then.
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

// "Auto: X" when the backend decided, warning tone when an explicit request was declined.
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

/** Read from status, never the request; a declined control's reason is in the tooltip. */
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
            : denseTransformerBuildLabel(status)
        }
        badge={<ResolvedBadge status={status} controlKey="transformer_quant" />}
      />
      <BuildRow
        label="Text encoder"
        value={
          status.text_encoder_quant
            ? formatResolvedValue("text_encoder_quant", status.text_encoder_quant)
            // On the native engine no TE quant is not the same as bf16.
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

function reportLoadFailure(message: string | null | undefined, fallback: string): void {
  const text = (message || "").trim();
  if (text && isPrecisionRefusal(text)) {
    toast.error(PRECISION_REFUSAL_TITLE, { description: text });
    return;
  }
  toast.error(text || fallback);
}

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
          <SelectTrigger className="h-8 w-[calc(160px*var(--ui-space-scale,1))] max-sm:w-[min(calc(160px*var(--ui-space-scale,1)),50vw)] text-xs">
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

function RecipePopover({
  video,
  onRestore,
  active,
}: {
  video: GalleryVideo;
  onRestore: (video: GalleryVideo) => void;
  active: boolean;
}) {
  // PopoverContent portals to body, so the inert page wrapper cannot contain it.
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
      <PopoverContent
        align="end"
        side="top"
        collisionPadding={12}
        className="flex max-h-[var(--radix-popover-content-available-height)] w-80 flex-col gap-0 overflow-hidden p-0"
      >
        <div className="shrink-0 border-b border-border/60 px-4 py-2.5">
          <p className="text-sm font-semibold">Generation settings</p>
          <p className="text-ui-11 text-muted-foreground">{formatTimestamp(video.created_at)}</p>
        </div>
        <div className="flex min-h-0 flex-col gap-2 overflow-y-auto overscroll-contain px-4 py-3 text-xs">
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
              precision that did not run. */}
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
        <div className="shrink-0 border-t border-border/60 px-3 py-2.5">
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
  displayRepoId?: string;
};
/** Carries the deferred loadOrStage arguments so the H3 partition choice only adds `h3Task`. */
type PendingH3Load = {
  repoId: string;
  opts: VideoLoadOptions;
  source: ModelSelectorChangeMeta["source"];
  token: number;
  familyOverrideRequired: boolean;
};

const H3_BF16_REPO = "MiniMaxAI/MiniMax-H3";

/** Both entry points need this. On-device copies are matched on the last path segment or an
 * explicit family, since a Hub-id equality test misses them. */
function isH3PipelinePick(repoId: string, kind: VideoLoadOptions["kind"], familyOverride?: string): boolean {
  if (kind !== "pipeline") return false;
  if (familyOverride?.trim().toLowerCase() === "minimax-h3") return true;
  const id = repoId.toLowerCase();
  if (id === H3_BF16_REPO.toLowerCase()) return true;
  const leaf = id.replace(/\\/g, "/").replace(/\/+$/, "").split("/").at(-1) ?? "";
  return leaf === H3_BF16_REPO.split("/")[1].toLowerCase();
}

// What a pick optimistically replaced, so a load that never takes can put it all back. The
// quant label and the recipe move together at pick time, so they roll back together.
type PickRevert = {
  prev: string | null;
  steps: number;
  guidance: number;
  commitRecipeClaim?: () => void;
  releaseRecipeClaim?: () => void;
  // A field the user changed after the pick is theirs, not ours to put back.
  appliedSteps?: number;
  appliedGuidance?: number;
  modelSeeded?: boolean;
  familySeeded?: boolean;
};
type VideoLoadAdvanced = Pick<
  VideoLoadRequest,
  | "memory_mode"
  | "speed_mode"
  | "attention_backend"
  | "transformer_cache"
  | "transformer_quant"
  | "family_override"
  | "gpu_ids"
>;

function VideoGate({ children }: { children: ReactNode }) {
  return (
    <div className="diffusion-surface flex h-full min-h-0 min-w-0 flex-1 flex-col items-center justify-center gap-3 pt-[var(--studio-content-top-inset,0px)] text-center text-sm text-muted-foreground">
      {children}
    </div>
  );
}

/** The root guard never bounces /video: Apple Silicon may be chat-only yet run video, so the
 * page decides from /api/system/hardware. */
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

  // Only an explicit false hides the generator; older backends send null.
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
  const t = useT();
  const initialReadySent = useRef(false);
  const isMobileShell = useIsMobileShell();
  const { pinned } = useSidebar();
  const hostClass = useHostClass();
  const denseQuantSchemes = useDenseQuantSchemes();
  const nvfp4Diffusion = useNvfp4Diffusion();
  const nvfp4DiffusionKnown = useNvfp4DiffusionKnown();
  const videoModels = useVideoModels(hostClass, denseQuantSchemes);
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompt, setPrompt] = useState(() => readLastPrompt("video"));
  const [exampleDismissed, setExampleDismissed] = useState(() => isExampleDismissed("video"));
  const [negativePrompt, setNegativePrompt] = useState("");
  const [negativeOpen, setNegativeOpen] = useState(false);
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const modelSeeded = useRef(false);
  const familySeeded = useRef(false);
  // A preset chosen during the download is newer, so the pick's defaults and rollback must not land.
  const pickRecipeSuperseded = useRef<(() => boolean) | null>(null);
  const revertPick = useCallback((r: PickRevert) => {
    setQuant(r.prev);
    setPendingModelDefaults(null);
    // Equality cannot tell untouched from the same number chosen again.
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
  // Without this the Default preset reads as "modified" for the whole download.
  const [pendingModelDefaults, setPendingModelDefaults] = useState<{
    steps: number;
    guidance: number;
  } | null>(null);
  const [seed, setSeed] = useState("");
  const [resolutionIdx, setResolutionIdx] = useState(0);
  const [resolutionIntent, setResolutionIntent] = useState<[number, number]>(
    FALLBACK_RESOLUTION_PRESETS[0]!,
  );
  const [firstFrame, setFirstFrame] = useState<string | null>(null);
  const [lastFrame, setLastFrame] = useState<string | null>(null);
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
  const [flowShift, setFlowShift] = useState<number | null>(null);
  const [audioFlowShift, setAudioFlowShift] = useState<number | null>(null);
  const [numFrames, setNumFrames] = useState(
    FALLBACK_FRAME_STEP * 3 + FALLBACK_FRAME_OFFSET,
  );
  const [durationIntentSeconds, setDurationIntentSeconds] = useState(
    numFrames / FALLBACK_FPS,
  );
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_video_advanced_open",
  );
  const [livePreviewOff, setLivePreviewOff] = usePersistedToggle("unsloth_video_live_preview_off");
  const livePreview = !livePreviewOff;
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  // Persisted because status names the device but not which card; a stored id is only a hint.
  const [selectedGpu, setSelectedGpu] = usePersistedChoice(
    "unsloth_video_gpu_choice",
    "auto",
  );
  const gpuChoices = useDiffusionGpuChoices();
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [attentionBackend, setAttentionBackend] = useState<
    "auto" | "native" | "cudnn" | "flash3" | "sage"
  >("auto");
  const [transformerCache, setTransformerCache] = useState<"auto" | "off" | "fbcache" | "static">("auto");
  const [transformerQuant, setTransformerQuant] = useState<
    "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8"
  >("auto");
  useEffect(() => {
    setTransformerQuant((v) => nvfp4SelectionFallback(v, nvfp4DiffusionKnown, nvfp4Diffusion));
  }, [nvfp4Diffusion, nvfp4DiffusionKnown, transformerQuant]);
  const lastLoad = useRef<({ repoId: string } & VideoLoadOptions) | null>(null);
  const [canReapply, setCanReapply] = useState(false);

  const [busy, setBusy] = useState<Busy>(null);
  const [stopping, setStopping] = useState(false);
  useEffect(() => {
    if (busy !== "generating") setStopping(false);
  }, [busy]);
  const [genStep, setGenStep] = useState<VideoGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // Background tabs clamp setInterval, so returning fires one immediate poll.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<VideoStatus | null>(null);
  const { familyOverride, setFamilyOverride, familySelect, opaqueKind, selectorModelId } = useFamilyOverride(status, status?.supported_families);
  // Controlled so the body-portaled selector force-closes when off-tab.
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [pendingH3Load, setPendingH3Load] = useState<PendingH3Load | null>(null);
  const tour = useGuidedTourController({
    id: "video",
    steps: videoTourSteps,
    enabled: active,
  });
  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  const [videos, setVideos] = useState<GalleryVideo[]>(() => galleryCache.videos);
  const { rootStyle: railRootStyle } = useMediaRailWidth("video");
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [clearingGallery, setClearingGallery] = useState(false);
  // Radix does not call onOpenChange for a parent-forced close, so reset during render.
  if (!active && clearConfirmOpen) setClearConfirmOpen(false);
  // 3 plays per selected clip, then pause.
  const playCountRef = useRef(0);
  useEffect(() => {
    playCountRef.current = 0;
  }, [selectedId]);
  // The keep-alive layout only hides the page, and display:none does not pause media.
  const previewRef = useRef<HTMLVideoElement | null>(null);
  const activeRef = useRef(active);
  useEffect(() => {
    activeRef.current = active;
    if (!active) previewRef.current?.pause();
  }, [active]);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    Object.fromEntries([...galleryCache.srcById].map(([id, e]) => [id, e.url])),
  );
  const [thumbnailById, setThumbnailById] = useState<Record<string, string>>(() =>
    galleryCache.thumbnailById.toRecord(),
  );
  const [thumbnailFailedIds, setThumbnailFailedIds] = useState<ReadonlySet<string>>(
    () => new Set(galleryCache.thumbnailFailed),
  );
  const visibleThumbnailIds = useRef(new Set<string>());
  const loadingMore = useRef(false);
  // The page stays mounted across tab switches, so only a true unmount flips this.
  const isMounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loadToastId = useRef<string | number | null>(null);
  const lastLoadSig = useRef<string | null>(null);
  // Carries the pick's step/guidance recipe too, or a cancelled pick leaves it on the resident model.
  const quantRevert = useRef<PickRevert | null>(null);
  // Staging does not set `busy`, so a second pick can overwrite quantRevert mid-plan.
  const stagedQuantRevert = useRef<PickRevert | null>(null);
  const pickSeq = useRef(0);
  // handleLoad overwrites lastLoad at start, and a failure leaves the old model.
  const lastLoadRevert = useRef<{ prev: typeof lastLoad.current; canReapply: boolean } | null>(null);
  // Resolving and staging do not set `busy`. Lazy state, since a ref cannot be written in render.
  const [pickGuard] = useState(createPickGuard);

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);
  const pickToast = useDiffusionPickToast();

  // A ref keeps onClick stable; the toast is built before handleCancelLoad exists.
  const cancelLoadRef = useRef<() => void>(() => {});
  const cancelLoadFromToast = useCallback(() => cancelLoadRef.current(), []);
  // Requests awaiting a response compare against this and discard their result.
  const cancelSeq = useRef(0);
  // The compensating unload carries no identity, so it must not fire once a newer load owns the page.
  const loadSeq = useRef(0);
  // begin_load refuses a second live load, so a new pick must wait for this to settle.
  const pendingStart = useRef<Promise<unknown> | null>(null);

  // The compensating unload failed, so the load is STILL running; handleUnload reports that.
  const loadTrackingRestored = useRef(false);

  // Shared with the indicator eject.
  const dropResidentState = useCallback(() => {
    // Cancel, not release, or a resolving pick would reload what was just ejected.
    pickGuard.cancel();
    pickToast.dismissAll();
    // Clearing the timer stops the next tick, not a request awaiting its response.
    cancelSeq.current += 1;
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    // Leaving this set would let Reapply reload the freed model.
    lastLoad.current = null;
    setCanReapply(false);
    // Stopping the poll also stops the branch that hands back a pick that never loaded.
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [dismissLoadToast, pickGuard, pickToast, revertPick]);

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
  const livePreviewSrc =
    busy === "generating" && livePreview ? (genStep?.preview ?? undefined) : undefined;
  const [viewer, setViewer] = useState<{ id: string; from: Playback } | null>(null);
  const viewerVideoRef = useRef<HTMLVideoElement | null>(null);
  const viewerPositioned = useRef(false);
  const handback = useRef<{ id: string; playback: Playback } | null>(null);
  const navigateToChat = useNavigate();
  const revealLabel = useRevealLabel();
  const viewerVideo = viewer ? (videos.find((video) => video.id === viewer.id) ?? null) : null;
  const viewerSrc = viewerVideo ? srcById[viewerVideo.id] : undefined;
  if (viewer && (!active || !viewerVideo)) setViewer(null);
  const openViewer = () => {
    if (!selected || !selectedSrc) return;
    const inline = previewRef.current;
    viewerPositioned.current = false;
    handback.current = null;
    setViewer({ id: selected.id, from: readPlayback(inline, INLINE_PLAYBACK) });
    inline?.pause();
  };
  const recordViewer = (video: HTMLVideoElement) => {
    if (!viewer || video !== viewerVideoRef.current) return;
    handback.current = {
      id: viewer.id,
      playback: readPlayback(video, viewer.from, viewerPositioned.current),
    };
  };
  const closeViewer = () => {
    if (viewerVideoRef.current) recordViewer(viewerVideoRef.current);
    setViewer(null);
  };
  const shownId = selected?.id;
  useEffect(() => {
    const last = handback.current;
    if (viewer || !last || last.id !== shownId) return;
    const inline = previewRef.current;
    if (!selectedSrc || !inline) return;
    handback.current = null;
    const { playback } = last;
    inline.currentTime = playback.time;
    inline.muted = playback.muted;
    inline.volume = playback.volume;
    if (playback.playing && activeRef.current) void playWithMutedFallback(inline);
    else inline.pause();
  }, [viewer, shownId, selectedSrc]);

  const resolutionPresets = useMemo<Array<[number, number]>>(() => {
    const presets = status?.defaults?.resolution_presets;
    if (presets && presets.length > 0) {
      return presets.map((p) => [p[0], p[1]] as [number, number]);
    }
    return FALLBACK_RESOLUTION_PRESETS;
  }, [status?.defaults?.resolution_presets]);
  const livePreviewPreset = resolutionPresets[resolutionIdx] ?? resolutionPresets[0];
  const livePreviewBox = livePreviewPreset
    ? { maxWidth: livePreviewPreset[0], maxHeight: livePreviewPreset[1] }
    : undefined;

  const frameStep = status?.defaults?.frame_step ?? FALLBACK_FRAME_STEP;
  const frameOffset = status?.defaults?.frame_offset ?? FALLBACK_FRAME_OFFSET;
  const fps = status?.defaults?.fps ?? FALLBACK_FPS;
  const durationTargets =
    status?.defaults?.duration_presets ?? FALLBACK_DURATION_TARGETS;

  // Frame counts closest to ~1s/2s/3s/5s at the current fps, deduped.
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

  useEffect(() => {
    setResolutionIdx((idx) =>
      idx === MATCH_SOURCE_RESOLUTION || idx < resolutionPresets.length ? idx : 0,
    );
  }, [resolutionPresets.length]);

  const supportsKeyframes = status?.supports_keyframes === true;
  // First when present, else last, matching the backend.
  const canvasKeyframe = firstFrame ?? lastFrame;

  const supportsReferences = status?.supports_references === true;
  const hasReferenceRoom = hasReferenceCapacity(
    referenceImages.length,
    referenceVideos.length,
    referenceAudios.length,
  );
  // Only Diffusers supports the 2048px reference policy.
  const canPickReferenceSize = supportsReferences && status?.engine !== "sd_cpp";

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
      // A negative prompt in effect must be visible, as in restoreSettings.
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
    (repoId: string, effectiveFamilyOverride = familyOverride) => {
      const revert = quantRevert.current;
      if (revert && !revert.releaseRecipeClaim) {
        const claim = claimVideoRecipe();
        revert.commitRecipeClaim = claim.commit;
        revert.releaseRecipeClaim = claim.release;
      }
      // Baselined per pick, including one that inherits an earlier pick's rollback.
      const claimedAt = videoFormClaimId();
      pickRecipeSuperseded.current = () => videoFormClaimId() !== claimedAt;
      const recommended = defaultsFor(defaultsKeyFor(repoId, effectiveFamilyOverride));
      setPendingModelDefaults(recommended);
      setSteps(recommended.steps);
      setGuidance(recommended.guidance);
      if (revert) {
        revert.modelSeeded ??= modelSeeded.current;
        revert.familySeeded ??= familySeeded.current;
        revert.appliedSteps = recommended.steps;
        revert.appliedGuidance = recommended.guidance;
      }
      // revertPick restores both markers so a saved recipe can still outrank a discovered resident.
      modelSeeded.current = true;
      familySeeded.current = true;
    },
    [claimVideoRecipe, familyOverride, videoFormClaimId],
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

  // Select "match source" only after the keyframe passes the aspect check.
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
    // Apply the family's default clip length, or every default run stays a ~1s clip.
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

  // On mount with a model loaded only refreshStatus runs, so seed from status. Keyed on the schedule,
  // since a GGUF repo holds several variants.
  const loadedModelKey = status?.loaded
    ? `${status.repo_id ?? ""}|${defaultSteps ?? ""}|${defaultGuidance ?? ""}|${defaultFlowShift ?? ""}|${defaultAudioFlowShift ?? ""}`
    : null;
  const prevLoadedModelRef = useRef<string | null>(null);
  useEffect(() => {
    const modelChanged = loadedModelKey !== prevLoadedModelRef.current;
    prevLoadedModelRef.current = loadedModelKey;
    if (modelChanged && loadedModelKey && defaultSteps != null && defaultGuidance != null) {
      setPendingModelDefaults(null);
      // A stored recipe outranks model defaults on the first seed.
      const applyDefaults = shouldApplyModelDefaults(
        modelSeeded.current,
        videoPresets.storedRecipe,
        pickRecipeSuperseded.current?.() ?? false,
      );
      // This status confirms the pick. Runs after the family effect on the same status.
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

  // Reseed from the LOADED build so declined requests snap back. Keyed on the load-time half only:
  // transformer_cache is rewritten at generation time.
  const resolvedKey = status?.loaded ? resolvedSeedKey(status.resolved) : null;
  useEffect(() => {
    const record = status?.loaded ? status.resolved : null;
    if (!record) return;
    const family = resolvedFamilyOverrideSelection(record.family_override);
    if (family) setFamilyOverride(family);
    const quant = resolvedSelectValue(record.transformer_quant, (v) =>
      // The engaged value spells "no quant" as "off"; the option is "none".
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
      // Map the dispatcher's name back to the option.
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
      // Deleted or cleared while minting; caching would strand an entry.
      if (galleryCache.deleted.has(video.id) || galleryCache.epoch !== epochAtStart) return;
      galleryCache.srcById.set(video.id, { url, mintedAt: Date.now() });
      if (isMounted.current) setSrcById((prev) => ({ ...prev, [video.id]: url }));
    } catch {
      // Leave it without a src; the card shows a placeholder.
    } finally {
      galleryCache.inflight.delete(video.id);
    }
  }, []);

  // The selected clip's card is usually off-strip, so its poster would evict first.
  const protectedThumbnailIds = useCallback((extra?: string) => {
    const keep = new Set(visibleThumbnailIds.current);
    if (galleryCache.selectedId) keep.add(galleryCache.selectedId);
    if (extra) keep.add(extra);
    return keep;
  }, []);

  // On visibility too: a fully cached strip fetches nothing.
  const pruneThumbnails = useCallback(() => {
    const evicted = galleryCache.thumbnailById.prune(protectedThumbnailIds());
    if (evicted.length === 0 || !isMounted.current) return;
    setThumbnailById((prev) => {
      const next = { ...prev };
      for (const id of evicted) delete next[id];
      return next;
    });
  }, [protectedThumbnailIds]);

  const ensureThumbnail = useCallback((video: GalleryVideo): Promise<boolean> => {
    const cached = galleryCache.thumbnailById.get(video.id);
    if (cached) {
      galleryCache.thumbnailById.touch(video.id);
      return Promise.resolve(true);
    }
    if (galleryCache.thumbnailFailed.has(video.id)) return Promise.resolve(false);
    const existing = galleryCache.thumbnailInflight.get(video.id);
    if (existing) return existing;
    const epochAtStart = galleryCache.epoch;
    // Skip decoding once the record was deleted or the gallery cleared.
    const stale = () => galleryCache.deleted.has(video.id) || galleryCache.epoch !== epochAtStart;
    const request = (async () => {
      try {
        // Retried because the undecodable marker is permanent for the session.
        const fetched = await withThumbnailRetries(() =>
          videoThumbnailQueue.run(() =>
            stale() ? Promise.resolve(null) : fetchGalleryVideoThumbnail(video.id),
          ),
        );
        if (!fetched) return false;
        if (stale()) {
          URL.revokeObjectURL(fetched.url);
          return false;
        }
        galleryCache.thumbnailById.set(video.id, fetched.url, fetched.bytes);
        // Unprotected, an oversized poster evicts every neighbour, then itself.
        const evicted = galleryCache.thumbnailById.prune(protectedThumbnailIds(video.id));
        if (isMounted.current) {
          setThumbnailById((prev) => {
            const next = { ...prev, [video.id]: fetched.url };
            for (const id of evicted) delete next[id];
            return next;
          });
        }
        return true;
      } catch {
        galleryCache.thumbnailFailed.add(video.id);
        if (isMounted.current) {
          setThumbnailFailedIds(new Set(galleryCache.thumbnailFailed));
        }
        return false;
      } finally {
        galleryCache.thumbnailInflight.delete(video.id);
      }
    })();
    galleryCache.thumbnailInflight.set(video.id, request);
    return request;
  }, [protectedThumbnailIds]);

  // Terminal: the cache hit ends retries, and refetching rejected bytes spins.
  const handlePosterError = useCallback((id: string) => {
    galleryCache.thumbnailById.delete(id);
    galleryCache.thumbnailFailed.add(id);
    if (!isMounted.current) return;
    setThumbnailById((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setThumbnailFailedIds(new Set(galleryCache.thumbnailFailed));
  }, []);

  // A media error on a playing clip means its link died early (the server restarted, changing
  // its signing secret). Re-mint once per clip per session.
  const remintSrc = useCallback(
    (video: GalleryVideo) => {
      if (galleryCache.refreshed.has(video.id)) return;
      galleryCache.refreshed.add(video.id);
      galleryCache.srcById.delete(video.id);
      void ensureSrc(video);
    },
    [ensureSrc],
  );

  // Posters, not video elements: WebKit creates decoders and queues per element even with
  // preload="metadata".
  const stripRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const root = stripRef.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    // disconnect() delivers no final entry, so a resynced-away id would protect its blob forever.
    const listed = new Set(videos.map((v) => v.id));
    let stranded = false;
    for (const id of visibleThumbnailIds.current) {
      if (listed.has(id)) continue;
      visibleThumbnailIds.current.delete(id);
      stranded = true;
    }
    if (stranded) pruneThumbnails();
    const io = new IntersectionObserver(
      (entries) => {
        let left = false;
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.clipId;
          if (!id) continue;
          if (!entry.isIntersecting) {
            if (visibleThumbnailIds.current.delete(id)) left = true;
            continue;
          }
          visibleThumbnailIds.current.add(id);
          galleryCache.thumbnailById.touch(id);
          const clip = videos.find((v) => v.id === id);
          if (clip) void ensureThumbnail(clip);
        }
        if (left) pruneThumbnails();
      },
      // rootMargin applies to the root box, so the root must be the horizontally scrolling strip.
      { root, rootMargin: "0px 600px" },
    );
    for (const card of root.querySelectorAll("[data-clip-id]")) io.observe(card);
    return () => io.disconnect();
  }, [videos, ensureThumbnail, pruneThumbnails]);

  // The preview player is what the user watches, so the selected clip is fetched whether or not its card is on screen.
  useEffect(() => {
    if (!selected) return;
    void ensureThumbnail(selected);
    void ensureSrc(selected);
  }, [selected, ensureSrc, ensureThumbnail]);

  // Bumped by every LOCAL change to the strip. A resync started before one holds a snapshot the
  // server listing cannot reconcile with what the user just did, so it drops it.
  const stripEpoch = useRef(0);
  // Bumped when the window grows from the server; a resync refetches rather than drops.
  const pageEpoch = useRef(0);
  // Only the newest resync may apply.
  const resyncSeq = useRef(0);
  // The epoch is an EDGE; a page applies only while no shelf mutation is in flight.
  const pendingShelfMutations = useRef(0);

  const loadGallery = useCallback(async () => {
    try {
      // Fenced: tiles stay actionable during the load, so a pre-pin snapshot would undo an action.
      const page = await fetchWhileStable(
        () => stripEpoch.current,
        () => getVideoGallery(0, PAGE_SIZE),
      );
      if (!page) return;
      pageEpoch.current += 1;
      // Otherwise one backend restart during an open bricks every poster.
      if (galleryCache.thumbnailFailed.size > 0) {
        galleryCache.thumbnailFailed.clear();
        setThumbnailFailedIds(new Set());
      }
      galleryCache.videos = page.videos;
      galleryCache.hasMore = page.has_more;
      setVideos(page.videos);
      setHasMore(page.has_more);
      // Without IntersectionObserver (jsdom / old webview) keep the eager poster fetch.
      if (typeof IntersectionObserver === "undefined") {
        page.videos.forEach((video) => void ensureThumbnail(video));
      }
    } catch {
      // Best-effort: a failed gallery load should not block the page.
    }
  }, [ensureThumbnail]);

  const loadMore = useCallback(async () => {
    if (loadingMore.current || !galleryCache.hasMore) return;
    loadingMore.current = true;
    try {
      // An archive landing across this GET shifts a clip past the page boundary.
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
        page.videos.forEach((video) => void ensureThumbnail(video));
      }
    } catch {
      // transient; the user can scroll again to retry
    } finally {
      loadingMore.current = false;
    }
  }, [ensureThumbnail]);

  // WebM/GIF transcode can take seconds and 501s without the codec.
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

  const handleQuickDownload = useCallback(
    async (video: GalleryVideo) => {
      const cached = galleryCache.srcById.get(video.id);
      let url =
        cached && Date.now() - cached.mintedAt < VIDEO_LINK_REFRESH_MS ? cached.url : null;
      if (!url) {
        try {
          url = await fetchGalleryVideoSignedUrl(video.id);
        } catch (err) {
          toast.error("Could not save video", {
            description: err instanceof Error ? err.message : undefined,
          });
          return;
        }
      }
      await handleDownload(url, video, "mp4");
    },
    [handleDownload],
  );

  // `discardLink` is for a real delete; an archived clip keeps its link.
  const dropFromStrip = useCallback((id: string, discardLink: boolean) => {
    visibleThumbnailIds.current.delete(id);
    if (discardLink) {
      galleryCache.srcById.delete(id);
      galleryCache.thumbnailById.delete(id);
      galleryCache.thumbnailFailed.delete(id);
      galleryCache.refreshed.delete(id);
      galleryCache.deleted.add(id);
      setSrcById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setThumbnailById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setThumbnailFailedIds(new Set(galleryCache.thumbnailFailed));
    }
    stripEpoch.current += 1;
    // Avoid a side effect inside a setVideos updater.
    const at = galleryCache.videos.findIndex((v) => v.id === id);
    const next = removeGalleryItem(galleryCache.videos, id);
    galleryCache.videos = next;
    setVideos(next);
    setSelectedId((cur) => nextSelectedId(next, id, cur, at));
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      // Held for the round trip: the server shortens the shelf while processing this.
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

  /** Unpinning can drop a clip past the window and promote an unloaded one into it. */
  const resyncWindow = useCallback(
    async (count: number, stillFresh?: () => boolean) => {
      const ticket = (resyncSeq.current += 1);
      for (let attempt = 0; attempt < RESYNC_MAX_ATTEMPTS; attempt += 1) {
        const paged = pageEpoch.current;
        // Sized against the live window so a page appended meanwhile is covered.
        const wanted = Math.max(count, galleryCache.videos.length, PAGE_SIZE);
        const collected: GalleryVideo[] = [];
        let more = false;
        while (collected.length < wanted) {
          // Only the remainder, not a whole page.
          const page = await getVideoGallery(
            collected.length,
            Math.min(PAGE_SIZE, wanted - collected.length),
          );
          collected.push(...page.videos);
          more = page.has_more;
          if (!page.has_more || page.videos.length === 0) break;
        }
        // Checked here because the window is applied before this returns.
        if (stillFresh && !stillFresh()) return;
        if (resyncSeq.current !== ticket) return;
        // Pagination moved: only server data, so take another pass.
        if (pageEpoch.current !== paged) continue;
        galleryCache.videos = collected;
        galleryCache.hasMore = more;
        setVideos(collected);
        setHasMore(more);
        if (typeof IntersectionObserver === "undefined") {
          collected.forEach((video) => void ensureThumbnail(video));
        }
        return;
      }
    },
    [ensureThumbnail],
  );

  // The page stays mounted, so resync on archive restore; loadGallery would cut to page one.
  useEffect(
    () =>
      subscribeGalleryChanged("videos", () => {
        // Bumped first so in-flight reads are discarded.
        stripEpoch.current += 1;
        // A generation or new page landing meanwhile would be overwritten by an older snapshot.
        const epoch = stripEpoch.current;
        void resyncWindow(
          galleryCache.videos.length,
          () => stripEpoch.current === epoch,
        ).catch(() => void loadGallery());
      }),
    [loadGallery, resyncWindow],
  );

  // The last clicked pin state per id, so a slow failure cannot roll back a later success.
  const { isFavorite, toggleFavorite } = useLibraryFavorites();
  const pinAttempt = useRef(new Map<string, number>());
  const pinSeq = useRef(0);

  const handleTogglePin = useCallback(
    async (id: string, pinned: boolean) => {
      const loadedCount = galleryCache.videos.length;
      // A failed unpin returns the clip to its old position, not the front.
      const orderBefore = pinnedOrder(galleryCache.videos);
      // A per-attempt token: pin, unpin, pin stores true twice.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      const epoch = stripEpoch.current;
      setVideos((prev) => {
        const next = applyPin(prev, id, pinned);
        galleryCache.videos = next;
        return next;
      });
      try {
        // One queue: the server stamps `pinned_at` when it runs, so concurrent requests could reorder.
        await serializeById("video-pin", () => setGalleryVideoFlags(id, { pinned }));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to pin video");
        // Roll back only while this is still the user's latest intent.
        if (pinAttempt.current.get(id) === attempt) {
          pinAttempt.current.delete(id);
          stripEpoch.current += 1;
          setVideos((prev) => {
            const next = pinned
              ? applyPin(prev, id, false)
              : restorePinOrder(prev, id, orderBefore);
            galleryCache.videos = next;
            return next;
          });
        }
        return;
      }
      if (pinAttempt.current.get(id) !== attempt) return;
      pinAttempt.current.delete(id);
      // Only unpinning can open a gap in the window.
      if (!pinned && loadedCount > 0) {
        try {
          // Fenced against a pin clicked while this GET is in flight.
          await resyncWindow(loadedCount, () => stripEpoch.current === epoch);
        } catch {
          // Best-effort: the strip may be short one clip until a reload.
        }
      }
    },
    [resyncWindow],
  );

  const handleMove = useCallback(
    async (id: string, afterId: string | null) => {
      const next = moveGalleryItem(galleryCache.videos, id, afterId);
      if (next === galleryCache.videos) return;
      const guessedPinned = Boolean(next.find((i) => i.id === id)?.pinned);
      // Takes a pin token too, so a later pin is not undone by this response.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      galleryCache.videos = next;
      setVideos(next);
      try {
        // Shares the pin queue, since both rewrite the order.
        const record = await serializeById("video-pin", () => moveGalleryVideo(id, afterId));
        if (pinAttempt.current.get(id) !== attempt) return;
        pinAttempt.current.delete(id);
        setVideos((prev) => {
          const patched = prev.map((i) =>
            i.id === id ? { ...i, pinned: record.pinned, order_at: record.order_at } : i,
          );
          const out =
            Boolean(record.pinned) === guessedPinned ? patched : sortGalleryItems(patched);
          galleryCache.videos = out;
          return out;
        });
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to move video");
        stripEpoch.current += 1;
        const epoch = stripEpoch.current;
        try {
          await resyncWindow(galleryCache.videos.length, () => stripEpoch.current === epoch);
        } catch {
          void loadGallery();
        }
      }
    },
    [resyncWindow, loadGallery],
  );
  const stripReorder = useStripReorder((id, afterId) => void handleMove(id, afterId));

  const handleArchive = useCallback(
    async (id: string) => {
      // Held for the round trip: the server shortens the shelf while processing this.
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
      galleryCache.thumbnailById.clear();
      galleryCache.thumbnailFailed.clear();
      // Else a regenerated id joins a fenced promise and spins forever.
      galleryCache.thumbnailInflight.clear();
      visibleThumbnailIds.current.clear();
      galleryCache.refreshed.clear();
      // Discard every in-flight mint, listed or not.
      galleryCache.epoch += 1;
      stripEpoch.current += 1;
      galleryCache.videos = [];
      galleryCache.hasMore = false;
      galleryCache.selectedId = null;
      setSrcById({});
      setThumbnailById({});
      setThumbnailFailedIds(new Set());
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
      const presetIdx = resolutionPresets.findIndex(
        ([w, h]) => w === video.width && h === video.height,
      );
      if (presetIdx >= 0) {
        setResolutionIntent([video.width, video.height]);
        setResolutionIdx(presetIdx);
      }
      if (durationOptions.some((o) => o.frames === video.num_frames)) {
        setDurationIntentSeconds(video.num_frames / fps);
        setNumFrames(video.num_frames);
      }
      toast.success("Settings restored to inputs");
    },
    [resolutionPresets, durationOptions, fps],
  );

  // No periodic poll corrects a stale read, so only the newest ticket may write.
  const statusTicket = useRef(0);
  const setStatusIfNewest = useCallback(
    (ticket: number, next: VideoStatus) => {
      if (ticket === statusTicket.current) setStatus(next);
    },
    [],
  );

  // null when the read failed or was superseded.
  const refreshStatus = useCallback(async (): Promise<VideoStatus | null> => {
    const ticket = ++statusTicket.current;
    try {
      const next = await getVideoStatus();
      setStatusIfNewest(ticket, next);
      return ticket === statusTicket.current ? next : null;
    } catch {
      return null;
    }
  }, [setStatusIfNewest]);

  // An idle auto-unload frees the runtime silently, so re-read after a refusal or every retry 409s.
  const resyncAfterGenerateRefusal = useCallback(async () => {
    // /video/status reports committed state, so it says not loaded for a just-started load;
    // loadSeq fences that.
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

  // The model may have been evicted while off-tab.
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
          if (initialSelection) {
            void ensureThumbnail(initialSelection);
            await ensureSrc(initialSelection);
          }
        })(),
      ]);
      if (cancelled || initialReadySent.current) return;
      initialReadySent.current = true;
      onInitialReady?.();
    })();
    return () => {
      cancelled = true;
    };
  }, [active, ensureSrc, ensureThumbnail, loadGallery, onInitialReady, refreshStatus]);

  useEffect(() => {
    const repoId = status?.loaded ? status.repo_id : null;
    if (!repoId || lastLoad.current || status?.model_kind !== "pipeline") return;
    const h3Task = status.h3_task === "fl2va" || status.h3_task === "ref2va" ? status.h3_task : undefined;
    lastLoad.current = { repoId, kind: "pipeline", displayRepoId: status.display_repo_id ?? undefined, h3Task };
    setCanReapply(true);
  }, [status?.display_repo_id, status?.h3_task, status?.loaded, status?.model_kind, status?.repo_id]);

  // The indicator eject skips handleUnload, so mirror it here minus the unload.
  useEffect(
    () =>
      subscribeModelEjected("video", () => {
        dropResidentState();
        // That eject cancelled the replacement load, and its progress poll is the only thing that clears
        // `busy`, which dropResidentState just stopped; leaving it set locks the page. Narrowed to
        // "loading" so a generation is left alone, and held until the start settles.
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

  // Collapse the body-ported model selector when leaving the tab so returning to /video does not pop it back open.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
  }, [active]);

  // Poll load-progress until the background load reaches "ready" or "error", updating the persistent toast in place.
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
          // Cancelled while this read was in flight, so it describes a pipeline being torn down. Drop
          // it and refresh NOTHING: the unload's own response is authoritative.
          return;
        }
        setStatusIfNewest(ticket, loaded);
        toast.success("Model loaded");
        setBusy(null);
        quantRevert.current?.commitRecipeClaim?.();
        quantRevert.current = null;
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
        // The previous model is still resident.
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.canReapply);
          lastLoadRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        // Cancelled or evicted: terminal, else this loop spins forever.
        dismissLoadToast();
        setBusy(null);
        if (quantRevert.current) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
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

  // The unload failed so the load still runs; restore its poll and toast. refreshStatus cannot,
  // since a first load is not resident yet.
  const restoreLoadTracking = useCallback(() => {
    loadTrackingRestored.current = true;
    setBusy("loading");
    lastLoadSig.current = null;
    loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, undefined, cancelLoadFromToast));
    void pollLoadProgress();
  }, [pollLoadProgress, cancelLoadFromToast]);

  const stopGenPoll = useCallback(() => {
    if (genPollTimer.current) clearInterval(genPollTimer.current);
    genPollTimer.current = null;
    if (genVisibilityListener.current) {
      document.removeEventListener("visibilitychange", genVisibilityListener.current);
      genVisibilityListener.current = null;
    }
  }, []);

  // Shared with the mount-time resume.
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
            // Sorted, not prepended: a new clip is unpinned and goes after the pinned group.
            const clip = p.video;
            // Archiving cannot revoke a response already on the wire.
            if (!galleryCache.archived.has(clip.id) && !galleryCache.deleted.has(clip.id)) {
              stripEpoch.current += 1;
              setVideos((prev) =>
                sortGalleryItems([clip, ...prev.filter((v) => v.id !== clip.id)]),
              );
              setSelectedId(clip.id);
              void ensureThumbnail(clip);
              void ensureSrc(clip);
            }
          } else if (p.phase === "failed") {
            const msg = p.error || "Video generation failed";
            // The user's own Cancel surfaces as the cancelled sentinel; not an error.
            if (!msg.toLowerCase().includes("cancelled"))
              toast.error(msg, { action: generationFailureLogsAction(msg) });
          }
          return;
        }
        setGenStep((prev) => {
          if (!p.active) return null;
          if (prev && sameGenerateProgress(prev, p)) return prev;
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
  }, [ensureSrc, ensureThumbnail, stopGenPoll]);

  useEffect(() => {
    void (async () => {
      await refreshStatus();
      // Loads run on a backend daemon thread that survives navigation, so resume tracking.
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
      // Generation also runs on a daemon thread, so a reload re-enters the poll loop.
      try {
        const g = await getVideoGenerateProgress();
        if (g.active) {
          setBusy("generating");
          setGenStep(g.phase === "queued" ? null : g);
          startGenPoll();
        } else if (g.phase === "completed" && g.video) {
          // The terminal record persists until the next job, covering a finish after the mount fetch.
          const clip = g.video;
          // A client racing the delete must not merge a record whose file is gone.
          if (!galleryCache.deleted.has(clip.id) && !galleryCache.archived.has(clip.id)) {
            stripEpoch.current += 1;
            setVideos((prev) =>
              prev.some((v) => v.id === clip.id) ? prev : sortGalleryItems([clip, ...prev]),
            );
            void ensureThumbnail(clip);
            void ensureSrc(clip);
          }
        } else if (g.phase === "failed") {
          // Kept until the next job, so a reload still shows the failure.
          const msg = g.error || "Video generation failed";
          if (!msg.toLowerCase().includes("cancelled"))
              toast.error(msg, { action: generationFailureLogsAction(msg) });
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
  }, [refreshStatus, dismissLoadToast, pollLoadProgress, startGenPoll, stopGenPoll, ensureSrc, ensureThumbnail, cancelLoadFromToast]);

  // Stable because route effects depend on loadOrStage.
  const loadControlsRef = useRef({
    memoryMode,
    speedMode,
    attentionBackend,
    transformerCache,
    transformerQuant,
    familyOverride,
    selectedGpu,
    gpuChoices,
  });
  loadControlsRef.current = {
    memoryMode,
    speedMode,
    attentionBackend,
    transformerCache,
    transformerQuant,
    familyOverride,
    selectedGpu,
    gpuChoices,
  };
  const currentLoadAdvanced = useCallback(
    (kind: "gguf" | "single_file" | "pipeline", familyOverrideRequired = true): VideoLoadAdvanced => {
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
        family_override: familyOverrideRequired ? explicitFamily(controls.familyOverride) : undefined,
        // Dropped when the chosen card is gone so a stale pick loads automatically instead of 400ing.
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
      const advanced = currentLoadAdvanced("gguf", false);
      const plan = await getVideoDownloadPlan({
        model_path: repoId,
        gguf_filename: meta.ggufFilename,
        model_kind: "gguf",
        hf_token: hfApiToken(getHfToken()),
        transformer_quant: advanced.transformer_quant,
        memory_mode: advanced.memory_mode,
        family_override: advanced.family_override,
        // The plan sizes its file set against the chosen card.
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
    // True when the background load STARTED; callers may revert optimistic state on false.
    async (
      repoId: string,
      opts: VideoLoadOptions,
      pinned?: VideoLoadAdvanced,
      pickToastId?: string,
    ): Promise<boolean> => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      // Read before the request: a Cancel in flight can reach the backend first and stop nothing.
      const startSeq = cancelSeq.current;
      const startLoad = ++loadSeq.current;
      // Published now and settled in the finally, so a cancel waits for the whole path.
      let settleLoad: () => void = () => {};
      const inFlight = new Promise<void>((resolve) => {
        settleLoad = resolve;
      });
      pendingStart.current = inFlight;
      const settle = (started: boolean): boolean => {
        settleLoad();
        if (pendingStart.current === inFlight) pendingStart.current = null;
        return started;
      };
      setBusy("loading");
      dismissLoadToast();
      lastLoadSig.current = null;
      const handedOver = pickToast.take(pickToastId);
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, handedOver, cancelLoadFromToast));
      // A load that fails to START leaves the previous model resident.
      const prevLastLoad = lastLoad.current;
      const prevCanReapply = canReapply;
      const advanced = pinned ?? currentLoadAdvanced(opts.kind);
      // Spread so the H3 partition rides along for Reapply.
      lastLoad.current = { repoId, ...opts };
      setCanReapply(true);
      lastLoadRevert.current = { prev: prevLastLoad, canReapply: prevCanReapply };
      try {
        const startRequest = loadVideoModel({
          model_path: repoId,
          display_repo_id: opts.displayRepoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          memory_mode: advanced.memory_mode,
          speed_mode: advanced.speed_mode,
          attention_backend: advanced.attention_backend,
          transformer_cache: advanced.transformer_cache,
          transformer_quant: advanced.transformer_quant,
          family_override: advanced.family_override,
          // Not an Advanced control: the partition is chosen per pick.
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
        // Cancelled mid-start: the unload may have landed before this load registered, so unload again
        // unless a newer load owns the page.
        if (startLoad === loadSeq.current) {
          try {
            await unloadVideoModel();
          } catch {
            // Not best-effort: this is the only request that can still stop the load.
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
      pickToast,
    ],
  );

  // Downloads go through the Hub download manager, as on Images.
  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: VideoLoadOptions;
    advanced: VideoLoadAdvanced;
    // A download outlives its pick and must not evict a newer one.
    token: number;
    toastId?: string;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  handleLoadRef.current = handleLoad;
  // A download finishing while hidden must not evict the visible page's model; the pick is held.
  const stagedLoadDeferred = useRef(false);
  // A deferred load can still be refused, so it needs the same rollback. `owned` is read before the
  // call so a newer pick's label is left alone.
  const runStagedLoad = useCallback(
    (pending: NonNullable<typeof pendingStagedLoad.current>) => {
      if (pendingStagedLoad.current === pending) pendingStagedLoad.current = null;
      if (!pickGuard.isLatest(pending.token)) {
        pickToast.dismiss(pending.toastId);
        return;
      }
      const owned = stagedQuantRevert.current;
      void handleLoadRef.current(pending.repoId, pending.opts, pending.advanced, pending.toastId).then((started) => {
        if (started) return;
        if (quantRevert.current && quantRevert.current === owned) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        if (stagedQuantRevert.current === owned) stagedQuantRevert.current = null;
      });
    },
    [pickGuard, revertPick, pickToast],
  );
  const { stage, progress: stagedProgress } = useStagedDownload({
    scopeId: "diffusion",
    onReady: () => {
      if (!active) {
        stagedLoadDeferred.current = true;
        pickToast.setPhase(pendingStagedLoad.current?.toastId, "ready");
        return;
      }
      const pending = pendingStagedLoad.current;
      if (pending) runStagedLoad(pending);
    },
    onCancelled: () => {
      // As on Images: an incomplete plan must not leave an intent for a late completion.
      pickToast.dismiss(pendingStagedLoad.current?.toastId);
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      // No load started, so no poll will roll back the optimistic quant label; only for this job's pick.
      if (quantRevert.current && quantRevert.current === stagedQuantRevert.current) {
        revertPick(quantRevert.current);
        quantRevert.current = null;
      }
      stagedQuantRevert.current = null;
    },
  });
  usePickToastProgress(pickToast, stagedProgress);

  useEffect(() => {
    if (!active || !stagedLoadDeferred.current) return;
    stagedLoadDeferred.current = false;
    const pending = pendingStagedLoad.current;
    if (pending) runStagedLoad(pending);
  }, [active, runStagedLoad]);

  // `token` lets an awaiting caller drop out when a newer pick takes the page.
  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: VideoLoadOptions,
      source: ModelSelectorChangeMeta["source"] = "hub",
      token?: number,
      familyOverrideRequired = false,
    ): Promise<boolean> => {
      // Every Hub pick needs the plan: a cached checkpoint can still miss its base repo's encoder or VAE.
      // Bumped before the non-hub return so a local pick invalidates one in flight.
      const pick = ++pickSeq.current;
      // The previous pick's staged intent dies with it, or its onReady loads the abandoned model.
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      stagedQuantRevert.current = null;
      pickToast.dismissAll();
      const owns = () => token === undefined || pickGuard.holds(token);
      if (!owns()) return true;
      const advanced = currentLoadAdvanced(opts.kind, familyOverrideRequired);
      if (source !== "hub") return handleLoadRef.current(repoId, opts, advanced);
      const pickToastId = pickToast.show();
      // Read before the await: a newer pick replaces quantRevert.
      const ownRevert = quantRevert.current;
      let incompatible: string | null = null;
      try {
        const plan = await getVideoDownloadPlan({
          model_path: opts.displayRepoId ?? repoId,
          gguf_filename: opts.filename,
          model_kind: opts.kind,
          // Without the token the lookup fails on a gated base and drops the companion entry.
          hf_token: hfApiToken(getHfToken()),
          transformer_quant: advanced.transformer_quant,
          memory_mode: advanced.memory_mode,
          family_override: advanced.family_override,
          // The two H3 denoisers are separate downloads.
          h3_task: opts.h3Task,
          gpu_ids: advanced.gpu_ids,
        });
        // Superseded; report started so this pick's `.then` leaves the newer label alone.
        if (pick !== pickSeq.current || !owns()) {
          pickToast.dismiss(pickToastId);
          return true;
        }
        // The shared envelope's half of the incompatible-pairing check; video has no live path here.
        incompatible = plan.incompatible_reason ?? null;
        if (!incompatible && plan.entries.length > 0) {
          const entries = diffusionStagingEntries(plan.entries, repoId, opts);
          if (entries.length === 0) return handleLoadRef.current(repoId, opts, advanced);
          pendingStagedLoad.current = {
            repoId,
            opts,
            advanced,
            token: token ?? pickGuard.claim(),
            toastId: pickToastId,
          };
          stagedQuantRevert.current = ownRevert;
          const staged = stage(entries);
          pickToast.setPhase(pickToastId, "downloading", staged);
          return true;
        }
      } catch {
        // No plan (older backend, metadata hiccup): fall back to the load's own download.
      }
      // Re-checked: a rejected plan after a newer pick must not reach the fallback load.
      if (pick !== pickSeq.current || !owns()) {
        pickToast.dismiss(pickToastId);
        return true;
      }
      if (incompatible) {
        pickToast.dismiss(pickToastId);
        toast.error(incompatible);
        return false;
      }
      return handleLoadRef.current(repoId, opts, advanced, pickToastId);
    },
    [stage, pickGuard, currentLoadAdvanced, pickToast],
  );

  // A GGUF pick can arrive with only a repo id. The backend rejects a gguf load with no filename
  // and a pipeline load of a GGUF repo, so name the file from the listing first.
  const loadGgufRepoPick = useCallback(
    async (
      repoId: string,
      quantHint: string | null,
      source: ModelSelectorChangeMeta["source"] = "hub",
      localPath?: string | null,
      effectiveFamilyOverride = familyOverride,
    ): Promise<boolean> => {
      // Claimed here so every entry point is covered.
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
        onAmbiguous: () =>
          toast.error("Pick a quantization for this model to load it"),
        onResolved: (filename) => {
          quantRevert.current = revert;
          setQuant(quantHint ?? filename);
          // The LTX variant lives in the checkpoint name, not the repo id.
          applyVideoModelDefaults(`${repoId}/${filename}`, effectiveFamilyOverride);
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

  // A pick rejected after beginPick() already retired its predecessor, so restore resident state here.
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
      // The same ending as Cancel: the pick already parked its rollback and never loaded.
      setPendingH3Load((pending) => {
        if (pending) abandonPick();
        return null;
      });
    }
  }, [abandonPick, active, pickGuard]);

  // A diffusion model picked from the chat picker arrives as ?model= on this route. This route's
  // own match, never `strict: false`: that resolves to the ROOT match, whose search is
  // whatever route is live, and /hub names its selection with the same param.
  const routeSearch = useSearch({ from: "/video", shouldThrow: false });
  const navigateSelf = useNavigate();
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    // A hidden page owns no query: both diffusion pages stay mounted.
    if (!active) return;
    if (!videoPresets.hydrated) return;
    const wanted = routeSearch?.model;
    // Released once the query is gone, or re-picking the same model becomes a dead click.
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    // `quant` is a filename, so a label goes through resolution; fields, not the rebuilt object.
    const routed = { quant: routeSearch?.quant, ggufQuant: routeSearch?.ggufQuant };
    const routedFilename = routedGgufFilename(routed);
    const routedLabel = routedGgufLabel(routed);
    const key = `${wanted}|${routeSearch?.quant ?? ""}|${routeSearch?.ggufQuant ?? ""}`;
    if (handledRouteModel.current === key) return;
    handledRouteModel.current = key;
    setFamilyOverride("auto");
    // Owns the page like a direct pick, so an earlier staged download cannot land on top.
    const token = pickGuard.claim();
    void navigateSelf({ to: "/video", search: {}, replace: true });
    // A label means a GGUF repo and is not loadable as a filename.
    if (routedLabel) {
      // Deferred: the load it fires owns the state a direct pick sets.
      void Promise.resolve().then(() =>
        loadGgufRepoPick(wanted, routedLabel, "hub", null, "auto"),
      );
      return;
    }
    // The chat picker only forwards GGUF filenames, so a curated single-file artifact needs the catalog.
    const pick = diffusionRoutePick(
      wanted,
      routedFilename ?? undefined,
      loadSpecFor(wanted, VIDEO_CATALOG),
    );
    // The catalog lists the repo, not its files.
    if (pick.opts.kind === "gguf" && !pick.opts.filename) {
      void Promise.resolve().then(() => loadGgufRepoPick(pick.repoId, null, "hub", null, "auto"));
      return;
    }
    // Routed intent owns the label and recipe, and rolls both back if the load never lands.
    const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
    quantRevert.current = revert;
    setQuant(pick.opts.kind === "pipeline" ? null : (pick.opts.filename ?? null));
    applyVideoModelDefaults(
      pick.opts.filename ? `${pick.repoId}/${pick.opts.filename}` : pick.repoId,
      "auto",
    );
    if (isH3PipelinePick(pick.repoId, pick.opts.kind)) {
      setPendingH3Load({
        repoId: pick.repoId,
        opts: pick.opts,
        source: "hub",
        token,
        familyOverrideRequired: false,
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

  // Page back until the ?item= clip loads. A counter retires lookups, since clearing the query must
  // not cancel its own.
  const routedItem = active ? routeSearch?.item : undefined;
  const routedLookup = useRef(0);
  useEffect(() => {
    if (!active) routedLookup.current += 1;
  }, [active]);
  useEffect(() => {
    if (!routedItem) return;
    const lookup = ++routedLookup.current;
    void navigateSelf({ to: "/video", search: {}, replace: true });
    void loadGalleryUntil({
      has: () => galleryCache.videos.some((entry) => entry.id === routedItem),
      count: () => galleryCache.videos.length,
      hasMore: () => galleryCache.hasMore,
      refresh: loadGallery,
      loadMore,
      busy: () => loadingMore.current,
      cancelled: () => lookup !== routedLookup.current,
    }).then((found) => {
      if (lookup !== routedLookup.current) return;
      if (found) {
        setSelectedId(routedItem);
      } else {
        toast(translate("library.toast.clipNotFound"), {
          description: translate("library.toast.notFoundDescription"),
        });
      }
    });
  }, [routedItem, navigateSelf, loadGallery, loadMore]);


  // The task dialog defers the load out of the branch that snapshotted the rollback, so the two
  // ways out carry that branch's two endings: choosing runs the load and reverts if it never
  // starts, cancelling abandons the pick.
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
        pending.familyOverrideRequired,
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
    pickToast.dismissAll();
  }, [abandonPick, pickGuard, pickToast]);

  const handleReapply = useCallback(() => {
    // Status wins when another client replaced the model; the ref covers this page's own load.
    const l = lastLoad.current;
    if (l) {
      void handleLoad(l.repoId, {
        kind: l.kind,
        filename: l.filename,
        h3Task: l.h3Task,
        displayRepoId: l.displayRepoId,
      });
    }
  }, [handleLoad]);

  // Every pick supersedes the last: a staged download outlives its pick and plans may be in flight.
  const beginPick = useCallback(() => {
    pickSeq.current += 1;
    pendingStagedLoad.current = null;
    stagedLoadDeferred.current = false;
    stagedQuantRevert.current = null;
    pickToast.dismissAll();
  }, [pickToast]);

  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      if (busy !== null) return;
      beginPick();
      // Before any branch, since staging never sets `busy`.
      const token = pickGuard.claim();
      const pipelineTarget = diffusionPipelineLoadTarget(id, meta);
      const { displayRepoId } = pipelineTarget;
      const familyOverrideRequired = meta.familyOverrideRequired === true;
      const nextFamilyOverride = familyOverrideRequired ? familyOverride : "auto";
      if (!familyOverrideRequired) setFamilyOverride("auto");
      const spec = loadSpecFor(id, VIDEO_CATALOG);
      if (spec && spec.kind !== "gguf") {
      // Carry a pending entry forward: a superseded staged pick left its optimistic state in place.
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(null);
        // The distilled variant lives in the filename; otherwise defaults fall to LTX 40-step/CFG-4.
        applyVideoModelDefaults(spec.filename ? `${id}/${spec.filename}` : id, nextFamilyOverride);
        if (isH3PipelinePick(id, spec.kind, nextFamilyOverride)) {
          setPendingH3Load({
            repoId: pipelineTarget.repoId,
            opts: { kind: spec.kind, filename: spec.filename, displayRepoId },
            source: pipelineTarget.source,
            token,
            familyOverrideRequired,
          });
          return;
        }
        void loadOrStage(
          pipelineTarget.repoId,
          { kind: spec.kind, filename: spec.filename, displayRepoId },
          pipelineTarget.source,
          token,
        ).then((started) => {
            if (!started && pickGuard.holds(token)) {
              revertPick(revert);
              quantRevert.current = null;
            }
          });
        return;
      }
      // Optimistic, reverted if the load fails to START; the poll owns the after-start revert.
      if (meta.ggufVariant && meta.ggufFilename) {
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(meta.ggufVariant);
        // The variant (distilled vs dev) lives in the filename.
        applyVideoModelDefaults(`${id}/${meta.ggufFilename}`, nextFamilyOverride);
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
      // A local .gguf has no variant/filename; split the path into (parent dir, basename).
      if (meta.isGguf) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        if (!filename.toLowerCase().endsWith(".gguf")) {
          void loadGgufRepoPick(
            id,
            meta.ggufVariant ?? null,
            meta.source,
            meta.source === "local" ? id : null,
            nextFamilyOverride,
          );
          return;
        }
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(filename);
        applyVideoModelDefaults(id, nextFamilyOverride);
        void handleLoad(dir, { kind: "gguf", filename }, currentLoadAdvanced("gguf", false)).then((started) => {
          if (!started) {
            revertPick(revert);
            quantRevert.current = null;
          }
        });
        return;
      }
      // The pipeline route rejects a bare file, and only after evicting the resident model.
      if (meta.source === "local" && id.toLowerCase().endsWith(".safetensors")) {
        const norm = id.replace(/\\/g, "/");
        const slash = norm.lastIndexOf("/");
        const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
        const dir = slash >= 0 ? norm.slice(0, slash) : ".";
        const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
        quantRevert.current = revert;
        setQuant(filename);
        applyVideoModelDefaults(id, nextFamilyOverride);
        void handleLoad(dir, { kind: "single_file", filename }, currentLoadAdvanced("single_file", false)).then((started) => {
          if (!started) {
            revertPick(revert);
            quantRevert.current = null;
          }
        });
        return;
      }
      // The backend rejects a pipeline load of a single-file GGUF repo.
      if (spec?.kind === "gguf" || meta.ggufVariant) {
        void loadGgufRepoPick(
          id,
          spec?.filename ?? meta.ggufVariant ?? null,
          meta.source,
          meta.source === "local" ? id : null,
          nextFamilyOverride,
        );
        return;
      }
      // The backend gates loads to unsloth/* repos, the family bases, or on-device paths.
      if (!pipelineTarget.onDevice && !id.toLowerCase().startsWith("unsloth/")) {
        toast.error("Only unsloth or on-device video models can be loaded here");
        abandonPick();
        return;
      }
      // Its own rollback, or an older staged download could revert over this pick.
      const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
      quantRevert.current = revert;
      setQuant(null);
      applyVideoModelDefaults(id, nextFamilyOverride);
      // The on-device H3 pipeline lands here and needs the partition question too.
      if (isH3PipelinePick(id, "pipeline", nextFamilyOverride)) {
        setPendingH3Load({
          repoId: pipelineTarget.repoId,
          opts: { kind: "pipeline", displayRepoId },
          source: pipelineTarget.source,
          token,
          familyOverrideRequired,
        });
        return;
      }
      void loadOrStage(pipelineTarget.repoId, { kind: "pipeline", displayRepoId }, pipelineTarget.source, token, familyOverrideRequired).then((started) => {
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
      currentLoadAdvanced,
      handleLoad,
      familyOverride,
      loadGgufRepoPick,
      loadOrStage,
      pickGuard,
      quant,
      revertPick,
    ],
  );

  // True only when the backend accepted the unload.
  const handleUnload = useCallback(async (): Promise<boolean> => {
    dropResidentState();
    loadTrackingRestored.current = false;
    setBusy("unloading");
    try {
      setStatusIfNewest(++statusTicket.current, await unloadVideoModel());
      setQuant(null);
      // Wait for any in-flight load start to finish, or the older handler skips its compensating unload
      // and leaves a large load running with no cancel.
      const pending = pendingStart.current;
      if (pending) {
        try {
          await pending;
        } catch {
          // Its own handler reports the failure; this only waits for the window to close.
        }
      }
      // Restored tracking means the compensating unload failed and the load still runs.
      return !loadTrackingRestored.current;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
      return false;
    } finally {
      // A restore during the wait set "loading" deliberately; keep its Cancel controls.
      setBusy((prev) => (prev === "unloading" ? null : prev));
    }
  }, [refreshStatus, dropResidentState]);

  // Cancelling a load IS the unload; only cache remains, so reloading resumes.
  const handleCancelLoad = useCallback(async () => {
    const wasLoading = busy === "loading";
    if (await handleUnload()) {
      toast.info("Stopped loading the model", {
        description: "Anything already downloaded stays cached, so loading it again resumes.",
      });
      return;
    }
    // Already restored in handleUnload; a second restore duplicates the toast and poll.
    if (!wasLoading || loadTrackingRestored.current) return;
    restoreLoadTracking();
  }, [busy, handleUnload, restoreLoadTracking]);

  useEffect(() => {
    cancelLoadRef.current = () => void handleCancelLoad();
  }, [handleCancelLoad]);

  const handleCancelGenerate = useCallback(async () => {
    setStopping(true);
    try {
      const { cancelled } = await cancelVideoGeneration();
      if (!cancelled) setStopping(false);
    } catch {
      setStopping(false);
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
    // Pinned now, even if random, so the recipe records it.
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

    // Only after validation, so a rejected attempt is not kept.
    saveLastPrompt("video", prompt);
    setBusy("generating");
    setGenStep(null);
    // The POST only starts the job (minutes, and the secure tunnel caps responses near 100s).
    try {
      await generateVideo({
        prompt: prompt.trim(),
        // Only when guidance uses it, so the recipe does not record an ignored prompt.
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
        live_preview: livePreview,
      });
    } catch (err) {
      if (!isMounted.current) return;
      const refusal = err instanceof Error ? err.message : "Video generation failed";
      toast.error(refusal, { action: generationFailureLogsAction(refusal) });
      setBusy(null);
      setGenStep(null);
      // The refusal can be "No video model is loaded"; re-read status.
      void resyncAfterGenerateRefusal();
      return;
    }
    startGenPoll();
  }, [
    livePreview,
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

  const advancedControls = (
    <>
      <AdvancedSelect {...familySelect} badge={<ResolvedBadge status={status} controlKey="family_override" />} />
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
        hint="Auto compiles every model at load: a clip takes minutes to denoise, so the one-time compile always pays for itself within a single run. eager = fused kernels, no compile. max adds TF32 + fused QKV, plus the step cache on 20+ step models."
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
      {/* The dense transformer_quant fast path engages only on a full-pipeline load, so gate the
          control and otherwise show why it is unavailable. */}
      {!status?.loaded || status.model_kind === "pipeline" ? (
        <AdvancedSelect
          label="Precision"
          hint="How the model computes. Auto picks the fastest precision the hardware supports (INT8 on every capable GPU, then FP8 where the card has it) by quantising the transformer onto low-precision tensor cores, and keeps plain bf16 when the device or memory plan can't take it. Off always runs bf16."
          badge={<ResolvedBadge status={status} controlKey="transformer_quant" />}
          value={transformerQuant}
          onValueChange={(v) => setTransformerQuant(v as typeof transformerQuant)}
          options={[
            ["auto", "Auto (fastest for GPU)"],
            ["none", "Off (bf16)"],
            // Low-precision schemes need dense tensor cores, which Mac or CPU-only hosts lack.
            ...(hostOffersDensePrecision(hostClass)
              ? withNvfp4Option(
                  [
                    ["fp8", "FP8"],
                    ["int8", "INT8"],
                    ["nvfp4", "NVFP4 (Blackwell)"],
                    ["mxfp8", "MXFP8 (Blackwell)"],
                  ] as [string, string][],
                  nvfp4Diffusion,
                )
              : []),
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
        hint="Attention kernel. Auto upgrades to cuDNN fused attention on NVIDIA when a speed profile is active. sage is INT8 attention (SageAttention 2; without a local install Studio fetches the Hugging Face kernels-hub build, which runs on Ampere, Ada and Hopper GPUs, and any other GPU keeps the default): fast (10-40%) but can black-frame some families (Qwen, Wan), so it never engages automatically."
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
        hint="Static skip extrapolates middle steps on a fixed schedule (12+ steps) and keeps the compile and CUDA graph; Wan2.2 A14B and LTX-2 run uncached. Auto uses it for text-to-video (no keyframes or references), at or above the step count it was measured at, on Wan2.2 TI2V 5B on every speed tier but Off/Eager, and on HunyuanVideo 1.5 and MiniMax-H3 on Max only. First-Block-Cache reuses the transformer tail across steps (larger quality cost); Auto turns it on for other many-step models on Max only. UNSLOTH_DIFFUSION_AUTO_STEP_SKIP=0 stops Auto from picking Static skip."
        badge={<ResolvedBadge status={status} controlKey="transformer_cache" />}
        value={transformerCache}
        onValueChange={(v) => setTransformerCache(v as typeof transformerCache)}
        options={[
          ["auto", "Auto"],
          ["off", "Off"],
          ["fbcache", "First-Block-Cache"],
          ["static", "Static skip"],
        ]}
      />
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
          Live preview
          <InfoHint>Show a rough preview of the first frame while the clip denoises. Costs no measurable speed and never changes the final video.</InfoHint>
        </span>
        <Switch
          checked={livePreview}
          onCheckedChange={(on) => setLivePreviewOff(!on)}
          aria-label="Live preview"
        />
      </div>
      <LoadedBuildSummary status={status} />
    </>
  );

  return (
    // The chat-style layout has no outer top inset, so apply the content inset here, as chat does.
    <div
      {...{ [MEDIA_RAIL_ROOT_ATTR]: "" }}
      style={railRootStyle}
      className="diffusion-surface @container relative flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]"
    >
      <MediaRailResizeHandle kind="video" placement="page" className="hidden @[50rem]:block" />
      {/* Portals to body, and this page stays mounted off-route, so gate it like the composer. */}
      {active && <GuidedTour {...tour.tourProps} />}
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
        {/* Squarer than the shared dialog's rounded-4xl: at this width the default reads as a lozenge
            rather than a panel. */}
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
      <div className="pointer-events-none relative z-40 grid h-[calc(48px*var(--ui-space-scale,1))] shrink-0 grid-cols-[minmax(0,var(--media-rail-width,calc(408px*var(--ui-space-scale,1))))_minmax(13rem,1fr)] @max-[30rem]:grid-cols-[minmax(0,1fr)_auto]">
        <div
          className={cn(
            "pointer-events-none flex h-full min-w-0 items-start overflow-hidden @[50rem]:border-r @[50rem]:border-border/60",
            isMobileShell
              ? "pl-12"
              : // Collapsed desktop sidebar: clear the titlebar buttons, as Chat does.
                !pinned && isTauri
                ? "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
                : "pl-[var(--studio-media-header-left-inset,1.5rem)]",
          )}
        >
          {/* min-w-0: without it a long resident model name pushes the Library link off a phone screen. */}
          <div className="pointer-events-auto flex min-w-0 max-w-full items-center gap-2 overflow-hidden pt-[var(--studio-chat-header-padding-top,11px)]">
            <ModelSelector
              triggerDataTour="video-model"
              models={videoModels}
              value={selectorModelId}
              loadedModelIdOverride={selectorModelId}
              activeGgufVariant={quant}
              onValueChange={handleModelSelect}
              resolveDownloadFootprint={resolveDownloadFootprint}
              onEject={status?.loaded ? handleUnload : undefined}
              variant="ghost"
              className="!h-[calc(34px*var(--ui-space-scale,1))]"
              task={VIDEO_GEN_TASKS}
              catalog={VIDEO_CATALOG}
              opaqueKind={opaqueKind}
              hubCapability="diffusion"
              placeholder="Select video model"
              open={active && selectorOpen}
              onOpenChange={(o) => setSelectorOpen(active && o)}
            />
            {/* The load's own cancel, beside the selector rather than inside it: the selector's eject needs
                a resident model, so it is hidden for exactly the span a first load runs. A real button,
                so it is keyboard reachable. Says "load", never "download": that Cancel stops another job. */}
            {busy === "loading" && (
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    aria-label="Cancel load"
                    className="!h-[calc(34px*var(--ui-space-scale,1))] rounded-full text-xs"
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
        <div className="grid h-full min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-2">
          {status?.loaded && (
            <div className="pointer-events-auto col-start-1 mt-[var(--studio-chat-header-padding-top,11px)] flex h-[var(--studio-chat-control-height,34px)] min-w-0 flex-wrap content-start gap-x-3 overflow-hidden pl-4 text-ui-11 leading-[var(--studio-chat-control-height,34px)]">
              {status.family && <StatusChip label="Family" value={status.family} />}
              {status.engine && <StatusChip label="Engine" value={status.engine} />}
              {status.model_kind && <StatusChip label="Kind" value={status.model_kind} />}
              {status.offload_policy && (
                <StatusChip label="Offload" value={status.offload_policy} />
              )}
              {status.speed_mode && <StatusChip label="Speed" value={status.speed_mode} />}
            </div>
          )}
          <div className="pointer-events-none col-start-2 flex min-w-0 items-start justify-end pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
            <div className="pointer-events-auto flex min-w-0 items-center gap-2">
              <LibraryPageLink
                tab="videos"
                labelClassName="hidden @[50rem]:inline"
                arrowClassName="hidden @[50rem]:block"
              />
            </div>
          </div>
        </div>
      </div>

      {/* overflow-x-hidden: an unset overflow-x computes to auto beside overflow-y-auto, letting a
          wide row pan the page sideways on a phone. */}
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto overflow-x-hidden @[50rem]:flex-row @[50rem]:overflow-hidden">
        <div
          data-tour="video-settings"
          className="flex w-full shrink-0 flex-col border-b border-border/60 @[50rem]:w-[min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100%-13rem))] @[50rem]:overflow-hidden @[50rem]:border-r @[50rem]:border-b-0"
        >
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar panel-scroll-fade-action flex min-h-0 flex-1 flex-col gap-4 px-10 max-sm:px-5 pt-9 pb-6 @[50rem]:overflow-y-auto",
              settingsFadeClass,
            )}
          >
          <div className="mb-2 flex items-start justify-between gap-3">
            <div className="min-w-0 grid gap-1.5">
              <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none text-foreground">
                <HugeiconsIcon icon={FlimSlateIcon} className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0" />
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
              data-type-to-activate="prompt"
              rows={4}
              placeholder={exampleDismissed ? undefined : VIDEO_EXAMPLE_PROMPT}
              value={prompt}
              onFocus={() => {
                if (exampleDismissed) return;
                dismissExample("video");
                setExampleDismissed(true);
              }}
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
                  // The tag in the prompt is the position.
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
          {/* The scroll mask fades; leave the footer unpainted to avoid dark-mode banding. */}
          <div className="relative z-10 flex shrink-0 flex-wrap justify-center gap-2 px-4 pt-0.5 pb-4">
            {busy === "generating" ? (
              <Button
                // Kept in step with the Images Stop control.
                className="relative z-10 h-11 px-8 hover:bg-muted dark:hover:bg-muted"
                variant="outline"
                onClick={handleCancelGenerate}
              >
                <Spinner className="mr-2 size-4" />
                {stopButtonLabel({ stopping, done: null, count: 1, idle: "Cancel" })}
              </Button>
            ) : (
              <>
                <Button
                  className="relative z-10 h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                  onClick={handleGenerate}
                  disabled={busy !== null || !status?.loaded}
                >
                  Generate
                </Button>
                {status?.loaded && canReapply && (
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <Button
                        className="relative z-10 h-11 px-5"
                        variant="secondary"
                        disabled={busy !== null}
                        onClick={handleReapply}
                      >
                        <HugeiconsIcon icon={Refresh01Icon} className="mr-2 size-4" />
                        Reapply
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Reload the current model with the advanced options</TooltipContent>
                  </Tooltip>
                )}
              </>
            )}
          </div>
        </div>

        <div
          data-tour="video-preview"
          className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden @[50rem]:min-h-0"
        >
          {viewer && viewerVideo && viewerSrc && (
            <MediaViewer
              open={true}
              onOpenChange={(open) => !open && closeViewer()}
              title={viewerVideo.prompt || t("library.viewer.untitledVideo")}
              meta={`Generated · ${viewerVideo.width} × ${viewerVideo.height} · ${Math.round(viewerVideo.duration_s)}s`}
              media={true}
              noun="video"
              actions={{
                primary: {
                  label: t("library.menu.chatAboutThis"),
                  icon: MessageCircleIcon,
                  onClick: () =>
                    void chatAboutMedia(
                      navigateToChat,
                      () => fetchWithFreshLink(viewerSrc, () => fetchGalleryVideoSignedUrl(viewerVideo.id)),
                      viewerVideo.prompt,
                      "video",
                    ),
                },
                onDownload: () => void handleQuickDownload(viewerVideo),
                reveal: revealLabel
                  ? { label: revealLabel, onClick: () => revealInFolder(`video:${viewerVideo.id}`) }
                  : undefined,
                favorite: isFavorite(`video:${viewerVideo.id}`),
                onToggleFavorite: () => toggleFavorite(`video:${viewerVideo.id}`),
                onAddToProject: (projectId) => addGalleryVideoToProject(viewerVideo.id, projectId),
                onDelete: () => {
                  viewerVideoRef.current?.pause();
                  handback.current = null;
                  setViewer(null);
                  void handleDelete(viewerVideo.id);
                },
              }}
            >
              <video
                ref={viewerVideoRef}
                src={viewerSrc}
                controls
                playsInline
                muted={viewer.from.muted}
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget;
                  if (viewer.from.time) video.currentTime = viewer.from.time;
                  video.volume = viewer.from.volume;
                  viewerPositioned.current = true;
                  if (viewer.from.playing) void playWithMutedFallback(video);
                }}
                onTimeUpdate={(event) => recordViewer(event.currentTarget)}
                onVolumeChange={(event) => recordViewer(event.currentTarget)}
                onError={(event) => {
                  const from = readPlayback(event.currentTarget, viewer.from, viewerPositioned.current);
                  viewerPositioned.current = false;
                  setViewer((current) => current && { ...current, from });
                  remintSrc(viewerVideo);
                }}
                className="size-full object-contain"
              />
            </MediaViewer>
          )}
          <div className="hover-scrollbar relative flex flex-1 items-center justify-center overflow-auto p-6">
            {livePreviewSrc ? (
              // Scaled to clip size from the preview's own aspect; the finished clip replaces it.
              <img
                src={livePreviewSrc}
                alt="Live preview of the first frame being generated"
                data-testid="video-live-preview"
                style={livePreviewBox}
                className="size-full object-contain shadow-sm"
              />
            ) : selected && selectedSrc ? (
              <>
                {/* autoPlay + muted + playsInline so it plays inline without a gesture; controls let the user
                    scrub. onEnded replays up to 3 total plays, reset per selection. */}
                <video
                  key={selected.id}
                  ref={previewRef}
                  src={selectedSrc}
                  controls
                  // A clip finishing behind the open viewer must not play under it.
                  autoPlay={viewer === null}
                  muted
                  playsInline
                  onPlay={() => {
                    playCountRef.current += 1;
                  }}
                  onEnded={(e) => {
                    // Not while hidden: a replay would restart audio on another page.
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
                    <HugeiconsIcon icon={Volume02Icon} className="size-3.5" />
                    Audio
                  </div>
                )}
                {/* No button borders: focus returning from a menu would draw one. Keyboard focus tints instead. */}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-xl bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur [&_[data-slot=button]]:border-0 [&_[data-slot=button]:focus-visible]:bg-muted">
                  {/* Not a click on the clip itself: Chrome's ⋮ menu, WebKit's centred play button and the
                      first click of a double-click to fullscreen all land on the frame, above the controls. */}
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    aria-label={t("library.viewer.openVideo")}
                    title={t("library.viewer.openVideo")}
                    onClick={(event) => {
                      // Safari does not focus a clicked button, and the viewer returns focus to what had it.
                      event.currentTarget.focus();
                      openViewer();
                    }}
                  >
                    <HugeiconsIcon icon={ArrowExpand01Icon} className="size-4" />
                  </Button>
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
                    favorite={isFavorite(`video:${selected.id}`)}
                    onToggleFavorite={() => toggleFavorite(`video:${selected.id}`)}
                    onTogglePin={() =>
                      void handleTogglePin(selected.id, !selected.pinned)
                    }
                    onToggleArchive={() => void handleArchive(selected.id)}
                    onDelete={() => void handleDelete(selected.id)}
                    onDownload={() => void handleQuickDownload(selected)}
                    onAddToProject={(projectId) => addGalleryVideoToProject(selected.id, projectId)}
                  />
                </div>
              </>
            ) : selected ? (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Spinner className="size-8" />
                <p className="text-sm">Loading…</p>
              </div>
            ) : busy === "generating" ? null : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <HugeiconsIcon icon={FlimSlateIcon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a video model to load"}
                </p>
              </div>
            )}

            {/* Live generation progress: a per-step bar with the phase label and ETA, centered when there
                is nothing else to show. */}
            {busy === "generating" && (
              <div
                className={cn(
                  "pointer-events-none absolute flex justify-center px-4",
                  selectedSrc || livePreviewSrc ? "inset-x-0 bottom-4" : "inset-0 items-center",
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
                    progressLabel={genStep ? genStepLabel(genStep, Boolean(status?.has_audio)) : null}
                  />
                </div>
              </div>
            )}
          </div>

          {(videos.length > 0 || busy === "generating") && (
            <div
              ref={stripRef}
              {...stripReorder.stripProps}
              className="hover-scrollbar flex shrink-0 items-stretch gap-2 overflow-x-auto border-t border-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] p-3"
              onScroll={(e) => {
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past clips stay browsable while
                  the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center overflow-hidden rounded-[10px] bg-muted/50 ring-2 ring-primary/30">
                  {livePreviewSrc ? (
                    <img src={livePreviewSrc} alt="" className="size-full object-cover" />
                  ) : (
                    <Spinner className="size-5 text-muted-foreground" />
                  )}
                </div>
              )}
              {/* The card is a wrapper, not a button: the actions menu must be the select button's SIBLING,
                  since a button inside a button is invalid. data-clip-id rides the wrapper so the observer
                  still sees it. */}
              {videos.map((video) => (
                <div
                  key={video.id}
                  data-clip-id={video.id}
                  {...stripReorder.tileProps(video.id)}
                  className={cn(
                    "group relative h-16 w-24 shrink-0",
                    stripReorder.draggingId === video.id && "opacity-40",
                  )}
                >
                  {stripReorder.cue?.id === video.id && (
                    <StripDropLine edge={stripReorder.cue.edge} />
                  )}
                <Tooltip>
                <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  onClick={() => setSelectedId(video.id)}
                  className="relative flex size-full flex-col justify-end overflow-hidden rounded-[10px] bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {thumbnailById[video.id] ? (
                    <img
                      src={thumbnailById[video.id]}
                      alt=""
                      draggable={false}
                      onError={() => handlePosterError(video.id)}
                      className="absolute inset-0 size-full object-cover"
                    />
                  ) : thumbnailFailedIds.has(video.id) ? (
                    <span className="absolute inset-0 flex items-center justify-center">
                      <HugeiconsIcon
                        icon={FlimSlateIcon}
                        className="size-4 text-muted-foreground"
                      />
                    </span>
                  ) : (
                    <span className="absolute inset-0 flex items-center justify-center">
                      <Spinner className="size-4 text-muted-foreground" />
                    </span>
                  )}
                  {/* A terse caption strip so cards read at a glance. Left/bottom padding clears the rounded
                      corner and the selection border. */}
                  <span className="relative z-10 truncate bg-gradient-to-t from-black/70 to-transparent px-2 pb-1 pt-2 text-left text-ui-9 font-medium leading-none text-white">
                    {clipMeta(video)}
                  </span>
                  {video.id === selected?.id && (
                    <span className="pointer-events-none absolute inset-0 z-20 rounded-[10px] border border-border bg-white/35 dark:border-[rgb(255_255_255_/_calc(0.25*var(--contrast-edge-gain,1)))] dark:bg-white/20" />
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
                {video.pinned && (
                  <GalleryPinBadge
                    noun="video"
                    className="left-0.5 top-0.5 z-30"
                    onUnpin={() => void handleTogglePin(video.id, false)}
                  />
                )}
                <div className="absolute right-0.5 top-0.5 z-30">
                  <GalleryItemMenu
                    variant="overlay"
                    noun="video"
                    active={active}
                    pinned={Boolean(video.pinned)}
                    archived={Boolean(video.archived)}
                    favorite={isFavorite(`video:${video.id}`)}
                    onToggleFavorite={() => toggleFavorite(`video:${video.id}`)}
                    onTogglePin={() => void handleTogglePin(video.id, !video.pinned)}
                    onToggleArchive={() => void handleArchive(video.id)}
                    onDelete={() => void handleDelete(video.id)}
                    onDownload={() => void handleQuickDownload(video)}
                    onAddToProject={(projectId) => addGalleryVideoToProject(video.id, projectId)}
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
