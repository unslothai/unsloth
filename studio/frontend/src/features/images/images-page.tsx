// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { generationFailureLogsAction } from "@/features/settings/lib/view-logs-action";
import { readImageModel, rememberImageModel, matchesRememberedModel, componentFilesMatch, type RememberedImageModel } from "./image-model-recall";
import { componentFileFields, splitComponentFileList } from "./component-files";
import {
  type ReactNode,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ArrowExpand01Icon,
  ArrowLeftRightIcon,
  ArrowUpDownIcon,
  Refresh01Icon,
  Delete02Icon,
  Download01Icon,
  Image03Icon,
  ImageAdd02Icon,
  InformationCircleIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { MessageCircleIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { MediaViewer } from "@/components/media-viewer";
import { shortPrompt } from "@/lib/prompt-text";

import { ImageDropzone } from "@/components/image-dropzone";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { buildImagesTourSteps } from "./tour";
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
import {
  explicitFamily,
  resolvedFamilyOverrideSelection,
  useFamilyOverride,
} from "@/features/model-picker/components/model-selector/family-override";
import { IMAGE_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import {
  type HostClass,
  hostOffersDensePrecision,
} from "@/features/model-picker/components/model-selector/host-artifact-policy";
import {
  IMAGE_CATALOG,
  catalogToModelOptions,
  curatedArtifactTakesDenseQuant,
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
import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { GalleryItemMenu, GalleryPinBadge } from "@/components/gallery-item-menu";
import { MediaRailResizeHandle } from "@/components/media-rail-resize-handle";
import { MEDIA_RAIL_ROOT_ATTR, useMediaRailWidth } from "@/hooks/use-media-rail-width";
import { StripDropLine } from "@/components/gallery-strip-reorder";
import { useStripReorder } from "@/hooks/use-strip-reorder";
import { LibraryPageLink } from "@/components/media-page-link";
import { translate, useT } from "@/i18n";
import {
  chatAboutMedia,
  revealInFolder,
  useLibraryFavorites,
  useRevealLabel,
} from "@/features/library";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  type NewRecordProbeBaseline,
  applyPin,
  fetchNextPage,
  fetchWhileStable,
  hasUnknownRecord,
  mergeGenerated,
  moveGalleryItem,
  newRecordProbeBaseline,
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
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useImageWorkflowStore } from "./stores/image-workflow-store";
import {
  WORKFLOW_EXAMPLE_PROMPTS,
  WORKFLOW_TABS,
  recipeWorkflowLabel,
  type WorkflowId,
} from "./workflows";
import { ParamSlider } from "@/features/chat";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import {
  type ImageGenerationPresetParams,
  MediaGenerationPresetControl,
  useMediaGenerationPresets,
} from "@/features/generation-presets";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { formatBytes, formatEta } from "@/features/hub/lib/format";
import { generatePhaseLabel, sameGenerateProgress } from "@/lib/media-generate-phase";
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
import { useDiffusionPickToast, usePickToastProgress } from "@/lib/use-diffusion-pick-toast";
import {
  PRECISION_REFUSAL_TITLE,
  denseTextEncoderBuildLabel,
  denseTransformerBuildLabel,
  isNativeEngineStatus,
  formatResolvedValue,
  isDenseQuantKind,
  isPrecisionRefusal,
  memoryRecipeValue,
  resolvedBadge,
  resolvedSeedKey,
  resolvedSelectValue,
} from "@/lib/resolved-precision";
import { diffusionPipelineLoadTarget, diffusionStagingEntries } from "@/lib/diffusion-pipeline-load-target";
import {
  routedGgufFilename,
  routedGgufLabel,
} from "@/lib/diffusion-route-search";
import { toast } from "@/lib/toast";
import { loadGalleryUntil } from "@/lib/gallery-deep-link";
import { subscribeModelEjected } from "@/lib/model-lifecycle-events";
import {
  DEFAULT_GEN,
  defaultsFor,
  defaultsKeyFor,
  loadedRecipeFor,
  residentDefaultsKey,
  residentRecipeFor,
  resolutionFor,
} from "./image-generation-defaults";
import {
  MIN_DIM,
  type SizeLimits,
  DEFAULT_SIZE_LIMITS,
  fitSize,
  restorableSize,
  sizeLimitsFrom,
  snapDim,
} from "./image-size";
import {
  ANNOTATION_COLORS,
  type EditSizing,
  REFERENCE_DETAIL_LABELS,
  TRANSPARENCY_CHECKER,
  additionalImageNumber,
  conditionedRequestFields,
  maxAdditionalImages,
  presetsWithin,
  resolveEditSize,
  restoreInputsNote,
  seedReferenceResolution,
  withLocalizedHint,
  withTransparencyPrompt,
} from "./edit-conditioning";
import { LocalizedEditCanvas } from "./localized-edit-canvas";

import {
  type ControlNetSpecInput,
  type DiffusionControlNetInfo,
  type DiffusionGenerateProgress,
  type DiffusionGenerateResponse,
  type DiffusionLoadProgress,
  type DiffusionLoadRequest,
  type DiffusionLoraInfo,
  type DiffusionStatus,
  type LocalizedEditMode,
  type GalleryImage,
  type LoraSpecInput,
  GenerateResponseLostError,
  cancelDiffusionGeneration,
  deleteGalleryImage,
  fetchGalleryBlob,
  fetchGalleryResponse,
  fetchGalleryObjectUrl,
  galleryThumbnailUrl,
  generateDiffusionImage,
  getDiffusionLoadProgress,
  getDiffusionStatus,
  getGallery,
  getGenerateProgress,
  listDiffusionControlNets,
  listDiffusionLoras,
  addGalleryImageToProject,
  moveGalleryImage,
  setGalleryImageFlags,
  getDiffusionDownloadPlan,
  loadDiffusionModel,
  unloadDiffusionModel,
} from "./api";
import {
  shouldContinueGenerating,
  shouldReportGenerateError,
  stopButtonLabel,
} from "./lib/generation-stop";
import {
  ALLOW_OVERSIZED_HINT,
  ALLOW_OVERSIZED_LABEL,
  allowOversizedField,
  GENERATE_ANYWAY_LABEL,
  MEMORY_REFUSAL_TITLE,
  shouldOfferGenerateAnyway,
  shouldRunQueuedOversizedRetry,
} from "./lib/memory-refusal";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useStagedDownload, type StagedDownloadEntry } from "@/features/hub/download-manager";
import { DiffusionTrainPanel } from "./train/diffusion-train-panel";
import {
  TrainBaseSelector,
  type TrainFamilyOption,
} from "./train/train-base-selector";

function withEngagedFamily(
  { repoId, kind, filename, textEncoderFiles, vaeFile }: RememberedImageModel,
  status: Pick<DiffusionStatus, "resolved">,
): RememberedImageModel {
  const family = explicitFamily(resolvedFamilyOverrideSelection(status.resolved?.family_override));
  return {
    repoId,
    kind,
    ...(filename ? { filename } : {}),
    ...(family ? { familyOverride: family } : {}),
    ...(textEncoderFiles?.length ? { textEncoderFiles } : {}),
    ...(vaeFile ? { vaeFile } : {}),
  };
}

function sendsTransformerQuant(kind: string | null | undefined, repoId: string): boolean {
  return (
    isDenseQuantKind(kind) &&
    curatedArtifactTakesDenseQuant(repoId, IMAGE_CATALOG) !== false
  );
}

function useImageModels(
  host: HostClass,
  denseQuantSchemes: readonly string[],
): ModelOption[] {
  return useMemo(
    () => catalogToModelOptions(IMAGE_CATALOG, host, denseQuantSchemes),
    [host, denseQuantSchemes],
  );
}

// Images each conditioned workflow consumed, for the restore toast. Keys are the backend's
// workflow strings; txt2img is absent because it restores completely.
const CONDITIONED_WORKFLOW_INPUTS: Record<string, string> = {
  img2img: "the source image",
  inpaint: "the source image and mask",
  outpaint: "the source image",
  upscale: "the source image",
  edit: "the source image",
  reference: "the source and reference images",
  controlnet: "the control image",
};

const ASPECT_RATIOS: Record<string, [number, number]> = {
  "1:1": [1, 1],
  "3:2": [3, 2],
  "4:3": [4, 3],
  "16:9": [16, 9],
  "21:9": [21, 9],
};
const ASPECT_OPTIONS = ["custom", ...Object.keys(ASPECT_RATIOS)];
const ASPECT_LABELS: Record<string, string> = {
  "1:1": "Square",
  "3:2": "Photo",
  "4:3": "Landscape",
  "16:9": "Widescreen",
  "21:9": "Ultrawide",
};

const CONTROL_TYPE_LABELS: Record<string, string> = {
  passthrough: "Passthrough (already a map)",
  canny: "Canny (trace edges)",
  depth: "Depth (map)",
  pose: "Pose (map)",
};

// The number box accepts higher typed values on purpose.
const RUNS_SLIDER_MAX = 128;
const DIM_OPTIONS = [
  256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280,
  1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2752,
];

const MAX_OUTPUT_DEFAULT = DEFAULT_SIZE_LIMITS.maxSide;

function dimOptions(limits: SizeLimits): number[] {
  return DIM_OPTIONS.filter((n) => n <= limits.maxSide && n % limits.multiple === 0);
}

function DimensionSelect({
  icon,
  label,
  value,
  open,
  onOpenChange,
  onChange,
  limits = DEFAULT_SIZE_LIMITS,
}: {
  icon: IconSvgElement;
  label: string;
  value: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChange: (value: number) => void;
  limits?: SizeLimits;
}) {
  const [draft, setDraft] = useState(String(value));
  const [lastValue, setLastValue] = useState(value);
  if (value !== lastValue) {
    setLastValue(value);
    setDraft(String(value));
  }
  const commit = () => {
    const typed = Number(draft);
    const next = snapDim(Number.isFinite(typed) && typed > 0 ? typed : value, limits);
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
    <div className="flex h-9 flex-1 items-center gap-2 rounded-full border border-border bg-background px-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:focus-within:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]">
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
        <DropdownMenuContent align="end" className="max-h-[min(--spacing(72),var(--radix-dropdown-menu-content-available-height))] overflow-y-auto">
          {dimOptions(limits).map((n) => (
            <DropdownMenuItem key={n} onSelect={() => pick(n)}>
              <span className="tabular-nums">{n}</span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

function matchAspect(width: number, height: number): { key: string; portrait: boolean } {
  const target = Math.max(width, height) / Math.min(width, height);
  const found = Object.entries(ASPECT_RATIOS).find(
    ([, [a, b]]) => Math.abs(target - a / b) < 0.01,
  );
  return { key: found ? found[0] : "custom", portrait: height > width };
}

// The blob budget never evicts a visible image; object URLs are revoked only on delete.
const IMAGE_BLOB_BUDGET_BYTES = 192 * 1024 * 1024;

const galleryCache: {
  images: GalleryImage[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  srcById: BlobUrlCache;
  inflight: Set<string>;
  thumbById: BlobUrlCache;
  thumbInflight: Set<string>;
  deleted: Set<string>;
} = {
  images: [],
  hasMore: false,
  selectedId: null,
  quant: null,
  srcById: new BlobUrlCache(IMAGE_BLOB_BUDGET_BYTES),
  inflight: new Set(),
  thumbById: new BlobUrlCache(32 * 1024 * 1024),
  thumbInflight: new Set(),
  deleted: new Set(),
};

const PAGE_SIZE = 50;

const RESYNC_MAX_ATTEMPTS = 3;

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
    // WebKit can silently return PNG bytes when an encoder is unavailable.
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
      outputFormat = "png";
      outputBlob = null;
    }
  }

  const filename = exportFilename(image, outputFormat);
  try {
    if (outputBlob) {
      await downloadFile(outputBlob, filename, outputBlob.type);
    } else if (isTauri) {
      // WebKit can display the cached object URL but fail to fetch it again.
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

function genStepLabel(p: DiffusionGenerateProgress): string {
  return generatePhaseLabel({ ...p, total: p.total_steps }, { formatEta });
}

const SETTLE_POLL_MS = 1000;
const SETTLE_MAX_MS = 6 * 60 * 60 * 1000; // a native-CPU batch can run for hours
const SETTLE_MAX_FAILS = 5;

/** Idle progress alone is ambiguous: success needs progress seen active or a new gallery record. */
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
  throw new Error("Timed out waiting for the image generation to finish.");
}

const LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast items-center gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

function loadToastDescription(p: DiffusionLoadProgress) {
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

// `onCancel` is the only control that reaches a first load in flight.
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

const IMAGE_PROMPT_BOX = "image-prompt-box rounded-lg";

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

const COMPONENT_FILES_HINT =
  "Optional. Use separate ComfyUI text encoder / VAE .safetensors files instead of downloading the base model's. Absolute path, or relative to the model folder (e.g. ../text_encoders/clip_l.safetensors). Only for single-file or GGUF transformers.";

function AdvancedTextField({
  label,
  hint,
  placeholder,
  value,
  onValueChange,
  multiline = false,
}: {
  label: string;
  hint?: ReactNode;
  placeholder?: string;
  value: string;
  onValueChange: (v: string) => void;
  multiline?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="flex shrink-0 items-center gap-1 whitespace-nowrap text-xs font-medium text-muted-foreground">
        {label}
        {hint && <InfoHint>{hint}</InfoHint>}
      </span>
      {multiline ? (
        <Textarea
          aria-label={label}
          rows={2}
          spellCheck={false}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          className="min-h-0 resize-y font-mono text-xs"
        />
      ) : (
        <Input
          aria-label={label}
          spellCheck={false}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          className="h-8 font-mono text-xs"
        />
      )}
    </div>
  );
}

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
  badge?: ReactNode;
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
          <SelectTrigger aria-label={label} className="h-8 w-[calc(160px*var(--ui-space-scale,1))] max-sm:w-[min(calc(160px*var(--ui-space-scale,1)),50vw)] text-xs">
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

// Exports the mask at NATIVE resolution (white = repaint).
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

type ExtendSides = { left: boolean; right: boolean; top: boolean; bottom: boolean };

function scaleToCanvas(source: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const dst = document.createElement("canvas");
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext("2d");
  if (!dctx) throw new Error("Could not scale the extended canvas");
  dctx.drawImage(source, 0, 0, w, h);
  return dst;
}

async function buildOutpaint(
  src: string,
  sides: ExtendSides,
  pct: number,
): Promise<{ image: string; mask: string }> {
  const source = await loadImage(src);
  // Scale the source first: growing all sides 100% is 9x area, and oversized canvases no-op drawImage.
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
  ictx.drawImage(img, l, t, w, h);
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
  mctx.fillStyle = "#ffffff";
  mctx.fillRect(0, 0, nw, nh);
  mctx.fillStyle = "#000000";
  mctx.fillRect(l + ol, t + ot, w - ol - or, h - ot - ob);

  // Per-side rounding can overshoot the backend's 4096px limit.
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

function RecipePopover({
  image,
  onRestore,
  active,
}: {
  image: GalleryImage;
  onRestore: (image: GalleryImage) => void;
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
          <p className="text-ui-11 text-muted-foreground">{formatTimestamp(image.created_at)}</p>
        </div>
        <div className="flex min-h-0 flex-col gap-2 overflow-y-auto overscroll-contain px-4 py-3 text-xs">
          <RecipeRow label="Prompt" value={image.prompt} wrap />
          {image.negative_prompt ? (
            <RecipeRow label="Negative" value={image.negative_prompt} wrap />
          ) : null}
          {image.model ? <RecipeRow label="Model" value={image.model} /> : null}
          {image.gguf_filename ? <RecipeRow label="File" value={image.gguf_filename} mono /> : null}
          {image.transformer_quant ? (
            <RecipeRow label="Transformer" value={image.transformer_quant} />
          ) : null}
          {image.text_encoder_quant ? (
            <RecipeRow label="TE quant" value={image.text_encoder_quant} />
          ) : null}
          {image.memory_mode ||
          (image.offload_policy && image.offload_policy !== "none") ? (
            <RecipeRow
              label="Memory"
              value={memoryRecipeValue(image.memory_mode, image.offload_policy)}
            />
          ) : null}
          {image.speed_mode ? (
            <RecipeRow
              label="Speed"
              value={formatResolvedValue("speed_mode", image.speed_mode)}
            />
          ) : null}
          {image.attention_backend ? (
            <RecipeRow
              label="Attention"
              value={formatResolvedValue("attention_backend", image.attention_backend)}
            />
          ) : null}
          {image.transformer_cache ? (
            <RecipeRow
              label="Step cache"
              value={formatResolvedValue("transformer_cache", image.transformer_cache)}
            />
          ) : null}
          {image.cpu_offload ? (
            <RecipeRow label="CPU offload" value={formatResolvedValue("cpu_offload", true)} />
          ) : null}
          {image.baked_loras?.length ? (
            <RecipeRow label="Baked" value={image.baked_loras.join(", ")} wrap />
          ) : null}
          {image.loras?.length ? (
            <RecipeRow label="LoRAs" value={image.loras.join(", ")} wrap />
          ) : null}
          {image.controlnet ? <RecipeRow label="ControlNet" value={image.controlnet} wrap /> : null}
          <RecipeRow label="Workflow" value={recipeWorkflowLabel(image.workflow)} />
          {image.strength != null &&
          (image.workflow === "img2img" ||
            image.workflow === "edit" ||
            image.workflow === "inpaint" ||
            image.workflow === "upscale") ? (
            <RecipeRow label="Strength" value={String(image.strength)} />
          ) : null}
          {image.upscale != null && image.workflow === "upscale" ? (
            <RecipeRow label="Upscale" value={`${image.upscale}×`} />
          ) : null}
          {image.reference_image_count != null &&
          (image.workflow === "reference" || image.workflow === "edit") ? (
            <RecipeRow label="References" value={String(image.reference_image_count)} />
          ) : null}
          {image.localized_edit ? (
            <RecipeRow
              label="Edit mode"
              value={image.localized_edit.charAt(0).toUpperCase() + image.localized_edit.slice(1)}
            />
          ) : null}
          <RecipeRow label="Size" value={`${image.width} × ${image.height}`} />
          <RecipeRow label="Steps" value={String(image.steps)} />
          <RecipeRow label="Guidance" value={String(image.guidance)} />
          <RecipeRow label="Seed" value={String(image.seed)} mono />
        </div>
        <div className="shrink-0 border-t border-border/60 px-3 py-2.5">
          <Button size="sm" className="w-full gap-1.5" onClick={() => onRestore(image)}>
            <HugeiconsIcon icon={Refresh01Icon} className="size-4" />
            Restore these settings
          </Button>
        </div>
      </PopoverContent>
    </Popover>
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

/** What the LOADED model runs, from status; the Advanced selects show what was asked for. */
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
      {status.component_files && Object.keys(status.component_files).length > 0 ? (
        <BuildRow
          label="Text encoder / VAE files"
          value={Object.entries(status.component_files)
            .map(([component, file]) => `${component}: ${file}`)
            .join(", ")}
        />
      ) : null}
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
            // sd.cpp reports no attention backend, so "Native SDPA" would be wrong there.
            :
              isNativeEngineStatus(status)
              ? "sd.cpp built-in"
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

type Busy = "loading" | "unloading" | "generating" | null;
type ImageLoadOptions = { kind: "gguf" | "single_file" | "pipeline"; filename?: string; displayRepoId?: string };
type LastLoad = { repoId: string } & ImageLoadOptions & Pick<RememberedImageModel, "textEncoderFiles" | "vaeFile">;

type PickRevert = {
  prev: string | null;
  steps: number;
  guidance: number;
  commitRecipeClaim?: () => void;
  releaseRecipeClaim?: () => void;
  // A field the user changed after the pick is theirs, not ours to put back.
  appliedSteps?: number;
  appliedGuidance?: number;
};

type LoadAdvanced = Pick<
  DiffusionLoadRequest,
  | "cpu_offload"
  | "speed_mode"
  | "transformer_quant"
  | "text_encoder_quant"
  | "attention_backend"
  | "memory_mode"
  | "transformer_cache"
  | "family_override"
  | "loras"
  | "gpu_ids"
  | "text_encoder_file"
  | "vae_file"
>;

function openImageLabel(t: ReturnType<typeof useT>, prompt: string): string {
  const text = shortPrompt(prompt);
  return text ? t("library.viewer.openImageNamed", { prompt: text }) : t("library.viewer.openImage");
}

export function ImagesPage({
  active = true,
  onInitialReady,
}: {
  active?: boolean;
  onInitialReady?: () => void;
}) {
  const t = useT();
  const initialReadySent = useRef(false);
  const [rememberedModel, setRememberedModel] = useState(readImageModel);
  const pendingRecalledGeneration = useRef<{ model: RememberedImageModel; load: number; workflow: WorkflowId; allowOversized?: boolean } | null>(null);
  const { isMobile, pinned } = useSidebar();
  const hostClass = useHostClass();
  const denseQuantSchemes = useDenseQuantSchemes();
  const nvfp4Diffusion = useNvfp4Diffusion();
  const nvfp4DiffusionKnown = useNvfp4DiffusionKnown();
  const imageModels = useImageModels(hostClass, denseQuantSchemes);
  const { rootStyle: railRootStyle } = useMediaRailWidth("images");
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  const [prompts, setPrompts] = useState<Record<WorkflowId, string>>(() =>
    Object.fromEntries(
      WORKFLOW_TABS.map(({ id }) => [id, readLastPrompt(`images:${id}`)]),
    ) as Record<WorkflowId, string>,
  );
  const [examplesDismissed, setExamplesDismissed] = useState<Record<WorkflowId, boolean>>(() =>
    Object.fromEntries(
      WORKFLOW_TABS.map(({ id }) => [id, isExampleDismissed(`images:${id}`)]),
    ) as Record<WorkflowId, boolean>,
  );
  const setPromptFor = useCallback((id: WorkflowId, next: SetStateAction<string>) => {
    setPrompts((prev) => ({
      ...prev,
      [id]: typeof next === "function" ? next(prev[id]) : next,
    }));
  }, []);
  const setPrompt = useCallback(
    (next: SetStateAction<string>) =>
      setPromptFor(useImageWorkflowStore.getState().workflow, next),
    [setPromptFor],
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
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [aspect, setAspect] = useState("1:1");
  const [portrait, setPortrait] = useState(false);
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const pickRecipeSuperseded = useRef<(() => boolean) | null>(null);
  // The recipe the last pick applied, so a load that reveals the family can replace a fallback.
  const pickDefaults = useRef<{ steps: number; guidance: number } | null>(null);
  // Put back everything a pick optimistically applied. Setters are stable, so this never re-renders on its own.
  const revertPick = useCallback((r: PickRevert) => {
    setQuant(r.prev);
    setPendingModelDefaults(null);
    pickDefaults.current = null;
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
  // Without this the Default preset reads as "modified" for the whole download.
  const [pendingModelDefaults, setPendingModelDefaults] = useState<{
    steps: number;
    guidance: number;
  } | null>(null);
  const [seed, setSeed] = useState("");
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);
  const workflow = useImageWorkflowStore((s) => s.workflow);
  const prompt = prompts[workflow];
  const setWorkflow = useImageWorkflowStore((s) => s.setWorkflow);
  const supported = useImageWorkflowStore((s) => s.supported);
  const setSupported = useImageWorkflowStore((s) => s.setSupported);
  const [initImage, setInitImage] = useState<string | null>(null);
  const [strength, setStrength] = useState(0.6);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [brushPct, setBrushPct] = useState(8);
  const [maskResetKey, setMaskResetKey] = useState(0);
  const [extendPct, setExtendPct] = useState(25);
  const [extendSides, setExtendSides] = useState<ExtendSides>({
    left: true,
    right: true,
    top: true,
    bottom: true,
  });
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [upscaleStrength, setUpscaleStrength] = useState(0.35);
  // "" holds a cleared slot so others do not renumber.
  const [referenceImages, setReferenceImages] = useState<string[]>([]);
  const [referenceResolution, setReferenceResolution] = useState<number | null>(null);
  const [editSizing, setEditSizing] = useState<EditSizing>("source");
  const [matchResolution, setMatchResolution] = useState(1024);
  const [localizedMode, setLocalizedMode] = useState<LocalizedEditMode | null>(null);
  const [localizedLayer, setLocalizedLayer] = useState<string | null>(null);
  const [localizedColor, setLocalizedColor] = useState(ANNOTATION_COLORS[0].value);
  const [localizedColors, setLocalizedColors] = useState<string[]>([]);
  const [localizedResetKey, setLocalizedResetKey] = useState(0);
  const [loras, setLoras] = useState<LoraSpecInput[]>([]);
  const [availableLoras, setAvailableLoras] = useState<DiffusionLoraInfo[]>([]);
  const pageMode = useImageWorkflowStore((s) => s.pageMode);
  const setPageMode = useImageWorkflowStore((s) => s.setPageMode);
  const tourSteps = useMemo(
    () => buildImagesTourSteps({ pageMode }),
    [pageMode],
  );
  const tour = useGuidedTourController({
    id: "images",
    steps: tourSteps,
    enabled: active,
  });
  const [trainFamilies, setTrainFamilies] = useState<TrainFamilyOption[]>([]);
  const [trainFamilyName, setTrainFamilyName] = useState("flux.1");
  const [trainBaseChoice, setTrainBaseChoice] = useState("");
  const [loraRefreshKey, setLoraRefreshKey] = useState(0);
  const [controlnetId, setControlnetId] = useState<string>("");
  const [controlImage, setControlImage] = useState<string | null>(null);
  const [controlType, setControlType] = useState<string>("passthrough");
  const [controlStrength, setControlStrength] = useState(0.7);
  const [availableControlNets, setAvailableControlNets] = useState<DiffusionControlNetInfo[]>([]);
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_images_advanced_open",
  );
  const [allowOversized, setAllowOversized] = usePersistedToggle(
    "unsloth_images_allow_oversized",
  );
  const [livePreviewOff, setLivePreviewOff] = usePersistedToggle("unsloth_images_live_preview_off");
  const livePreview = !livePreviewOff;
  const oversizedOnce = useRef(false);
  const [oversizedRetryQueued, setOversizedRetryQueued] = useState(false);
  const [modelSelectionAction, setModelSelectionAction] = useState<"load" | "download">("load");
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  const [transformerQuant, setTransformerQuant] = useState<
    "none" | "auto" | "int8" | "fp8" | "nvfp4" | "mxfp8"
  >("auto");
  const [textEncoderQuant, setTextEncoderQuant] = useState<
    "auto" | NonNullable<DiffusionLoadRequest["text_encoder_quant"]>
  >("auto");
  const [attentionBackend, setAttentionBackend] = useState<"auto" | "native" | "cudnn" | "flash3" | "sage">(
    "auto",
  );
  useEffect(() => {
    setTransformerQuant((v) => nvfp4SelectionFallback(v, nvfp4DiffusionKnown, nvfp4Diffusion));
    setTextEncoderQuant((v) => nvfp4SelectionFallback(v, nvfp4DiffusionKnown, nvfp4Diffusion));
  }, [nvfp4Diffusion, nvfp4DiffusionKnown, transformerQuant, textEncoderQuant]);
  const [memoryMode, setMemoryMode] = useState<"auto" | "fast" | "balanced" | "low_vram">("auto");
  const [textEncoderFiles, setTextEncoderFiles] = useState("");
  const [vaeFile, setVaeFile] = useState("");
  // "auto", or the physical index to pin this load to; offered only on a multi-card CUDA/ROCm
  // host. Persisted, unlike the selects around it: status carries the device a pipeline is on
  // but not which card, so a refresh would reset it to Auto. A stale id is dropped on send.
  const [selectedGpu, setSelectedGpu] = usePersistedChoice(
    "unsloth_image_gpu_choice",
    "auto",
  );
  const gpuChoices = useDiffusionGpuChoices();
  const [transformerCache, setTransformerCache] = useState<"auto" | "off" | "fbcache" | "static">("auto");
  const [cpuOffload, setCpuOffload] = useState(false);
  // The last load descriptor, so "Reapply" can reload the same model with new advanced options without re-picking it.
  const lastLoad = useRef<LastLoad | null>(null);
  // Render-safe mirror of whether a page-initiated load supplied a complete Reapply target.
  const [canReapply, setCanReapply] = useState(false);
  const seededResident = useRef<string | null>(null);

  const [busy, setBusy] = useState<Busy>(null);
  const [genDone, setGenDone] = useState<number | null>(null);
  const [stopping, setStopping] = useState(false);
  const [genStep, setGenStep] = useState<DiffusionGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // Background tabs clamp setInterval, so returning fires one immediate poll.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  const { familyOverride, setFamilyOverride, familySelect, opaqueKind, selectorModelId } = useFamilyOverride(status, status?.supported_families);
  const conditioning = status?.loaded ? (status.conditioning ?? null) : null;
  const sizeLimits = useMemo(() => sizeLimitsFrom(conditioning), [conditioning]);
  const unifiedEdit = Boolean(conditioning?.unified_edit);
  const referenceResolutions = conditioning?.reference_resolutions ?? [];
  const maxExtras = maxAdditionalImages(conditioning, workflow === "edit" && unifiedEdit ? localizedMode : null);
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [aspectOpen, setAspectOpen] = useState(false);
  const [images, setImages] = useState<GalleryImage[]>(() => galleryCache.images);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    galleryCache.srcById.toRecord(),
  );
  const [thumbById, setThumbById] = useState<Record<string, string>>(() =>
    galleryCache.thumbById.toRecord(),
  );
  const [srcErrors, setSrcErrors] = useState<Record<string, boolean>>({});
  const loadingMore = useRef(false);
  const stripRef = useRef<HTMLDivElement | null>(null);
  const visibleIds = useRef<Set<string>>(new Set());
  // Tab switches keep this mounted, so a batch keeps generating off-tab.
  const isMounted = useRef(true);
  // The backend cancel only reaches the current denoise, so later runs must be stopped here.
  const cancelRequested = useRef(false);
  // True only once the backend answered {cancelled: true}; otherwise later errors are real failures.
  const cancelAcked = useRef(false);
  const runToken = useRef(0);
  // Without it a late duplicate Stop hits whatever runs next. A token, since the clear is async.
  const cancelInFlight = useRef<number | null>(null);
  // A pending cancel can outlive its run via 401 refresh-and-replay and hit a newer run.
  const cancelAbort = useRef<AbortController | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loadToastId = useRef<string | number | null>(null);
  const lastLoadSig = useRef<string | null>(null);
  // `{ prev }` distinguishes "revert to null" from "nothing pending"; also carries the recipe.
  const quantRevert = useRef<PickRevert | null>(null);
  // Staging does not set `busy`, so a second pick can overwrite quantRevert meanwhile.
  const stagedQuantRevert = useRef<PickRevert | null>(null);
  const pickSeq = useRef(0);
  // handleLoad overwrites lastLoad.current at load start.
  const lastLoadRevert = useRef<{ prev: typeof lastLoad.current } | null>(null);
  const pendingDeploy = useRef<{ loraId: string; family: string } | null>(null);
  // Lazy state, not a ref, since a ref cannot be written during render.
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
  const residentDefaults = residentDefaultsKey(status?.repo_id ?? "", status?.base_repo, status?.resolved?.family_override);
  const { steps: residentSteps, guidance: residentGuidance } = residentRecipeFor(
    residentDefaults,
    status?.generation_defaults,
  );
  const imageDefaultRecipe = useMemo<ImageGenerationPresetParams>(() => {
    const recommended = pendingModelDefaults ?? { steps: residentSteps, guidance: residentGuidance };
    // Reset restores the resident build's canvas, the same one the seed above applied. A constant
    // here would quietly undo it and put a 24 GB card back over its budget.
    const size = resolutionFor(status?.base_repo ?? status?.repo_id ?? "", {
      modelKind: status?.model_kind,
      transformerQuant: status?.transformer_quant,
      transformerQuantSource: status?.resolved?.transformer_quant?.source,
    });
    return {
      negativePrompt: "",
      width: size.width,
      height: size.height,
      steps: recommended.steps,
      guidance: recommended.guidance,
      batchSize: 1,
      runs: 1,
    };
  }, [
    pendingModelDefaults,
    residentSteps,
    residentGuidance,
    status?.base_repo,
    status?.repo_id,
    status?.model_kind,
    status?.transformer_quant,
    status?.resolved?.transformer_quant?.source,
  ]);
  const applyImagePresetParams = useCallback((params: ImageGenerationPresetParams) => {
    setNegativePrompt(params.negativePrompt);
    // A negative prompt in effect must be visible.
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
    (repoId: string, effectiveFamilyOverride = familyOverride, forceLoad = false) => {
      if (modelSelectionAction === "download" && !forceLoad) return;
      const revert = quantRevert.current;
      if (revert && !revert.releaseRecipeClaim) {
        const claim = claimImageRecipe();
        revert.commitRecipeClaim = claim.commit;
        revert.releaseRecipeClaim = claim.release;
      }
      const claimedAt = imageFormClaimId();
      pickRecipeSuperseded.current = () => imageFormClaimId() !== claimedAt;
      const recommended = defaultsFor(defaultsKeyFor(repoId, effectiveFamilyOverride));
      pickDefaults.current = recommended;
      setPendingModelDefaults(recommended);
      setSteps(recommended.steps);
      setGuidance(recommended.guidance);
      if (revert) {
        revert.appliedSteps = recommended.steps;
        revert.appliedGuidance = recommended.guidance;
      }
    },
    [claimImageRecipe, familyOverride, imageFormClaimId, modelSelectionAction],
  );

  const dismissLoadToast = useCallback(() => {
    if (loadToastId.current != null) toast.dismiss(loadToastId.current);
    loadToastId.current = null;
  }, []);
  const pickToast = useDiffusionPickToast();

  const cancelLoadRef = useRef<() => void>(() => {});
  const cancelLoadFromToast = useCallback(() => cancelLoadRef.current(), []);
  // In-flight requests compare against this and discard their result after a cancel.
  const cancelSeq = useRef(0);
  // The compensating unload carries no identity, so it must not fire once a newer load owns the page.
  const loadSeq = useRef(0);
  // Settles only after handleLoad finishes, since begin_load refuses a second load while one is live.
  const pendingStart = useRef<Promise<unknown> | null>(null);

  // Set when the compensating unload failed, so the cancelled load is STILL running.
  const loadTrackingRestored = useRef(false);

  const dropResidentState = useCallback(() => {
    pickGuard.cancel();
    pickToast.dismissAll();
    // Clearing the timer stops the next tick, not requests already awaiting a response.
    cancelSeq.current += 1;
    // Otherwise a discarded adapter would be mixed into an unrelated model.
    pendingDeploy.current = null;
    if (pollTimer.current) clearTimeout(pollTimer.current);
    pollTimer.current = null;
    dismissLoadToast();
    lastLoadSig.current = null;
    lastLoad.current = null;
    setCanReapply(false);
    // Stopping the poll skips its revert branch, so revert here as it would.
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [dismissLoadToast, pickGuard, pickToast, revertPick]);

  useEffect(() => {
    galleryCache.images = images;
    galleryCache.hasMore = hasMore;
    galleryCache.selectedId = selectedId;
    galleryCache.quant = quant;
  }, [images, hasMore, selectedId, quant]);

  // Tracked in a ref so it skips first load and unload (a restore can precede the load).
  const loraCapable = Boolean(status?.loaded && status?.supports_lora);
  const prevLoraFamilyRef = useRef<string | null | undefined>(undefined);
  const bakedLorasOnLoad = useRef(false);
  useEffect(() => {
    if (!loraCapable) {
      // Keep the selection: it may have just been restored.
      setAvailableLoras([]);
      return;
    }
    const fam = status?.family ?? null;
    const prev = prevLoraFamilyRef.current;
    if (prev != null && prev !== fam) {
      setLoras([]);
    }
    prevLoraFamilyRef.current = fam;
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
        // Clear only the options: selections are valid without being in the catalog.
        if (!cancelled) setAvailableLoras([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loraCapable, status?.family, loraRefreshKey]);

  // torchao int8/fp8 builds take adapters only at load time, so drop the selection rather than 400.
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

  const controlnetCapable = Boolean(status?.loaded && status?.supports_controlnet);
  const negativeCapable = status?.supports_negative_prompt !== false;
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

  const controlTypeOptions = useMemo(() => {
    const cn = availableControlNets.find((c) => c.id === controlnetId);
    const types = cn?.control_types?.length ? cn.control_types : ["passthrough", "canny"];
    return types;
  }, [availableControlNets, controlnetId]);

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
  const selectedThumb = selected ? thumbById[selected.id] : undefined;
  const livePreviewSrc =
    busy === "generating" && livePreview ? (genStep?.preview ?? undefined) : undefined;
  const [viewerId, setViewerId] = useState<string | null>(null);
  const viewerImage = viewerId ? (images.find((image) => image.id === viewerId) ?? null) : null;
  const viewerSrc = viewerImage ? srcById[viewerImage.id] : undefined;
  if (viewerId && (!active || !viewerImage)) setViewerId(null);
  const viewerIdRef = useRef<string | null>(null);
  useEffect(() => {
    viewerIdRef.current = viewerId;
  }, [viewerId]);
  const openViewer = () => selected && selectedSrc && setViewerId(selected.id);
  const navigateToChat = useNavigate();
  const revealLabel = useRevealLabel();

  const ensureSrc = useCallback(async (image: GalleryImage) => {
    if (galleryCache.srcById.has(image.id) || galleryCache.inflight.has(image.id)) return;
    galleryCache.inflight.add(image.id);
    setSrcErrors((prev) => {
      if (!prev[image.id]) return prev;
      const next = { ...prev };
      delete next[image.id];
      return next;
    });
    try {
      const { url, bytes } = await fetchGalleryObjectUrl(image.url);
      if (galleryCache.deleted.has(image.id)) {
        URL.revokeObjectURL(url);
        return;
      }
      galleryCache.srcById.set(image.id, url, bytes);
      const evicted = galleryCache.srcById.prune(
        new Set([
          image.id,
          ...visibleIds.current,
          galleryCache.selectedId ?? "",
          viewerIdRef.current ?? "",
        ]),
      );
      setSrcById((prev) => {
        const next = { ...prev, [image.id]: url };
        for (const id of evicted) delete next[id];
        return next;
      });
    } catch {
      if (!galleryCache.deleted.has(image.id)) {
        setSrcErrors((prev) => ({ ...prev, [image.id]: true }));
      }
    } finally {
      galleryCache.inflight.delete(image.id);
    }
  }, []);

  const ensureThumb = useCallback(async (image: GalleryImage) => {
    if (
      galleryCache.thumbById.has(image.id) ||
      galleryCache.srcById.has(image.id) ||
      galleryCache.thumbInflight.has(image.id)
    )
      return;
    galleryCache.thumbInflight.add(image.id);
    try {
      const { url, bytes } = await fetchGalleryObjectUrl(galleryThumbnailUrl(image.url));
      if (galleryCache.deleted.has(image.id)) {
        URL.revokeObjectURL(url);
        return;
      }
      galleryCache.thumbById.set(image.id, url, bytes);
      const evicted = galleryCache.thumbById.prune(
        new Set([image.id, ...visibleIds.current, galleryCache.selectedId ?? ""]),
      );
      setThumbById((prev) => {
        const next = { ...prev, [image.id]: url };
        for (const id of evicted) delete next[id];
        return next;
      });
    } catch {
      // Leave it without a src; the tile shows a placeholder.
    } finally {
      galleryCache.thumbInflight.delete(image.id);
    }
  }, []);

  const stripEpoch = useRef(0);
  const pageEpoch = useRef(0);
  const resyncSeq = useRef(0);
  // The epoch is an EDGE, so a page starting after the bump sees it unchanged.
  const pendingShelfMutations = useRef(0);

  const loadGallery = useCallback(async () => {
    try {
      // Fenced: tiles are actionable during the load, and a pre-pin snapshot would undo an action.
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
      if (typeof IntersectionObserver === "undefined") {
        page.images.forEach((image) => void ensureThumb(image));
      }
    } catch {
      // Best-effort: a failed gallery load should not block the page.
    }
  }, [ensureThumb]);

  const loadMore = useCallback(async () => {
    if (loadingMore.current || !galleryCache.hasMore) return;
    loadingMore.current = true;
    try {
      // An archive during this GET shifts a record past the page boundary, where no page returns it.
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
        page.images.forEach((image) => void ensureThumb(image));
      }
    } catch {
      // transient; the user can scroll again to retry
    } finally {
      loadingMore.current = false;
    }
  }, [ensureThumb]);

  useEffect(() => {
    const root = stripRef.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.imageId;
          if (!id) continue;
          if (!entry.isIntersecting) {
            visibleIds.current.delete(id);
            continue;
          }
          visibleIds.current.add(id);
          galleryCache.srcById.touch(id);
          galleryCache.thumbById.touch(id);
          const image = images.find((i) => i.id === id);
          if (image) void ensureThumb(image);
        }
      },
      // rootMargin applies to the ROOT box only, so the root must be the scrolling strip.
      { root, rootMargin: "0px 600px" },
    );
    for (const tile of root.querySelectorAll("[data-image-id]")) io.observe(tile);
    return () => io.disconnect();
  }, [images, ensureThumb]);

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
      galleryCache.srcById.delete(id);
      galleryCache.thumbById.delete(id);
      galleryCache.deleted.add(id);
      setSrcErrors((prev) => {
        if (!prev[id]) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setSrcById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setThumbById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
    visibleIds.current.delete(id);
    stripEpoch.current += 1;
    // No setSelectedId inside a setImages updater: that would run a side effect during dispatch.
    const at = galleryCache.images.findIndex((i) => i.id === id);
    const next = removeGalleryItem(galleryCache.images, id);
    galleryCache.images = next;
    setImages(next);
    setSelectedId((cur) => nextSelectedId(next, id, cur, at));
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      // Held for the whole round trip: the server shortens the shelf while processing this.
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

  /** Unpinning can drop an image past the window and promote an unloaded one into it. */
  const resyncWindow = useCallback(
    async (count: number, stillFresh?: () => boolean) => {
      const ticket = (resyncSeq.current += 1);
      for (let attempt = 0; attempt < RESYNC_MAX_ATTEMPTS; attempt += 1) {
        const paged = pageEpoch.current;
        const wanted = Math.max(count, galleryCache.images.length, PAGE_SIZE);
        const collected: GalleryImage[] = [];
        let more = false;
        while (collected.length < wanted) {
          const page = await getGallery(
            collected.length,
            Math.min(PAGE_SIZE, wanted - collected.length),
          );
          collected.push(...page.images);
          more = page.has_more;
          if (!page.has_more || page.images.length === 0) break;
        }
        if (stillFresh && !stillFresh()) return;
        if (resyncSeq.current !== ticket) return;
        if (pageEpoch.current !== paged) continue;
        galleryCache.images = collected;
        galleryCache.hasMore = more;
        setImages(collected);
        setHasMore(more);
        if (typeof IntersectionObserver === "undefined") {
          collected.forEach((image) => void ensureThumb(image));
        }
        return;
      }
    },
    [ensureThumb],
  );

  // This page stays mounted across routes, so an archive restore needs a resync.
  useEffect(
    () =>
      subscribeGalleryChanged("images", () => {
        // Bumped FIRST so in-flight reads are discarded.
        stripEpoch.current += 1;
        const epoch = stripEpoch.current;
        void resyncWindow(
          galleryCache.images.length,
          () => stripEpoch.current === epoch,
        ).catch(() => void loadGallery());
      }),
    [loadGallery, resyncWindow],
  );

  const { isFavorite, toggleFavorite } = useLibraryFavorites();
  const pinAttempt = useRef(new Map<string, number>());
  const pinSeq = useRef(0);

  const handleTogglePin = useCallback(
    async (id: string, pinned: boolean) => {
      const loadedCount = galleryCache.images.length;
      const orderBefore = pinnedOrder(galleryCache.images);
      // A per-attempt token, not the target boolean: pin, unpin, pin stores true twice.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      const epoch = stripEpoch.current;
      setImages((prev) => {
        const next = applyPin(prev, id, pinned);
        galleryCache.images = next;
        return next;
      });
      try {
        // The server stamps `pinned_at` on PATCH, so requests run one at a time to follow click order.
        await serializeById("image-pin", () => setGalleryImageFlags(id, { pinned }));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to pin image");
        if (pinAttempt.current.get(id) === attempt) {
          pinAttempt.current.delete(id);
          stripEpoch.current += 1;
          setImages((prev) => {
            const next = pinned
              ? applyPin(prev, id, false)
              : restorePinOrder(prev, id, orderBefore);
            galleryCache.images = next;
            return next;
          });
        }
        return;
      }
      if (pinAttempt.current.get(id) !== attempt) return;
      pinAttempt.current.delete(id);
      if (!pinned && loadedCount > 0) {
        try {
          await resyncWindow(loadedCount, () => stripEpoch.current === epoch);
        } catch {
          // Best-effort: the strip is still usable, just possibly short one image until a reload.
        }
      }
    },
    [resyncWindow],
  );

  const handleMove = useCallback(
    async (id: string, afterId: string | null) => {
      const next = moveGalleryItem(galleryCache.images, id, afterId);
      if (next === galleryCache.images) return;
      const guessedPinned = Boolean(next.find((i) => i.id === id)?.pinned);
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      galleryCache.images = next;
      setImages(next);
      try {
        const record = await serializeById("image-pin", () => moveGalleryImage(id, afterId));
        if (pinAttempt.current.get(id) !== attempt) return;
        pinAttempt.current.delete(id);
        setImages((prev) => {
          const patched = prev.map((i) =>
            i.id === id ? { ...i, pinned: record.pinned, order_at: record.order_at } : i,
          );
          const out =
            Boolean(record.pinned) === guessedPinned ? patched : sortGalleryItems(patched);
          galleryCache.images = out;
          return out;
        });
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to move image");
        stripEpoch.current += 1;
        const epoch = stripEpoch.current;
        try {
          await resyncWindow(galleryCache.images.length, () => stripEpoch.current === epoch);
        } catch {
          void loadGallery();
        }
      }
    },
    [resyncWindow, loadGallery],
  );
  const handleQuickDownload = useCallback(
    async (image: GalleryImage) => {
      const src = srcById[image.id];
      if (src) {
        await downloadImage(src, image, "png");
        return;
      }
      try {
        const blob = await fetchGalleryBlob(image.url);
        await downloadFile(blob, exportFilename(image, "png"), blob.type);
      } catch (error) {
        if (isDownloadCancelled(error)) return;
        toast.error("Could not save image", {
          description: error instanceof Error ? error.message : undefined,
        });
      }
    },
    [srcById],
  );
  const stripReorder = useStripReorder((id, afterId) => void handleMove(id, afterId));

  const handleArchive = useCallback(
    async (id: string) => {
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

  const restoreSettings = useCallback((image: GalleryImage) => {
    const restoredNegative = image.guidance > 0 ? (image.negative_prompt ?? "") : "";
    setNegativePrompt(restoredNegative);
    if (restoredNegative) setNegativeOpen(true);
    setSteps(image.steps);
    setGuidance(image.guidance);
    // Use the BASE batch seed, or a batch replay advances it again.
    setSeed(String(image.batch_seed ?? image.seed));
    const restored = restorableSize(image.width, image.height, image.workflow, sizeLimits);
    setWidth(restored.width);
    setHeight(restored.height);
    setBatchSize(image.batch_size ?? 1);
    const m = matchAspect(restored.width, restored.height);
    setAspect(m.key);
    setPortrait(m.portrait);
    // Split on the LAST colon so an id containing ':' survives.
    const restoredLoras: LoraSpecInput[] = [];
    for (const entry of image.loras ?? []) {
      const idx = entry.lastIndexOf(":");
      if (idx <= 0) continue;
      const id = entry.slice(0, idx);
      const weight = Number(entry.slice(idx + 1));
      if (id && Number.isFinite(weight)) restoredLoras.push({ id, weight });
    }
    setLoras(restoredLoras);
    if (typeof image.strength === "number") {
      if (image.workflow === "upscale") setUpscaleStrength(image.strength);
      else setStrength(image.strength);
    }
    if (typeof image.upscale === "number") setUpscaleFactor(image.upscale);
    // Conditioning images are not persisted. Edit and Reference reopen so Generate stays blocked.
    const reopened: WorkflowId =
      image.workflow === "edit" ? "edit" : image.workflow === "reference" ? "reference" : "create";
    setWorkflow(reopened);
    setPromptFor(reopened, image.prompt);
    setInitImage(null);
    setMaskImage(null);
    setReferenceImages(
      reopened === "create" ? [] : Array.from({ length: image.reference_image_count ?? 0 }, () => ""),
    );
    if (typeof image.reference_resolution === "number") {
      setReferenceResolution(image.reference_resolution);
    }
    setLocalizedMode(reopened === "edit" ? (image.localized_edit ?? null) : null);
    setLocalizedLayer(null);
    setLocalizedColors([]);
    if (reopened === "edit") setEditSizing("custom");
    setControlnetId("");
    setControlImage(null);
    const rescaled =
      restored.width !== image.width || restored.height !== image.height
        ? { description: `Size scaled to ${restored.width} × ${restored.height} to fit the ${MIN_DIM}-${sizeLimits.maxSide} range.` }
        : undefined;
    const conditioned =
      restoreInputsNote(image) ?? CONDITIONED_WORKFLOW_INPUTS[image.workflow ?? ""];
    if (conditioned) {
      toast.success(`Settings restored. Add ${conditioned} again to reproduce this image.`, rescaled);
    } else {
      toast.success("Settings restored to inputs", rescaled);
    }
  }, [setPromptFor, setWorkflow, sizeLimits]);

  const ratioHW = (a: number, b: number) => (portrait ? a / b : b / a);
  const changeAspect = (key: string) => {
    setAspect(key);
    if (key === "custom") return;
    const [a, b] = ASPECT_RATIOS[key];
    setHeight(snapDim(width * ratioHW(a, b), sizeLimits));
  };
  const changeWidth = (v: number) => {
    setWidth(v);
    if (aspect === "custom") return;
    const [a, b] = ASPECT_RATIOS[aspect];
    setHeight(snapDim(v * ratioHW(a, b), sizeLimits));
  };
  const changeHeight = (v: number) => {
    setHeight(v);
    if (aspect === "custom") return;
    const [a, b] = ASPECT_RATIOS[aspect];
    setWidth(snapDim(v / ratioHW(a, b), sizeLimits));
  };
  const flipDimensions = () => {
    setWidth(height);
    setHeight(width);
    setPortrait((p) => !p);
  };

  // Only the newest status ticket may write: an older read can answer after an eject.
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

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

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

  // The indicator eject does not run handleUnload, so mirror it minus the unload.
  useEffect(
    () =>
      subscribeModelEjected("image", () => {
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

  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
    setAspectOpen(false);
  }, [active]);

  const pollLoadProgress = useCallback(async () => {
    const seq = cancelSeq.current;
    try {
      const p = await getDiffusionLoadProgress();
      if (seq !== cancelSeq.current) return;
      if (p.phase === "ready") {
        dismissLoadToast();
        const ticket = ++statusTicket.current;
        const loaded = await getDiffusionStatus();
        if (seq !== cancelSeq.current) {
          // Cancelled mid-read: drop it, the unload's own response is authoritative.
          return;
        }
        setStatusIfNewest(ticket, loaded);
        toast.success("Model loaded");
        if (lastLoad.current && matchesRememberedModel(lastLoad.current, loaded)) {
          const remembered = withEngagedFamily(lastLoad.current, loaded);
          rememberImageModel(remembered);
          setRememberedModel(remembered);
        }
        setBusy(null);
        // A fallback-recipe pick takes the loaded family recipe on an untouched form (else SDXL runs 9 steps, CFG 0).
        const loadedRecipe = loadedRecipeFor(
          pickDefaults.current,
          residentDefaultsKey(loaded.repo_id ?? "", loaded.base_repo, loaded.resolved?.family_override),
          loaded.generation_defaults,
        );
        pickDefaults.current = null;
        if (loadedRecipe && !pickRecipeSuperseded.current?.()) {
          setSteps((cur) => (cur === DEFAULT_GEN.steps ? loadedRecipe.steps : cur));
          setGuidance((cur) => (cur === DEFAULT_GEN.guidance ? loadedRecipe.guidance : cur));
        }
        // Load succeeded: the optimistic quant is now the real one, so drop the pending revert.
        quantRevert.current?.commitRecipeClaim?.();
        quantRevert.current = null;
        setPendingModelDefaults(null);
        lastLoadRevert.current = null;
        return;
      }
      if (p.phase === "error") {
        pendingRecalledGeneration.current = null;
        dismissLoadToast();
        reportLoadFailure(p.error, "Failed to load model");
        setBusy(null);
        // A load that failed AFTER starting leaves the previous pipeline loaded.
        if (quantRevert.current) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.prev != null);
          lastLoadRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        pendingRecalledGeneration.current = null;
        // Cancelled or evicted. Terminal, else this loop spins forever.
        dismissLoadToast();
        setBusy(null);
        if (quantRevert.current) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        if (lastLoadRevert.current) {
          lastLoad.current = lastLoadRevert.current.prev;
          setCanReapply(lastLoadRevert.current.prev != null);
          lastLoadRevert.current = null;
        }
        void refreshStatus();
        return;
      }
      // Include bytes_total: the estimate lands as a 0->real jump while the rest holds.
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

  // The unload failed, so restore the poll and toast. refreshStatus cannot: a first load is not resident.
  const restoreLoadTracking = useCallback(() => {
    loadTrackingRestored.current = true;
    setBusy("loading");
    lastLoadSig.current = null;
    loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, undefined, cancelLoadFromToast));
    void pollLoadProgress();
  }, [pollLoadProgress, cancelLoadFromToast]);

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
          void loadGallery();
          void refreshStatus();
          return;
        }
        setGenStep((prev) => {
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
  }, [loadGallery, refreshStatus]);

  useEffect(() => {
    void (async () => {
      await refreshStatus();
      // A backend load survives navigation, so resume tracking one in flight.
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

  // Seed from a resident model's recipe, else flux.1-dev generates garbage at 9 steps.
  const residentSeeded = useRef(false);
  useEffect(() => {
    const repoId = status?.loaded ? status.repo_id : null;
    if (!repoId) return;
    if (lastLoad.current) return;
    const seedKey = `${repoId}\0${residentDefaults}\0${residentSteps}\0${residentGuidance}`;
    if (seededResident.current === seedKey) return;
    seededResident.current = seedKey;
    // Only a full pipeline is reloadable by repo id; a resident GGUF has no filename.
    if (status?.model_kind === "pipeline") {
      lastLoad.current = { repoId, kind: "pipeline", displayRepoId: status.display_repo_id ?? undefined };
    }
    if (!residentSeeded.current) {
      residentSeeded.current = true;
      if (imagePresets.storedRecipe) return;
    }
    const d = { steps: residentSteps, guidance: residentGuidance };
    setPendingModelDefaults(null);
    setSteps(d.steps);
    setGuidance(d.guidance);
    // Read the canvas from the ENGAGED build, so a declined quant request keeps 1024.
    const size = resolutionFor(status?.base_repo ?? repoId, {
      modelKind: status?.model_kind,
      transformerQuant: status?.transformer_quant,
      transformerQuantSource: status?.resolved?.transformer_quant?.source,
    });
    setWidth(size.width);
    setHeight(size.height);
    const matched = matchAspect(size.width, size.height);
    setAspect(matched.key);
    setPortrait(matched.portrait);
  }, [
    imagePresets.storedRecipe,
    residentDefaults,
    residentSteps,
    residentGuidance,
    status?.display_repo_id,
    status?.loaded,
    status?.repo_id,
    status?.base_repo,
    status?.model_kind,
    status?.transformer_quant,
    status?.resolved?.transformer_quant?.source,
  ]);

  // Keyed on the load-time half of `resolved`: the backend rewrites the rest at generation time.
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
    const encoder = resolvedSelectValue(record.text_encoder_quant, (v) =>
      // "off" maps to Dense, NOT Default, or a pinned Dense would silently take the family scheme.
      (["auto", "none", "fp8", "fp8_dynamic", "int8", "nvfp4"] as const).find(
        (o) => o === v || (o === "none" && v === "off"),
      ) ?? null,
    );
    if (encoder) setTextEncoderQuant(encoder);
    const memory = resolvedSelectValue(record.memory_mode, (v) =>
      (["auto", "fast", "balanced", "low_vram"] as const).find((o) => o === v) ?? null,
    );
    if (memory) setMemoryMode(memory);
    const attention = resolvedSelectValue(record.attention_backend, (v) =>
      (["auto", "native", "cudnn", "flash3", "sage"] as const).find(
        (o) => o === v || `_native_${o}` === v,
      ) ?? null,
    );
    if (attention) setAttentionBackend(attention);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- resolvedKey stands for the record
  }, [resolvedKey]);

  const bakedLorasFor = useCallback(
    (repoId: string, preserveSelection = false): LoraSpecInput[] => {
      const sameTarget = repoId === (lastLoad.current?.repoId ?? status?.repo_id ?? null);
      if (!sameTarget && !preserveSelection) return [];
      return loras
        .map((l) => ({ id: l.id.trim(), weight: l.weight }))
        .filter((l) => l.id && l.weight > 0);
    },
    [loras, status?.repo_id],
  );

  const currentLoadAdvanced = useCallback(
    (repoId: string, familyOverrideRequired = true, preserveSelection = false): LoadAdvanced => {
      const baked = bakedLorasFor(repoId, preserveSelection);
      return {
        cpu_offload: cpuOffload,
        speed_mode: speedMode === "auto" ? undefined : speedMode,
        transformer_quant: transformerQuant === "auto" ? undefined : transformerQuant,
        text_encoder_quant: textEncoderQuant === "auto" ? undefined : textEncoderQuant,
        attention_backend: attentionBackend === "auto" ? undefined : attentionBackend,
        memory_mode: memoryMode === "auto" ? undefined : memoryMode,
        transformer_cache: transformerCache === "auto" ? undefined : transformerCache,
        family_override: familyOverrideRequired ? explicitFamily(familyOverride) : undefined,
        loras: baked.length > 0 ? baked : undefined,
        gpu_ids:
          selectedGpu !== "auto" &&
          gpuChoices.some((d) => String(d.index) === selectedGpu)
            ? [Number(selectedGpu)]
            : undefined,
        text_encoder_file: (() => {
          const files = splitComponentFileList(textEncoderFiles);
          return files.length > 0 ? files : undefined;
        })(),
        vae_file: vaeFile.trim() || undefined,
      };
    },
    [
      bakedLorasFor,
      cpuOffload,
      speedMode,
      transformerQuant,
      textEncoderQuant,
      attentionBackend,
      memoryMode,
      transformerCache,
      familyOverride,
      selectedGpu,
      gpuChoices,
      textEncoderFiles,
      vaeFile,
    ],
  );

  const handleLoad = useCallback(
    async (
      repoId: string,
      opts: ImageLoadOptions,
      // A staged download plans its files at pick time, so the load must use those values.
      pinned?: LoadAdvanced,
      pickToastId?: string,
    ): Promise<boolean> => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
      // Read BEFORE the start request: a Cancel's unload can reach the backend first and stop nothing.
      const startSeq = cancelSeq.current;
      const startLoad = ++loadSeq.current;
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
      // Snapshot first: a load that fails to START leaves the previous model resident.
      const prevLastLoad = lastLoad.current;
      // torchao int8/fp8 takes adapters only at load time, so a reload must keep the selection.
      const advanced = pinned ?? currentLoadAdvanced(repoId);
      const bakeLoras = advanced.loras ?? [];
      bakedLorasOnLoad.current = bakeLoras.length > 0;
      const componentFiles = componentFileFields(opts.kind, advanced.text_encoder_file, advanced.vae_file);
      lastLoad.current = {
        repoId,
        kind: opts.kind,
        filename: opts.filename,
        displayRepoId: opts.displayRepoId,
        textEncoderFiles: componentFiles.text_encoder_file,
        vaeFile: componentFiles.vae_file,
      };
      setCanReapply(true);
      lastLoadRevert.current = { prev: prevLastLoad };
      try {
        const startRequest = loadDiffusionModel({
          model_path: repoId,
          display_repo_id: opts.displayRepoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          hf_token: hfApiToken(getHfToken()),
          cpu_offload: advanced.cpu_offload,
          speed_mode: advanced.speed_mode,
          transformer_quant: sendsTransformerQuant(opts.kind, repoId)
            ? advanced.transformer_quant
            : undefined,
          text_encoder_quant: advanced.text_encoder_quant,
          attention_backend: advanced.attention_backend,
          memory_mode: advanced.memory_mode,
          transformer_cache: advanced.transformer_cache,
          family_override: advanced.family_override,
          loras: bakeLoras.length > 0 ? bakeLoras : undefined,
          gpu_ids: advanced.gpu_ids,
          ...componentFiles,
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
        // The cancel's unload may have landed before this load registered, so unload once more.
        if (startLoad === loadSeq.current) {
          try {
            await unloadDiffusionModel();
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
    [pollLoadProgress, refreshStatus, dismissLoadToast, currentLoadAdvanced, cancelLoadFromToast, pickToast],
  );

  const handleInitChange = useCallback((dataUrl: string | null) => {
    setInitImage(dataUrl);
    setMaskImage(null);
    setMaskResetKey((k) => k + 1);
  }, []);

  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: ImageLoadOptions;
    // Staging does not set `busy`, so the user can change settings while the download runs.
    advanced: LoadAdvanced;
    token: number;
    toastId?: string;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  handleLoadRef.current = handleLoad;
  // Both diffusion pages stay mounted and a load evicts the GPU holder, so defer until visible.
  const stagedLoadDeferred = useRef(false);
  // `owned` is read BEFORE the call, so a newer pick's label is left alone.
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
  const downloadOnlyPlans = useRef<StagedDownloadEntry[][]>([]);
  const pendingLoadEntries = useRef<StagedDownloadEntry[] | null>(null);
  const stagedPlan = useRef<"download" | { token: number } | null>(null);

  const { stage, progress: stagedProgress } = useStagedDownload({
    scopeId: "diffusion",
    onReady: () => {
      if (stagedPlan.current === "download") {
        finishDownloadOnlyPlan();
        return;
      }
      const finished =
        pendingStagedLoad.current?.token === stagedPlan.current?.token
          ? pendingStagedLoad.current
          : null;
      if (finished) {
        pendingLoadEntries.current = null;
      }
      stagedPlan.current = null;
      if (startQueuedDownload()) {
        pickToast.setPhase(finished?.toastId, "waiting");
        return;
      }
      resumePendingLoad();
    },
    onCancelled: () => {
      const cancelled = stagedPlan.current;
      if (cancelled === "download") {
        finishDownloadOnlyPlan();
        return;
      }
      stagedPlan.current = null;
      if (pendingStagedLoad.current?.token !== cancelled?.token) {
        startQueuedDownload();
        return;
      }
      pendingLoadEntries.current = null;
      pickToast.dismiss(pendingStagedLoad.current?.toastId);
      pendingStagedLoad.current = null;
      stagedLoadDeferred.current = false;
      // Staging starts no load, so the optimistic label must be reverted here, for this job's pick only.
      if (quantRevert.current && quantRevert.current === stagedQuantRevert.current) {
        revertPick(quantRevert.current);
        quantRevert.current = null;
      }
      stagedQuantRevert.current = null;
      startQueuedDownload();
    },
  });
  usePickToastProgress(pickToast, stagedProgress);

  function planKey(entries: StagedDownloadEntry[]) {
    return JSON.stringify(
      entries.map((e) => [e.repoId, e.ggufFilename ?? "", [...e.files].sort()]).sort(),
    );
  }

  function queuedDownloadKeys() {
    return new Set(downloadOnlyPlans.current.map(planKey));
  }

  function startQueuedDownload() {
    const next = downloadOnlyPlans.current[0];
    if (!next) return false;
    stagedPlan.current = "download";
    stage(next);
    return true;
  }

  function finishDownloadOnlyPlan() {
    stagedPlan.current = null;
    downloadOnlyPlans.current.shift();
    if (startQueuedDownload()) return;
    resumePendingLoad();
  }

  function resumePendingLoad() {
    const entries = pendingLoadEntries.current;
    const pending = pendingStagedLoad.current;
    if (!pending || !pickGuard.isLatest(pending.token)) {
      pendingLoadEntries.current = null;
      pickToast.dismiss(pending?.toastId);
      return;
    }
    if (entries) {
      stagedPlan.current = { token: pending.token };
      pickToast.setPhase(pending.toastId, "downloading", stage(entries));
    } else if (!active) {
      stagedLoadDeferred.current = true;
      pickToast.setPhase(pending.toastId, "ready");
    } else {
      runStagedLoad(pending);
    }
  }

  useEffect(() => {
    if (!active || !stagedLoadDeferred.current) return;
    stagedLoadDeferred.current = false;
    const pending = pendingStagedLoad.current;
    if (pending) runStagedLoad(pending);
  }, [active, runStagedLoad]);

  // Image models can need a separate text encoder/VAE repo, so plan on every Hub pick.
  const requestDownloadPlan = useCallback(
    (
      repoId: string,
      opts: ImageLoadOptions,
      advanced: LoadAdvanced,
    ) =>
      getDiffusionDownloadPlan({
        model_path: repoId,
        gguf_filename: opts.filename,
        model_kind: opts.kind,
        // Must match what handleLoad sends, or a gated base plans no companion.
        hf_token: hfApiToken(getHfToken()),
        cpu_offload: advanced.cpu_offload,
        speed_mode: advanced.speed_mode,
        transformer_quant: sendsTransformerQuant(opts.kind, repoId)
          ? advanced.transformer_quant
          : undefined,
        text_encoder_quant: advanced.text_encoder_quant,
        memory_mode: advanced.memory_mode,
        family_override: advanced.family_override,
        // A baked LoRA always runs the dense build path, which changes the file set.
        loras: advanced.loras,
        gpu_ids: advanced.gpu_ids,
        ...componentFileFields(opts.kind, advanced.text_encoder_file, advanced.vae_file),
      }),
    [],
  );

  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: ImageLoadOptions,
      source: ModelSelectorChangeMeta["source"] = "hub",
      token?: number,
      familyOverrideRequired = false,
      downloadSnapshot?: LoadAdvanced,
    ): Promise<boolean> => {
      const downloadOnly = downloadSnapshot !== undefined || modelSelectionAction === "download";
      // Plans resolve in response order and staging never sets `busy`. Download-only picks
      // supersede nothing, so they do not bump the sequence.
      const pick = downloadOnly ? pickSeq.current : ++pickSeq.current;
      // A pick that stages nothing never calls stage(), so the old staged job must die here.
      const owns = () => token === undefined || downloadOnly || pickGuard.holds(token);
      // A download-only pick must not retire a staged load, or that model never loads.
      if (!downloadSnapshot && !downloadOnly) {
        pendingStagedLoad.current = null;
        pendingLoadEntries.current = null;
        stagedLoadDeferred.current = false;
        stagedQuantRevert.current = null;
        pickToast.dismissAll();
        if (!owns()) return true;
      }
      const advanced = downloadSnapshot ?? currentLoadAdvanced(repoId, familyOverrideRequired);
      if (source !== "hub" && !downloadOnly) return handleLoadRef.current(repoId, opts, advanced);
      const pickToastId = downloadOnly ? undefined : pickToast.show();
      // Read before the await: a pick made meanwhile replaces quantRevert. Download-only owns no slot.
      const ownRevert = downloadSnapshot || downloadOnly ? null : quantRevert.current;
      // Acted on outside the try: a throwing refusal would otherwise fall through to the load.
      let incompatible: string | null = null;
      try {
        const plan = await requestDownloadPlan(opts.displayRepoId ?? repoId, opts, advanced);
        if (!downloadOnly && (pick !== pickSeq.current || !owns())) {
          pickToast.dismiss(pickToastId);
          return true;
        }
        if (downloadOnly && plan.plan_failed) {
          throw new Error("Required asset metadata is incomplete. Retry when it is available.");
        }
        incompatible = plan.incompatible_reason ?? null;
        if (!incompatible && plan.entries.length > 0) {
          if (!downloadOnly) {
            pendingStagedLoad.current = {
              repoId,
              opts,
              advanced,
              token: token ?? pickGuard.claim(),
              toastId: pickToastId,
            };
            stagedQuantRevert.current = ownRevert;
          }
          const entries = diffusionStagingEntries(plan.entries, repoId, opts);
          if (entries.length === 0) {
            pickToast.dismiss(pickToastId);
            if (downloadOnly) return true;
            pendingStagedLoad.current = null;
            return handleLoadRef.current(repoId, opts, advanced);
          }
          if (downloadOnly) {
            // Same files queued twice would download twice and the second start could come back "busy".
            if (queuedDownloadKeys().has(planKey(entries))) return true;
            downloadOnlyPlans.current.push(entries);
            if (stagedPlan.current === null) {
              stagedPlan.current = "download";
              stage(entries);
            }
          } else {
            pendingLoadEntries.current = entries;
            const pending = pendingStagedLoad.current;
            if (downloadOnlyPlans.current.length === 0 && pending) {
              stagedPlan.current = { token: pending.token };
              pickToast.setPhase(pickToastId, "downloading", stage(entries));
            } else {
              pickToast.setPhase(pickToastId, "queued");
            }
          }
          return true;
        }
      } catch (error) {
        if (downloadOnly) {
          toast.error("Could not plan the download", {
            description: error instanceof Error ? error.message : "Try selecting the model again.",
          });
          return true;
        }
      }
      // Re-checked: a plan rejected after a newer pick would otherwise reach the fallback load.
      if (!downloadOnly && (pick !== pickSeq.current || !owns())) {
        pickToast.dismiss(pickToastId);
        return true;
      }
      if (incompatible) {
        pickToast.dismiss(pickToastId);
        toast.error(incompatible);
        return downloadOnly;
      }
      if (downloadOnly) {
        toast.info("No downloads were planned for this selection");
        return true;
      }
      return handleLoadRef.current(repoId, opts, advanced, pickToastId);
    },
    [stage, currentLoadAdvanced, requestDownloadPlan, modelSelectionAction, pickGuard, revertPick, pickToast],
  );

  const resolveDownloadFootprint = useCallback(
    async (repoId: string, meta: ModelSelectorChangeMeta) => {
      if (!meta.ggufFilename) return null;
      const plan = await requestDownloadPlan(
        repoId,
        { kind: "gguf", filename: meta.ggufFilename },
        currentLoadAdvanced(repoId, false),
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

  const beginPick = useCallback(() => {
    pendingRecalledGeneration.current = null;
    pickSeq.current += 1;
    pendingStagedLoad.current = null;
    pendingLoadEntries.current = null;
    stagedLoadDeferred.current = false;
    stagedQuantRevert.current = null;
    pickToast.dismissAll();
  }, [pickToast]);

  // The backend rejects a gguf load with no filename, so name the file from the listing first.
  const loadGgufRepoPick = useCallback(
    async (
      repoId: string,
      quantHint: string | null,
      source: ModelSelectorChangeMeta["source"] = "hub",
      localPath?: string | null,
      effectiveFamilyOverride = familyOverride,
    ): Promise<boolean> => {
      const downloadOnly = modelSelectionAction === "download";
      const token = downloadOnly ? 0 : pickGuard.claim();
      const downloadSnapshot = downloadOnly ? currentLoadAdvanced(repoId, false) : undefined;
      const isCurrent = () => isMounted.current &&
        (downloadOnly || pickGuard.holds(token));
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
          toast.error(downloadOnly
            ? "Pick a quantization for this model to download it"
            : "Pick a quantization for this model to load it"),
        onResolved: (filename) => {
          if (downloadOnly) return;
          quantRevert.current = revert;
          setQuant(quantHint ?? filename);
          applyImageModelDefaults(repoId, effectiveFamilyOverride);
        },
        onNotStarted: () => {
          if (!downloadOnly && quantRevert.current === revert) {
            revertPick(revert);
            quantRevert.current = null;
          }
        },
        load: (filename) =>
          loadOrStage(repoId, { kind: "gguf", filename }, source, token, false, downloadSnapshot),
      });
    },
    [applyImageModelDefaults, currentLoadAdvanced, loadOrStage, modelSelectionAction, pickGuard, quant, revertPick],
  );

  // Both pages stay mounted, so a hidden page must not load after the user switched.
  useEffect(() => {
    if (!active) pickGuard.release();
  }, [active, pickGuard]);

  // Not `strict: false`: that resolves the ROOT match, and /hub uses the same param.
  const routeSearch = useSearch({ from: "/images", shouldThrow: false });
  const navigateSelf = useNavigate();
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    if (!active) return;
    if (!imagePresets.hydrated) return;
    const wanted = routeSearch?.model;
    // Release the marker once the query is gone, or re-picking the same file is a dead click.
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    // `routeSearch` is rebuilt every render, so depend on the two fields.
    const routed = { quant: routeSearch?.quant, ggufQuant: routeSearch?.ggufQuant };
    const routedFilename = routedGgufFilename(routed);
    const routedLabel = routedGgufLabel(routed);
    const key = `${wanted}|${routeSearch?.quant ?? ""}|${routeSearch?.ggufQuant ?? ""}`;
    if (handledRouteModel.current === key) return;
    handledRouteModel.current = key;
    // A Download only arrival owns nothing and must leave a staging load its claim.
    const downloadOnlyPick = modelSelectionAction === "download";
    const token = downloadOnlyPick ? undefined : pickGuard.claim();
    if (!downloadOnlyPick) setFamilyOverride("auto");
    void navigateSelf({ to: "/images", search: {}, replace: true });
    if (routedLabel) {
      void Promise.resolve().then(() =>
        loadGgufRepoPick(wanted, routedLabel, "hub", null, "auto"),
      );
      return;
    }
    const pick = diffusionRoutePick(
      wanted,
      routedFilename ?? undefined,
      loadSpecFor(wanted, IMAGE_CATALOG),
    );
    if (pick.opts.kind === "gguf" && !pick.opts.filename) {
      void Promise.resolve().then(() => loadGgufRepoPick(pick.repoId, null, "hub", null, "auto"));
      return;
    }
    const revert: PickRevert | null = downloadOnlyPick
      ? null
      : (quantRevert.current ?? { prev: quant, steps, guidance });
    if (revert) {
      quantRevert.current = revert;
      setQuant(pick.opts.kind === "pipeline" ? null : (pick.opts.filename ?? null));
      applyImageModelDefaults(wanted, "auto");
    }
    void loadOrStage(pick.repoId, pick.opts, "hub", token).then((started) => {
      if (!started && revert && token !== undefined && pickGuard.holds(token) && quantRevert.current === revert) {
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
    modelSelectionAction,
    navigateSelf,
    pickGuard,
    quant,
    revertPick,
  ]);

  // A counter, not effect cleanup, retires a lookup: clearing the query must not cancel its own.
  const routedItem = active ? routeSearch?.item : undefined;
  const routedLookup = useRef(0);
  useEffect(() => {
    if (!active) routedLookup.current += 1;
  }, [active]);
  useEffect(() => {
    if (!routedItem) return;
    const lookup = ++routedLookup.current;
    void navigateSelf({ to: "/images", search: {}, replace: true });
    void loadGalleryUntil({
      has: () => galleryCache.images.some((entry) => entry.id === routedItem),
      count: () => galleryCache.images.length,
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
        toast(translate("library.toast.imageNotFound"), {
          description: translate("library.toast.notFoundDescription"),
        });
      }
    });
  }, [routedItem, navigateSelf, loadGallery, loadMore]);

  // Reload the current model with the current advanced options.
  const handleReapply = useCallback(() => {
    const l = lastLoad.current;
    if (l) void handleLoad(l.repoId, { kind: l.kind, filename: l.filename, displayRepoId: l.displayRepoId });
  }, [handleLoad]);

  // Every pick supersedes the previous one; direct-local branches bypass loadOrStage.
  const abandonPick = useCallback(() => {
    if (quantRevert.current) {
      revertPick(quantRevert.current);
      quantRevert.current = null;
    }
  }, [revertPick]);

  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // A Download only selection does not take over the page, or a staged load never loads.
      const downloadOnlyPick = modelSelectionAction === "download";
      // The backend 409s a second load; download-only picks submit none, so allow them.
      if (busy !== null && !downloadOnlyPick) return;
      if (!downloadOnlyPick) beginPick();
      // Before any branch, since staging never sets `busy`.
      const token = downloadOnlyPick ? undefined : pickGuard.claim();
      const stillOwnsPick = (): boolean => token !== undefined && pickGuard.holds(token);
      const pipelineTarget = diffusionPipelineLoadTarget(id, meta);
      const { displayRepoId } = pipelineTarget;
      const familyOverrideRequired = meta.familyOverrideRequired === true;
      const nextFamilyOverride = familyOverrideRequired ? familyOverride : "auto";
      if (!downloadOnlyPick && !familyOverrideRequired) setFamilyOverride("auto");
      const spec = loadSpecFor(id, IMAGE_CATALOG);
      if (spec && spec.kind !== "gguf") {
      // Carried forward: a superseded staged pick left its optimistic state, which must not be snapshotted.
        // Download only relabels nothing and claims no revert: quantRevert is ONE slot a staged load may own.
        const revert: PickRevert | null = downloadOnlyPick
          ? null
          : (quantRevert.current ?? { prev: quant, steps, guidance });
        if (revert) {
          quantRevert.current = revert;
          setQuant(null);
          applyImageModelDefaults(id, nextFamilyOverride);
        }
        void loadOrStage(
          pipelineTarget.repoId,
          { kind: spec.kind, filename: spec.filename, displayRepoId },
          pipelineTarget.source,
          token,
        ).then((started) => {
            if (!started && revert && stillOwnsPick()) {
              revertPick(revert);
              quantRevert.current = null;
            }
          });
        return;
      }
      if (meta.ggufVariant && meta.ggufFilename) {
        const revert: PickRevert | null = downloadOnlyPick
          ? null
          : (quantRevert.current ?? { prev: quant, steps, guidance });
        if (revert) {
          quantRevert.current = revert;
          setQuant(meta.ggufVariant);
          applyImageModelDefaults(id, nextFamilyOverride);
        }
        void loadOrStage(
          id,
          { kind: "gguf", filename: meta.ggufFilename },
          meta.source,
          token,
        ).then((started) => {
          if (!started && revert && stillOwnsPick()) {
            revertPick(revert);
            quantRevert.current = null;
          }
        });
        return;
      }
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
        const revert: PickRevert | null = downloadOnlyPick
          ? null
          : (quantRevert.current ?? { prev: quant, steps, guidance });
        if (revert) {
          quantRevert.current = revert;
          setQuant(filename);
          applyImageModelDefaults(id, nextFamilyOverride);
        }
        void loadOrStage(dir, { kind: "gguf", filename }, meta.source, token).then((started) => {
          if (!started && revert && stillOwnsPick()) {
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
        const revert: PickRevert | null = downloadOnlyPick
          ? null
          : (quantRevert.current ?? { prev: quant, steps, guidance });
        if (revert) {
          quantRevert.current = revert;
          setQuant(filename);
          applyImageModelDefaults(id, nextFamilyOverride);
        }
        void loadOrStage(dir, { kind: "single_file", filename }, meta.source, token).then((started) => {
          if (!started && revert && stillOwnsPick()) {
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
      if (!pipelineTarget.onDevice && !id.toLowerCase().startsWith("unsloth/")) {
        // A refused Download only pick retires nothing: it never claimed the page.
        toast.error("Only unsloth or on-device image models can be loaded here");
        if (!downloadOnlyPick) abandonPick();
        return;
      }
      const revert: PickRevert | null = downloadOnlyPick
        ? null
        : (quantRevert.current ?? { prev: quant, steps, guidance });
      if (revert) {
        quantRevert.current = revert;
        setQuant(null);
        applyImageModelDefaults(id, nextFamilyOverride);
      }
      void loadOrStage(pipelineTarget.repoId, { kind: "pipeline", displayRepoId }, pipelineTarget.source, token, familyOverrideRequired).then((started) => {
        if (!started && revert && stillOwnsPick()) {
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
      currentLoadAdvanced,
      familyOverride,
      handleLoad,
      loadGgufRepoPick,
      loadOrStage,
      pickGuard,
      quant,
      revertPick,
    ],
  );

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
      pickGuard.cancel();
      pickToast.dismissAll();
      pendingDeploy.current = { loraId: stem, family: args.family };
      if (args.trigger.trim()) setPrompt(args.trigger.trim());
      setPageMode("create");
      const revert: PickRevert = quantRevert.current ?? { prev: quant, steps, guidance };
      quantRevert.current = revert;
      setQuant(null);
      applyImageModelDefaults(args.baseRepo, "auto", true);
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
    [applyImageModelDefaults, busy, handleLoad, pickGuard, pickToast, quant, revertPick, setPageMode],
  );

  const handleUnload = useCallback(async (): Promise<boolean> => {
    pendingRecalledGeneration.current = null;
    dropResidentState();
    loadTrackingRestored.current = false;
    setBusy("unloading");
    try {
      setStatusIfNewest(++statusTicket.current, await unloadDiffusionModel());
      setQuant(null);
      // Wait for any in-flight load start to finish, compensating unload included, or the older load
      // keeps running with no toast or cancel control.
      const pending = pendingStart.current;
      if (pending) {
        try {
          await pending;
        } catch {
          // Its own handler reports the failure; this only waits for the window to close.
        }
      }
      // Tracking RESTORED means the compensating unload failed and the load is still running.
      return !loadTrackingRestored.current;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to unload model");
      void refreshStatus();
      return false;
    } finally {
      // Not an unconditional clear: a restore may have set "loading" deliberately.
      setBusy((prev) => (prev === "unloading" ? null : prev));
    }
  }, [refreshStatus, dropResidentState]);

  // The unload cancels the load; it leaves only cache, so reloading resumes.
  const handleCancelLoad = useCallback(async () => {
    const wasLoading = busy === "loading";
    if (await handleUnload()) {
      toast.info("Stopped loading the model", {
        description: "Anything already downloaded stays cached, so loading it again resumes.",
      });
      return;
    }
    // Already restored in handleUnload; a second restore would duplicate the toast and poll.
    if (!wasLoading || loadTrackingRestored.current) return;
    restoreLoadTracking();
  }, [busy, handleUnload, restoreLoadTracking]);

  useEffect(() => {
    cancelLoadRef.current = () => void handleCancelLoad();
  }, [handleCancelLoad]);

  const buildKey = status?.loaded
    ? [
        status.repo_id,
        status.base_repo,
        status.model_kind,
        status.transformer_quant,
        status.resolved?.transformer_quant?.source ?? "",
        (status.conditioning?.reference_resolutions ?? []).join(","),
      ].join("|")
    : null;
  const [seededBuild, setSeededBuild] = useState<string | null>(null);
  if (buildKey !== seededBuild) {
    setSeededBuild(buildKey);
    if (status?.loaded) {
      const tier = resolutionFor(status.base_repo ?? status.repo_id ?? "", {
        modelKind: status.model_kind,
        transformerQuant: status.transformer_quant,
        transformerQuantSource: status.resolved?.transformer_quant?.source,
      }).width;
      setReferenceResolution(
        seedReferenceResolution(status.conditioning?.reference_resolutions ?? [], tier),
      );
      setMatchResolution(tier);
    }
  }

  const limitsKey = `${sizeLimits.multiple}|${sizeLimits.maxSide}|${sizeLimits.maxPixels}`;
  const [fittedLimits, setFittedLimits] = useState(limitsKey);
  if (limitsKey !== fittedLimits) {
    setFittedLimits(limitsKey);
    const fitted = fitSize(width, height, sizeLimits);
    if (fitted.width !== width) setWidth(fitted.width);
    if (fitted.height !== height) setHeight(fitted.height);
  }

  const [sourceRead, setSourceRead] = useState<{
    src: string;
    width: number;
    height: number;
  } | null>(null);
  const sourceDims = sourceRead && sourceRead.src === initImage ? sourceRead : null;
  useEffect(() => {
    if (!initImage) return;
    let live = true;
    loadImage(initImage)
      .then((img) => {
        if (live) setSourceRead({ src: initImage, width: img.naturalWidth, height: img.naturalHeight });
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [initImage]);

  const editSize = useMemo(
    () =>
      resolveEditSize(editSizing, sourceDims, matchResolution, { width, height }, sizeLimits),
    [editSizing, sourceDims, matchResolution, width, height, sizeLimits],
  );
  const officialPresets = useMemo(() => presetsWithin(sizeLimits), [sizeLimits]);
  const showOfficialPresets = sizeLimits.maxSide > MAX_OUTPUT_DEFAULT && officialPresets.length > 0;
  const unifiedEditActive = workflow === "edit" && unifiedEdit;
  const onLocalizedLayer = useCallback((dataUrl: string | null) => setLocalizedLayer(dataUrl), []);
  const onLocalizedColors = useCallback((names: string[]) => setLocalizedColors(names), []);

  const renderAdditionalImages = (numberOf: (i: number) => number, hint: string) => (
    <>
      {referenceImages.map((img, i) => (
        <Field key={i} label={`Image ${numberOf(i)}`} hint={hint}>
          <div className="space-y-1.5">
            <ImageDropzone
              value={img}
              onChange={(v) =>
                setReferenceImages((prev) => prev.map((p, j) => (j === i ? (v ?? "") : p)))
              }
              removeLabel={`Remove image ${numberOf(i)}`}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="w-full"
              onClick={() => setReferenceImages((prev) => prev.filter((_, j) => j !== i))}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
              Remove image {numberOf(i)}
            </Button>
          </div>
        </Field>
      ))}
      {referenceImages.length > maxExtras && (
        <p className="text-xs text-destructive">
          This model takes {maxExtras} additional image{maxExtras === 1 ? "" : "s"} here. Remove{" "}
          {referenceImages.length - maxExtras} to generate.
        </p>
      )}
      {referenceImages.length < maxExtras && (
        <Button
          type="button"
          variant="secondary"
          size="sm"
          className="w-full"
          disabled={!initImage}
          onClick={() => setReferenceImages((prev) => [...prev, ""])}
        >
          <HugeiconsIcon icon={ImageAdd02Icon} className="size-3.5" />
          Add image {numberOf(referenceImages.length)}
        </Button>
      )}
    </>
  );

  const engineNotes = conditioning?.notes?.length ? (
    <ul className="space-y-1 text-ui-11 leading-snug text-muted-foreground">
      {conditioning.notes.map((note) => (
        <li key={note}>{note}</li>
      ))}
    </ul>
  ) : null;

  const referenceDetailControl =
    referenceResolutions.length > 0 && referenceResolution != null ? (
      <Field
        label="Reference detail"
        hint="The resolution every input image is resized to (by area) before the model reads it. Separate from the output size: higher keeps more detail from the inputs and costs more memory and time for every image. High (2048) with many images needs a large GPU."
      >
        <Select
          value={String(referenceResolution)}
          onValueChange={(v) => setReferenceResolution(Number(v))}
        >
          <SelectTrigger aria-label="Reference detail">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {referenceResolutions.map((r) => (
              <SelectItem key={r} value={String(r)}>
                {REFERENCE_DETAIL_LABELS[r] ?? String(r)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
    ) : null;

  const sizeControls = (
    <>
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
          workflow === "transform"
            ? "Caps the output size. The source image is scaled down to fit inside this box, keeping its aspect ratio, so the result may be smaller than the values shown."
            : workflow === "inpaint" ||
                workflow === "extend" ||
                workflow === "upscale" ||
                (workflow === "edit" && !unifiedEdit)
              ? "Not used by this workflow: the output size comes from the source image. Upload a smaller image to generate at a smaller size."
              : `Width and height in pixels. Sizes run from ${MIN_DIM} to ${sizeLimits.maxSide} in steps of ${sizeLimits.multiple}${sizeLimits.maxPixels < sizeLimits.maxSide * sizeLimits.maxSide ? `, up to ${(sizeLimits.maxPixels / 1e6).toFixed(1)} megapixels` : ""}. Most models are trained around 1 megapixel, so much larger sizes can look worse.`
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
            limits={sizeLimits}
          />
          <DimensionSelect
            icon={ArrowUpDownIcon}
            label="Height"
            value={height}
            open={active && heightOpen}
            onOpenChange={(o) => setHeightOpen(active && o)}
            onChange={changeHeight}
            limits={sizeLimits}
          />
        </div>
      </Field>
      {showOfficialPresets && (
        <Field
          label="2K presets"
          hint="The model's native 2K sizes. They take several times the memory and time of a 1 megapixel image."
        >
          <Select
            value=""
            onValueChange={(v) => {
              const preset = officialPresets.find((p) => `${p.width}x${p.height}` === v);
              if (!preset) return;
              setWidth(preset.width);
              setHeight(preset.height);
              const m = matchAspect(preset.width, preset.height);
              setAspect(m.key);
              setPortrait(m.portrait);
            }}
          >
            <SelectTrigger aria-label="2K presets">
              <SelectValue placeholder="Choose a 2K size" />
            </SelectTrigger>
            <SelectContent>
              {officialPresets.map((p) => (
                <SelectItem key={`${p.width}x${p.height}`} value={`${p.width}x${p.height}`}>
                  {`${p.label} (${p.width} × ${p.height})`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      )}
    </>
  );

  const handleGenerate = useCallback(async () => {
    const allowOversizedSent = allowOversizedField(allowOversized, oversizedOnce.current) === true;
    oversizedOnce.current = false;
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
    const unifiedEditRun = isEdit && unifiedEdit;
    if (unifiedEditRun && localizedMode && !localizedLayer) {
      toast.error(
        localizedMode === "annotate"
          ? "Draw the annotations on the source image, or turn Localize off"
          : "Paint the region on the source image, or turn Localize off",
      );
      return;
    }
    const emptySlot = (isReference || unifiedEditRun) ? referenceImages.findIndex((img) => !img) : -1;
    if (emptySlot >= 0) {
      const n = isReference ? emptySlot + 2 : additionalImageNumber(emptySlot, localizedMode);
      toast.error(`Image ${n} is empty. Add it, or remove that slot.`);
      return;
    }
    if ((isReference || unifiedEditRun) && referenceImages.filter(Boolean).length > maxExtras) {
      toast.error(
        `This model takes at most ${maxExtras + 1 + (unifiedEditRun && localizedMode === "mask" ? 1 : 0)} input images in total. Remove ${referenceImages.filter(Boolean).length - maxExtras}.`,
      );
      return;
    }
    if (isExtend && !(extendSides.left || extendSides.right || extendSides.top || extendSides.bottom)) {
      toast.error("Pick at least one side to extend");
      return;
    }

    let condInit: string | undefined;
    let condMask: string | undefined;
    let condStrength: number | undefined;
    let condUpscale: number | undefined;
    let condFields: ReturnType<typeof conditionedRequestFields> | undefined;
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
        condStrength = 1;
        condFields = { workflow: "outpaint" };
      } else if (isUpscale) {
        condInit = initImage ?? undefined;
        condUpscale = upscaleFactor;
        condStrength = upscaleStrength;
      } else if (isReference || unifiedEditRun) {
        condFields = conditionedRequestFields({
          workflow: isReference ? "reference" : "edit",
          initImage: initImage!,
          extras: referenceImages,
          referenceResolution,
          conditioning,
          localized:
            unifiedEditRun && localizedMode && localizedLayer
              ? { mode: localizedMode, image: localizedLayer }
              : null,
        });
      } else if (isEdit) {
        condFields = { workflow: "edit", init_image: initImage ?? undefined };
      }
    } catch {
      toast.error("Could not prepare the source image");
      return;
    }
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

    const sent = unifiedEditRun ? editSize : fitSize(width, height, sizeLimits);
    const w = sent.width;
    const h = sent.height;
    if (!unifiedEditRun || editSizing === "custom") {
      if (w !== width) setWidth(w);
      if (h !== height) setHeight(h);
    }

    const runs = Number.isFinite(count) && count >= 1 ? Math.floor(count) : 1;
    if (runs !== count) setCount(runs);

    // Offsets added to a seed near the 2**53-1 cap could 422 a later run after GPU work.
    if (baseSeed > Number.MAX_SAFE_INTEGER - (runs * batchSize - 1)) {
      toast.error("Seed too large for this run count and batch size; use a smaller seed");
      return;
    }

    saveLastPrompt(`images:${workflow}`, prompt);
    setBusy("generating");
    setGenDone(0);
    setGenStep(null);
    cancelRequested.current = false;
    setStopping(false);
    cancelAcked.current = false;
    cancelInFlight.current = null;
    cancelAbort.current?.abort();
    cancelAbort.current = null;
    runToken.current += 1;
    let pollInFlight = false;
    const pollGenerateOnce = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const p = await getGenerateProgress();
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
    if (genVisibilityListener.current)
      document.removeEventListener("visibilitychange", genVisibilityListener.current);
    genVisibilityListener.current = () => {
      if (document.visibilityState === "visible") void pollGenerateOnce();
    };
    document.addEventListener("visibilitychange", genVisibilityListener.current);
    genPollTimer.current = setInterval(() => void pollGenerateOnce(), 300);
    // Captured BEFORE the first POST: settleLostGeneration needs a record outside it as proof.
    const knownIds = new Set(galleryCache.images.map((image) => image.id));
    try {
      for (let i = 0; i < runs; i++) {
        if (
          !shouldContinueGenerating({
            mounted: isMounted.current,
            stopRequested: cancelRequested.current,
          })
        )
          break;
        // Frozen BEFORE the POST so both halves of the probe describe the same moment.
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
            negative_prompt:
              negativeCapable && guidance > 0 ? negativePrompt.trim() || undefined : undefined,
            width: w,
            height: h,
            steps,
            guidance,
            // The native engine seeds image j at seed+j, so offset by batch size.
            seed: baseSeed + i * batchSize,
            batch_size: batchSize,
            init_image: condInit,
            mask_image: condMask,
            strength: condStrength,
            upscale: condUpscale,
            allow_oversized: allowOversizedSent ? true : undefined,
            live_preview: livePreview,
            ...condFields,
            // Gated on loraCapable, since a restore can leave adapters in state.
            loras: (() => {
              if (!loraCapable) return undefined;
              const active = loras
                .map((l) => ({ id: l.id.trim(), weight: l.weight }))
                .filter((l) => l.id && l.weight > 0);
              return active.length ? active : undefined;
            })(),
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
          // The tunnel caps the response near 100s; retrying would duplicate the work.
          if (!(err instanceof GenerateResponseLostError)) throw err;
          await settleLostGeneration(() => isMounted.current, probeBaseline);
          if (!isMounted.current) break;
          await loadGallery();
          galleryCache.images.forEach((image) => knownIds.add(image.id));
          setGenDone(i + 1);
          continue;
        }
        if (!isMounted.current) break;
        // Sorted, not prepended: new images are unpinned and go after the pinned group.
        stripEpoch.current += 1;
        // Deduplicated: a resync may already have fetched the record.
        setImages((prev) => mergeGenerated(prev, res.images));
        res.images.forEach((image) => knownIds.add(image.id));
        if (res.images[0]) setSelectedId(res.images[0].id);
        res.images.forEach((image) => void ensureSrc(image));
        setGenDone(i + 1);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Image generation failed";
      // Only a Stop the backend confirmed explains an error away.
      const report = shouldReportGenerateError({
        message: msg,
        stopRequested: cancelRequested.current && cancelAcked.current,
      });
      if (report && shouldOfferGenerateAnyway({ error: err, allowOversizedSent })) {
        toast.error(MEMORY_REFUSAL_TITLE, {
          description: msg,
          duration: 20_000,
          action: {
            label: GENERATE_ANYWAY_LABEL,
            onClick: () => setOversizedRetryQueued(true),
          },
        });
      } else if (report)
        toast.error(msg, { action: generationFailureLogsAction(msg) });
    } finally {
      if (genPollTimer.current) clearInterval(genPollTimer.current);
      genPollTimer.current = null;
      if (genVisibilityListener.current) {
        document.removeEventListener("visibilitychange", genVisibilityListener.current);
        genVisibilityListener.current = null;
      }
      cancelRequested.current = false;
      // Await on EVERY exit: a run can change status (compile, cancelled native run), so Generate could 409.
      if (isMounted.current) await refreshStatus();
      setBusy(null);
      setGenDone(null);
      setGenStep(null);
      setStopping(false);
    }
  }, [allowOversized, livePreview, prompt, negativePrompt, negativeCapable, width, height, steps, guidance, seed, batchSize, count, workflow, initImage, maskImage, strength, extendPct, extendSides, upscaleFactor, upscaleStrength, referenceImages, loras, loraCapable, controlnetCapable, controlnetId, controlImage, controlType, controlStrength, ensureSrc, loadGallery, refreshStatus, unifiedEdit, localizedMode, localizedLayer, maxExtras, referenceResolution, conditioning, editSize, editSizing, sizeLimits]);

  // Latch FIRST, so a multi-run request stops even if the POST races the finishing run.
  const handleCancelGenerate = useCallback(async () => {
    cancelRequested.current = true;
    setStopping(true);
    // One Stop on the wire at a time: a second POST would target whatever is active when it arrives.
    const token = runToken.current;
    if (cancelInFlight.current === token) return;
    cancelInFlight.current = token;
    const abort = new AbortController();
    cancelAbort.current = abort;
    try {
      const { cancelled } = await cancelDiffusionGeneration(abort.signal);
      cancelAcked.current = Boolean(cancelled);
      if (!cancelled) setStopping(false);
    } catch {
      if (!abort.signal.aborted) {
        cancelAcked.current = false;
        setStopping(false);
        toast.error("Could not reach the server to stop this generation; it is still running");
      }
    } finally {
      if (cancelInFlight.current === token) cancelInFlight.current = null;
      if (cancelAbort.current === abort) cancelAbort.current = null;
    }
  }, []);

  const handleGenerateWithRecall = useCallback(async () => {
    if (busy !== null || !imagePresets.hydrated) return;
    if (status?.loaded) {
      const kind = status.model_kind;
      if (
        status.repo_id &&
        (kind === "pipeline" ||
          ((kind === "gguf" || kind === "single_file") && status.gguf_filename))
      ) {
        const model = withEngagedFamily(
          lastLoad.current &&
            matchesRememberedModel(lastLoad.current, status) &&
            componentFilesMatch(lastLoad.current, status.component_files)
            ? lastLoad.current
            : rememberedModel &&
                matchesRememberedModel(rememberedModel, status) &&
                componentFilesMatch(rememberedModel, status.component_files)
              ? rememberedModel
              : { repoId: status.repo_id, kind, filename: status.gguf_filename ?? undefined },
          status,
        );
        rememberImageModel(model);
        setRememberedModel(model);
      }
      await handleGenerate();
      return;
    }
    if (!rememberedModel || !prompt.trim()) {
      toast.info(
        rememberedModel
          ? "Enter a prompt first."
          : "Pick an image model first.",
      );
      return;
    }
    pendingRecalledGeneration.current = {
      model: rememberedModel,
      load: loadSeq.current + 1,
      workflow,
      allowOversized: oversizedOnce.current,
    };
    const started = await handleLoad(
      rememberedModel.repoId,
      { kind: rememberedModel.kind, filename: rememberedModel.filename },
      // Only the family the remembered load engaged; the live selection belongs to whatever is picked next.
      {
        ...currentLoadAdvanced(rememberedModel.repoId, false, true),
        family_override: rememberedModel.familyOverride,
        // The recalled build's own encoder / VAE files, not whatever the fields hold now.
        text_encoder_file: rememberedModel.textEncoderFiles,
        vae_file: rememberedModel.vaeFile,
      },
    );
    if (!started) pendingRecalledGeneration.current = null;
  }, [
    busy,
    currentLoadAdvanced,
    handleGenerate,
    handleLoad,
    imagePresets.hydrated,
    prompt,
    rememberedModel,
    status,
    workflow,
  ]);
  useEffect(() => {
    if (!shouldRunQueuedOversizedRetry({ queued: oversizedRetryQueued, busy })) return;
    setOversizedRetryQueued(false);
    oversizedOnce.current = true;
    void handleGenerateWithRecall().finally(() => {
      oversizedOnce.current = false;
    });
  }, [oversizedRetryQueued, busy, handleGenerateWithRecall]);

  useEffect(() => {
    const pending = pendingRecalledGeneration.current;
    if (!pending) return;
    if (!active || pending.load !== loadSeq.current) {
      pendingRecalledGeneration.current = null;
      return;
    }
    if (busy !== null || !status?.loaded) return;
    pendingRecalledGeneration.current = null;
    const tab = WORKFLOW_TABS.find(
      (candidate) => candidate.id === pending.workflow,
    );
    if (
      pending.workflow !== workflow ||
      !tab ||
      !(status.workflows ?? []).includes(tab.requires ?? "txt2img")
    ) {
      toast.info(
        "Choose a workflow supported by the loaded model before generating.",
      );
      return;
    }
    if (!matchesRememberedModel(pending.model, status)) return;
    oversizedOnce.current = pending.allowOversized === true;
    void handleGenerate().finally(() => {
      oversizedOnce.current = false;
    });
  }, [active, busy, handleGenerate, status, workflow]);

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

  useEffect(() => {
    if (supported === null || pageMode !== "create") return;
    if (!supported.includes(workflow) && supported[0]) {
      setWorkflow(supported[0]);
    }
  }, [supported, workflow, setWorkflow, pageMode]);

  const activeWorkflowTab =
    WORKFLOW_TABS.find((t) => t.id === workflow) ?? WORKFLOW_TABS[0];

  const advancedControls = (
    <>
      <AdvancedSelect {...familySelect} badge={<ResolvedBadge status={status} controlKey="family_override" />} />
      <AdvancedSelect
        label="On model selection"
        hint="Choose Download only to prepare the selected model and its required assets without loading it. Applies to the next model you select; progress and cancellation appear in Downloads."
        value={modelSelectionAction}
        onValueChange={(v) => setModelSelectionAction(v as typeof modelSelectionAction)}
        options={[
          ["load", "Download and load"],
          ["download", "Download only"],
        ]}
      />
      <AdvancedSelect
        label="Speed"
        hint="Auto picks per model: GGUF compiles at load; a dense model keeps the first two images exact and eager, then compiles from the 3rd (~2x from there). eager = fused kernels, no compile. default/max add torch.compile (max also TF32 + fused QKV, plus the step cache on 20+ step models)."
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
      {!status?.loaded
      || (sendsTransformerQuant(status.model_kind, status.repo_id ?? "")
          && !isNativeEngineStatus(status)) ? (
        <AdvancedSelect
          label="Precision"
          hint="How the model computes. Auto picks the fastest precision the hardware supports (INT8 on every capable GPU, then FP8 where the card has it) and quantises the transformer onto low-precision tensor cores. A GGUF pick reaches it by loading the FULL base model instead of the GGUF, and falls back to the GGUF as-is when the device, VRAM or disk can't take it; an official pipeline is already dense and is quantised in place, falling back to plain BF16. Off runs the checkpoint as-is."
          badge={<ResolvedBadge status={status} controlKey="transformer_quant" />}
          value={transformerQuant}
          onValueChange={(v) => setTransformerQuant(v as typeof transformerQuant)}
          options={[
            ["auto", "Auto (fastest for GPU)"],
            ["none", "Off (run the checkpoint as-is)"],
            // Mac and CPU-only hosts cannot run the dense tensor-core path these schemes need.
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
          <span className="text-xs text-muted-foreground/60">
            Runs this checkpoint's own precision
          </span>
        </div>
      )}
      <AdvancedSelect
        label="Text encoder precision"
        hint={`Shrinks the text encoder to save memory, at some cost to image quality. Default lets the model choose, which on Qwen-Image 2.1 means its hosted FP8 encoder (8.75 GB rather than 16.3); pick Dense (bf16) to pin the released encoder. FP8 (storage) is the safe pick and the only one that works with CPU offload. FP8 (compute) needs an RTX 40 series or newer.${nvfp4Diffusion ? " NVFP4 is the smallest." : ""} The loaded build below reports what was applied.`}
        badge={<ResolvedBadge status={status} controlKey="text_encoder_quant" />}
        value={textEncoderQuant}
        onValueChange={(v) => setTextEncoderQuant(v as typeof textEncoderQuant)}
        options={withNvfp4Option(
          [
            ["auto", "Default"],
            // Omitting the field no longer means dense, since a family default can pick a scheme.
            ["none", "Dense (bf16)"],
            ["fp8", "FP8 (storage)"],
            ["fp8_dynamic", "FP8 (compute)"],
            ["int8", "INT8"],
            ["nvfp4", "NVFP4"],
          ] as [string, string][],
          nvfp4Diffusion,
        )}
      />
      <AdvancedTextField
        label="Text encoder file(s)"
        hint={COMPONENT_FILES_HINT}
        multiline
        placeholder="../text_encoders/clip_l.safetensors"
        value={textEncoderFiles}
        onValueChange={setTextEncoderFiles}
      />
      <AdvancedTextField
        label="VAE file"
        hint={COMPONENT_FILES_HINT}
        placeholder="../vae/ae.safetensors"
        value={vaeFile}
        onValueChange={setVaeFile}
      />
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
        hint="Static skip extrapolates middle steps on a fixed schedule (12+ steps) and keeps the compile and CUDA graph. Auto uses it for text-to-image, at or above the step count it was measured at, on the models where it stayed close to the full render: Qwen-Image, Qwen-Image-2.1, FLUX.1 Krea dev, FLUX.2 klein base 4B on every speed tier but Off/Eager, and FLUX.1 dev and HunyuanImage 2.1 on Max only. First-Block-Cache reuses the transformer tail across steps (~1.4x, larger quality cost); Auto turns it on for other many-step models on Max only. UNSLOTH_DIFFUSION_AUTO_STEP_SKIP=0 stops Auto from picking Static skip."
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
          CPU offload
          <InfoHint>Offload to CPU to fit low-VRAM cards (slower). Overridden by Memory mode when that is not Auto.</InfoHint>
          <ResolvedBadge status={status} controlKey="cpu_offload" />
        </span>
        <Switch checked={cpuOffload} onCheckedChange={setCpuOffload} />
      </div>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
          {ALLOW_OVERSIZED_LABEL}
          <InfoHint>{ALLOW_OVERSIZED_HINT}</InfoHint>
        </span>
        <Switch
          checked={allowOversized}
          onCheckedChange={setAllowOversized}
          aria-label={ALLOW_OVERSIZED_LABEL}
        />
      </div>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
          Live preview
          <InfoHint>Show a rough preview of the image while it denoises. Costs no measurable speed and never changes the final image.</InfoHint>
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
    <div
      {...{ [MEDIA_RAIL_ROOT_ATTR]: "" }}
      style={railRootStyle}
      className="diffusion-surface @container relative flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]"
    >
      <MediaRailResizeHandle kind="images" placement="page" className="hidden @[50rem]:block" />
      {/* Portals to body, and this page stays mounted off-route, so gate it like the composer. */}
      {active && <GuidedTour {...tour.tourProps} />}
      <div className="pointer-events-none relative z-40 grid h-[calc(48px*var(--ui-space-scale,1))] shrink-0 grid-cols-[minmax(0,var(--media-rail-width,calc(408px*var(--ui-space-scale,1))))_minmax(13rem,1fr)] @max-[30rem]:grid-cols-[minmax(0,1fr)_auto]">
        <div
          className={cn(
            "pointer-events-none flex h-full min-w-0 items-start overflow-hidden pr-3 @[50rem]:border-r @[50rem]:border-border/60",
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
                triggerDataTour="images-model"
                models={imageModels}
                value={selectorModelId}
                loadedModelIdOverride={selectorModelId}
                activeGgufVariant={quant}
                onValueChange={handleModelSelect}
                resolveDownloadFootprint={resolveDownloadFootprint}
                onEject={status?.loaded ? handleUnload : undefined}
                variant="ghost"
                className="!h-[calc(34px*var(--ui-space-scale,1))] max-w-full gap-1 overflow-hidden pl-3 pr-1 @[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"
                triggerLabelClassName="text-ui-14 @[68rem]:text-ui-16"
                task={IMAGE_GEN_TASKS}
                catalog={IMAGE_CATALOG}
                opaqueKind={opaqueKind}
                hubCapability="diffusion"
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
                    className="!h-[calc(34px*var(--ui-space-scale,1))] shrink-0 rounded-full text-xs"
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
              dataTour="images-mode"
              ariaLabel="Page mode"
              value={pageMode}
              onValueChange={(v) => setPageMode(v as "create" | "train")}
              fit={true}
              className="h-[calc(34px*var(--ui-space-scale,1))] [&>button]:h-[calc(34px*var(--ui-space-scale,1))] [&>button]:px-3 @[68rem]:[&>button]:px-11 @max-[30rem]:[&>button]:px-2.5 @max-[30rem]:[&>button>span]:sr-only"
              tabs={[
                { value: "create", label: "Create", icon: <HugeiconsIcon icon={SparklesIcon} className="size-3.5" /> },
                { value: "train", label: "Train", icon: <HugeiconsIcon icon={TestTubeOutlineIcon} className="size-3.5" /> },
              ]}
            />
          </div>
          <div className="pointer-events-none col-start-3 flex min-w-0 items-start justify-end pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
            <div className="pointer-events-auto flex min-w-0 items-center gap-2">
              <LibraryPageLink
                tab="images"
                labelClassName="hidden @[50rem]:inline"
                arrowClassName="hidden @[50rem]:block"
              />
            </div>
          </div>
        </div>
      </div>
      {pageMode === "train" ? (
        <DiffusionTrainPanel
          active={active && pageMode === "train"}
          loadedFamily={status?.family ?? null}
          loadedBaseRepo={
            // For a GGUF load repo_id is a checkpoint path, not a trainable base.
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
        <div
          data-tour="images-settings"
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
                  <HugeiconsIcon
                    icon={activeWorkflowTab.icon}
                    className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0"
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
                            // No border: the fill marks the state, and index.css blanks mouse-focus rings anyway.
                            "rounded-lg px-2 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                            on
                              ? "bg-primary/15 text-foreground hover:bg-primary/20 dark:bg-primary/25 dark:hover:bg-primary/30"
                              : "bg-muted text-muted-foreground hover:bg-muted/70 hover:text-foreground dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
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
                  label={unifiedEdit ? "Image 1" : "Reference image"}
                  hint="A reference the model draws on (subject, style, or composition) while generating a NEW image from your prompt at the size below. Unlike Transform, it is not a redraw of this image, so there is no strength."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} />
                </Field>
                {renderAdditionalImages(
                  (i) => i + 2,
                  "An extra reference combined with the others (e.g. one for the subject, one for the style). Refer to it in the prompt by its number.",
                )}
                {referenceDetailControl}
                {engineNotes}
              </>
            )}

            {workflow === "edit" && !unifiedEdit && (
              <Field
                label="Source image"
                hint="The image to edit. Describe the change in the prompt below (e.g. 'make it night', 'add a red hat', 'change the background to a beach')."
              >
                <ImageDropzone value={initImage} onChange={handleInitChange} />
              </Field>
            )}

            {unifiedEditActive && (
              <>
                {initImage && localizedMode ? (
                  <Field
                    label="Image 1 (source)"
                    hint={
                      localizedMode === "annotate"
                        ? "Draw outlines or marks around what to change, in the colours your instruction names."
                        : localizedMode === "paint"
                          ? "Paint the area to change in white. The model sees the white paint on the source."
                          : "Paint the area to change. It is sent as a white-on-black mask, Image 2, right after the source."
                    }
                  >
                    <div className="space-y-1.5">
                      <LocalizedEditCanvas
                        image={initImage}
                        mode={localizedMode}
                        color={localizedColor}
                        brushPct={brushPct}
                        resetKey={localizedResetKey}
                        onLayerChange={onLocalizedLayer}
                        onColorsChange={onLocalizedColors}
                      />
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="w-full"
                        onClick={() => handleInitChange(null)}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                        Remove source image
                      </Button>
                    </div>
                  </Field>
                ) : (
                  <Field
                    label="Image 1 (source)"
                    hint="The image to edit. Describe the change in the instruction below; add more images to combine them, and refer to them as Image 2, Image 3 and so on."
                  >
                    <ImageDropzone value={initImage} onChange={handleInitChange} />
                  </Field>
                )}
                {conditioning?.localized_edit_modes?.length ? (
                  <Field
                    label="Localize"
                    hint="Point the edit at a region. It guides a generative edit: the model redraws the whole image and is asked to change the marked area, so pixels outside it are not guaranteed to stay identical."
                  >
                    <Select
                      value={localizedMode ?? "off"}
                      onValueChange={(v) => {
                        setLocalizedMode(v === "off" ? null : (v as LocalizedEditMode));
                        setLocalizedLayer(null);
                        setLocalizedColors([]);
                      }}
                    >
                      <SelectTrigger aria-label="Localize">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="off">Off (whole image)</SelectItem>
                        {conditioning.localized_edit_modes.includes("annotate") && (
                          <SelectItem value="annotate">Colour annotations</SelectItem>
                        )}
                        {conditioning.localized_edit_modes.includes("paint") && (
                          <SelectItem value="paint">White paint on the source</SelectItem>
                        )}
                        {conditioning.localized_edit_modes.includes("mask") && (
                          <SelectItem value="mask">Separate mask (Image 2)</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </Field>
                ) : null}
                {localizedMode && (
                  <>
                    {localizedMode === "annotate" && (
                      <Field label="Annotation colour">
                        <div className="flex gap-2">
                          {ANNOTATION_COLORS.map((c) => (
                            <button
                              key={c.value}
                              type="button"
                              aria-label={c.name}
                              aria-pressed={localizedColor === c.value}
                              onClick={() => setLocalizedColor(c.value)}
                              className={cn(
                                "size-7 rounded-full border-2",
                                localizedColor === c.value ? "border-foreground" : "border-transparent",
                              )}
                              style={{ backgroundColor: c.value }}
                            />
                          ))}
                        </div>
                      </Field>
                    )}
                    <SliderField
                      label="Brush size"
                      value={brushPct}
                      min={1}
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
                        disabled={!localizedLayer}
                        onClick={() => setLocalizedResetKey((k) => k + 1)}
                      >
                        Clear
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="flex-1"
                        onClick={() =>
                          setPrompt((p) => withLocalizedHint(p, localizedMode, localizedColors))
                        }
                      >
                        Add region wording
                      </Button>
                    </div>
                  </>
                )}
                {renderAdditionalImages(
                  (i) => additionalImageNumber(i, localizedMode),
                  "Another input the instruction can refer to by its number, such as a person, product or style to bring into the source.",
                )}
                <Field
                  label="Output size"
                  hint="Match Image 1 keeps the source's proportions at the chosen size. Custom uses the aspect ratio and resolution below. The size shown is the size generated."
                >
                  <div className="flex items-center gap-2">
                    <Select
                      value={editSizing}
                      onValueChange={(v) => setEditSizing(v as EditSizing)}
                    >
                      <SelectTrigger aria-label="Output size" className="flex-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="source">Match Image 1</SelectItem>
                        <SelectItem value="custom">Custom</SelectItem>
                      </SelectContent>
                    </Select>
                    {editSizing === "source" && (
                      <Select
                        value={String(matchResolution)}
                        onValueChange={(v) => setMatchResolution(Number(v))}
                      >
                        <SelectTrigger aria-label="Match size" className="w-[calc(120px*var(--ui-space-scale,1))]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[512, 768, 1024, 1536, 2048]
                            .filter((r) => r <= sizeLimits.maxSide)
                            .map((r) => (
                              <SelectItem key={r} value={String(r)}>
                                {r === 2048 ? "2K" : `${r} px`}
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                  <p className="text-ui-11 tabular-nums text-muted-foreground">
                    {editSizing === "source" && !sourceDims
                      ? "Add Image 1 to size the output from it."
                      : `${editSize.width} × ${editSize.height}`}
                  </p>
                </Field>
                {editSizing === "custom" && sizeControls}
                {referenceDetailControl}
                {engineNotes}
                {unifiedEdit && (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="w-full"
                    onClick={() => setPrompt((p) => withTransparencyPrompt(p))}
                  >
                    Ask for a transparent background
                  </Button>
                )}
              </>
            )}

            <Field label={workflow === "edit" ? "Instruction" : "Prompt"}>
              <Textarea
                data-type-to-activate="prompt"
                rows={4}
                className={cn(IMAGE_PROMPT_BOX, "min-h-32")}
                placeholder={
                  examplesDismissed[workflow] ? undefined : WORKFLOW_EXAMPLE_PROMPTS[workflow]
                }
                value={prompt}
                onFocus={() => {
                  if (examplesDismissed[workflow]) return;
                  dismissExample(`images:${workflow}`);
                  setExamplesDismissed((prev) => ({ ...prev, [workflow]: true }));
                }}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </Field>
            {negativeCapable && (
              <NegativePromptField
                value={negativePrompt}
                onChange={setNegativePrompt}
                open={negativeOpen}
                onOpenChange={setNegativeOpen}
                hint="What to steer the image away from. Only used when guidance is above 0."
                textareaClassName={IMAGE_PROMPT_BOX}
              />
            )}
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
                      // Key on the index: sel.id is the editable input value, so keying on it drops focus.
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
            {!unifiedEditActive && sizeControls}

            <div className="pt-2">
              <SliderField
                label="Steps"
                hint="Number of denoising steps. Start with the selected model's default; more steps take longer and may not improve quality."
                value={steps}
                min={1}
                max={50}
                step={1}
                onChange={setSteps}
              />
            </div>
            <SliderField
              label="Guidance"
              hint="Controls how strongly the model follows the prompt. Start with the selected model's default; distilled models may require low or zero guidance."
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
          {/* Leave the footer unpainted to avoid dark-mode banding. */}
          <div className="relative z-10 flex shrink-0 flex-wrap justify-center gap-2 px-4 pt-0.5 pb-4">
            {busy === "generating" ? (
              <Button
                className="relative z-10 h-11 px-8 hover:bg-muted dark:hover:bg-muted"
                variant="outline"
                onClick={handleCancelGenerate}
              >
                <Spinner className="mr-2 size-4" />
                {stopButtonLabel({ stopping, done: genDone, count })}
              </Button>
            ) : (
              <>
                <Button
                  className="relative z-10 h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                  onClick={handleGenerateWithRecall}
                  disabled={busy !== null || !imagePresets.hydrated || (!status?.loaded && !rememberedModel)}
                >
                  Generate
                </Button>
                {status?.loaded && (canReapply || status?.model_kind === "pipeline") && (
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
          data-tour="images-preview"
          className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden @[50rem]:min-h-0"
        >
          {viewerImage && viewerSrc && (
            <MediaViewer
              open={true}
              onOpenChange={(open) => !open && setViewerId(null)}
              title={viewerImage.prompt || t("library.viewer.untitledImage")}
              meta={`Generated · ${viewerImage.width} × ${viewerImage.height}`}
              media={true}
              noun="image"
              actions={{
                primary: {
                  label: t("library.menu.chatAboutThis"),
                  icon: MessageCircleIcon,
                  onClick: () =>
                    void chatAboutMedia(
                      navigateToChat,
                      // WebKit shows the object URL but cannot refetch it.
                      () => fetchGalleryResponse(viewerImage.url),
                      viewerImage.prompt,
                      "image",
                    ),
                },
                onDownload: () => void handleQuickDownload(viewerImage),
                reveal: revealLabel
                  ? { label: revealLabel, onClick: () => revealInFolder(`image:${viewerImage.id}`) }
                  : undefined,
                favorite: isFavorite(`image:${viewerImage.id}`),
                onToggleFavorite: () => toggleFavorite(`image:${viewerImage.id}`),
                onAddToProject: (projectId) => addGalleryImageToProject(viewerImage.id, projectId),
                onDelete: () => {
                  setViewerId(null);
                  void handleDelete(viewerImage.id);
                },
              }}
            >
              <img src={viewerSrc} alt={viewerImage.prompt} className="size-full object-contain" />
            </MediaViewer>
          )}
          <div className="hover-scrollbar relative flex flex-1 items-center justify-center overflow-auto p-6 px-10 @[50rem]:pt-[calc(60px*var(--ui-space-scale,1))]">
            {livePreviewSrc ? (
              <img
                src={livePreviewSrc}
                alt="Live preview of the image being generated"
                data-testid="images-live-preview"
                style={{ maxWidth: width, maxHeight: height }}
                className="size-full object-contain shadow-sm"
              />
            ) : selected && selectedSrc ? (
              <>
                <img
                  src={selectedSrc}
                  alt={selected.prompt}
                  style={TRANSPARENCY_CHECKER}
                  role="button"
                  tabIndex={0}
                  aria-label={openImageLabel(t, selected.prompt)}
                  onClick={openViewer}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      openViewer();
                    }
                  }}
                  className="max-h-full max-w-full cursor-zoom-in object-contain shadow-sm"
                />
                {/* No button borders: focus returning from a menu would draw one. */}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-xl bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur [&_[data-slot=button]]:border-0 [&_[data-slot=button]:focus-visible]:bg-muted">
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    aria-label={t("library.viewer.openImage")}
                    title={t("library.viewer.openImage")}
                    onClick={(event) => {
                      // Safari does not focus a clicked button, and the viewer returns focus to what had it.
                      event.currentTarget.focus();
                      openViewer();
                    }}
                  >
                    <HugeiconsIcon icon={ArrowExpand01Icon} className="size-4" />
                  </Button>
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
                    favorite={isFavorite(`image:${selected.id}`)}
                    onToggleFavorite={() => toggleFavorite(`image:${selected.id}`)}
                    onTogglePin={() =>
                      void handleTogglePin(selected.id, !selected.pinned)
                    }
                    onToggleArchive={() => void handleArchive(selected.id)}
                    onDelete={() => void handleDelete(selected.id)}
                    onDownload={() => void handleQuickDownload(selected)}
                    onAddToProject={(projectId) => addGalleryImageToProject(selected.id, projectId)}
                  />
                </div>
              </>
            ) : selected ? (
              <>
                {selectedThumb && (
                  <img
                    src={selectedThumb}
                    alt={selected.prompt}
                    style={{ maxWidth: selected.width, maxHeight: selected.height }}
                    className="size-full object-contain"
                  />
                )}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-xl bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur [&_[data-slot=button]]:border-0 [&_[data-slot=button]:focus-visible]:bg-muted">
                  {srcErrors[selected.id] ? (
                    <>
                      <span role="alert" className="sr-only">Full-resolution download failed.</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-1.5"
                        title="Full-resolution download failed. Retry downloading."
                        onClick={() => void ensureSrc(selected)}
                      >
                        <HugeiconsIcon icon={Refresh01Icon} className="size-4" />
                        Retry download
                      </Button>
                    </>
                  ) : (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="gap-1.5"
                      disabled
                      title="Loading full-resolution image…"
                    >
                      <Spinner className="size-4" label="Loading full-resolution image" />
                      Loading image…
                    </Button>
                  )}
                </div>
              </>
            ) : busy === "generating" ? null : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <HugeiconsIcon icon={Image03Icon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded
                    ? "Enter a prompt and hit Generate."
                    : "Select a diffusion model to load"}
                </p>
              </div>
            )}

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
              {...stripReorder.stripProps}
              className="hover-scrollbar flex shrink-0 gap-2 overflow-x-auto border-t border-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] px-10 max-sm:px-5 py-3"
              onScroll={(e) => {
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center overflow-hidden rounded-lg bg-muted/50 ring-2 ring-primary/30">
                  {livePreviewSrc ? (
                    <img src={livePreviewSrc} alt="" className="size-full object-cover" />
                  ) : (
                    <Spinner className="size-5 text-muted-foreground" />
                  )}
                </div>
              )}
              {images.map((image) => (
                <div
                  key={image.id}
                  data-image-id={image.id}
                  {...stripReorder.tileProps(image.id)}
                  className={cn(
                    "group relative size-16 shrink-0",
                    stripReorder.draggingId === image.id && "opacity-40",
                  )}
                >
                  {stripReorder.cue?.id === image.id && (
                    <StripDropLine edge={stripReorder.cue.edge} />
                  )}
                  <button
                    type="button"
                    onClick={() => setSelectedId(image.id)}
                    className="relative size-full overflow-hidden rounded-[10px] bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {(thumbById[image.id] ?? srcById[image.id]) ? (
                      <img
                        src={thumbById[image.id] ?? srcById[image.id]}
                        alt={image.prompt}
                        draggable={false}
                        className="size-full object-cover"
                      />
                    ) : (
                      <span className="flex size-full items-center justify-center">
                        <Spinner className="size-4 text-muted-foreground" />
                      </span>
                    )}
                    {/* Selection marker on a non-focusable overlay, so the button's focus state cannot mask it. */}
                    {image.id === selected?.id && (
                      <span className="pointer-events-none absolute inset-0 rounded-[10px] border border-border bg-white/35 dark:border-[rgb(255_255_255_/_calc(0.25*var(--contrast-edge-gain,1)))] dark:bg-white/20" />
                    )}
                  </button>
                  {image.pinned && (
                    <GalleryPinBadge
                      noun="image"
                      className="bottom-0.5 left-0.5"
                      onUnpin={() => void handleTogglePin(image.id, false)}
                    />
                  )}
                  <div className="absolute right-0.5 top-0.5">
                    <GalleryItemMenu
                      variant="overlay"
                      noun="image"
                      active={active}
                      pinned={Boolean(image.pinned)}
                      archived={Boolean(image.archived)}
                      favorite={isFavorite(`image:${image.id}`)}
                      onToggleFavorite={() => toggleFavorite(`image:${image.id}`)}
                      onTogglePin={() => void handleTogglePin(image.id, !image.pinned)}
                      onToggleArchive={() => void handleArchive(image.id)}
                      onDelete={() => void handleDelete(image.id)}
                    onDownload={() => void handleQuickDownload(image)}
                    onAddToProject={(projectId) => addGalleryImageToProject(image.id, projectId)}
                    />
                  </div>
                </div>
              ))}
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
