// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { generationFailureLogsAction } from "@/features/settings/lib/view-logs-action";
import { readImageModel, redirectRetiredImagePick, rememberImageModel, matchesRememberedModel, componentFilesMatch, type RememberedImageModel } from "./image-model-recall";
import { componentFileFields, splitComponentFileList } from "./component-files";
import type { CompareContext } from "./compare/compare-axes";
import type { CompareRun } from "./compare/compare-run";
// Statically imported: this page is already its own lazy chunk, and a nested import() of the panel split modules
// the startup set shares out of their eager chunks (+3 startup chunks, measured with bundle:check).
import ComparePanel from "./compare/compare-panel";
import {
  type DragEvent,
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
  Delete02Icon,
  DiceFaces03Icon,
  Download01Icon,
  Edit03Icon,
  FolderAddIcon,
  Image03Icon,
  ImageAdd02Icon,
  ImageCropIcon,
  InformationCircleIcon,
  PlusSignIcon,
  Refresh01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { MessageCircleIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { MediaViewer } from "@/components/media-viewer";
import { shortPrompt } from "@/lib/prompt-text";

import { ImageDropzone, useGalleryImageDropSource } from "@/components/image-dropzone";
import {
  readNativeAttachmentFile,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { useImageCrop } from "@/hooks/use-image-crop";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { buildImagesTourSteps } from "./tour";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
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
  ExternalModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { GalleryItemMenu, GalleryPinBadge } from "@/components/gallery-item-menu";
import { MediaRailResizeHandle } from "@/components/media-rail-resize-handle";
import { MEDIA_RAIL_ROOT_ATTR, useMediaRailWidth } from "@/hooks/use-media-rail-width";
import { StripDropLine } from "@/components/gallery-strip-reorder";
import { useStripReorder } from "@/hooks/use-strip-reorder";
import { MediaApiButton } from "@/components/media-api-button";
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
  subscribeArchiveChanged,
  subscribeGalleryChanged,
} from "@/lib/gallery-flags";
import {
  dismissExample,
  isExampleDismissed,
  readLastPrompt,
  saveLastPrompt,
} from "@/lib/last-prompt";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useImageEditHandoffStore } from "@/stores/image-edit-handoff";
import { isWorkflowEnabled, useImageWorkflowStore } from "./stores/image-workflow-store";
import {
  WORKFLOW_EXAMPLE_PROMPTS,
  WORKFLOW_TABS,
  recipeWorkflowLabel,
  type WorkflowId,
} from "./workflows";
import {
  ParamSlider,
  buildExternalModelId,
  isDecisionConnection,
  parseExternalModelId,
  useExternalProvidersStore,
} from "@/features/chat";
import { ModelLoadDescription } from "@/features/chat/components/model-load-status";
import { cancelConnectionImageGeneration } from "./connection-cancel";
import {
  type ImageGenerationPresetParams,
  MediaGenerationPresetControl,
  useMediaGenerationPresets,
} from "@/features/generation-presets";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { formatBytes, formatEta } from "@/features/hub/lib/format";
import { OnDeviceFoldersDialog } from "@/features/hub/catalog/on-device-folders-dialog";
import { generatePhaseLabel, sameGenerateProgress } from "@/lib/media-generate-phase";
import { Recycle } from "lucide-react";
import { NegativePromptField, PROMPT_BOX_CLASS } from "@/components/negative-prompt-field";
import {
  ASPECT_RATIOS,
  AspectRatioSelect,
  DimensionSelect,
  matchAspect,
} from "@/components/media-size-controls";
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
import { useLoadAnywayPrompt } from "@/components/load-anyway-dialog";
import { readAllowOversizedLoad, retryWhileCurrent, shouldOfferLoadAnyway } from "@/lib/unified-memory-load-refusal";
import { evictedSlotsFrom, evictedToastOptions, resyncEvicted, type EvictedSlot } from "@/lib/evicted-toast";
import { failureMessage, isMultiModelRefusal, MULTI_MODEL_REFUSAL_TITLE, refusalHeadersOf } from "@/lib/multi-model-refusal";
import { watchSavingPhase } from "@/lib/swap-save-phase";
import { modelMemorySettingsAction } from "@/features/settings/lib/model-memory-action";
import { MAX_UPLOAD_IMAGE_SIDE, blobToDataUrl, fitImageDataUrl } from "@/lib/image-fit";
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
  stepsSliderMax,
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
import type { AppliedMask } from "./lib/mask-from-text";
import { applyMaskToLayer } from "./mask-apply";
import { MaskFromTextPanel } from "./mask-from-text-panel";
import { DetailerField, useDetailerSegmenter } from "./detailer-field";
import {
  detailerAvailable,
  detailerNotApplied,
  detailerEngagesNegative,
  detailerErrors,
  detailerModelCapable,
  detailerRecordOf,
  detailerRequestField,
  detailerSummary,
  readDetailerOptions,
  restoredDetailer,
  saveDetailerOptions,
  type DetailerOptions,
} from "./lib/detailer";

import {
  type ControlNetSpecInput,
  type DiffusionControlNetInfo,
  type DiffusionGenerateProgress,
  type DiffusionGenerateRequest,
  type DiffusionGenerateResponse,
  type DiffusionLoadProgress,
  type DiffusionLoadRequest,
  type DiffusionLoraInfo,
  type DiffusionSamplerOption,
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
  listDiffusionLoras,
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
import { drawInfoStamp, infoStampText } from "./lib/info-stamp";
import { lastSeedOf, reusableSeed } from "./lib/seed-controls";
import { PasteParametersDialog } from "./paste-parameters-dialog";
import {
  INPAINT_PADDING_DEFAULT,
  INPAINT_PADDING_MAX,
  type InpaintMaskRequestFields,
  inpaintMaskRequestFields,
  MASK_BLUR_DEFAULT,
  MASK_BLUR_MAX,
  inpaintOptionsNotApplied,
  restoredInpaintMaskOptions,
} from "./lib/inpaint-options";
import {
  copyPngTextChunks,
  type ImportedRecipe,
  importedRecipe,
  importNotApplied,
  literalPromptNotApplied,
  variationApplies,
  textualInversionLabel,
  isGenerationParametersPaste,
  loraTagNames,
  matchLoraTagsForImport,
  MAX_LORAS,
  lorasOverLimit,
  newerRecipeSchema,
  pastedParameters,
  pngRecipe,
  readEmbeddedRecipe,
  recipeSettings,
  unknownRecipeKeys,
  withJpegExif,
  withWebpExif,
} from "./lib/image-metadata";
import { useLibrarySettingsStore } from "@/features/library/settings-store";
import {
  ALLOW_OVERSIZED_HINT,
  ALLOW_OVERSIZED_LABEL,
  allowOversizedField,
  GENERATE_ANYWAY_LABEL,
  MEMORY_REFUSAL_TITLE,
  shouldOfferGenerateAnyway,
  shouldRunQueuedOversizedRetry,
} from "./lib/memory-refusal";
import {
  AUTOMATIC_SCHEDULE,
  DEFAULT_SAMPLER,
  normalizeSamplerChoice,
  readSamplerChoices,
  type SamplerChoice,
  samplerRecipeLabel,
  samplerRequestFields,
  SAMPLER_LABELS,
  SCHEDULE_LABELS,
  withSamplerChoice,
} from "./lib/sampler-choice";
import {
  EMPTY_SHIFT,
  shiftFromRecipe,
  shiftNotApplied,
  shiftRequestFields,
  splitInstalledEmbeddings,
  type ShiftValue,
  withEmbeddingTrigger,
} from "./lib/shift-choice";
import { listDiffusionEmbeddings } from "./embeddings-api";
import { addGalleryImageToProject, listDiffusionControlNets, moveGalleryImage } from "./images-page-api";
import { EmbeddingsField, ShiftField } from "./shift-embeddings-controls";
import { HiresControls, UpscaleModelField } from "./hires-controls";
import {
  HIRES_OFF,
  HIRES_UPSCALER_LABELS,
  hiresBaseSize,
  hiresEngagesNegative,
  hiresFromRecipe,
  hiresRecordUsedNegative,
  hiresNotApplied,
  hiresRequest,
  hiresSummary,
  hiresVisible,
  resolveRestoredHires,
  upscaleModelRequest,
  type HiresChoice,
} from "./lib/hires-choice";
import {
  enhancePromptRequestField,
  enhanceRestoreFor,
  enhancementNotice,
  hasEnhancedPrompt,
  ENHANCE_UNSUPPORTED_FAMILY,
  overrideAfterFamilyChange,
  promptUpsamplerCapability,
  readEnhanceChoices,
  resolveEnhancePrompt,
  upsamplerVariantFor,
  withEnhanceChoice,
} from "./lib/prompt-enhance";
import { EnhancedPromptRecipeRow, EnhancePromptControls } from "./prompt-enhance-panel";
import {
  ENGINE_HINT,
  ENGINE_OPTIONS,
  type EngineChoice,
  engineModelKey,
  engineRequestField,
  engineSelectVisible,
  loadedEngineChoice,
  nativeLoadAdvanced,
  nativeReloadTarget,
  offersCompiledLoraReload,
  offersNativeLoraReload,
  reapplyChangesEngine,
  reapplyTarget,
  readEngineChoices,
  reloadSource,
  residentReloadTarget,
  rememberLoadedEngine,
  sendsEngine,
  withEngineChoice,
} from "./lib/engine-choice";
import {
  canVaryImage,
  DEFAULT_SD_CONTROLS,
  FALLBACK_VARY_STRENGTH,
  parseVariationSeed,
  randomVariationSeed,
  sdControlsFromRecipe,
  sdControlsNotRestored,
  sdControlsRecipeRows,
  sdControlsRequestFields,
  type SdControlsChoice,
  varyBlockedReason,
} from "./lib/sd-controls";
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

/** Whether this pick may receive a transformer precision request. Unknown repos defer to the backend. */
function sendsTransformerQuant(kind: string | null | undefined, repoId: string): boolean {
  return (
    isDenseQuantKind(kind) &&
    curatedArtifactTakesDenseQuant(repoId, IMAGE_CATALOG) !== false
  );
}

// Curated models come from the shared catalog, one group per model with its artifacts as data and
// the load kind per artifact from loadSpecFor. Built per render, since a host that can only run
// the native engine is not offered pipeline rows.
function useImageModels(
  host: HostClass,
  denseQuantSchemes: readonly string[],
): ModelOption[] {
  return useMemo(
    () => catalogToModelOptions(IMAGE_CATALOG, host, denseQuantSchemes),
    [host, denseQuantSchemes],
  );
}

// Workflow tabs. `requires` is the backend workflow id (status.workflows) the model must
// support; null = always. Each conditioned workflow names the images it consumed for the restore
// toast, since a recipe keeps the scalar settings but not the uploads. Keys are the backend's own
// workflow strings; txt2img is absent because it restores completely.
// Own keys only: an imported id like "constructor" must not read Object.prototype.
const labelOf = (labels: Record<string, string>, id: string) =>
  Object.hasOwn(labels, id) ? labels[id] : id;

// Identifies a loaded model for the once-per-model Only masked notice.
const onlyMaskedKeyOf = (s: DiffusionStatus): string => `${s.repo_id ?? ""}|${s.family ?? ""}`;
// The load an import started under: model plus the engine and build, so a reload of the same repo on
// another engine or quant counts as a change. Status carries no per-load id; an identical reload leaves
// the same LoRA catalog, so its matches stay valid.
const loadedModelKey = (s: DiffusionStatus | null): string | null =>
  s?.loaded
    ? JSON.stringify([
        s.repo_id,
        s.base_repo,
        s.family,
        s.engine,
        s.gguf_filename,
        s.dtype,
        s.transformer_quant,
        s.text_encoder_quant,
      ])
    : null;

const CONDITIONED_WORKFLOW_INPUTS: Record<string, string> = {
  img2img: "the source image",
  inpaint: "the source image and mask",
  outpaint: "the source image",
  upscale: "the source image",
  edit: "the source image",
  reference: "the source and reference images",
  controlnet: "the control image",
};

// Friendly labels for ControlNet control types; unknown types fall back to a capitalized "(map)" label.
const CONTROL_TYPE_LABELS: Record<string, string> = {
  passthrough: "Passthrough (already a map)",
  canny: "Canny (trace edges)",
  depth: "Depth (map)",
  pose: "Pose (map)",
};

// Convenient drag range for the Runs slider; the number box accepts higher typed values on purpose.
const RUNS_SLIDER_MAX = 128;
// Offered sizes; a locked ratio can derive one off-list, so the current value is added in.
const DIM_OPTIONS = [
  256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280,
  1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2752,
];

// Larger limits than this unlock the 2K presets.
const MAX_OUTPUT_DEFAULT = DEFAULT_SIZE_LIMITS.maxSide;

function dimOptions(limits: SizeLimits): number[] {
  return DIM_OPTIONS.filter((n) => n <= limits.maxSide && n % limits.multiple === 0);
}


// Module cache of the backend-persisted gallery, so a tab switch re-renders instantly; object
// URLs are revoked only on delete. The 192 MB blob budget (~100-200 images) never evicts a
// visible image.
const IMAGE_BLOB_BUDGET_BYTES = 192 * 1024 * 1024;

const galleryCache: {
  images: GalleryImage[];
  hasMore: boolean;
  selectedId: string | null;
  quant: string | null;
  srcById: BlobUrlCache;
  // Ids with a fetch in flight, so concurrent ensureSrc calls do not double-fetch and leak a duplicate object URL.
  inflight: Set<string>;
  // Keep thumbnails separate from originals used for viewing and downloads.
  thumbById: BlobUrlCache;
  thumbInflight: Set<string>;
  // Ids deleted while their PNG was still downloading: a fetch landing afterwards must throw its blob away.
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

// Re-encoded client-side (JPEG flattened onto white), with the info stamp when it is on. Every
// format carries the stored recipe: PNG keeps its text chunks, JPEG / WebP get it as EXIF.
async function encodeExport(
  src: string,
  original: Uint8Array,
  image: GalleryImage,
  format: ImageExportFormat,
  stamp: boolean,
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
  if (stamp) {
    drawInfoStamp(ctx, canvas.width, canvas.height, infoStampText(image));
  }
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
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const recipe = pngRecipe(original);
  const out =
    format === "png"
      ? copyPngTextChunks(original, bytes)
      : format === "jpeg"
        ? withJpegExif(bytes, recipe)
        : withWebpExif(bytes, recipe, canvas.width, canvas.height);
  return new Blob([out], { type: `image/${format}` });
}

async function downloadImage(
  image: GalleryImage,
  format: ImageExportFormat = "png",
  src?: string,
) {
  const stamp = useLibrarySettingsStore.getState().imageInfoStamp;
  let ownedUrl: string | null = null;
  try {
    let output: Blob;
    let outputFormat = format;
    if (format === "png" && !stamp) {
      // The stored bytes verbatim. WebKit can display the cached object URL but fail to fetch it
      // again, so the native save re-fetches the authenticated original.
      if (src && !isTauri) {
        const filename = exportFilename(image, "png");
        await downloadUrl(src, filename);
        return;
      }
      output = await fetchGalleryBlob(image.url);
    } else {
      const original = await fetchGalleryBlob(image.url);
      const source = src ?? (ownedUrl = URL.createObjectURL(original));
      try {
        output = await encodeExport(
          source,
          new Uint8Array(await original.arrayBuffer()),
          image,
          format,
          stamp,
        );
      } catch {
        outputFormat = "png";
        output = original;
      }
    }
    const filename = exportFilename(image, outputFormat);
    await downloadFile(output, filename, output.type || "image/png");
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
  } finally {
    if (ownedUrl) {
      URL.revokeObjectURL(ownedUrl);
    }
  }
}

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function genStepLabel(p: DiffusionGenerateProgress): string {
  return generatePhaseLabel({ ...p, total: p.total_steps }, { formatEta });
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
  if (p.phase === "saving_chat_context") {
    return (
      <ModelLoadDescription
        title="Saving chat context…"
        message="Saving the chat model's context so it can resume after this load."
      />
    );
  }
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

const NO_SAMPLERS: DiffusionSamplerOption[] = [];
const IMAGE_CONNECTION_TYPES = new Set(["custom", "vllm", "llama_cpp", "lemonade", "ollama"]);

const IDLE_PROGRESS: DiffusionLoadProgress = {
  phase: null,
  bytes_downloaded: 0,
  bytes_total: 0,
  fraction: 0,
  error: null,
};

// What the load toast shows while the backend saves the chat model's context ahead of this load.
const SAVING_PROGRESS: DiffusionLoadProgress = { ...IDLE_PROGRESS, phase: "saving_chat_context" };


// The request one Generate run sends: a compare cell's own, else the form's.
function compareRequest(
  run: CompareRun | null,
  i: number,
  baseSeed: number,
  body: DiffusionGenerateRequest,
): DiffusionGenerateRequest {
  return run ? run.request(i, body, baseSeed) : body;
}

// One row: label, track, value. The Images sliders are Chat ParamSlider, so both pages share one
// control.
function SliderField({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
}: {
  label: string;
  hint?: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
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
      disabled={disabled}
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
        "shrink-0 rounded-full px-1 py-px text-ui-9 font-medium uppercase tracking-wider",
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

// Brush-based mask editor for inpainting, exporting a grayscale PNG mask at NATIVE resolution
// (white = repaint). `brushPct` sizes the brush against the shorter side.
function MaskCanvas({
  image,
  brushPct,
  resetKey,
  onMaskChange,
  applyMask,
}: {
  image: string;
  brushPct: number;
  resetKey: number;
  onMaskChange: (dataUrl: string | null) => void;
  /** A "Mask from text" result; each new key is drawn into the mask once, then the brush keeps editing it. */
  applyMask?: AppliedMask | null;
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

  // Applied once per key, after the mask is sized for this image; Clear or a new image starts blank as before. A
  // result from before this canvas mounted (another workflow's painter) is not replayed.
  const appliedKey = useRef<number | null>(applyMask?.key ?? null);
  useEffect(() => {
    if (!applyMask || !ready || appliedKey.current === applyMask.key) return;
    appliedKey.current = applyMask.key;
    const disp = dispRef.current;
    const mask = maskRef.current;
    if (!disp || !mask) return;
    let live = true;
    applyMaskToLayer({
      layer: mask,
      display: disp,
      dataUrl: applyMask.dataUrl,
      additive: applyMask.additive,
      overlay: "mask",
      isCurrent: () => live,
    })
      .then((url) => {
        // An empty Replace leaves the mask black: nothing is selected, so Generate's empty-mask check still holds.
        if (live && url !== null) onMaskChange(applyMask.empty ? null : url);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [applyMask, ready, onMaskChange]);

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
    <div className="relative overflow-hidden rounded-lg border border-border bg-muted/30">
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
  onRestore: (image: GalleryImage, exactCaption?: boolean) => void;
  active: boolean;
}) {
  // Controlled + force-closed off-tab: PopoverContent portals to body, so the inert page wrapper cannot contain it.
  const [open, setOpen] = useState(false);
  // Also clear the flag when leaving the tab so it does not reopen on return.
  useEffect(() => {
    if (!active) setOpen(false);
  }, [active]);
  const viaConnection = image.model_kind === "connection";
  return (
    <Popover open={active && open} onOpenChange={(o) => setOpen(active && o)}>
      <PopoverTrigger asChild>
        <Button size="sm" variant="ghost" className="gap-1.5">
          <HugeiconsIcon icon={InformationCircleIcon} className="size-4" />
          Recipe
        </Button>
      </PopoverTrigger>
      {/* Fits the viewport: only the settings scroll, and overflow-hidden keeps the corners round. */}
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
          {hasEnhancedPrompt(image) ? (
            <EnhancedPromptRecipeRow caption={(image.enhanced_prompt ?? "").trim()} />
          ) : null}
          {image.negative_prompt ? (
            <RecipeRow label="Negative" value={image.negative_prompt} wrap />
          ) : null}
          {image.model ? <RecipeRow label="Model" value={image.model} /> : null}
          {/* The load-time build, so the recipe still names the pipeline once the model is unloaded: the
              repo id alone does not say which quant ran. */}
          {image.gguf_filename ? <RecipeRow label="File" value={image.gguf_filename} mono /> : null}
          {image.transformer_quant ? (
            <RecipeRow label="Transformer" value={image.transformer_quant} />
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
          {image.upscale_model && image.workflow === "upscale" ? (
            <RecipeRow label="Upscaler" value={HIRES_UPSCALER_LABELS[image.upscale_model] ?? image.upscale_model.replace(/^local:/, "")} />
          ) : null}
          {image.hires_scale != null ? (
            <RecipeRow label="Hires fix" value={hiresSummary(hiresFromRecipe(image))} wrap />
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
          {!viaConnection && <RecipeRow label="Steps" value={String(image.steps)} />}
          {!viaConnection && <RecipeRow label="Guidance" value={String(image.guidance)} />}
          {samplerRecipeLabel(image.sampler, image.schedule_type) ? (
            <RecipeRow
              label="Sampler"
              value={samplerRecipeLabel(image.sampler, image.schedule_type) ?? ""}
            />
          ) : null}
          {sdControlsRecipeRows(image).map(([label, value]) => (
            <RecipeRow key={label} label={label} value={value} />
          ))}
          {!viaConnection && <RecipeRow label="Seed" value={String(image.seed)} mono />}
          {detailerSummary(detailerRecordOf(image.detailer)) ? (
            <RecipeRow label="Detailer" value={detailerSummary(detailerRecordOf(image.detailer)) ?? ""} />
          ) : null}
        </div>
        <div className="shrink-0 border-t border-border/60 px-3 py-2.5">
          <Button size="sm" className="w-full gap-1.5" onClick={() => onRestore(image)}>
            <HugeiconsIcon icon={Refresh01Icon} className="size-4" />
            Restore these settings
          </Button>
          {/* The caption that actually ran, with Enhance prompt off, so the run reproduces without the upsampler. */}
          {image.prompt_enhancement === "applied" && hasEnhancedPrompt(image) ? (
            <Button
              size="sm"
              variant="ghost"
              className="mt-1 w-full gap-1.5"
              onClick={() => onRestore(image, true)}
            >
              Restore with the enhanced prompt
            </Button>
          ) : null}
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

/** What the LOADED model is actually running, read from status and never from the request:
 *  transformer and text-encoder precision, memory mode with its offload behaviour, and the
 *  attention backend. The Advanced selects say what was ASKED for; this says what happened. */
function LoadedBuildSummary({ status }: { status: DiffusionStatus | null }) {
  if (!status?.loaded) return null;
  const offload = status.offload_policy ?? "none";
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border/60 px-2.5 py-2 text-ui-11">
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

/** Report a failed load. A refused precision or a Model memory refusal is a long actionable
 *  sentence, so it becomes a toast description under a short title. Nothing is left half-loaded. */
function reportLoadFailure(failure: unknown, fallback: string): void {
  const text = failureMessage(failure).trim();
  if (isMultiModelRefusal(null, refusalHeadersOf(failure), text)) {
    toast.error(MULTI_MODEL_REFUSAL_TITLE, { description: text || fallback, action: modelMemorySettingsAction() });
    return;
  }
  if (text && isPrecisionRefusal(text)) {
    toast.error(PRECISION_REFUSAL_TITLE, { description: text });
    return;
  }
  toast.error(text || fallback);
}

type Busy = "loading" | "unloading" | "generating" | null;
// baseRepo: only a reload of a resident model the page did not load sends one, to rebuild the same companions.
type ImageLoadOptions = { kind: "gguf" | "single_file" | "pipeline"; filename?: string; displayRepoId?: string; baseRepo?: string };
type LastLoad = { repoId: string } & ImageLoadOptions & Pick<RememberedImageModel, "textEncoderFiles" | "vaeFile">;

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
  | "text_encoder_quant"
  | "attention_backend"
  | "memory_mode"
  | "transformer_cache"
  | "family_override"
  | "loras"
  | "gpu_ids"
  | "text_encoder_file"
  | "vae_file"
  | "engine"
  | "allow_oversized_load"
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
  const imageInfoStamp = useLibrarySettingsStore((state) => state.imageInfoStamp);
  const [rememberedModel, setRememberedModel] = useState(readImageModel);
  const pendingRecalledGeneration = useRef<{ model: RememberedImageModel; load: number; workflow: WorkflowId; allowOversized?: boolean } | null>(null);
  const { isMobile, pinned } = useSidebar();
  const hostClass = useHostClass();
  const denseQuantSchemes = useDenseQuantSchemes();
  const nvfp4Diffusion = useNvfp4Diffusion();
  const nvfp4DiffusionKnown = useNvfp4DiffusionKnown();
  const imageModels = useImageModels(hostClass, denseQuantSchemes);
  const connections = useExternalProvidersStore((s) => s.providers);
  const connectionsEnabled = useExternalProvidersStore((s) => s.connectionsEnabled);
  const externalModels = useMemo<ExternalModelOption[]>(
    () =>
      connectionsEnabled
        ? connections
            .filter((c) => IMAGE_CONNECTION_TYPES.has(c.providerType) && !isDecisionConnection(c))
            .flatMap((c) =>
              c.models.map((model) => ({
                id: buildExternalModelId(c.id, model),
                name: model,
                providerId: c.id,
                providerName: c.name,
                providerType: c.providerType,
              })),
            )
        : [],
    [connections, connectionsEnabled],
  );
  const [externalModelId, setExternalModelId] = useState<string | null>(null);
  const runTarget = useRef<"connection" | undefined>(undefined);
  const external = useMemo(
    () =>
      externalModels.some((m) => m.id === externalModelId)
        ? parseExternalModelId(externalModelId)
        : null,
    [externalModels, externalModelId],
  );
  const { rootStyle: railRootStyle } = useMediaRailWidth("images");
  const [quant, setQuant] = useState<string | null>(galleryCache.quant);
  // One prompt per workflow, starting from the last one generated with.
  const [prompts, setPrompts] = useState<Record<WorkflowId, string>>(() =>
    Object.fromEntries(
      WORKFLOW_TABS.map(({ id }) => [id, readLastPrompt(`images:${id}`)]),
    ) as Record<WorkflowId, string>,
  );
  // Workflows whose example hint is gone: it shows as a placeholder until the box is first focused.
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
  // Writes the active workflow's prompt, read at call time so a stale closure cannot write another's.
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
  // width/height are the source of truth; `aspect` locks their proportion and `portrait` tracks
  // orientation, so Flip keeps the lock.
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [aspect, setAspect] = useState("1:1");
  const [portrait, setPortrait] = useState(false);
  // Z-Image-Turbo official defaults: 9 steps (= 8 DiT forwards), guidance 0 (distilled, CFG-free).
  const [steps, setSteps] = useState(DEFAULT_GEN.steps);
  const [guidance, setGuidance] = useState(DEFAULT_GEN.guidance);
  const [samplerChoices, setSamplerChoices] = useState(readSamplerChoices);
  // Engine pick per GGUF checkpoint, recorded only by a load that landed with it.
  const [engineChoices, setEngineChoices] = useState(readEngineChoices);
  const [enhanceChoices, setEnhanceChoices] = useState(readEnhanceChoices);
  // The family it was set under ("" = before a model loaded, which applies to whichever family loads next).
  const [enhanceOverrideState, setEnhanceOverrideState] = useState<{ family: string; on: boolean } | null>(null);
  // A recipe restored while unloaded: Generate recalls the model, uses this, and stores it for that model.
  const [restoredSampler, setRestoredSampler] = useState<Partial<SamplerChoice> | null>(null);
  // Whether the user has taken the recipe since the pick still waiting for its status: a preset
  // selected while the model downloaded is newer than that pick.
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
  // The recipe a pick optimistically claimed until status confirms it or a failed load reverts
  // it; without this the Default preset reads as "modified" for the whole download.
  const [pendingModelDefaults, setPendingModelDefaults] = useState<{
    steps: number;
    guidance: number;
  } | null>(null);
  const [seed, setSeed] = useState("");
  const [maskBlur, setMaskBlur] = useState(MASK_BLUR_DEFAULT);
  const [inpaintOnlyMasked, setInpaintOnlyMasked] = useState(false);
  const [inpaintPadding, setInpaintPadding] = useState(INPAINT_PADDING_DEFAULT);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  // Batch size = images per forward pass (VRAM-heavy); count = sequential loops.
  const [batchSize, setBatchSize] = useState(1);
  const [count, setCount] = useState(1);
  // Active workflow tab: create = text-to-image, transform = img2img, inpaint = mask-guided
  // redraw. Workflow and page mode live in a store so the sidebar submenu can drive them.
  const workflow = useImageWorkflowStore((s) => s.workflow);
  const prompt = prompts[workflow];
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
  // The latest "Mask from text" result, handed to whichever painter is showing; a new key applies it once.
  const [textMask, setTextMask] = useState<AppliedMask | null>(null);
  const onTextMask = useCallback(
    (mask: { dataUrl: string; additive: boolean }) =>
      setTextMask((prev) => ({ ...mask, key: (prev?.key ?? 0) + 1 })),
    [],
  );
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
  const [upscaleModel, setUpscaleModel] = useState<string | null>(null);
  const [hires, setHires] = useState<HiresChoice>(HIRES_OFF);
  // Reference and Edit: the ADDITIONAL images after the source. "" holds a cleared slot so others do not renumber.
  const [referenceImages, setReferenceImages] = useState<string[]>([]);
  const [referenceResolution, setReferenceResolution] = useState<number | null>(null);
  const [editSizing, setEditSizing] = useState<EditSizing>("source");
  const [matchResolution, setMatchResolution] = useState(1024);
  const [localizedMode, setLocalizedMode] = useState<LocalizedEditMode | null>(null);
  const [localizedLayer, setLocalizedLayer] = useState<string | null>(null);
  const [localizedColor, setLocalizedColor] = useState(ANNOTATION_COLORS[0].value);
  const [localizedColors, setLocalizedColors] = useState<string[]>([]);
  const [localizedResetKey, setLocalizedResetKey] = useState(0);
  // LoRA adapters selected for the next generation (id + weight), plus the list the picker offers.
  const [loras, setLoras] = useState<LoraSpecInput[]>([]);
  const [availableLoras, setAvailableLoras] = useState<DiffusionLoraInfo[]>([]);
  // The family availableLoras was fetched for; undefined while none is current (loading, failed, no model).
  const [loraCatalogFamily, setLoraCatalogFamily] = useState<string | null | undefined>(undefined);
  // Fine-tuned (trained in Unsloth) first, for both the suggestions and the Add LoRA prefill.
  const sortedLoras = useMemo(
    () => [...availableLoras].sort((a, b) => Number(Boolean(b.fine_tuned)) - Number(Boolean(a.fine_tuned))),
    [availableLoras],
  );
  // Page mode: "create" is the generation workspace, "train" the LoRA training workspace.
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
  // Train family + base live here so the top bar can pick them, replacing the generation model selector on Train.
  const [trainFamilies, setTrainFamilies] = useState<TrainFamilyOption[]>([]);
  const [trainFamilyName, setTrainFamilyName] = useState("flux.1");
  const [trainBaseChoice, setTrainBaseChoice] = useState("");
  // Bumped when a training run completes, so the LoRA discovery effect rescans without a model reload.
  const [loraRefreshKey, setLoraRefreshKey] = useState(0);
  const [loraFoldersOpen, setLoraFoldersOpen] = useState(false);
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
  const [allowOversized, setAllowOversized] = usePersistedToggle(
    "unsloth_images_allow_oversized",
  );
  // Live latent preview while denoising, on by default: only the opt-out is stored.
  const [livePreviewOff, setLivePreviewOff] = usePersistedToggle("unsloth_images_live_preview_off");
  const livePreview = !livePreviewOff;
  const oversizedOnce = useRef(false);
  // The compare grid run handleGenerate is to send instead of the form's runs (compare/).
  const compareOnce = useRef<CompareRun | null>(null);
  const [sdChoice, setSdChoice] = useState<SdControlsChoice>(DEFAULT_SD_CONTROLS);
  // Set by Vary for the next Generate only: the seed to keep and the variation seed to slerp toward.
  const varyOnce = useRef<{ seed: number; variationSeed: number; strength: number } | null>(null);
  // Set only around a confirmed Load anyway retry.
  const oversizedLoadOnce = useRef(false);
  const lastLoadAttempt = useRef<{ overrideSent: boolean; retry: () => void } | null>(null);
  // Synced after render (handleLoadRef is written during render).
  const retryLoadRef = useRef<((repoId: string, opts: ImageLoadOptions, pinned?: LoadAdvanced) => Promise<boolean>) | null>(null);
  const { offer: offerLoadAnyway, dismiss: dismissLoadAnyway, dialog: loadAnywayDialog } = useLoadAnywayPrompt();
  // Queued: "Generate anyway" is clickable before the refused run releases busy.
  const [oversizedRetryQueued, setOversizedRetryQueued] = useState(false);
  // Advanced (load-time) options; "auto"/"off"/"none" map to the backend defaults. Changing one
  // while loaded shows "Reapply".
  const [modelSelectionAction, setModelSelectionAction] = useState<"load" | "download">("load");
  const [speedMode, setSpeedMode] = useState<"auto" | "off" | "eager" | "default" | "max">("auto");
  // The Engine select edits the LOADED checkpoint's next load (Reapply); any other pick uses its stored choice.
  const [engineChoice, setEngineChoice] = useState<EngineChoice>("auto");
  // The loaded build the Engine select was last seeded from; cleared by a failed load to force a reseed.
  const [engineSeed, setEngineSeed] = useState("");
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
  // Repo id whose defaults were already seeded from a discovered resident model, so we seed
  // once and never clobber a manual edit.
  const seededResident = useRef<string | null>(null);

  const [busy, setBusy] = useState<Busy>(null);
  // Bumped when a load ends: a reload of the same repo and file (other precision, memory or LoRA settings) stops the
  // upsampler on the backend, so Enhance prompt warms it again.
  const [loadEpoch, setLoadEpoch] = useState(0);
  const wasLoading = useRef(false);
  useEffect(() => {
    if (wasLoading.current && busy !== "loading") setLoadEpoch((n) => n + 1);
    wasLoading.current = busy === "loading";
  }, [busy]);
  const [genDone, setGenDone] = useState<number | null>(null);
  const [stopping, setStopping] = useState(false);
  const [genStep, setGenStep] = useState<DiffusionGenerateProgress | null>(null);
  const genPollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  // visibilitychange handler active while a generation poll runs: background tabs clamp
  // setInterval, so returning fires one immediate poll.
  const genVisibilityListener = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  const { familyOverride, setFamilyOverride, familySelect, opaqueKind, selectorModelId } = useFamilyOverride(status, status?.supported_families);
  // A connection is not bound by the resident model's size grid or presets.
  const conditioning = !external && status?.loaded ? (status.conditioning ?? null) : null;
  const sizeLimits = useMemo(() => sizeLimitsFrom(conditioning), [conditioning]);
  const unifiedEdit = Boolean(conditioning?.unified_edit);
  // Edit-only model whose Edit takes more images after the source (Qwen-Image-Edit-Plus).
  const multiImageEdit = !unifiedEdit && Boolean(conditioning?.multi_image_edit);
  const referenceResolutions = conditioning?.reference_resolutions ?? [];
  const maxExtras = maxAdditionalImages(conditioning, workflow === "edit" && unifiedEdit ? localizedMode : null);
  // Controlled so the body-portaled overlays force-close while this page is mounted but off-tab.
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [aspectOpen, setAspectOpen] = useState(false);
  // Records come from the backend (durable); srcById maps each id to its object URL or data URL.
  const [images, setImages] = useState<GalleryImage[]>(() => galleryCache.images);
  const [hasMore, setHasMore] = useState(() => galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(() => galleryCache.selectedId);
  // The base seed of this session's last run, for "Reuse last seed"; before one, the newest record's.
  const [sessionSeed, setSessionSeed] = useState<number | null>(null);
  // The strip lists active images only, so the gallery fallback also needs the archive to be empty: an
  // archived image may be the newest. Checked again whenever the strip changes (an archive drops a row).
  const [archivedEmpty, setArchivedEmpty] = useState<boolean | null>(null);
  // Bumped when the archive changes elsewhere (Settings restore or delete), which the strip may not show.
  const [archiveChangeKey, setArchiveChangeKey] = useState(0);
  useEffect(
    () => subscribeArchiveChanged("images", () => setArchiveChangeKey((k) => k + 1)),
    [],
  );
  useEffect(() => {
    if (sessionSeed !== null || hasMore) return;
    let cancelled = false;
    getGallery(0, 1, true)
      .then((page) => {
        if (!cancelled) setArchivedEmpty(page.images.length === 0);
      })
      .catch(() => {
        if (!cancelled) setArchivedEmpty(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionSeed, hasMore, images.length, archiveChangeKey]);
  const lastSeed = useMemo(
    () => reusableSeed(sessionSeed, images, hasMore || archivedEmpty !== true),
    [sessionSeed, images, hasMore, archivedEmpty],
  );
  const [srcById, setSrcById] = useState<Record<string, string>>(() =>
    galleryCache.srcById.toRecord(),
  );
  const [thumbById, setThumbById] = useState<Record<string, string>>(() =>
    galleryCache.thumbById.toRecord(),
  );
  const [srcErrors, setSrcErrors] = useState<Record<string, boolean>>({});
  // Guards a "load more" so a fast scroll cannot fire several at once.
  const loadingMore = useRef(false);
  // Observer root for loading thumbnails near the visible strip.
  const stripRef = useRef<HTMLDivElement | null>(null);
  // Ids currently intersecting the strip; the blob cache never evicts these.
  const visibleIds = useRef<Set<string>>(new Set());
  // False once the page truly unmounts. Tab switches keep it mounted, so a batch keeps generating off-tab.
  const isMounted = useRef(true);
  // Set by Stop for one handleGenerate call: the backend cancel only reaches the denoise running
  // RIGHT NOW, so a count > 1 request would start its next run straight after.
  const cancelRequested = useRef(false);
  // True only once the backend answered {cancelled: true}. Anything else, a POST that never landed
  // or {cancelled: false} because the run was already past its last cancellation check, means it was
  // NOT stopped, so an error raised afterwards is a real failure.
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
  // What the load POST unloaded to make room, shown on the "Model loaded" toast it ends with.
  const pendingEvicted = useRef<EvictedSlot[]>([]);
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
  const residentDefaults = residentDefaultsKey(
    status?.repo_id ?? "",
    status?.base_repo,
    status?.resolved?.family_override,
    status?.gguf_filename,
  );
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
    (
      repoId: string,
      effectiveFamilyOverride = familyOverride,
      forceLoad = false,
      ggufFilename: string | null = null,
    ) => {
      if (modelSelectionAction === "download" && !forceLoad) return;
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
      const recommended = defaultsFor(defaultsKeyFor(repoId, effectiveFamilyOverride, ggufFilename));
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
    // Cancel picks that could reload the ejected model. Download-only work carries no pick token
    // at all now, so it keeps running without needing an exception here -- the exception it used to
    // need was inert anyway, since the marker was cleared when the PLAN resolved rather than when
    // the download finished.
    pickGuard.cancel();
    // That pick can no longer load, so its toast must not keep promising it will.
    pickToast.dismissAll();
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
  }, [dismissLoadToast, pickGuard, pickToast, revertPick]);

  // Mirror to the module cache so a tab switch re-renders instantly.
  useEffect(() => {
    galleryCache.images = images;
    galleryCache.hasMore = hasMore;
    galleryCache.selectedId = selectedId;
    galleryCache.quant = quant;
  }, [images, hasMore, selectedId, quant]);

  // Refresh the LoRA picker when the loaded family changes, since a LoRA is family-specific.
  // Not on first load or unload (a restore can precede the load), so track it in a ref.
  const loraCapable = Boolean(!external && status?.loaded && status?.supports_lora);
  const prevLoraFamilyRef = useRef<string | null | undefined>(undefined);
  // Whether the load in flight baked the LoRA selection into the build (see handleLoad).
  const bakedLorasOnLoad = useRef(false);
  useEffect(() => {
    setLoraCatalogFamily(undefined);
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
        if (cancelled) return;
        setAvailableLoras(list);
        setLoraCatalogFamily(fam);
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
  const controlnetSupported = Boolean(status?.loaded && status?.supports_controlnet);
  const controlnetCapable = !external && controlnetSupported;
  // What Create sends as a ControlNet run (strength 0 runs plain text-to-image): its SDXL pipeline has no CFG rescale.
  const controlnetEngaged =
    controlnetCapable && Boolean(controlnetId && controlImage) && workflow === "create" && controlStrength > 0;
  const negativeCapable = !external && status?.supports_negative_prompt !== false;
  // A model that reads no guidance here runs at its default, which the request and recipe then carry.
  const guidanceFixed = !external && status?.loaded === true && status.supports_guidance === false;
  const effectiveGuidance = guidanceFixed
    ? (status?.generation_defaults?.guidance ?? DEFAULT_GEN.guidance)
    : guidance;
  // The guidance a negative prompt needs: 0 with no model loaded or on an older backend, the old rule.
  const negativeMinGuidance =
    status?.loaded && status.negative_prompt_min_guidance !== undefined
      ? status.negative_prompt_min_guidance
      : 0;
  // A connection runs none of the local engine's Hires fix / model upscalers (the route refuses both).
  const hiresCapability = !external && status?.loaded ? (status.hires ?? null) : null;
  const upscaleModels = !external && status?.loaded ? status.upscale_models : undefined;
  const livePreviewUnavailable =
    !external && status?.loaded === true && status.supports_live_preview === false;
  const samplerOptions = (!external && status?.loaded ? status.samplers : null) ?? NO_SAMPLERS;
  const sdCaps = (!external && status?.loaded ? status.sd_controls : null) ?? null;
  // The checkpoint, not residentDefaults: community SDXL files all share the SDXL base repo.
  const samplerModelKey = status?.loaded ? `${status.repo_id ?? ""}|${status.gguf_filename ?? ""}` : "";
  const samplerChoice = normalizeSamplerChoice(
    samplerOptions,
    restoredSampler ?? samplerChoices[samplerModelKey],
  );
  const changeSamplerChoice = useCallback(
    (next: Partial<SamplerChoice>) => {
      setRestoredSampler(null);
      setSamplerChoices((all) =>
        withSamplerChoice(
          all,
          samplerModelKey,
          normalizeSamplerChoice(samplerOptions, { ...samplerChoice, ...next }),
        ),
      );
    },
    [samplerChoice, samplerModelKey, samplerOptions],
  );
  const shiftControl = !external && status?.loaded ? (status.shift_control ?? null) : null;
  const [shiftState, setShiftState] = useState<{ key: string; value: ShiftValue }>({
    key: "",
    value: EMPTY_SHIFT,
  });
  const shiftValue = shiftState.key === samplerModelKey ? shiftState.value : EMPTY_SHIFT;
  const setShiftValue = useCallback(
    (value: ShiftValue) => setShiftState({ key: samplerModelKey, value }),
    [samplerModelKey],
  );
  const embeddingsEnabled = !external && status?.loaded === true && !!status.textual_inversion;
  // Detailer (face fix): off by default; its options persist, the switch does not.
  const [detailerOn, setDetailerOn] = useState(false);
  const [detailerOptions, setDetailerOptionsState] = useState<DetailerOptions>(readDetailerOptions);
  const setDetailerOptions = useCallback((next: DetailerOptions) => {
    setDetailerOptionsState(next);
    saveDetailerOptions(next);
  }, []);
  const detailerCapable = detailerModelCapable(status, Boolean(external));
  const detailerSegmenter = useDetailerSegmenter(detailerCapable, loadedModelKey(status) ?? "");
  const detailerReady = detailerAvailable(status, Boolean(external), detailerSegmenter);
  // Hires CFG or the Detailer's CFG can turn CFG on for that pass alone; it then reads the negative prompt.
  const negativeEngaged =
    negativeCapable &&
    (negativeMinGuidance === null ||
      effectiveGuidance > negativeMinGuidance ||
      hiresEngagesNegative(hires, hiresCapability, workflow, negativeMinGuidance) ||
      detailerEngagesNegative(detailerOn, detailerReady, detailerOptions, negativeMinGuidance));
  const insertEmbedding = useCallback((trigger: string, target: "prompt" | "negative") => {
    if (target === "negative") setNegativePrompt((p) => withEmbeddingTrigger(p, trigger));
    else setPrompt((p) => withEmbeddingTrigger(p, trigger));
  }, [setPrompt]);
  // Enhance prompt (FLUX.3's prompt upsampler): stored per family, defaulting to the family's own default. A
  // restore or "Use as prompt" sets a one-off override that the next toggle or family change drops.
  const upsamplerCapability = promptUpsamplerCapability(status, Boolean(external));
  const enhanceFamilyKey = upsamplerCapability.supported ? (status?.family ?? "") : "";
  // A family change consumes the override (overrideAfterFamilyChange), a switch to a model or connection without an
  // upsampler included.
  const enhanceTrackKey =
    enhanceFamilyKey || (external || status?.loaded ? ENHANCE_UNSUPPORTED_FAMILY : "");
  useEffect(() => {
    if (enhanceTrackKey) setEnhanceOverrideState((prev) => overrideAfterFamilyChange(prev, enhanceTrackKey));
  }, [enhanceTrackKey]);
  const enhanceOverride =
    enhanceOverrideState &&
    (enhanceOverrideState.family === "" || enhanceOverrideState.family === enhanceFamilyKey)
      ? enhanceOverrideState.on
      : null;
  const setEnhanceOverride = useCallback(
    (on: boolean | null) =>
      setEnhanceOverrideState(on === null ? null : { family: enhanceFamilyKey, on }),
    [enhanceFamilyKey],
  );
  const enhanceOn = resolveEnhancePrompt({
    capability: upsamplerCapability,
    stored: enhanceChoices[enhanceFamilyKey],
    override: enhanceOverride,
  });
  const changeEnhancePrompt = useCallback(
    (on: boolean) => {
      setEnhanceOverride(null);
      setEnhanceChoices((all) =>
        withEnhanceChoice(all, enhanceFamilyKey, on, upsamplerCapability.defaultOn),
      );
    },
    [enhanceFamilyKey, setEnhanceOverride, upsamplerCapability.defaultOn],
  );
  useEffect(() => {
    if (!controlnetSupported) {
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
  }, [controlnetSupported, status?.family]);

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
  const selectedThumb = selected ? thumbById[selected.id] : undefined;
  // The in-flight run's live preview, when the backend streams one and the toggle is on.
  const livePreviewSrc =
    busy === "generating" && livePreview ? (genStep?.preview ?? undefined) : undefined;
  const [viewerId, setViewerId] = useState<string | null>(null);
  const viewerImage = viewerId ? (images.find((image) => image.id === viewerId) ?? null) : null;
  const viewerSrc = viewerImage ? srcById[viewerImage.id] : undefined;
  if (viewerId && (!active || !viewerImage)) setViewerId(null);
  // Pruning below must not revoke the image on screen.
  const viewerIdRef = useRef<string | null>(null);
  useEffect(() => {
    viewerIdRef.current = viewerId;
  }, [viewerId]);
  const openViewer = () => selected && selectedSrc && setViewerId(selected.id);
  const navigateToChat = useNavigate();
  const revealLabel = useRevealLabel();

  // Fetch (once) the object URL for a record's PNG; cached across remounts.
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
      // Evict the coldest off-screen images this one pushed over budget; on-screen and open tiles are protected.
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

  // Fetch a thumbnail unless the tile already has a cached image or pending request.
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
        page.images.forEach((image) => void ensureThumb(image));
      }
    } catch {
      // Best-effort: a failed gallery load should not block the page.
    }
  }, [ensureThumb]);

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
        page.images.forEach((image) => void ensureThumb(image));
      }
    } catch {
      // transient; the user can scroll again to retry
    } finally {
      loadingMore.current = false;
    }
  }, [ensureThumb]);

  // Load thumbnails as tiles approach the visible strip.
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
          galleryCache.thumbById.touch(id);
          const image = images.find((i) => i.id === id);
          if (image) void ensureThumb(image);
        }
      },
      // rootMargin applies to the ROOT box only, so the root must be the scrolling strip; the
      // sideways margin fetches a few tiles early.
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
      galleryCache.srcById.delete(id); // revokes the URL with the entry
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
          collected.forEach((image) => void ensureThumb(image));
        }
        return;
      }
    },
    [ensureThumb],
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
  const { isFavorite, toggleFavorite } = useLibraryFavorites();
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

  // Drag-to-reorder: applied optimistically, then the server's record (key and pin) is adopted.
  const handleMove = useCallback(
    async (id: string, afterId: string | null) => {
      const next = moveGalleryItem(galleryCache.images, id, afterId);
      if (next === galleryCache.images) return;
      const guessedPinned = Boolean(next.find((i) => i.id === id)?.pinned);
      // Takes a pin token too: a pin clicked after this drop must not be undone by its response.
      const attempt = (pinSeq.current += 1);
      pinAttempt.current.set(id, attempt);
      stripEpoch.current += 1;
      galleryCache.images = next;
      setImages(next);
      try {
        // Shares the pin queue, since both rewrite the order.
        const record = await serializeById("image-pin", () => moveGalleryImage(id, afterId));
        if (pinAttempt.current.get(id) !== attempt) return;
        pinAttempt.current.delete(id);
        setImages((prev) => {
          const patched = prev.map((i) =>
            i.id === id ? { ...i, pinned: record.pinned, order_at: record.order_at } : i,
          );
          // Re-sort only if the local pin guess was wrong.
          const out =
            Boolean(record.pinned) === guessedPinned ? patched : sortGalleryItems(patched);
          galleryCache.images = out;
          return out;
        });
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to move image");
        // Restore the server's order.
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
    (image: GalleryImage) => downloadImage(image, "png", srcById[image.id]),
    [srcById],
  );
  const stripReorder = useStripReorder((id, afterId) => void handleMove(id, afterId));
  const imagesRef = useRef(images);
  imagesRef.current = images;
  useGalleryImageDropSource(
    useCallback(async (id: string) => {
      const image = imagesRef.current.find((item) => item.id === id);
      return image ? fetchGalleryBlob(image.url) : null;
    }, []),
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

  // `via`: "import" names what was not applied in its own toast; "vary" restores silently for the Vary run.
  const restoreSettings = useCallback((image: GalleryImage, exactCaption = false, via: "restore" | "import" | "vary" = "restore") => {
    const viaConnection = image.model_kind === "connection";
    // Negative prompt only applies when guidance>0 (the base pass's, or Hires CFG's for the refine pass); do not
    // restore a hidden value.
    const restoredNegative =
      image.guidance > 0 || hiresRecordUsedNegative(image) ? (image.negative_prompt ?? "") : "";
    setNegativePrompt(restoredNegative);
    if (restoredNegative) setNegativeOpen(true);
    if (!viaConnection) {
      setSteps(image.steps);
      setGuidance(image.guidance);
    }
    const restoredChoice = {
      sampler: image.sampler ?? DEFAULT_SAMPLER,
      scheduleType: image.schedule_type ?? AUTOMATIC_SCHEDULE,
    };
    if (samplerOptions.length > 0) changeSamplerChoice(restoredChoice);
    else setRestoredSampler(restoredChoice);
    setSdChoice(sdControlsFromRecipe(image));
    // Restore from the BASE batch seed, not this image's derived seed, or a replay with batch_size
    // advances it again.
    if (!viaConnection) setSeed(String(image.batch_seed ?? image.seed));
    // A Hires fix image records its output size; the Width / Height controls take the base pass.
    const base = hiresBaseSize(image) ?? { width: image.width, height: image.height };
    const restored = restorableSize(base.width, base.height, image.workflow, sizeLimits);
    setHires(resolveRestoredHires(hiresFromRecipe(image), status?.hires));
    setUpscaleModel(image.upscale_model ?? null);
    setWidth(restored.width);
    setHeight(restored.height);
    // The batch shared one base seed, so a batch_index>0 image only reproduces by replaying the whole batch.
    setBatchSize(image.batch_size ?? 1);
    setShiftValue(shiftFromRecipe(image));
    const m = matchAspect(restored.width, restored.height);
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
    const detail = restoredDetailer(image);
    setDetailerOn(detail.enabled);
    if (detail.enabled) setDetailerOptions({ ...readDetailerOptions(), ...detail.options });
    const maskOptions = restoredInpaintMaskOptions(image);
    if (maskOptions.maskBlur !== undefined) setMaskBlur(maskOptions.maskBlur);
    if (maskOptions.onlyMasked !== undefined) setInpaintOnlyMasked(maskOptions.onlyMasked);
    if (maskOptions.padding !== undefined) setInpaintPadding(maskOptions.padding);
    // Conditioning images are not persisted, so every upload is cleared. Edit and Reference reopen
    // their own workflow, so Generate stays blocked until the inputs are supplied again instead of
    // replaying as text-to-image; the others return to Create.
    const reopened: WorkflowId =
      image.workflow === "edit" ? "edit" : image.workflow === "reference" ? "reference" : "create";
    setWorkflow(reopened);
    // An enhanced run restores the user's prompt with Enhance prompt on, or (exactCaption) the caption it ran with
    // the toggle off. Records without an outcome leave the toggle alone.
    const enhanceRestore = enhanceRestoreFor(image, exactCaption);
    setPromptFor(reopened, enhanceRestore.prompt);
    if (enhanceRestore.enhance !== null) setEnhanceOverride(enhanceRestore.enhance);
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
    // The control image is not persisted, so clear any stale ControlNet selection.
    setControlnetId("");
    setControlImage(null);
    // The Recipe popover goes on showing the recorded size, so a restore that had to move it says
    // so rather than leaving the two silently disagreeing.
    const rescaled =
      restored.width !== base.width || restored.height !== base.height
        ? { description: `Size scaled to ${restored.width} × ${restored.height} to fit the ${MIN_DIM}-${sizeLimits.maxSide} range.` }
        : undefined;
    // Say so, rather than letting a conditioned image restore as a plain Create that generates something unrelated.
    const conditioned =
      restoreInputsNote(image) ?? CONDITIONED_WORKFLOW_INPUTS[image.workflow ?? ""];
    if (via === "vary") return;
    if (conditioned) {
      toast.success(`Settings restored. Add ${conditioned} again to reproduce this image.`, rescaled);
    } else {
      toast.success("Settings restored to inputs", rescaled);
    }
    // sd.cpp has no off switch for (word:1.2) syntax, so a literal-prompt run would replay weighted.
    const literal =
      via === "restore"
        ? literalPromptNotApplied(image, Boolean(status?.loaded && isNativeEngineStatus(status)), sdCaps)
        : null;
    // Generate omits (or clamps) the SD controls this load lacks, so name them rather than claim a full restore.
    const notApplied = [
      ...(literal ? [literal] : []),
      ...(via === "restore" && status?.loaded ? sdControlsNotRestored(image, sdCaps) : []),
      // Restored into the hidden control, which Generate omits: say so, as an import does.
      ...(via === "restore" ? detailerNotApplied(detail.enabled, detailerReady) : []),
    ];
    if (notApplied.length > 0) {
      toast.warning("Some settings in this image were not applied", {
        description: `Not applied: ${notApplied.join(", ")}. This model or build does not support them.`,
      });
    }
  }, [changeSamplerChoice, samplerOptions.length, sdCaps, setDetailerOptions, setEnhanceOverride, setPromptFor, setShiftValue, setWorkflow, sizeLimits, status, detailerReady]);

  // Like dropping an image on ComfyUI: Unsloth's recipe first, else AUTOMATIC1111 / ComfyUI metadata.
  const importFileRef = useRef<HTMLInputElement>(null);
  // Notified once per loaded model while Only masked is on but that model's pipeline cannot crop, so
  // the hidden switch never drops the setting silently, whatever turned it on or switched the model.
  const onlyMaskedNotified = useRef<string | null>(null);
  useEffect(() => {
    const key =
      status?.loaded && inpaintOnlyMasked && status.supports_inpaint_only_masked !== true
        ? onlyMaskedKeyOf(status)
        : null;
    if (key === null) {
      onlyMaskedNotified.current = null;
      return;
    }
    if (onlyMaskedNotified.current === key) return;
    onlyMaskedNotified.current = key;
    const missed = inpaintOptionsNotApplied(
      { inpaint_only_masked: true, inpaint_padding: inpaintPadding },
      false,
    );
    toast.warning("Some settings were not applied", {
      description: `Not applied: ${missed.join(", ")}. This model or build does not support them.`,
    });
  }, [status, inpaintOnlyMasked, inpaintPadding]);
  // A later import supersedes one still waiting on the LoRA lookup.
  const importSeq = useRef(0);
  // The status as of now, for an import resuming after its LoRA lookup.
  const statusRef = useRef(status);
  statusRef.current = status;
  const applyImportedRecipe = useCallback(
    async (recipe: ImportedRecipe): Promise<boolean> => {
      const seq = ++importSeq.current;
      const modelBefore = loadedModelKey(status);
      let tagged = await matchLoraTagsForImport(
        recipe.lora_tags,
        // Only a finished fetch for the loaded family: right after a load or family switch it is empty or stale.
        loraCapable && loraCatalogFamily === (status?.family ?? null),
        availableLoras,
        // A LoRA is family-specific, so a loaded model narrows the lookup like the picker does.
        () => listDiffusionLoras(status?.loaded ? (status.family ?? undefined) : undefined),
      );
      if (seq !== importSeq.current) return false;
      // A load or family switch landed during the lookup: its matches are for the old model.
      if (loadedModelKey(statusRef.current) !== modelBefore) {
        tagged = {
          loras: [],
          unmatched: [],
          notes: loraTagNames(recipe.lora_tags).map((n) => `LoRA ${n} (model changed during import)`),
        };
      }
      // TI hashes: the prompt names each embedding, so on a model that takes them an installed one applies.
      const tiNames = Array.isArray(recipe.textual_inversions)
        ? recipe.textual_inversions.filter((x): x is string => typeof x === "string")
        : [];
      let tiMatch: { installed: string[]; missing: string[] } | null = null;
      if (tiNames.length > 0 && embeddingsEnabled && status?.family) {
        const triggers = await listDiffusionEmbeddings(status.family).then(
          (r) => r.embeddings.map((e) => e.trigger),
          () => null,
        );
        if (seq !== importSeq.current) return false;
        // A list fetched for a model that is no longer loaded says nothing: the names are reported plainly.
        if (triggers && loadedModelKey(statusRef.current) === modelBefore)
          tiMatch = splitInstalledEmbeddings(tiNames, triggers);
      }
      const merged =
        tagged.loras.length > 0
          ? { ...recipe, loras: [...(Array.isArray(recipe.loras) ? recipe.loras : []), ...tagged.loras] }
          : recipe;
      const settings = recipeSettings(merged, { width, height, steps, guidance });
      const localized = settings.localized_edit;
      restoreSettings(
        {
          ...settings,
          id: "",
          url: "",
          seed: settings.seed ?? 0,
          batch_index: 0,
          created_at: 0,
          localized_edit:
            localized === "annotate" || localized === "paint" || localized === "mask" ? localized : null,
        },
        false,
        "import",
      );
      if (settings.seed === null && settings.model_kind !== "connection") setSeed("");
      // A variation seed this model or workflow cannot apply is named below, not left armed (the backend refuses it).
      if (!variationApplies(merged, sdCaps))
        setSdChoice((c) => ({ ...c, variationSeed: "", variationStrength: 0 }));
      const loaded = status?.display_repo_id ?? status?.repo_id ?? null;
      const quant = typeof recipe.gguf_filename === "string" ? recipe.gguf_filename : null;
      if (settings.model && settings.model !== loaded) {
        toast(`Made with ${settings.model}${quant ? ` (${quant})` : ""}`, {
          // Loading a model applies its own default steps and guidance.
          description: "Load that model, then load settings from this image again to reproduce it.",
        });
      } else if (settings.model && status?.loaded) {
        // Same model, another build: null on either side means dense / no quant.
        const differs = (["gguf_filename", "transformer_quant", "text_encoder_quant", "transformer_cache", "attention_backend"] as const).filter(
          (key) =>
            Object.hasOwn(recipe, key) &&
            status[key] !== undefined &&
            (recipe[key] ?? null) !== (status[key] ?? null),
        );
        if (differs.length > 0) {
          toast(
            `Made with ${differs.map((key) => (key === "gguf_filename" ? String(recipe[key] ?? "the dense model") : `${key.replace(/_/g, " ")} ${String(recipe[key] ?? "off")}`)).join(", ")}`,
            { description: "Reload the model with that build to reproduce this image exactly." },
          );
        }
      }
      const notApplied = [
        ...importNotApplied(
          recipe,
          status?.loaded ? status.family : null,
          Boolean(status?.loaded && isNativeEngineStatus(status)),
          sdCaps,
        ),
        ...tagged.unmatched.map((name) => `LoRA ${name} (not installed)`),
        ...lorasOverLimit(merged).map((id) => `LoRA ${id} (over the ${MAX_LORAS} adapter limit)`),
        ...tagged.notes,
        ...shiftNotApplied(recipe, shiftControl),
        ...hiresNotApplied(hiresFromRecipe(settings), status?.loaded ? status.hires : null),
        ...unknownRecipeKeys(recipe),
      ];
      // Restored into the hidden control when this model cannot inpaint or SAM 3 is missing: say so.
      notApplied.unshift(...detailerNotApplied(Boolean(settings.detailer), detailerReady));
      if (tiMatch) {
        const { installed, missing } = tiMatch;
        const resolved = new Map<string, string | null>([
          ...installed.map((n): [string, null] => [textualInversionLabel(n), null]),
          ...missing.map((n): [string, string] => [textualInversionLabel(n), `${textualInversionLabel(n)} (not installed)`]),
        ]);
        const kept = notApplied.flatMap((item) => {
          if (!resolved.has(item)) return [item];
          const label = resolved.get(item);
          return label ? [label] : [];
        });
        notApplied.splice(0, notApplied.length, ...kept);
      }
      const baked = Array.isArray(recipe.baked_loras)
        ? recipe.baked_loras.filter((x): x is string => typeof x === "string")
        : [];
      if (baked.length > 0) notApplied.unshift(`baked LoRAs ${baked.join(", ")} (load-time)`);
      if (typeof recipe.controlnet === "string" && recipe.controlnet) {
        notApplied.unshift(`ControlNet ${recipe.controlnet}`);
      }
      if (status?.loaded && status.supports_lora === false && settings.loras.length > 0) {
        notApplied.unshift("LoRAs");
      }
      if (status?.loaded) {
        const missed = inpaintOptionsNotApplied(settings, status.supports_inpaint_only_masked === true);
        notApplied.unshift(...missed);
        // Named in this toast, so the Only masked effect does not repeat it for this model.
        if (missed.length > 0) onlyMaskedNotified.current = onlyMaskedKeyOf(status);
      }
      if (status?.supports_negative_prompt === false && settings.negative_prompt && settings.guidance > 0) {
        notApplied.unshift("negative prompt");
      }
      if (status?.loaded && samplerOptions.length === 0) {
        if (settings.schedule_type && settings.schedule_type !== AUTOMATIC_SCHEDULE) {
          notApplied.unshift(`schedule ${labelOf(SCHEDULE_LABELS, settings.schedule_type)}`);
        }
        if (settings.sampler && settings.sampler !== DEFAULT_SAMPLER) {
          notApplied.unshift(`sampler ${labelOf(SAMPLER_LABELS, settings.sampler)}`);
        }
      } else if (samplerOptions.length > 0) {
        const offered = samplerOptions.find((o) => o.id === (settings.sampler ?? DEFAULT_SAMPLER));
        if (!offered) {
          notApplied.unshift(`sampler ${labelOf(SAMPLER_LABELS, settings.sampler ?? DEFAULT_SAMPLER)}`);
        } else if (
          settings.schedule_type &&
          !offered.schedule_types.includes(settings.schedule_type)
        ) {
          notApplied.unshift(`schedule ${labelOf(SCHEDULE_LABELS, settings.schedule_type)}`);
        }
      }
      if (notApplied.length > 0 || newerRecipeSchema(recipe)) {
        toast.warning("Some settings in this image were not applied", {
          description:
            (notApplied.length > 0 ? `Not applied: ${notApplied.join(", ")}. ` : "") +
            (newerRecipeSchema(recipe)
              ? "It was made with a newer Unsloth; update to use every setting."
              : "This model or build does not support them."),
        });
      }
      return true;
    },
    [restoreSettings, width, height, steps, guidance, status, samplerOptions, availableLoras, loraCapable, loraCatalogFamily, sdCaps, embeddingsEnabled, shiftControl, detailerReady],
  );
  const importSettingsFromFile = useCallback(
    async (file: File | Uint8Array) => {
      let recipe = null;
      try {
        const bytes = file instanceof Uint8Array ? file : new Uint8Array(await file.arrayBuffer());
        recipe = importedRecipe(readEmbeddedRecipe(bytes));
      } catch {
        recipe = null;
      }
      if (!recipe) {
        toast.error("No generation settings found in this image", {
          description: "Images saved by Unsloth, ComfyUI, or tools that write a parameters block can be loaded.",
        });
        return;
      }
      await applyImportedRecipe(recipe);
    },
    [applyImportedRecipe],
  );
  const importSettingsFromText = useCallback(
    async (text: string): Promise<boolean> => {
      const recipe = pastedParameters(text);
      if (!recipe) {
        toast.error("No generation parameters found in that text", {
          description: "Paste the prompt, the Negative prompt: line and the Steps: ... line as AUTOMATIC1111, Forge or Civitai copy them.",
        });
        return false;
      }
      return applyImportedRecipe(recipe);
    },
    [applyImportedRecipe],
  );
  // Desktop: Tauri delivers OS drops as paths and suppresses the DOM drop event.
  const importNativePath = useCallback(
    async (path: string | undefined) => {
      if (!path) return;
      if (!["png", "jpg", "jpeg", "webp"].includes(path.split(".").pop()?.toLowerCase() ?? "")) {
        toast.error("Drop a PNG, JPEG or WebP image to load its settings");
        return;
      }
      try {
        const intent = await registerNativeAttachmentPath(path);
        const file = await readNativeAttachmentFile(intent.path.token);
        await importSettingsFromFile(Uint8Array.from(atob(file.base64), (c) => c.charCodeAt(0)));
      } catch (error) {
        toast.error("Could not read the image", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [importSettingsFromFile],
  );
  const previewDropRef = useNativeDropTarget({ onDrop: (paths) => void importNativePath(paths[0]) });
  const onImportDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    if (event.dataTransfer.types.includes("Files")) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    }
  }, []);
  const onImportDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!event.dataTransfer.types.includes("Files")) return;
      // Claimed even for a file this cannot read, or the browser navigates away to it.
      event.preventDefault();
      const file = Array.from(event.dataTransfer.files).find((f) => f.type.startsWith("image/"));
      if (file) {
        void importSettingsFromFile(file);
      } else {
        toast.error("Drop a PNG, JPEG or WebP image to load its settings");
      }
    },
    [importSettingsFromFile],
  );

  // A locked ratio keeps the paired dimension in step; "custom" frees both, Flip swaps W/H. ratioHW is h/w for [a,b].
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

  // A status read started before an eject can answer after the one that followed it, and this
  // page has no periodic poll to correct it, so only the newest ticket may write.
  const statusTicket = useRef(0);
  const setStatusIfNewest = useCallback(
    (ticket: number, next: DiffusionStatus) => {
      if (ticket !== statusTicket.current) return;
      setStatus(next);
      // Any accepted status, not only the poll's ready branch: a load that finished while the page was closed.
      setEngineChoices((all) => rememberLoadedEngine(all, next));
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

  // Collapse the body-ported popovers when leaving the tab: the open flag stays set, so
  // returning would pop them back open.
  useEffect(() => {
    if (active) return;
    setSelectorOpen(false);
    setAspectOpen(false);
  }, [active]);

  const offerLoadAnywayFor = useCallback(
    (failure: unknown, code?: string | null): boolean => {
      const attempt = lastLoadAttempt.current;
      if (!attempt || !shouldOfferLoadAnyway({ code, failure, overrideSent: attempt.overrideSent })) return false;
      offerLoadAnyway(failureMessage(failure) || "Failed to load model", attempt.retry);
      return true;
    },
    [offerLoadAnyway],
  );

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
        if (loaded.loaded && loaded.model_kind === "gguf") {
          const key = engineModelKey(loaded.repo_id, loaded.gguf_filename);
          setEngineChoices((all) => withEngineChoice(all, key, loadedEngineChoice(loaded)));
        }
        const evicted = pendingEvicted.current;
        pendingEvicted.current = [];
        toast.success("Model loaded", evictedToastOptions(evicted));
        if (lastLoad.current && matchesRememberedModel(lastLoad.current, loaded)) {
          const remembered = withEngagedFamily(lastLoad.current, loaded);
          rememberImageModel(remembered);
          setRememberedModel(remembered);
        }
        setBusy(null);
        // A fallback-recipe pick takes the loaded family recipe on an untouched form (else SDXL runs 9 steps, CFG 0).
        const loadedRecipe = loadedRecipeFor(
          pickDefaults.current,
          residentDefaultsKey(
            loaded.repo_id ?? "",
            loaded.base_repo,
            loaded.resolved?.family_override,
            loaded.gguf_filename,
          ),
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
        // Status owns the recipe from here, so the pick's claim on Default expires with it.
        setPendingModelDefaults(null);
        // lastLoad.current already holds the now-resident pick, so drop its revert too.
        lastLoadRevert.current = null;
        return;
      }
      if (p.phase === "error") {
        pendingRecalledGeneration.current = null;
        pendingEvicted.current = [];
        dismissLoadToast();
        if (!offerLoadAnywayFor(p.error, p.error_code)) reportLoadFailure(p.error, "Failed to load model");
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
        // And for the Engine select: a worker that failed before publishing leaves the resident provenance, so the
        // seed would not move and the next Reapply would replay the failed pick.
        setEngineSeed("");
        // A failed load may have freed a previously-loaded model, so resync to the real backend state.
        void refreshStatus();
        return;
      }
      if (p.phase === null) {
        pendingRecalledGeneration.current = null;
        pendingEvicted.current = [];
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
  }, [dismissLoadToast, refreshStatus, cancelLoadFromToast, offerLoadAnywayFor]);

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
        if (p.active) runTarget.current = p.target ?? undefined;
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
          runTarget.current = g.target ?? undefined;
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
    const seedKey = `${repoId}\0${residentDefaults}\0${residentSteps}\0${residentGuidance}`;
    if (seededResident.current === seedKey) return;
    seededResident.current = seedKey;
    // Wire Reapply to the resident model too: a pipeline by repo id, a GGUF by repo id + file (so an
    // Engine change on it can take effect). Set before the recipe decision below, a separate question.
    const resident = residentReloadTarget({
      loaded: true,
      repo_id: repoId,
      model_kind: status?.model_kind,
      gguf_filename: status?.gguf_filename,
      display_repo_id: status?.display_repo_id,
      base_repo: status?.base_repo,
      component_files: status?.component_files,
    });
    if (resident) lastLoad.current = resident;
    // A stored recipe is the user's own choice, so it outranks the resident model's defaults on the first seed.
    if (!residentSeeded.current) {
      residentSeeded.current = true;
      if (imagePresets.storedRecipe) return;
    }
    const d = { steps: residentSteps, guidance: residentGuidance };
    setPendingModelDefaults(null);
    setSteps(d.steps);
    setGuidance(d.guidance);
    // The canvas is part of the resident model's defaults, not a constant: a quantised build shrinks
    // the weights and leaves the activations alone, so on Qwen-Image-2.1 the canvas is what decides
    // whether the load fits. Read from the ENGAGED build, so a declined quant request keeps 1024.
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
    status?.gguf_filename,
    status?.component_files,
    status?.transformer_quant,
    status?.resolved?.transformer_quant?.source,
  ]);

  const loadedEngineKey =
    status?.loaded && status.model_kind === "gguf" ? engineModelKey(status.repo_id, status.gguf_filename) : "";
  const loadedEngine = loadedEngineChoice(status);
  // Reseed the select when the loaded build changes, during render rather than in an effect.
  if (engineSeed !== `${loadedEngineKey}|${loadedEngine}`) {
    setEngineSeed(`${loadedEngineKey}|${loadedEngine}`);
    setEngineChoice(loadedEngine);
  }
  const engineForced = status?.engine_forced ?? null;
  const engineSelection = useRef({ key: loadedEngineKey, choice: engineChoice, stored: engineChoices, forced: engineForced });
  useEffect(() => {
    engineSelection.current = { key: loadedEngineKey, choice: engineChoice, stored: engineChoices, forced: engineForced };
  }, [loadedEngineKey, engineChoice, engineChoices, engineForced]);
  // A load's engine: an explicit one it was handed, else the Engine select for the loaded checkpoint, else
  // the choice that checkpoint last loaded with. GGUF only; Auto omits the field.
  const engineFor = useCallback(
    (repoId: string, opts: ImageLoadOptions, advanced: LoadAdvanced) => {
      if (!sendsEngine(opts.kind, opts.filename)) return undefined;
      if (advanced.engine) return engineRequestField(advanced.engine);
      const { key, choice, stored } = engineSelection.current;
      const target = engineModelKey(repoId, opts.filename);
      return engineRequestField(target && target === key ? choice : stored[target]);
    },
    [],
  );

  // Reseed the Advanced selects from the LOADED build, so a declined request snaps to what
  // engaged and Precision never advertises a scheme the model is not running. Keyed on the
  // LOAD-TIME half of the record: the backend rewrites the speed/attention/cache entries at
  // GENERATION time, and the whole record threw away a Precision picked but not yet loaded.
  const resolvedKey = status?.loaded ? resolvedSeedKey(status.resolved) : null;
  useEffect(() => {
    const record = status?.loaded ? status.resolved : null;
    if (!record) return;
    const family = resolvedFamilyOverrideSelection(record.family_override);
    if (family) setFamilyOverride(family);
    const quant = resolvedSelectValue(record.transformer_quant, (v) =>
      // The engaged value spells "no quant" as "off"; the select's option for it is "none".
      (["auto", "none", "int8", "fp8", "nvfp4", "mxfp8"] as const).find(
        (o) => o === v || (o === "none" && v === "off"),
      ) ?? null,
    );
    if (quant) setTransformerQuant(quant);
    const encoder = resolvedSelectValue(record.text_encoder_quant, (v) =>
      // The engaged value spells the dense encoder "off"; the select's option for it is "none".
      // It maps to Dense, NOT to Default: an unset request is already caught upstream by
      // `source === "auto"`, so reaching here with "off" means dense was pinned or a scheme was
      // declined. Folding it into Default would snap a pinned Dense back to Default, and the next
      // reapply would omit the field and silently take the family's scheme instead.
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
      // The engaged value uses the dispatcher's own name; map it back to the option.
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

  // One snapshot of every Advanced control a load sends, so a staged pick can pin the values it planned against.
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
        // Dropped when the chosen card is gone, so a stale pick loads automatically instead of 400ing.
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
    // Resolves true when the background load STARTED (callers may revert optimistic picker state on false).
    async (
      repoId: string,
      opts: ImageLoadOptions,
      // The Advanced values this load must use when pinned earlier: a staged download plans its file
      // set at pick time and loads minutes later, so live state could outrun the staged files.
      pinned?: LoadAdvanced,
      // Reuse the pick toast when loading starts.
      pickToastId?: string,
    ): Promise<boolean> => {
      setExternalModelId(null);
      // Cancel any prior poll loop so two cannot run at once.
      if (pollTimer.current) clearTimeout(pollTimer.current);
      // Read BEFORE the start request goes out: a Cancel pressed while it is in flight sends an
      // unload that can reach the backend first, find no load registered, and stop nothing.
      const startSeq = cancelSeq.current;
      const startLoad = ++loadSeq.current;
      // An older pick's Load anyway prompt would retry a model this load supersedes.
      dismissLoadAnyway();
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
      const handedOver = pickToast.take(pickToastId);
      loadToastId.current = toast(null, loadToastArgs(IDLE_PROGRESS, handedOver, cancelLoadFromToast));
      // Remember what was loaded so "Reapply" can reload it. Snapshot the prior target first: a load
      // that fails to START leaves the previous model resident.
      const prevLastLoad = lastLoad.current;
      // A torchao int8/fp8 transformer takes adapters only at LOAD time and /images/generate then
      // rejects a new set, so a reload must keep the selection. Ignored by bf16 / bnb-4bit.
      const picked = pinned ?? currentLoadAdvanced(repoId);
      // A native pick drops what the native engine refuses (explicit precision, separate component files).
      const advanced = nativeLoadAdvanced(
        picked,
        engineFor(repoId, opts, picked),
        engineSelection.current.forced,
        sendsEngine(opts.kind, opts.filename),
      );
      const bakeLoras = advanced.loras ?? [];
      // Whether THIS load carries the selection into the build, so a quantized load that did not can drop it.
      bakedLorasOnLoad.current = bakeLoras.length > 0;
      const componentFiles = componentFileFields(opts.kind, advanced.text_encoder_file, advanced.vae_file);
      // A retired FLUX.3 full copy (named, or a downloaded snapshot of one) loads the base it copies, as the backend
      // redirects it, so the load, Reapply and the remembered model all name the id the status will report.
      const redirected = redirectRetiredImagePick({
        repoId,
        ...opts,
        textEncoderFiles: componentFiles.text_encoder_file,
        vaeFile: componentFiles.vae_file,
      });
      // A redirect loads another checkpoint, whose base the backend infers.
      const baseRepo = redirected.repoId === repoId ? opts.baseRepo : undefined;
      repoId = redirected.repoId;
      opts = { kind: redirected.kind, filename: redirected.filename, displayRepoId: redirected.displayRepoId, baseRepo };
      lastLoad.current = {
        repoId,
        kind: opts.kind,
        filename: opts.filename,
        displayRepoId: opts.displayRepoId,
        baseRepo: opts.baseRepo,
        textEncoderFiles: componentFiles.text_encoder_file,
        vaeFile: componentFiles.vae_file,
      };
      setCanReapply(true);
      // Carry the prior target so the async poll can restore it if the background load fails after starting.
      lastLoadRevert.current = { prev: prevLastLoad };
      const overrideSent = advanced.allow_oversized_load === true || readAllowOversizedLoad();
      const retryPick = pickSeq.current;
      const retryOversized = retryWhileCurrent(
        () => retryPick === pickSeq.current && startLoad === loadSeq.current,
        () => void retryLoadRef.current?.(repoId, opts, { ...advanced, allow_oversized_load: true }),
      );
      lastLoadAttempt.current = { overrideSent, retry: retryOversized };
      // The POST may first save the chat model's context; show that on this load's toast meanwhile.
      const savePoll = watchSavingPhase(getDiffusionLoadProgress, () => {
        if (startSeq === cancelSeq.current && startLoad === loadSeq.current && loadToastId.current != null) {
          toast(null, loadToastArgs(SAVING_PROGRESS, loadToastId.current, cancelLoadFromToast));
        }
      });
      try {
        // Returns immediately; the load runs in the background and we poll. The backend infers the
        // family and base repo from the id, and the saved HF token covers gated bases.
        const startRequest = loadDiffusionModel({
          model_path: repoId,
          display_repo_id: opts.displayRepoId,
          model_kind: opts.kind,
          gguf_filename: opts.filename,
          base_repo: opts.baseRepo,
          hf_token: hfApiToken(getHfToken()),
          engine: engineFor(repoId, opts, advanced),
          cpu_offload: advanced.cpu_offload,
          speed_mode: advanced.speed_mode,
          // Do not carry a saved precision into a known incompatible artifact.
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
          allow_oversized_load: advanced.allow_oversized_load,
          ...componentFiles,
        });
        const answer = await startRequest;
        pendingEvicted.current = evictedSlotsFrom(answer);
        resyncEvicted(pendingEvicted.current, "image");
      } catch (err) {
        pendingEvicted.current = [];
        lastLoad.current = prevLastLoad;
        setCanReapply(prevLastLoad != null);
        lastLoadRevert.current = null;
        // A refused load keeps the resident model and its provenance, so the seed would not move: reseed the
        // Engine select from the resident build rather than replay the refused pick on the next Reapply.
        setEngineSeed("");
        dismissLoadToast();
        if (!offerLoadAnywayFor(err)) reportLoadFailure(err, "Failed to start load");
        setBusy(null);
        void refreshStatus();
        return settle(false);
      } finally {
        savePoll.stop();
      }
      if (startSeq !== cancelSeq.current) {
        pendingEvicted.current = [];
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
    [pollLoadProgress, refreshStatus, dismissLoadToast, currentLoadAdvanced, cancelLoadFromToast, pickToast, engineFor, offerLoadAnywayFor, dismissLoadAnyway],
  );

  // Set or clear the Transform/Inpaint source image; always drop the painted mask, which is
  // sized to the previous source.
  const handleInitChange = useCallback((dataUrl: string | null) => {
    setInitImage(dataUrl);
    setMaskImage(null);
    setMaskResetKey((k) => k + 1);
  }, []);
  const applyEditSource = useCallback(
    (dataUrl: string) => {
      // FLUX.2 klein and FLUX.3 edit through Reference; without this fallback the source lands hidden on Create.
      const target: WorkflowId | null = isWorkflowEnabled("edit", supported)
        ? "edit"
        : isWorkflowEnabled("transform", supported)
          ? "transform"
          : isWorkflowEnabled("reference", supported)
            ? "reference"
            : null;
      if (target) setWorkflow(target);
      handleInitChange(dataUrl);
      if (target === "edit") {
        toast.success("Image added to Edit. Describe the change and hit Generate.");
      } else if (target === "transform") {
        toast.success("Image added to Transform", {
          description: "The loaded model cannot follow edit instructions, so it redraws the image from your prompt.",
        });
      } else if (target === "reference") {
        toast.success("Image added to Reference. Describe the change and hit Generate.");
      } else {
        toast.info("Image set as the source", {
          description: "The loaded model cannot edit images. Load an edit model, then pick Edit.",
        });
      }
    },
    [supported, setWorkflow, handleInitChange],
  );
  // Only the newest click applies: the gallery fetch can outlive a second one.
  const editRequest = useRef(0);
  const editGalleryImage = useCallback(
    async (image: GalleryImage) => {
      const claimed = ++editRequest.current;
      let dataUrl: string;
      try {
        const blob = await fetchGalleryBlob(image.url);
        dataUrl = await fitImageDataUrl(await blobToDataUrl(blob), MAX_UPLOAD_IMAGE_SIDE);
      } catch (error) {
        if (claimed === editRequest.current) {
          toast.error("Could not open the image for editing", {
            description: error instanceof Error ? error.message : String(error),
          });
        }
        return;
      }
      if (claimed === editRequest.current) applyEditSource(dataUrl);
    },
    [applyEditSource],
  );
  const pendingLibraryEdit = useImageEditHandoffStore((s) => s.pending);
  // Held until the loaded model's workflows are known, or a later snap to Create drops the source.
  const capabilitiesKnown = external
    ? supported !== null
    : status !== null && (!status.loaded || supported !== null);
  useEffect(() => {
    if (!pendingLibraryEdit || !capabilitiesKnown) return;
    const dataUrl = useImageEditHandoffStore.getState().take();
    if (!dataUrl) return;
    editRequest.current += 1;
    applyEditSource(dataUrl);
  }, [pendingLibraryEdit, capabilitiesKnown, applyEditSource]);
  const editImageItem = (image: GalleryImage) => (
    <DropdownMenuItem onClick={() => void editGalleryImage(image)}>
      <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
      Edit image
    </DropdownMenuItem>
  );
  const { openCrop: openInpaintCrop, cropDialog: inpaintCropDialog } = useImageCrop(
    initImage,
    handleInitChange,
    "Source image",
  );

  // Downloads go through the Hub download manager like every other model, so the load finds a
  // warm cache. In a ref, so the callback is not a render dep.
  const pendingStagedLoad = useRef<{
    repoId: string;
    opts: ImageLoadOptions;
    // The Advanced values the plan was built from: staging does not set `busy`, so the user can
    // change precision or LoRAs while the download runs.
    advanced: LoadAdvanced;
    // The pick that staged it: a download outlives its pick, so it must not evict a newer one when it lands.
    token: number;
    toastId?: string;
  } | null>(null);
  const handleLoadRef = useRef(handleLoad);
  useEffect(() => {
    retryLoadRef.current = handleLoad;
  }, [handleLoad]);
  handleLoadRef.current = handleLoad;
  // Set when a staged download finished while this page was hidden: both diffusion pages stay
  // mounted and a load evicts whatever holds the GPU. The pick fires on return.
  const stagedLoadDeferred = useRef(false);
  // Both deferred paths run the load minutes after the pick was reported started, so both need the
  // same rollback: a deferred load can still be REFUSED, and staging polls nothing. `owned` is read
  // BEFORE the call, so a newer pick's label is left alone.
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
  // Each download-only selection keeps its complete plan until it finishes or is cancelled.
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
        // Loading waits for queued download-only plans.
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
      startQueuedDownload();
    },
  });
  usePickToastProgress(pickToast, stagedProgress);

  /** A staged file set, as the identity two picks of the same model share. */
  function planKey(entries: StagedDownloadEntry[]) {
    return JSON.stringify(
      entries.map((e) => [e.repoId, e.ggufFilename ?? "", [...e.files].sort()]).sort(),
    );
  }

  /** The file sets already queued, including the one downloading right now (the queue head). */
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

  // Ask for a cache-aware plan on every Hub pick: the picker knows only whether the checkpoint
  // is cached, and image models can need a separate text encoder/VAE repo. True = accepted.
  const requestDownloadPlan = useCallback(
    (
      repoId: string,
      opts: ImageLoadOptions,
      picked: LoadAdvanced,
    ) => {
      // Normalised exactly as handleLoad does, so the plan describes the load that will run.
      const advanced = nativeLoadAdvanced(
        picked,
        engineFor(repoId, opts, picked),
        engineSelection.current.forced,
        sendsEngine(opts.kind, opts.filename),
      );
      return getDiffusionDownloadPlan({
        model_path: repoId,
        gguf_filename: opts.filename,
        base_repo: opts.baseRepo,
        model_kind: opts.kind,
        // The same token and Advanced values handleLoad sends, so the plan describes the load that
        // will run: without the token a gated base plans no companion, and without the controls it
        // stages shards the load never opens.
        hf_token: hfApiToken(getHfToken()),
        engine: engineFor(repoId, opts, advanced),
        cpu_offload: advanced.cpu_offload,
        speed_mode: advanced.speed_mode,
        // Keep the planned precision identical to the load request.
        transformer_quant: sendsTransformerQuant(opts.kind, repoId)
          ? advanced.transformer_quant
          : undefined,
        text_encoder_quant: advanced.text_encoder_quant,
        memory_mode: advanced.memory_mode,
        family_override: advanced.family_override,
        // The backend prefetch decision reads the adapter selection too: a baked LoRA always runs the
        // dense build path, and omitting it staged too little.
        loras: advanced.loras,
        // The plan route preflights precision and sizes the file set against the card the load will
        // use, so a selection the load carries has to reach the plan.
        gpu_ids: advanced.gpu_ids,
        // A Load anyway retry sends it on the plan too, or the plan refuses the pick again.
        allow_oversized_load: advanced.allow_oversized_load,
        ...componentFileFields(opts.kind, advanced.text_encoder_file, advanced.vae_file),
      });
    },
    [engineFor],
  );

  const loadOrStage = useCallback(
    async (
      repoId: string,
      opts: ImageLoadOptions,
      source: ModelSelectorChangeMeta["source"] = "hub",
      token?: number,
      familyOverrideRequired = false,
      downloadSnapshot?: LoadAdvanced,
      // A reload of the resident model loads whatever the picker's Load / Download only action says.
      forceLoad = false,
    ): Promise<boolean> => {
      const downloadOnly =
        downloadSnapshot !== undefined || (!forceLoad && modelSelectionAction === "download");
      // Read before the first await: a confirmed Load anyway retry sets it only around this call.
      const oversizedRetry = oversizedLoadOnce.current && !downloadOnly;
      // Staging never sets `busy`, so a second pick passes the guard while this plan is in flight, and
      // plans resolve in response order rather than pick order. Bumped before the non-hub return too, so
      // a local pick invalidates an in-flight hub plan. Never for a download-only pick: it supersedes
      // nothing, and retiring the sequence made the load it interrupted fail its own stale check below
      // and return silently, leaving the selected model neither staged nor loaded.
      const pick = downloadOnly ? pickSeq.current : ++pickSeq.current;
      if (!downloadOnly) dismissLoadAnyway();
      // The previous pick's staged intent dies with it: a pick that stages nothing never calls
      // stage(), so the queue keeps the older job and its onReady loads the abandoned model.
      // Only load intents have an owner; a download-only pick holds no token and answers true.
      const owns = () => token === undefined || downloadOnly || pickGuard.holds(token);
      // Resolved downloads retain their snapshot without replacing a newer load intent. Neither does
      // a download-only pick: it adds files to fetch, so retiring the staged load here left the model
      // that load was waiting for fully downloaded and never loaded.
      if (!downloadSnapshot && !downloadOnly) {
        pendingStagedLoad.current = null;
        pendingLoadEntries.current = null;
        stagedLoadDeferred.current = false;
        stagedQuantRevert.current = null;
        pickToast.dismissAll();
        if (!owns()) return true;
      }
      // ONE snapshot for the plan and the load it fires: the download runs for minutes without setting `busy`.
      let advanced = downloadSnapshot ?? currentLoadAdvanced(repoId, familyOverrideRequired);
      // A retired FLUX.3 full copy is planned, staged and loaded as the base it copies, so the staged
      // checkpoint is the one the load opens (the backend redirects the same picks).
      const component = componentFileFields(opts.kind, advanced.text_encoder_file, advanced.vae_file);
      const redirected = redirectRetiredImagePick({
        repoId,
        ...opts,
        textEncoderFiles: component.text_encoder_file,
        vaeFile: component.vae_file,
      });
      if (redirected.repoId !== repoId) {
        repoId = redirected.repoId;
        opts = { kind: redirected.kind, filename: redirected.filename, displayRepoId: redirected.displayRepoId };
        advanced = downloadSnapshot ?? currentLoadAdvanced(repoId, familyOverrideRequired);
        // A local snapshot of the copy is now a Hub load of the base: plan and stage it like one.
        source = "hub";
      }
      // The retry's override rides the same snapshot, so its plan and its load both carry it.
      if (oversizedRetry) advanced = { ...advanced, allow_oversized_load: true };
      // Resolve the engine into the snapshot, so the plan and the load it fires cannot read different selects.
      if (!advanced.engine && sendsEngine(opts.kind, opts.filename)) {
        advanced = { ...advanced, engine: engineFor(repoId, opts, advanced) ?? "auto" };
      }
      if (source !== "hub" && !downloadOnly) return handleLoadRef.current(repoId, opts, advanced);
      // Show feedback before the potentially slow Hub metadata request.
      const pickToastId = downloadOnly ? undefined : pickToast.show();
      // Read before the await: a pick made while the plan resolves replaces quantRevert, and this
      // job must not revert it. A download-only pick claims no slot at all: it never relabels the
      // selector, so there is nothing here it could own, and reverting what it found would restore
      // the PREVIOUS resident under a staged load that is still coming and then leave that load
      // nothing to commit.
      const ownRevert = downloadSnapshot || downloadOnly ? null : quantRevert.current;
      // Read inside the try, acted on outside it: refusing from in there would fall through to the
      // load if the refusal itself threw.
      let incompatible: string | null = null;
      let incompatibleCode: string | null = null;
      try {
        const plan = await requestDownloadPlan(opts.displayRepoId ?? repoId, opts, advanced);
        // Only load intents are superseded; accepted downloads keep their own plans.
        if (!downloadOnly && (pick !== pickSeq.current || !owns())) {
          pickToast.dismiss(pickToastId);
          return true;
        }
        if (downloadOnly && plan.plan_failed) {
          throw new Error("Required asset metadata is incomplete. Retry when it is available.");
        }
        incompatible = plan.incompatible_reason ?? null;
        incompatibleCode = plan.incompatible_code ?? null;
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
            // Picking the same model twice plans the same files twice. Queueing both downloaded
            // every byte twice, and the second start could come back "busy" against the first.
            if (queuedDownloadKeys().has(planKey(entries))) return true;
            downloadOnlyPlans.current.push(entries);
            // A late plan waits for the active file set, including a normal load's download.
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
        // No plan (older backend, metadata hiccup): fall back to the load's own download.
      }
      // Re-checked: a plan that REJECTED after a newer pick would otherwise reach the fallback load.
      if (!downloadOnly && (pick !== pickSeq.current || !owns())) {
        pickToast.dismiss(pickToastId);
        return true;
      }
      if (incompatible) {
        pickToast.dismiss(pickToastId);
        const overrideSent = advanced.allow_oversized_load === true || readAllowOversizedLoad();
        if (!downloadOnly && shouldOfferLoadAnyway({ code: incompatibleCode, overrideSent })) {
          const retrySource = source;
          offerLoadAnyway(incompatible, retryWhileCurrent(() => pick === pickSeq.current, () => {
            oversizedLoadOnce.current = true;
            try {
              void loadOrStageRef.current(repoId, opts, retrySource, undefined, familyOverrideRequired, undefined, true);
            } finally {
              oversizedLoadOnce.current = false;
            }
          }));
        } else {
          toast.error(incompatible);
        }
        return downloadOnly;
      }
      if (downloadOnly) {
        toast.info("No downloads were planned for this selection");
        return true;
      }
      return handleLoadRef.current(repoId, opts, advanced, pickToastId);
    },
    [stage, currentLoadAdvanced, requestDownloadPlan, modelSelectionAction, pickGuard, revertPick, pickToast, engineFor, offerLoadAnyway, dismissLoadAnyway],
  );
  const loadOrStageRef = useRef(loadOrStage);
  useEffect(() => {
    loadOrStageRef.current = loadOrStage;
  }, [loadOrStage]);

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
    dismissLoadAnyway();
  }, [pickToast, dismissLoadAnyway]);

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
      // Normal loads belong to the latest selection. A download-only pick claims nothing and
      // retires nothing: it fetches files, and a staged load already in flight still owns the page.
      const downloadOnly = modelSelectionAction === "download";
      const token = downloadOnly ? 0 : pickGuard.claim();
      const downloadSnapshot = downloadOnly ? currentLoadAdvanced(repoId, false) : undefined;
      const isCurrent = () => isMounted.current &&
        (downloadOnly || pickGuard.holds(token));
      // onResolved skips the label for a download-only pick, so this snapshot is never installed
      // for one; taking the slot over would strand a staged load's own baseline in it.
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
          toast.error(downloadOnly
            ? "Pick a quantization for this model to download it"
            : "Pick a quantization for this model to load it"),
        // Optimistic label, reverted if the load never starts, like the curated GGUF branch below.
        onResolved: (filename) => {
          if (downloadOnly) return;
          quantRevert.current = revert;
          setQuant(quantHint ?? filename);
          applyImageModelDefaults(repoId, effectiveFamilyOverride, false, filename);
        },
        onNotStarted: () => {
          // Nothing was applied for a download-only pick, so there is nothing to hand back, and
          // what sits in the slot is the staged load's own baseline.
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
    // This arrival owns the page like a direct pick, so a download staged by an earlier one cannot
    // land on top -- unless it is a Download only arrival, which owns nothing and must leave a load
    // already staging with its claim, or resumePendingLoad drops that load once its files arrive.
    const downloadOnlyPick = modelSelectionAction === "download";
    const token = downloadOnlyPick ? undefined : pickGuard.claim();
    if (!downloadOnlyPick) {
      setFamilyOverride("auto");
      setExternalModelId(null);
    }
    void navigateSelf({ to: "/images", search: {}, replace: true });
    // A label means a GGUF repo whatever the catalog says, and is not loadable, so resolve it
    // rather than routing it as a filename.
    if (routedLabel) {
      // Deferred, not inline: resolution is a request, and the load it fires owns the state a direct pick sets.
      void Promise.resolve().then(() =>
        loadGgufRepoPick(wanted, routedLabel, "hub", null, "auto"),
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
      void Promise.resolve().then(() => loadGgufRepoPick(pick.repoId, null, "hub", null, "auto"));
      return;
    }
    // Match every direct picker branch: the routed intent owns both the visible build label and
    // the Default recipe, and a load that never becomes resident rolls both back. A Download only
    // arrival owns neither, and quantRevert is one slot a staged load may already hold.
    const revert: PickRevert | null = downloadOnlyPick
      ? null
      : (quantRevert.current ?? { prev: quant, steps, guidance });
    if (revert) {
      quantRevert.current = revert;
      setQuant(pick.opts.kind === "pipeline" ? null : (pick.opts.filename ?? null));
      applyImageModelDefaults(
        wanted,
        "auto",
        false,
        pick.opts.kind === "gguf" ? (pick.opts.filename ?? null) : null,
      );
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

  // A Library "View in" link arrives as ?item=: select that image, paging back until it loads. A
  // counter, not effect cleanup, retires a lookup: clearing the query must not cancel its own.
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
    // Against the current status: another client may have replaced the model this page last loaded.
    const l = reapplyTarget(lastLoad.current, status);
    if (!l) return;
    const opts = { kind: l.kind, filename: l.filename, displayRepoId: l.displayRepoId, baseRepo: l.baseRepo };
    // An engine change reads the other engine's companion files: plan and stage them first.
    if (reapplyChangesEngine(engineChoice, status)) {
      void loadOrStage(l.repoId, opts, reloadSource(l.repoId), undefined, true, undefined, true);
      return;
    }
    void handleLoad(l.repoId, opts);
  }, [engineChoice, handleLoad, loadOrStage, status]);

  // The LoRA note's action: the same GGUF pick, on the native engine. A resident model the page did not load
  // reloads from its status, which names the checkpoint.
  const handleNativeReload = useCallback(() => {
    const l = nativeReloadTarget(lastLoad.current, status);
    if (!l) return;
    setEngineChoice("sd_cpp");
    // Now, not on the next render: the plan and the staged snapshot read the select through this ref.
    engineSelection.current = { ...engineSelection.current, choice: "sd_cpp" };
    // Through the plan, which stages the native companions (VAE / text encoders) the diffusers load never fetched.
    void loadOrStage(
      l.repoId,
      { kind: l.kind, filename: l.filename, displayRepoId: l.displayRepoId, baseRepo: l.baseRepo },
      reloadSource(l.repoId),
      undefined,
      true,
      undefined,
      true,
    );
  }, [loadOrStage, status]);

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

  // The chat picker emits (modelId, quant + filename) for a GGUF, or just (modelId) for a curated safetensors pick.
  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      // A Download only selection fetches files; it does not take over the page. Retiring the staged
      // intent and claiming the page for it stranded a load that was already downloading: that model
      // finished downloading and then never loaded, with no toast and nothing to retry from.
      const downloadOnlyPick = modelSelectionAction === "download";
      // Ignore picks while a load/generation/unload is in flight: the backend rejects a second load
      // with a 409. A download-only pick submits no load, so that cannot happen, and the selector
      // stays interactive during a generation: refusing it there was a silent dead click.
      if (busy !== null && !downloadOnlyPick) return;
      if (meta.source === "external") {
        if (busy !== null) return;
        if (quantRevert.current && quantRevert.current === stagedQuantRevert.current) {
          revertPick(quantRevert.current);
          quantRevert.current = null;
        }
        pickGuard.cancel();
        pickToast.dismissAll();
        beginPick();
        setExternalModelId(id);
        return;
      }
      if (!downloadOnlyPick) setExternalModelId(null);
      if (!downloadOnlyPick) beginPick();
      // This pick owns the page now, so one still awaiting a listing or a plan drops out. Before any
      // branch, since staging never sets `busy`.
      const token = downloadOnlyPick ? undefined : pickGuard.claim();
      // A download-only pick holds no token, and an absent token owns nothing to hand back.
      const stillOwnsPick = (): boolean => token !== undefined && pickGuard.holds(token);
      const pipelineTarget = diffusionPipelineLoadTarget(id, meta);
      const { displayRepoId } = pipelineTarget;
      const familyOverrideRequired = meta.familyOverrideRequired === true;
      const nextFamilyOverride = familyOverrideRequired ? familyOverride : "auto";
      if (!downloadOnlyPick && !familyOverrideRequired) setFamilyOverride("auto");
      // Curated non-GGUF model: load as a full pipeline or single-file safetensors.
      const spec = loadSpecFor(id, IMAGE_CATALOG);
      if (spec && spec.kind !== "gguf") {
      // Carried forward when one is already pending: a superseded staged pick left its optimistic
      // quant and recipe in state, so snapshotting now would record THAT and restore a model that
      // never loaded. Leaving the old entry would also let that download revert this pick.
        // A Download only pick fetches files and never becomes the resident model, so it
        // relabels nothing and claims no revert: quantRevert is ONE slot, and a staged load
        // already in flight owns what is in it and still has to commit it.
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
      // GGUF quant pick from the variant expander. Optimistic for instant feedback, but reverted if
      // the load fails to START or later in the poll: the old pipeline stays loaded either way.
      if (meta.ggufVariant && meta.ggufFilename) {
        // A Download only pick fetches files and never becomes the resident model, so it
        // relabels nothing and claims no revert: quantRevert is ONE slot, and a staged load
        // already in flight owns what is in it and still has to commit it.
        const revert: PickRevert | null = downloadOnlyPick
          ? null
          : (quantRevert.current ?? { prev: quant, steps, guidance });
        if (revert) {
          quantRevert.current = revert;
          setQuant(meta.ggufVariant);
          applyImageModelDefaults(id, nextFamilyOverride, false, meta.ggufFilename);
        }
        void loadOrStage(
          id,
          { kind: "gguf", filename: meta.ggufFilename },
          meta.source,
          token,
        ).then((started) => {
          // `quantRevert` is one slot, so only the pick that set the label may take it back.
          if (!started && revert && stillOwnsPick()) {
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
            nextFamilyOverride,
          );
          return;
        }
        // A direct pick carries no curated variant label, so surface the filename or the selector
        // keeps advertising the old quant. Optimistic, reverted if the load never starts.
        // A Download only pick fetches files and never becomes the resident model, so it
        // relabels nothing and claims no revert: quantRevert is ONE slot, and a staged load
        // already in flight owns what is in it and still has to commit it.
        const revert: PickRevert | null = downloadOnlyPick
          ? null
          : (quantRevert.current ?? { prev: quant, steps, guidance });
        if (revert) {
          quantRevert.current = revert;
          setQuant(filename);
          applyImageModelDefaults(id, nextFamilyOverride, false, filename);
        }
        void loadOrStage(dir, { kind: "gguf", filename }, meta.source, token).then((started) => {
          // Guarded like every sibling branch: quantRevert is one slot, so a pick that no longer
          // owns the page must not hand back a label a newer pick has already set.
          if (!started && revert && stillOwnsPick()) {
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
        // A Download only pick fetches files and never becomes the resident model, so it
        // relabels nothing and claims no revert: quantRevert is ONE slot, and a staged load
        // already in flight owns what is in it and still has to commit it.
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
      // A GGUF repo with no filename: these used to fall through to the pipeline branch below, which
      // the backend rejects for a single-file GGUF repo.
      if (spec?.kind === "gguf" || meta.ggufVariant) {
        // An artifact that names its file short-circuits the listing; otherwise the label is the hint.
        void loadGgufRepoPick(
          id,
          spec?.filename ?? meta.ggufVariant ?? null,
          meta.source,
          meta.source === "local" ? id : null,
          nextFamilyOverride,
        );
        return;
      }
      // Otherwise treat it as a full diffusers repo. The backend gates loads to unsloth/* repos or on-device paths.
      if (!pipelineTarget.onDevice && !id.toLowerCase().startsWith("unsloth/")) {
        // A refused Download only pick retires nothing: it never claimed the page, and the rollback
        // slot it would clear belongs to whatever load is still staging.
        toast.error("Only unsloth or on-device image models can be loaded here");
        if (!downloadOnlyPick) abandonPick();
        return;
      }
      // Optimistically clear the quant label, revert it if the load never starts.
      // A Download only pick fetches files and never becomes the resident model, so it
      // relabels nothing and claims no revert: quantRevert is ONE slot, and a staged load
      // already in flight owns what is in it and still has to commit it.
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
      pickToast,
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
      pickToast.dismissAll();
      setExternalModelId(null);
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

  // Resolves true when the backend accepted the unload; handleCancelLoad reports the cancel only then.
  const handleUnload = useCallback(async (): Promise<boolean> => {
    pendingRecalledGeneration.current = null;
    // Ejecting cancels any in-flight replacement load, so tear down its client-side tracking too
    // or the toast leaks forever.
    dropResidentState();
    loadTrackingRestored.current = false;
    setBusy("unloading");
    try {
      setStatusIfNewest(++statusTicket.current, await unloadDiffusionModel());
      setQuant(null);
      // Hold the page until any load start still in flight has run to its END, compensating unload and
      // all. Without the fence an eject landing before the start registered returned success and cleared
      // busy, so the next pick was refused while the older load carried on: that older handler, seeing
      // the newer loadSeq, skips its compensating unload and leaves a multi-gigabyte load running with
      // no toast and no cancel control.
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

  // Seed reference detail and match-source area from the build's canvas tier when the loaded build changes.
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

  // Keep the size on the loaded model's grid and inside its bounds, so the value shown is the value sent.
  const limitsKey = `${sizeLimits.multiple}|${sizeLimits.maxSide}|${sizeLimits.maxPixels}`;
  const [fittedLimits, setFittedLimits] = useState(limitsKey);
  if (limitsKey !== fittedLimits) {
    setFittedLimits(limitsKey);
    const fitted = fitSize(width, height, sizeLimits);
    if (fitted.width !== width) setWidth(fitted.width);
    if (fitted.height !== height) setHeight(fitted.height);
  }

  // Keyed to the image it was read from, so a stale read never sizes a newer source.
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

  // `numberOf` is the image number the model sees for slot i.
  const renderAdditionalImages = (numberOf: (i: number) => number, hint: string) => (
    <>
      {referenceImages.map((img, i) => (
        <Field key={i} label={`Image ${numberOf(i)}`} hint={hint}>
          <div className="space-y-1.5">
            <ImageDropzone
              value={img}
              onChange={(v) =>
                // Keep the slot in place (empty string when cleared) so other slots do not renumber mid-edit;
                // empty slots are dropped at send time.
                setReferenceImages((prev) => prev.map((p, j) => (j === i ? (v ?? "") : p)))
              }
              removeLabel={`Remove image ${numberOf(i)}`}
              cropSubject={`Image ${numberOf(i)}`}
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
          <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
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

  // Aspect ratio, Resolution and 2K presets. Unified Edit shows them under Output size when Custom is
  // picked, next to the choice that reveals them; every other workflow keeps them in place.
  const sizeControls = (
    <>
      <Field
        label="Aspect ratio"
        hint="Pick a ratio to lock the proportions, then set the size below. Flip swaps width and height."
      >
        <AspectRatioSelect
          value={aspect}
          onValueChange={changeAspect}
          open={active && aspectOpen}
          onOpenChange={(o) => setAspectOpen(active && o)}
          portrait={portrait}
          onFlip={flipDimensions}
        />
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
            options={dimOptions(sizeLimits)}
            snap={(n) => snapDim(n, sizeLimits)}
          />
          <DimensionSelect
            icon={ArrowUpDownIcon}
            label="Height"
            value={height}
            open={active && heightOpen}
            onOpenChange={(o) => setHeightOpen(active && o)}
            onChange={changeHeight}
            options={dimOptions(sizeLimits)}
            snap={(n) => snapDim(n, sizeLimits)}
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
    // Consume before any early return so it never leaks into a later run.
    const allowOversizedSent = allowOversizedField(allowOversized, oversizedOnce.current) === true;
    oversizedOnce.current = false;
    const vary = sdCaps?.variation_seed ? varyOnce.current : null;
    varyOnce.current = null;
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    if (restoredSampler && samplerOptions.length > 0) {
      setRestoredSampler(null);
      setSamplerChoices((all) => withSamplerChoice(all, samplerModelKey, samplerChoice));
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
    const multiEditRun = isEdit && multiImageEdit;
    const emptySlot = (isReference || unifiedEditRun || multiEditRun) ? referenceImages.findIndex((img) => !img) : -1;
    if (emptySlot >= 0) {
      const n = unifiedEditRun ? additionalImageNumber(emptySlot, localizedMode) : emptySlot + 2;
      toast.error(`Image ${n} is empty. Add it, or remove that slot.`);
      return;
    }
    if ((isReference || unifiedEditRun || multiEditRun) && referenceImages.filter(Boolean).length > maxExtras) {
      toast.error(
        `This model takes at most ${maxExtras + 1 + (unifiedEditRun && localizedMode === "mask" ? 1 : 0)} input images in total. Remove ${referenceImages.filter(Boolean).length - maxExtras}.`,
      );
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
    let condFields: ReturnType<typeof conditionedRequestFields> | undefined;
    let condInpaint: InpaintMaskRequestFields = {};
    try {
      if (isTransform) {
        condInit = initImage ?? undefined;
        condStrength = strength;
      } else if (isInpaint) {
        condInit = initImage ?? undefined;
        condMask = maskImage ?? undefined;
        condStrength = strength;
        condInpaint = inpaintMaskRequestFields(
          { maskBlur, onlyMasked: inpaintOnlyMasked, padding: inpaintPadding },
          status?.supports_inpaint_only_masked === true,
        );
      } else if (isExtend) {
        const built = await buildOutpaint(initImage!, extendSides, extendPct);
        condInit = built.image;
        condMask = built.mask;
        condStrength = 1; // the new border is blank canvas: redraw it fully
        // Runs as inpaint; the label only keeps the recipe saying Extend.
        condFields = { workflow: "outpaint" };
      } else if (isUpscale) {
        // Hires fix: the backend enlarges the source and re-denoises at this low strength, gaining
        // detail without changing content.
        condInit = initImage ?? undefined;
        condUpscale = upscaleFactor;
        condStrength = upscaleStrength;
      } else if (isReference || unifiedEditRun || multiEditRun) {
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
        // Edit-only model: the source alone; the prompt IS the instruction and the output takes its size.
        condFields = { workflow: "edit", init_image: initImage ?? undefined };
      }
    } catch {
      toast.error("Could not prepare the source image");
      return;
    }
    // Resolve a base seed up front, so each sequential image gets a distinct, reproducible seed (base + i).
    let baseSeed: number;
    if (vary) {
      baseSeed = vary.seed;
    } else if (seed.trim() && !external) {
      const n = Number(seed);
      if (!Number.isInteger(n) || n < 0 || n > Number.MAX_SAFE_INTEGER) {
        toast.error("Seed must be a non-negative integer");
        return;
      }
      baseSeed = n;
    } else {
      baseSeed = Math.floor(Math.random() * 2 ** 32);
    }
    const runChoice: SdControlsChoice = vary
      ? { ...sdChoice, variationSeed: String(vary.variationSeed), variationStrength: vary.strength }
      : sdChoice;
    const varying = Boolean(sdCaps?.variation_seed) && workflow === "create" && runChoice.variationStrength > 0;
    // A compare grid checks it at any form strength: a Var. strength axis can engage it with the form's at 0.
    const checkVariation = varying || (compareOnce.current !== null && Boolean(sdCaps?.variation_seed));
    if (checkVariation && runChoice.variationSeed.trim() && parseVariationSeed(runChoice.variationSeed) === null) {
      toast.error("Variation seed must be an integer from 0 to 4294967295");
      return;
    }
    const baseVariationSeed = varying
      ? (parseVariationSeed(runChoice.variationSeed) ?? randomVariationSeed())
      : null;

    // Snap to the model's grid and bounds so a half-typed value cannot 400.
    const sent = unifiedEditRun ? editSize : fitSize(width, height, sizeLimits);
    const w = sent.width;
    const h = sent.height;
    if (!unifiedEditRun || editSizing === "custom") {
      if (w !== width) setWidth(w);
      if (h !== height) setHeight(h);
    }

    // A large run count is legitimate, so no upper cap: floor at 1 and ignore non-numeric input.
    const runs = Number.isFinite(count) && count >= 1 ? Math.floor(count) : 1;
    if (runs !== count) setCount(runs);
    // A compare grid sends one request per cell, in the order compare/ chose.
    const compare = compareOnce.current;
    const runTotal = compare ? compare.cellCount : runs;

    // An explicit seed near the 2**53-1 backend cap can overflow once the per-run and in-batch
    // offsets are added, 422ing a later run after earlier images generated. Fail before GPU work. A compare grid adds
    // no offset (every cell runs at batch size 1 on the base seed or its Seed axis value, both already in range).
    if (!compare && baseSeed > Number.MAX_SAFE_INTEGER - (runs * batchSize - 1)) {
      toast.error("Seed too large for this run count and batch size; use a smaller seed");
      return;
    }

    // Saved only once the request passes validation, so a rejected attempt is not kept.
    saveLastPrompt(`images:${workflow}`, prompt);
    setBusy("generating");
    setGenDone(0);
    setGenStep(null);
    // Fresh run: a Stop from the PREVIOUS run must not cancel this one.
    cancelRequested.current = false;
    setStopping(false);
    cancelAcked.current = false;
    // Per-run, like the two above: a cancel POST still outstanding from the PREVIOUS run would
    // leave the guard set and swallow this run's own Stop.
    cancelInFlight.current = null;
    // Any Stop still pending belongs to the run that just ended, so it must not reach the server now.
    cancelAbort.current?.abort();
    cancelAbort.current = null;
    runToken.current += 1;
    runTarget.current = external ? "connection" : undefined;
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
    // Every gallery id this page has seen, captured BEFORE the first POST and grown as records
    // arrive: settleLostGeneration proves a lost POST landed by finding a record outside it.
    const knownIds = new Set(galleryCache.images.map((image) => image.id));
    try {
      for (let i = 0; i < runTotal; i++) {
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
        // Offset runs by the batch size: the native engine seeds image j at seed+j, so a +1 offset
        // would regenerate batch-mates.
        const runBaseSeed = varying ? baseSeed : baseSeed + i * batchSize;
        let res: DiffusionGenerateResponse;
        try {
          res = await generateDiffusionImage(compareRequest(compare, i, baseSeed, {
            prompt: prompt.trim(),
            // Only send a negative prompt when guidance uses it, so the recipe does not record one the model ignored.
            negative_prompt: negativeEngaged ? negativePrompt.trim() || undefined : undefined,
            width: w,
            height: h,
            steps,
            guidance: effectiveGuidance,
            seed: runBaseSeed,
            batch_size: batchSize,
            // Transform/Inpaint/Extend send the source image (and mask) with a strength; the backend
            // derives output size from the image.
            init_image: condInit,
            mask_image: condMask,
            strength: condStrength,
            ...condInpaint,
            upscale: condUpscale,
            upscale_model: isUpscale ? upscaleModelRequest(upscaleModel, upscaleModels) : undefined,
            hires: hiresRequest(hires, hiresCapability, workflow),
            allow_oversized: allowOversizedSent ? true : undefined,
            live_preview: livePreview,
            ...samplerRequestFields(samplerOptions, samplerChoice),
            ...sdControlsRequestFields(
              sdCaps,
              runChoice,
              // A compare grid keeps the form's variation seed for every cell (its axis, if any, replaces it), so
              // only the swept setting differs between cells.
              baseVariationSeed !== null ? (baseVariationSeed + (compare ? 0 : i * batchSize)) % 2 ** 32 : null,
              varying,
              !controlnetEngaged,
            ),
            ...shiftRequestFields(shiftControl, shiftValue),
            ...enhancePromptRequestField(upsamplerCapability, enhanceOn),
            ...condFields,
            ...detailerRequestField(detailerOn, detailerReady, detailerOptions),
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
            ...(external && { provider_id: external.providerId, model: external.modelId }),
          }));
        } catch (err) {
          // The POST response was lost while the backend kept generating (the secure-mode tunnel caps
          // the response near 100s). Retrying would duplicate the work, so wait the run out.
          if (!(err instanceof GenerateResponseLostError)) throw err;
          // A record outside the baseline proves the request reached the backend. Taken per attempt, so
          // it reflects what the client could see when THIS post went out.
          await settleLostGeneration(() => isMounted.current, probeBaseline);
          if (!isMounted.current) break;
          await loadGallery();
          // A compare cell takes its record from what the gallery gained (matched by seed in compare/).
          compare?.onImages(i, galleryCache.images.filter((image) => !knownIds.has(image.id)));
          // loadGallery refreshes the module cache synchronously, so this run's records are folded in
          // before the next run.
          galleryCache.images.forEach((image) => knownIds.add(image.id));
          // The run landed without its records, so Reuse last seed takes the seed it was sent.
          if (!external) setSessionSeed(compare?.sentSeed(i) ?? runBaseSeed);
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
        const runSeed = lastSeedOf(res.images);
        if (runSeed !== null) setSessionSeed(runSeed);
        // Asked for enhancement and did not get it: say so once, without blocking the result.
        if (enhanceOn && upsamplerCapability.supported) {
          const notice = enhancementNotice(res.images[0]?.prompt_enhancement);
          if (notice) toast.info(notice.title, { id: "prompt-enhancement-notice", description: notice.description });
        }
        res.images.forEach((image) => void ensureSrc(image));
        const detailFailed = detailerErrors(res.images);
        if (detailFailed.length > 0) {
          toast.warning("The Detailer did not run on this image", {
            id: "images-detailer-error",
            description: detailFailed.join("; "),
          });
        }
        compare?.onImages(i, res.images);
        setGenDone(i + 1);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Image generation failed";
      // The user's own Stop comes back as the backend's cancelled sentinel (409), so it is not
      // toasted. Only a Stop the backend confirmed explains an error away: a POST that never
      // landed, or {cancelled: false}, means whatever it raised is a real failure.
      const report = shouldReportGenerateError({
        message: msg,
        stopRequested: cancelRequested.current && cancelAcked.current,
      });
      const offerAnyway = report && shouldOfferGenerateAnyway({ error: err, allowOversizedSent });
      // Not for a compare cell: the queued retry would run after the grid has ended, as a plain form generation.
      if (offerAnyway && compare) {
        toast.error(MEMORY_REFUSAL_TITLE, {
          description: `${msg} To run the grid anyway, turn on "${ALLOW_OVERSIZED_LABEL}" and generate it again.`,
          duration: 20_000,
        });
      } else if (offerAnyway) {
        toast.error(MEMORY_REFUSAL_TITLE, {
          description: msg,
          duration: 20_000,
          action: {
            label: GENERATE_ANYWAY_LABEL,
            onClick: () => {
              varyOnce.current = vary;
              setOversizedRetryQueued(true);
            },
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
      // Refresh on EVERY exit, not just the successful one, and AWAIT it before Generate comes back: a
      // generation can change server-side status (Speed=Auto compiles on the 3rd LoRA-free run and
      // supports_lora flips false) and a cancelled native run can leave no model at all, so re-enabling
      // first would offer a button that 409s.
      if (isMounted.current) await refreshStatus();
      setBusy(null);
      setGenDone(null);
      setGenStep(null);
      setStopping(false);
    }
  }, [external, allowOversized, livePreview, enhanceOn, upsamplerCapability, prompt, negativePrompt, negativeEngaged, restoredSampler, samplerModelKey, samplerOptions, samplerChoice, sdCaps, sdChoice, shiftControl, shiftValue, width, height, steps, effectiveGuidance, seed, batchSize, count, workflow, initImage, maskImage, strength, maskBlur, inpaintOnlyMasked, inpaintPadding, status?.supports_inpaint_only_masked, extendPct, extendSides, upscaleFactor, upscaleStrength, upscaleModel, upscaleModels, hires, hiresCapability, referenceImages, loras, loraCapable, controlnetCapable, controlnetEngaged, controlnetId, controlImage, controlType, controlStrength, ensureSrc, loadGallery, refreshStatus, unifiedEdit, multiImageEdit, localizedMode, localizedLayer, maxExtras, referenceResolution, conditioning, editSize, editSizing, sizeLimits, detailerOn, detailerReady, detailerOptions]);

  // Stop the in-flight generation. Latch FIRST, so a multi-run request stops even if the POST
  // races the run that is already finishing.
  const handleCancelGenerate = useCallback(async () => {
    cancelRequested.current = true;
    setStopping(true);
    // One Stop on the wire at a time. The button stays enabled so the click still latches, but a
    // second POST would target whatever is active when IT arrives.
    const token = runToken.current;
    if (cancelInFlight.current === token) return;
    cancelInFlight.current = token;
    const abort = new AbortController();
    cancelAbort.current = abort;
    try {
      const { cancelled } = runTarget.current
        ? await cancelConnectionImageGeneration(abort.signal)
        : await cancelDiffusionGeneration(abort.signal);
      cancelAcked.current = Boolean(cancelled);
      if (!cancelled) setStopping(false);
    } catch {
      // An abort means the next run dropped this one on purpose, so there is nothing to report.
      // Otherwise the request never landed and the denoise runs on, so the click is not handled.
      if (!abort.signal.aborted) {
        cancelAcked.current = false;
        setStopping(false);
        toast.error("Could not reach the server to stop this generation; it is still running");
      }
    } finally {
      // Only if they are still ours: a slow cancel from an earlier run must not release the guard a later run set.
      if (cancelInFlight.current === token) cancelInFlight.current = null;
      if (cancelAbort.current === abort) cancelAbort.current = null;
    }
  }, []);

  // Compare grid: handleGenerate reads the run from compareOnce, so a grid shares Stop, progress and the gallery
  // merge with Generate. Text-to-image on a loaded local model only.
  const [compareRun, setCompareRun] = useState<CompareRun | null>(null);
  const runCompare = useCallback(
    async (run: CompareRun) => {
      // Like Generate: the stored preset must replace the default steps / size before any request is sent.
      if (busy !== null || !imagePresets.hydrated) return;
      compareOnce.current = run;
      setCompareRun(run);
      try {
        await handleGenerate();
      } finally {
        compareOnce.current = null;
        setCompareRun(null);
      }
    },
    [busy, handleGenerate, imagePresets.hydrated],
  );
  const compareAvailable = !external && Boolean(status?.loaded) && workflow === "create";
  const compareSdControls = sdCaps;
  const compareContext = useMemo<CompareContext>(
    () => ({
      prompt,
      negativePrompt,
      negativeCapable,
      negativeMinGuidance,
      guidanceFixed,
      guidance: effectiveGuidance,
      samplers: samplerOptions,
      samplerChoice,
      shift: shiftControl,
      loraCapable,
      loras: loras
        .filter((l) => l.id.trim())
        .map((l) => ({
          id: l.id,
          weight: l.weight,
          label: availableLoras.find((a) => a.id === l.id)?.display_name ?? l.id.split(/[\\/]/).pop() ?? l.id,
        })),
      sdControls: compareSdControls,
      hiresEngagesNegative: hiresEngagesNegative(hires, hiresCapability, workflow, negativeMinGuidance),
      // The predicate the request uses (strength 0 runs plain text-to-image, where CFG rescale applies).
      controlnetEngaged,
    }),
    // samplerChoice is rebuilt every render; its two fields are what matter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prompt, negativePrompt, negativeCapable, negativeMinGuidance, guidanceFixed, effectiveGuidance, samplerOptions, samplerChoice.sampler, samplerChoice.scheduleType, shiftControl, loraCapable, loras, availableLoras, compareSdControls, controlnetEngaged, hires, hiresCapability, workflow],
  );

  const handleGenerateWithRecall = useCallback(async () => {
    if (busy !== null || !imagePresets.hydrated) return;
    if (external) {
      await handleGenerate();
      return;
    }
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
      // "Generate anyway" on an unloaded model: the retry's finally clears the one-shot before this runs.
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
    external,
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

  const canVary =
    !external && Boolean(status?.loaded && sdCaps?.variation_seed) && workflow === "create" && selected !== null &&
    canVaryImage(selected, status?.display_repo_id ?? status?.repo_id ?? null);
  const varyBlocked = canVary && selected ? varyBlockedReason(selected, loraCapable, sdCaps, status ?? {}) : null;
  // Vary runs the SELECTED image's recipe, not the form: restore it, then generate once the form has rendered.
  const [varyQueued, setVaryQueued] = useState(false);
  const handleVary = useCallback(() => {
    if (busy !== null || !selected || !sdCaps?.variation_seed || varyBlocked) return;
    varyOnce.current = {
      seed: selected.seed,
      variationSeed: randomVariationSeed(),
      strength: sdCaps.vary_strength ?? FALLBACK_VARY_STRENGTH,
    };
    restoreSettings(selected, true, "vary");
    setVaryQueued(true);
  }, [busy, restoreSettings, sdCaps, selected, varyBlocked]);
  useEffect(() => {
    if (!varyQueued || busy !== null) return;
    setVaryQueued(false);
    void handleGenerate();
  }, [varyQueued, busy, handleGenerate]);

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

  // Publish what the loaded model can do, so the sidebar submenu dims the rest. null while
  // nothing is loaded, which leaves every workflow open to set up first.
  useEffect(() => {
    if (external) {
      setSupported(["create"]);
      return;
    }
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
  }, [external, status?.loaded, status?.workflows, setSupported]);

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
      {!external && engineSelectVisible(status) && (
        <AdvancedSelect
          label="Engine"
          hint={
            status?.engine_forced
              ? `${ENGINE_HINT} The server pins UNSLOTH_DIFFUSION_ENGINE=${status.engine_forced}, which wins over this choice.`
              : ENGINE_HINT
          }
          badge={<ResolvedBadge status={status} controlKey="engine" />}
          value={engineChoice}
          onValueChange={(v) => setEngineChoice(v as EngineChoice)}
          options={ENGINE_OPTIONS}
        />
      )}
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
      {/* Use the same precision eligibility rule as the load request. Native sd.cpp reports
          model_kind "gguf" as well, to be recallable by exact checkpoint, but runs no torchao
          path: gate it out by ENGINE or it offers FP8/INT8/NVFP4 with no badge and snaps back. */}
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
            // The explicit low-precision schemes need the dense tensor-core path, which a Mac or
            // CPU-only host cannot run, so the picker does not list what the loader would refuse.
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
            // Opt-out: now that a family default can pick a scheme, omitting the field is no longer the dense request.
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
        hint="Attention kernel. Auto upgrades to cuDNN fused attention on NVIDIA when a speed profile is active. sage is INT8 attention (SageAttention 2; without a local install Unsloth fetches the Hugging Face kernels-hub build, which runs on Ampere, Ada and Hopper GPUs, and any other GPU keeps the default): fast (10-40%) but can black-frame some families (Qwen, Wan), so it never engages automatically."
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
          <InfoHint>
            {livePreviewUnavailable
              ? "The loaded model shows no preview while it denoises on this engine."
              : "Show a rough preview of the image while it denoises. Costs no measurable speed and never changes the final image."}
          </InfoHint>
        </span>
        <Switch
          checked={livePreview && !livePreviewUnavailable}
          onCheckedChange={(on) => setLivePreviewOff(!on)}
          disabled={livePreviewUnavailable}
          aria-label="Live preview"
        />
      </div>
      {sdCaps && (
        <>
          {sdCaps.prompt_weighting && (
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
                Prompt weighting
                <InfoHint>
                  AUTOMATIC1111 syntax: (word:1.2) weights a word, (word) is 1.1x, [word] is 1/1.1x,
                  \( and \) type a literal bracket, BREAK starts a new 75-token chunk. Prompts past 75
                  tokens are encoded in chunks instead of cut off. A prompt without this syntax that fits in
                  75 tokens renders exactly as before. Off reads brackets as plain text.
                </InfoHint>
              </span>
              <Switch
                checked={sdChoice.promptWeighting}
                onCheckedChange={(on) => setSdChoice((c) => ({ ...c, promptWeighting: on }))}
                aria-label="Prompt weighting"
              />
            </div>
          )}
          {sdCaps.clip_skip && (
            <SliderField
              label="CLIP skip"
              hint={`AUTOMATIC1111 numbering: 1 uses the model's own text-encoder layer. SDXL already reads the penultimate layer, so 1 and 2 render the same (Pony and Illustrious cards often say 2). ${sdCaps.clip_skip.max > 2 ? "3 and up read earlier layers." : ""}`}
              value={sdChoice.clipSkip}
              min={1}
              max={sdCaps.clip_skip.max}
              step={1}
              onChange={(v) => setSdChoice((c) => ({ ...c, clipSkip: v }))}
            />
          )}
          {sdCaps.cfg_rescale && (
            <SliderField
              label="CFG rescale"
              hint={
                controlnetEngaged
                  ? "Not available with a ControlNet: the SDXL ControlNet pipeline has no CFG rescale. Set the ControlNet strength to 0 or clear it to use this."
                  : sdCaps.cfg_rescale.recommended
                  ? `Rescales the guided prediction to fix overexposure. This is a v-prediction model: ${sdCaps.cfg_rescale.recommended} is the value from the zero-terminal-SNR paper. 0 = off.`
                  : "Rescales the guided prediction to fix overexposure, mostly for v-prediction checkpoints (0.7 there). 0 = off."
              }
              value={sdChoice.cfgRescale}
              min={0}
              max={sdCaps.cfg_rescale.max}
              step={0.05}
              onChange={(v) => setSdChoice((c) => ({ ...c, cfgRescale: v }))}
              disabled={controlnetEngaged}
            />
          )}
          {sdCaps.variation_seed && workflow === "create" && (
            <>
              <SliderField
                label="Variation strength"
                hint="Blends the seed's starting noise toward the variation seed's (AUTOMATIC1111 Variation seed): 0 = off, small values keep the composition, 1 = the variation seed alone. Text to image only. A batch keeps one seed and steps the variation seed."
                value={sdChoice.variationStrength}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => setSdChoice((c) => ({ ...c, variationStrength: v }))}
              />
              {sdChoice.variationStrength > 0 && (
                <Field label="Variation seed" hint="Leave empty for a random variation seed each run.">
                  <Input
                    placeholder="Random if empty"
                    value={sdChoice.variationSeed}
                    onChange={(e) => setSdChoice((c) => ({ ...c, variationSeed: e.target.value }))}
                    aria-label="Variation seed"
                  />
                </Field>
              )}
            </>
          )}
        </>
      )}
      <LoadedBuildSummary status={status} />
    </>
  );

  return (
    // The chat-style layout gives this page no outer top inset, so it applies the content inset itself, as chat does.
    <div
      {...{ [MEDIA_RAIL_ROOT_ATTR]: "" }}
      style={railRootStyle}
      className="diffusion-surface @container relative flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]"
    >
      {/* Page-level, so the handle covers the divider through the header too (Create and Train). */}
      <MediaRailResizeHandle kind="images" placement="page" className="hidden @[50rem]:block" />
      {/* Portals to body, and this page stays mounted off-route, so gate it like the composer. */}
      {active && <GuidedTour {...tour.tourProps} />}
      {/* Keep the tabs centered over the preview at every width: the model rail holds at its
          (draggable) width when space permits and shrinks only to preserve the controls. */}
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
                externalModels={externalModels}
                value={(external && externalModelId) || selectorModelId}
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
                hubCapability="image-gen"
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
              className="h-[calc(34px*var(--ui-space-scale,1))] [&>button]:h-[calc(34px*var(--ui-space-scale,1))] [&>button]:px-3 @[68rem]:[&>button]:px-11 @max-[30rem]:[&>button]:px-2.5 @max-[30rem]:[&>button>[data-pill-tab-label]]:sr-only"
              tabs={[
                { value: "create", label: "Create", icon: <HugeiconsIcon icon={SparklesIcon} className="size-3.5" /> },
                { value: "train", label: "Train", icon: <HugeiconsIcon icon={TestTubeOutlineIcon} className="size-3.5" /> },
              ]}
            />
          </div>
          <div className="pointer-events-none col-start-3 flex min-w-0 items-start justify-end pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
            <div className="pointer-events-auto flex min-w-0 items-center gap-2">
              <MediaApiButton
                kind="images"
                model={status?.loaded ? (status.repo_id ?? null) : null}
              />
              <LibraryPageLink
                tab="images"
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
        <div
          data-tour="images-settings"
          className="flex w-full shrink-0 flex-col border-b border-border/60 @[50rem]:w-[min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100%-13rem))] @[50rem]:overflow-hidden @[50rem]:border-r @[50rem]:border-b-0"
        >
          {/* pl-0.5 keeps focus rings off the scroll container's edge. */}
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar panel-scroll-fade-action flex min-h-0 flex-1 flex-col gap-4 px-10 max-sm:px-5 pt-9 pb-6 @[50rem]:overflow-y-auto",
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
                  onImportFromImage={() => importFileRef.current?.click()}
                  onPasteParameters={() => setPasteOpen(true)}
                />
              )}
              <PasteParametersDialog
                open={pasteOpen}
                onOpenChange={setPasteOpen}
                text={pasteText}
                onTextChange={setPasteText}
                onApply={importSettingsFromText}
              />
            </div>

            {workflow === "transform" && (
              <>
                <Field
                  label="Source image"
                  hint="The image to transform. Generation redraws it guided by your prompt; the Strength below controls how far."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} cropSubject="Source image" />
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
                        applyMask={textMask}
                      />
                    </Field>
                    <MaskFromTextPanel image={initImage} onMask={onTextMask} />
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
                        onClick={openInpaintCrop}
                      >
                        <HugeiconsIcon icon={ImageCropIcon} className="size-3.5" />
                        Crop
                      </Button>
                      {inpaintCropDialog}
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
                    <SliderField
                      label="Mask blur"
                      hint="Softens the mask edge by this many pixels so the repainted region blends into the source. 0 keeps the hard edge."
                      value={maskBlur}
                      min={0}
                      max={MASK_BLUR_MAX}
                      step={1}
                      onChange={setMaskBlur}
                    />
                    {status?.supports_inpaint_only_masked && (
                      <>
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
                            Only masked
                            <InfoHint>
                              Repaint just the masked area plus padding at full resolution, then paste it back. Sharper small edits on big images; the model sees less of the surroundings.
                            </InfoHint>
                          </span>
                          <Switch
                            checked={inpaintOnlyMasked}
                            onCheckedChange={setInpaintOnlyMasked}
                            aria-label="Only masked"
                          />
                        </div>
                        {inpaintOnlyMasked && (
                          <SliderField
                            label="Padding"
                            hint="Pixels of context kept around the mask when repainting only the masked area."
                            value={inpaintPadding}
                            min={0}
                            max={INPAINT_PADDING_MAX}
                            step={4}
                            onChange={setInpaintPadding}
                          />
                        )}
                      </>
                    )}
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
                  <ImageDropzone value={initImage} onChange={handleInitChange} cropSubject="Source image" />
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
                            // mouse focus anyway (index.css blanks it).
                            "rounded-full px-2 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                            on
                              ? "bg-wash-15 text-foreground hover:bg-wash-20 dark:bg-wash-25 dark:hover:bg-wash-30"
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
                  <ImageDropzone value={initImage} onChange={handleInitChange} cropSubject="Source image" />
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
                <UpscaleModelField models={upscaleModels} value={upscaleModel} onChange={setUpscaleModel} />
              </>
            )}

            {workflow === "reference" && (
              <>
                <Field
                  label={unifiedEdit ? "Image 1" : "Reference image"}
                  hint="A reference the model draws on (subject, style, or composition) while generating a NEW image from your prompt at the size below. Unlike Transform, it is not a redraw of this image, so there is no strength."
                >
                  <ImageDropzone value={initImage} onChange={handleInitChange} cropSubject={unifiedEdit ? "Image 1" : "Reference image"} />
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
              <>
                <Field
                  label={multiImageEdit ? "Image 1 (source)" : "Source image"}
                  hint="The image to edit. Describe the change in the prompt below (e.g. 'make it night', 'add a red hat', 'change the background to a beach')."
                >
                  <ImageDropzone
                    value={initImage}
                    onChange={handleInitChange}
                    cropSubject={multiImageEdit ? "Image 1" : "Source image"}
                  />
                </Field>
                {multiImageEdit &&
                  renderAdditionalImages(
                    (i) => i + 2,
                    "Another input the instruction can refer to by its number, such as a person, product or style to bring into the source. The output keeps the source's size.",
                  )}
              </>
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
                        applyMask={textMask}
                      />
                      {localizedMode !== "annotate" && (
                        <MaskFromTextPanel image={initImage} onMask={onTextMask} />
                      )}
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
                    <ImageDropzone value={initImage} onChange={handleInitChange} cropSubject="Image 1" />
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
                        <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
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

            <Field
              label={workflow === "edit" ? "Instruction" : "Prompt"}
              hint={external ? undefined : status?.prompt_hint || undefined}
            >
              <Textarea
                data-type-to-activate="prompt"
                rows={4}
                className={cn(PROMPT_BOX_CLASS, "min-h-32")}
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
                onPaste={(e) => {
                  // A whole AUTOMATIC1111 / Forge / Civitai block fills the form instead of the prompt.
                  const text = e.clipboardData.getData("text/plain");
                  if (isGenerationParametersPaste(text)) {
                    e.preventDefault();
                    void importSettingsFromText(text);
                  }
                }}
              />
            </Field>
            {upsamplerCapability.supported && (
              <EnhancePromptControls
                enabled={enhanceOn}
                onEnabledChange={changeEnhancePrompt}
                modelKey={`${status?.repo_id ?? ""}|${status?.gguf_filename ?? ""}|${loadEpoch}`}
                prompt={prompt}
                previewRequest={() => {
                  // Edit and Reference condition on their images, so the caption describes them too.
                  const conditioned = workflow === "edit" || workflow === "reference";
                  const images = conditioned
                    ? [initImage, ...referenceImages].filter((src): src is string => Boolean(src))
                    : [];
                  return {
                    prompt: prompt.trim(),
                    variant: upsamplerVariantFor(status?.family),
                    ...(images.length ? { reference_images: images } : {}),
                  };
                }}
                onUseCaption={(caption) => {
                  setPrompt(caption);
                  setEnhanceOverride(false);
                }}
              />
            )}
            {negativeCapable && (
              <NegativePromptField
                value={negativePrompt}
                onChange={setNegativePrompt}
                open={negativeOpen}
                onOpenChange={setNegativeOpen}
                hint={
                  negativeMinGuidance === null
                    ? "What to steer the image away from."
                    : `What to steer the image away from. Only used when guidance is above ${negativeMinGuidance}.`
                }
              />
            )}
            {!external && !loraCapable && offersNativeLoraReload(status) && (
              <div className="flex items-center justify-between gap-2 rounded-lg border border-dashed px-3 py-2 text-xs text-muted-foreground">
                <span>LoRAs need the native engine for this GGUF; diffusers runs it without them.</span>
                <Button size="sm" variant="secondary" className="h-7 shrink-0 text-xs" disabled={busy !== null} onClick={handleNativeReload}>
                  Reload with native engine
                </Button>
              </div>
            )}
            {!external && !loraCapable && offersCompiledLoraReload(status) && (
              <div className="flex items-center justify-between gap-2 rounded-lg border border-dashed px-3 py-2 text-xs text-muted-foreground">
                <span>Speed compiled this model after a few images, so LoRAs are off. Reload it to add LoRAs.</span>
                {(canReapply || residentReloadTarget(status) !== null) && (
                  <Button size="sm" variant="secondary" className="h-7 shrink-0 text-xs" disabled={busy !== null} onClick={handleReapply}>
                    Reload model
                  </Button>
                )}
              </div>
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
                      {/* A datalist has no groups, so the label carries the fine-tuned mark. */}
                      {sortedLoras.map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.fine_tuned ? `${a.display_name} (fine-tuned)` : a.display_name}
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
                  <div className="flex gap-2">
                    {loras.length < 8 && (
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="flex-1"
                        onClick={() => {
                          // Prefill with the first unused suggestion when a curated catalog exists, else an empty row.
                          const taken = new Set(loras.map((l) => l.id));
                          const next = sortedLoras.find((a) => !taken.has(a.id));
                          setLoras((prev) => [
                            ...prev,
                            next ? { id: next.id, weight: next.weight_default || 1 } : { id: "", weight: 1 },
                          ]);
                        }}
                      >
                        <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                        Add LoRA
                      </Button>
                    )}
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      title="Add a custom models folder. Image LoRAs exported from Unsloth in it show up here."
                      onClick={() => setLoraFoldersOpen(true)}
                    >
                      <HugeiconsIcon icon={FolderAddIcon} className="size-3.5" />
                      Folders
                    </Button>
                  </div>
                </div>
                <OnDeviceFoldersDialog
                  open={loraFoldersOpen}
                  onOpenChange={setLoraFoldersOpen}
                  onInventoryChange={() => setLoraRefreshKey((k) => k + 1)}
                />
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
                      <ImageDropzone value={controlImage} onChange={setControlImage} cropSubject="Control image" />
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

            {/* First of the one-line sliders, so it takes a bigger break than the gap gives. */}
            {!external && (
              <>
                <div className="pt-2">
                  <SliderField
                    label="Steps"
                    hint="Number of denoising steps. Start with the selected model's default; more steps take longer and may not improve quality."
                    value={steps}
                    min={1}
                    max={stepsSliderMax(pendingModelDefaults?.steps ?? residentSteps, steps)}
                    step={1}
                    onChange={setSteps}
                  />
                </div>
                <SliderField
                  label="Guidance"
                  hint={
                    guidanceFixed
                      ? "The loaded model is distilled to run without guidance, so this value has no effect and stays at its default."
                      : "Controls how strongly the model follows the prompt. Start with the selected model's default; distilled models may require low or zero guidance."
                  }
                  value={effectiveGuidance}
                  min={0}
                  max={15}
                  step={0.5}
                  onChange={setGuidance}
                  disabled={guidanceFixed}
                />
              </>
            )}
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
            {samplerOptions.length > 1 && (
              <Field
                label="Sampler"
                hint="How each denoising step is taken. Model default keeps the scheduler the model ships with. Community checkpoints often name one, such as DPM++ 2M with Karras."
                className="pt-2"
              >
                <div className="flex items-center gap-2">
                  <Select
                    value={samplerChoice.sampler}
                    onValueChange={(v) => changeSamplerChoice({ sampler: v })}
                  >
                    <SelectTrigger className="h-8 flex-1 text-xs" aria-label="Sampler">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {samplerOptions.map((o) => (
                        <SelectItem key={o.id} value={o.id}>
                          {o.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select
                    value={samplerChoice.scheduleType}
                    onValueChange={(v) => changeSamplerChoice({ scheduleType: v })}
                  >
                    <SelectTrigger className="h-8 w-32 text-xs" aria-label="Schedule type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(
                        samplerOptions.find((o) => o.id === samplerChoice.sampler)
                          ?.schedule_types ?? [AUTOMATIC_SCHEDULE]
                      ).map((t) => (
                        <SelectItem key={t} value={t}>
                          {SCHEDULE_LABELS[t] ?? t}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </Field>
            )}
            <ShiftField control={shiftControl} value={shiftValue} onChange={setShiftValue} />
            {hiresVisible(hiresCapability, workflow) ? (
              <HiresControls capability={hiresCapability} choice={hires} onChange={setHires} width={width} height={height} />
            ) : !external && workflow === "create" && status?.hires_unavailable_reason ? (
              <p className="pt-2 text-xs text-muted-foreground">{status.hires_unavailable_reason}</p>
            ) : null}
            <EmbeddingsField
              family={status?.family}
              enabled={embeddingsEnabled}
              onInsert={insertEmbedding}
            />
            <DetailerField
              available={detailerReady}
              enabled={detailerOn}
              onEnabledChange={setDetailerOn}
              options={detailerOptions}
              onOptionsChange={setDetailerOptions}
            />
            {!external && (
              <>
                <Field
                  label="Seed"
                  hint="Leave empty for a fresh random seed each run."
                  className="pt-2"
                >
                  <div className="flex items-center gap-1.5">
                    <Input
                      placeholder="Random if empty"
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      className="flex-1"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="size-8 shrink-0"
                      aria-label="Random seed"
                      title="Random seed each run"
                      disabled={seed === ""}
                      onClick={() => setSeed("")}
                    >
                      <HugeiconsIcon icon={DiceFaces03Icon} className="size-4" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="size-8 shrink-0"
                      aria-label="Reuse last seed"
                      title={lastSeed === null ? "No image generated yet" : `Reuse last seed (${lastSeed})`}
                      disabled={lastSeed === null}
                      onClick={() => {
                        if (lastSeed !== null) setSeed(String(lastSeed));
                      }}
                    >
                      <Recycle className="size-4" strokeWidth={1.5} />
                    </Button>
                  </div>
                </Field>

                {compareAvailable && (
                  <ComparePanel
                    context={compareContext}
                    busy={busy !== null || compareRun !== null || !imagePresets.hydrated}
                    seedFieldEmpty={!seed.trim()}
                    variationSeed={sdCaps?.variation_seed ? parseVariationSeed(sdChoice.variationSeed) : null}
                    modelName={status?.repo_id ?? null}
                    width={width}
                    height={height}
                    steps={steps}
                    guidance={effectiveGuidance}
                    onRun={runCompare}
                  />
                )}

                <AdvancedDisclosure open={advancedOpen} onOpenChange={setAdvancedOpen}>
                  {advancedControls}
                </AdvancedDisclosure>
              </>
            )}

          </div>
          {/* The scroll mask provides the fade; leave the footer unpainted to avoid dark-mode banding. */}
          <div className="relative z-10 flex shrink-0 flex-wrap justify-center gap-2 px-4 pt-0.5 pb-4">
            {busy === "generating" ? (
              /* Replaces Generate while a run is in flight. Every workflow funnels through the same
                 handler, so one control stops all of them. */
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
                <Button variant="dark"
                  className="relative z-10 h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                  onClick={handleGenerateWithRecall}
                  disabled={busy !== null || !imagePresets.hydrated || (!status?.loaded && !rememberedModel && !external)}
                >
                  Generate
                </Button>
                {canVary && (
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      {/* A disabled button gets no pointer events; the span keeps the reason reachable. */}
                      <span className="relative z-10 inline-flex">
                        <Button
                          className="h-11 px-5"
                          variant="secondary"
                          disabled={busy !== null || varyBlocked !== null}
                          onClick={handleVary}
                        >
                          Vary
                        </Button>
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      {varyBlocked ?? (
                        <>
                          Restore the selected image&apos;s settings and re-run its seed with a new variation seed at
                          strength {sdCaps?.vary_strength ?? FALLBACK_VARY_STRENGTH}: same composition, small changes
                        </>
                      )}
                    </TooltipContent>
                  </Tooltip>
                )}
                {!external && status?.loaded && (canReapply || residentReloadTarget(status) !== null) && (
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
          ref={previewDropRef}
          data-tour="images-preview"
          className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden @[50rem]:min-h-0"
          onDragOver={onImportDragOver}
          onDrop={onImportDrop}
        >
          <input
            ref={importFileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            data-testid="images-import-settings-input"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void importSettingsFromFile(file);
            }}
          />
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
                      // The authenticated original: WebKit shows the object URL but cannot refetch it.
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
              // Live latent preview: a small projection of the image being denoised, scaled up to the
              // requested size (the aspect comes from the preview itself). The finished image replaces it.
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
                {/* Actions grouped in one glass toolbar so they stay legible over any image. Size and seed
                    live in the Recipe popover. */}
                {/* No button borders: focus returning from a menu would draw one. Keyboard focus tints instead. */}
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-full bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur [&_[data-slot=button]]:border-0 [&_[data-slot=button]:focus-visible]:bg-muted">
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
                        onClick={() => void downloadImage(selected, "png", selectedSrc)}
                      >
                        PNG (lossless)
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => void downloadImage(selected, "jpeg", selectedSrc)}
                      >
                        JPEG (smaller)
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => void downloadImage(selected, "webp", selectedSrc)}
                      >
                        WebP
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        onClick={() => useSettingsDialogStore.getState().openDialog("library")}
                        className="justify-between gap-4 text-muted-foreground"
                      >
                        Info stamp
                        <span className="text-xs">{imageInfoStamp ? "On" : "Off"}</span>
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
                    leadingItems={editImageItem(selected)}
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
                <div className="absolute bottom-4 right-4 flex items-center gap-0.5 rounded-full bg-background/80 p-1 shadow-lg ring-1 ring-border backdrop-blur [&_[data-slot=button]]:border-0 [&_[data-slot=button]:focus-visible]:bg-muted">
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
                {/* Same icon as the Images nav item. */}
                <HugeiconsIcon icon={Image03Icon} className="size-12" strokeWidth={1.5} />
                <p className="text-sm">
                  {status?.loaded || external
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
                  selectedSrc || livePreviewSrc ? "inset-x-0 bottom-4" : "inset-0 items-center",
                )}
              >
                <div className="w-72 max-w-full rounded-xl bg-background/85 p-3 shadow-lg ring-1 ring-border backdrop-blur">
                  <ModelLoadDescription
                    // Drop the chat min-height: this floating card has no layout to stabilise.
                    className="min-h-0"
                    title={
                      compareRun && genDone != null
                        ? compareRun.title(genDone)
                        : genDone != null && count > 1
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
              // The rule spans the pane; only the thumbnail contents receive the 40px gutter.
              className="hover-scrollbar flex shrink-0 gap-2 overflow-x-auto border-t border-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] px-10 max-sm:px-5 py-3"
              onScroll={(e) => {
                // Near the right edge: pull the next older page (infinite scroll).
                const el = e.currentTarget;
                if (el.scrollWidth - el.scrollLeft - el.clientWidth < 400) void loadMore();
              }}
            >
              {/* In-progress generation: a placeholder tile at the front so past images stay browsable while
                  the new one renders. */}
              {busy === "generating" && (
                <div className="flex size-16 shrink-0 animate-pulse items-center justify-center overflow-hidden rounded-lg bg-muted/50 ring-2 ring-wash-30">
                  {livePreviewSrc ? (
                    <img src={livePreviewSrc} alt="" className="size-full object-cover" />
                  ) : (
                    <Spinner className="size-5 text-muted-foreground" />
                  )}
                </div>
              )}
              {/* The tile is a wrapper, not a button: the actions menu must be the select button's SIBLING,
                  since a button inside a button is invalid. data-image-id rides the wrapper so the
                  observer still sees it. */}
              {images.map((image) => (
                <div
                  key={image.id}
                  data-image-id={image.id}
                  {...stripReorder.tileProps(image.id)}
                  className={cn(
                    "group relative size-16 shrink-0",
                    // Fade the tile being dragged.
                    stripReorder.draggingId === image.id && "opacity-40",
                  )}
                >
                  {stripReorder.cue?.id === image.id && (
                    <StripDropLine edge={stripReorder.cue.edge} />
                  )}
                  <button
                    type="button"
                    onClick={() => setSelectedId(image.id)}
                    className="relative size-full overflow-hidden rounded-lg bg-muted/40 outline-none ring-1 ring-transparent transition-shadow hover:ring-border focus-visible:ring-2 focus-visible:ring-ring"
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
                      <span className="pointer-events-none absolute inset-0 rounded-lg border border-border bg-white/35 dark:border-[rgb(255_255_255_/_calc(0.25*var(--contrast-edge-gain,1)))] dark:bg-white/20" />
                    )}
                  </button>
                  {/* Pin marker and Unpin button, bottom-left so it clears the menu. */}
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
                      leadingItems={editImageItem(image)}
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
      {loadAnywayDialog}
    </div>
  );
}
