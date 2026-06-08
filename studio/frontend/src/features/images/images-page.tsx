// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SectionCard } from "@/components/section-card";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import {
  Add01Icon,
  Delete02Icon,
  FileImageIcon,
  PaintBrush02Icon,
  SparklesIcon,
  GpuIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  fetchDiffusionStatus,
  generateDiffusionImage,
  generateDiffusionVideo,
  loadDiffusionModel,
  unloadDiffusionModel,
  type DiffusionOffloadPolicy,
  type DiffusionGenerateResponse,
  type DiffusionReferencePreset,
  type DiffusionStatus,
  type DiffusionVideoGenerateResponse,
} from "./api";
import { type PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

// Curated short list of working diffusion GGUFs. Picked to span
// size + license so any GPU class has at least one viable option:
//   FLUX.2 klein base 4B -> low-VRAM Apache 2.0 baseline
//   FLUX.2 klein 4B      -> distilled 4-step baseline
//   FLUX.2 klein 9B      -> distilled 4-step higher quality baseline
//   FLUX.2 dev           -> 50-step gated baseline
//   FLUX.1 dev           -> older but widely tested gated baseline
//
// Filenames mirror the Hub canonical case (lowercase 'flux-2-klein-4b')
// and base_repo is set explicitly so the backend never falls back to the
// family default. The CLI on the backend can load anything supported by
// detect_family(); this list just keeps the picker compact for the v1 UI.
type CuratedDiffusionModel = {
  label: string;
  repo_id: string;
  default_gguf?: string;
  base_repo: string;
  text_encoder_gguf_repo?: string;
  text_encoder_gguf_filename?: string;
  text_encoder_gguf_component?: "text_encoder" | "text_encoder_2" | "text_encoder_3";
  family: string;
  default_steps: number;
  default_guidance: number;
  default_width?: number;
  default_height?: number;
  default_num_frames?: number;
  default_frame_rate?: number;
  notes: string;
};

const CURATED_MODELS: CuratedDiffusionModel[] = [
  {
    label: "FLUX.2 klein base 4B (Q4_K_M, Apache 2.0)",
    repo_id: "unsloth/FLUX.2-klein-base-4B-GGUF",
    default_gguf: "flux-2-klein-base-4b-Q4_K_M.gguf",
    base_repo: "black-forest-labs/FLUX.2-klein-base-4B",
    family: "flux.2-klein",
    default_steps: 50,
    default_guidance: 4.0,
    notes: "Apache 2.0, ungated. Official base settings: 50 steps, guidance 4.",
  },
  {
    label: "FLUX.2 klein 4B (Q4_K_M, distilled)",
    repo_id: "unsloth/FLUX.2-klein-4B-GGUF",
    default_gguf: "flux-2-klein-4b-Q4_K_M.gguf",
    // Distilled GGUF must pair with the distilled base, not the Base
    // checkpoint. The Hub model card for the GGUF lists
    // base_model: black-forest-labs/FLUX.2-klein-4B.
    base_repo: "black-forest-labs/FLUX.2-klein-4B",
    family: "flux.2-klein",
    default_steps: 4,
    default_guidance: 1.0,
    notes: "Distilled klein 4B. Official distilled settings: 4 steps, guidance 1.",
  },
  {
    label: "FLUX.2 klein 9B (Q4_K_M, gated)",
    repo_id: "unsloth/FLUX.2-klein-9B-GGUF",
    default_gguf: "flux-2-klein-9b-Q4_K_M.gguf",
    base_repo: "black-forest-labs/FLUX.2-klein-9B",
    family: "flux.2-klein",
    default_steps: 4,
    default_guidance: 1.0,
    notes: "Higher quality distilled. Official distilled settings: 4 steps, guidance 1.",
  },
  {
    label: "FLUX.2 dev (Q4_K_M + text UD-Q4, gated)",
    repo_id: "unsloth/FLUX.2-dev-GGUF",
    default_gguf: "flux2-dev-Q4_K_M.gguf",
    base_repo: "black-forest-labs/FLUX.2-dev",
    text_encoder_gguf_repo: "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
    text_encoder_gguf_filename: "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
    family: "flux.2",
    default_steps: 50,
    default_guidance: 4.0,
    notes: "Uses GGUF transformer + GGUF Mistral text encoder. Official dev settings: 50 steps, guidance 4.",
  },
  {
    label: "FLUX.1 dev (Q4_K_S, city96, gated)",
    repo_id: "city96/FLUX.1-dev-gguf",
    default_gguf: "flux1-dev-Q4_K_S.gguf",
    base_repo: "black-forest-labs/FLUX.1-dev",
    family: "flux.1",
    default_steps: 24,
    default_guidance: 3.5,
    notes: "12 GB VRAM. Older but widely tested. Requires HF access to FLUX.1 dev.",
  },
  {
    label: "Qwen-Image 2512 (Q4_K_M)",
    repo_id: "unsloth/Qwen-Image-2512-GGUF",
    default_gguf: "qwen-image-2512-Q4_K_M.gguf",
    base_repo: "Qwen/Qwen-Image-2512",
    family: "qwen-image-2512",
    default_steps: 50,
    default_guidance: 4.0,
    notes: "Text-to-image. Uses true CFG internally; leave negative prompt blank for the backend's Qwen default.",
  },
  {
    label: "Z-Image Turbo (Q4_K_M)",
    repo_id: "unsloth/Z-Image-Turbo-GGUF",
    default_gguf: "z-image-turbo-Q4_K_M.gguf",
    base_repo: "Tongyi-MAI/Z-Image-Turbo",
    family: "z-image-turbo",
    default_steps: 9,
    default_guidance: 0,
    notes: "Fast text-to-image Z-Image Turbo GGUF.",
  },
  {
    label: "Z-Image (Q4_K_M)",
    repo_id: "unsloth/Z-Image-GGUF",
    default_gguf: "z-image-Q4_K_M.gguf",
    base_repo: "Tongyi-MAI/Z-Image",
    family: "z-image",
    default_steps: 50,
    default_guidance: 4,
    notes: "Text-to-image Z-Image base GGUF.",
  },
];

const DEFAULT_PRESET = CURATED_MODELS[0];

const CURATED_VIDEO_MODELS: CuratedDiffusionModel[] = [
  {
    label: "LTX 2.3 Distilled",
    repo_id: "diffusers/LTX-2.3-Distilled-Diffusers",
    base_repo: "diffusers/LTX-2.3-Distilled-Diffusers",
    family: "ltx2-3-distilled",
    default_steps: 8,
    default_guidance: 1.0,
    default_width: 1536,
    default_height: 1024,
    default_num_frames: 121,
    default_frame_rate: 24,
    notes: "Fast LTX 2.3 distilled video pipeline. Default final output is 1536x1024, 121 frames.",
  },
  {
    label: "LTX 2.3 Base",
    repo_id: "diffusers/LTX-2.3-Diffusers",
    base_repo: "diffusers/LTX-2.3-Diffusers",
    family: "ltx2-3-base",
    default_steps: 30,
    default_guidance: 3.0,
    default_width: 1536,
    default_height: 1024,
    default_num_frames: 121,
    default_frame_rate: 24,
    notes: "Higher quality LTX 2.3 base video pipeline.",
  },
  {
    label: "Wan 2.2 T2V A14B",
    repo_id: "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    base_repo: "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    family: "wan2-2-t2v",
    default_steps: 40,
    default_guidance: 4.0,
    default_width: 1280,
    default_height: 720,
    default_num_frames: 81,
    default_frame_rate: 16,
    notes: "Wan text-to-video pipeline. Heavier than LTX; use lower resolutions for preview runs.",
  },
];

const DEFAULT_VIDEO_PRESET = CURATED_VIDEO_MODELS[0];

type DiffusionOffloadPolicySelection = "auto" | DiffusionOffloadPolicy;
type MediaKind = "image" | "video";
type GenerationTask = "auto" | "text_to_image" | "image_to_image" | "edit" | "inpaint";
type EnhanceMode = "off" | "upscale" | "creative_upscale" | "large_tiled";
type UpscaleEngine = "pixel" | "super_resolution";
type UpscaleMethod = "nearest" | "bilinear" | "bicubic" | "lanczos";
type ReferenceRole =
  | "edit_source"
  | "init_frame"
  | "style"
  | "object_identity"
  | "person_identity"
  | "structure"
  | "reference";
type ImageAsset = {
  id: string;
  name: string;
  b64: string;
  previewUrl: string;
  width: number;
  height: number;
};
type ReferenceImage = ImageAsset & {
  role: ReferenceRole;
};
type GeneratedMediaResult =
  | ({ kind: "image" } & DiffusionGenerateResponse)
  | ({ kind: "video" } & DiffusionVideoGenerateResponse);

const GENERATION_TASKS: Array<{ value: GenerationTask; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "text_to_image", label: "Text" },
  { value: "image_to_image", label: "Image" },
  { value: "edit", label: "Edit" },
  { value: "inpaint", label: "Inpaint" },
];

const REFERENCE_ROLE_LABELS: Record<ReferenceRole, string> = {
  edit_source: "Edit source",
  init_frame: "Initial frame",
  style: "Style",
  object_identity: "Object ID",
  person_identity: "Person ID",
  structure: "Structure",
  reference: "Reference",
};

function isReferenceRole(value: string): value is ReferenceRole {
  return value in REFERENCE_ROLE_LABELS;
}

function isGenerationTask(value: string): value is GenerationTask {
  return GENERATION_TASKS.some((task) => task.value === value);
}

function referencePresetSummary(preset: DiffusionReferencePreset): string {
  const parts = [
    preset.implementation,
    preset.confidence,
    preset.default_steps != null ? `${preset.default_steps} steps` : null,
    preset.default_guidance_scale != null
      ? `guidance ${preset.default_guidance_scale}`
      : null,
    preset.default_strength != null
      ? `strength ${preset.default_strength.toFixed(2)}`
      : null,
  ].filter(Boolean);
  return parts.join(" · ");
}

const ENHANCE_MODES: Array<{ value: EnhanceMode; label: string }> = [
  { value: "off", label: "Off" },
  { value: "upscale", label: "Upscale" },
  { value: "creative_upscale", label: "Creative" },
  { value: "large_tiled", label: "Large tiled" },
];

const OFFLOAD_POLICY_OPTIONS: Array<{
  value: DiffusionOffloadPolicySelection;
  label: string;
  description: string;
}> = [
  {
    value: "auto",
    label: "Auto",
    description: "Use Studio's recommended policy for this model.",
  },
  {
    value: "aggressive",
    label: "Aggressive",
    description: "Lowest VRAM: CPU-resident GGUF weights, CPU offload hooks, tiled VAE.",
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "CPU-resident GGUF weights without full CPU offload.",
  },
  {
    value: "less_aggressive",
    label: "Less aggressive",
    description: "Keep diffusion weights on GPU and offload GGUF text weights.",
  },
  {
    value: "none",
    label: "No offload",
    description: "Keep the pipeline resident on the selected device.",
  },
];

const RESOLUTION_PRESETS: Array<{ label: string; w: number; h: number }> = [
  { label: "Square 1024", w: 1024, h: 1024 },
  { label: "Square 768", w: 768, h: 768 },
  { label: "Square 512", w: 512, h: 512 },
  { label: "Portrait 832x1216", w: 832, h: 1216 },
  { label: "Landscape 1216x832", w: 1216, h: 832 },
];

function readImageAsset(file: File): Promise<ImageAsset> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const image = new Image();
      image.onerror = () => reject(new Error(`${file.name} is not a valid image`));
      image.onload = () => {
        resolve({
          id: `${file.name}-${file.lastModified}-${file.size}-${crypto.randomUUID()}`,
          name: file.name,
          b64: dataUrl,
          previewUrl: URL.createObjectURL(file),
          width: image.naturalWidth,
          height: image.naturalHeight,
        });
      };
      image.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

async function readReferenceImage(
  file: File,
  role: ReferenceRole = "edit_source",
): Promise<ReferenceImage> {
  return {
    ...(await readImageAsset(file)),
    role,
  };
}

function clampGenerationDimension(value: number | undefined): number | undefined {
  if (!value || !Number.isFinite(value)) return undefined;
  return Math.max(64, Math.min(2048, Math.round(value / 8) * 8));
}

export function ImagesPage() {
  const [status, setStatus] = useState<DiffusionStatus | null>(null);
  const [refreshingStatus, setRefreshingStatus] = useState(false);
  const [busy, setBusy] = useState<"idle" | "loading" | "unloading" | "generating">("idle");

  const [mediaKind, setMediaKind] = useState<MediaKind>("image");
  const [presetIndex, setPresetIndex] = useState(0);
  const [customRepoId, setCustomRepoId] = useState("");
  const [customGguf, setCustomGguf] = useState("");
  const [customBaseRepo, setCustomBaseRepo] = useState("");
  const [customTextEncoderRepo, setCustomTextEncoderRepo] = useState("");
  const [customTextEncoderGguf, setCustomTextEncoderGguf] = useState("");
  const [customFamily, setCustomFamily] = useState<string>("auto");
  const [controlNetEnabled, setControlNetEnabled] = useState(false);
  const [controlNetRepo, setControlNetRepo] = useState("");
  const [controlNetWeightName, setControlNetWeightName] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [hfToken, setHfToken] = useState("");
  const [offloadPolicy, setOffloadPolicy] = useState<DiffusionOffloadPolicySelection>("auto");

  const [prompt, setPrompt] = useState("a tiny ginger sloth coding in a sunlit treehouse, photorealistic");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [generationTask, setGenerationTask] = useState<GenerationTask>("auto");
  const [referenceImages, setReferenceImages] = useState<ReferenceImage[]>([]);
  const [maskImages, setMaskImages] = useState<ImageAsset[]>([]);
  const [paintedMaskDirty, setPaintedMaskDirty] = useState(false);
  const [maskBrushSize, setMaskBrushSize] = useState(64);
  const [controlImages, setControlImages] = useState<ReferenceImage[]>([]);
  const [strength, setStrength] = useState(0.6);
  const [controlScale, setControlScale] = useState(1.0);
  const [controlStart, setControlStart] = useState(0.0);
  const [controlEnd, setControlEnd] = useState(1.0);
  const [enhanceMode, setEnhanceMode] = useState<EnhanceMode>("off");
  const [upscaleEngine, setUpscaleEngine] = useState<UpscaleEngine>("pixel");
  const [upscaleModelRepo, setUpscaleModelRepo] = useState("");
  const [upscaleScale, setUpscaleScale] = useState(2);
  const [upscaleMethod, setUpscaleMethod] = useState<UpscaleMethod>("lanczos");
  const [creativeUpscalerRepo, setCreativeUpscalerRepo] = useState("");
  const [creativeUpscalerPipelineClass, setCreativeUpscalerPipelineClass] = useState("");
  const [creativeUpscaleSteps, setCreativeUpscaleSteps] = useState(12);
  const [creativeUpscaleGuidance, setCreativeUpscaleGuidance] = useState(4);
  const [tileSize, setTileSize] = useState(768);
  const [tileOverlap, setTileOverlap] = useState(64);
  const [vaeTiling, setVaeTiling] = useState<"auto" | "on" | "off">("auto");
  const [steps, setSteps] = useState(DEFAULT_PRESET.default_steps);
  const [guidance, setGuidance] = useState(DEFAULT_PRESET.default_guidance);
  const [resolutionIdx, setResolutionIdx] = useState(0);
  const [videoWidth, setVideoWidth] = useState(DEFAULT_VIDEO_PRESET.default_width);
  const [videoHeight, setVideoHeight] = useState(DEFAULT_VIDEO_PRESET.default_height);
  const [videoFrames, setVideoFrames] = useState(DEFAULT_VIDEO_PRESET.default_num_frames);
  const [videoFrameRate, setVideoFrameRate] = useState(DEFAULT_VIDEO_PRESET.default_frame_rate);
  const [seed, setSeed] = useState<string>("");

  const [results, setResults] = useState<GeneratedMediaResult[]>([]);
  const lastErrorRef = useRef<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const maskInputRef = useRef<HTMLInputElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskDrawingRef = useRef(false);
  const maskLastPointRef = useRef<{ x: number; y: number } | null>(null);
  const controlInputRef = useRef<HTMLInputElement | null>(null);
  const referenceImagesRef = useRef<ReferenceImage[]>([]);
  const maskImagesRef = useRef<ImageAsset[]>([]);
  const controlImagesRef = useRef<ReferenceImage[]>([]);

  const activeModelPresets =
    mediaKind === "video" ? CURATED_VIDEO_MODELS : CURATED_MODELS;
  const preset =
    activeModelPresets[presetIndex] ??
    (mediaKind === "video" ? DEFAULT_VIDEO_PRESET : DEFAULT_PRESET);
  const resolution = RESOLUTION_PRESETS[resolutionIdx];

  const applyPresetDefaults = useCallback((nextPreset: CuratedDiffusionModel) => {
    setSteps(nextPreset.default_steps);
    setGuidance(nextPreset.default_guidance);
    if (nextPreset.default_width && nextPreset.default_height) {
      setVideoWidth(nextPreset.default_width);
      setVideoHeight(nextPreset.default_height);
    }
    if (nextPreset.default_num_frames) {
      setVideoFrames(nextPreset.default_num_frames);
    }
    if (nextPreset.default_frame_rate) {
      setVideoFrameRate(nextPreset.default_frame_rate);
    }
  }, []);

  const handleMediaKindChange = useCallback(
    (nextKind: MediaKind) => {
      if (nextKind === mediaKind) return;
      const nextPreset =
        nextKind === "video" ? DEFAULT_VIDEO_PRESET : DEFAULT_PRESET;
      setMediaKind(nextKind);
      setUseCustom(false);
      setPresetIndex(0);
      setResolutionIdx(0);
      applyPresetDefaults(nextPreset);
    },
    [applyPresetDefaults, mediaKind],
  );

  // Round 30 P2 #12: split the fetch from the spinner toggle so the
  // mount + auto-poll effects can call the fetch without the
  // synchronous setRefreshingStatus(true) that tripped
  // react-hooks/set-state-in-effect.
  const fetchAndUpdateStatus = useCallback(async () => {
    try {
      const next = await fetchDiffusionStatus();
      setStatus(next);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (lastErrorRef.current !== msg) {
        lastErrorRef.current = msg;
        toast.error("Could not fetch image-model status", { description: msg });
      }
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    setRefreshingStatus(true);
    try {
      await fetchAndUpdateStatus();
    } finally {
      setRefreshingStatus(false);
    }
  }, [fetchAndUpdateStatus]);

  useEffect(() => {
    // Defer the mount fetch out of the synchronous effect body so the
    // setStatus call inside fetchAndUpdateStatus does not trip the
    // react-hooks/set-state-in-effect rule.
    const id = window.setTimeout(() => {
      void fetchAndUpdateStatus();
    }, 0);
    return () => window.clearTimeout(id);
  }, [fetchAndUpdateStatus]);

  // Round 27 P2: when the backend is mid-load (is_loading=true) the
  // status label froze at "Loading..." until the user clicked
  // Refresh. Auto-poll every 2 s while a load is in flight so the
  // UI tracks real backend progress.
  useEffect(() => {
    if (!status?.is_loading) return;
    const id = window.setInterval(() => {
      void fetchAndUpdateStatus();
    }, 2000);
    return () => window.clearInterval(id);
  }, [status?.is_loading, fetchAndUpdateStatus]);

  useEffect(() => {
    referenceImagesRef.current = referenceImages;
  }, [referenceImages]);

  useEffect(() => {
    maskImagesRef.current = maskImages;
  }, [maskImages]);

  useEffect(() => {
    controlImagesRef.current = controlImages;
  }, [controlImages]);

  const inpaintSource = useMemo(
    () =>
      referenceImages.find((image) => image.role === "edit_source") ??
      referenceImages[0] ??
      null,
    [referenceImages],
  );

  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !inpaintSource) return;
    canvas.width = inpaintSource.width;
    canvas.height = inpaintSource.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPaintedMaskDirty(false);
    maskDrawingRef.current = false;
    maskLastPointRef.current = null;
  }, [inpaintSource]);

  useEffect(() => {
    return () => {
      for (const image of referenceImagesRef.current) {
        URL.revokeObjectURL(image.previewUrl);
      }
      for (const image of controlImagesRef.current) {
        URL.revokeObjectURL(image.previewUrl);
      }
    };
  }, []);

  const handleLoad = useCallback(async () => {
    setBusy("loading");
    try {
      const repo = useCustom ? customRepoId.trim() : preset.repo_id;
      const gguf = useCustom ? customGguf.trim() || undefined : preset.default_gguf;
      // Custom mode lets the user pin a family explicitly because
      // detect_family is substring-based and exotic repo names (custom
      // fine-tunes, third-party mirrors) frequently fail to match.
      // "auto" leaves the override blank and lets the backend infer.
      const family = useCustom
        ? customFamily === "auto"
          ? undefined
          : customFamily
        : preset.family;
      // Always pass base_repo for curated entries; custom-repo mode
      // now also lets the user pin one because private / mirrored
      // GGUFs (e.g. a 9B klein transformer) would otherwise fall
      // back to the family-default 4B base and 500 on load. Empty
      // string still falls back to the backend's smart-base /
      // repo-id defaults.
      const baseRepo = useCustom
        ? customBaseRepo.trim() || undefined
        : preset.base_repo;
      const textEncoderRepo = useCustom
        ? customTextEncoderRepo.trim() || undefined
        : preset.text_encoder_gguf_repo;
      const textEncoderGguf = useCustom
        ? customTextEncoderGguf.trim() || undefined
        : preset.text_encoder_gguf_filename;
      const textEncoderComponent = useCustom
        ? undefined
        : preset.text_encoder_gguf_component;
      if (!repo) {
        toast.error("Pick a model first");
        return;
      }
      if (mediaKind === "image" && controlNetEnabled && !controlNetRepo.trim()) {
        toast.error("Enter a ControlNet repo first");
        return;
      }
      if (
        mediaKind === "image" &&
        enhanceMode === "creative_upscale" &&
        !creativeUpscalerRepo.trim()
      ) {
        toast.error("Enter a creative upscaler repo first");
        return;
      }
      if (
        mediaKind === "image" &&
        enhanceMode === "upscale" &&
        upscaleEngine === "super_resolution" &&
        !upscaleModelRepo.trim()
      ) {
        toast.error("Enter a super-resolution model repo first");
        return;
      }
      const next = await loadDiffusionModel({
        repo_id: repo,
        gguf_filename: gguf,
        base_repo: baseRepo,
        text_encoder_gguf_repo: textEncoderRepo,
        text_encoder_gguf_filename: textEncoderGguf,
        text_encoder_gguf_component: textEncoderComponent,
        controlnet_repo: mediaKind === "image" && controlNetEnabled
          ? controlNetRepo.trim() || undefined
          : undefined,
        controlnet_weight_name: mediaKind === "image" && controlNetEnabled
          ? controlNetWeightName.trim() || undefined
          : undefined,
        controlnet_conditioning_scale:
          mediaKind === "image" && controlNetEnabled ? controlScale : undefined,
        control_guidance_start:
          mediaKind === "image" && controlNetEnabled ? controlStart : undefined,
        control_guidance_end:
          mediaKind === "image" && controlNetEnabled ? controlEnd : undefined,
        upscaler_repo:
          mediaKind === "image" && enhanceMode === "creative_upscale"
            ? creativeUpscalerRepo.trim() || undefined
            : mediaKind === "image" &&
                enhanceMode === "upscale" &&
                upscaleEngine === "super_resolution"
              ? upscaleModelRepo.trim() || undefined
            : undefined,
        upscaler_mode:
          mediaKind === "image" && enhanceMode === "creative_upscale"
            ? "diffusion"
            : mediaKind === "image" &&
                enhanceMode === "upscale" &&
                upscaleEngine === "super_resolution"
              ? "super_resolution"
              : undefined,
        upscaler_pipeline_class:
          mediaKind === "image" && enhanceMode === "creative_upscale"
            ? creativeUpscalerPipelineClass.trim() || undefined
            : undefined,
        upscaler_scale:
          mediaKind === "image" &&
          (enhanceMode === "creative_upscale" ||
            (enhanceMode === "upscale" && upscaleEngine === "super_resolution"))
            ? upscaleScale
            : undefined,
        family,
        hf_token: hfToken.trim() || undefined,
        runtime: {
          memory_mode: offloadPolicy === "auto" ? "auto" : "manual",
          offload_policy: offloadPolicy === "auto" ? undefined : offloadPolicy,
          attention_backend: "auto",
          torch_compile: "auto",
        },
        parameters: {
          width: mediaKind === "video" ? videoWidth : resolution.w,
          height: mediaKind === "video" ? videoHeight : resolution.h,
          num_frames: mediaKind === "video" ? videoFrames : undefined,
          frame_rate: mediaKind === "video" ? videoFrameRate : undefined,
          batch_size: 1,
          guidance_scale: guidance,
          enhance:
            mediaKind === "image"
              ? {
                  mode: enhanceMode,
                  tiling: {
                    enabled: enhanceMode === "large_tiled" ? "on" : "auto",
                    tile_size: tileSize,
                    overlap: tileOverlap,
                    vae_decode: vaeTiling,
                  },
                }
              : undefined,
        },
      });
      setStatus(next);
      toast.success(`Loaded ${mediaKind} model`, { description: next.repo_id ?? undefined });
    } catch (err) {
      toast.error(`Failed to load ${mediaKind} model`, {
        description: err instanceof Error ? err.message : String(err),
      });
      // Backend clears its old pipeline before allocating the new one;
      // a failed swap leaves status.is_loaded=false while our local
      // copy still says loaded. Re-fetch so Generate disables and the
      // user does not see a stale "Loaded:" label.
      await refreshStatus();
    } finally {
      setBusy("idle");
    }
  }, [
    useCustom,
    mediaKind,
    customRepoId,
    customGguf,
    customBaseRepo,
    customTextEncoderRepo,
    customTextEncoderGguf,
    customFamily,
    controlNetEnabled,
    controlNetRepo,
    controlNetWeightName,
    controlScale,
    controlStart,
    controlEnd,
    enhanceMode,
    upscaleEngine,
    upscaleModelRepo,
    creativeUpscalerRepo,
    creativeUpscalerPipelineClass,
    upscaleScale,
    tileSize,
    tileOverlap,
    vaeTiling,
    preset,
    resolution,
    videoWidth,
    videoHeight,
    videoFrames,
    videoFrameRate,
    guidance,
    hfToken,
    offloadPolicy,
    refreshStatus,
  ]);

  const handleUnload = useCallback(async () => {
    setBusy("unloading");
    try {
      await unloadDiffusionModel();
      await refreshStatus();
    } catch (err) {
      toast.error("Failed to unload image model", {
        description: err instanceof Error ? err.message : String(err),
      });
      // Round 27 P2: a partial unload (subprocess refused to terminate,
      // 503 from the backend) used to leave the UI showing the old
      // "Loaded:" label even though the backend state was half torn
      // down. Refresh so the button states match reality (mirrors
      // handleLoad above which always re-fetches on catch).
      await refreshStatus();
    } finally {
      setBusy("idle");
    }
  }, [refreshStatus]);

  const handleReferenceFiles = useCallback(async (files: FileList | File[]) => {
    const selected = Array.from(files).filter((file) => file.type.startsWith("image/"));
    if (selected.length === 0) return;
    try {
      const defaultRole: ReferenceRole = mediaKind === "video" ? "init_frame" : "edit_source";
      const next = await Promise.all(
        selected.slice(0, 8).map((file) => readReferenceImage(file, defaultRole)),
      );
      setReferenceImages((prev) => {
        const merged = [...prev, ...next].slice(0, 8);
        for (const image of [...prev, ...next].slice(8)) {
          URL.revokeObjectURL(image.previewUrl);
        }
        return merged;
      });
    } catch (err) {
      toast.error("Could not add reference image", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (imageInputRef.current) {
        imageInputRef.current.value = "";
      }
    }
  }, [mediaKind]);

  const handleControlFiles = useCallback(async (files: FileList | File[]) => {
    const selected = Array.from(files).filter((file) => file.type.startsWith("image/"));
    if (selected.length === 0) return;
    try {
      const next = await Promise.all(
        selected.slice(0, 8).map((file) => readReferenceImage(file)),
      );
      setControlImages((prev) => {
        const merged = [...prev, ...next].slice(0, 8);
        for (const image of [...prev, ...next].slice(8)) {
          URL.revokeObjectURL(image.previewUrl);
        }
        return merged;
      });
    } catch (err) {
      toast.error("Could not add control image", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (controlInputRef.current) {
        controlInputRef.current.value = "";
      }
    }
  }, []);

  const handleMaskFiles = useCallback(async (files: FileList | File[]) => {
    const selected = Array.from(files).filter((file) => file.type.startsWith("image/"));
    if (selected.length === 0) return;
    try {
      const next = await Promise.all(selected.slice(0, 4).map(readImageAsset));
      setMaskImages((prev) => {
        const merged = [...prev, ...next].slice(0, 4);
        for (const image of [...prev, ...next].slice(4)) {
          URL.revokeObjectURL(image.previewUrl);
        }
        return merged;
      });
    } catch (err) {
      toast.error("Could not add mask image", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (maskInputRef.current) {
        maskInputRef.current.value = "";
      }
    }
  }, []);

  const removeReferenceImage = useCallback((id: string) => {
    setReferenceImages((prev) => {
      const removed = prev.find((image) => image.id === id);
      if (removed) URL.revokeObjectURL(removed.previewUrl);
      return prev.filter((image) => image.id !== id);
    });
  }, []);

  const updateReferenceRole = useCallback((id: string, role: ReferenceRole) => {
    setReferenceImages((prev) =>
      prev.map((image) => (image.id === id ? { ...image, role } : image)),
    );
  }, []);

  const addGeneratedImageAsReference = useCallback(
    (
      result: DiffusionGenerateResponse,
      imageB64: string,
      imageIndex: number,
      role: ReferenceRole,
    ) => {
      const dataUrl = `data:${result.image_mime};base64,${imageB64}`;
      const asset: ReferenceImage = {
        id: `generated-${Date.now()}-${crypto.randomUUID()}`,
        name:
          imageIndex > 0
            ? `generated-output-${imageIndex + 1}.png`
            : "generated-output.png",
        b64: dataUrl,
        previewUrl: dataUrl,
        width: result.width,
        height: result.height,
        role,
      };
      setReferenceImages((prev) => [asset, ...prev].slice(0, 8));
      if (role === "init_frame") {
        setMediaKind("video");
        const nextWidth = clampGenerationDimension(result.width);
        const nextHeight = clampGenerationDimension(result.height);
        if (nextWidth) setVideoWidth(nextWidth);
        if (nextHeight) setVideoHeight(nextHeight);
        toast.success("Added generated image as the video input frame");
      } else {
        setMediaKind("image");
        setGenerationTask("edit");
        toast.success("Added generated image as an image input");
      }
    },
    [],
  );

  const removeMaskImage = useCallback((id: string) => {
    setMaskImages((prev) => {
      const removed = prev.find((image) => image.id === id);
      if (removed) URL.revokeObjectURL(removed.previewUrl);
      return prev.filter((image) => image.id !== id);
    });
  }, []);

  const clearPaintedMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    maskDrawingRef.current = false;
    maskLastPointRef.current = null;
    setPaintedMaskDirty(false);
  }, []);

  const drawMaskPoint = useCallback(
    (event: PointerEvent<HTMLCanvasElement>) => {
      const canvas = maskCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
      const last = maskLastPointRef.current;
      ctx.save();
      ctx.strokeStyle = "white";
      ctx.fillStyle = "white";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = maskBrushSize;
      if (last) {
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(x, y, maskBrushSize / 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      maskLastPointRef.current = { x, y };
      setPaintedMaskDirty(true);
    },
    [maskBrushSize],
  );

  const startMaskStroke = useCallback(
    (event: PointerEvent<HTMLCanvasElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId);
      maskDrawingRef.current = true;
      maskLastPointRef.current = null;
      drawMaskPoint(event);
    },
    [drawMaskPoint],
  );

  const continueMaskStroke = useCallback(
    (event: PointerEvent<HTMLCanvasElement>) => {
      if (!maskDrawingRef.current) return;
      drawMaskPoint(event);
    },
    [drawMaskPoint],
  );

  const endMaskStroke = useCallback((event: PointerEvent<HTMLCanvasElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    maskDrawingRef.current = false;
    maskLastPointRef.current = null;
  }, []);

  const applyReferencePreset = useCallback(
    (preset: DiffusionReferencePreset) => {
      if (isGenerationTask(preset.task)) {
        setGenerationTask(preset.task);
      }
      if (preset.default_steps != null) {
        setSteps(preset.default_steps);
      }
      if (preset.default_guidance_scale != null) {
        setGuidance(preset.default_guidance_scale);
      }
      if (preset.default_strength != null) {
        setStrength(preset.default_strength);
      }
      toast.success("Reference preset applied", {
        description: referencePresetSummary(preset),
      });
    },
    [],
  );

  const removeControlImage = useCallback((id: string) => {
    setControlImages((prev) => {
      const removed = prev.find((image) => image.id === id);
      if (removed) URL.revokeObjectURL(removed.previewUrl);
      return prev.filter((image) => image.id !== id);
    });
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    if (
      mediaKind === "image" &&
      enhanceMode === "creative_upscale" &&
      status?.upscaler?.mode !== "diffusion"
    ) {
      toast.error("Creative upscale needs a loaded upscaler", {
        description: "Enter a creative upscaler repo, reload the model, then generate.",
      });
      return;
    }
    if (
      mediaKind === "image" &&
      enhanceMode === "upscale" &&
      upscaleEngine === "super_resolution" &&
      status?.upscaler?.mode !== "super_resolution"
    ) {
      toast.error("Model upscale needs a loaded upscaler", {
        description: "Enter a super-resolution model repo, reload the model, then generate.",
      });
      return;
    }
    setBusy("generating");
    try {
      // Reject non-integer seeds and clamp to the [-2^63, 2^64 - 1]
      // range the backend's torch.Generator can actually pack. JSON
      // serialises BigInts as plain integers, so we keep the wire
      // format compatible and avoid the Number(seed) precision loss
      // (>= 2^53 silently rounds, producing a different image than
      // the seed the user typed). When the seed fits a safe integer
      // it goes through unchanged; larger seeds ride along as their
      // BigInt-derived string via the wire-format BigInt JSON helper
      // in the api layer.
      const seedStr = seed.trim();
      let parsedSeed: number | bigint | undefined;
      if (seedStr) {
        if (!/^-?\d+$/.test(seedStr)) {
          toast.error("Seed must be an integer");
          return;
        }
        let big: bigint;
        try {
          big = BigInt(seedStr);
        } catch {
          toast.error("Seed must be an integer");
          return;
        }
        const SEED_MIN = -(BigInt(2) ** BigInt(63));
        const SEED_MAX = BigInt(2) ** BigInt(64) - BigInt(1);
        if (big < SEED_MIN || big > SEED_MAX) {
          toast.error(
            "Seed must be in [-2^63, 2^64 - 1] (the torch.Generator range)",
          );
          return;
        }
        // Use a plain Number when it fits a safe integer so the
        // existing api.ts JSON serialiser does not break on BigInt;
        // otherwise pass the BigInt and let api.ts emit it as a JSON
        // number via a custom replacer.
        const SAFE_MAX = BigInt(Number.MAX_SAFE_INTEGER);
        const SAFE_MIN = -SAFE_MAX;
        parsedSeed = big >= SAFE_MIN && big <= SAFE_MAX ? Number(big) : big;
      }
      if (mediaKind === "video") {
        const videoInputs = referenceImages.map((image) => ({
          id: image.id,
          type: "image" as const,
          role: image.role,
          mime: "image/png",
          b64: image.b64,
        }));
        const out = await generateDiffusionVideo({
          prompt,
          negative_prompt: negativePrompt.trim() || undefined,
          inputs: videoInputs.length > 0 ? videoInputs : undefined,
          num_inference_steps: steps,
          guidance_scale: guidance,
          width: videoWidth,
          height: videoHeight,
          num_frames: videoFrames,
          frame_rate: videoFrameRate,
          seed: parsedSeed,
        });
        const result: GeneratedMediaResult = { ...out, kind: "video" };
        setResults((prev) => [result, ...prev].slice(0, 12));
        return;
      }
      const activeReferenceImages =
        generationTask === "text_to_image" ? [] : referenceImages;
      const imageB64s = activeReferenceImages.map((image) => image.b64);
      const structuredInputs = activeReferenceImages.map((image) => ({
        id: image.id,
        type: "image" as const,
        role: image.role,
        mime: "image/png",
        b64: image.b64,
      }));
      const activeMaskImages = generationTask === "inpaint" ? maskImagesRef.current : [];
      const maskInputs = activeMaskImages.map((image) => ({
        id: image.id,
        type: "image" as const,
        role: "mask",
        mime: "image/png",
        b64: image.b64,
      }));
      if (generationTask === "inpaint") {
        if (imageB64s.length === 0) {
          toast.error("Inpaint needs a source image");
          return;
        }
        const canvas = maskCanvasRef.current;
        if (paintedMaskDirty && canvas) {
          maskInputs.push({
            id: "painted-mask",
            type: "image" as const,
            role: "mask",
            mime: "image/png",
            b64: canvas.toDataURL("image/png"),
          });
        }
        if (maskInputs.length === 0) {
          toast.error("Inpaint needs a mask");
          return;
        }
      }
      const activeControlImages =
        controlNetEnabled && status?.controlnet ? controlImages : [];
      const controlB64s = activeControlImages.map((image) => image.b64);
      const enhance =
        enhanceMode === "off"
          ? undefined
          : {
              mode: enhanceMode,
              upscale: {
                enabled: true,
                mode:
                  enhanceMode === "creative_upscale"
                    ? ("diffusion" as const)
                    : enhanceMode === "upscale" && upscaleEngine === "super_resolution"
                      ? ("super_resolution" as const)
                    : ("pixel" as const),
                scale: upscaleScale,
                method: upscaleMethod,
                tile_size: tileSize,
                tile_overlap: tileOverlap,
                num_inference_steps:
                  enhanceMode === "creative_upscale"
                    ? creativeUpscaleSteps
                    : undefined,
                guidance_scale:
                  enhanceMode === "creative_upscale"
                    ? creativeUpscaleGuidance
                    : undefined,
              },
              tiling: {
                enabled: enhanceMode === "large_tiled" ? ("on" as const) : ("auto" as const),
                tile_size: tileSize,
                overlap: tileOverlap,
                vae_decode: vaeTiling,
              },
            };
      const out = await generateDiffusionImage({
        prompt,
        negative_prompt: negativePrompt.trim() || undefined,
        task: generationTask,
        inputs:
          structuredInputs.length > 0 || maskInputs.length > 0
            ? [...structuredInputs, ...maskInputs]
            : undefined,
        control_image_b64: controlB64s.length === 1 ? controlB64s[0] : undefined,
        control_images_b64: controlB64s.length > 1 ? controlB64s : undefined,
        strength:
          imageB64s.length > 0 &&
          generationTask !== "text_to_image" &&
          generationTask !== "edit"
            ? strength
            : undefined,
        controlnet_conditioning_scale:
          controlB64s.length > 0 ? controlScale : undefined,
        control_guidance_start: controlB64s.length > 0 ? controlStart : undefined,
        control_guidance_end: controlB64s.length > 0 ? controlEnd : undefined,
        num_inference_steps: steps,
        guidance_scale: guidance,
        width: resolution.w,
        height: resolution.h,
        seed: parsedSeed,
        enhance,
      });
      const result: GeneratedMediaResult = { ...out, kind: "image" };
      setResults((prev) => [result, ...prev].slice(0, 12));
    } catch (err) {
      toast.error(`${mediaKind === "video" ? "Video" : "Image"} generation failed`, {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setBusy("idle");
    }
  }, [
    prompt,
    negativePrompt,
    mediaKind,
    generationTask,
    referenceImages,
    paintedMaskDirty,
    controlImages,
    controlNetEnabled,
    status,
    strength,
    controlScale,
    controlStart,
    controlEnd,
    enhanceMode,
    upscaleEngine,
    upscaleScale,
    upscaleMethod,
    creativeUpscaleSteps,
    creativeUpscaleGuidance,
    tileSize,
    tileOverlap,
    vaeTiling,
    steps,
    guidance,
    resolution,
    videoWidth,
    videoHeight,
    videoFrames,
    videoFrameRate,
    seed,
  ]);

  const statusLabel = useMemo(() => {
    if (!status) return refreshingStatus ? "Checking..." : "Not loaded";
    if (status.is_loading) return "Loading...";
    if (status.is_loaded) {
      const dev = status.device ? ` on ${status.device}` : "";
      return `Loaded: ${status.repo_id ?? "(unknown)"} (${status.family ?? "unknown"})${dev}`;
    }
    return "Not loaded";
  }, [status, refreshingStatus]);

  // FLUX.2 / FLUX.2 klein pipelines do NOT accept negative_prompt and
  // would 500 if we sent one through. The backend strips the field
  // defensively but hiding it client-side keeps the UI honest.
  // Round 29 P2 #12: also honour the user-picked customFamily when no
  // model is loaded yet, so a Custom HF repo with family flux.2 /
  // flux.2-klein hides the negative-prompt field correctly.
  const supportsNegativePrompt = useMemo(() => {
    const family = status?.family;
    if (!family) {
      let candidate: string | undefined;
      if (useCustom) {
        candidate = customFamily === "auto" ? undefined : customFamily;
      } else {
        candidate = preset.family;
      }
      if (!candidate) return true;
      return !candidate.startsWith("flux.2");
    }
    return !family.startsWith("flux.2");
  }, [status, useCustom, customFamily, preset.family]);

  const activeFamilyName = status?.family || (useCustom ? customFamily : preset.family);
  const activeFamily = useMemo(() => {
    if (!status?.supported_families) return null;
    return (
      status.supported_families.find((family) => family.name === activeFamilyName) ??
      null
    );
  }, [status, activeFamilyName]);
  const referencePresets = useMemo(
    () =>
      status?.sampling_contract?.reference_presets?.length
        ? status.sampling_contract.reference_presets
        : (activeFamily?.reference_presets ?? []),
    [status, activeFamily],
  );
  const supportedGenerationTasks = useMemo(() => {
    const contractTasks = status?.sampling_contract?.image_tasks;
    const familyTasks = activeFamily?.image_tasks;
    const sourceTasks = contractTasks?.length ? contractTasks : familyTasks;
    if (!sourceTasks?.length) return null;
    const tasks = new Set<GenerationTask>(["auto"]);
    for (const task of sourceTasks) {
      if (isGenerationTask(task)) tasks.add(task);
    }
    return tasks;
  }, [status?.sampling_contract?.image_tasks, activeFamily?.image_tasks]);
  useEffect(() => {
    if (supportedGenerationTasks != null && !supportedGenerationTasks.has(generationTask)) {
      const id = window.setTimeout(() => setGenerationTask("auto"), 0);
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [supportedGenerationTasks, generationTask]);
  const referenceRoleOptions = useMemo(() => {
    if (mediaKind === "video") {
      return ["init_frame", "reference"] as ReferenceRole[];
    }
    const roles = referencePresets
      .map((preset) => preset.role as ReferenceRole)
      .filter((role): role is ReferenceRole => isReferenceRole(role));
    const unique = Array.from(new Set<ReferenceRole>(roles));
    return unique.length > 0 ? unique : (["edit_source", "reference"] as ReferenceRole[]);
  }, [mediaKind, referencePresets]);
  const activeReferencePresets = useMemo(() => {
    if (referenceImages.length === 0) return [];
    const activeRoles = new Set(referenceImages.map((image) => image.role));
    const seen = new Set<string>();
    return referencePresets.filter((preset) => {
      if (!isReferenceRole(preset.role) || !activeRoles.has(preset.role)) return false;
      const key = `${preset.role}:${preset.id}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [referencePresets, referenceImages]);

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4 sm:p-6">
      <SectionCard
        icon={<HugeiconsIcon icon={GpuIcon} className="size-5" strokeWidth={1.5} />}
        title="Local generation"
        description={
          "Run diffusion image and video models from Hugging Face on your own GPU."
        }
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <Label>Media</Label>
            <div className="grid grid-cols-2 rounded-md border border-border bg-muted/30 p-1">
              {(["image", "video"] as MediaKind[]).map((kind) => (
                <button
                  key={kind}
                  type="button"
                  onClick={() => handleMediaKindChange(kind)}
                  className={[
                    "min-h-8 px-4 text-xs font-medium capitalize transition",
                    mediaKind === kind
                      ? "rounded bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground",
                  ].join(" ")}
                >
                  {kind}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <Label>Model</Label>
            <Select
              value={useCustom ? "custom" : String(presetIndex)}
              onValueChange={(v) => {
                if (v === "custom") {
                  setUseCustom(true);
                } else {
                  setUseCustom(false);
                  const idx = Number(v);
                  const nextPreset = activeModelPresets[idx] ?? preset;
                  setPresetIndex(idx);
                  applyPresetDefaults(nextPreset);
                  setResolutionIdx(0);
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Pick a model" />
              </SelectTrigger>
              <SelectContent>
                {activeModelPresets.map((m, idx) => (
                  <SelectItem key={m.repo_id} value={String(idx)}>
                    {m.label}
                  </SelectItem>
                ))}
                <SelectItem value="custom">Custom HF repo...</SelectItem>
              </SelectContent>
            </Select>
            {!useCustom && (
              <p className="text-xs text-muted-foreground">{preset.notes}</p>
            )}
          </div>

          {useCustom && (
            <div className="flex flex-col gap-2">
              <Label>HF repo id</Label>
              <Input
                value={customRepoId}
                onChange={(e) => setCustomRepoId(e.target.value)}
                placeholder={
                  mediaKind === "video"
                    ? "diffusers/LTX-2.3-Distilled-Diffusers"
                    : "unsloth/FLUX.2-klein-4B-GGUF"
                }
              />
              <Label>GGUF filename (optional)</Label>
              <Input
                value={customGguf}
                onChange={(e) => setCustomGguf(e.target.value)}
                placeholder={
                  mediaKind === "video"
                    ? "optional video transformer GGUF"
                    : "FLUX.2-klein-4B-Q4_K_S.gguf"
                }
              />
              <Label>Base diffusers repo (optional)</Label>
              <Input
                value={customBaseRepo}
                onChange={(e) => setCustomBaseRepo(e.target.value)}
                placeholder={
                  mediaKind === "video"
                    ? "diffusers/LTX-2.3-Distilled-Diffusers"
                    : "black-forest-labs/FLUX.2-klein-9B"
                }
              />
              <p className="text-xs text-muted-foreground">
                {"Optional. Defaults to the family base. Set this when "}
                {"your GGUF expects a non-default base (for example a 9B "}
                {"transformer that would otherwise fall back to a 4B base)."}
              </p>
              <Label>Text encoder GGUF repo (FLUX.2 optional)</Label>
              <Input
                value={customTextEncoderRepo}
                onChange={(e) => setCustomTextEncoderRepo(e.target.value)}
                placeholder="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
              />
              <Label>Text encoder GGUF filename (FLUX.2 optional)</Label>
              <Input
                value={customTextEncoderGguf}
                onChange={(e) => setCustomTextEncoderGguf(e.target.value)}
                placeholder="Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
              />
              <p className="text-xs text-muted-foreground">
                {"Only needed when loading FLUX.2 dev with a GGUF text encoder. "}
                {"Leave blank for non-FLUX.2 models or standard diffusers text encoders."}
              </p>
              <Label>Pipeline family (override)</Label>
              <Select
                value={customFamily}
                onValueChange={setCustomFamily}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect from repo id</SelectItem>
                  <SelectItem value="flux.2-klein">FLUX.2 klein</SelectItem>
                  <SelectItem value="flux.2">FLUX.2</SelectItem>
                  <SelectItem value="flux.1">FLUX.1</SelectItem>
                  <SelectItem value="qwen-image">Qwen-Image</SelectItem>
                  <SelectItem value="qwen-image-2512">Qwen-Image 2512</SelectItem>
                  <SelectItem value="qwen-image-edit">Qwen-Image Edit</SelectItem>
                  <SelectItem value="qwen-image-edit-2509">Qwen-Image Edit 2509</SelectItem>
                  <SelectItem value="qwen-image-edit-2511">Qwen-Image Edit 2511</SelectItem>
                  <SelectItem value="qwen-image-layered">Qwen-Image Layered</SelectItem>
                  <SelectItem value="z-image">Z-Image</SelectItem>
                  <SelectItem value="z-image-turbo">Z-Image Turbo</SelectItem>
                  <SelectItem value="ernie-image">ERNIE-Image</SelectItem>
                  <SelectItem value="ernie-image-turbo">ERNIE-Image Turbo</SelectItem>
                  <SelectItem value="stable-diffusion-3">Stable Diffusion 3</SelectItem>
                  <SelectItem value="stable-diffusion-xl">Stable Diffusion XL</SelectItem>
                  <SelectItem value="ltx2-3-distilled">LTX 2.3 Distilled</SelectItem>
                  <SelectItem value="ltx2-3-base">LTX 2.3 Base</SelectItem>
                  <SelectItem value="wan2-2-t2v">Wan 2.2 T2V</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {"Set this when your repo name does not contain "}
                {"a recognised family substring (e.g. private fine-tunes)."}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <Label>Hugging Face token (only for gated repos)</Label>
            <Input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              autoComplete="off"
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label>VRAM policy</Label>
            <Select
              value={offloadPolicy}
              onValueChange={(value) => setOffloadPolicy(value as DiffusionOffloadPolicySelection)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {OFFLOAD_POLICY_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {OFFLOAD_POLICY_OPTIONS.find((option) => option.value === offloadPolicy)?.description}
            </p>
          </div>

          {mediaKind === "image" ? (
          <div className="flex flex-col gap-2 border-t border-border pt-3">
            <div className="flex items-center justify-between gap-3">
              <Label htmlFor="diffusion-controlnet-enabled">ControlNet</Label>
              <Switch
                id="diffusion-controlnet-enabled"
                checked={controlNetEnabled}
                onCheckedChange={setControlNetEnabled}
              />
            </div>
            {controlNetEnabled ? (
              <>
                <Label htmlFor="diffusion-controlnet-repo">ControlNet repo</Label>
                <Input
                  id="diffusion-controlnet-repo"
                  value={controlNetRepo}
                  onChange={(e) => setControlNetRepo(e.target.value)}
                  placeholder="InstantX/Qwen-Image-ControlNet-Union"
                />
                <Label htmlFor="diffusion-controlnet-weight">Weight filename</Label>
                <Input
                  id="diffusion-controlnet-weight"
                  value={controlNetWeightName}
                  onChange={(e) => setControlNetWeightName(e.target.value)}
                  placeholder="optional: controlnet.safetensors"
                />
                <p className="text-xs text-muted-foreground">
                  {"Reload the model after changing this. The adapter shares "}
                  {"the loaded pipeline and follows the selected VRAM policy."}
                </p>
              </>
            ) : null}
            {status?.controlnet ? (
              <p className="text-xs text-muted-foreground">
                {`Loaded ControlNet: ${status.controlnet.repo}`}
              </p>
            ) : null}
          </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-2">
            <Button
              onClick={handleLoad}
              disabled={busy !== "idle"}
              data-testid="diffusion-load"
            >
              {busy === "loading" ? <Spinner className="mr-2 size-4" /> : null}
              {`Load ${mediaKind} model`}
            </Button>
            <Button
              variant="outline"
              onClick={handleUnload}
              disabled={busy !== "idle" || !status?.is_loaded}
              data-testid="diffusion-unload"
            >
              Unload
            </Button>
            <Button
              variant="ghost"
              onClick={() => void refreshStatus()}
              disabled={refreshingStatus}
            >
              Refresh status
            </Button>
            <span
              className="ml-auto text-xs text-muted-foreground"
              data-testid="diffusion-status"
            >
              {statusLabel}
            </span>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        icon={<HugeiconsIcon icon={PaintBrush02Icon} className="size-5" strokeWidth={1.5} />}
        title={mediaKind === "video" ? "Video prompt" : "Prompt"}
        description="The pipeline runs on the GPU you launched Unsloth Studio on."
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <Label htmlFor="diffusion-prompt">Prompt</Label>
            <Textarea
              id="diffusion-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              data-testid="diffusion-prompt"
            />
          </div>
          {supportsNegativePrompt ? (
            <div className="flex flex-col gap-1">
              <Label htmlFor="diffusion-negative">Negative prompt (optional)</Label>
              <Textarea
                id="diffusion-negative"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                rows={2}
              />
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {"FLUX.2 and FLUX.2 klein do not accept a negative prompt. "}
              {"Steer the output via the main prompt instead."}
            </p>
          )}

          {mediaKind === "image" ? (
          <>
          <div className="flex flex-col gap-3 border-t border-border pt-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <Label>Mode</Label>
              <div className="grid grid-cols-5 rounded-md border border-border bg-muted/30 p-1">
                {GENERATION_TASKS.map((task) => {
                  const disabled =
                    supportedGenerationTasks != null &&
                    !supportedGenerationTasks.has(task.value);
                  return (
                    <button
                      key={task.value}
                      type="button"
                      onClick={() => {
                        if (!disabled) setGenerationTask(task.value);
                      }}
                      disabled={disabled}
                      aria-disabled={disabled}
                      title={disabled ? "This loaded model does not expose that image task." : undefined}
                      className={[
                        "min-h-8 px-2 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-40",
                        generationTask === task.value
                          ? "rounded bg-background text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground",
                      ].join(" ")}
                    >
                      {task.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(event) => {
                if (event.target.files) void handleReferenceFiles(event.target.files);
              }}
            />
            <input
              ref={controlInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(event) => {
                if (event.target.files) void handleControlFiles(event.target.files);
              }}
            />
            <input
              ref={maskInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(event) => {
                if (event.target.files) void handleMaskFiles(event.target.files);
              }}
            />
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <Label>Reference images</Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => imageInputRef.current?.click()}
                >
                  <HugeiconsIcon icon={Add01Icon} className="mr-2 size-4" strokeWidth={1.5} />
                  Add
                </Button>
              </div>
              {referenceImages.length > 0 ? (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {referenceImages.map((image) => (
                    <div
                      key={image.id}
                      className="relative overflow-hidden rounded-md border border-border bg-muted/20"
                    >
                      <img
                        src={image.previewUrl}
                        alt={image.name}
                        className="aspect-square w-full object-cover"
                      />
                      <button
                        type="button"
                        aria-label={`Remove ${image.name}`}
                        onClick={() => removeReferenceImage(image.id)}
                        className="absolute right-1 top-1 grid size-7 place-items-center rounded bg-background/90 text-foreground shadow-sm hover:bg-background"
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" strokeWidth={1.5} />
                      </button>
                      <div className="flex items-center gap-1 px-2 py-1 text-[11px] text-muted-foreground">
                        <HugeiconsIcon icon={FileImageIcon} className="size-3.5" strokeWidth={1.5} />
                        <span className="truncate">
                          {image.width}x{image.height}
                        </span>
                      </div>
                      <div className="border-t border-border px-1.5 py-1.5">
                        <Select
                          value={image.role}
                          onValueChange={(value) =>
                            updateReferenceRole(image.id, value as ReferenceRole)
                          }
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {referenceRoleOptions.map((role) => (
                              <SelectItem key={role} value={role}>
                                {REFERENCE_ROLE_LABELS[role]}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-border bg-muted/20 text-sm text-muted-foreground hover:bg-muted/30 hover:text-foreground"
                >
                  <HugeiconsIcon icon={FileImageIcon} className="mr-2 size-4" strokeWidth={1.5} />
                  Add a source image
                </button>
              )}
              {activeReferencePresets.length > 0 ? (
                <div className="flex flex-col gap-2 border-t border-border pt-2">
                  {activeReferencePresets.map((preset) => (
                    <div
                      key={`${preset.role}:${preset.id}`}
                      className="flex items-center justify-between gap-3"
                    >
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium">
                          {preset.label}
                        </div>
                        <div className="truncate text-xs text-muted-foreground">
                          {referencePresetSummary(preset)}
                        </div>
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => applyReferencePreset(preset)}
                      >
                        Apply
                      </Button>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>

            {mediaKind === "image" &&
            generationTask !== "text_to_image" &&
            generationTask !== "edit" &&
            referenceImages.length > 0 ? (
              <div className="flex flex-col gap-1">
                <Label>Strength: {strength.toFixed(2)}</Label>
                <Slider
                  aria-label="Image strength"
                  min={0}
                  max={1}
                  step={0.01}
                  value={[strength]}
                  onValueChange={(v) => setStrength(v[0] ?? strength)}
                />
              </div>
            ) : null}

            {generationTask === "inpaint" ? (
              <div className="flex flex-col gap-3 border-t border-border pt-3">
                <div className="flex items-center justify-between gap-2">
                  <Label>Mask</Label>
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => maskInputRef.current?.click()}
                    >
                      <HugeiconsIcon icon={Add01Icon} className="mr-2 size-4" strokeWidth={1.5} />
                      Upload
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={clearPaintedMask}
                      disabled={!inpaintSource}
                    >
                      Clear
                    </Button>
                  </div>
                </div>
                {inpaintSource ? (
                  <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_180px]">
                    <div className="relative overflow-hidden rounded-md border border-border bg-muted/20">
                      <img
                        src={inpaintSource.previewUrl}
                        alt={inpaintSource.name}
                        className="block w-full"
                        draggable={false}
                      />
                      <canvas
                        ref={maskCanvasRef}
                        aria-label="Inpaint mask canvas"
                        className="absolute inset-0 h-full w-full touch-none opacity-50 mix-blend-screen"
                        onPointerDown={startMaskStroke}
                        onPointerMove={continueMaskStroke}
                        onPointerUp={endMaskStroke}
                        onPointerCancel={endMaskStroke}
                        onPointerLeave={endMaskStroke}
                      />
                    </div>
                    <div className="flex flex-col gap-3">
                      <div className="flex flex-col gap-1">
                        <Label>Brush: {maskBrushSize}px</Label>
                        <Slider
                          aria-label="Mask brush size"
                          min={8}
                          max={192}
                          step={1}
                          value={[maskBrushSize]}
                          onValueChange={(v) => setMaskBrushSize(v[0] ?? maskBrushSize)}
                        />
                      </div>
                      {maskImages.length > 0 ? (
                        <div className="grid grid-cols-2 gap-2">
                          {maskImages.map((image) => (
                            <div
                              key={image.id}
                              className="relative overflow-hidden rounded-md border border-border bg-muted/20"
                            >
                              <img
                                src={image.previewUrl}
                                alt={image.name}
                                className="aspect-square w-full object-cover"
                              />
                              <button
                                type="button"
                                aria-label={`Remove ${image.name}`}
                                onClick={() => removeMaskImage(image.id)}
                                className="absolute right-1 top-1 grid size-7 place-items-center rounded bg-background/90 text-foreground shadow-sm hover:bg-background"
                              >
                                <HugeiconsIcon icon={Delete02Icon} className="size-4" strokeWidth={1.5} />
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => imageInputRef.current?.click()}
                    className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-border bg-muted/20 text-sm text-muted-foreground hover:bg-muted/30 hover:text-foreground"
                  >
                    <HugeiconsIcon icon={FileImageIcon} className="mr-2 size-4" strokeWidth={1.5} />
                    Add a source image
                  </button>
                )}
              </div>
            ) : null}

            {controlNetEnabled || status?.controlnet ? (
              <div className="flex flex-col gap-3 border-t border-border pt-3">
                <div className="flex items-center justify-between gap-2">
                  <Label>Control images</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => controlInputRef.current?.click()}
                  >
                    <HugeiconsIcon icon={Add01Icon} className="mr-2 size-4" strokeWidth={1.5} />
                    Add
                  </Button>
                </div>
                {controlImages.length > 0 ? (
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {controlImages.map((image) => (
                      <div
                        key={image.id}
                        className="relative overflow-hidden rounded-md border border-border bg-muted/20"
                      >
                        <img
                          src={image.previewUrl}
                          alt={image.name}
                          className="aspect-square w-full object-cover"
                        />
                        <button
                          type="button"
                          aria-label={`Remove ${image.name}`}
                          onClick={() => removeControlImage(image.id)}
                          className="absolute right-1 top-1 grid size-7 place-items-center rounded bg-background/90 text-foreground shadow-sm hover:bg-background"
                        >
                          <HugeiconsIcon icon={Delete02Icon} className="size-4" strokeWidth={1.5} />
                        </button>
                        <div className="flex items-center gap-1 px-2 py-1 text-[11px] text-muted-foreground">
                          <HugeiconsIcon icon={FileImageIcon} className="size-3.5" strokeWidth={1.5} />
                          <span className="truncate">
                            {image.width}x{image.height}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => controlInputRef.current?.click()}
                    className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-border bg-muted/20 text-sm text-muted-foreground hover:bg-muted/30 hover:text-foreground"
                  >
                    <HugeiconsIcon icon={FileImageIcon} className="mr-2 size-4" strokeWidth={1.5} />
                    Add a control map
                  </button>
                )}
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div className="flex flex-col gap-1">
                    <Label>Scale: {controlScale.toFixed(2)}</Label>
                    <Slider
                      aria-label="ControlNet scale"
                      min={0}
                      max={2}
                      step={0.01}
                      value={[controlScale]}
                      onValueChange={(v) => setControlScale(v[0] ?? controlScale)}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label>Start: {controlStart.toFixed(2)}</Label>
                    <Slider
                      aria-label="ControlNet start"
                      min={0}
                      max={1}
                      step={0.01}
                      value={[controlStart]}
                      onValueChange={(v) => setControlStart(v[0] ?? controlStart)}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label>End: {controlEnd.toFixed(2)}</Label>
                    <Slider
                      aria-label="ControlNet end"
                      min={0}
                      max={1}
                      step={0.01}
                      value={[controlEnd]}
                      onValueChange={(v) => setControlEnd(v[0] ?? controlEnd)}
                    />
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          <div className="flex flex-col gap-3 border-t border-border pt-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <Label>Enhance</Label>
              <div className="grid grid-cols-4 rounded-md border border-border bg-muted/30 p-1">
                {ENHANCE_MODES.map((mode) => (
                  <button
                    key={mode.value}
                    type="button"
                    onClick={() => setEnhanceMode(mode.value)}
                    className={[
                      "min-h-8 px-2 text-xs font-medium transition",
                      enhanceMode === mode.value
                        ? "rounded bg-background text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground",
                    ].join(" ")}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>

            {enhanceMode !== "off" ? (
              <>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div className="flex flex-col gap-1">
                    <Label>Scale: {upscaleScale.toFixed(1)}x</Label>
                    <Slider
                      aria-label="Upscale scale"
                      min={1}
                      max={4}
                      step={0.5}
                      value={[upscaleScale]}
                      onValueChange={(v) => setUpscaleScale(v[0] ?? upscaleScale)}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label>Tile: {tileSize}px</Label>
                    <Slider
                      aria-label="Enhance tile size"
                      min={256}
                      max={1536}
                      step={128}
                      value={[tileSize]}
                      onValueChange={(v) => setTileSize(v[0] ?? tileSize)}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label>Overlap: {tileOverlap}px</Label>
                    <Slider
                      aria-label="Enhance tile overlap"
                      min={0}
                      max={256}
                      step={16}
                      value={[tileOverlap]}
                      onValueChange={(v) => setTileOverlap(v[0] ?? tileOverlap)}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="flex flex-col gap-1">
                    <Label>Resampling</Label>
                    <Select
                      value={upscaleMethod}
                      onValueChange={(value) => setUpscaleMethod(value as UpscaleMethod)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="lanczos">Lanczos</SelectItem>
                        <SelectItem value="bicubic">Bicubic</SelectItem>
                        <SelectItem value="bilinear">Bilinear</SelectItem>
                        <SelectItem value="nearest">Nearest</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label>VAE tiling</Label>
                    <Select
                      value={vaeTiling}
                      onValueChange={(value) => setVaeTiling(value as "auto" | "on" | "off")}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="auto">Auto</SelectItem>
                        <SelectItem value="on">On</SelectItem>
                        <SelectItem value="off">Off</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </>
            ) : null}

            {enhanceMode === "upscale" ? (
              <div className="grid grid-cols-1 gap-3 border-t border-border pt-3 sm:grid-cols-2">
                <div className="flex flex-col gap-1">
                  <Label>Upscale engine</Label>
                  <Select
                    value={upscaleEngine}
                    onValueChange={(value) => setUpscaleEngine(value as UpscaleEngine)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pixel">Built-in</SelectItem>
                      <SelectItem value="super_resolution">Model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-1">
                  <Label>Upscaler status</Label>
                  <div className="flex min-h-10 items-center rounded-md border border-border px-3 text-sm text-muted-foreground">
                    {upscaleEngine === "super_resolution" && status?.upscaler?.mode === "super_resolution"
                      ? `Loaded: ${status.upscaler.repo ?? "super-resolution upscaler"}`
                      : upscaleEngine === "pixel"
                        ? "Built-in"
                        : "Not loaded"}
                  </div>
                </div>
                {upscaleEngine === "super_resolution" ? (
                  <div className="flex flex-col gap-1 sm:col-span-2">
                    <Label>Super-resolution model repo</Label>
                    <Input
                      value={upscaleModelRepo}
                      onChange={(e) => setUpscaleModelRepo(e.target.value)}
                      placeholder="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
                    />
                    <p className="text-xs text-muted-foreground">
                      {"Reload the model after changing this. Built-in upscale does not load an extra model."}
                    </p>
                  </div>
                ) : null}
              </div>
            ) : null}

            {enhanceMode === "creative_upscale" ? (
              <div className="grid grid-cols-1 gap-3 border-t border-border pt-3 sm:grid-cols-2">
                <div className="flex flex-col gap-1 sm:col-span-2">
                  <Label>Creative upscaler repo</Label>
                  <Input
                    value={creativeUpscalerRepo}
                    onChange={(e) => setCreativeUpscalerRepo(e.target.value)}
                    placeholder="stabilityai/stable-diffusion-x4-upscaler"
                  />
                  <p className="text-xs text-muted-foreground">
                    {"Reload the model after changing this. Sharp upscale and large tiled modes do not need an extra model."}
                  </p>
                </div>
                <div className="flex flex-col gap-1">
                  <Label>Pipeline class</Label>
                  <Input
                    value={creativeUpscalerPipelineClass}
                    onChange={(e) => setCreativeUpscalerPipelineClass(e.target.value)}
                    placeholder="StableDiffusionUpscalePipeline"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label>Upscaler status</Label>
                  <div className="flex min-h-10 items-center rounded-md border border-border px-3 text-sm text-muted-foreground">
                    {status?.upscaler?.mode === "diffusion"
                      ? `Loaded: ${status.upscaler.repo ?? "diffusion upscaler"}`
                      : "Not loaded"}
                  </div>
                </div>
                <div className="flex flex-col gap-1">
                  <Label>Creative steps: {creativeUpscaleSteps}</Label>
                  <Slider
                    aria-label="Creative upscale steps"
                    min={1}
                    max={80}
                    step={1}
                    value={[creativeUpscaleSteps]}
                    onValueChange={(v) => setCreativeUpscaleSteps(v[0] ?? creativeUpscaleSteps)}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label>Creative guidance: {creativeUpscaleGuidance.toFixed(1)}</Label>
                  <Slider
                    aria-label="Creative upscale guidance"
                    min={0}
                    max={15}
                    step={0.1}
                    value={[creativeUpscaleGuidance]}
                    onValueChange={(v) => setCreativeUpscaleGuidance(v[0] ?? creativeUpscaleGuidance)}
                  />
                </div>
              </div>
            ) : null}
          </div>
          </>
          ) : null}

          {mediaKind === "video" ? (
            <div className="flex flex-col gap-3 border-t border-border pt-3">
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={(event) => {
                  if (event.target.files) void handleReferenceFiles(event.target.files);
                }}
              />
              <div className="flex items-center justify-between gap-2">
                <Label>Input frames</Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => imageInputRef.current?.click()}
                >
                  <HugeiconsIcon icon={Add01Icon} className="mr-2 size-4" strokeWidth={1.5} />
                  Add
                </Button>
              </div>
              {referenceImages.length > 0 ? (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {referenceImages.map((image) => (
                    <div
                      key={image.id}
                      className="relative overflow-hidden rounded-md border border-border bg-muted/20"
                    >
                      <img
                        src={image.previewUrl}
                        alt={image.name}
                        className="aspect-square w-full object-cover"
                      />
                      <button
                        type="button"
                        aria-label={`Remove ${image.name}`}
                        onClick={() => removeReferenceImage(image.id)}
                        className="absolute right-1 top-1 grid size-7 place-items-center rounded bg-background/90 text-foreground shadow-sm hover:bg-background"
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" strokeWidth={1.5} />
                      </button>
                      <div className="flex items-center gap-1 px-2 py-1 text-[11px] text-muted-foreground">
                        <HugeiconsIcon icon={FileImageIcon} className="size-3.5" strokeWidth={1.5} />
                        <span className="truncate">
                          {image.width}x{image.height}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  className="flex min-h-24 items-center justify-center rounded-md border border-dashed border-border bg-muted/20 text-sm text-muted-foreground hover:bg-muted/30 hover:text-foreground"
                >
                  <HugeiconsIcon icon={FileImageIcon} className="mr-2 size-4" strokeWidth={1.5} />
                  Add an initial frame
                </button>
              )}
            </div>
          ) : null}

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="flex flex-col gap-1">
              {mediaKind === "video" ? (
                <>
                  <Label htmlFor="diffusion-video-width">Width</Label>
                  <Input
                    id="diffusion-video-width"
                    type="number"
                    min={64}
                    max={2048}
                    step={8}
                    value={videoWidth}
                    onChange={(e) => setVideoWidth(Number(e.target.value) || videoWidth)}
                  />
                </>
              ) : (
                <>
                  <Label>Resolution</Label>
                  <Select
                    value={String(resolutionIdx)}
                    onValueChange={(v) => setResolutionIdx(Number(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {RESOLUTION_PRESETS.map((r, idx) => (
                        <SelectItem key={r.label} value={String(idx)}>
                          {r.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </>
              )}
            </div>
            <div className="flex flex-col gap-1">
              {mediaKind === "video" ? (
                <>
                  <Label htmlFor="diffusion-video-height">Height</Label>
                  <Input
                    id="diffusion-video-height"
                    type="number"
                    min={64}
                    max={2048}
                    step={8}
                    value={videoHeight}
                    onChange={(e) => setVideoHeight(Number(e.target.value) || videoHeight)}
                  />
                </>
              ) : (
                <>
                  <Label>Steps: {steps}</Label>
                  <Slider
                    aria-label="Inference steps"
                    min={1}
                    max={60}
                    step={1}
                    value={[steps]}
                    onValueChange={(v) => setSteps(v[0] ?? steps)}
                  />
                </>
              )}
            </div>
            <div className="flex flex-col gap-1">
              <Label>Guidance: {guidance.toFixed(1)}</Label>
              <Slider
                aria-label="Guidance scale"
                min={0}
                max={15}
                step={0.1}
                value={[guidance]}
                onValueChange={(v) => setGuidance(v[0] ?? guidance)}
              />
            </div>
          </div>

          {mediaKind === "video" ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="flex flex-col gap-1">
                <Label>Steps: {steps}</Label>
                <Slider
                  aria-label="Video inference steps"
                  min={1}
                  max={80}
                  step={1}
                  value={[steps]}
                  onValueChange={(v) => setSteps(v[0] ?? steps)}
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="diffusion-video-frames">Frames</Label>
                <Input
                  id="diffusion-video-frames"
                  type="number"
                  min={1}
                  max={513}
                  step={1}
                  value={videoFrames}
                  onChange={(e) => setVideoFrames(Number(e.target.value) || videoFrames)}
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label htmlFor="diffusion-video-fps">Frame rate</Label>
                <Input
                  id="diffusion-video-fps"
                  type="number"
                  min={1}
                  max={240}
                  step={1}
                  value={videoFrameRate}
                  onChange={(e) => setVideoFrameRate(Number(e.target.value) || videoFrameRate)}
                />
              </div>
            </div>
          ) : null}

          <div className="flex flex-col gap-1">
            <Label htmlFor="diffusion-seed">Seed (optional)</Label>
            <Input
              id="diffusion-seed"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="leave empty for random"
              inputMode="numeric"
            />
          </div>

          <div>
            <Button
              size="lg"
              onClick={handleGenerate}
              disabled={busy !== "idle" || !status?.is_loaded}
              data-testid="diffusion-generate"
            >
              {busy === "generating" ? <Spinner className="mr-2 size-4" /> : null}
              {`Generate ${mediaKind}`}
            </Button>
          </div>
        </div>
      </SectionCard>

      {results.length > 0 && (
        <SectionCard
          icon={<HugeiconsIcon icon={SparklesIcon} className="size-5" strokeWidth={1.5} />}
          title="Results"
          description="Most recent first."
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {results.flatMap((r, idx) => {
              if (r.kind === "video") {
                return [
                  <figure key={`${idx}-video`} className="flex flex-col gap-2">
                    <video
                      src={`data:${r.video_mime};base64,${r.video_b64}`}
                      controls
                      playsInline
                      className="h-auto w-full rounded-md border border-border bg-black object-contain"
                      data-testid="diffusion-result-video"
                    />
                    <figcaption className="text-xs text-muted-foreground">
                      {r.width}x{r.height} - {r.num_frames} frames - {r.frame_rate} fps - {r.num_inference_steps} steps - g={(r.guidance_scale ?? 0).toFixed(1)}
                      {r.seed_str
                        ? ` - seed ${r.seed_str}`
                        : r.seed !== null && r.seed !== undefined
                        ? ` - seed ${r.seed}`
                        : ""} -
                      {` ${(r.duration_ms / 1000).toFixed(1)}s`}
                    </figcaption>
                  </figure>,
                ];
              }
              const renderedImages =
                r.images_b64 && r.images_b64.length > 0 ? r.images_b64 : [r.image_b64];
              return renderedImages.map((imageB64, imageIdx) => (
                <figure key={`${idx}-${imageIdx}`} className="flex flex-col gap-2">
                  <img
                    src={`data:${r.image_mime};base64,${imageB64}`}
                    alt={`Generated image ${idx + 1}${imageIdx > 0 ? ` layer ${imageIdx + 1}` : ""}`}
                    // h-auto + object-contain so portrait / landscape
                    // outputs render at their true aspect ratio instead
                    // of being cropped into a square thumbnail.
                    className="h-auto w-full rounded-md border border-border object-contain"
                    data-testid="diffusion-result-image"
                  />
                  <figcaption className="text-xs text-muted-foreground">
                    {r.width}x{r.height} - {r.num_inference_steps} steps - g={(r.guidance_scale ?? 0).toFixed(1)}
                    {imageIdx > 0 ? ` - output ${imageIdx + 1}` : ""}
                    {/* Prefer seed_str (full uint64 precision) since the
                         numeric seed gets rounded by JSON.parse above
                         Number.MAX_SAFE_INTEGER and would otherwise
                         display a value that does not reproduce. */}
                    {r.seed_str
                      ? ` - seed ${r.seed_str}`
                      : r.seed !== null && r.seed !== undefined
                      ? ` - seed ${r.seed}`
                      : ""} -
                    {` ${(r.duration_ms / 1000).toFixed(1)}s`}
                  </figcaption>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        addGeneratedImageAsReference(r, imageB64, imageIdx, "edit_source")
                      }
                    >
                      Use as image input
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        addGeneratedImageAsReference(r, imageB64, imageIdx, "init_frame")
                      }
                    >
                      Use as video input
                    </Button>
                  </div>
                </figure>
              ));
            })}
          </div>
        </SectionCard>
      )}
    </div>
  );
}
