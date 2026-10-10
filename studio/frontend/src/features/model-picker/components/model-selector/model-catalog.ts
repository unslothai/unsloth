// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One canonical name per diffusion model, its artifacts, and a device-aware router. See model-catalog.check.ts.

import { normalizeDenseQuantSchemes } from "../../../../lib/dense-quant-schemes.ts";
import {
  AUDIO_CPP_MODELS,
  type AudioCppModel,
  type AudioCppTask,
  audioCppDisplayName,
} from "../../../audio/audio-cpp-catalog.ts";
import {
  type GgufFitClass,
  type GgufVariantSizes,
  classifyGgufFit as classifyGgufFitForDevice,
  ggufVariantFitSizeBytes,
} from "../../../../lib/gguf-fit.ts";
import {
  type HostClass,
  curatedArtifactIsOfferable,
  densePerfSuffix,
  ggufPerfSuffix,
  h3PerfSuffix,
  hostRunsDenseQuant,
} from "./host-artifact-policy.ts";
import type { ModelCapabilities } from "./model-capabilities";
import type { ModelOption } from "./types";

export type ArtifactFormat = "gguf" | "fp8" | "bnb-4bit" | "bf16";
/** The dense torchao schemes a host reports in `/api/system.dense_quant_schemes`. */
export type DenseQuantScheme = "fp8" | "int8";
export type LoadKind = "gguf" | "single_file" | "pipeline";

export interface ModelArtifact {
  repoId: string;
  /** Vendor repo this unsloth mirror copies byte for byte; its id resolves here too. */
  upstreamRepoId?: string;
  format: ArtifactFormat;
  loadKind: LoadKind;
  filename?: string;
  /** Second-level row label ("GGUF", "FP8", "BF16", "BF16 - 720p"). */
  label: string;
  denseQuantable?: boolean;
  /** Hosted pre-quantised checkpoint an auto load fetches instead of the dense bf16 shards. */
  prequantRepo?: string;
  /** Size (GB) per scheme, transformer only; int8 stores scales fp8 does not. */
  prequantSizeGb?: Readonly<Partial<Record<DenseQuantScheme, number>>>;
  /** Resident size for routing. Omitted = never auto-picked unless downloaded (GGUF self-fits). */
  approxSizeGb?: number;
  /** Measured CPU-offload fit tiers; a met tier bypasses the resident 70% rule. */
  offloadFitTiers?: readonly OffloadFitTier[];
  keywords?: readonly string[];
  /** Parameter count for the size chip; a fallback when the Hub listing reports none. */
  totalParams?: number;
  /** Gated on the Hub: a bare group click skips it unless already downloaded. */
  gated?: boolean;
  /** Fixed quant when a specialized runtime pins one exact GGUF file. */
  deviceQuant?: string;
}

export interface CatalogGroup {
  /** Canonical display id, owner spelled once ("unsloth/Qwen-Image-2512"). */
  canonicalId: string;
  displayName: string;
  description: string;
  scope: "image" | "video" | "audio";
  /** Audio-only task tag driving the Audio page's Speak/Transcribe mode interlock. */
  task?: "tts" | "stt";
  /** Descending quality order: bf16, fp8, bnb-4bit, gguf. The router walks it. */
  artifacts: ModelArtifact[];
  /** Cross-owner ids for this group; suffix stripping never merges owners. */
  aliases?: readonly string[];
  /** Capability glyph fallback; the Hub listing's tags win. */
  capabilities?: Partial<ModelCapabilities>;
  /** Leads the Recommended list whatever the dropdown sort, in catalog order among pinned groups. */
  pinToTop?: boolean;
}


const gguf = (repoId: string, extra?: Partial<ModelArtifact>): ModelArtifact => ({
  repoId,
  format: "gguf",
  loadKind: "gguf",
  label: "GGUF",
  keywords: ["gguf", "quantized"],
  ...extra,
});

const bnb4bit = (
  repoId: string,
  approxSizeGb: number,
  extra?: Partial<ModelArtifact>,
): ModelArtifact => ({
  repoId,
  format: "bnb-4bit",
  loadKind: "pipeline",
  label: "bnb-4bit",
  approxSizeGb,
  keywords: ["4bit", "bnb", "nf4", "bitsandbytes"],
  ...extra,
});

const fp8Pipeline = (
  repoId: string,
  approxSizeGb: number,
  extra: Partial<ModelArtifact> = {},
): ModelArtifact => ({
  repoId,
  format: "fp8",
  loadKind: "pipeline",
  label: "FP8",
  approxSizeGb,
  keywords: ["fp8", "float8"],
  ...extra,
});

const bf16Pipeline = (
  repoId: string,
  approxSizeGb?: number,
  extra?: Partial<ModelArtifact>,
): ModelArtifact => ({
  repoId,
  format: "bf16",
  loadKind: "pipeline",
  label: "BF16",
  approxSizeGb,
  keywords: ["bf16", "safetensors", "full precision"],
  denseQuantable: true,
  ...extra,
});

const bf16Mirror = (
  upstreamRepoId: string,
  approxSizeGb?: number,
  extra?: Partial<ModelArtifact>,
): ModelArtifact =>
  bf16Pipeline(`unsloth/${upstreamRepoId.split("/")[1]}`, approxSizeGb, {
    upstreamRepoId,
    ...extra,
  });

// from_single_file against the family base repo for the VAE / text encoder.
const bf16Single = (
  repoId: string,
  filename: string,
  approxSizeGb: number,
  extra?: Partial<ModelArtifact>,
): ModelArtifact => ({
  repoId,
  format: "bf16",
  loadKind: "single_file",
  filename,
  label: "BF16",
  approxSizeGb,
  keywords: ["bf16", "safetensors", "full precision"],
  denseQuantable: true,
  ...extra,
});

const AUDIO_GGUF_DESCRIPTIONS: Record<AudioCppTask, string> = {
  tts: "Text-to-speech",
  music: "Text-to-music",
  asr: "Speech-to-text",
  sep: "Source separation",
};

function audioGgufDescription(model: AudioCppModel): string {
  const workflows = model.workflows;
  if (!workflows || workflows.includes("speak")) {
    return AUDIO_GGUF_DESCRIPTIONS[model.task];
  }
  const clones = workflows.includes("clone");
  const converts = workflows.includes("convert");
  if (clones && converts) return "Voice cloning and conversion";
  if (converts) return "Voice conversion";
  if (clones) return "Voice cloning";
  return AUDIO_GGUF_DESCRIPTIONS[model.task];
}

// Audio GGUFs are plain GGUF rows; the backend routes them by GGUF header.
const audioGgufGroups = (tasks: readonly AudioCppTask[]): CatalogGroup[] =>
  AUDIO_CPP_MODELS.filter((model) => tasks.includes(model.task)).map((model) => ({
    canonicalId: model.id,
    displayName: audioCppDisplayName(model.id),
    description: audioGgufDescription(model),
    scope: "audio",
    task: model.task === "asr" ? "stt" : "tts",
    artifacts: [gguf(model.id)],
  }));

// Sizes are resident GB used only for routing; GGUF entries have none (pickDefaultQuant sizes them).

export const IMAGE_CATALOG: CatalogGroup[] = [
  {
    canonicalId: "unsloth/Z-Image-Turbo",
    displayName: "Z-Image-Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      // Resident bf16 GiB, not the Hub's fp32 DiT: 11.5 DiT + 7.5 Qwen3-4B + 0.2 VAE.
      bf16Mirror("Tongyi-MAI/Z-Image-Turbo", 19.1, {
        totalParams: 6154908736,
        prequantRepo: "unsloth/Z-Image-Turbo-FP8",
        prequantSizeGb: { fp8: 5.86, int8: 5.86 },
      }),
      bnb4bit("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", 8, { totalParams: 3210823936 }),
      gguf("unsloth/Z-Image-Turbo-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/Z-Image",
    displayName: "Z-Image",
    description: "Text-to-image",
    scope: "image",
    artifacts: [gguf("unsloth/Z-Image-GGUF")],
  },
  {
    canonicalId: "unsloth/Qwen-Image-2.1",
    displayName: "Qwen-Image 2.1",
    description: "Text-to-image and image editing",
    scope: "image",
    // The int8 half has no artifact row, so alias it for pasted ids.
    aliases: ["unsloth/Qwen-Image-2.1-FP8"],
    pinToTop: true,
    artifacts: [
      bf16Mirror("Qwen/Qwen-Image-2.1", 33, {
        totalParams: 7115124736,
        prequantRepo: "unsloth/Qwen-Image-2.1-FP8",
        prequantSizeGb: { fp8: 7.12, int8: 7.26 },
      }),
      gguf("unsloth/Qwen-Image-2.1-GGUF"),
    ],
  },
  {
    // A different denoiser with its own FP8/INT8 checkpoints; no unsloth mirror, so the vendor pipeline is the row.
    canonicalId: "Qwen/Qwen-Image-2.1-Turbo",
    displayName: "Qwen-Image 2.1 Turbo",
    description: "Text-to-image and image editing in 8 steps",
    scope: "image",
    // Same reason as 2.1's alias: the int8 half of the prequant repo has no artifact row.
    aliases: ["unsloth/Qwen-Image-2.1-Turbo-FP8"],
    artifacts: [
      bf16Pipeline("Qwen/Qwen-Image-2.1-Turbo", 33, {
        totalParams: 7115124736,
        prequantRepo: "unsloth/Qwen-Image-2.1-Turbo-FP8",
        prequantSizeGb: { fp8: 7.12, int8: 7.26 },
      }),
      gguf("unsloth/Qwen-Image-2.1-Turbo-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/Qwen-Image-2512",
    displayName: "Qwen-Image 2512",
    description: "Text-to-image",
    scope: "image",
    // The int8 half is reached via prequant_variant_repos; alias it so a pasted id still resolves.
    aliases: ["unsloth/Qwen-Image-2512-FP8"],
    artifacts: [
      bf16Mirror("Qwen/Qwen-Image-2512", 54, {
        totalParams: 20430401088,
        prequantRepo: "unsloth/Qwen-Image-2512-FP8",
        prequantSizeGb: { fp8: 19.06, int8: 25.4 },
      }),
      // No FP8 row: that repo holds torch .pt checkpoints, not single-file safetensors.
      bnb4bit("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", 14, { totalParams: 10850871408 }),
      gguf("unsloth/Qwen-Image-2512-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/Qwen-Image",
    displayName: "Qwen-Image",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("Qwen/Qwen-Image", 54, {
        totalParams: 20430401088,
        prequantRepo: "unsloth/Qwen-Image-FP8",
        prequantSizeGb: { fp8: 19.06, int8: 31.73 },
      }),
      gguf("unsloth/Qwen-Image-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-schnell",
    displayName: "FLUX.1 schnell",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("black-forest-labs/FLUX.1-schnell", 32, {
        totalParams: 11891178560,
        prequantRepo: "unsloth/FLUX.1-schnell-FP8",
        prequantSizeGb: { fp8: 11.09, int8: 14.13 },
      }),
      gguf("unsloth/FLUX.1-schnell-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-dev",
    displayName: "FLUX.1 dev",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("black-forest-labs/FLUX.1-dev", 32, { totalParams: 11901408320 }),
      gguf("unsloth/FLUX.1-dev-GGUF"),
    ],
  },
  {
    // Krea finetune of FLUX.1-dev, same layout, so it runs under the flux.1 family.
    canonicalId: "black-forest-labs/FLUX.1-Krea-dev",
    displayName: "FLUX.1 Krea dev",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("black-forest-labs/FLUX.1-Krea-dev", 32, { totalParams: 11901408320 }),
      gguf("QuantStack/FLUX.1-Krea-dev-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.2-klein-4B",
    displayName: "FLUX.2 klein 4B",
    description: "Text-to-image",
    scope: "image",
    artifacts: [gguf("unsloth/FLUX.2-klein-4B-GGUF")],
  },
  {
    canonicalId: "unsloth/FLUX.2-klein-9B",
    displayName: "FLUX.2 klein 9B",
    description: "Text-to-image",
    scope: "image",
    artifacts: [gguf("unsloth/FLUX.2-klein-9B-GGUF")],
  },
  {
    canonicalId: "unsloth/Qwen-Image-Edit-2511",
    displayName: "Qwen-Image-Edit 2511",
    description: "Image editing",
    scope: "image",
    artifacts: [
      bf16Mirror("Qwen/Qwen-Image-Edit-2511", 54, { totalParams: 20430401088 }),
      gguf("unsloth/Qwen-Image-Edit-2511-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-Kontext-dev",
    displayName: "FLUX.1 Kontext dev",
    description: "Image editing",
    scope: "image",
    artifacts: [
      bf16Mirror("black-forest-labs/FLUX.1-Kontext-dev", 32, { totalParams: 11901408320 }),
      gguf("unsloth/FLUX.1-Kontext-dev-GGUF"),
    ],
  },
  {
    canonicalId: "krea/Krea-2-Turbo",
    displayName: "Krea 2 Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("krea/Krea-2-Turbo", 18, {
        totalParams: 12820073036,
        prequantRepo: "unsloth/Krea-2-Turbo-FP8",
        prequantSizeGb: { fp8: 11.95, int8: 12.19 },
      }),
    ],
  },
  {
    // Ships fp32, cast on load; no upstream GGUF, so the pipeline is the only artifact.
    canonicalId: "Alpha-VLLM/Lumina-Image-2.0",
    displayName: "Lumina Image 2.0",
    description: "Text-to-image",
    scope: "image",
    artifacts: [bf16Mirror("Alpha-VLLM/Lumina-Image-2.0", 11, { totalParams: 2609769152 })],
  },
  {
    // bf16 only: the QuantStack GGUF was unpublished and has no vetted replacement.
    canonicalId: "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    displayName: "HunyuanImage 2.1",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 50, { totalParams: 17425795520 }),
    ],
  },
  {
    // The MIT repos lack text_encoder_4; the backend assembles it from the unsloth mirror (+16 GB).
    canonicalId: "HiDream-ai/HiDream-I1-Full",
    displayName: "HiDream I1",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("HiDream-ai/HiDream-I1-Full", 63, { totalParams: 17105733184 }),
      bf16Mirror("HiDream-ai/HiDream-I1-Dev", 63, {
        label: "BF16 - Dev (distilled)",
        keywords: ["bf16", "dev", "distilled"],
        totalParams: 17105733184,
      }),
      bf16Mirror("HiDream-ai/HiDream-I1-Fast", 63, {
        label: "BF16 - Fast (distilled)",
        keywords: ["bf16", "fast", "distilled"],
        totalParams: 17105733184,
      }),
    ],
  },
  {
    // No bf16 repo exists: -fp8 is raw float8 (~46 GB cast), -nf4-diffusers is bnb-4bit.
    canonicalId: "ideogram-ai/ideogram-4",
    displayName: "Ideogram 4",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      // Both Ideogram repos are gated, so neither can auto-route anonymously.
      fp8Pipeline("ideogram-ai/ideogram-4-fp8", 46, { gated: true, totalParams: 9281557760 }),
      bnb4bit("ideogram-ai/ideogram-4-nf4-diffusers", 11, { gated: true, totalParams: 4785317809 }),
    ],
  },
  {
    canonicalId: "stabilityai/sdxl-turbo",
    displayName: "SDXL Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("stabilityai/sdxl-turbo", 8, {
        label: "Safetensors",
        totalParams: 2567463684,
        denseQuantable: false,
      }),
    ],
  },
  {
    canonicalId: "stabilityai/stable-diffusion-xl-base-1.0",
    displayName: "SDXL Base 1.0",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Mirror("stabilityai/stable-diffusion-xl-base-1.0", 8, {
        label: "Safetensors",
        totalParams: 2567463684,
        denseQuantable: false,
      }),
    ],
  },
];

export const VIDEO_CATALOG: CatalogGroup[] = [
  {
    canonicalId: "MiniMaxAI/MiniMax-H3",
    displayName: "MiniMax H3",
    description: "Text, image and reference to video with synchronized audio",
    scope: "video",
    aliases: ["Comfy-Org/MiniMax-H3"],
    capabilities: { audio: true },
    artifacts: [
      bf16Pipeline("MiniMaxAI/MiniMax-H3", 145, {
        // Measured tiers in GiB (nvidia.py / main.py), while video.py estimators use decimal GB: never
        // copy figures across or the conversion applies twice and sends capable hosts to GGUF.
        offloadFitTiers: [
          { gpuGb: 30, systemRamGb: 80, requiresQuantisedStreaming: true },
          { gpuGb: 74, systemRamGb: 140 },
          { gpuGb: 123, systemRamGb: 80 },
        ],
      }),
      gguf("unsloth/MiniMax-H3-GGUF", {
        label: "GGUF",
        keywords: [
          "gguf",
          "quantized",
          "fl2va",
          "ref2va",
          "keyframes",
          "references",
        ],
        totalParams: 20_111_438_744,
      }),
    ],
  },
  {
    // Keyed on an existing artifact: unsloth/LTX-2.3 was never published and would bypass owner guards.
    canonicalId: "Lightricks/LTX-2.3",
    displayName: "LTX 2.3 distilled",
    description: "Text-to-video with audio",
    scope: "video",
    capabilities: { audio: true },
    artifacts: [
      bf16Single(
        "Lightricks/LTX-2.3",
        "ltx-2.3-22b-distilled.safetensors",
        90,
      ),
      // No FP8 artifact: the LTX-2.3 loader refuses the official scaled-FP8 single file.
      gguf("unsloth/LTX-2.3-GGUF", { totalParams: 21_005_004_544 }),
    ],
  },
  {
    canonicalId: "Lightricks/LTX-2",
    displayName: "LTX 2 (base)",
    description: "Text-to-video with audio",
    scope: "video",
    capabilities: { audio: true },
    artifacts: [bf16Pipeline("Lightricks/LTX-2", 90, { totalParams: 18876174592 })],
  },
  {
    canonicalId: "Wan-AI/Wan2.2-TI2V-5B",
    displayName: "Wan 2.2 TI2V 5B",
    description: "Text-to-video 720p",
    scope: "video",
    artifacts: [bf16Pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 30, { totalParams: 4999787712 })],
  },
  {
    canonicalId: "Wan-AI/Wan2.2-T2V-A14B",
    displayName: "Wan 2.2 T2V A14B (MoE)",
    description: "Text-to-video, dual-expert",
    scope: "video",
    artifacts: [bf16Pipeline("Wan-AI/Wan2.2-T2V-A14B-Diffusers", 114, { totalParams: 14288491584 })],
  },
  {
    canonicalId: "hunyuanvideo-community/HunyuanVideo-1.5",
    displayName: "HunyuanVideo 1.5",
    description: "Text-to-video",
    scope: "video",
    artifacts: [
      // pickDefaultArtifact sorts only by format, so catalog order picks the first fitting resolution.
      bf16Pipeline("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", 52, {
        label: "BF16 - 720p",
        keywords: ["bf16", "720p"],
        totalParams: 8326608160,
      }),
      bf16Pipeline("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", 40, {
        label: "BF16 - 480p",
        keywords: ["bf16", "480p"],
        totalParams: 8326608160,
      }),
    ],
  },
];

// stt groups map to the sidecar models in stt-model-catalog.ts; their sizes are informational.
export const AUDIO_CATALOG: CatalogGroup[] = [
  {
    canonicalId: "unsloth/orpheus-3b-0.1-ft",
    displayName: "Orpheus TTS 3B",
    description: "Text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("unsloth/orpheus-3b-0.1-ft", 7, { label: "Safetensors" }),
      gguf("unsloth/orpheus-3b-0.1-ft-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/csm-1b",
    displayName: "Sesame CSM 1B",
    description: "Text-to-speech",
    scope: "audio",
    task: "tts",
    // No GGUF: the llama.cpp TTS path has no CSM decoder.
    artifacts: [bf16Pipeline("unsloth/csm-1b", 6, { label: "Safetensors" })],
  },
  {
    canonicalId: "unsloth/Spark-TTS-0.5B",
    displayName: "Spark TTS 0.5B",
    description: "Text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [bf16Pipeline("unsloth/Spark-TTS-0.5B", 3, { label: "Safetensors" })],
  },
  {
    canonicalId: "unsloth/Llama-OuteTTS-1.0-1B",
    displayName: "Oute TTS 1B",
    description: "Text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("unsloth/Llama-OuteTTS-1.0-1B", 4, { label: "Safetensors" }),
    ],
  },
  {
    canonicalId: "bosonai/higgs-tts-2-3b-base",
    displayName: "Higgs TTS 2 3B",
    description: "Text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("bosonai/higgs-tts-2-3b-base", 12, {
        label: "Safetensors",
      }),
    ],
  },
  {
    canonicalId: "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    displayName: "MOSS TTS Local v1.5",
    description: "48 kHz stereo text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", 10, {
        label: "Safetensors",
      }),
    ],
  },
  {
    canonicalId: "OpenMOSS-Team/MOSS-TTS-Nano-100M",
    displayName: "MOSS TTS Nano 100M",
    description: "CPU-friendly text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("OpenMOSS-Team/MOSS-TTS-Nano-100M", 1, {
        label: "Safetensors",
      }),
    ],
  },
  {
    canonicalId: "multimodalart/higgs-audio-v3-tts-4b-transformers",
    displayName: "Higgs Audio v3 TTS 4B",
    description: "Text-to-speech",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("multimodalart/higgs-audio-v3-tts-4b-transformers", 10, {
        label: "Safetensors",
      }),
    ],
  },
  {
    canonicalId: "MiniMaxAI/MiniMax-Music3",
    displayName: "MiniMax Music 3",
    description: "Lyrics-to-music · NVIDIA CUDA",
    scope: "audio",
    task: "tts",
    artifacts: [
      bf16Pipeline("MiniMaxAI/MiniMax-Music3", 67, {
        label: "Diffusers",
        // 67 GB is the download footprint; the BF16 ModularPipeline fits a 24 GB GPU.
        offloadFitTiers: [{ gpuGb: 24, systemRamGb: 0 }],
      }),
    ],
  },
  ...audioGgufGroups(["tts", "music", "sep"]),
  // Llasa is deliberately absent: XCodec2 is not decodable here. Re-add with an xcodec2 decoder.
  {
    canonicalId: "unslothai/Qwen3-ASR-0.6B-GGUF",
    displayName: "Qwen3-ASR 0.6B",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [
      gguf("unslothai/Qwen3-ASR-0.6B-GGUF", { deviceQuant: "Q8_0" }),
    ],
  },
  {
    canonicalId: "unslothai/Qwen3-ASR-1.7B-GGUF",
    displayName: "Qwen3-ASR 1.7B",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [
      gguf("unslothai/Qwen3-ASR-1.7B-GGUF", { deviceQuant: "Q8_0" }),
    ],
  },
  {
    canonicalId: "unsloth/whisper-large-v3-turbo",
    displayName: "Whisper Large v3 Turbo",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [
      bf16Pipeline("unsloth/whisper-large-v3-turbo", 2, { label: "Safetensors" }),
    ],
  },
  {
    canonicalId: "unsloth/whisper-large-v3",
    displayName: "Whisper Large v3",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [
      bf16Pipeline("unsloth/whisper-large-v3", 4, { label: "Safetensors" }),
    ],
  },
  {
    canonicalId: "unsloth/whisper-small",
    displayName: "Whisper Small",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [bf16Pipeline("unsloth/whisper-small", 1, { label: "Safetensors" })],
  },
  {
    canonicalId: "unsloth/whisper-base",
    displayName: "Whisper Base",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [bf16Pipeline("unsloth/whisper-base", 1, { label: "Safetensors" })],
  },
  {
    canonicalId: "unsloth/whisper-tiny",
    displayName: "Whisper Tiny",
    description: "Speech-to-text",
    scope: "audio",
    task: "stt",
    artifacts: [bf16Pipeline("unsloth/whisper-tiny", 1, { label: "Safetensors" })],
  },
  ...audioGgufGroups(["asr"]),
];


// Stripped repeatedly, longest-first, off the name part; owner is preserved.
const ARTIFACT_SUFFIXES = [
  "-unsloth-bnb-4bit",
  "-nf4-diffusers",
  "-bnb-4bit",
  "-bnb4bit",
  "-fp8-dynamic",
  "-safetensors",
  "-diffusers",
  "-nvfp4",
  "-gguf",
  "-int8",
  "-4bit",
  "-nf4",
  "-fp8",
  "-bf16",
] as const;

/** Owner-preserving generic key: lowercase, artifact suffixes stripped off the name.
 *  "unsloth/Qwen-Image-2512-GGUF" to "unsloth/qwen-image-2512". */
export function canonicalKeyFor(repoId: string): string {
  const lowered = repoId.trim().toLowerCase();
  const slash = lowered.indexOf("/");
  const owner = slash >= 0 ? lowered.slice(0, slash + 1) : "";
  let name = slash >= 0 ? lowered.slice(slash + 1) : lowered;
  let stripped = true;
  while (stripped) {
    stripped = false;
    for (const suffix of ARTIFACT_SUFFIXES) {
      if (name.endsWith(suffix) && name.length > suffix.length) {
        name = name.slice(0, -suffix.length);
        stripped = true;
      }
    }
  }
  return owner + name;
}

/** Case-preserving display name with artifact suffixes stripped; the load id is untouched. */
export function stripArtifactSuffixesForDisplay(repoId: string): string {
  const trimmed = repoId.trim();
  const slash = trimmed.indexOf("/");
  const owner = slash >= 0 ? trimmed.slice(0, slash + 1) : "";
  let name = slash >= 0 ? trimmed.slice(slash + 1) : trimmed;
  let stripped = true;
  while (stripped) {
    stripped = false;
    const lowered = name.toLowerCase();
    for (const suffix of ARTIFACT_SUFFIXES) {
      if (lowered.endsWith(suffix) && name.length > suffix.length) {
        name = name.slice(0, -suffix.length);
        stripped = true;
        break;
      }
    }
  }
  return owner + name;
}

interface CatalogIndex {
  /** exact lowercased artifact/alias/canonical id -> group */
  byId: Map<string, CatalogGroup>;
  /** canonical suffix-stripped key -> group */
  byKey: Map<string, CatalogGroup>;
  /** exact lowercased artifact id -> artifact */
  artifactById: Map<string, ModelArtifact>;
}

const indexCache = new WeakMap<CatalogGroup[], CatalogIndex>();

function indexFor(catalog: CatalogGroup[]): CatalogIndex {
  const cached = indexCache.get(catalog);
  if (cached) return cached;
  const byId = new Map<string, CatalogGroup>();
  const byKey = new Map<string, CatalogGroup>();
  const artifactById = new Map<string, ModelArtifact>();
  for (const group of catalog) {
    byId.set(group.canonicalId.toLowerCase(), group);
    byKey.set(canonicalKeyFor(group.canonicalId), group);
    for (const alias of group.aliases ?? []) {
      byId.set(alias.toLowerCase(), group);
      // An alias also claims its stripped key so sibling artifacts of the aliased owner group correctly.
      byKey.set(canonicalKeyFor(alias), group);
    }
    for (const artifact of group.artifacts) {
      for (const id of [artifact.repoId, artifact.upstreamRepoId]) {
        if (!id) continue;
        byId.set(id.toLowerCase(), group);
        byKey.set(canonicalKeyFor(id), group);
        artifactById.set(id.toLowerCase(), artifact);
      }
    }
  }
  const built = { byId, byKey, artifactById };
  indexCache.set(catalog, built);
  return built;
}

/** The group a repo id belongs to, or null for unknown repos (callers render those ungrouped). */
export function groupForRepoId(
  repoId: string,
  catalog: CatalogGroup[],
): CatalogGroup | null {
  const index = indexFor(catalog);
  const lowered = repoId.trim().toLowerCase();
  return index.byId.get(lowered) ?? index.byKey.get(canonicalKeyFor(lowered)) ?? null;
}

/** The exact curated artifact for a repo id (null when the repo only matches a group by key/alias). */
export function artifactForRepoId(
  repoId: string,
  catalog: CatalogGroup[],
): { group: CatalogGroup; artifact: ModelArtifact } | null {
  const index = indexFor(catalog);
  const artifact = index.artifactById.get(repoId.trim().toLowerCase());
  if (!artifact) return null;
  const group = index.byId.get(repoId.trim().toLowerCase());
  return group ? { group, artifact } : null;
}

const BYTES_PER_GB = 1024 ** 3;

/** Curated size (bytes) of an exact artifact id; undefined for GGUF. */
export function curatedSizeBytesFor(
  repoId: string,
  catalog: CatalogGroup[],
): number | undefined {
  const gb = artifactForRepoId(repoId, catalog)?.artifact.approxSizeGb;
  return gb && gb > 0 ? gb * BYTES_PER_GB : undefined;
}

/** Curated parameter count; a fallback, callers must prefer the listing's total. */
export function curatedTotalParamsFor(
  repoId: string,
  catalog: CatalogGroup[],
): number | undefined {
  const params = artifactForRepoId(repoId, catalog)?.artifact.totalParams;
  return params && params > 0 ? params : undefined;
}

/** Group-level curated capabilities; the listing's tags win. */
export function curatedCapabilitiesFor(
  repoId: string,
  catalog: CatalogGroup[],
): ModelCapabilities | undefined {
  const group = groupForRepoId(repoId, catalog);
  if (!group) return undefined;
  const declared = group.capabilities;
  return {
    vision: declared?.vision ?? false,
    reasoning: declared?.reasoning ?? false,
    audio: declared?.audio ?? false,
    // From the group's scope, which beats the name heuristic.
    imageGen: declared?.imageGen ?? group.scope === "image",
    videoGen: declared?.videoGen ?? group.scope === "video",
  };
}

export function curatedDisplayNameFor(
  repoId: string,
  catalog: CatalogGroup[],
  host: HostClass = "unknown",
  denseQuantSchemes: readonly string[] = [],
): string | null {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return null;
  // Trigger and row must read the same, or the model renames itself as the popover opens.
  if (curatedPerfSuffix(hit, host, denseQuantSchemes)) {
    return (
      curatedRowLabelFor(repoId, catalog, host, denseQuantSchemes)?.name ??
      hit.group.displayName
    );
  }
  return hit.group.artifacts.length > 1
    ? `${hit.group.displayName} (${hit.artifact.label})`
    : hit.group.displayName;
}

// Labels are "FORMAT" or "FORMAT - QUALIFIER"; format and resolution become chips.
const LABEL_PART_SEPARATOR = " - ";
const GGUF_SUFFIX_RE = /-gguf$/i;
const RESOLUTION_RE = /^\d{3,4}p$/i;

/** Whether a known artifact can accept transformer quantisation. Unknown ids defer to the backend. */
export function curatedArtifactTakesDenseQuant(
  repoId: string,
  catalog: CatalogGroup[],
): boolean | undefined {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return undefined;
  // GGUF reaches quantisation through dense base-weight substitution.
  if (hit.artifact.format === "gguf") return true;
  return (
    hit.artifact.format === "bf16" &&
    hit.artifact.loadKind === "pipeline" &&
    hit.artifact.denseQuantable === true
  );
}

/** Whether this row runs the dense quant path on this host. Artifact and host only: the actual
 *  precision depends on request inputs and is reported by `resolved`. */
function artifactUsesDenseQuant(
  group: CatalogGroup,
  artifact: ModelArtifact,
  host: HostClass,
): boolean {
  return hostRunsDenseQuant(host) && artifactTakesDenseQuant(group, artifact);
}

function artifactTakesDenseQuant(
  group: CatalogGroup,
  artifact: ModelArtifact,
): boolean {
  return (
    group.scope === "image" &&
    artifact.format === "bf16" &&
    artifact.loadKind === "pipeline" &&
    artifact.denseQuantable === true
  );
}

function curatedPerfSuffix(
  hit: { group: CatalogGroup; artifact: ModelArtifact },
  host: HostClass,
  denseQuantSchemes: readonly string[],
): string | null {
  // Audio GGUFs run the whisper.cpp sidecar, which has no dense sibling to be slow against.
  if (
    hit.artifact.format === "gguf" &&
    (hit.group.scope === "image" || hit.group.scope === "video")
  ) {
    return ggufPerfSuffix(host);
  }
  if (artifactUsesDenseQuant(hit.group, hit.artifact, host)) {
    return densePerfSuffix(denseQuantSchemes);
  }
  return h3PerfSuffix(hit.artifact.repoId, host, denseQuantSchemes);
}

/** A curated row as name plus chips. Null for unknown ids. */
export function curatedRowLabelFor(
  repoId: string,
  catalog: CatalogGroup[],
  host: HostClass = "unknown",
  denseQuantSchemes: readonly string[] = [],
): { name: string; tags: string[] } | null {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return null;
  // Only where the host can run both rows.
  const perf = curatedPerfSuffix(hit, host, denseQuantSchemes);
  // Match the first word so an existing "Fast" in the name is not duplicated.
  const perfWord = perf?.split(" ")[0];
  const qualify = (name: string) =>
    perf && perfWord && !new RegExp(`\\b${perfWord}\\b`, "i").test(name)
      ? `${name} (${perf})`
      : name;
  // The repo name already ends in -GGUF, so no chip.
  if (hit.artifact.format === "gguf") {
    const leaf = hit.artifact.repoId.split("/").pop() ?? hit.artifact.repoId;
    return { name: qualify(GGUF_SUFFIX_RE.test(leaf) ? leaf : `${leaf}-GGUF`), tags: [] };
  }
  if (hit.group.artifacts.length <= 1) return { name: qualify(hit.group.displayName), tags: [] };
  const [format, ...rest] = hit.artifact.label.split(LABEL_PART_SEPARATOR);
  // The chip is the stored precision; "Safetensors" is left off (the format dot says it).
  const tags = [format.trim()].filter((tag) => tag && tag.toLowerCase() !== "safetensors");
  const kept: string[] = [];
  for (const part of rest) {
    if (RESOLUTION_RE.test(part.trim())) tags.push(part.trim());
    else kept.push(part);
  }
  const name =
    kept.length > 0
      ? `${hit.group.displayName} (${kept.join(LABEL_PART_SEPARATOR)})`
      : hit.group.displayName;
  return { name: qualify(name), tags };
}

/** Back-compat: the flat ModelOption list the ModelSelector `models` prop expects, one option per ARTIFACT. */
export function catalogToModelOptions(
  catalog: CatalogGroup[],
  host: HostClass = "unknown",
  denseQuantSchemes: readonly string[] = [],
): ModelOption[] {
  const options: ModelOption[] = [];
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      // The `models` prop is built only here, so this covers trigger name and seed ids.
      if (!curatedArtifactIsOfferable(artifact.repoId, host)) continue;
      options.push({
        id: artifact.repoId,
        name:
          curatedDisplayNameFor(artifact.repoId, catalog, host, denseQuantSchemes) ??
          group.displayName,
        description: group.description,
        descriptionSuffix: artifact.label,
        isGguf: artifact.format === "gguf",
        deviceQuant: artifact.deviceQuant,
      });
    }
  }
  return options;
}

/** How to load a curated artifact. Null for unknown ids (GGUF picks carry their own variant
 *  metadata; local paths and hub GGUFs resolve elsewhere). */
export function loadSpecFor(
  repoId: string,
  catalog: CatalogGroup[],
): { kind: LoadKind; filename?: string } | null {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return null;
  return { kind: hit.artifact.loadKind, filename: hit.artifact.filename };
}

const GGUF_QUANT_TOKENS = [
  "q2",
  "q3",
  "q4",
  "q5",
  "q6",
  "q8",
  "q4_k_m",
  "q5_k_m",
  "q6_k",
  "q8_0",
  "bf16",
  "f16",
] as const;

export function groupMatchesQuery(group: CatalogGroup, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (group.canonicalId.toLowerCase().includes(q)) return true;
  if (group.displayName.toLowerCase().includes(q)) return true;
  if (group.description.toLowerCase().includes(q)) return true;
  for (const alias of group.aliases ?? []) {
    if (alias.toLowerCase().includes(q)) return true;
  }
  for (const artifact of group.artifacts) {
    if (artifact.repoId.toLowerCase().includes(q)) return true;
    if (artifact.label.toLowerCase().includes(q)) return true;
    for (const keyword of artifact.keywords ?? []) {
      if (keyword.includes(q) || q.includes(keyword)) return true;
    }
    if (artifact.format === "gguf" && GGUF_QUANT_TOKENS.some((t) => q === t)) {
      return true;
    }
  }
  return false;
}


export interface OffloadFitTier {
  gpuGb: number;
  systemRamGb: number;
  /** Only offered where `/api/system.quantised_streaming` says group offload streams torchao weights. */
  requiresQuantisedStreaming?: boolean;
}

function offloadTierMet(tier: OffloadFitTier, budget: DeviceBudget): boolean {
  if (tier.requiresQuantisedStreaming && budget.quantisedStreaming !== true) return false;
  return budget.gpuGb >= tier.gpuGb && budget.systemRamGb >= tier.systemRamGb;
}

export interface DeviceBudget {
  /** Total GPU memory in GB (0/undefined = unknown or none). */
  gpuGb: number;
  systemRamGb: number;
  /** The user's saved VRAM Budget. Absent falls back to the loader's default. */
  budgetFraction?: number;
  /** GPUs gpuGb sums, for the loader's per-card VRAM reserve. Absent means one. */
  gpuCount?: number;
  /** Dense quant schemes this host runs, best first; empty keeps the bf16 sizing rule. */
  denseQuantSchemes?: readonly string[];
  /** Group offload can stream torchao weights; absent = unknown, so streamed tiers are not offered. */
  quantisedStreaming?: boolean;
  /** Backend-reported extra offload tiers per lower-cased repo id. They only widen catalog tiers
   *  and are ignored for artifacts without them. */
  extraOffloadFitTiers?: Readonly<Record<string, readonly OffloadFitTier[]>>;
}

function artifactOffloadTiers(
  artifact: ModelArtifact,
  budget: DeviceBudget,
): readonly OffloadFitTier[] {
  const own = artifact.offloadFitTiers ?? [];
  if (!own.length) return own;
  const extra = budget.extraOffloadFitTiers?.[artifact.repoId.trim().toLowerCase()];
  return extra?.length ? [...own, ...extra] : own;
}

/** GGUF fit, delegated to the formula the Hub badge uses. */
export function classifyGgufFit(
  sizeBytes: number,
  budget: DeviceBudget,
): GgufFitClass {
  // Nothing measured, so do not invent a verdict.
  if ((budget.gpuGb || 0) <= 0 && (budget.systemRamGb || 0) <= 0) return "fits";
  if (sizeBytes <= 0) return "fits";
  return classifyGgufFitForDevice(sizeBytes, budget);
}

/** Fit rule for media GGUFs: the diffusion backend cannot offload and budgets free memory at a
 *  0.85 margin (`diffusion_memory.py`), so the llama.cpp formula must not judge them. */
export function classifyMediaGgufFit(
  sizeBytes: number,
  gpuGb: number,
  systemRamGb: number,
): GgufFitClass {
  const gpuBudgetGb = gpuGb * 0.7;
  const totalBudgetGb = gpuBudgetGb + systemRamGb * 0.7;
  const gb = sizeBytes / 1024 ** 3;
  if (gb <= 0 || gb <= gpuBudgetGb) return "fits";
  if (gpuBudgetGb <= 0) return gb <= totalBudgetGb ? "fits" : "oom";
  // "partial" describes a spill out of the card.
  return gb <= totalBudgetGb ? "partial" : "oom";
}

/** Only `oom` fails to load; `partial` runs by offloading to CPU. */
export function ggufFitRuns(fit: GgufFitClass): boolean {
  return fit !== "oom";
}

/** `fits` and `ram` keep a reserve; `marginal` and `partial` run at the edge. */
export function ggufFitIsComfortable(fit: GgufFitClass): boolean {
  return fit === "fits" || fit === "ram";
}

/** With a repo default, keep it wherever it loads, else step down. Without one, the largest
 *  comfortable quant, else the largest that runs, else the smallest. Null when nothing is sized. */
export function recommendedQuantForDevice<T extends GgufVariantSizes>(
  variants: readonly T[],
  fitOf: (sizeBytes: number) => GgufFitClass,
  preferred: T | null = null,
): T | null {
  // Judged on the download footprint, which includes companion GGUFs (mmproj, drafter).
  const fitOfVariant = (variant: T) => fitOf(ggufVariantFitSizeBytes(variant));
  // Zero size means unknown (old backend or stat error) and must not compete as comfortable.
  const sized = variants.filter((variant) => ggufVariantFitSizeBytes(variant) > 0);
  if (sized.length === 0) return null;
  const bySizeDesc = [...sized].sort(
    (left, right) => right.size_bytes - left.size_bytes,
  );
  // Nothing runs: pick the smallest footprint, since companions vary per checkpoint.
  const smallestFootprint = bySizeDesc.reduce((best, variant) =>
    ggufVariantFitSizeBytes(variant) < ggufVariantFitSizeBytes(best)
      ? variant
      : best,
  );
  // Step down from the default when it cannot load, never up past it.
  const candidates =
    preferred && variants.includes(preferred)
      ? ggufVariantFitSizeBytes(preferred) <= 0 ||
        fitOfVariant(preferred) !== "oom"
        ? [preferred]
        : bySizeDesc.filter(
            (variant) => variant.size_bytes < preferred.size_bytes,
          )
      : bySizeDesc;
  return (
    candidates.find((variant) => ggufFitIsComfortable(fitOfVariant(variant))) ??
    candidates.find((variant) => fitOfVariant(variant) !== "oom") ??
    smallestFootprint
  );
}

export interface QuantVariant {
  quant: string;
  filename: string;
  size_bytes: number;
  downloaded?: boolean;
}

/** The quant a bare group/repo click loads: largest downloaded non-OOM quant, else the repo default
 *  when non-OOM, else the largest fitting, else the smallest overall. */
export function pickDefaultQuant(
  variants: QuantVariant[],
  defaultVariant: string | null,
  budget: DeviceBudget,
): QuantVariant | null {
  if (!variants || variants.length === 0) return null;
  const anyBudget = (budget.gpuGb || 0) > 0 || (budget.systemRamGb || 0) > 0;
  const runs = (v: QuantVariant) =>
    ggufFitRuns(classifyGgufFit(v.size_bytes, budget));
  const downloadedFitting = variants
    .filter((v) => v.downloaded && runs(v))
    .sort((a, b) => b.size_bytes - a.size_bytes);
  if (downloadedFitting.length > 0) return downloadedFitting[0];
  const byQuant = (quant: string | null) =>
    quant ? (variants.find((v) => v.quant === quant) ?? null) : null;
  if (!anyBudget) return byQuant(defaultVariant) ?? variants[0];
  const defaultV = byQuant(defaultVariant);
  if (defaultV && runs(defaultV)) {
    return defaultV;
  }
  const fitting = variants
    .filter(runs)
    .sort((a, b) => b.size_bytes - a.size_bytes);
  if (fitting.length > 0) return fitting[0];
  const smallest = [...variants].sort((a, b) => a.size_bytes - b.size_bytes);
  return smallest[0] ?? null;
}

export interface RoutingInput extends DeviceBudget {
  isDownloaded: (repoId: string) => boolean;
}

const FORMAT_QUALITY: Record<ArtifactFormat, number> = {
  bf16: 0,
  fp8: 1,
  "bnb-4bit": 2,
  gguf: 3,
};

const BF16_BYTES_PER_PARAM = 2;

/** Pre-quantised transformer plus companions, clamped to the dense figure. Schemes are a ladder:
 *  the first rung fitting `allowanceGb` is what the backend loads. */
function residentSizeGb(
  group: CatalogGroup,
  artifact: ModelArtifact,
  budget: DeviceBudget,
  allowanceGb?: number,
): number | undefined {
  const dense = artifact.approxSizeGb;
  const schemes = normalizeDenseQuantSchemes(budget.denseQuantSchemes);
  if (schemes.length === 0 || dense === undefined) return dense;
  if (!artifactTakesDenseQuant(group, artifact)) return dense;
  const denseTransformerGb = artifact.totalParams
    ? (artifact.totalParams * BF16_BYTES_PER_PARAM) / BYTES_PER_GB
    : 0;
  const companionsGb = Math.max(0, dense - denseTransformerGb);
  let firstHosted: number | undefined;
  for (const scheme of schemes) {
    const quantised = artifact.prequantSizeGb?.[scheme as DenseQuantScheme];
    if (quantised === undefined) continue;
    const sizeGb = Math.min(dense, quantised + companionsGb);
    if (firstHosted === undefined) firstHosted = sizeGb;
    if (allowanceGb === undefined || sizeGb <= allowanceGb) return sizeGb;
  }
  return firstHosted ?? dense;
}

function fitsArtifactBudget(
  group: CatalogGroup,
  artifact: ModelArtifact,
  budget: DeviceBudget,
): boolean {
  if (artifact.offloadFitTiers?.length) {
    return artifactOffloadTiers(artifact, budget).some(
      (tier) => offloadTierMet(tier, budget),
    );
  }
  const allowanceGb = budget.gpuGb * 0.7;
  const sizeGb = residentSizeGb(group, artifact, budget, allowanceGb);
  if (sizeGb === undefined) return false;
  return sizeGb <= allowanceGb;
}

/** The artifact a bare group click loads. Sized artifacts normally use the 0.7 * GPU budget;
 *  measured offload tiers can override that fit check. */
export function pickDefaultArtifact(
  group: CatalogGroup,
  input: RoutingInput,
): ModelArtifact {
  const artifacts = [...group.artifacts].sort(
    (a, b) => FORMAT_QUALITY[a.format] - FORMAT_QUALITY[b.format],
  );
  const ggufArtifact = artifacts.find((a) => a.format === "gguf") ?? null;
  const downloaded = artifacts.filter((a) => input.isDownloaded(a.repoId));
  if (downloaded.length > 0) {
    const fitting = downloaded.find(
      (a) => a.format !== "gguf" && fitsArtifactBudget(group, a, input),
    );
    if (fitting) return fitting;
    const downloadedGguf = downloaded.find((a) => a.format === "gguf");
    if (downloadedGguf) return downloadedGguf;
    return downloaded.sort(
      (a, b) => (a.approxSizeGb ?? Infinity) - (b.approxSizeGb ?? Infinity),
    )[0];
  }
  if (!input.gpuGb || input.gpuGb <= 0) {
    return ggufArtifact ?? artifacts[0];
  }
  for (const artifact of artifacts) {
    // Skip a gated, not-downloaded artifact: auto-routing there fails without a token.
    if (
      artifact.format !== "gguf" &&
      !artifact.gated &&
      fitsArtifactBudget(group, artifact, input)
    ) {
      return artifact;
    }
  }
  if (ggufArtifact) return ggufArtifact;
  return artifacts.sort(
    (a, b) => (a.approxSizeGb ?? Infinity) - (b.approxSizeGb ?? Infinity),
  )[0];
}

/** Fit verdict and badge estimate for one curated artifact; undefined if unknown.
 *  Uses GPU memory, or RAM for unified-memory hosts and transcription fallback.
 *  Measured offload tiers return the catalog size without an allowance. */
export function curatedArtifactFit(
  repoId: string,
  catalog: CatalogGroup[],
  budget: DeviceBudget,
): {
  fits: boolean;
  sizeGb?: number;
  allowanceGb?: number;
  deviceGb?: number;
  device?: "GPU" | "RAM";
} | undefined {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit || hit.artifact.format === "gguf") return undefined;
  const { group, artifact } = hit;
  if (budget.gpuGb <= 0 && budget.systemRamGb <= 0) return undefined;
  if (artifact.offloadFitTiers?.length) {
    return {
      fits: fitsArtifactBudget(group, artifact, budget),
      sizeGb: artifact.approxSizeGb,
    };
  }
  // STT retries on CPU so RAM counts, but the model lands on one device: the larger, not the sum.
  const onRam =
    group.task === "stt" ? budget.systemRamGb > budget.gpuGb : budget.gpuGb <= 0;
  const deviceGb = onRam ? budget.systemRamGb : budget.gpuGb;
  const allowanceGb = deviceGb * 0.7;
  const sizeGb = residentSizeGb(group, artifact, budget, allowanceGb);
  if (sizeGb === undefined) return undefined;
  return {
    fits: sizeGb <= allowanceGb,
    sizeGb,
    allowanceGb,
    deviceGb,
    device: onRam ? "RAM" : "GPU",
  };
}

export function curatedArtifactFitsDevice(
  repoId: string,
  catalog: CatalogGroup[],
  budget: DeviceBudget,
): boolean | undefined {
  return curatedArtifactFit(repoId, catalog, budget)?.fits;
}

export function catalogGroupFitsDevice(
  group: CatalogGroup,
  budget: DeviceBudget,
  isDownloaded: (repoId: string) => boolean,
): boolean {
  const budgetGb =
    Math.max(0, budget.gpuGb || 0) * 0.7 +
    Math.max(0, budget.systemRamGb || 0) * 0.7;
  if (budgetGb <= 0) return true;
  return group.artifacts.some((a) => {
    if (isDownloaded(a.repoId)) return true;
    // A GGUF ladder self-fits (llama-server offloads), matching pickDefaultArtifact.
    if (a.format === "gguf") return true;
    if (a.offloadFitTiers?.length) {
      return artifactOffloadTiers(a, budget).some(
        (tier) => offloadTierMet(tier, budget),
      );
    }
    // Same quantised sizing as the row badge and pickDefaultArtifact.
    const sizeGb = residentSizeGb(group, a, budget, budgetGb);
    return sizeGb !== undefined && sizeGb <= budgetGb;
  });
}
