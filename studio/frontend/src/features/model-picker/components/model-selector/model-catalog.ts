// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One canonical name per diffusion model, its published artifacts (GGUF quants, prequant FP8 /
// bnb-4bit repos, official BF16 pipelines), and a deterministic router picking the best
// artifact for the device. Pure helpers, no React/DOM deps. See model-catalog.check.ts.

import {
  type GgufFitClass,
  classifyGgufFit as classifyGgufFitForDevice,
} from "../../../../lib/gguf-fit.ts";
import {
  type HostClass,
  curatedArtifactIsOfferable,
  h3PerfSuffix,
} from "./host-artifact-policy.ts";
import type { ModelCapabilities } from "./model-capabilities";
import type { ModelOption } from "./types";

export type ArtifactFormat = "gguf" | "fp8" | "bnb-4bit" | "bf16";
export type LoadKind = "gguf" | "single_file" | "pipeline";

export interface ModelArtifact {
  /** Exact artifact repo id (the pre-grouping id, stays loadable/searchable). */
  repoId: string;
  format: ArtifactFormat;
  loadKind: LoadKind;
  /** single_file loads name their exact checkpoint inside the repo. */
  filename?: string;
  /** Second-level row label ("GGUF", "FP8", "BF16 (official)", "BF16 - 720p"). */
  label: string;
  /** Curated resident-size estimate for routing. Omitted = unknown: never auto-picked unless
   *  downloaded. GGUF omits it too, since its quant ladder self-fits via pickDefaultQuant. */
  approxSizeGb?: number;
  /** Measured CPU-offload fit tiers. Any tier whose GPU and available-RAM floors are both met can
   *  auto-route this artifact without the resident 70% rule. */
  offloadFitTiers?: readonly { gpuGb: number; systemRamGb: number }[];
  /** Extra search tokens beyond the id/label ("4bit", "nf4", ...). */
  keywords?: readonly string[];
  /** Parameter count of THIS artifact's checkpoint, for the row's size chip. Only a fallback: the
   *  Hub listing's `expand=gguf` total wins wherever it reports one, and rows the listing never
   *  returns have no other source (ids like "MiniMax-H3-GGUF" carry no "<n>B" token). */
  totalParams?: number;
  /** Gated on the Hub (license + token). A bare group click skips it when not downloaded and falls
   *  through to an open artifact; an already-downloaded gated artifact is still returned. */
  gated?: boolean;
  /** Fixed quant when a specialized runtime pins one exact GGUF file. */
  deviceQuant?: string;
}

export interface CatalogGroup {
  /** Canonical display id, owner spelled once ("unsloth/Qwen-Image-2512"). */
  canonicalId: string;
  displayName: string;
  /** Row meta line ("Text-to-image", "Image editing", "Text-to-video with audio"). */
  description: string;
  scope: "image" | "video" | "audio";
  /** Audio-only task tag driving the Audio page's Speak/Transcribe mode interlock. */
  task?: "tts" | "stt";
  /** Descending quality order: bf16, fp8, bnb-4bit, gguf. The router walks it. */
  artifacts: ModelArtifact[];
  /** Cross-owner ids that resolve to this group. Suffix stripping never merges two owners on its own. */
  aliases?: readonly string[];
  /** What the model can do, for the row's capability glyphs. Same fallback rule as
   *  `ModelArtifact.totalParams`: the Hub listing's tags win where it returns the row, since
   *  `detectCapabilities` reads tags then repo-name keywords and a name like "MiniMax-H3-GGUF"
   *  says nothing about the audio track the model emits. */
  capabilities?: Partial<ModelCapabilities>;
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
  label: "BF16 (official)",
  approxSizeGb,
  keywords: ["bf16", "safetensors", "full precision"],
  ...extra,
});

// A bf16 single-file DiT checkpoint: from_single_file against the family base repo for the VAE /
// text encoder, like the fp8 single-file checkpoints.
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
  label: "BF16 (official)",
  approxSizeGb,
  keywords: ["bf16", "safetensors", "full precision"],
  ...extra,
});

// Sizes are steady resident estimates (GB) used only for routing; missing = never auto-pick
// unless downloaded. GGUF entries carry none: pickDefaultQuant sizes the .gguf files.

export const IMAGE_CATALOG: CatalogGroup[] = [
  {
    canonicalId: "unsloth/Z-Image-Turbo",
    displayName: "Z-Image-Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("Tongyi-MAI/Z-Image-Turbo", 30, { totalParams: 6154908736 }),
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
    canonicalId: "unsloth/Qwen-Image-2512",
    displayName: "Qwen-Image 2512",
    description: "Text-to-image",
    scope: "image",
    // The prequant repo is real and public and the backend reaches its int8 half through
    // prequant_repos. It has no artifact row here, so alias it to keep a pasted id finding it.
    aliases: ["unsloth/Qwen-Image-2512-FP8"],
    artifacts: [
      bf16Pipeline("Qwen/Qwen-Image-2512", 54, { totalParams: 20430401088 }),
      // No FP8 row: unsloth/Qwen-Image-2512-FP8 holds torch prequant .pt checkpoints, not a single-file
      // .safetensors, and fp8 is denied for this family anyway (_FAMILY_SCHEME_DENY: qwen-image
      // renders every frame black under fp8).
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
      bf16Pipeline("Qwen/Qwen-Image", 54, { totalParams: 20430401088 }),
      gguf("unsloth/Qwen-Image-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-schnell",
    displayName: "FLUX.1 schnell",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      // Apache-2.0 but still gated on the Hub (gated: "auto", a contact-info form), so an anonymous
      // download 401s exactly like dev. The licence and the gate are independent.
      bf16Pipeline("black-forest-labs/FLUX.1-schnell", 32, { gated: true, totalParams: 11891178560 }),
      gguf("unsloth/FLUX.1-schnell-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-dev",
    displayName: "FLUX.1 dev",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      // FLUX.1-dev is gated (license acceptance + token), like FLUX.1-schnell above.
      bf16Pipeline("black-forest-labs/FLUX.1-dev", 32, { gated: true, totalParams: 11901408320 }),
      gguf("unsloth/FLUX.1-dev-GGUF"),
    ],
  },
  {
    // Krea guidance-distilled FLUX.1-dev finetune: same arch/layout as dev, so it runs under the
    // flux.1 family. The base repo is gated like dev; QuantStack publishes the open GGUF quants.
    canonicalId: "black-forest-labs/FLUX.1-Krea-dev",
    displayName: "FLUX.1 Krea dev",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("black-forest-labs/FLUX.1-Krea-dev", 32, { gated: true, totalParams: 11901408320 }),
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
      bf16Pipeline("Qwen/Qwen-Image-Edit-2511", 54, { totalParams: 20430401088 }),
      gguf("unsloth/Qwen-Image-Edit-2511-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-Kontext-dev",
    displayName: "FLUX.1 Kontext dev",
    description: "Image editing",
    scope: "image",
    artifacts: [
      // FLUX.1-Kontext-dev is gated on the Hub (license acceptance + token).
      bf16Pipeline("black-forest-labs/FLUX.1-Kontext-dev", 32, { gated: true, totalParams: 11901408320 }),
      gguf("unsloth/FLUX.1-Kontext-dev-GGUF"),
    ],
  },
  {
    canonicalId: "krea/Krea-2-Turbo",
    displayName: "Krea 2 Turbo",
    description: "Text-to-image",
    scope: "image",
    // Gated on the Hub, and the group's only artifact, so a bare click has nothing open to fall
    // through to: the picker must show the gate rather than start a download that 401s.
    artifacts: [bf16Pipeline("krea/Krea-2-Turbo", 18, { gated: true, totalParams: 12820073036 })],
  },
  {
    // 2.6B DiT + Gemma2-2B encoder, ~11 GB bf16-resident (ships fp32, cast on load). Apache-2.0,
    // ungated. No upstream GGUF quants, so the official pipeline is the only artifact.
    canonicalId: "Alpha-VLLM/Lumina-Image-2.0",
    displayName: "Lumina Image 2.0",
    description: "Text-to-image",
    scope: "image",
    artifacts: [bf16Pipeline("Alpha-VLLM/Lumina-Image-2.0", 11, { totalParams: 2609769152 })],
  },
  {
    // 17B dual-stream 2K-native DiT with a Qwen2.5-VL encoder. bf16 only: the QuantStack GGUF
    // consumer route was unpublished, and an entry the Hub cannot serve renders as a one-click
    // download that fails partway through. No vetted replacement: unsloth has an FP8 mirror but
    // no GGUF, and the third-party GGUF repos are a different lineage. At 24 GB the bf16 still
    // fits the 61.6 GB budget, so the group stays visible; the quant ladder is what is gone.
    // The mirror guider components load natively on diffusers 0.39. The retired GGUF repo 404s to an
    // authed request and 401s anonymously; QuantStack itself is alive and still ships its other GGUF
    // repos, so this is one model withdrawn rather than a publisher going away.
    canonicalId: "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    displayName: "HunyuanImage 2.1",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 50, { totalParams: 17425795520 }),
    ],
  },
  {
    // 17B MoE DiT + four text encoders. The MIT repos ship no Llama text_encoder_4, so the backend
    // assembles it from the open unsloth mirror at load time (+16 GB): ~63 GB bf16-resident, a
    // datacenter pick. Full is the undistilled base, Dev and Fast its distillations.
    canonicalId: "HiDream-ai/HiDream-I1-Full",
    displayName: "HiDream I1",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("HiDream-ai/HiDream-I1-Full", 63, { totalParams: 17105733184 }),
      bf16Pipeline("HiDream-ai/HiDream-I1-Dev", 63, {
        label: "BF16 - Dev (distilled)",
        keywords: ["bf16", "dev", "distilled"],
        totalParams: 17105733184,
      }),
      bf16Pipeline("HiDream-ai/HiDream-I1-Fast", 63, {
        label: "BF16 - Fast (distilled)",
        keywords: ["bf16", "fast", "distilled"],
        totalParams: 17105733184,
      }),
    ],
  },
  {
    // No bf16 repo exists for Ideogram 4: -fp8 stores its two DiTs as raw float8 (~46 GB after the
    // bf16 cast); -nf4-diffusers is the bnb-4bit export (~11 GB).
    canonicalId: "ideogram-ai/ideogram-4",
    displayName: "Ideogram 4",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      // Both Ideogram repos are gated on the Hub, so neither can be auto-routed anonymously.
      fp8Pipeline("ideogram-ai/ideogram-4-fp8", 46, { gated: true, totalParams: 9281557760 }),
      bnb4bit("ideogram-ai/ideogram-4-nf4-diffusers", 11, { gated: true, totalParams: 4785317809 }),
    ],
  },
  // SDXL Turbo and Base are different checkpoints with different step/guidance defaults, so two groups.
  {
    canonicalId: "stabilityai/sdxl-turbo",
    displayName: "SDXL Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [bf16Pipeline("stabilityai/sdxl-turbo", 8, { label: "Safetensors", totalParams: 2567463684 })],
  },
  {
    canonicalId: "stabilityai/stable-diffusion-xl-base-1.0",
    displayName: "SDXL Base 1.0",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("stabilityai/stable-diffusion-xl-base-1.0", 8, {
        label: "Safetensors",
        totalParams: 2567463684,
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
        // Cluster measurements at the default 1344x768, 124-frame preset: the lower-GPU tier holds one
        // 66 GB component at a time and keeps the full model in RAM. THESE ARE GiB and the backend
        // estimators they mirror are decimal GB, so the two sets must never be copied across:
        // nvidia.py memory_total_gb and main.py available_gb divide by 1024, while video.py divides
        // runtime bytes by 1_000_000_000. Converted, these are 79.5 / 150.3 and 132.1 / 85.9 GB,
        // matching the estimators' 78.74 / 150 and 132 / 85. Copying the decimal figures across
        // applies the conversion twice and sends capable hosts to GGUF.
        offloadFitTiers: [
          { gpuGb: 74, systemRamGb: 140 },
          { gpuGb: 123, systemRamGb: 80 },
        ],
      }),
      // One official bundle for both denoiser partitions. The GGUF lister labels every variant, so both
      // stay explicit under one repo id.
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
    // The distilled 2.3 release: Lightricks' own bf16/fp8 single-file DiT checkpoints (loaded against
    // the already-trusted LTX-2 base for the VAE / Gemma3 encoder) plus the GGUF quants, which
    // keep the ~50 GB encoder in bf16 so consumer GPUs route to GGUF. Keyed on the artifact that
    // exists: unsloth/LTX-2.3 was never published, and an `unsloth/*` id that is not an artifact
    // clears both owner guards. It still resolves here via the GGUF's suffix-stripped key.
    // unsloth/LTX-2.3 was never published (404), and an `unsloth/*` id that is not an artifact clears
    // both the picker's owner guard and the backend's, so a pick reaching the fall-through was loaded
    // as a pipeline and only died at the Hub. Lightricks/LTX-2.3 IS an artifact below.
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
      // No FP8 artifact: the LTX-2.3 loader refuses the official scaled-FP8 single file (it carries
      // .weight_scale/.input_scale), so a click would start a ~76 GB download that always fails.
      // 21.0B is what the Hub reports, and carrying it keeps the row identical when offline.
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
      // Highest-quality first: pickDefaultArtifact sorts only by FORMAT, so these two bf16 artifacts
      // keep catalog order and the fit loop returns the first that fits. 720p (52 GB) precedes
      // 480p (40 GB), so an 80 GB card picks 720p.
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

// The Audio page's curated list. tts groups load into the main slot via /api/inference/load
// (Orpheus is the only family the llama.cpp TTS path also serves as GGUF); native TTS
// families use their official Transformers/Diffusers interfaces. stt groups map to the
// dictation sidecar models in stt-model-catalog.ts, so their sizes are informational only.
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
    // No GGUF artifact: the llama.cpp TTS path has no csm decode, so CSM runs transformers-only.
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
        // 67 GB is the repository/download footprint, not resident VRAM: the publisher's ModularPipeline
        // path loads in BF16 on a 24 GB CUDA GPU.
        offloadFitTiers: [{ gpuGb: 24, systemRamGb: 0 }],
      }),
    ],
  },
  // Llasa is deliberately absent: it speaks XCodec2 (65,536 <|s_N|> tokens), which is in neither
  // _AUDIO_TOKEN_PATTERNS nor AudioCodecManager, so a curated row loaded and then failed at
  // generation. Unsloth can still TRAIN Llasa (unsloth_Llasa-3B.yaml); this catalog only feeds the
  // Generate picker. Re-add both rows together with an xcodec2 decoder.
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
  // Both sidecars carry tiny/base (GGML_STT_REPOS, STT_MODEL_REPOS) and Voice settings lists them;
  // only this picker was missing them.
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
];


// Artifact/format suffixes stripped (repeatedly, longest-first) off the NAME part of a repo id.
// Owner is preserved: cross-owner merges happen only via the alias tables above.
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

/** Case-preserving display name with artifact suffixes stripped, so uncurated rows read as their
 *  base model. The format badge carries the artifact kind; the load id is untouched. */
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

// Rebuilt only on a new catalog array identity; the curated arrays are module constants, so in
// practice once per catalog.
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
      // An alias also claims its own suffix-stripped key so sibling artifacts of the aliased owner group correctly.
      byKey.set(canonicalKeyFor(alias), group);
    }
    for (const artifact of group.artifacts) {
      byId.set(artifact.repoId.toLowerCase(), group);
      byKey.set(canonicalKeyFor(artifact.repoId), group);
      artifactById.set(artifact.repoId.toLowerCase(), artifact);
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

/** Curated size (bytes) of an exact artifact id; undefined when unsized (every GGUF entry: its
 *  quant ladder self-fits). */
export function curatedSizeBytesFor(
  repoId: string,
  catalog: CatalogGroup[],
): number | undefined {
  const gb = artifactForRepoId(repoId, catalog)?.artifact.approxSizeGb;
  return gb && gb > 0 ? gb * BYTES_PER_GB : undefined;
}

/** Curated parameter count for an exact artifact id, or undefined. A FALLBACK for rows the Hub
 *  listing does not return: callers must prefer the listing's own total. */
export function curatedTotalParamsFor(
  repoId: string,
  catalog: CatalogGroup[],
): number | undefined {
  const params = artifactForRepoId(repoId, catalog)?.artifact.totalParams;
  return params && params > 0 ? params : undefined;
}

/** Curated capabilities for any id belonging to a group that declares them. Same fallback rule as
 *  `curatedTotalParamsFor`: the listing's tags win. Group-level, since every artifact of a
 *  model can do what the model can do. */
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
    // From the group's scope rather than a declaration: every catalog entry is a generator of one or
    // the other, and the scope already says which. Beats the name heuristic, which has to guess
    // a family from a repo id.
    imageGen: declared?.imageGen ?? group.scope === "image",
    videoGen: declared?.videoGen ?? group.scope === "video",
  };
}

/** Human-facing name of an exact curated artifact, including the artifact label when its model has
 *  more than one selectable representation. */
export function curatedDisplayNameFor(
  repoId: string,
  catalog: CatalogGroup[],
  host: HostClass = "unknown",
): string | null {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return null;
  // A row that earns a speed qualifier must read the same closed as open: this helper names the
  // trigger and curatedRowLabelFor names the row, so a divergence would rename the model as
  // the popover opens.
  if (h3PerfSuffix(repoId, host)) {
    return curatedRowLabelFor(repoId, catalog, host)?.name ?? hit.group.displayName;
  }
  return hit.group.artifacts.length > 1
    ? `${hit.group.displayName} (${hit.artifact.label})`
    : hit.group.displayName;
}

// Artifact labels are written as "FORMAT" or "FORMAT - QUALIFIER". The format head and a
// resolution qualifier are chips; anything else stays in the name, since it is the only thing
// telling two rows of one group apart.
const LABEL_PART_SEPARATOR = " - ";
const OFFICIAL_SUFFIX_RE = /\s*\(official\)$/i;
const GGUF_SUFFIX_RE = /-gguf$/i;
const RESOLUTION_RE = /^\d{3,4}p$/i;

/** A curated row as name plus chips. The name used to carry the artifact inside brackets ("MiniMax
 *  H3 (BF16 (official))"), which pushed the part a user scans for behind the part they do
 *  not. Null for ids outside the catalog. */
export function curatedRowLabelFor(
  repoId: string,
  catalog: CatalogGroup[],
  host: HostClass = "unknown",
): { name: string; tags: string[] } | null {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return null;
  // Only where the host can run both rows, so the qualifier compares things the user can pick
  // between rather than advertising a speed they cannot have.
  const perf = h3PerfSuffix(repoId, host);
  const qualify = (name: string) => (perf ? `${name} (${perf})` : name);
  // GGUF reads like a text model's row: the repo name already ends in -GGUF, so a chip would only repeat the suffix.
  if (hit.artifact.format === "gguf") {
    const leaf = hit.artifact.repoId.split("/").pop() ?? hit.artifact.repoId;
    return { name: qualify(GGUF_SUFFIX_RE.test(leaf) ? leaf : `${leaf}-GGUF`), tags: [] };
  }
  // A group with one artifact has nothing to distinguish, so it stays bare.
  if (hit.group.artifacts.length <= 1) return { name: qualify(hit.group.displayName), tags: [] };
  const [format, ...rest] = hit.artifact.label.split(LABEL_PART_SEPARATOR);
  const tags = [format.replace(OFFICIAL_SUFFIX_RE, "").trim()].filter(Boolean);
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
): ModelOption[] {
  const options: ModelOption[] = [];
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      // A host that can only run the native engine is not offered the pipeline rows it would be refused
      // at load. This is the one place the `models` prop is built, so filtering here covers the
      // trigger name and the picker's seed ids together.
      if (!curatedArtifactIsOfferable(artifact.repoId, host)) continue;
      options.push({
        id: artifact.repoId,
        name: curatedDisplayNameFor(artifact.repoId, catalog, host) ?? group.displayName,
        description: `${group.description} - ${artifact.label}`,
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

// Quant-class tokens that should match every GGUF artifact ("q4" finds the group whose GGUF repo publishes Q4_K_M).
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

/** Whether a (lowercased, trimmed) query matches the group: canonical id, display name, artifact
 *  id, label/keyword, or quant-class token. */
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


export interface DeviceBudget {
  /** Total GPU memory in GB (0/undefined = unknown or none). */
  gpuGb: number;
  /** Available system RAM in GB (for the GGUF offload tier). */
  systemRamGb: number;
  /** The user's saved VRAM Budget. Absent falls back to the loader's default. */
  budgetFraction?: number;
  /** GPUs gpuGb sums, for the loader's per-card VRAM reserve. Absent means one. */
  gpuCount?: number;
}

/** GGUF fit, delegated to the one formula the Hub badge already uses. This used to carry its own
 *  rule (0.7 * GPU + 0.7 * RAM, raw file size) whose comment claimed to match `_select_gpus`;
 *  it did not, so chat hid quants the loader would have taken and ignored the VRAM Budget. */
export function classifyGgufFit(
  sizeBytes: number,
  budget: DeviceBudget,
): GgufFitClass {
  // Nothing measured, so a verdict would be invented. Callers that know the budget really is zero
  // decide that for themselves.
  if ((budget.gpuGb || 0) <= 0 && (budget.systemRamGb || 0) <= 0) return "fits";
  if (sizeBytes <= 0) return "fits";
  return classifyGgufFitForDevice(sizeBytes, budget);
}

/** Fit rule for a GGUF the IMAGES / VIDEO / AUDIO pickers offer, the one case the shared llama.cpp
 *  formula must not judge: those loads go through the diffusion backend, whose budget is free
 *  memory minus a reserve at a 0.85 margin (`diffusion_memory.py`) and which cannot offload.
 *  On a 64 GiB Mac that planner allows about 43.5 GiB where llama.cpp allows 62.1; this rule
 *  allows 44.8. It is not the diffusion planner either, only the closer of the two. */
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
  // "partial", where this rule used to say "tight": the state it describes is a spill out of the
  // card. Tight now means a full GPU load with no room to spare.
  return gb <= totalBudgetGb ? "partial" : "oom";
}

/** Whether a verdict still loads. Only `oom` clears neither budget; `marginal` and `partial` run,
 *  the second by offloading to CPU. */
export function ggufFitRuns(fit: GgufFitClass): boolean {
  return fit !== "oom";
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
  // No budget knowledge at all: trust the repo default.
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
  /** Whether an artifact repo already has weights on disk. */
  isDownloaded: (repoId: string) => boolean;
}

const FORMAT_QUALITY: Record<ArtifactFormat, number> = {
  bf16: 0,
  fp8: 1,
  "bnb-4bit": 2,
  gguf: 3,
};

function fitsArtifactBudget(artifact: ModelArtifact, budget: DeviceBudget): boolean {
  if (artifact.offloadFitTiers?.length) {
    return artifact.offloadFitTiers.some(
      (tier) => budget.gpuGb >= tier.gpuGb && budget.systemRamGb >= tier.systemRamGb,
    );
  }
  if (artifact.approxSizeGb === undefined) return false;
  return artifact.approxSizeGb <= budget.gpuGb * 0.7;
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
      (a) => a.format !== "gguf" && fitsArtifactBudget(a, input),
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
    // Skip a gated, NOT-downloaded artifact: auto-routing there fails the download without
    // license/token access. The downloaded branch above still returns gated artifacts.
    if (artifact.format !== "gguf" && !artifact.gated && fitsArtifactBudget(artifact, input)) {
      return artifact;
    }
  }
  if (ggufArtifact) return ggufArtifact;
  return artifacts.sort(
    (a, b) => (a.approxSizeGb ?? Infinity) - (b.approxSizeGb ?? Infinity),
  )[0];
}

/** Whether ONE curated artifact loads on this device, by the rule `pickDefaultArtifact` routes
 *  with, since a row click loads that exact artifact. System RAM is not part of a
 *  discrete-GPU budget: a pipeline goes wholly on the card unless the catalog states a
 *  measured offload tier or the loader falls back to CPU, which only transcription does. A
 *  unified-memory host reports RAM and no GPU, and there the RAM is the card. Undefined where
 *  nothing can be judged, so the caller shows no verdict rather than a wrong one. */
export function curatedArtifactFitsDevice(
  repoId: string,
  catalog: CatalogGroup[],
  budget: DeviceBudget,
): boolean | undefined {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit || hit.artifact.format === "gguf") return undefined;
  const { group, artifact } = hit;
  if (budget.gpuGb <= 0 && budget.systemRamGb <= 0) return undefined;
  if (artifact.offloadFitTiers?.length) return fitsArtifactBudget(artifact, budget);
  if (artifact.approxSizeGb === undefined) return undefined;
  // Transcription retries a failed device load on CPU (stt_sidecar.py), so RAM is a real budget
  // there, but the WHOLE model goes to whichever device it lands on, so it is the larger of
  // the two and not their sum. An image, video or TTS load rejects CPU offload.
  const deviceGb =
    group.task === "stt"
      ? Math.max(budget.gpuGb, budget.systemRamGb)
      : budget.gpuGb > 0
        ? budget.gpuGb
        : budget.systemRamGb;
  return artifact.approxSizeGb <= deviceGb * 0.7;
}

/** Whether the "fit on device" toggle keeps a group, including measured offload tiers when an
 *  artifact provides them. */
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
    // A GGUF quant ladder self-fits (llama-server offloads), so it is always a runnable fallback,
    // matching pickDefaultArtifact.
    if (a.format === "gguf") return true;
    if (a.offloadFitTiers?.length) {
      return a.offloadFitTiers.some(
        (tier) => budget.gpuGb >= tier.gpuGb && budget.systemRamGb >= tier.systemRamGb,
      );
    }
    return a.approxSizeGb !== undefined && a.approxSizeGb <= budgetGb;
  });
}
