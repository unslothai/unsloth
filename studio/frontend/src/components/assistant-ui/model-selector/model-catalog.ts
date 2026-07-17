// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One canonical name per diffusion model, with its published artifacts (GGUF quants, prequant
// FP8 / bnb-4bit repos, official BF16 pipelines) as a second level, plus a deterministic router
// picking the best artifact for the device. Pure helpers -- no React/DOM deps, so easy to test
// (see model-catalog.check.ts, run via `npm run catalog:check`).

import type { ModelOption } from "./types";

export type ArtifactFormat = "gguf" | "fp8" | "bnb-4bit" | "bf16";
export type LoadKind = "gguf" | "single_file" | "pipeline";

export interface ModelArtifact {
  /** Exact artifact repo id (the pre-grouping id -- stays loadable/searchable). */
  repoId: string;
  format: ArtifactFormat;
  loadKind: LoadKind;
  /** single_file loads name their exact checkpoint inside the repo. */
  filename?: string;
  /** Second-level row label ("GGUF", "FP8", "BF16 (official)", "BF16 - 720p"). */
  label: string;
  /** Curated resident-size estimate for routing. Omitted = unknown: never
   *  auto-picked unless already downloaded. GGUF artifacts omit it too -- their
   *  per-quant ladder self-fits via pickDefaultQuant. */
  approxSizeGb?: number;
  /** Extra search tokens beyond the id/label ("4bit", "nf4", ...). */
  keywords?: readonly string[];
  /** Gated on the Hub (license acceptance + token needed to download). A bare group
   *  click must not auto-route to it when it isn't already downloaded -- the download
   *  would fail for a user without access -- so the not-downloaded ladder skips it and
   *  falls through to an open artifact (e.g. the GGUF). An already-downloaded gated
   *  artifact is still returned (the user clearly has access). */
  gated?: boolean;
}

export interface CatalogGroup {
  /** Canonical display id, owner spelled once ("unsloth/Qwen-Image-2512"). */
  canonicalId: string;
  displayName: string;
  /** Row meta line ("Text-to-image", "Image editing", "Text-to-video with audio"). */
  description: string;
  scope: "image" | "video";
  /** Descending quality order: bf16, fp8, bnb-4bit, gguf. The router walks it. */
  artifacts: ModelArtifact[];
  /** Cross-owner ids that resolve to this group. Suffix stripping never merges
   *  two owners on its own, so arbitrary cached repos cannot be mis-grouped. */
  aliases?: readonly string[];
}

// ── artifact constructors (keep the data tables terse) ─────────────────────────

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

const fp8Single = (
  repoId: string,
  filename: string,
  approxSizeGb: number,
): ModelArtifact => ({
  repoId,
  format: "fp8",
  loadKind: "single_file",
  filename,
  label: "FP8",
  approxSizeGb,
  keywords: ["fp8", "float8"],
});

const fp8Pipeline = (repoId: string, approxSizeGb: number): ModelArtifact => ({
  repoId,
  format: "fp8",
  loadKind: "pipeline",
  label: "FP8",
  approxSizeGb,
  keywords: ["fp8", "float8"],
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

// A bf16 single-file DiT checkpoint (e.g. Lightricks' distilled LTX-2.3): loads
// via from_single_file against the family base repo for the VAE / text encoder,
// same load path as the fp8 single-file checkpoints.
const bf16Single = (
  repoId: string,
  filename: string,
  approxSizeGb: number,
): ModelArtifact => ({
  repoId,
  format: "bf16",
  loadKind: "single_file",
  filename,
  label: "BF16 (official)",
  approxSizeGb,
  keywords: ["bf16", "safetensors", "full precision"],
});

// ── curated catalogs ────────────────────────────────────────────────────────────
// Sizes are steady resident estimates (GB) used only for routing; a missing size means "never
// auto-pick unless downloaded". GGUF entries carry no size -- the quant ladder
// (pickDefaultQuant) sizes the individual .gguf files.

export const IMAGE_CATALOG: CatalogGroup[] = [
  {
    canonicalId: "unsloth/Z-Image-Turbo",
    displayName: "Z-Image-Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("Tongyi-MAI/Z-Image-Turbo", 30),
      bnb4bit("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", 8),
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
    artifacts: [
      bf16Pipeline("Qwen/Qwen-Image-2512", 54),
      fp8Single(
        "unsloth/Qwen-Image-2512-FP8",
        "qwen-image-2512-fp8.safetensors",
        24,
      ),
      bnb4bit("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", 14),
      gguf("unsloth/Qwen-Image-2512-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/Qwen-Image",
    displayName: "Qwen-Image",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("Qwen/Qwen-Image", 54),
      gguf("unsloth/Qwen-Image-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-schnell",
    displayName: "FLUX.1 schnell",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("black-forest-labs/FLUX.1-schnell", 32),
      gguf("unsloth/FLUX.1-schnell-GGUF"),
    ],
  },
  {
    canonicalId: "unsloth/FLUX.1-dev",
    displayName: "FLUX.1 dev",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      // FLUX.1-dev is gated (license acceptance + token); FLUX.1-schnell above is Apache-2.0.
      bf16Pipeline("black-forest-labs/FLUX.1-dev", 32, { gated: true }),
      gguf("unsloth/FLUX.1-dev-GGUF"),
    ],
  },
  {
    // Krea's guidance-distilled FLUX.1-dev finetune ("opinionated aesthetics"): same
    // arch/layout as FLUX.1-dev, so it runs under the existing flux.1 family. The base
    // repo is gated like dev; QuantStack publishes the open GGUF quants.
    canonicalId: "black-forest-labs/FLUX.1-Krea-dev",
    displayName: "FLUX.1 Krea dev",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("black-forest-labs/FLUX.1-Krea-dev", 32, { gated: true }),
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
      bf16Pipeline("Qwen/Qwen-Image-Edit-2511", 54),
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
      bf16Pipeline("black-forest-labs/FLUX.1-Kontext-dev", 32, { gated: true }),
      gguf("unsloth/FLUX.1-Kontext-dev-GGUF"),
    ],
  },
  {
    canonicalId: "krea/Krea-2-Turbo",
    displayName: "Krea 2 Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [bf16Pipeline("krea/Krea-2-Turbo", 18)],
  },
  {
    // No bf16 repo exists for Ideogram 4: -fp8 stores its two DiTs as raw
    // float8 (~46 GB resident after the bf16 cast); -nf4-diffusers is the
    // bnb-4bit export (~11 GB).
    canonicalId: "ideogram-ai/ideogram-4",
    displayName: "Ideogram 4",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      fp8Pipeline("ideogram-ai/ideogram-4-fp8", 46),
      bnb4bit("ideogram-ai/ideogram-4-nf4-diffusers", 11),
    ],
  },
  // SDXL Turbo and Base are different checkpoints with different step/guidance
  // defaults -- two groups, not two formats of one model.
  {
    canonicalId: "stabilityai/sdxl-turbo",
    displayName: "SDXL Turbo",
    description: "Text-to-image",
    scope: "image",
    artifacts: [bf16Pipeline("stabilityai/sdxl-turbo", 8, { label: "Safetensors" })],
  },
  {
    canonicalId: "stabilityai/stable-diffusion-xl-base-1.0",
    displayName: "SDXL Base 1.0",
    description: "Text-to-image",
    scope: "image",
    artifacts: [
      bf16Pipeline("stabilityai/stable-diffusion-xl-base-1.0", 8, {
        label: "Safetensors",
      }),
    ],
  },
];

export const VIDEO_CATALOG: CatalogGroup[] = [
  {
    // The distilled 2.3 release: Lightricks' own bf16/fp8 single-file DiT checkpoints (loaded
    // against the LTX-2 base for the VAE / Gemma3 text encoder, both already trusted) plus the GGUF
    // quants. The single-file checkpoints keep the ~50 GB Gemma3-27B encoder in bf16, so their
    // resident footprint is datacenter-scale; consumer GPUs route to GGUF, which offloads.
    canonicalId: "unsloth/LTX-2.3",
    displayName: "LTX 2.3 distilled",
    description: "Text-to-video with audio",
    scope: "video",
    artifacts: [
      bf16Single(
        "Lightricks/LTX-2.3",
        "ltx-2.3-22b-distilled.safetensors",
        90,
      ),
      // No FP8 artifact: the LTX-2.3 loader refuses the official scaled-FP8 single file (it carries
      // .weight_scale/.input_scale tensors) and points users to GGUF/BF16, so advertising it would
      // route a bare click or manual pick to a ~76 GB download that always fails on load.
      gguf("unsloth/LTX-2.3-GGUF"),
    ],
  },
  {
    canonicalId: "Lightricks/LTX-2",
    displayName: "LTX 2 (base)",
    description: "Text-to-video with audio",
    scope: "video",
    artifacts: [bf16Pipeline("Lightricks/LTX-2", 90)],
  },
  {
    canonicalId: "Wan-AI/Wan2.2-TI2V-5B",
    displayName: "Wan 2.2 TI2V 5B",
    description: "Text-to-video 720p",
    scope: "video",
    artifacts: [bf16Pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 30)],
  },
  {
    canonicalId: "Wan-AI/Wan2.2-T2V-A14B",
    displayName: "Wan 2.2 T2V A14B (MoE)",
    description: "Text-to-video, dual-expert",
    scope: "video",
    artifacts: [bf16Pipeline("Wan-AI/Wan2.2-T2V-A14B-Diffusers", 114)],
  },
  {
    canonicalId: "hunyuanvideo-community/HunyuanVideo-1.5",
    displayName: "HunyuanVideo 1.5",
    description: "Text-to-video",
    scope: "video",
    artifacts: [
      // Highest-quality first: pickDefaultArtifact sorts only by FORMAT, so among these two bf16
      // artifacts it keeps catalog order and the fit loop returns the FIRST that fits. The 720p (52
      // GB) must precede the 480p (40 GB) so a bare click on a GPU where 720p fits (e.g. 80 GB,
      // 0.7*budget=56) picks 720p, falling back to 480p only on smaller cards.
      bf16Pipeline("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", 52, {
        label: "BF16 - 720p",
        keywords: ["bf16", "720p"],
      }),
      bf16Pipeline("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", 40, {
        label: "BF16 - 480p",
        keywords: ["bf16", "480p"],
      }),
    ],
  },
];

// ── canonical keys and lookups ──────────────────────────────────────────────────

// Artifact/format suffixes stripped (repeatedly, longest-first) off the NAME part
// of a repo id to reach its generic key. Owner is preserved: cross-owner merges
// happen only through the explicit alias tables above.
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

/** Owner-preserving generic key: lowercase, artifact suffixes stripped off the
 *  name part. "unsloth/Qwen-Image-2512-GGUF" -> "unsloth/qwen-image-2512". */
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

/** Case-preserving display name: the artifact suffixes stripped off the name part
 *  while keeping the original casing and the owner prefix. Used by the diffusion
 *  pickers so rows OUTSIDE the curated catalog (arbitrary hub or cached repos)
 *  still read as their base model name ("ERNIE-Image-Turbo-GGUF" ->
 *  "ERNIE-Image-Turbo"); the format badge next to the row carries the artifact
 *  kind. The id used for loading is never touched. */
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

// Rebuilt only when a new catalog array identity shows up (the curated arrays
// are module constants, so in practice this builds twice: images + video).
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
      // An alias also claims its own suffix-stripped key so sibling artifacts
      // of the aliased owner group correctly (Qwen/Qwen-Image-2512-FP8 etc.).
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

/** The group a repo id belongs to, or null for unknown repos (callers render
 *  those ungrouped, exactly as before the catalog existed). */
export function groupForRepoId(
  repoId: string,
  catalog: CatalogGroup[],
): CatalogGroup | null {
  const index = indexFor(catalog);
  const lowered = repoId.trim().toLowerCase();
  return index.byId.get(lowered) ?? index.byKey.get(canonicalKeyFor(lowered)) ?? null;
}

/** The exact curated artifact for a repo id (null when the repo only matches a
 *  group by key/alias -- e.g. a cached quant repo we know but did not curate). */
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

/** Back-compat: the flat ModelOption list the ModelSelector's `models` prop
 *  expects, one option per ARTIFACT (old ids keep working everywhere). */
export function catalogToModelOptions(catalog: CatalogGroup[]): ModelOption[] {
  const options: ModelOption[] = [];
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      options.push({
        id: artifact.repoId,
        name:
          group.artifacts.length > 1
            ? `${group.displayName} (${artifact.label})`
            : group.displayName,
        description: `${group.description} - ${artifact.label}`,
        isGguf: artifact.format === "gguf",
      });
    }
  }
  return options;
}

/** How to load a curated artifact: replaces the pages' SAFETENSORS_MODELS /
 *  PIPELINE_MODELS lookup tables. Null for unknown ids (GGUF picks carry their
 *  own variant metadata; local paths and hub GGUFs resolve elsewhere). */
export function loadSpecFor(
  repoId: string,
  catalog: CatalogGroup[],
): { kind: LoadKind; filename?: string } | null {
  const hit = artifactForRepoId(repoId, catalog);
  if (!hit) return null;
  return { kind: hit.artifact.loadKind, filename: hit.artifact.filename };
}

// Quant-class tokens that should match every GGUF artifact ("q4" finds the
// group whose GGUF repo publishes Q4_K_M, etc.).
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

/** Whether a (lowercased, trimmed) query matches the group: canonical id,
 *  display name, any artifact id, any label/keyword, or a quant-class token. */
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

// ── device fit + routing ─────────────────────────────────────────────────────────

export interface DeviceBudget {
  /** Total GPU memory in GB (0/undefined = unknown or none). */
  gpuGb: number;
  /** Available system RAM in GB (for the GGUF offload tier). */
  systemRamGb: number;
}

/** GGUF fit classification matching llama-server's _select_gpus logic:
 *  fits = model <= 0.7 * GPU; tight = fits with 0.7 * RAM offload; oom = neither.
 *  Extracted from GgufVariantExpander so the badge and the router agree. */
export function classifyGgufFit(
  sizeBytes: number,
  budget: DeviceBudget,
): "fits" | "tight" | "oom" {
  const gpuBudgetGb = (budget.gpuGb || 0) * 0.7;
  const totalBudgetGb = gpuBudgetGb + (budget.systemRamGb || 0) * 0.7;
  if (totalBudgetGb <= 0) return "fits";
  const gb = sizeBytes / 1024 ** 3;
  if (gb <= 0 || gb <= gpuBudgetGb) return "fits";
  if (gpuBudgetGb <= 0) return gb <= totalBudgetGb ? "fits" : "oom";
  if (gb <= totalBudgetGb) return "tight";
  return "oom";
}

export interface QuantVariant {
  quant: string;
  filename: string;
  size_bytes: number;
  downloaded?: boolean;
}

/** The quant a bare group/repo click should load. Preference order:
 *  largest downloaded non-OOM quant, the repo default when non-OOM, the largest
 *  fitting quant, then the smallest overall (closest to running). Mirrors the
 *  expander's effectiveRecommended, extended to prefer what is already on disk. */
export function pickDefaultQuant(
  variants: QuantVariant[],
  defaultVariant: string | null,
  budget: DeviceBudget,
): QuantVariant | null {
  if (!variants || variants.length === 0) return null;
  const totalBudgetGb =
    (budget.gpuGb || 0) * 0.7 + (budget.systemRamGb || 0) * 0.7;
  const downloadedFitting = variants
    .filter((v) => v.downloaded && classifyGgufFit(v.size_bytes, budget) !== "oom")
    .sort((a, b) => b.size_bytes - a.size_bytes);
  if (downloadedFitting.length > 0) return downloadedFitting[0];
  const byQuant = (quant: string | null) =>
    quant ? (variants.find((v) => v.quant === quant) ?? null) : null;
  // No budget knowledge at all: trust the repo default.
  if (totalBudgetGb <= 0) return byQuant(defaultVariant) ?? variants[0];
  const defaultV = byQuant(defaultVariant);
  if (defaultV && classifyGgufFit(defaultV.size_bytes, budget) !== "oom") {
    return defaultV;
  }
  const fitting = variants
    .filter((v) => classifyGgufFit(v.size_bytes, budget) !== "oom")
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

function fitsResident(artifact: ModelArtifact, gpuGb: number): boolean {
  if (artifact.approxSizeGb === undefined) return false;
  return artifact.approxSizeGb <= gpuGb * 0.7;
}

/** The artifact a bare group click loads. Deterministic ladder:
 *  1. Downloaded first: the highest-quality downloaded artifact that fits the
 *     0.7 * GPU budget; else a downloaded GGUF (its quant ladder self-fits);
 *     else the smallest-footprint downloaded artifact.
 *  2. No budget known: the GGUF artifact when the group has one (the backend
 *     GGUF path plans offload itself and the Precision auto ladder still
 *     upgrades capable GPUs), else the first artifact.
 *  3. Best sized artifact that fits resident, walking descending quality
 *     (BF16 official, FP8, bnb-4bit). Unknown sizes never auto-picked.
 *  4. Fallback: GGUF, else the smallest-footprint artifact. */
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
      (a) => a.format !== "gguf" && fitsResident(a, input.gpuGb),
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
    // Skip a gated, NOT-downloaded artifact: auto-routing to it would fail the download for a
    // user without license/token access, so fall through to an open artifact (the GGUF below).
    // The downloaded branch above still returns a gated artifact the user already fetched.
    if (artifact.format !== "gguf" && !artifact.gated && fitsResident(artifact, input.gpuGb)) {
      return artifact;
    }
  }
  if (ggufArtifact) return ggufArtifact;
  return artifacts.sort(
    (a, b) => (a.approxSizeGb ?? Infinity) - (b.approxSizeGb ?? Infinity),
  )[0];
}

/** Whether the "fit on device" toggle should keep a catalog group. A group stays
 *  visible when at least one artifact can actually run here: one already on disk
 *  (the user has it), a GGUF (its per-quant ladder self-fits and the backend GGUF
 *  path plans its own offload), or a sized artifact whose curated footprint is
 *  within the 0.7*GPU + 0.7*RAM budget. A group of only over-budget (or unsized)
 *  full-precision artifacts -- e.g. the LTX-2 base at 90 GB or Wan2.2-A14B at
 *  114 GB on a consumer card -- is hidden, so a bare click on a fit-filtered list
 *  can no longer start an OOM load the toggle was meant to hide. An unknown device
 *  budget keeps everything (we cannot tell). Mirrors fitsResident/the Recommended
 *  fit predicate, extended across a group's formats. */
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
    // A GGUF's quant ladder self-fits (llama-server offloads), so it is always a
    // runnable fallback -- matching pickDefaultArtifact returning it on any budget.
    if (a.format === "gguf") return true;
    return a.approxSizeGb !== undefined && a.approxSizeGb <= budgetGb;
  });
}
