// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { normalizeDenseQuantSchemes } from "../../../../lib/dense-quant-schemes.ts";

/** Media-picker runtime capability. The backend reports dense quant, since its name alone cannot
 *  distinguish unsupported accelerators. */
/** "cpu-only" is a "gguf-only" host with no accelerator at all (backend cpu): it runs GGUF image models on the native
 *  sd.cpp engine only, so a family that has a GGUF row offers nothing else there (``curatedArtifactIsOfferable``). */
export type HostClass = "unknown" | "gguf-only" | "cpu-only" | "accelerated" | "dense-quant";

const ACCELERATED_BACKENDS = new Set(["cuda", "rocm", "xpu"]);

const GGUF_ONLY_BACKENDS = new Set(["mlx", "cpu"]);

export function classifyHost({
  deviceType,
  deviceBackend,
  budgetKnown,
  denseQuantSupported,
}: {
  deviceType?: string | null;
  deviceBackend?: string | null;
  budgetKnown?: boolean;
  denseQuantSupported?: boolean;
}): HostClass {
  const backend = (deviceBackend ?? "").trim().toLowerCase();
  // Backend outranks the OS: deviceType can be the browser's platform (config/env.ts).
  if (backend && budgetKnown && ACCELERATED_BACKENDS.has(backend)) {
    return denseQuantSupported ? "dense-quant" : "accelerated";
  }
  // Modular Diffusers needs mem_get_info, which torch.mps lacks.
  if (deviceType === "mac") return "gguf-only";
  if (!(backend && budgetKnown)) return "unknown";
  if (backend === "cpu") return "cpu-only";
  if (GGUF_ONLY_BACKENDS.has(backend)) return "gguf-only";
  // An unrecognised backend is a new accelerator, not a CPU.
  return "unknown";
}

/** No GPU dense path: a Mac (MLX / MPS) or a CPU-only host. */
export function hostIsGgufOnly(host: HostClass): boolean {
  return host === "gguf-only" || host === "cpu-only";
}

export function hostIsAccelerated(host: HostClass): boolean {
  return host === "accelerated" || host === "dense-quant";
}

export function hostRunsDenseQuant(host: HostClass): boolean {
  return host === "dense-quant";
}

/** The two H3 rows differ by roughly 10x in throughput. */
const H3_PIPELINE_ID = "minimaxai/minimax-h3";

/** Refused without an accelerator, keyed by id: MiniMax-H3's Modular workflow is unplaceable
  *  (enable_auto_cpu_offload needs mem_get_info), and the MiniMax-H3-Image pipeline streams through
  *  CUDA / ROCm; its GGUF row runs there instead. */
const UNPLACEABLE_WITHOUT_ACCELERATOR = new Set([H3_PIPELINE_ID, "minimaxai/minimax-h3-image"]);

/** The part of a catalog group the CPU-only rule reads (structural, so this module imports no catalog types). */
export interface OfferableGroup {
  scope: string;
  artifacts: ReadonlyArray<{ repoId: string; format: string }>;
  nativeGgufRoute?: boolean;
}

/** Only browse rows are filtered; a model already on disk keeps its row. On a CPU-only host an image / video family
 *  with a GGUF row offers only its GGUF rows: its bf16 pipeline ("Fast FP8" on a GPU), FP8 / INT8 / NVFP4 / bnb rows
 *  would load dense weights on torch CPU (FLUX.3 Klein: 20.3 GB), and the backend refuses them there. A family with a
 *  native route but no curated GGUF row (``nativeGgufRoute``) offers nothing; families with no GGUF route keep theirs. */
export function curatedArtifactIsOfferable(
  repoId: string,
  host: HostClass,
  group?: OfferableGroup | null,
): boolean {
  if (!hostIsGgufOnly(host)) return true;
  if (UNPLACEABLE_WITHOUT_ACCELERATOR.has(repoId.trim().toLowerCase())) return false;
  if (host !== "cpu-only" || !group || (group.scope !== "image" && group.scope !== "video")) return true;
  const id = repoId.trim().toLowerCase();
  const artifact = group.artifacts.find((a) => a.repoId.trim().toLowerCase() === id);
  if (!artifact || artifact.format === "gguf") return true;
  return !(group.nativeGgufRoute || group.artifacts.some((a) => a.format === "gguf"));
}

const H3_GGUF_ID = "unsloth/minimax-h3-gguf";

/** Dense 8-bit schemes that share the one user-facing label; see ``densePerfSuffix``. */
const DENSE_EIGHT_BIT = new Set(["int8", "fp8", "mxfp8"]);

export function densePerfSuffix(
  denseQuantSchemes?: readonly string[] | null,
): string {
  const scheme = normalizeDenseQuantSchemes(denseQuantSchemes)[0];
  if (!scheme) return "Fast";
  // Names the fast tier, not the scheme: 8-bit schemes all render as FP8 so reordering does not rename.
  return DENSE_EIGHT_BIT.has(scheme) ? "Fast FP8" : `Fast ${scheme.toUpperCase()}`;
}

/** A GGUF runs the native engine and is the slow row beside a dense one; null off an accelerator. */
export function ggufPerfSuffix(host: HostClass): string | null {
  return hostIsAccelerated(host) ? "Slow" : null;
}

/** An "unknown" host keeps the full list so an older backend does not lose controls. */
export function hostOffersDensePrecision(host: HostClass): boolean {
  return !hostIsGgufOnly(host);
}

export function h3PerfSuffix(
  repoId: string,
  host: HostClass,
  denseQuantSchemes?: readonly string[] | null,
): string | null {
  if (!hostIsAccelerated(host)) return null;
  const id = repoId.trim().toLowerCase();
  if (id === H3_PIPELINE_ID) return densePerfSuffix(denseQuantSchemes);
  if (id === H3_GGUF_ID) return "Slow";
  return null;
}

/** Video pipelines whose fp8 / int8 load runs a quantized DiT (a hosted pre-quantized checkpoint, else quantized at
 *  load). Other video rows stay bare: the dense-quant row rule is image-only. */
const DENSE_QUANT_VIDEO_PIPELINES = new Set(["lightricks/ltx-2.5-diffusers"]);

/** Speed qualifier for those video rows, from the same host scheme list as the image rows. */
export function videoDensePerfSuffix(
  repoId: string,
  host: HostClass,
  denseQuantSchemes?: readonly string[] | null,
): string | null {
  if (!hostRunsDenseQuant(host)) return null;
  return DENSE_QUANT_VIDEO_PIPELINES.has(repoId.trim().toLowerCase())
    ? densePerfSuffix(denseQuantSchemes)
    : null;
}
