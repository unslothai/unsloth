// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { normalizeDenseQuantSchemes } from "../../../../lib/dense-quant-schemes.ts";

/** Media-picker runtime capability. The backend reports dense quant, since its name alone cannot
 *  distinguish unsupported accelerators. */
export type HostClass = "unknown" | "gguf-only" | "accelerated" | "dense-quant";

/** Backends whose diffusion pipelines the media pages can place. */
const ACCELERATED_BACKENDS = new Set(["cuda", "rocm", "xpu"]);

/** Backends that can only run the native GGUF engine. */
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
  /** Backend-reported dense quant capability. */
  denseQuantSupported?: boolean;
}): HostClass {
  const backend = (deviceBackend ?? "").trim().toLowerCase();
  // A resolved accelerated backend outranks the OS, because no Mac reports one: on macOS the
  // backend resolves to mps, mlx or cpu. deviceType can still be the BROWSER's platform -- it
  // falls back to detectLocalPlatform() until an authenticated reply carries device_type
  // (config/env.ts) -- so a Mac browser driving a remote CUDA server would otherwise classify
  // that Linux host as gguf-only.
  if (backend && budgetKnown && ACCELERATED_BACKENDS.has(backend)) {
    return denseQuantSupported ? "dense-quant" : "accelerated";
  }
  // Mac outranks the rest of the backend string. Apple GPUs report as available and Studio may
  // name the backend mlx or cpu depending on what torch found, but no Mac can place a Modular
  // Diffusers workflow: it needs mem_get_info, which torch.mps does not expose, and video.py
  // refuses the load.
  if (deviceType === "mac") return "gguf-only";
  if (!(backend && budgetKnown)) return "unknown";
  if (GGUF_ONLY_BACKENDS.has(backend)) return "gguf-only";
  // An unrecognised backend is a new accelerator, not a CPU. Show what we show today.
  return "unknown";
}

/** Whether the host can place a diffusion pipeline. */
export function hostIsAccelerated(host: HostClass): boolean {
  return host === "accelerated" || host === "dense-quant";
}

/** Whether the dense torchao transformer quant can engage on this host. */
export function hostRunsDenseQuant(host: HostClass): boolean {
  return host === "dense-quant";
}

/** The H3 group, whose two rows differ by roughly 10x in throughput. */
const H3_PIPELINE_ID = "minimaxai/minimax-h3";

/** The curated artifacts a host without an accelerator is refused at load, by repo id. Keyed on
 *  the id, NOT on `format !== "gguf"`: everything else in the catalogs loads here, since the
 *  diffusion pipelines run on MPS, video_capability() certifies Apple Silicon, and the STT rows
 *  run through the whisper.cpp sidecar. Only MiniMax-H3's Modular Diffusers workflow is
 *  genuinely unplaceable, since enable_auto_cpu_offload needs mem_get_info. */
const UNPLACEABLE_WITHOUT_ACCELERATOR = new Set([H3_PIPELINE_ID]);

/** Whether a curated artifact is worth offering on this host. Only browse rows are filtered. A
 *  model already on disk keeps its row wherever it came from. */
export function curatedArtifactIsOfferable(
  repoId: string,
  host: HostClass,
): boolean {
  if (host !== "gguf-only") return true;
  return !UNPLACEABLE_WITHOUT_ACCELERATOR.has(repoId.trim().toLowerCase());
}

const H3_GGUF_ID = "unsloth/minimax-h3-gguf";

/** The speed qualifier for a dense-quant row, naming the precision that will run. An empty list (an
 *  older backend) falls back to a bare "Fast". Only the first, best entry is read. */
/** Dense 8-bit schemes that share the one user-facing label; see ``densePerfSuffix``. */
const DENSE_EIGHT_BIT = new Set(["int8", "fp8", "mxfp8"]);

export function densePerfSuffix(
  denseQuantSchemes?: readonly string[] | null,
): string {
  const scheme = normalizeDenseQuantSchemes(denseQuantSchemes)[0];
  if (!scheme) return "Fast";
  // The row names the fast TIER, not the scheme the load settles on: int8 and mxfp8 render as FP8,
  // the name users know for the 8-bit fast path, so reordering the ladder does not rename the model.
  // The loaded-models card's resolved record still names the real scheme. A 4-bit scheme names
  // itself, being a different quality tier.
  return DENSE_EIGHT_BIT.has(scheme) ? "Fast FP8" : `Fast ${scheme.toUpperCase()}`;
}

/** Speed qualifier for a GGUF diffusion row. A GGUF runs the native engine, which has no
 *  low-precision tensor-core path and no compiled dense transformer, so it is the slow row wherever
 *  a dense row sits beside it. Null off an accelerator, where it is the only row that runs. */
export function ggufPerfSuffix(host: HostClass): string | null {
  return hostIsAccelerated(host) ? "Slow" : null;
}

/** Whether the Precision control should offer the dense low-precision schemes. A Mac or CPU-only host
 *  is refused them at load. An "unknown" host keeps the full list, since an older backend reporting
 *  no capability must not lose controls it can honour. */
export function hostOffersDensePrecision(host: HostClass): boolean {
  return host !== "gguf-only";
}

/** Speed qualifier for H3 artifacts, naming its precision from the same host scheme list. */
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
