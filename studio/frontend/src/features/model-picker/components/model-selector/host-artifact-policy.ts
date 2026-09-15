// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Media-picker runtime capability. Dense quant is reported by the backend because the backend
 *  name alone cannot distinguish unsupported accelerators. */
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
  // Mac outranks the backend string. Apple GPUs report as available and Unsloth may name the
  // backend mlx or cpu, but no Mac can place a Modular Diffusers workflow: it needs
  // mem_get_info, which torch.mps does not expose, and video.py refuses the load.
  if (deviceType === "mac") return "gguf-only";
  const backend = (deviceBackend ?? "").trim().toLowerCase();
  if (!(backend && budgetKnown)) return "unknown";
  if (GGUF_ONLY_BACKENDS.has(backend)) return "gguf-only";
  if (ACCELERATED_BACKENDS.has(backend)) {
    return denseQuantSupported ? "dense-quant" : "accelerated";
  }
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

/** Speed qualifier for H3 artifacts. It omits precision because auto may select INT8 or retain BF16. */
export function h3PerfSuffix(repoId: string, host: HostClass): string | null {
  if (!hostIsAccelerated(host)) return null;
  const id = repoId.trim().toLowerCase();
  if (id === H3_PIPELINE_ID) return "Fast";
  if (id === H3_GGUF_ID) return "Slow";
  return null;
}
