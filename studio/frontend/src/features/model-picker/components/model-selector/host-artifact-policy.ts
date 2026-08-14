// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the host can actually run, as opposed to what fits on it.
 *
 * Capability, not size: `curatedArtifactFitsDevice` already answers "is there room", and the two
 * must stay apart. A 128 GB Mac has room for the H3 pipeline and still cannot load it.
 *
 * `unknown` is not a formality. The GPU hook starts at `available: false, budgetKnown: false`, so
 * a boolean would read every first render as CPU-only and blink the non-GGUF rows out and back on
 * a real GPU host.
 */
export type HostClass = "unknown" | "gguf-only" | "accelerated";

/** Backends whose diffusion pipelines the media pages can place. */
const ACCELERATED_BACKENDS = new Set(["cuda", "rocm", "xpu"]);

/** Backends that can only run the native GGUF engine. */
const GGUF_ONLY_BACKENDS = new Set(["mlx", "cpu"]);

export function classifyHost({
  deviceType,
  deviceBackend,
  budgetKnown,
}: {
  deviceType?: string | null;
  deviceBackend?: string | null;
  budgetKnown?: boolean;
}): HostClass {
  // Mac outranks the backend string. Apple GPUs report as available and Studio may name the
  // backend mlx or cpu depending on what torch found, but no Mac can place a Modular Diffusers
  // workflow: it needs mem_get_info, which torch.mps does not expose. video.py refuses the load.
  if (deviceType === "mac") return "gguf-only";
  const backend = (deviceBackend ?? "").trim().toLowerCase();
  if (!(backend && budgetKnown)) return "unknown";
  if (GGUF_ONLY_BACKENDS.has(backend)) return "gguf-only";
  if (ACCELERATED_BACKENDS.has(backend)) return "accelerated";
  // An unrecognised backend is a new accelerator, not a CPU. Show what we show today.
  return "unknown";
}

/** The H3 group, whose two rows differ by roughly 10x in throughput. */
const H3_PIPELINE_ID = "minimaxai/minimax-h3";

/**
 * The curated artifacts a host without an accelerator is refused at load, by repo id.
 *
 * Keyed on the id, NOT on `format !== "gguf"`. Everything else in the catalogs loads here: the
 * diffusion pipelines resolve their device through the shared target and run on MPS (Z-Image is
 * tuned for MPS bfloat16 by name), video_capability() certifies Apple Silicon, and the STT rows
 * run through the whisper.cpp sidecar whatever format the catalog labels them. Only MiniMax-H3's
 * Modular Diffusers workflow is genuinely unplaceable: enable_auto_cpu_offload needs
 * mem_get_info, torch.mps has none, and video.py raises before the download.
 */
const UNPLACEABLE_WITHOUT_ACCELERATOR = new Set([H3_PIPELINE_ID]);

/** Whether a curated artifact is worth offering on this host.
 *
 * Only browse rows are filtered. A model already on disk keeps its row wherever it came from. */
export function curatedArtifactIsOfferable(
  repoId: string,
  host: HostClass,
): boolean {
  if (host !== "gguf-only") return true;
  return !UNPLACEABLE_WITHOUT_ACCELERATOR.has(repoId.trim().toLowerCase());
}

const H3_GGUF_ID = "unsloth/minimax-h3-gguf";

/**
 * The speed qualifier for a curated row, or null when it earns none.
 *
 * Deliberately keyed on the two H3 ids rather than on `format !== "gguf"`. Most non-GGUF rows are
 * plain bf16 or bnb-4bit, Auto precision resolves to int8 or bf16 on cards without fp8, and only
 * H3 has the measured gap this wording claims.
 */
export function h3PerfSuffix(repoId: string, host: HostClass): string | null {
  if (host !== "accelerated") return null;
  const id = repoId.trim().toLowerCase();
  if (id === H3_PIPELINE_ID) return "Fast FP8";
  if (id === H3_GGUF_ID) return "Slow";
  return null;
}
