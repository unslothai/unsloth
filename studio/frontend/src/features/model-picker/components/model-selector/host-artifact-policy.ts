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

/** Auto selects FP8 or INT8 according to backend capability. */
export const DENSE_QUANT_PRECISION_CHIP = "FP8 / INT8";

/** The transformer precision the page will request for the next load. `undefined` means the caller
 *  has no control to report, in which case the row describes the default (auto) request. */
export type RequestedPrecision = string | null | undefined;

/** The explicit schemes the backend says this host can run, or undefined when it did not say
 *  (an older backend). Undefined defers to the host class alone. */
export type DenseQuantSchemes = readonly string[] | undefined;

/** The runtime-precision chip for a dense-quant row under `precision`, or null when the load will
 *  run the checkpoint as-is or would be refused.
 *
 *  Auto never refuses: it walks down to whatever the card has, or to BF16, so it keeps the pair.
 *  An explicit scheme fails closed, so a card that cannot run it (fp8 on Ampere) must not be
 *  labelled fast at all -- the click would end in a precision refusal rather than a load. */
export function denseQuantPrecisionChip(
  precision: RequestedPrecision,
  schemes?: DenseQuantSchemes,
): string | null {
  const value = (precision ?? "auto").trim().toLowerCase();
  if (value === "" || value === "auto") return DENSE_QUANT_PRECISION_CHIP;
  if (value === "none" || value === "off") return null;
  if (schemes && !schemes.includes(value)) return null;
  return value.toUpperCase();
}

/** Load controls that keep the backend on bf16 whatever Precision asks for. Each is decidable from
 *  the request alone, and the loader refuses an EXPLICIT scheme under every one of them, so a row
 *  under any of these must read exactly as it does with Precision=Off.
 *
 *  Speed=Off is the one that depends on the precision: it rewrites an AUTO quant to off, but an
 *  explicit scheme still runs, so the caller passes `precision` and it is judged here rather than
 *  lumped in with the rest. */
export function loadControlsBlockDenseQuant({
  precision,
  speedMode,
  memoryMode,
  cpuOffload,
}: {
  precision: RequestedPrecision;
  speedMode?: string | null;
  memoryMode?: string | null;
  cpuOffload?: boolean;
}): boolean {
  const speed = (speedMode ?? "").trim().toLowerCase();
  const memory = (memoryMode ?? "").trim().toLowerCase();
  // Eager never compiles, and an uncompiled torchao transformer loses to the bf16 it replaces.
  if (speed === "eager") return true;
  // balanced / low_vram name their offload policy outright, and offload hooks move modules with
  // Module.to(), which torchao tensors do not survive. The bare flag forces it when no mode is set.
  if (memory === "balanced" || memory === "low_vram") return true;
  if (!memory && cpuOffload) return true;
  // Bit-exact output is incompatible with an automatic quant, so the backend rewrites auto to off.
  return speed === "off" && (precision ?? "auto").trim().toLowerCase() === "auto";
}

/** The precision a row should describe, given what this host can actually run. An explicit scheme
 *  the card lacks is refused at load, so the row has to read exactly as it does with Precision=Off
 *  rather than advertise a click that ends in the refusal. */
export function effectiveRowPrecision(
  precision: RequestedPrecision,
  schemes?: DenseQuantSchemes,
): RequestedPrecision {
  return denseQuantPrecisionChip(precision, schemes) === null ? "none" : precision;
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
