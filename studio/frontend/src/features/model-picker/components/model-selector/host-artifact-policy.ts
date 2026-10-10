// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { normalizeDenseQuantSchemes } from "../../../../lib/dense-quant-schemes.ts";

/** Media-picker runtime capability. The backend reports dense quant, since its name alone cannot
 *  distinguish unsupported accelerators. */
export type HostClass = "unknown" | "gguf-only" | "accelerated" | "dense-quant";

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
  if (GGUF_ONLY_BACKENDS.has(backend)) return "gguf-only";
  // An unrecognised backend is a new accelerator, not a CPU.
  return "unknown";
}

export function hostIsAccelerated(host: HostClass): boolean {
  return host === "accelerated" || host === "dense-quant";
}

export function hostRunsDenseQuant(host: HostClass): boolean {
  return host === "dense-quant";
}

/** The two H3 rows differ by roughly 10x in throughput. */
const H3_PIPELINE_ID = "minimaxai/minimax-h3";

/** Refused without an accelerator, keyed by id: only MiniMax-H3's Modular workflow is
  *  unplaceable (enable_auto_cpu_offload needs mem_get_info). */
const UNPLACEABLE_WITHOUT_ACCELERATOR = new Set([H3_PIPELINE_ID]);

/** Only browse rows are filtered; a model already on disk keeps its row. */
export function curatedArtifactIsOfferable(
  repoId: string,
  host: HostClass,
): boolean {
  if (host !== "gguf-only") return true;
  return !UNPLACEABLE_WITHOUT_ACCELERATOR.has(repoId.trim().toLowerCase());
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
  return host !== "gguf-only";
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
