// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CpuFallbackReason, MmprojFallbackReason } from "../types/api";

/**
 * Said when the whole model, not just the projector, ended up on the CPU.
 *
 * Lived as a duplicated string literal in chat-adapter.ts and
 * use-chat-model-runtime.ts. It is here so the two load paths cannot describe the
 * same condition differently, which is what let them drift apart below.
 */
export const CPU_FALLBACK_MESSAGE =
  "The auto-selected Vulkan backend crashed during startup, so GPU acceleration is disabled for this model session.";

const MMPROJ_FALLBACK_MESSAGES: Record<MmprojFallbackReason, string> = {
  // Three routes reach this one placement: the fit estimate predicting the
  // projector will not fit in VRAM, a GPU allocation failure at startup, and a
  // bare signal crash with no non-projector diagnostic. So the message names the
  // outcome and no cause. "Could not start on the GPU" is untrue of the predicted
  // route, which never attempts a GPU load, and "does not fit in VRAM" is untrue
  // of the crash routes -- and that one is worse than vague, because it sends
  // someone whose GPU runtime is broken off to cut context and offload layers.
  cpu_offload:
    "Unsloth is running the vision projector in system memory rather than on the GPU. Image input remains available, but image processing may be slower.",
  projector_incompatible:
    "The vision projector is incompatible with the installed llama.cpp build, so Unsloth reloaded this model in text-only mode. Update Unsloth, then reload the model to restore image input.",
  projector_startup_failure:
    "The vision projector could not start on the GPU or CPU, so Unsloth reloaded this model in text-only mode. Free memory or check the GPU logs, then reload the model to restore image input.",
};

export function isTextOnlyMmprojFallback(
  reason: MmprojFallbackReason | null | undefined,
): boolean {
  return (
    reason === "projector_incompatible" ||
    reason === "projector_startup_failure"
  );
}

export function mmprojFallbackMessage(reason: MmprojFallbackReason): string {
  return MMPROJ_FALLBACK_MESSAGES[reason];
}

export function mmprojLoadNotice(
  modelName: string,
  reason: MmprojFallbackReason,
): { title: string; description: string } {
  return {
    title:
      reason === "cpu_offload"
        ? `${modelName} loaded with vision on CPU`
        : `${modelName} loaded without vision`,
    description: mmprojFallbackMessage(reason),
  };
}

/**
 * The one description of how a load was degraded, covering BOTH fallbacks.
 *
 * The two load paths each wrote this inline as
 * `mmproj ? mmprojMessage : cpu ? cpuMessage : undefined`, which silently drops the
 * CPU message whenever both are set. That combination is not hypothetical: on a
 * CPU-fallback replay `llama_cpp.py` preserves `_cpu_fallback_reason` (it only clears
 * it when `not _replaying_cpu_fallback`) and clears `_mmproj_fallback_reason` so the
 * projector can fail again in that same launch. A low-VRAM machine whose Vulkan
 * backend crashed is exactly where the projector then falls back too -- the case the
 * feature was written for.
 *
 * The user was told "loaded without vision" and never told the model was on the CPU,
 * so a silently unaccelerated session read as a deliberate, explained one.
 *
 * `baseTitle` is suffixed rather than replaced because the two callers name models
 * differently: the auto-load passes a label that already reads "Loaded X (Q4_K_M)",
 * the explicit load passes "X loaded". Both take the same suffixes, which is why one
 * helper can serve them.
 */
export function loadFallbackNotice(
  baseTitle: string,
  cpuFallbackReason: CpuFallbackReason | null | undefined,
  mmprojFallbackReason: MmprojFallbackReason | null | undefined,
): { title: string; description: string | undefined; degraded: boolean } {
  const textOnly = isTextOnlyMmprojFallback(mmprojFallbackReason);

  // When the whole model is on the CPU, saying the projector is too adds nothing --
  // "on CPU" already covers it. Losing vision entirely is a different fact and is
  // always said.
  let suffix = "";
  if (cpuFallbackReason && textOnly) {
    suffix = " on CPU, without vision";
  } else if (cpuFallbackReason) {
    suffix = " on CPU";
  } else if (mmprojFallbackReason === "cpu_offload") {
    suffix = " with vision on CPU";
  } else if (textOnly) {
    suffix = " without vision";
  }

  // Both sentences, CPU first: it is the broader condition, and the projector
  // sentence reads as a detail under it.
  const parts: string[] = [];
  if (cpuFallbackReason) {
    parts.push(CPU_FALLBACK_MESSAGE);
  }
  if (mmprojFallbackReason) {
    parts.push(mmprojFallbackMessage(mmprojFallbackReason));
  }

  return {
    title: `${baseTitle}${suffix}`,
    description: parts.length > 0 ? parts.join(" ") : undefined,
    degraded: Boolean(cpuFallbackReason || mmprojFallbackReason),
  };
}
