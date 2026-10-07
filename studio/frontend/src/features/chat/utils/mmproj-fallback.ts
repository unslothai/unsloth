// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CpuFallbackReason, MmprojFallbackReason } from "../types/api";

/** Shared so both load paths describe a CPU fallback identically. */
export const CPU_FALLBACK_MESSAGE =
  "The auto-selected Vulkan backend crashed during startup, so GPU acceleration is disabled for this model session.";

const MMPROJ_FALLBACK_MESSAGES: Record<MmprojFallbackReason, string> = {
  // Three causes lead here (predicted no-fit, GPU alloc failure, crash), so name only the outcome.
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

/** Covers both fallbacks: a CPU-fallback replay can set both, and neither may be dropped. */
export function loadFallbackNotice(
  baseTitle: string,
  cpuFallbackReason: CpuFallbackReason | null | undefined,
  mmprojFallbackReason: MmprojFallbackReason | null | undefined,
  offloadNotice?: { titleSuffix: string; description: string } | null,
): { title: string; description: string | undefined; degraded: boolean } {
  const textOnly = isTextOnlyMmprojFallback(mmprojFallbackReason);
  const offload = cpuFallbackReason ? null : (offloadNotice ?? null);

  let suffix = "";
  if (cpuFallbackReason && textOnly) {
    suffix = " on CPU, without vision";
  } else if (cpuFallbackReason) {
    suffix = " on CPU";
  } else if (offload) {
    suffix = textOnly
      ? `${offload.titleSuffix}, without vision`
      : offload.titleSuffix;
  } else if (mmprojFallbackReason === "cpu_offload") {
    suffix = " with vision on CPU";
  } else if (textOnly) {
    suffix = " without vision";
  }

  const parts: string[] = [];
  if (cpuFallbackReason) {
    parts.push(CPU_FALLBACK_MESSAGE);
  }
  if (offload) {
    parts.push(offload.description);
  }
  if (mmprojFallbackReason) {
    parts.push(mmprojFallbackMessage(mmprojFallbackReason));
  }

  return {
    title: `${baseTitle}${suffix}`,
    description: parts.length > 0 ? parts.join(" ") : undefined,
    degraded: Boolean(cpuFallbackReason || offload || mmprojFallbackReason),
  };
}
