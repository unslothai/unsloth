// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so it is testable: env.ts reaches import.meta.env, which only vite can load.

export type HealthVerdict = {
  chat_only?: boolean;
  chat_only_reason?: string | null;
  chat_only_detail?: string | null;
  /** The repair a closed AMD device node needs, when it fully explains the verdict. */
  chat_only_hint?: string | null;
  hardware_detecting?: boolean;
  hardware_detection_deferred?: boolean;
};

export type ResolvedVerdict = {
  chatOnly: boolean;
  chatOnlyReason: string | null;
  chatOnlyDetail: string | null;
  chatOnlyHint: string | null;
};

export function isProvisionalVerdict(data: HealthVerdict): boolean {
  return data.hardware_detecting === true;
}

/** Under UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 nothing settles, so waiting would stall every load. */
export function isDetectionDeferred(data: HealthVerdict): boolean {
  return data.hardware_detection_deferred === true;
}


/** Only reasons that leave no video device; Apple Silicon runs video on Metal regardless of MLX. */
export function videoNavHint(
  chatOnlyMeasured: boolean,
  chatOnlyReason: string | null,
  // Developer mode's CPU video opt-in (/api/health video_cpu_opt_in). Waives only the plain
  // no-GPU host; VideoPage still answers from the backend's own video verdict.
  videoCpuOptIn = false,
): string | undefined {
  if (!chatOnlyMeasured) return undefined;
  if (chatOnlyReason === "no_gpu" && videoCpuOptIn) return undefined;
  // Not "or a GPU": an Intel Mac's own dGPU is not one the video pipelines can use, so offering
  // that reads as a fix that would not work. Mirrors the backend's message for this host.
  if (chatOnlyReason === "intel_mac")
    return "Video generation requires Apple Silicon. This Intel Mac has no Metal device to run it.";
  if (chatOnlyReason === "rosetta")
    return "Unsloth is running under Rosetta (x86_64 Python). Re-run the installer from a native arm64 terminal to enable video generation.";
  // The GPU is there; PyTorch is what cannot reach it (a Windows update that resolved torch
  // from PyPI leaves a +cpu wheel beside two working cards), so "get a GPU" is a lie.
  if (
    chatOnlyReason === "torch_cpu_build" ||
    chatOnlyReason === "torch_cuda_unavailable"
  )
    return "Video generation needs a working PyTorch GPU build. This machine's GPUs were detected but PyTorch cannot use them; repair the installation.";
  if (chatOnlyReason === "no_gpu") return "Video generation needs an NVIDIA or AMD GPU.";
  return undefined;
}

/** A provisional reply keeps the previous values: its chat_only would tell a GPU host it has none. */
export function resolveVerdict(
  data: HealthVerdict,
  previous: ResolvedVerdict,
): ResolvedVerdict {
  if (isDetectionDeferred(data)) {
    // Nothing will settle, so take the backend's conservative chat_only instead of the browser guess.
    return {
      chatOnly: data.chat_only ?? true,
      chatOnlyReason: data.chat_only_reason ?? previous.chatOnlyReason,
      // The detail belongs to its reason; never keep a stale one beside a new reason.
      chatOnlyDetail:
        data.chat_only_reason === undefined
          ? previous.chatOnlyDetail
          : (data.chat_only_detail ?? null),
      chatOnlyHint:
        data.chat_only_reason === undefined
          ? previous.chatOnlyHint
          : (data.chat_only_hint ?? null),
    };
  }
  if (isProvisionalVerdict(data)) return previous;
  return {
    chatOnly: data.chat_only ?? false,
    chatOnlyReason: data.chat_only_reason ?? null,
    chatOnlyDetail: data.chat_only_detail ?? null,
    chatOnlyHint: data.chat_only_hint ?? null,
  };
}
