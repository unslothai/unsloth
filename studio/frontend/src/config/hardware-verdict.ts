// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The /api/health hardware verdict: whether a reply carries a measured one or the backend's
// pre-detection default, and what the shell derives from it. Its own import-free module so it is
// testable: env.ts reaches import.meta.env through api-base.ts, which only vite can load, and a
// derivation left in a component can only be checked by grepping its source.

export type HealthVerdict = {
  chat_only?: boolean;
  chat_only_reason?: string | null;
  /** What blocked that reason, when the backend knows something specific. */
  chat_only_detail?: string | null;
  hardware_detecting?: boolean;
  hardware_detection_deferred?: boolean;
};

export type ResolvedVerdict = {
  chatOnly: boolean;
  chatOnlyReason: string | null;
  chatOnlyDetail: string | null;
};

/** True while the backend is still measuring the host, so chat_only is its
 * pre-detection default rather than an answer. */
export function isProvisionalVerdict(data: HealthVerdict): boolean {
  return data.hardware_detecting === true;
}

/** True when the backend deferred detection rather than started it.
 * UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 stops health kicking detection at all, so nothing
 * settles until a hardware-dependent operation runs, and waiting would stall every load. */
export function isDetectionDeferred(data: HealthVerdict): boolean {
  return data.hardware_detection_deferred === true;
}


/** The Video row's tooltip, or undefined when the row stays navigable.
 *
 * Only the chat-only reasons that leave no video device at all. Apple Silicon runs video on Metal
 * with or without a healthy MLX stack, so "mlx_unavailable" is not one of them: those hosts reach
 * VideoPage, which reports the backend's video verdict.
 */
export function videoNavHint(
  chatOnlyMeasured: boolean,
  chatOnlyReason: string | null,
): string | undefined {
  if (!chatOnlyMeasured) return undefined;
  // Not "or a GPU": an Intel Mac's own dGPU is not one the video pipelines can use, so offering
  // that reads as a fix that would not work. Mirrors the backend's message for this host.
  if (chatOnlyReason === "intel_mac")
    return "Video generation requires Apple Silicon. This Intel Mac has no Metal device to run it.";
  // The GPU is there; PyTorch is what cannot reach it (a Windows update that resolved torch
  // from PyPI leaves a +cpu wheel beside two working cards). Offering "get a GPU" to that host
  // is the lie the reason exists to stop, and it is not the fix either.
  if (
    chatOnlyReason === "torch_cpu_build" ||
    chatOnlyReason === "torch_cuda_unavailable"
  )
    return "Video generation needs a working PyTorch GPU build. This machine's GPUs were detected but PyTorch cannot use them; repair the installation.";
  if (chatOnlyReason === "no_gpu") return "Video generation needs an NVIDIA or AMD GPU.";
  return undefined;
}

/** The verdict to store. A provisional reply keeps the previous values: storing
 * its chat_only tells a GPU host it has no GPU, and beforeLoad redirects on that. */
export function resolveVerdict(
  data: HealthVerdict,
  previous: ResolvedVerdict,
): ResolvedVerdict {
  if (isDetectionDeferred(data)) {
    // Nothing will settle, so keeping `previous` would leave the browser-platform default
    // (chatOnly false off macOS) in place all session and offer Train on a CPU-only host.
    // Take the backend's conservative chat_only, keeping any reason already explained.
    return {
      chatOnly: data.chat_only ?? true,
      chatOnlyReason: data.chat_only_reason ?? previous.chatOnlyReason,
      // The detail belongs to the reason, so it travels with it: keeping a stale one
      // beside a new reason would name a blocker for a verdict it never explained.
      chatOnlyDetail:
        data.chat_only_reason === undefined
          ? previous.chatOnlyDetail
          : (data.chat_only_detail ?? null),
    };
  }
  if (isProvisionalVerdict(data)) return previous;
  return {
    chatOnly: data.chat_only ?? false,
    chatOnlyReason: data.chat_only_reason ?? null,
    chatOnlyDetail: data.chat_only_detail ?? null,
  };
}
