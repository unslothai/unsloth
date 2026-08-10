// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether a /api/health reply carries a measured hardware verdict or the backend's
// pre-detection default. Its own import-free module so it is testable: env.ts reaches
// import.meta.env through api-base.ts, which only vite can load.

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
