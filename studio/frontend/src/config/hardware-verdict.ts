// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether a /api/health reply carries a measured hardware verdict or the
// backend's pre-detection default. Its own module, no imports, so it is testable:
// env.ts reaches import.meta.env through api-base.ts, which only vite can load.

export type HealthVerdict = {
  chat_only?: boolean;
  chat_only_reason?: string | null;
  hardware_detecting?: boolean;
  hardware_detection_deferred?: boolean;
};

export type ResolvedVerdict = {
  chatOnly: boolean;
  chatOnlyReason: string | null;
};

/** True while the backend is still measuring the host, so chat_only is its
 * pre-detection default rather than an answer. */
export function isProvisionalVerdict(data: HealthVerdict): boolean {
  return data.hardware_detecting === true;
}

/** True when the backend has deferred detection rather than started it.

 * UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 stops health kicking detection at all, so
 * nothing settles until a hardware-dependent operation runs. Waiting for a
 * measurement that is not coming would stall every load, including /login, for
 * the whole retry window.
 */
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
    // Nothing will settle, so keeping `previous` would leave the browser-platform
    // default (chatOnly false off macOS) in place for the session and offer Train on
    // a CPU-only host. Take the backend's conservative pre-detection chat_only
    // instead, and keep any reason already being explained.
    return {
      chatOnly: data.chat_only ?? true,
      chatOnlyReason: data.chat_only_reason ?? previous.chatOnlyReason,
    };
  }
  if (isProvisionalVerdict(data)) return previous;
  return {
    chatOnly: data.chat_only ?? false,
    chatOnlyReason: data.chat_only_reason ?? null,
  };
}
