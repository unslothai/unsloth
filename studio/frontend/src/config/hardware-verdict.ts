// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether a /api/health reply carries a measured hardware verdict or the
// backend's pre-detection default. Its own module, no imports, so it is testable:
// env.ts reaches import.meta.env through api-base.ts, which only vite can load.

export type HealthVerdict = {
  chat_only?: boolean;
  chat_only_reason?: string | null;
  hardware_detecting?: boolean;
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

/** The verdict to store. A provisional reply keeps the previous values: storing
 * its chat_only tells a GPU host it has no GPU, and beforeLoad redirects on that. */
export function resolveVerdict(
  data: HealthVerdict,
  previous: ResolvedVerdict,
): ResolvedVerdict {
  if (isProvisionalVerdict(data)) return previous;
  return {
    chatOnly: data.chat_only ?? false,
    chatOnlyReason: data.chat_only_reason ?? null,
  };
}
