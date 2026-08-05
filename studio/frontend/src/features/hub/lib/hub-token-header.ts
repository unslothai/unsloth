// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Internal Unsloth API header; direct Hugging Face calls use Authorization.

export const HUB_HF_TOKEN_HEADER = "X-Unsloth-HF-Token";

// Stamped by the backend's hub proxy routes. Its absence on a 404 means the
// backend predates them and the SPA catch-all answered, which is not a Hub
// failure and must not be reported as one. Idea from #7893.
export const HUB_PROXY_MARKER_HEADER = "X-Unsloth-HF-Proxy";

export function hubTokenHeader(
  token?: string | null,
): Record<string, string> {
  return token ? { [HUB_HF_TOKEN_HEADER]: token } : {};
}
