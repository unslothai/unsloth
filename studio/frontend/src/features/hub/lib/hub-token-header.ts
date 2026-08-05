// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Internal Unsloth API header; direct Hugging Face calls use Authorization.

export const HUB_HF_TOKEN_HEADER = "X-Unsloth-HF-Token";

// Set by the hf-proxy route on responses it produced, so a hub status passed
// through it is not confused with one from the Studio API itself. Lives here
// because both the hub and auth features read it.
export const HUB_PROXY_MARKER_HEADER = "X-Unsloth-HF-Proxy";

export function hubTokenHeader(
  token?: string | null,
): Record<string, string> {
  return token ? { [HUB_HF_TOKEN_HEADER]: token } : {};
}
