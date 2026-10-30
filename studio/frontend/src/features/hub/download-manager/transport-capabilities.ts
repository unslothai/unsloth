// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of api.ts, which imports the auth barrel the test runner cannot load.

export interface DownloadTransportCapability {
  available: boolean | null;
  reason: string | null;
}

export interface DownloadTransportCapabilities {
  http: DownloadTransportCapability;
  xet: DownloadTransportCapability;
  // Server-side: only the backend sees RAM, the hf_xet build and recent Xet failures.
  auto_resolves_to?: "xet" | "http";
  auto_reason?: string | null;
  // Server-side: only the backend knows which huggingface_hub writer is installed.
  partials_resumable?: boolean;
}

export const DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK: DownloadTransportCapabilities = {
  http: { available: true, reason: null },
  xet: {
    available: null,
    reason: "Couldn't verify Xet support with the Unsloth backend.",
  },
  // Unknown backend state: stay on Xet; the download-time ladder still falls back to HTTP.
  auto_resolves_to: "xet",
  auto_reason: null,
  // Unverified means do not promise a byte resume.
  partials_resumable: false,
};
export function normalizeDownloadTransportCapability(
  value: unknown,
  fallback: DownloadTransportCapability,
): DownloadTransportCapability {
  if (!value || typeof value !== "object") {
    return fallback;
  }
  const candidate = value as { available?: unknown; reason?: unknown };
  return {
    available:
      typeof candidate.available === "boolean"
        ? candidate.available
        : fallback.available,
    reason:
      typeof candidate.reason === "string"
        ? candidate.reason
        : candidate.reason === null
          ? null
          : fallback.reason,
  };
}

export function normalizeDownloadTransportCapabilities(
  value: unknown,
): DownloadTransportCapabilities {
  if (!value || typeof value !== "object") {
    return DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK;
  }
  const candidate = value as {
    http?: unknown;
    xet?: unknown;
    auto_resolves_to?: unknown;
    auto_reason?: unknown;
    partials_resumable?: unknown;
  };
  return {
    http: normalizeDownloadTransportCapability(candidate.http, {
      available: true,
      reason: null,
    }),
    xet: normalizeDownloadTransportCapability(
      candidate.xet,
      DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK.xet,
    ),
    // Carry the verdict through, or Auto always resolves to Xet.
    auto_resolves_to:
      candidate.auto_resolves_to === "http" || candidate.auto_resolves_to === "xet"
        ? candidate.auto_resolves_to
        : DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK.auto_resolves_to,
    auto_reason:
      typeof candidate.auto_reason === "string" ? candidate.auto_reason : null,
    partials_resumable: candidate.partials_resumable === true,
  };
}
