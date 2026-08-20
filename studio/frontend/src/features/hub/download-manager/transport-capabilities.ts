


// Kept OUT of api.ts on purpose: api.ts imports the auth barrel, which the test runner cannot
// load, so anything defined there is untestable. That is how a normalizer silently dropping the
// backend's Auto verdict shipped green.

import { PRODUCT_NAME } from "@/config/branding";

export interface DownloadTransportCapability {
  available: boolean | null;
  reason: string | null;
}

export interface DownloadTransportCapabilities {
  http: DownloadTransportCapability;
  xet: DownloadTransportCapability;
  // What "auto" resolves to right now, and why. Server-side because only the backend can see this
  // machine's RAM, hf_xet build, and recent Xet failures.
  auto_resolves_to?: "xet" | "http";
  auto_reason?: string | null;
  // Whether an interrupted HTTP transfer leaves bytes the next attempt can append to. Server-side
  // because only the backend knows which huggingface_hub writer is installed.
  partials_resumable?: boolean;
}

export const DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK: DownloadTransportCapabilities = {
  http: { available: true, reason: null },
  xet: {
    available: null,
    reason: `Couldn't verify Xet support with the ${PRODUCT_NAME} backend.`,
  },
  // Unknown backend state: stay on Xet, the download-time ladder still falls back to HTTP.
  auto_resolves_to: "xet",
  auto_reason: null,
  // Unverified means "do not promise a resume". Continuing a partial is honest either way; only
  // the byte-resume wording would be a lie.
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
    // Carry the backend's verdict through. Dropping these left effectiveTransportMode("auto")
    // reading undefined, so Auto always resolved to Xet whatever the machine's health was.
    auto_resolves_to:
      candidate.auto_resolves_to === "http" || candidate.auto_resolves_to === "xet"
        ? candidate.auto_resolves_to
        : DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK.auto_resolves_to,
    auto_reason:
      typeof candidate.auto_reason === "string" ? candidate.auto_reason : null,
    // Anything but an explicit true (older backend, junk value) stays false.
    partials_resumable: candidate.partials_resumable === true,
  };
}
