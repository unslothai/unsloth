


// Kept OUT of api.ts on purpose: api.ts imports the auth barrel, which the test runner cannot
// load, so anything defined there is untestable. That is how a normalizer silently dropping the
// backend's Auto verdict shipped green.

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
}

export const DOWNLOAD_TRANSPORT_CAPABILITIES_FALLBACK: DownloadTransportCapabilities = {
  http: { available: true, reason: null },
  xet: {
    available: null,
    reason: "Couldn't verify Xet support with the Unsloth backend.",
  },
  // Unknown backend state: stay on Xet, the download-time ladder still falls back to HTTP.
  auto_resolves_to: "xet",
  auto_reason: null,
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
  };
}
