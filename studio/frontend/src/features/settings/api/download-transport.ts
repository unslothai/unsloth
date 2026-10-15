// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const DOWNLOAD_TRANSPORT_EVENT = "unsloth-download-transport-change";

export type DownloadTransportMode = "auto" | "xet" | "http";

export type DownloadTransportSettings = {
  mode: DownloadTransportMode;
  xetAvailable: boolean;
  xetUnavailableReason: string | null;
  autoResolvesTo: "xet" | "http";
  autoReason: string | null;
};

type ApiDownloadTransportSettings = {
  mode: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  xet_available: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  xet_unavailable_reason: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_resolves_to: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_reason: string | null;
};

let cachedTransport: DownloadTransportSettings | null = null;
let inFlightTransport: Promise<DownloadTransportSettings> | null = null;
// A refresh must not reuse a GET answered before it asked.
let inFlightIsRefresh = false;
// Newest request wins so an older overlapping GET cannot re-cache a replaced value.
let latestRequest = 0;

export function subscribeDownloadTransportSettings(
  listener: (settings: DownloadTransportSettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<DownloadTransportSettings>).detail);
  };
  window.addEventListener(DOWNLOAD_TRANSPORT_EVENT, handleChange);
  return () =>
    window.removeEventListener(DOWNLOAD_TRANSPORT_EVENT, handleChange);
}

function asMode(value: string, fallback: DownloadTransportMode) {
  return value === "auto" || value === "xet" || value === "http"
    ? value
    : fallback;
}

function fromApi(
  settings: ApiDownloadTransportSettings,
): DownloadTransportSettings {
  return {
    // Unknown values read as auto, never as a user choice.
    mode: asMode(settings.mode, "auto"),
    xetAvailable: settings.xet_available,
    xetUnavailableReason: settings.xet_unavailable_reason,
    autoResolvesTo: settings.auto_resolves_to === "xet" ? "xet" : "http",
    autoReason: settings.auto_reason,
  };
}

function cacheTransport(settings: DownloadTransportSettings) {
  cachedTransport = settings;
  window.dispatchEvent(
    new CustomEvent(DOWNLOAD_TRANSPORT_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchDownloadTransportSettings(): Promise<DownloadTransportSettings> {
  const res = await authFetch("/api/settings/download-transport");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load download transport settings"),
    );
  }
  return fromApi(await res.json());
}

/** `refresh` skips the cache; refreshing callers never share a plain hydration GET. */
export async function loadDownloadTransportSettings(
  opts: { refresh?: boolean } = {},
) {
  if (cachedTransport && !opts.refresh) {
    return cachedTransport;
  }
  if (inFlightTransport && (!opts.refresh || inFlightIsRefresh)) {
    return inFlightTransport;
  }
  const request = ++latestRequest;
  inFlightIsRefresh = Boolean(opts.refresh);
  const pending = fetchDownloadTransportSettings()
    .then((settings) => {
      if (request === latestRequest) {
        return cacheTransport(settings);
      }
      // Superseded: returning the stale payload would revert the newer write.
      return cachedTransport ?? settings;
    })
    .finally(() => {
      if (inFlightTransport === pending) {
        inFlightTransport = null;
        inFlightIsRefresh = false;
      }
    });
  inFlightTransport = pending;
  return pending;
}

// Serialize writes so the last selection is also the last PUT to land.
let writeQueue: Promise<unknown> = Promise.resolve();

async function putDownloadTransport(
  mode: DownloadTransportMode,
): Promise<DownloadTransportSettings> {
  const res = await authFetch("/api/settings/download-transport", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update download transport"),
    );
  }
  // Any GET issued before this write is now stale.
  latestRequest += 1;
  return cacheTransport(fromApi(await res.json()));
}

export function updateDownloadTransportSettings(
  mode: DownloadTransportMode,
): Promise<DownloadTransportSettings> {
  // Survive a rejected write, or one failure strands every later selection.
  const next = writeQueue.catch(() => undefined).then(() => putDownloadTransport(mode));
  writeQueue = next.catch(() => undefined);
  return next;
}
