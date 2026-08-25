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
    // Auto, as an install with nothing picked runs: a server that answers with something else
    // must not be read as a choice the user made.
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

/** The install's setting. `refresh` skips the cache, for a caller that must not act on a
 * value another browser may have changed since this tab loaded it. An in-flight read is
 * still shared: two refreshing callers a moment apart want the same answer, not two GETs. */
export async function loadDownloadTransportSettings(
  opts: { refresh?: boolean } = {},
) {
  if (cachedTransport && !opts.refresh) {
    return cachedTransport;
  }
  inFlightTransport ??= fetchDownloadTransportSettings()
    .then(cacheTransport)
    .finally(() => {
      inFlightTransport = null;
    });
  return inFlightTransport;
}

// Writes run one at a time. Two quick selections used to race, and the earlier PUT landing
// last left the database on the mode the user did NOT pick while this browser showed the one
// they did. Chained rather than cancelled, so the last selection is also the last write.
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
  return cacheTransport(fromApi(await res.json()));
}

export function updateDownloadTransportSettings(
  mode: DownloadTransportMode,
): Promise<DownloadTransportSettings> {
  // The queue must survive a rejected write, or one failure strands every later selection.
  const next = writeQueue.catch(() => undefined).then(() => putDownloadTransport(mode));
  writeQueue = next.catch(() => undefined);
  return next;
}
