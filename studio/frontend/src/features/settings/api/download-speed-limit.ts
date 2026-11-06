// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { clearDownloadTransportCapabilitiesCache } from "@/features/hub";
import { readFastApiError } from "@/lib/format-fastapi-error";

const DOWNLOAD_SPEED_LIMIT_EVENT = "unsloth-download-speed-limit-change";

/** Mbit/s; null means no limit. */
export type DownloadSpeedLimitSettings = {
  mbps: number | null;
};

export function subscribeDownloadSpeedLimitSettings(
  listener: (settings: DownloadSpeedLimitSettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<DownloadSpeedLimitSettings>).detail);
  };
  window.addEventListener(DOWNLOAD_SPEED_LIMIT_EVENT, handleChange);
  return () =>
    window.removeEventListener(DOWNLOAD_SPEED_LIMIT_EVENT, handleChange);
}

export async function loadDownloadSpeedLimitSettings(): Promise<DownloadSpeedLimitSettings> {
  const res = await authFetch("/api/settings/download-speed-limit");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load download speed limit"),
    );
  }
  return (await res.json()) as DownloadSpeedLimitSettings;
}

export async function updateDownloadSpeedLimitSettings(
  mbps: number | null,
): Promise<DownloadSpeedLimitSettings> {
  const res = await authFetch("/api/settings/download-speed-limit", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mbps }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update download speed limit"),
    );
  }
  const settings = (await res.json()) as DownloadSpeedLimitSettings;
  clearDownloadTransportCapabilitiesCache();
  window.dispatchEvent(
    new CustomEvent(DOWNLOAD_SPEED_LIMIT_EVENT, { detail: settings }),
  );
  return settings;
}
