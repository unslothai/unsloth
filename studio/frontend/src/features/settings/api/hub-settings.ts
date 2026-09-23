// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchDeviceType } from "@/config/env";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { HubSource } from "@/lib/hf-endpoint";

export type HubSettings = {
  /** Empty means the official Hugging Face Hub. */
  hfEndpoint: string;
  datasetsServerFollowsEndpoint: boolean;
  source: HubSource;
  /** Differs from `source` only when the ModelScope adapter could not start. */
  activeSource: HubSource;
};

type ApiHubSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_endpoint: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  datasets_server_follows_endpoint: boolean;
  source: HubSource;
  // biome-ignore lint/style/useNamingConvention: API schema
  active_source: HubSource;
};

function fromApi(settings: ApiHubSettings): HubSettings {
  return {
    hfEndpoint: settings.hf_endpoint,
    datasetsServerFollowsEndpoint: settings.datasets_server_follows_endpoint,
    source: settings.source,
    activeSource: settings.active_source,
  };
}

export async function loadHubSettings(): Promise<HubSettings> {
  const res = await authFetch("/api/settings/hub");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load hub settings"));
  }
  return fromApi(await res.json());
}

export class InvalidHubEndpointError extends Error {}

/** Owner only. The catalog follows as soon as this resolves. */
export async function updateHubSource(source: HubSource): Promise<HubSettings> {
  const res = await authFetch("/api/settings/hub/source", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to switch the model source"));
  }
  const saved = fromApi(await res.json());
  await fetchDeviceType({ force: true }).catch(() => undefined);
  return saved;
}

export type HubEndpointSettings = Pick<
  HubSettings,
  "hfEndpoint" | "datasetsServerFollowsEndpoint"
>;

export async function updateHubSettings(
  settings: HubEndpointSettings,
): Promise<HubSettings> {
  const res = await authFetch("/api/settings/hub", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      hf_endpoint: settings.hfEndpoint,
      // biome-ignore lint/style/useNamingConvention: API schema
      datasets_server_follows_endpoint: settings.datasetsServerFollowsEndpoint,
    }),
  });
  if (!res.ok) {
    const message = await readFastApiError(res, "Failed to save hub settings");
    throw res.status === 400 || res.status === 422
      ? new InvalidHubEndpointError(message)
      : new Error(message);
  }
  const saved = fromApi(await res.json());
  // The browser's Hub calls follow the endpoints /api/health reports.
  await fetchDeviceType({ force: true }).catch(() => undefined);
  return saved;
}
