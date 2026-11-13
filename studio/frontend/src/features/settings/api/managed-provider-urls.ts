// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

const ROUTE = "/api/settings/managed-provider-urls";

export type ManagedProviderUrlSettings = {
  allowed: boolean;
  defaultAllowed: boolean;
  // UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 overrides the switch for every account.
  lockedByEnvironment: boolean;
};

type ApiManagedProviderUrlSettings = {
  allowed: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_allowed?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  locked_by_environment?: boolean;
};

function fromApi(
  settings: ApiManagedProviderUrlSettings,
): ManagedProviderUrlSettings {
  return {
    allowed: settings.allowed,
    defaultAllowed: settings.default_allowed ?? false,
    lockedByEnvironment: settings.locked_by_environment ?? false,
  };
}

export async function loadManagedProviderUrls(): Promise<ManagedProviderUrlSettings> {
  const res = await authFetch(ROUTE);
  // 404 means an older backend: hide the row instead of showing an unactionable error.
  if (res.status === 404) {
    throw new SettingsRouteAbsentError(ROUTE);
  }
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Failed to load managed account connection settings",
      ),
    );
  }
  return fromApi(await res.json());
}

export async function updateManagedProviderUrls(
  allowed: boolean,
): Promise<ManagedProviderUrlSettings> {
  const res = await authFetch(ROUTE, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ allowed }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Failed to save managed account connection settings",
      ),
    );
  }
  return fromApi(await res.json());
}
