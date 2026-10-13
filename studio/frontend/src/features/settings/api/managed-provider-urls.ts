// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

const ROUTE = "/api/settings/managed-provider-urls";

export type ManagedProviderUrlSettings = {
  allowed: boolean;
  defaultAllowed: boolean;
  // UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 refuses private addresses for
  // every account, so the switch cannot take effect while it is set.
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
  // A backend older than this bundle does not serve the route. Told apart from a
  // failed read so the row can be hidden rather than shown with a red error the
  // owner cannot act on.
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
