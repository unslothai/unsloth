// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type KeylessApiAccessScope = "off" | "inference" | "full";
export type KeylessApiAccessExposure =
  | "colab"
  | "public_url"
  | "private_lan"
  | "network";

export type KeylessApiAccessSettings = {
  scope: KeylessApiAccessScope;
  tools: boolean;
  exposure: KeylessApiAccessExposure | null;
};

type ApiKeylessApiAccessSettings = {
  scope: KeylessApiAccessScope;
  tools: boolean;
  exposure?: KeylessApiAccessExposure | null;
};

function fromApi(
  settings: ApiKeylessApiAccessSettings,
): KeylessApiAccessSettings {
  return {
    scope: settings.scope,
    tools: settings.tools,
    exposure: settings.exposure ?? null,
  };
}

export async function loadKeylessApiAccess(): Promise<KeylessApiAccessSettings> {
  const res = await authFetch("/api/settings/keyless-api-access");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load keyless API access settings"),
    );
  }
  return fromApi(await res.json());
}

export async function updateKeylessApiAccess(
  scope: KeylessApiAccessScope,
  tools?: boolean,
): Promise<KeylessApiAccessSettings> {
  const res = await authFetch("/api/settings/keyless-api-access", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scope, tools: tools ?? null }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Failed to update keyless API access settings",
      ),
    );
  }
  return fromApi(await res.json());
}

/** who can reach this server once a scope is on */
export function keylessAudience(
  exposure: KeylessApiAccessExposure | null,
): string {
  switch (exposure) {
    case "public_url":
      return "Anyone with your public URL";
    case "network":
    case "private_lan":
      return "Anyone on your network";
    case "colab":
      return "Anyone who can reach this Colab runtime";
    default:
      return "Anything running on this computer";
  }
}
