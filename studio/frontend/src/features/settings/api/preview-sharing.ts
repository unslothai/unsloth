// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type PreviewSharingSettings = {
  enabled: boolean;
  defaultEnabled: boolean;
};

type ApiPreviewSharingSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
};

function fromApi(settings: ApiPreviewSharingSettings): PreviewSharingSettings {
  return {
    enabled: settings.enabled,
    defaultEnabled: settings.default_enabled,
  };
}

export async function loadPreviewSharing(): Promise<PreviewSharingSettings> {
  const res = await authFetch("/api/settings/preview-sharing");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load preview sharing settings"),
    );
  }
  return fromApi(await res.json());
}

export async function updatePreviewSharing(
  enabled: boolean,
): Promise<PreviewSharingSettings> {
  const res = await authFetch("/api/settings/preview-sharing", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update preview sharing settings"),
    );
  }
  return fromApi(await res.json());
}

// Rotate the server-side signing secret, invalidating every previously shared
// /p preview link in one step. Newly copied links keep working.
export async function rotatePreviewLinks(): Promise<void> {
  const res = await authFetch("/api/settings/preview-links/rotate", {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to revoke preview links"),
    );
  }
}
