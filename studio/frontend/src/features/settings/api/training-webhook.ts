// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type TrainingWebhookSettings = {
  enabled: boolean;
  url: string;
};

function fromApi(settings: TrainingWebhookSettings): TrainingWebhookSettings {
  return {
    enabled: settings.enabled,
    url: settings.url,
  };
}

export async function loadTrainingWebhookSettings(): Promise<TrainingWebhookSettings> {
  const res = await authFetch("/api/settings/notifications");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load notification settings"),
    );
  }
  return fromApi(await res.json());
}

export async function updateTrainingWebhookSettings(input: {
  enabled: boolean;
  url: string;
}): Promise<TrainingWebhookSettings> {
  const res = await authFetch("/api/settings/notifications", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update notification settings"),
    );
  }
  return fromApi(await res.json());
}

export async function sendTrainingWebhookTest(): Promise<void> {
  const res = await authFetch("/api/settings/notifications/test", {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Webhook test failed"));
  }
}
