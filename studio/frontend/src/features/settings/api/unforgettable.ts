// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { UnforgettableEpisodeExtras } from "@/features/unforgettable/lib/merge-extras";

const SETTINGS_EVENT = "unsloth-unforgettable-settings-change";

export type UnforgettableSettings = UnforgettableEpisodeExtras & {
  voter?: string | null;
  supervisor_url?: string | null;
  supervisor_timeout?: number;
  db_path?: string;
  namespace?: string;
};

let cached: UnforgettableSettings | null = null;
let inFlight: Promise<UnforgettableSettings> | null = null;

function publish(settings: UnforgettableSettings) {
  cached = settings;
  window.dispatchEvent(
    new CustomEvent(SETTINGS_EVENT, { detail: settings }),
  );
  return settings;
}

export function subscribeUnforgettableSettings(
  listener: (settings: UnforgettableSettings) => void,
) {
  const handle = (event: Event) => {
    listener((event as CustomEvent<UnforgettableSettings>).detail);
  };
  window.addEventListener(SETTINGS_EVENT, handle);
  return () => window.removeEventListener(SETTINGS_EVENT, handle);
}

async function fetchSettings(): Promise<UnforgettableSettings> {
  const res = await authFetch("/api/unforgettable/settings");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Unforgettable settings"),
    );
  }
  return (await res.json()) as UnforgettableSettings;
}

export async function loadUnforgettableSettings() {
  if (cached) return cached;
  inFlight ??= fetchSettings()
    .then(publish)
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

export function peekUnforgettableSettings() {
  return cached;
}

export async function updateUnforgettableSettings(
  patch: Partial<UnforgettableSettings>,
): Promise<UnforgettableSettings> {
  const res = await authFetch("/api/unforgettable/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update Unforgettable settings"),
    );
  }
  return publish((await res.json()) as UnforgettableSettings);
}
