// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export const DEFAULT_UPLOAD_LIMIT_MB = 500;
export const DEFAULT_UPLOAD_LIMIT_BYTES = DEFAULT_UPLOAD_LIMIT_MB * 1024 * 1024;

const UPLOAD_LIMIT_EVENT = "unsloth-upload-limit-change";

export type UploadLimitSettings = {
  maxUploadSizeMb: number;
  maxUploadSizeBytes: number;
  maxUploadSizeLabel: string;
  defaultUploadSizeMb: number;
  minUploadSizeMb: number;
  maxAllowedUploadSizeMb: number;
};

type ApiUploadLimitSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  max_upload_size_mb: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_upload_size_bytes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_upload_size_label: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_upload_size_mb: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  min_upload_size_mb: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  max_allowed_upload_size_mb: number;
};

let cachedUploadLimit: UploadLimitSettings | null = null;
let inFlightUploadLimit: Promise<UploadLimitSettings> | null = null;

export function getCachedUploadLimitBytes() {
  return cachedUploadLimit?.maxUploadSizeBytes ?? DEFAULT_UPLOAD_LIMIT_BYTES;
}

export function getCachedUploadLimitLabel() {
  return (
    cachedUploadLimit?.maxUploadSizeLabel ?? `${DEFAULT_UPLOAD_LIMIT_MB}MB`
  );
}

export function formatUploadSize(bytes: number) {
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function subscribeUploadLimitSettings(
  listener: (settings: UploadLimitSettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<UploadLimitSettings>).detail);
  };
  window.addEventListener(UPLOAD_LIMIT_EVENT, handleChange);
  return () => window.removeEventListener(UPLOAD_LIMIT_EVENT, handleChange);
}

function fromApi(settings: ApiUploadLimitSettings): UploadLimitSettings {
  return {
    maxUploadSizeMb: settings.max_upload_size_mb,
    maxUploadSizeBytes: settings.max_upload_size_bytes,
    maxUploadSizeLabel: settings.max_upload_size_label,
    defaultUploadSizeMb: settings.default_upload_size_mb,
    minUploadSizeMb: settings.min_upload_size_mb,
    maxAllowedUploadSizeMb: settings.max_allowed_upload_size_mb,
  };
}

function cacheUploadLimit(settings: UploadLimitSettings) {
  cachedUploadLimit = settings;
  window.dispatchEvent(
    new CustomEvent(UPLOAD_LIMIT_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchUploadLimitSettings(): Promise<UploadLimitSettings> {
  const res = await authFetch("/api/settings/upload-limit");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load upload limit"));
  }
  return fromApi(await res.json());
}

export async function loadUploadLimitSettings() {
  if (cachedUploadLimit) {
    return cachedUploadLimit;
  }
  inFlightUploadLimit ??= fetchUploadLimitSettings()
    .then(cacheUploadLimit)
    .finally(() => {
      inFlightUploadLimit = null;
    });
  return inFlightUploadLimit;
}

export async function updateUploadLimitSettings(
  maxUploadSizeMb: number,
): Promise<UploadLimitSettings> {
  const res = await authFetch("/api/settings/upload-limit", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ max_upload_size_mb: maxUploadSizeMb }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update upload limit"),
    );
  }
  return cacheUploadLimit(fromApi(await res.json()));
}
