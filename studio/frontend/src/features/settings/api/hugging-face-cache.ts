// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  bumpInventoryVersion,
  invalidateGgufVariantsCache,
} from "@/features/hub";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type HuggingFaceCacheSettings = {
  cacheHome: string;
  hubCache: string;
  xetCache: string;
  source: "default" | "studio" | "environment";
  editable: boolean;
  isCustom: boolean;
  available: boolean;
  writable: boolean;
  freeBytes: number | null;
  environmentVariable: string | null;
};

type ApiHuggingFaceCacheSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  cache_home: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  hub_cache: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  xet_cache: string;
  source: HuggingFaceCacheSettings["source"];
  editable: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  is_custom: boolean;
  available: boolean;
  writable: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  free_bytes: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  environment_variable: string | null;
};

function fromApi(value: ApiHuggingFaceCacheSettings): HuggingFaceCacheSettings {
  return {
    cacheHome: value.cache_home,
    hubCache: value.hub_cache,
    xetCache: value.xet_cache,
    source: value.source,
    editable: value.editable,
    isCustom: value.is_custom,
    available: value.available,
    writable: value.writable,
    freeBytes: value.free_bytes,
    environmentVariable: value.environment_variable,
  };
}

export async function loadHuggingFaceCacheSettings() {
  const response = await authFetch("/api/settings/hugging-face-cache");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to load the model cache location",
      ),
    );
  }
  return fromApi(await response.json());
}

export async function updateHuggingFaceCacheSettings(cacheHome: string | null) {
  const response = await authFetch("/api/settings/hugging-face-cache", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ cache_home: cacheHome }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to update the model cache location",
      ),
    );
  }
  const settings = fromApi(await response.json());
  bumpInventoryVersion();
  invalidateGgufVariantsCache();
  return settings;
}
