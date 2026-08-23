// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiErrorText } from "@/lib/format-fastapi-error";

export interface CachedModelPath {
  path: string;
  // biome-ignore lint/style/useNamingConvention: API response schema
  is_dir: boolean;
}

export interface BrowseEntry {
  name: string;
  // biome-ignore lint/style/useNamingConvention: API response schema
  has_models: boolean;
  hidden: boolean;
}

export interface BrowseFoldersResponse {
  current: string;
  parent: string | null;
  entries: BrowseEntry[];
  suggestions: string[];
  truncated?: boolean;
  // biome-ignore lint/style/useNamingConvention: API response schema
  model_files_here?: number;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(apiErrorText(response.status, body));
  }
  return body as T;
}

export async function getCachedModelPath(
  repoId: string,
  variant?: string,
  cachePath?: string | null,
): Promise<CachedModelPath> {
  const params = new URLSearchParams();
  params.set("repo_id", repoId);
  if (variant) {
    params.set("variant", variant);
  }
  if (cachePath) {
    params.set("cache_path", cachePath);
  }
  const response = await authFetch(
    `/api/models/cached-model-path?${params.toString()}`,
  );
  return parseJsonOrThrow<CachedModelPath>(response);
}

export async function revealCachedModel(
  repoId: string,
  variant?: string,
  cachePath?: string | null,
): Promise<void> {
  const payload: Record<string, string> = {
    // biome-ignore lint/style/useNamingConvention: API request schema
    repo_id: repoId,
  };
  if (variant) {
    payload.variant = variant;
  }
  if (cachePath) {
    payload.cache_path = cachePath;
  }
  const response = await authFetch("/api/models/reveal-cached-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await parseJsonOrThrow<unknown>(response);
}

export async function listRecommendedFolders(): Promise<string[]> {
  const response = await authFetch("/api/models/recommended-folders");
  const data = await parseJsonOrThrow<{ folders: string[] }>(response);
  return data.folders;
}

export async function browseFolders(
  path?: string,
  showHidden = false,
  signal?: AbortSignal,
): Promise<BrowseFoldersResponse> {
  const params = new URLSearchParams();
  if (path !== undefined && path !== null) {
    params.set("path", path);
  }
  if (showHidden) {
    params.set("show_hidden", "true");
  }
  const query = params.toString();
  const response = await authFetch(
    `/api/models/browse-folders${query ? `?${query}` : ""}`,
    signal ? { signal } : undefined,
  );
  return parseJsonOrThrow<BrowseFoldersResponse>(response);
}
