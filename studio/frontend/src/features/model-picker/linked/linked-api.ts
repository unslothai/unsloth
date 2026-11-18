// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantsResponse,
} from "@/features/hub/inventory/api";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { linkedProxyPath } from "./linked-id";

async function call<T>(
  instanceId: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await authFetch(linkedProxyPath(instanceId, path), init);
  if (!res.ok) throw new Error(await readFastApiError(res));
  return res.json() as Promise<T>;
}

export async function linkedCachedModels(instanceId: string): Promise<{
  gguf: CachedGgufRepo[];
  models: CachedModelRepo[];
}> {
  const [gguf, models] = await Promise.all([
    call<{ cached: CachedGgufRepo[] }>(instanceId, "/api/hub/cached-gguf"),
    call<{ cached: CachedModelRepo[] }>(instanceId, "/api/hub/cached-models"),
  ]);
  return { gguf: gguf.cached ?? [], models: models.cached ?? [] };
}

const variantsCache = new Map<string, Promise<GgufVariantsResponse>>();

export function linkedGgufVariants(
  instanceId: string,
  repoId: string,
  fresh = false,
): Promise<GgufVariantsResponse> {
  const key = `${instanceId}::${repoId}`;
  const hit = variantsCache.get(key);
  if (hit && !fresh) return hit;
  const promise = call<GgufVariantsResponse>(
    instanceId,
    `/api/hub/gguf-variants?${new URLSearchParams({ repo_id: repoId })}`,
  );
  variantsCache.set(key, promise);
  promise.catch(() => variantsCache.delete(key));
  return promise;
}
