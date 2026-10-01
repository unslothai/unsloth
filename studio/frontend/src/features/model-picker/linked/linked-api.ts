// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantsResponse,
} from "@/features/hub/inventory/api";
import {
  type LinkedInstance,
  testLinkedInstance,
} from "@/features/settings/api/linked-instances";
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

const post = <T>(instanceId: string, path: string, body: unknown) =>
  call<T>(instanceId, path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export async function linkedDefaultModels(
  instanceId: string,
): Promise<string[]> {
  const res = await call<{ default_models?: string[] }>(
    instanceId,
    "/api/models/list",
  );
  return res.default_models ?? [];
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

export type LinkedLoadStage =
  | {
      stage: "downloading";
      fraction: number | null;
      bytes: number;
      total: number;
    }
  | { stage: "loading"; fraction: number | null };

type DownloadStatus = { state: string; error?: string | null };
type DownloadProgress = {
  downloaded_bytes: number;
  expected_bytes: number;
  progress: number;
};
type LoadProgress = { phase: string | null; fraction: number };

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

async function downloadOnLinked(
  instanceId: string,
  repoId: string,
  variant: string | null,
  expectedBytes: number,
  onStage: (stage: LinkedLoadStage) => void,
): Promise<void> {
  await post(instanceId, "/api/hub/download", {
    repo_id: repoId,
    gguf_variant: variant,
    transport_mode: "auto",
  });
  const statusQuery = new URLSearchParams({
    repo_id: repoId,
    gguf_variant: variant ?? "",
  });
  const progressPath = variant
    ? `/api/hub/gguf-download-progress?${new URLSearchParams({ repo_id: repoId, variant, expected_bytes: String(expectedBytes) })}`
    : `/api/hub/download-progress?${new URLSearchParams({ repo_id: repoId, expected_bytes: String(expectedBytes) })}`;
  for (;;) {
    const [status, progress] = await Promise.all([
      call<DownloadStatus>(
        instanceId,
        `/api/hub/download-status?${statusQuery}`,
      ),
      call<DownloadProgress>(instanceId, progressPath).catch(() => null),
    ]);
    if (status.state === "complete") return;
    if (status.state === "error" || status.state === "cancelled") {
      throw new Error(
        status.error || "The download stopped on the linked instance.",
      );
    }
    const total = progress?.expected_bytes || expectedBytes;
    onStage({
      stage: "downloading",
      fraction:
        progress && total > 0
          ? Math.min(1, progress.downloaded_bytes / total)
          : null,
      bytes: progress?.downloaded_bytes ?? 0,
      total,
    });
    await sleep(1500);
  }
}

/**
 * Download (when needed) and load a chat model on a linked instance, then return the
 * `@name/<id>` this server routes to it.
 */
export async function loadChatOnLinked(
  instance: LinkedInstance,
  pick: {
    repoId: string;
    loadId?: string | null;
    variant?: string | null;
    downloaded: boolean;
    expectedBytes?: number;
  },
  onStage: (stage: LinkedLoadStage) => void,
): Promise<string> {
  const variant = pick.variant ?? null;
  if (!pick.downloaded) {
    await downloadOnLinked(
      instance.id,
      pick.repoId,
      variant,
      pick.expectedBytes ?? 0,
      onStage,
    );
  }
  onStage({ stage: "loading", fraction: null });
  let done = false;
  const poll = (async () => {
    while (!done) {
      await sleep(1200);
      const progress = await call<LoadProgress>(
        instance.id,
        "/api/inference/load-progress",
      ).catch(() => null);
      if (!done && progress?.phase && progress.phase !== "ready") {
        onStage({ stage: "loading", fraction: progress.fraction || null });
      }
    }
  })();
  try {
    await post(instance.id, "/api/inference/load", {
      model_path: pick.loadId || pick.repoId,
      gguf_variant: variant,
      max_seq_length: 0,
      load_in_4bit: true,
      is_lora: false,
      hf_token: null,
    });
  } finally {
    done = true;
    await poll;
  }
  const prefix = `@${instance.name}/`;
  const status = await testLinkedInstance(instance.id).catch(() => null);
  const repo = pick.repoId.toLowerCase();
  return (
    status?.loaded.find((id) => id.toLowerCase().startsWith(prefix + repo)) ??
    status?.loaded[0] ??
    prefix + pick.repoId
  );
}
