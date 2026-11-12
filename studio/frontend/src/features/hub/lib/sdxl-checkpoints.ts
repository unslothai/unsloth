// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";
import type { ImageCheckpointFile } from "../download-manager/required-assets";
import type { ModelSizeInfo } from "./dataset-size";
import { hubTokenHeader } from "./hub-token-header";

// ModelSizeInfo is already cached by endpoint, repo and token. Keep only confirmed header verdicts.
const headers = new WeakMap<ModelSizeInfo, Map<string, boolean>>();

export async function confirmedSdxlCheckpoints(
  repoId: string,
  info: ModelSizeInfo,
  token: string | undefined,
  signal: AbortSignal,
): Promise<ImageCheckpointFile[]> {
  const candidates = info.imageCheckpointCandidates ?? [];
  const verdicts = headers.get(info) ?? new Map<string, boolean>();
  headers.set(info, verdicts);
  const unchecked = candidates.filter(({ filename }) => !verdicts.has(filename));
  for (let i = 0; i < unchecked.length; i += 4) {
    if (signal.aborted) break;
    await Promise.all(unchecked.slice(i, i + 4).map(async ({ filename }) => {
      const query = new URLSearchParams({ repo_id: repoId, filename });
      if (info.revision) query.set("revision", info.revision);
      try {
        const res = await authFetch(`/api/hub/sdxl-checkpoint?${query}`, {
          headers: hubTokenHeader(token),
          signal,
        });
        if (res.ok) {
          const verdict: unknown = await res.json();
          if (typeof verdict === "boolean") verdicts.set(filename, verdict);
        }
      } catch {
        // A failed header read can be retried on the next visit.
      }
    }));
  }
  return candidates.filter(({ filename }) => verdicts.get(filename) === true);
}

// Offline reopen has no metadata to rediscover from; without this, Run falls back to a pipeline pick.
const STORAGE_KEY = "unsloth.hub.sdxlCheckpoints";
const MAX_REPOS = 50;

function readRemembered(): Record<string, ImageCheckpointFile[]> {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, ImageCheckpointFile[]>)
      : {};
  } catch {
    return {};
  }
}

export function rememberSdxlCheckpoints(repoId: string, files: ImageCheckpointFile[]): void {
  const key = repoId.toLowerCase();
  const next = readRemembered();
  delete next[key];
  next[key] = files;
  const keys = Object.keys(next);
  for (const old of keys.slice(0, Math.max(0, keys.length - MAX_REPOS))) delete next[old];
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Storage full or disabled: online discovery still works.
  }
}

export function rememberedSdxlCheckpoints(repoId: string): ImageCheckpointFile[] | undefined {
  const files = readRemembered()[repoId.toLowerCase()];
  return Array.isArray(files)
    ? files.filter((f) => typeof f?.filename === "string" && /^[^/\\]+\.safetensors$/i.test(f.filename))
    : undefined;
}
