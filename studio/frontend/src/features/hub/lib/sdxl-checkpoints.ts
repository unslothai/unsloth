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
