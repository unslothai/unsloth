// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withPlanBreakdown } from "@/features/hub/download-manager/download-breakdown";

type DiffusionPickSource = "hub" | "lora" | "exported" | "local" | "external";

/** A cached row pinned to its validated snapshot loads by path; `displayRepoId` keeps the logical id for planning. */
export function diffusionPipelineLoadTarget(
  model: string,
  meta: { loadId?: string | null; source: DiffusionPickSource },
): { repoId: string; displayRepoId?: string; source: DiffusionPickSource; onDevice: boolean } {
  const loadId = meta.loadId?.trim();
  return loadId && loadId !== model.trim()
    ? { repoId: loadId, displayRepoId: model, source: meta.source, onDevice: true }
    : { repoId: model, source: meta.source, onDevice: meta.source === "local" };
}

/** Plan entries to stage. A pinned snapshot is already on disk (and its Hub revision may move), so only companions download. */
export function diffusionStagingEntries(
  entries: readonly {
    repo_id: string;
    files: string[];
    bytes: number;
    file_bytes?: Record<string, number>;
    gguf_filename: string | null;
    checkpoint?: boolean;
  }[],
  repoId: string,
  opts: { filename?: string; displayRepoId?: string; checkpointBytes?: number },
) {
  const planRepoId = opts.displayRepoId ?? repoId;
  const staged = entries
    .map((e) => ({
      repoId: e.repo_id,
      files: e.files,
      bytes: e.bytes,
      fileBytes: e.file_bytes,
      ggufFilename: e.gguf_filename,
      // `??`, not `||`: a planner answering false is an answer; the fallback is only for older backends.
      checkpoint:
        e.checkpoint ?? (opts.filename ? e.files.includes(opts.filename) : e.repo_id === planRepoId),
    }))
    .filter((e) => planRepoId === repoId || !e.checkpoint);
  return withPlanBreakdown(staged, opts.checkpointBytes);
}
