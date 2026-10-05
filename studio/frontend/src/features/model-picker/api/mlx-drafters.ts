// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { MlxDrafter } from "@/lib/speculative-modes";

/** Cached drafters an MLX load of `modelPath` could name; empty when the backend cannot say. */
export async function fetchMlxDrafters(
  modelPath: string,
  signal?: AbortSignal,
): Promise<MlxDrafter[]> {
  const response = await authFetch("/api/inference/mlx-drafters", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({ model_path: modelPath }),
  });
  if (!response.ok) return [];
  const body = (await response.json()) as { drafters?: unknown };
  return Array.isArray(body.drafters)
    ? body.drafters
        .filter((d) => typeof d?.repo_id === "string" && typeof d?.kind === "string")
        .map((d) => ({ repo: d.repo_id, kind: d.kind }))
    : [];
}
