// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of api.ts, which the eager chat chunk carries: only the Images page lists embeddings.
import { authFetch } from "@/features/auth";
import { parseJson } from "./api";

// A textual-inversion embedding file (from GET /api/models/diffusion-embeddings); `trigger` is the file stem.
export interface DiffusionEmbeddingInfo {
  id: string;
  trigger: string;
  kind: string | null;
  vectors: number;
  size_bytes: number;
}

/** List textual-inversion embeddings usable by a model family. */
export async function listDiffusionEmbeddings(
  family: string,
): Promise<{ embeddings: DiffusionEmbeddingInfo[]; embeddings_dir?: string }> {
  const data = await parseJson<{ embeddings: DiffusionEmbeddingInfo[]; embeddings_dir?: string }>(
    await authFetch(`/api/models/diffusion-embeddings?family=${encodeURIComponent(family)}`),
  );
  return { embeddings: data.embeddings ?? [], embeddings_dir: data.embeddings_dir };
}
