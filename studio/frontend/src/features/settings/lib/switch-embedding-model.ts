// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type EmbeddingModelResolution,
  resolveEmbeddingModel,
  updateEmbeddingModelSettings,
} from "../api/embedding-model";
import { useEmbeddingModelStore } from "../stores/embedding-model-store";

export type EmbeddingSwitchResult =
  /** `needsDownload`: saved, but indexing needs the files on disk first. */
  | { status: "saved"; needsDownload: boolean }
  /** A newer pick from another surface won; say nothing. */
  | { status: "superseded" }
  | { status: "failed"; message: string };

/** Switch the embedding model from outside Settings (the RAG menu). Same order as the
 *  Settings picker: claim save order, resolve, then save what the resolve found. */
export async function switchEmbeddingModel(
  model: string,
  hfToken: string | undefined,
): Promise<EmbeddingSwitchResult> {
  const store = useEmbeddingModelStore.getState();
  const reservation = store.beginSave();
  let plan: EmbeddingModelResolution | null = null;
  try {
    plan = await resolveEmbeddingModel(model, { hfToken });
  } catch {
    // Resolve is advisory; the save still verifies.
  }
  if (!store.isSaveCurrent(reservation)) return { status: "superseded" };
  if (plan?.error) return { status: "failed", message: plan.error };
  try {
    const stood = await store.save(
      () =>
        updateEmbeddingModelSettings(model, {
          hfToken,
          ggufRepo: plan?.backend === "llama" ? (plan.downloadRepo ?? null) : null,
          backend: plan?.backend ?? null,
        }),
      reservation,
    );
    if (!stood) return { status: "superseded" };
    return {
      status: "saved",
      needsDownload: Boolean(plan && !plan.cached && plan.downloadRepo),
    };
  } catch (error) {
    return { status: "failed", message: error instanceof Error ? error.message : "" };
  }
}

/** "unsloth/bge-small-en-v1.5" -> "bge-small-en-v1.5"; local paths keep their last segment. */
export function embeddingModelName(model: string): string {
  const id = model.trim().replace(/[\\/]+$/, "");
  return id.slice(Math.max(id.lastIndexOf("/"), id.lastIndexOf("\\")) + 1) || id;
}

/** Owner of a repo id, or "" for a bare name. */
export function embeddingModelOwner(model: string): string {
  const id = model.trim();
  const slash = id.indexOf("/");
  return slash > 0 && !id.startsWith("/") && !id.startsWith(".") ? id.slice(0, slash) : "";
}
