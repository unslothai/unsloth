// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type EmbeddingModelSettings = {
  embeddingModel: string;
  defaultEmbeddingModel: string;
  isCustom: boolean;
};

type ApiEmbeddingModelSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  embedding_model: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_embedding_model: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  is_custom: boolean;
};

/** 409 from the backend: the model could not be verified as an embedding model
 * (wrong type, gated repo, or offline). Retry with force to save anyway. */
export class EmbeddingModelVerificationError extends Error {}

/** 403 from the backend: the repo is flagged unsafe by Hugging Face's security scan.
 * A hard block; force cannot bypass it, so it must not enter the "save anyway" flow. */
export class EmbeddingModelBlockedError extends Error {}

function fromApi(settings: ApiEmbeddingModelSettings): EmbeddingModelSettings {
  return {
    embeddingModel: settings.embedding_model,
    defaultEmbeddingModel: settings.default_embedding_model,
    isCustom: settings.is_custom,
  };
}

export async function loadEmbeddingModelSettings(): Promise<EmbeddingModelSettings> {
  const res = await authFetch("/api/settings/embedding-model");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load embedding model setting"),
    );
  }
  return fromApi(await res.json());
}

export async function updateEmbeddingModelSettings(
  embeddingModel: string,
  options?: { hfToken?: string; force?: boolean },
): Promise<EmbeddingModelSettings> {
  const res = await authFetch("/api/settings/embedding-model", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      embedding_model: embeddingModel,
      // biome-ignore lint/style/useNamingConvention: API schema
      hf_token: options?.hfToken || null,
      force: options?.force ?? false,
    }),
  });
  if (res.status === 403) {
    throw new EmbeddingModelBlockedError(
      await readFastApiError(res, "This model is blocked by a security scan"),
    );
  }
  if (res.status === 409) {
    throw new EmbeddingModelVerificationError(
      await readFastApiError(res, "Could not verify the embedding model"),
    );
  }
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save embedding model"),
    );
  }
  return fromApi(await res.json());
}

export async function resetEmbeddingModelSettings(): Promise<EmbeddingModelSettings> {
  const res = await authFetch("/api/settings/embedding-model", {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to reset embedding model"),
    );
  }
  return fromApi(await res.json());
}
