// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { bumpInventoryVersion } from "@/features/hub";
// Leaf module, not the barrel: the barrel re-exports .tsx panels, and tests stub
// it down to the cache bump alone.
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type EmbeddingModelSettings = {
  embeddingModel: string;
  embeddingGgufRepo: string;
  defaultEmbeddingModel: string;
  defaultEmbeddingGgufRepo: string;
  isCustom: boolean;
  /** THIS model is held in memory right now, for the status line. */
  loaded: boolean;
  /** ANY embedder is resident, so Unload has something to do. Saving a new model
   * does not release the old one, so not the same question as `loaded`. */
  backendLoaded: boolean;
};

type ApiEmbeddingModelSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  embedding_model: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  embedding_gguf_repo: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_embedding_model: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_embedding_gguf_repo: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  is_custom: boolean;
  loaded?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  backend_loaded?: boolean;
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
    embeddingGgufRepo: settings.embedding_gguf_repo,
    defaultEmbeddingModel: settings.default_embedding_model,
    defaultEmbeddingGgufRepo: settings.default_embedding_gguf_repo,
    isCustom: settings.is_custom,
    loaded: settings.loaded ?? false,
    // A backend predating this field answers only about the selected model,
    // which is the old behaviour and the right fallback.
    backendLoaded: settings.backend_loaded ?? settings.loaded ?? false,
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
  options?: {
    hfToken?: string;
    force?: boolean;
    ggufRepo?: string | null;
    backend?: EmbeddingModelResolution["backend"] | null;
  },
): Promise<EmbeddingModelSettings> {
  const res = await authFetch("/api/settings/embedding-model", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      embedding_model: embeddingModel,
      // biome-ignore lint/style/useNamingConvention: API schema
      hf_token: options?.hfToken || null,
      // biome-ignore lint/style/useNamingConvention: API schema
      gguf_repo: options?.ggufRepo ?? null,
      backend: options?.backend ?? null,
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
  const settings = fromApi(await res.json());
  bumpInventoryVersion();
  return settings;
}

/** What saving a model would need fetched, and whether it is already on disk. */
export type EmbeddingModelResolution = {
  embeddingModel: string;
  backend: "llama" | "sentence-transformers";
  /** Repo to hand the download manager; null when nothing needs fetching. */
  downloadRepo: string | null;
  /** The selected GGUF family (all shards when split), on llama-server. */
  files: string[] | null;
  cached: boolean;
  sizeBytes: number | null;
  /** Why the model is unusable here; the detail the save would refuse with. */
  error: string | null;
};

type ApiEmbeddingModelResolution = {
  // biome-ignore lint/style/useNamingConvention: API schema
  embedding_model: string;
  backend: "llama" | "sentence-transformers";
  // biome-ignore lint/style/useNamingConvention: API schema
  download_repo: string | null;
  files: string[] | null;
  cached: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  size_bytes: number | null;
  error: string | null;
};

export async function resolveEmbeddingModel(
  embeddingModel: string,
  options?: { hfToken?: string },
): Promise<EmbeddingModelResolution> {
  const params = new URLSearchParams({ model: embeddingModel });
  const res = await authFetch(
    `/api/settings/embedding-model/resolve?${params}`,
    // The token rides a header so a gated repo's credential stays out of the URL.
    { headers: hubTokenHeader(options?.hfToken) },
  );
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to check the embedding model"),
    );
  }
  const body = (await res.json()) as ApiEmbeddingModelResolution;
  return {
    embeddingModel: body.embedding_model,
    backend: body.backend,
    downloadRepo: body.download_repo,
    files: body.files,
    cached: body.cached,
    sizeBytes: body.size_bytes,
    error: body.error,
  };
}

export async function unloadEmbeddingModel(): Promise<EmbeddingModelSettings> {
  const res = await authFetch("/api/settings/embedding-model/unload", {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to unload the embedding model"),
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
  const settings = fromApi(await res.json());
  bumpInventoryVersion();
  return settings;
}
