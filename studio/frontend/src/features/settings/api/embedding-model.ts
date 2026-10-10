// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { bumpInventoryVersion } from "@/features/hub";
// Leaf module, not the barrel: tests stub the barrel down to the cache bump.
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type EmbeddingModelSettings = {
  embeddingModel: string;
  embeddingGgufRepo: string;
  defaultEmbeddingModel: string;
  defaultEmbeddingGgufRepo: string;
  isCustom: boolean;
  loaded: boolean;
  /** ANY embedder is resident; saving a new model does not release the old one. */
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

/** 409: not verifiable as an embedding model; retry with force to save anyway. */
export class EmbeddingModelVerificationError extends Error {}

/** 403: flagged unsafe by HF; force cannot bypass it. */
export class EmbeddingModelBlockedError extends Error {}

function fromApi(settings: ApiEmbeddingModelSettings): EmbeddingModelSettings {
  return {
    embeddingModel: settings.embedding_model,
    embeddingGgufRepo: settings.embedding_gguf_repo,
    defaultEmbeddingModel: settings.default_embedding_model,
    defaultEmbeddingGgufRepo: settings.default_embedding_gguf_repo,
    isCustom: settings.is_custom,
    loaded: settings.loaded ?? false,
    // Older backends only report the selected model.
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

export type EmbeddingModelResolution = {
  embeddingModel: string;
  backend: "llama" | "sentence-transformers";
  downloadRepo: string | null;
  files: string[] | null;
  cached: boolean;
  sizeBytes: number | null;
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
    // Header keeps a gated repo's token out of the URL.
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
