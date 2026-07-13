// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

const USER_ASSETS_BASE = "/api/user-assets";

export type UserAssetErrorDetail = {
  code?: string;
  message?: string;
  currentRevision?: number;
  current?: unknown;
  resource?: unknown;
  paths?: string[];
};

export class UserAssetApiError extends Error {
  readonly status: number;
  readonly detail: UserAssetErrorDetail;

  constructor(status: number, detail: UserAssetErrorDetail) {
    super(detail.message || `User asset request failed (${status}).`);
    this.name = "UserAssetApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function readError(response: Response): Promise<UserAssetApiError> {
  let detail: UserAssetErrorDetail = {};
  try {
    const body = (await response.json()) as {
      detail?: UserAssetErrorDetail | string;
      message?: string;
    };
    if (typeof body.detail === "string") detail = { message: body.detail };
    else if (body.detail && typeof body.detail === "object")
      detail = body.detail;
    else if (typeof body.message === "string")
      detail = { message: body.message };
  } catch {
    detail = { message: response.statusText || "Request failed." };
  }
  return new UserAssetApiError(response.status, detail);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(`${USER_ASSETS_BASE}${path}`, init);
  if (!response.ok) throw await readError(response);
  return response.json() as Promise<T>;
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export type UserAssetsBootstrap = {
  subject: string;
  importLedger: { source: string; recipes: string[]; executions: string[] };
};

export type LegacyImportItemResult = {
  id: string | null;
  outcome:
    | "imported"
    | "already_imported"
    | "redacted"
    | "id_retired"
    | "rejected"
    | "missing_parent";
  reason?: string | null;
  redactedPaths?: string[];
};

export type LegacyImportResult = {
  recipes: LegacyImportItemResult[];
  executions: LegacyImportItemResult[];
  summary: Record<string, number>;
};

export type RecipeAssetRecord<TPayload> = {
  id: string;
  name: string;
  payload: TPayload;
  learningRecipeId?: string | null;
  learningRecipeTitle?: string | null;
  revision: number;
  createdAt: number;
  updatedAt: number;
};

export function bootstrapUserAssets(): Promise<UserAssetsBootstrap> {
  return requestJson<UserAssetsBootstrap>("/bootstrap");
}

export async function listServerRecipes<TPayload>(): Promise<
  RecipeAssetRecord<TPayload>[]
> {
  const response = await requestJson<{
    recipes: RecipeAssetRecord<TPayload>[];
  }>("/recipes");
  return response.recipes;
}

export function getServerRecipe<TPayload>(id: string) {
  return requestJson<RecipeAssetRecord<TPayload>>(
    `/recipes/${encodeURIComponent(id)}`,
  );
}

export function createServerRecipe<TPayload>(
  recipe: Omit<
    RecipeAssetRecord<TPayload>,
    "revision" | "createdAt" | "updatedAt"
  >,
) {
  return requestJson<RecipeAssetRecord<TPayload>>(
    "/recipes",
    jsonInit("POST", recipe),
  );
}

export function updateServerRecipe<TPayload>(input: {
  id: string;
  name: string;
  payload: TPayload;
  learningRecipeId?: string | null;
  learningRecipeTitle?: string | null;
  revision: number;
}) {
  return requestJson<RecipeAssetRecord<TPayload>>(
    `/recipes/${encodeURIComponent(input.id)}`,
    jsonInit("PUT", input),
  );
}

export async function deleteServerRecipe(id: string, revision: number) {
  const response = await authFetch(
    `${USER_ASSETS_BASE}/recipes/${encodeURIComponent(id)}?revision=${revision}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw await readError(response);
}

export type RecipeExecutionPage<TExecution> = {
  executions: TExecution[];
  nextCursor: string | null;
};

export function listServerRecipeExecutions<TExecution>(
  recipeId: string,
  options: { cursor?: string | null; limit?: number } = {},
) {
  const params = new URLSearchParams();
  params.set("limit", String(options.limit ?? 100));
  if (options.cursor) params.set("cursor", options.cursor);
  return requestJson<RecipeExecutionPage<TExecution>>(
    `/recipes/${encodeURIComponent(recipeId)}/executions?${params.toString()}`,
  );
}

export function upsertServerRecipeExecution<TExecution>(input: {
  recipeId: string;
  executionId: string;
  metadata: object;
  revision?: number;
}) {
  return requestJson<TExecution>(
    `/recipes/${encodeURIComponent(input.recipeId)}/executions/${encodeURIComponent(input.executionId)}`,
    jsonInit("PUT", { ...input.metadata, revision: input.revision }),
  );
}

export function importLegacyUserAssets<
  TRecipe extends object,
  TExecution extends object,
>(input: {
  source: string;
  confirmSubject: string;
  recipes: TRecipe[];
  executions: TExecution[];
}) {
  return requestJson<LegacyImportResult>(
    "/legacy-import",
    jsonInit("POST", input),
  );
}

export type TrainingPresetRecord<TConfig = Record<string, unknown>> = {
  id: string;
  name: string;
  config: TConfig;
  revision: number;
  createdAt: number;
  updatedAt: number;
};

export async function listServerTrainingPresets<TConfig>() {
  const response = await requestJson<{
    presets: TrainingPresetRecord<TConfig>[];
  }>("/training-presets");
  return response.presets;
}

export function getServerTrainingPreset<TConfig>(id: string) {
  return requestJson<TrainingPresetRecord<TConfig>>(
    `/training-presets/${encodeURIComponent(id)}`,
  );
}

export function createServerTrainingPreset<TConfig>(input: {
  id: string;
  name: string;
  config: TConfig;
}) {
  return requestJson<TrainingPresetRecord<TConfig>>(
    "/training-presets",
    jsonInit("POST", input),
  );
}

export function updateServerTrainingPreset<TConfig>(input: {
  id: string;
  name: string;
  config: TConfig;
  revision: number;
}) {
  return requestJson<TrainingPresetRecord<TConfig>>(
    `/training-presets/${encodeURIComponent(input.id)}`,
    jsonInit("PUT", input),
  );
}

export async function deleteServerTrainingPreset(id: string, revision: number) {
  const response = await authFetch(
    `${USER_ASSETS_BASE}/training-presets/${encodeURIComponent(id)}?revision=${revision}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw await readError(response);
}
