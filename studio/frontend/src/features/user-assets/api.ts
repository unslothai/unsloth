// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthSubjectKey } from "@/features/auth";

const USER_ASSETS_BASE = "/api/user-assets";
const USER_ASSET_REQUEST_TIMEOUT_MS = 15_000;

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

type UserAssetRequestOptions = {
  expectedSubjectKey?: string;
};

async function request(
  path: string,
  init?: RequestInit,
  options: UserAssetRequestOptions = {},
): Promise<Response> {
  const controller = new AbortController();
  let timedOut = false;
  const abortFromCaller = () => controller.abort(init?.signal?.reason);
  if (init?.signal?.aborted) abortFromCaller();
  else init?.signal?.addEventListener("abort", abortFromCaller, { once: true });
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, USER_ASSET_REQUEST_TIMEOUT_MS);
  try {
    const method = (init?.method ?? "GET").toUpperCase();
    const mutatesUserAssets =
      method === "POST" || method === "PUT" || method === "DELETE";
    const expectedSubjectKey =
      options.expectedSubjectKey ??
      (mutatesUserAssets ? getAuthSubjectKey() : undefined);
    const guard = expectedSubjectKey ? { expectedSubjectKey } : undefined;
    return await authFetch(
      `${USER_ASSETS_BASE}${path}`,
      {
        ...init,
        signal: controller.signal,
      },
      guard,
    );
  } catch (error) {
    if (timedOut) {
      throw new Error("The persistence request timed out. Please try again.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    init?.signal?.removeEventListener("abort", abortFromCaller);
  }
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  options: UserAssetRequestOptions = {},
): Promise<T> {
  const response = await request(path, init, options);
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

function assetPathSegment(id: string): string {
  if (!id || id.includes("/") || id.includes("\\")) {
    throw new Error("Asset ids must be safe to use as one URL path segment.");
  }
  return encodeURIComponent(id);
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

export function bootstrapUserAssets(
  options: { signal?: AbortSignal; expectedSubjectKey?: string } = {},
): Promise<UserAssetsBootstrap> {
  return requestJson<UserAssetsBootstrap>(
    "/bootstrap",
    { signal: options.signal },
    options,
  );
}

export async function listServerRecipes<TPayload>(): Promise<
  RecipeAssetRecord<TPayload>[]
> {
  const response = await requestJson<{
    recipes: RecipeAssetRecord<TPayload>[];
  }>("/recipes");
  return response.recipes;
}

export function getServerRecipe<TPayload>(
  id: string,
  options: { signal?: AbortSignal } = {},
) {
  return requestJson<RecipeAssetRecord<TPayload>>(
    `/recipes/${assetPathSegment(id)}`,
    { signal: options.signal },
  );
}

export function createServerRecipe<TPayload>(
  recipe: Omit<
    RecipeAssetRecord<TPayload>,
    "revision" | "createdAt" | "updatedAt"
  >,
  options: { signal?: AbortSignal; expectedSubjectKey?: string } = {},
) {
  return requestJson<RecipeAssetRecord<TPayload>>(
    "/recipes",
    {
      ...jsonInit("POST", recipe),
      signal: options.signal,
    },
    options,
  );
}

export function updateServerRecipe<TPayload>(
  input: {
    id: string;
    name: string;
    payload: TPayload;
    learningRecipeId?: string | null;
    learningRecipeTitle?: string | null;
    revision: number;
  },
  options: { signal?: AbortSignal; expectedSubjectKey?: string } = {},
) {
  return requestJson<RecipeAssetRecord<TPayload>>(
    `/recipes/${assetPathSegment(input.id)}`,
    { ...jsonInit("PUT", input), signal: options.signal },
    options,
  );
}

export async function deleteServerRecipe(
  id: string,
  revision: number,
  options: UserAssetRequestOptions = {},
) {
  const response = await request(
    `/recipes/${assetPathSegment(id)}?revision=${revision}`,
    { method: "DELETE" },
    options,
  );
  if (!response.ok) throw await readError(response);
}

export type RecipeExecutionPage<TExecution> = {
  executions: TExecution[];
  nextCursor: string | null;
  resumable?: TExecution | null;
};

export function listServerRecipeExecutions<TExecution>(
  recipeId: string,
  options: { cursor?: string | null; limit?: number } = {},
) {
  const params = new URLSearchParams();
  params.set("limit", String(options.limit ?? 100));
  if (options.cursor) params.set("cursor", options.cursor);
  return requestJson<RecipeExecutionPage<TExecution>>(
    `/recipes/${assetPathSegment(recipeId)}/executions?${params.toString()}`,
  );
}

export function upsertServerRecipeExecution<TExecution>(
  input: {
    recipeId: string;
    executionId: string;
    metadata: object;
    revision?: number;
  },
  options: UserAssetRequestOptions = {},
) {
  return requestJson<TExecution>(
    `/recipes/${assetPathSegment(input.recipeId)}/executions/${assetPathSegment(input.executionId)}`,
    jsonInit("PUT", { ...input.metadata, revision: input.revision }),
    options,
  );
}

export function importLegacyUserAssets<
  TRecipe extends object,
  TExecution extends object,
>(
  input: {
    source: string;
    confirmSubject: string;
    recipes: TRecipe[];
    executions: TExecution[];
  },
  options: { signal?: AbortSignal; expectedSubjectKey?: string } = {},
) {
  return requestJson<LegacyImportResult>(
    "/legacy-import",
    {
      ...jsonInit("POST", input),
      signal: options.signal,
    },
    options,
  );
}
