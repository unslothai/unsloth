// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export const RECIPES_API = "/api/data-recipe/recipes";

export class RecipeApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function recipeRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await authFetch(`${RECIPES_API}${path}`, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new RecipeApiError(
      typeof detail === "string" ? detail : `Request failed (${res.status})`,
      res.status,
    );
  }
  return body as T;
}
