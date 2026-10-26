// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  ModelAnnotation,
  ModelAnnotations,
} from "../lib/model-annotations";

const URL = "/api/settings/model-annotations";

/** Fetch one account's annotations in a single request for the On Device inventory. */
export async function readModelAnnotations(): Promise<ModelAnnotations> {
  const response = await authFetch(URL);
  if (!response.ok)
    throw new Error(
      await readFastApiError(response, "Could not load model details"),
    );
  return ((await response.json()) as { annotations: ModelAnnotations })
    .annotations;
}

/** Replace one row's details; a confirmed null means its personal details were cleared. */
export async function writeModelAnnotation(
  modelId: string,
  annotation: ModelAnnotation,
): Promise<ModelAnnotation | null> {
  const response = await authFetch(URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, annotation }),
  });
  if (!response.ok)
    throw new Error(
      await readFastApiError(response, "Could not save model details"),
    );
  return ((await response.json()) as { annotation: ModelAnnotation | null })
    .annotation;
}
