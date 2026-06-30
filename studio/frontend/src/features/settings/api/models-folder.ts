// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ModelsFolder = {
  path: string;
};

// The path is resolved once at backend startup and never changes, so cache it
// and dedupe concurrent loads (same shape as the sibling settings loaders).
let cachedModelsFolder: ModelsFolder | null = null;
let inFlightModelsFolder: Promise<ModelsFolder> | null = null;

async function fetchModelsFolder(): Promise<ModelsFolder> {
  const res = await authFetch("/api/hub/models-folder");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load models folder"),
    );
  }
  const data = (await res.json()) as { path: string };
  return { path: data.path };
}

export async function loadModelsFolder(): Promise<ModelsFolder> {
  if (cachedModelsFolder) {
    return cachedModelsFolder;
  }
  inFlightModelsFolder ??= fetchModelsFolder()
    .then((folder) => {
      cachedModelsFolder = folder;
      return folder;
    })
    .finally(() => {
      inFlightModelsFolder = null;
    });
  return inFlightModelsFolder;
}
