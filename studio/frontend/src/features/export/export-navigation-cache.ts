// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type LocalModelInfo, listLocalModels } from "@/features/training";
import { type ModelCheckpoints, fetchCheckpoints } from "./api/export-api";

let cachedCheckpoints: ModelCheckpoints[] | null = null;
let checkpointsRequest: Promise<ModelCheckpoints[]> | null = null;
let cachedLocalModels: LocalModelInfo[] | null = null;
let localModelsRequest: Promise<LocalModelInfo[]> | null = null;

export function getCachedCheckpoints(): ModelCheckpoints[] | null {
  return cachedCheckpoints;
}

export function getCachedLocalModels(): LocalModelInfo[] | null {
  return cachedLocalModels;
}

export function refreshCheckpoints(): Promise<ModelCheckpoints[]> {
  if (checkpointsRequest) {
    return checkpointsRequest;
  }
  const request = fetchCheckpoints()
    .then((data) => {
      cachedCheckpoints = data.models;
      return data.models;
    })
    .finally(() => {
      checkpointsRequest = null;
    });
  checkpointsRequest = request;
  return request;
}

export function refreshLocalModels(): Promise<LocalModelInfo[]> {
  if (localModelsRequest) {
    return localModelsRequest;
  }
  const request = listLocalModels()
    .then((models) => {
      cachedLocalModels = models;
      return models;
    })
    .finally(() => {
      localModelsRequest = null;
    });
  localModelsRequest = request;
  return request;
}

export async function preloadExportData(): Promise<void> {
  const requests: Promise<unknown>[] = [];
  if (cachedCheckpoints === null) {
    requests.push(refreshCheckpoints());
  }
  if (cachedLocalModels === null) {
    requests.push(refreshLocalModels());
  }
  await Promise.allSettled(requests);
}
