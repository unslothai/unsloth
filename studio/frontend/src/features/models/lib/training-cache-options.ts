// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SelectedResourceRef } from "../types";

export type TrainingCacheOptions = {
  knownCached: boolean;
  localPath: string | null;
};

export function cacheOptionsForTraining(
  resource: Pick<SelectedResourceRef, "cacheState" | "localPath">,
): TrainingCacheOptions {
  if (resource.cacheState === "cached") {
    return {
      knownCached: true,
      localPath: resource.localPath,
    };
  }
  if (resource.cacheState === "local") {
    return {
      knownCached: false,
      localPath: resource.localPath,
    };
  }
  return {
    knownCached: false,
    localPath: null,
  };
}
