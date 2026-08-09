// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionTrainableFamily } from "../api";

function normalizedRepo(repo: string): string {
  return repo.trim().toLowerCase();
}

/** Resolve the inference checkpoint paired with the base a diffusion adapter trained on. */
export function resolveDiffusionDeployBase(
  family: DiffusionTrainableFamily | undefined,
  trainedBase: string,
): string {
  const key = normalizedRepo(trainedBase);
  const variant = Object.entries(family?.deploy_bases ?? {}).find(
    ([trainingRepo]) => normalizedRepo(trainingRepo) === key,
  );
  if (variant) return variant[1];

  // Backward-compatible family-wide mapping used by Krea 2 and older backends.
  if (
    family?.deploy_base &&
    family.base_repos.some((repo) => normalizedRepo(repo) === key)
  ) {
    return family.deploy_base;
  }
  return trainedBase;
}
