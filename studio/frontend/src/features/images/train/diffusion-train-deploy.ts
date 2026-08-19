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

/**
 * The TRAINING base paired with a checkpoint that is currently loaded for inference, or null.
 *
 * The inverse of the mapping above, and it is what the Train panel needs to preselect: the
 * distilled variants a user generates with are not trainable, so a loaded `...klein-9B` never
 * appears in `base_repos` and an exact-match preselect silently falls through to the FIRST entry.
 * For FLUX.2 Klein that is the 4B base, so opening Train with the 9B model loaded started a 9B
 * workflow from 4B weights unless the user noticed the selector.
 *
 * Only a pairing the family actually declares is returned, and only when the paired training repo
 * is offered, so this can never invent a base the backend would refuse.
 */
export function resolveDiffusionTrainingBase(
  family: DiffusionTrainableFamily | undefined,
  loadedBase: string,
): string | null {
  const key = normalizedRepo(loadedBase);
  if (!family || !key) return null;
  const pair = Object.entries(family.deploy_bases ?? {}).find(
    ([, inferenceRepo]) => normalizedRepo(inferenceRepo) === key,
  );
  const trainingRepo = pair?.[0];
  if (!trainingRepo) return null;
  const exact = family.base_repos.find(
    (repo) => normalizedRepo(repo) === normalizedRepo(trainingRepo),
  );
  if (exact) return exact;
  // A checkpoint loaded from the ungated MIRROR pairs with the mirror training id, and
  // /diffusion/info offers only the vendor ids, so the exact match above finds nothing and the
  // panel falls back to the first base -- for Klein the 4B, which is the very mix-up this
  // function exists to prevent, reached through Deploy's own mirror pairing. A mirror keeps the
  // upstream repo NAME, so fold to that; still only ever returning a base the backend offers.
  const name = normalizedRepo(trainingRepo.split("/").pop() ?? "");
  if (!name) return null;
  return (
    family.base_repos.find((repo) => normalizedRepo(repo.split("/").pop() ?? "") === name) ?? null
  );
}
