// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DiffusionTrainableFamily,
  DiffusionTrainingRunSummary,
  DiffusionTrainingStatus,
} from "../api";
import type { AdapterDeployRequest } from "../stores/image-workflow-store";

export type DiffusionDeployRun = Pick<
  DiffusionTrainingRunSummary,
  "base_model" | "family" | "catalog_path" | "instance_prompt"
>;

function normalizedRepo(repo: string): string {
  return repo.trim().toLowerCase();
}

export function resolveDiffusionDeployBase(
  family: DiffusionTrainableFamily | undefined,
  trainedBase: string,
): string {
  const key = normalizedRepo(trainedBase);
  const variant = Object.entries(family?.deploy_bases ?? {}).find(
    ([trainingRepo]) => normalizedRepo(trainingRepo) === key,
  );
  if (variant) return variant[1];

  if (
    family?.deploy_base &&
    family.base_repos.some((repo) => normalizedRepo(repo) === key)
  ) {
    return family.deploy_base;
  }
  return trainedBase;
}

export function diffusionDeployRequest(
  families: DiffusionTrainableFamily[] | undefined,
  run: DiffusionDeployRun,
): AdapterDeployRequest | null {
  if (!run.base_model) return null;
  const family = run.family || "";
  const rec = families?.find((f) => f.name === family);
  return {
    baseRepo: resolveDiffusionDeployBase(rec, run.base_model),
    family,
    families: rec?.lora_families ?? [family],
    catalogPath: run.catalog_path || "",
    trigger: run.instance_prompt?.trim() || "",
  };
}

// The job's own trigger, since the form may be edited or empty after a reload or resume; older backends omit it.
export function liveDeployRun(
  status: Pick<DiffusionTrainingStatus, "base_model" | "family" | "catalog_path" | "instance_prompt">,
  record: DiffusionDeployRun | null,
  form: { baseModel: string; family: string; trigger: string },
): DiffusionDeployRun {
  return {
    base_model: status.base_model || form.baseModel,
    family: status.family || form.family,
    catalog_path: status.catalog_path,
    instance_prompt:
      status.instance_prompt !== undefined
        ? status.instance_prompt
        : record ? record.instance_prompt : form.trigger,
  };
}

/**
 * Training base paired with a loaded inference checkpoint, or null. Distilled variants are not
 * trainable, so an exact match would fall back to the wrong base (Klein 4B).
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
  // A mirror keeps the upstream repo NAME, so fold to it; only an offered base is returned.
  const name = normalizedRepo(trainingRepo.split("/").pop() ?? "");
  if (!name) return null;
  return (
    family.base_repos.find(
      (repo) => normalizedRepo(repo.split("/").pop() ?? "") === name,
    ) ?? null
  );
}

// The catalog id is the file stem, or the full name when a same-stem file of the other format exists.
export function catalogFileName(catalogPath: string): string {
  return catalogPath.split(/[\\/]/).pop() ?? catalogPath;
}

// Same format only (a stale run must not pick a same-stem GGUF), stem before full name (`x.safetensors.safetensors`).
export function matchCatalogId(
  loras: { id: string; format: string }[],
  fileName: string,
): string | undefined {
  const match = /^(.*)\.(safetensors|gguf)$/i.exec(fileName);
  const ids = loras
    .filter((l) => !match || l.format === match[2].toLowerCase())
    .map((l) => l.id);
  const stem = match ? match[1] : fileName;
  return ids.find((id) => id === stem) ?? ids.find((id) => id === fileName);
}
