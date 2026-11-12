// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type NpuNamed = {
  id: string;
  model_path?: string;
  hf_repo?: string | null;
};

/** The Hugging Face repo id (owner/name), or the Lemonade id when the server sent no repo. */
export function npuRepoLabel(model: NpuNamed): string {
  return model.hf_repo?.trim() || model.id;
}

/** The repo name without its owner, as GGUF rows show theirs; the Lemonade id when there is no repo. */
export function npuDisplayName(model: NpuNamed): string {
  const repo = model.hf_repo?.trim();
  return (repo && repo.slice(repo.lastIndexOf("/") + 1)) || model.id;
}

/** npuDisplayName of the catalog model with this id or `lemonade:` path, or null when none has it. */
export function npuDisplayNameOf(
  models: readonly NpuNamed[] | null | undefined,
  idOrPath: string | null | undefined,
): string | null {
  if (!idOrPath) return null;
  const model = models?.find(
    (row) => row.model_path === idOrPath || row.id === idOrPath,
  );
  return model ? npuDisplayName(model) : null;
}

export function npuSearchText(
  model: NpuNamed & { checkpoint: string },
): string {
  return `${model.id} ${model.checkpoint} ${npuRepoLabel(model)}`;
}
