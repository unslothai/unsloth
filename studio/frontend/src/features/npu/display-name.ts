// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type NpuNamed = {
  id: string;
  model_path?: string;
  hf_repo?: string | null;
  local?: { folder: string } | null;
};

/** A Q4NX folder converted on the Export page, named by its folder. */
export function npuLocalLabel(folder: string): string | null {
  const parts = folder.split("/").filter(Boolean);
  const last = parts.pop();
  // A GGUF export with Q4NX on writes into <save folder>/npu-q4nx: name it by the save folder.
  return (last === "npu-q4nx" ? parts.pop() : last) || null;
}

export function npuRepoLabel(model: NpuNamed): string {
  return model.hf_repo?.trim() || model.id;
}

/** The repo name without its owner, as GGUF rows show theirs. */
export function npuDisplayName(model: NpuNamed): string {
  if (model.local) return npuLocalLabel(model.local.folder) || model.id;
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
