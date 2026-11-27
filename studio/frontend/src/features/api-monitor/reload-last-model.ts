// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads the durable last-local-load record, not the bounded, clearable monitor ring buffer.

export type LastLoadRecord = {
  id: string;
  kind: "gguf" | "model";
  ggufVariant: string | null;
};

export type ReloadTarget<C = unknown> = LastLoadRecord & { config: C | null };

export type ReloadDeps<C = unknown> = {
  readLastLoad: () => Promise<LastLoadRecord | null>;
  resolveConfig: (
    id: string,
    ggufVariant: string | null,
  ) => { config: C } | null;
  loadTarget: (target: ReloadTarget<C>) => Promise<void>;
};

export const RELOAD_MISSING_HISTORY_MESSAGE =
  "No previously loaded model found. Load a model from the picker first.";

export function resolveReloadTarget<C>(
  lastLoad: LastLoadRecord | null,
  resolveConfig: ReloadDeps<C>["resolveConfig"],
): ReloadTarget<C> | null {
  const id = lastLoad?.id.trim();
  if (!(lastLoad && id)) {
    return null;
  }
  // A GGUF record's null variant is a direct .gguf path; a model record's is an MLX precision folder.
  const ggufVariant = lastLoad.ggufVariant?.trim() || null;
  return {
    id,
    kind: lastLoad.kind,
    ggufVariant,
    config: resolveConfig(id, ggufVariant)?.config ?? null,
  };
}

export async function reloadLastModel<C>(
  deps: ReloadDeps<C>,
): Promise<ReloadTarget<C>> {
  const target = resolveReloadTarget(
    await deps.readLastLoad(),
    deps.resolveConfig,
  );
  if (!target) {
    throw new Error(RELOAD_MISSING_HISTORY_MESSAGE);
  }
  await deps.loadTarget(target);
  return target;
}
