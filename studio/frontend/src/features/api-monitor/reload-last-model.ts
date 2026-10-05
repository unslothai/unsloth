// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads the durable last-local-load record, not the monitor ring buffer: that one is bounded,
// clearable and can be disabled.

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
  // A null variant on a GGUF record is a direct .gguf file path, which the loader accepts as is.
  const ggufVariant =
    lastLoad.kind === "gguf" ? lastLoad.ggufVariant?.trim() || null : null;
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

// The monitor reports a llama.cpp model as "<id>:<quant>".
export function splitActiveModel(activeModel: string | null | undefined): {
  id: string;
  ggufVariant: string | null;
} {
  if (!activeModel) {
    return { id: "", ggufVariant: null };
  }
  const sep = activeModel.lastIndexOf(":");
  const variant = activeModel.slice(sep + 1);
  if (sep <= 0 || !variant || variant.includes("/")) {
    return { id: activeModel, ggufVariant: null };
  }
  return { id: activeModel.slice(0, sep), ggufVariant: variant };
}
