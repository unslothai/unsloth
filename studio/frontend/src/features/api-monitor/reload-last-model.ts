// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reload helper for the API monitor's "Reload previous" action (issue #11189).
// Plain module (no React/router/icons) so the node --test suite can drive it,
// mirroring unload-resident.ts / clear-monitor.ts.
//
// The durable source is the last-local-load record written on every successful
// Chat load (features/chat/utils/last-local-model-load.ts). The monitor ring
// buffer (entries[].model) is deliberately NOT used: it is bounded, clearable
// and can be disabled via UNSLOTH_STUDIO_DISABLE_API_MONITOR, so it cannot be
// the sole source for a recovery action.

export type ReloadableKind = "gguf" | "model";

export type LastLoadRecord = {
  id: string;
  kind: ReloadableKind;
  ggufVariant: string | null;
};

export type ReloadTarget = {
  /** model_path to send to the Chat runtime's selectModel. */
  id: string;
  kind: ReloadableKind;
  ggufVariant: string | null;
  /** Remembered per-model config (context, GPU, KV, spec, ...), if any. */
  config: unknown | null;
  /** Always true: a crash/unload leaves active_model null, and a stale ready
   *  state still needs a real reload to recover the worker. */
  forceReload: true;
  /** True when the Chat tab currently holds an external-provider selection.
   *  The reload must NOT clear it as a side effect; selectModel owns the
   *  switch behind its stop-running-chats confirm. */
  externalPreserved: boolean;
};

export type ReloadDeps = {
  /** Durable last-local-load record, or null when never loaded / cleared. */
  readLastLoad: () => LastLoadRecord | null | Promise<LastLoadRecord | null>;
  /** Current Chat runtime selection (params.checkpoint), may be "". */
  readSelectedCheckpoint: () => string;
  /** True when the given checkpoint id is an external-provider selection. */
  isExternalSelection: (checkpoint: string) => boolean;
  /** Remembered per-model config for (id, variant), or null for defaults.
   *  Injected so this module stays free of the model-picker import graph. */
  resolveConfig: (
    id: string,
    ggufVariant: string | null,
  ) => { config: unknown } | null;
  /** Performs the actual load via the shared Chat runtime (selectModel).
   *  It already gates on in-flight loads and active generations
   *  (confirmStopRunningChatsIfNeeded), so this helper must not duplicate
   *  those dialogs or construct a partial LoadModelRequest itself. */
  loadTarget: (target: ReloadTarget) => Promise<void>;
};

export const RELOAD_MISSING_HISTORY_MESSAGE =
  "No previously loaded model found. Load a model from the picker first.";

function normalizedId(id: string): string | null {
  const trimmed = id.trim();
  return trimmed ? trimmed : null;
}

/** Pure resolution: which model would a reload load, and why. */
export function resolveReloadTarget(
  lastLoad: LastLoadRecord | null,
  selectedCheckpoint: string,
  isExternalSelection: (checkpoint: string) => boolean,
  resolveConfig: ReloadDeps["resolveConfig"],
): { ok: true; target: ReloadTarget } | { ok: false; reason: string } {
  if (!lastLoad) {
    return { ok: false, reason: RELOAD_MISSING_HISTORY_MESSAGE };
  }
  const id = normalizedId(lastLoad.id);
  if (!id) {
    return { ok: false, reason: RELOAD_MISSING_HISTORY_MESSAGE };
  }
  if (lastLoad.kind === "gguf") {
    const variant = lastLoad.ggufVariant?.trim() || null;
    if (!variant) {
      return { ok: false, reason: RELOAD_MISSING_HISTORY_MESSAGE };
    }
    const resolved = resolveConfig(id, variant);
    return {
      ok: true,
      target: {
        id,
        kind: "gguf",
        ggufVariant: variant,
        config: resolved?.config ?? null,
        forceReload: true,
        externalPreserved:
          !!selectedCheckpoint && isExternalSelection(selectedCheckpoint),
      },
    };
  }
  if (lastLoad.kind !== "model") {
    return { ok: false, reason: RELOAD_MISSING_HISTORY_MESSAGE };
  }
  const resolved = resolveConfig(id, null);
  return {
    ok: true,
    target: {
      id,
      kind: "model",
      ggufVariant: null,
      config: resolved?.config ?? null,
      forceReload: true,
      externalPreserved:
        !!selectedCheckpoint && isExternalSelection(selectedCheckpoint),
    },
  };
}

/** Resolve the durable record then delegate to the shared Chat loader. */
export async function reloadLastModel(deps: ReloadDeps): Promise<ReloadTarget> {
  const lastLoad = await deps.readLastLoad();
  const selected = deps.readSelectedCheckpoint();
  const resolved = resolveReloadTarget(
    lastLoad,
    selected,
    deps.isExternalSelection,
    deps.resolveConfig,
  );
  if (!resolved.ok) {
    throw new Error(resolved.reason);
  }
  await deps.loadTarget(resolved.target);
  return resolved.target;
}
