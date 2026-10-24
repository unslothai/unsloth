// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One model's remembered run settings, edited from outside the Run settings panel.
//
// There is exactly one store: the per-model record Run settings saves with Remember (this
// browser's "unsloth_model_configs") plus its server mirror (the auto-switch override a gateway
// or API load applies). A control elsewhere -- Settings → Agents -- must not keep a copy of its
// own, so it writes through here: the same normalize, the same save, the same mirror call Run
// settings' persistConfig makes, and the same PER_MODEL_CONFIG_UPDATED_EVENT every reader
// (Run settings, the memory bar, Agents) already listens to.

import { syncModelOverride } from "../api/model-overrides";
import { cachedRepoConfigId, isStandaloneGgufPath } from "./model-identity";
import {
  type PerModelConfig,
  deletePerModelConfig,
  isDefaultConfig,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
  storedSpeculativeAuto,
} from "./per-model-config";

export type RememberedSettingsDeps = {
  sync: typeof syncModelOverride;
};

const DEFAULT_DEPS: RememberedSettingsDeps = { sync: syncModelOverride };

export type RememberedSettingsKey = { modelId: string; ggufVariant: string | null };

/** The key Run settings saves a loaded model's settings under, from its load path and quant:
 *  residentModelConfigTarget's rules (a loose .gguf has no variant, a quant-less cached snapshot
 *  is keyed by its repo id). */
export function rememberedSettingsKey(
  loadPath: string,
  ggufVariant: string | null | undefined,
): RememberedSettingsKey {
  const variant = isStandaloneGgufPath(loadPath) ? null : (ggufVariant ?? null);
  return {
    modelId: cachedRepoConfigId(loadPath, variant) ?? loadPath,
    ggufVariant: variant,
  };
}

/** The remembered settings Run settings would show for this model, or null when nothing is
 *  remembered in this browser. `aliasId` is the catalog spelling a picker record may use. */
export function readRememberedModelSettings(
  modelId: string,
  ggufVariant: string | null | undefined,
  aliasId?: string | null,
): PerModelConfig | null {
  const resolved = resolveInitialConfig(modelId, ggufVariant);
  if (resolved.remembered) {
    return resolved.config;
  }
  if (aliasId && aliasId !== modelId) {
    const alias = resolveInitialConfig(aliasId, ggufVariant);
    if (alias.remembered) {
      return alias.config;
    }
  }
  return null;
}

/** Apply `patch` to the model's remembered settings and mirror them to the server, exactly as
 *  Run settings saves with Remember ticked. A result equal to the defaults forgets the record
 *  instead (Run settings' "Default settings kept"), leaving launch flags only the server holds.
 *  Returns false when the browser refused the write; nothing is mirrored then. */
export function rememberModelSettingsPatch(
  modelId: string,
  ggufVariant: string | null | undefined,
  patch: Partial<PerModelConfig>,
  deps: RememberedSettingsDeps = DEFAULT_DEPS,
  /** The record the change applies to when it was read under another spelling (the alias). */
  base?: PerModelConfig | null,
): boolean {
  const own = resolveInitialConfig(modelId, ggufVariant);
  const current = own.remembered ? own.config : (base ?? own.config);
  // GGUF only: an MLX model has no Agents controls (the gateways load GGUF).
  const next = normalizePerModelConfig(
    storedSpeculativeAuto({ ...current, ...patch }, false),
  );
  const evicted: { modelId: string; ggufVariant: string | null }[] = [];
  const forget = isDefaultConfig(next);
  const saved = forget
    ? deletePerModelConfig(modelId, ggufVariant ?? null)
    : savePerModelConfig(modelId, ggufVariant, next, evicted);
  if (!saved) {
    return false;
  }
  deps.sync(
    modelId,
    ggufVariant,
    forget ? null : next,
    forget ? { keepLaunchFlags: true } : undefined,
  );
  for (const dropped of evicted) {
    deps.sync(dropped.modelId, dropped.ggufVariant, null, {
      keepLaunchFlags: true,
    });
  }
  return true;
}
