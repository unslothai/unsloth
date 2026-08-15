// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the hub barrel's React and download-manager exports.
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "@/features/hub/lib/model-identity";

/** The names one model pick answers to: the catalog id its picker row shows, and the
 * `model_path` its load sends, which differ whenever a cached row pins a snapshot dir. */
export type ModelPickNames = {
  id: string;
  loadPath?: string | null;
  ggufVariant?: string | null;
};

/** The resident model as `/api/inference/status` reports it. */
export type ResidentModelStatus = {
  active_model?: string | null;
  model_identifier?: string | null;
  gguf_variant?: string | null;
};

/**
 * Whether the model already resident on the inference server is the one this pick names.
 *
 * Both sides hold two strings for one model and they need not be the same one: a pinned
 * cached row loads by snapshot path while its picker row keeps the repo id, and the status
 * publishes the clean public id next to the raw load identifier. Loading a model that is
 * already resident returns `already_loaded` without touching llama-server, so a caller that
 * recognises it here can skip both the reload and the confirmation that goes with it.
 */
export function residentModelMatchesPick(
  status: ResidentModelStatus,
  pick: ModelPickNames,
): boolean {
  if (!status.active_model) {
    return false;
  }
  // a quant switch within one repo is a real reload, so the variant has to agree too
  if (
    normalizeGgufVariantIdentity(status.gguf_variant) !==
    normalizeGgufVariantIdentity(pick.ggufVariant)
  ) {
    return false;
  }
  const residentNames = new Set(
    [status.model_identifier, status.active_model]
      .filter((name): name is string => Boolean(name))
      .map(normalizeModelIdentity),
  );
  for (const name of [pick.id, pick.loadPath]) {
    if (name && residentNames.has(normalizeModelIdentity(name))) {
      return true;
    }
  }
  return false;
}
