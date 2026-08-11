// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { parseExternalModelId } from "@/features/chat/external-providers";

/** The connection fields the picker carries for every model a connection offers. */
export interface ExternalModelRef {
  id: string;
  providerId: string;
  providerName: string;
  providerType: string;
}

export interface MissingExternalModel {
  /** The model as the user picked it, e.g. `kimi-k2.5`. */
  modelName: string;
  /** The connection's display name, when it still offers at least one other model. */
  providerName: string | null;
  /** Registry key (e.g. openai, ollama), from the same sibling entry. */
  providerType: string | null;
}

/**
 * Describes an `external::<connectionId>::<modelId>` selection that its connection no
 * longer offers, or null for any other selection.
 *
 * Fetch Models replaces a connection's model list wholesale, and nothing reconciles the
 * active pick against it, so a model the provider dropped leaves the picker's option list
 * while its id stays in `params.checkpoint`. The picker's generic fallback cannot shorten
 * that id: `modelDisplayName` is an exact identity function for it, since the id is neither
 * path-shaped nor a `org/name` repo id, so the trigger printed the raw `external::…` string
 * for a model that can no longer be loaded (#8405).
 *
 * Parsing recovers the name the user actually picked, and the connection's display name
 * comes from any sibling model still listed under the same connection id. A connection that
 * dropped every model has no sibling to read, so `providerName` is null and the caller says
 * only that the model is gone.
 */
export function missingExternalModel(
  selected: string | null | undefined,
  externalModels: readonly ExternalModelRef[],
): MissingExternalModel | null {
  const parsed = parseExternalModelId(selected);
  if (!parsed) {
    return null;
  }
  // Still offered: the option carries a real name and never reaches the fallback.
  if (externalModels.some((option) => option.id === selected)) {
    return null;
  }
  const sibling = externalModels.find(
    (option) => option.providerId === parsed.providerId,
  );
  return {
    modelName: parsed.modelId,
    providerName: sibling?.providerName ?? null,
    providerType: sibling?.providerType ?? null,
  };
}
