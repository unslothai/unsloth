// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import {
  allowsManualModelIdsWithCatalog,
  parseExternalModelId,
} from "@/features/chat/external-providers";

/** The connection fields the picker carries for every model a connection offers. */
export interface ExternalModelRef {
  id: string;
  providerId: string;
  providerName: string;
  providerType: string;
}

/** The connection-level fields this helper needs beyond the enabled options: a connection caches
 *  the provider's whole catalogue in `availableModels`, while `models` holds only the subset
 *  the user ticked, and `externalModels` is built from `models` alone. */
export interface ExternalConnectionRef {
  id: string;
  name: string;
  providerType?: string;
  /** Cached catalogue from the provider's /models response, when one was ever stored. */
  availableModels?: readonly string[];
}

/** Why the selection has no option behind it. `disabled` is the user's own doing and reversible
 *  from the connection dialog; `dropped` is the provider's, and no re-ticking brings it back. */
export type MissingExternalModelState = "disabled" | "dropped";

export interface MissingExternalModel {
  /** The model as the user picked it, e.g. `kimi-k2.5`. */
  modelName: string;
  /** The connection's display name, when one can be read. */
  providerName: string | null;
  /** Registry key (e.g. openai, ollama), from the same source as the name. */
  providerType: string | null;
  /** Which of the two ways the option went away. */
  state: MissingExternalModelState;
}

/** Describes an `external::<connectionId>::<modelId>` selection with no option behind it, or null
 *  for any other selection. Fetch Models replaces a connection's model list wholesale and
 *  nothing reconciles the active pick, so a dropped model leaves the option list while its id
 *  stays in `params.checkpoint`, and the generic fallback cannot shorten that id, so the trigger
 *  printed the raw `external::` string (#8405). Parsing recovers the name the user picked.
 *  An option can go away two ways and only one is the provider's doing: unticking drops the id
 *  from `models` while it stays in `availableModels`, so the enabled list alone cannot tell
 *  them apart. `dropped` is therefore claimed only on positive evidence: a catalogue that was
 *  cached, is non-empty, does not list the model, and could have listed it. That last clause
 *  rules out connections whose dialog offers a manual model-ID box. Anything less is reported
 *  as `disabled`, which claims nothing about the provider. */
export function missingExternalModel(
  selected: string | null | undefined,
  externalModels: readonly ExternalModelRef[],
  connections: readonly ExternalConnectionRef[] = [],
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
  const connection = connections.find(
    (entry) => entry.id === parsed.providerId,
  );
  const catalog = connection?.availableModels;
  // Ollama, vLLM, llama.cpp and OpenRouter take typed-in model IDs beside the fetched list, and
  // the dialog saves those to `models` alone, so their catalogue never carried them and its
  // silence about one is not evidence of anything.
  const catalogCoversEveryId = !allowsManualModelIdsWithCatalog(
    connection?.providerType,
  );
  // `modelId` is decoded and `availableModels` holds raw provider ids, so these compare directly;
  // an empty catalogue is unknown rather than empty, since a connection with no enabled models
  // never gets one written.
  const dropped =
    connection == null ||
    (catalogCoversEveryId &&
      catalog != null &&
      catalog.length > 0 &&
      !catalog.includes(parsed.modelId));
  if (dropped) {
    return {
      modelName: parsed.modelId,
      providerName: sibling?.providerName ?? null,
      providerType: sibling?.providerType ?? null,
      state: "dropped",
    };
  }
  return {
    modelName: parsed.modelId,
    providerName: sibling?.providerName ?? connection.name,
    providerType: sibling?.providerType ?? connection.providerType ?? null,
    state: "disabled",
  };
}
