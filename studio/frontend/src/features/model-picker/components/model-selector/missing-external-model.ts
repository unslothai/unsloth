// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import {
  allowsManualModelIdsWithCatalog,
  parseExternalModelId,
} from "@/features/chat/external-providers";

export interface ExternalModelRef {
  id: string;
  providerId: string;
  providerName: string;
  providerType: string;
}

/** `availableModels` caches the whole catalogue; `models` holds only the ticked subset. */
export interface ExternalConnectionRef {
  id: string;
  name: string;
  providerType?: string;
  availableModels?: readonly string[];
}

/** `disabled` is the user's doing and reversible; `dropped` is the provider's. */
export type MissingExternalModelState = "disabled" | "dropped";

export interface MissingExternalModel {
  modelName: string;
  providerName: string | null;
  providerType: string | null;
  state: MissingExternalModelState;
}

/** Describes an `external::<connectionId>::<modelId>` selection with no option behind it, or null.
  *  `dropped` is claimed only when a non-empty cached catalogue that could list the model omits it. */
export function missingExternalModel(
  selected: string | null | undefined,
  externalModels: readonly ExternalModelRef[],
  connections: readonly ExternalConnectionRef[] = [],
): MissingExternalModel | null {
  const parsed = parseExternalModelId(selected);
  if (!parsed) {
    return null;
  }
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
  // Providers that accept typed-in model IDs never catalogue them, so silence proves nothing.
  const catalogCoversEveryId = !allowsManualModelIdsWithCatalog(
    connection?.providerType,
  );
  // An empty catalogue is unknown, since one is never written with no enabled models.
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
