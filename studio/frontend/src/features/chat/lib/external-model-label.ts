// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";
import { providerModelDisplayName } from "../model-display-names";
import { useExternalProvidersStore } from "../stores/external-providers-store";

function connectionProviderType(providerId: string): string | undefined {
  return useExternalProvidersStore
    .getState()
    .providers.find((provider) => provider.id === providerId)?.providerType;
}

/** Label of an `external::<conn>::<model>` selection (raw id is percent-encoded), else null. */
export function externalModelLabel(
  id: string | null | undefined,
): string | null {
  const selection = parseExternalModelId(id);
  if (!selection) return null;
  return providerModelDisplayName(
    connectionProviderType(selection.providerId),
    selection.modelId,
  );
}

/** Short label for a compare pane id; parses external ids first so they do not show whole. */
export function compareModelDisplayName(id: string): string {
  const selection = parseExternalModelId(id);
  if (selection) {
    const name = externalModelLabel(id) ?? selection.modelId;
    if (name !== selection.modelId) return name;
  }
  const value = selection?.modelId ?? id;
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}
