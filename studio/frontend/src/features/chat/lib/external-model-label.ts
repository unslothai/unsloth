// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { connectionModelLabel } from "../claude-agent";
import { parseExternalModelId } from "../external-providers";
import { useExternalProvidersStore } from "../stores/external-providers-store";

/** Model label of an `external::<conn>::<model>` selection (raw id is percent-encoded), else null. */
export function externalModelLabel(
  id: string | null | undefined,
): string | null {
  const parsed = parseExternalModelId(id);
  if (!parsed) return null;
  const provider = useExternalProvidersStore
    .getState()
    .providers.find((candidate) => candidate.id === parsed.providerId);
  return connectionModelLabel(provider?.providerType, parsed.modelId);
}

/** Short label for a compare pane id; parses external ids first so they do not show whole. */
export function compareModelDisplayName(id: string): string {
  const value = externalModelLabel(id) ?? id;
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}
