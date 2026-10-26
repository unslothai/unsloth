// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";

/** Model id of an `external::<conn>::<model>` selection (raw id is percent-encoded), else null. */
export function externalModelLabel(
  id: string | null | undefined,
): string | null {
  return parseExternalModelId(id)?.modelId ?? null;
}

/** Short label for a compare pane id; parses external ids first so they do not show whole. */
export function compareModelDisplayName(id: string): string {
  const value = externalModelLabel(id) ?? id;
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}
