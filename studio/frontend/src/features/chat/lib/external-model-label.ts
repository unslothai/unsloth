// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "../external-providers";

/** The model id a connection was asked for, e.g. `kimi-k2.5`, or null when *id* is not an
 *  `external::<connectionId>::<modelId>` selection. A connected model has no row in
 *  `store.models`, so every label chain that looks a checkpoint up there falls through to the
 *  checkpoint itself, and `buildExternalModelId` percent-encodes the model id, so the raw string
 *  carries no separator a generic shortener can use (#8405). Put this ahead of the raw id. */
export function externalModelLabel(
  id: string | null | undefined,
): string | null {
  return parseExternalModelId(id)?.modelId ?? null;
}

/** The short label for a compare pane's model id: the trailing segment, as the compare toasts have
 *  always shown for a local model. A compare pane takes its id straight from the picker and the
 *  compare header offers connected models, so `external::abc::openai%2Fgpt-5` reached the plain
 *  `id.split("/").pop()` this replaced and came back whole. Parsing first yields `gpt-5`. */
export function compareModelDisplayName(id: string): string {
  const value = externalModelLabel(id) ?? id;
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}
