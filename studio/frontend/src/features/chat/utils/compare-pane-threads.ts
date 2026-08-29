// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType, ThreadRecord } from "../types";

/**
 * Either compare component can be handed the pair the other saved (#9823), hence
 * the alias. Two lookups, not one predicate over both: `listStoredChatThreads`
 * sorts updatedAt-descending, so a single find returns the freshest row.
 */
export function resolveComparePaneThreadId(
  threads: ThreadRecord[],
  native: ModelType,
  alias: ModelType,
): string | undefined {
  return (
    threads.find((t) => t.modelType === native)?.id ??
    threads.find((t) => t.modelType === alias)?.id
  );
}
