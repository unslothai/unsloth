// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType, ThreadRecord } from "../types";

/**
 * One compare pane's thread out of a pair that may hold either persisted shape:
 * the generalized compare stores model1/model2, the LoRA one base/lora, and
 * which component mounts is decided from async global state, so either can be
 * handed the pair the other saved (#9823).
 *
 * Two lookups, native shape first. `listStoredChatThreads` sorts updatedAt
 * descending, so a single find matching both shapes returns whichever row is
 * freshest, which can give the two panes histories from different runs.
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
