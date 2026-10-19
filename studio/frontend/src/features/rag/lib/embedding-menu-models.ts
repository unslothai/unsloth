// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Current, default, then pinned models, each once. */
export function embeddingMenuModels(
  current: string,
  defaultModel: string,
  pinned: readonly string[],
): string[] {
  const models: string[] = [];
  for (const model of [current, defaultModel, ...pinned]) {
    const id = model.trim();
    if (id && !models.includes(id)) models.push(id);
  }
  return models;
}
