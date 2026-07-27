// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const UNTRAINABLE_MODEL_FORMATS = new Set(["gguf", "adapter"]);

export function isUntrainableModelFormat(
  format: string | null | undefined,
): boolean {
  return format != null && UNTRAINABLE_MODEL_FORMATS.has(format);
}
