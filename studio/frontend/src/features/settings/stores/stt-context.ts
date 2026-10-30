// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MTMD_STT_MODELS } from "./stt-model-catalog.ts";

/** Zero lets the backend fit the context to each recording. */
export const DEFAULT_STT_CONTEXT_LENGTH = 0;
export const STT_CONTEXT_LENGTHS = [
  1024, 2048, 4096, 8192, 16384, 32768, 65536,
] as const;

export function normalizeSttContextLength(value: unknown): number {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    (value === 0 || (value >= 1024 && value <= 65536))
    ? value
    : DEFAULT_STT_CONTEXT_LENGTH;
}

/** Other engines and custom models have different memory controls. */
export function sttContextBody(
  model: string,
  engine: string,
  contextLength: unknown,
): { context_length?: number } {
  return engine === "mtmd" && MTMD_STT_MODELS.has(model.trim())
    ? { context_length: normalizeSttContextLength(contextLength) }
    : {};
}
