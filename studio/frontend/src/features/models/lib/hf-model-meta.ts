// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const EMBEDDING_TAGS: ReadonlySet<string> = new Set([
  "sentence-transformers",
  "feature-extraction",
]);

export const DTYPE_BYTES: Readonly<Record<string, number>> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  NF4: 0.5,
  FP4: 0.5,
  INT4: 0.5,
  GPTQ: 0.5,
};

export function estimateSizeFromDtypes(
  params: Record<string, number> | undefined,
): number | undefined {
  if (!params) return undefined;
  let total = 0;
  for (const [dtype, count] of Object.entries(params)) {
    const bytesPerParam = DTYPE_BYTES[dtype.toUpperCase()] ?? 2;
    total += count * bytesPerParam;
  }
  return total > 0 ? total : undefined;
}

const GGUF_PATTERN = /-GGUF(?:$|-)/i;

export function isGgufLike(value: string | undefined | null): boolean {
  if (!value) return false;
  return value.toLowerCase().endsWith(".gguf") || GGUF_PATTERN.test(value);
}
