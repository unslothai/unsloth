// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GgufShardMode = "single" | "split";

export const GGUF_SHARD_SIZE_PRESETS = [
  "256MB",
  "512MB",
  "1GB",
  "2GB",
  "4GB",
  "8GB",
  "16GB",
] as const;

const GGUF_SHARD_SIZE_RE = /^(\d+)\s*([MG])B?$/i;
const MAX_PATH_COMPONENT_LENGTH = 255;

export function normalizeGgufShardSize(value: string): string | null {
  const match = GGUF_SHARD_SIZE_RE.exec(value.trim());
  if (!match || BigInt(match[1]) === 0n) {
    return null;
  }
  return `${BigInt(match[1])}${match[2].toUpperCase()}B`;
}

export function isValidGgufShardSize(value: string): boolean {
  return normalizeGgufShardSize(value) !== null;
}

export function ggufShardSaveDirectory(
  baseDirectory: string,
  shardSize: string | null,
): string {
  if (!shardSize || shardSize === "0") {
    return baseDirectory;
  }

  const suffix = `-split-${shardSize}`;
  const separator = Math.max(
    baseDirectory.lastIndexOf("/"),
    baseDirectory.lastIndexOf("\\"),
  );
  const parent = baseDirectory.slice(0, separator + 1);
  const name = baseDirectory.slice(separator + 1);
  return `${parent}${name.slice(0, MAX_PATH_COMPONENT_LENGTH - suffix.length)}${suffix}`;
}
