// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const GGUF_REPO_SUFFIX_PATTERN = /-GGUF$/i;
const GGUF_TOKEN_PATTERN = /(?:^|[-_.\\/])gguf(?:$|[-_.\\/])/i;

function normalizedLeaf(value: string): string {
  const normalized = value.trim().replace(/[\\/]+$/, "");
  return normalized.split(/[\\/]/).pop() ?? normalized;
}

export function hasGgufRepoSuffix(value: string | undefined | null): boolean {
  if (!value) return false;
  const leaf = normalizedLeaf(value);
  return leaf.length > 0 && GGUF_REPO_SUFFIX_PATTERN.test(leaf);
}

export function isGgufFilename(value: string | undefined | null): boolean {
  if (!value) return false;
  return normalizedLeaf(value).toLowerCase().endsWith(".gguf");
}

export function isGgufLike(value: string | undefined | null): boolean {
  return isGgufFilename(value) || hasGgufRepoSuffix(value);
}

export function hasGgufToken(value: string | undefined | null): boolean {
  if (!value) return false;
  return GGUF_TOKEN_PATTERN.test(value.trim());
}

export function isMmprojFilename(value: string | undefined | null): boolean {
  if (!value) return false;
  const leaf = normalizedLeaf(value).toLowerCase();
  return leaf.endsWith(".gguf") && leaf.includes("mmproj");
}
