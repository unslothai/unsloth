// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type Listener = () => void;

const listeners = new Set<Listener>();
const repoVersions = new Map<string, number>();
const REPO_VERSION_LIMIT = 256;
let globalVersion = 0;
let versionSequence = 0;

function repoKey(repoId: string): string {
  return repoId.trim().toLowerCase();
}

function emit(): void {
  for (const listener of [...listeners]) listener();
}

function nextVersion(): number {
  versionSequence += 1;
  return versionSequence;
}

export function bumpGgufVariantsCacheVersion(repoId?: string): void {
  const key = repoId ? repoKey(repoId) : "";
  if (key) {
    repoVersions.delete(key);
    repoVersions.set(key, nextVersion());
    while (repoVersions.size > REPO_VERSION_LIMIT) {
      const oldest = repoVersions.keys().next().value;
      if (!oldest) break;
      repoVersions.delete(oldest);
    }
  } else {
    globalVersion = nextVersion();
  }
  emit();
}

export function getGgufVariantsCacheVersion(repoId?: string | null): string {
  const key = repoId ? repoKey(repoId) : "";
  return `${globalVersion}:${key ? (repoVersions.get(key) ?? 0) : 0}`;
}

export function subscribeGgufVariantsCache(
  listener: Listener,
): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
