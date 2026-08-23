// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cachePathKey } from "@/lib/cache-path-key";
import { listGgufVariants } from "../inventory/api";

export async function downloadedGgufQuantsInCacheCopies(
  repoId: string,
  cacheCopies: readonly { cachePath: string }[],
  hfToken?: string,
): Promise<Set<string> | null> {
  const pathsByKey = new Map<string, string>();
  for (const copy of cacheCopies) {
    const path = copy.cachePath.trim();
    if (path) {
      pathsByKey.set(cachePathKey(path), path);
    }
  }
  const settled = await Promise.allSettled(
    [...pathsByKey.values()].map((localPath) =>
      listGgufVariants(repoId, hfToken, {
        preferLocalCache: true,
        offline: true,
        localPath,
      }),
    ),
  );
  if (settled.some((result) => result.status === "rejected")) {
    return null;
  }
  return new Set(
    settled.flatMap((result) =>
      result.status === "fulfilled"
        ? result.value.variants
            .filter((variant) => variant.downloaded === true)
            .map((variant) => variant.quant.trim().toLowerCase())
        : [],
    ),
  );
}

export async function remainingDownloadedGgufQuants(
  repoId: string,
  selectedCachePath: string | null | undefined,
  cacheCopies: readonly { cachePath: string }[],
  hfToken?: string,
): Promise<Set<string> | null> {
  const pathsByKey = new Map<string, string>();
  for (const copy of cacheCopies) {
    const path = copy.cachePath.trim();
    if (path) {
      pathsByKey.set(cachePathKey(path), path);
    }
  }
  if (!selectedCachePath) {
    return pathsByKey.size === 1 ? new Set() : null;
  }
  if (pathsByKey.size === 0) {
    return null;
  }
  const selected = cachePathKey(selectedCachePath);
  const otherPathsByKey = new Map<string, string>();
  for (const [candidate, path] of pathsByKey) {
    if (candidate === selected || selected.startsWith(`${candidate}/`)) {
      continue;
    }
    otherPathsByKey.set(candidate, path);
  }
  const otherPaths = [...otherPathsByKey.values()];
  if (otherPaths.length === 0) {
    return new Set();
  }
  return downloadedGgufQuantsInCacheCopies(
    repoId,
    otherPaths.map((cachePath) => ({ cachePath })),
    hfToken,
  );
}
