// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type CacheCopy = {
  repo_id: string;
  load_id?: string | null;
  cache_path?: string | null;
  active_cache?: boolean | null;
};
type Variant = {
  quant: string;
  filename: string;
  downloaded?: boolean;
  partial?: boolean;
};
export type ResolvedPinnedQuant = {
  repoId: string;
  quant: string;
  loadId?: string;
  cachePath?: string;
  filename: string;
};

export async function missingPinnedQuants(
  pins: readonly { repoId: string; quant: string }[],
  copies: readonly CacheCopy[],
  read: (copy: CacheCopy) => Promise<readonly Variant[]>,
): Promise<readonly { repoId: string; quant: string }[]> {
  // A failed scan is not evidence that the last copy has been deleted.
  const relevant = copies.filter((copy) =>
    pins.some((pin) => pin.repoId === copy.repo_id),
  );
  const variants = await Promise.all(relevant.map(read));
  return pins.filter(
    (pin) =>
      !relevant.some(
        (copy, index) =>
          copy.repo_id === pin.repoId &&
          variants[index].some(
            (variant) =>
              variant.quant === pin.quant &&
              variant.downloaded &&
              !variant.partial,
          ),
      ),
  );
}

export async function resolvePinnedQuantSources(
  pins: readonly { repoId: string; quant: string }[],
  copies: readonly CacheCopy[],
  read: (copy: CacheCopy) => Promise<readonly Variant[]>,
): Promise<ResolvedPinnedQuant[]> {
  const wanted = new Set(pins.map((pin) => pin.repoId));
  const candidates = copies
    .filter((copy) => wanted.has(copy.repo_id))
    .sort(
      (a, b) =>
        Number(b.active_cache === true) - Number(a.active_cache === true),
    );
  const lists = await Promise.all(
    candidates.map(async (copy) => {
      try {
        return await read(copy);
      } catch {
        return [];
      }
    }),
  );
  return pins.flatMap((pin) => {
    for (const [index, copy] of candidates.entries()) {
      if (copy.repo_id !== pin.repoId) continue;
      const variant = lists[index].find(
        (v) => v.quant === pin.quant && v.downloaded && !v.partial,
      );
      if (variant)
        return [
          {
            ...pin,
            loadId: copy.load_id || undefined,
            cachePath: copy.cache_path || undefined,
            filename: variant.filename,
          },
        ];
    }
    return [];
  });
}
