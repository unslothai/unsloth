// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface SoleQuantTarget {
  repoId: string;
  localSource: string | null;
  includeCacheLocations?: boolean;
  fingerprint: string;
  key: string;
}

/** A null quant (including a failed read) keeps the row expandable. */
export interface SoleQuantEntry<T> {
  key: string;
  quant: T | null;
}

/** Collapse only after a Hub-aware response verified every dependency. Counts quants on disk,
 *  not listed ones: a Hub answer lists every published quant. */
export function verifiedSoleHubVariant<
  T extends {
    downloaded?: boolean;
    partial?: boolean;
    update_available?: boolean;
    pending_drafter_filename?: string | null;
  },
>(
  variants: readonly T[],
  resolvedLocally: boolean,
  dependenciesResolved: boolean,
): T | null {
  if (resolvedLocally || !dependenciesResolved) return null;
  // A torn quant keeps the expander, where resume lives.
  if (variants.some((v) => v.partial === true)) return null;
  const downloaded = variants.filter((v) => v.downloaded === true);
  if (downloaded.length !== 1) return null;
  const sole = downloaded[0];
  if (sole.pending_drafter_filename) return null;
  // Only the expander carries the update action.
  if (sole.update_available === true) return null;
  return sole;
}

export function soleQuantKey(
  version: string | undefined,
  localSource: string | null,
  fingerprint = "",
): string {
  return `${version ?? ""}::${localSource ?? ""}::${fingerprint}`;
}

/** Includes download state: a sibling cancelled before any file landed moves neither bytes nor mtime. */
export function soleQuantFingerprint(repo: {
  size_bytes?: number;
  last_modified?: number;
  has_variant_state?: boolean;
}): string {
  return `${repo.size_bytes ?? ""}:${repo.last_modified ?? ""}:${
    repo.has_variant_state ? "state" : ""
  }`;
}

export function partitionSoleQuants<T>(
  targets: readonly SoleQuantTarget[],
  entries: ReadonlyMap<string, SoleQuantEntry<T>>,
  { enabled }: { enabled: boolean },
): {
  quants: ReadonlyMap<string, T>;
  pending: ReadonlySet<string>;
  stale: SoleQuantTarget[];
} {
  const quants = new Map<string, T>();
  const pending = new Set<string>();
  const stale: SoleQuantTarget[] = [];
  if (!enabled) return { quants, pending, stale };
  for (const target of targets) {
    const entry = entries.get(target.repoId);
    if (!entry || entry.key !== target.key) {
      pending.add(target.repoId);
      stale.push(target);
      continue;
    }
    if (entry.quant) quants.set(target.repoId, entry.quant);
  }
  return { quants, pending, stale };
}

/** Each repo is read once per key; a read whose repo moved on is dropped, not committed. */
export function createSoleQuantReader<T>({
  workers,
  read,
  commit,
}: {
  workers: number;
  read: (target: SoleQuantTarget) => Promise<T | null>;
  commit: (target: SoleQuantTarget, quant: T | null) => void;
}): { start: (targets: readonly SoleQuantTarget[]) => void } {
  const inFlight = new Map<string, string>();
  const queue: SoleQuantTarget[] = [];
  let active = 0;

  const owns = (target: SoleQuantTarget) =>
    inFlight.get(target.repoId) === target.key;

  const drain = async () => {
    while (queue.length > 0) {
      const target = queue.shift();
      if (!(target && owns(target))) continue;
      const quant = await read(target).catch(() => null);
      if (!owns(target)) continue;
      inFlight.delete(target.repoId);
      commit(target, quant);
    }
    active -= 1;
  };

  return {
    start(targets) {
      for (const target of targets) {
        if (owns(target)) continue;
        inFlight.set(target.repoId, target.key);
        queue.push(target);
      }
      while (active < workers && queue.length > 0) {
        active += 1;
        void drain();
      }
    },
  };
}

/** Compares fingerprints, never keys: dropping a listing bumps the key's cache version. */
export function takeDriftedRepos(
  targets: readonly SoleQuantTarget[],
  seen: Map<string, string>,
): string[] {
  const drifted: string[] = [];
  for (const target of targets) {
    const previous = seen.get(target.repoId);
    seen.set(target.repoId, target.fingerprint);
    if (previous !== undefined && previous !== target.fingerprint) {
      drifted.push(target.repoId);
    }
  }
  return drifted;
}
