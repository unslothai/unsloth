// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bookkeeping for the On Device sole-quant probe: what is known, what is being read, and what
// needs reading. Per repo, so one repo's invalidation leaves the rest alone. No React/DOM
// deps so it stays easy to test.

/** A repo to probe, tagged with the cache version and location it reads at, and separately with
 *  the bytes on disk behind it. */
export interface SoleQuantTarget {
  repoId: string;
  localSource: string | null;
  includeCacheLocations?: boolean;
  fingerprint: string;
  key: string;
}

/** A probe result. A null quant means the repo has no single complete quant, including when the
 *  read failed: either way the row stays expandable. */
export interface SoleQuantEntry<T> {
  key: string;
  quant: T | null;
}

/** Collapse a Hub row only after a Hub-aware response verified every dependency.
 *
 *  Counts the quants on disk, not the quants the repo offers. A dependency-aware answer is a
 *  Hub answer, which lists every published quant, so counting listed quants never matched a
 *  repo holding one. The local answer lists only what is on disk but cannot resolve
 *  dependencies, so the two conditions could not hold together. */
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
  // Drafter still missing: not loadable yet, so not collapsible.
  if (sole.pending_drafter_filename) return null;
  // Only the expander carries the update action, so an update must keep it.
  if (sole.update_available === true) return null;
  return sole;
}

/** Identity of one repo's probe. Moves when that repo's variants cache is invalidated, the row
 *  points at another directory, or the bytes on disk change under us. */
export function soleQuantKey(
  version: string | undefined,
  localSource: string | null,
  fingerprint = "",
): string {
  return `${version ?? ""}::${localSource ?? ""}::${fingerprint}`;
}

/** What the inventory reports about a repo's files. A change here means disk moved, including
 *  from outside this tab, so the cached listing is suspect. Download state is part of it: a
 *  sibling cancelled before any file landed moves neither the bytes nor the mtime, and the
 *  repo's own partial flag stays false while another quant is clean. */
export function soleQuantFingerprint(repo: {
  size_bytes?: number;
  last_modified?: number;
  has_variant_state?: boolean;
}): string {
  return `${repo.size_bytes ?? ""}:${repo.last_modified ?? ""}:${
    repo.has_variant_state ? "state" : ""
  }`;
}

/** Split the listed repos into resolved rows, repos still being read, and repos needing a read.
 *  A repo holds its result until its own key moves. */
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

/** Runs the reads. Kept apart from React so the ordering rules below can be exercised directly:
 *  a repo is read once per key, and a read whose repo has since moved on is dropped rather
 *  than committed over the newer result. */
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
      // Superseded before it started, so there is nothing to read.
      if (!(target && owns(target))) continue;
      const quant = await read(target).catch(() => null);
      // Superseded while reading: the newer read owns this repo now.
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

/** Repos whose bytes moved since they were last seen, recording what is seen now. Their cached
 *  listing predates the change, so it should be dropped. Compares the fingerprint alone, never
 *  the probe key: the key also carries the variants cache version, and dropping a listing
 *  bumps that version, so comparing keys would invalidate on its own effect forever. */
export function takeDriftedRepos(
  targets: readonly SoleQuantTarget[],
  seen: Map<string, string>,
): string[] {
  const drifted: string[] = [];
  for (const target of targets) {
    const previous = seen.get(target.repoId);
    seen.set(target.repoId, target.fingerprint);
    // First sight records only: there is no earlier listing to drop.
    if (previous !== undefined && previous !== target.fingerprint) {
      drifted.push(target.repoId);
    }
  }
  return drifted;
}
