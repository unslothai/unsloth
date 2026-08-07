// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { toast } from "@/lib/toast";

import { DOWNLOAD_KIND } from "./constants";
import { downloadManager } from "./download-manager-controller";
import { scopedVariant } from "./download-manager-types";
import { useRepoDownload } from "./use-repo-download";

/** One repo of a staged plan: the exact files to fetch and their declared size. */
export interface StagedDownloadEntry {
  repoId: string;
  files: string[];
  bytes: number;
  /** Set when this entry is a single-file GGUF checkpoint. Informational: it is fetched as a scoped job like every other entry. */
  ggufFilename?: string | null;
}

function entryKey(entry: StagedDownloadEntry): string {
  return `${entry.repoId}|${[...entry.files].sort().join(",")}`;
}

/** Runs a multi-repo download plan through the shared download manager, then calls `onReady` once every entry is on disk. Chat stages a single repo inline; the diffusion pages need two (a GGUF checkpoint plus its companion base) and read only part of each, so the entries go out as scoped jobs. Staging here rather than inside the load is what puts image and video downloads in the same panel, with the same progress, cancel, resume, disk preflight and manifest verification. */
export function useStagedDownload({
  scopeId,
  onReady,
  onCancelled,
}: {
  /** Scope label for entries that fetch a file subset (e.g. "diffusion"). */
  scopeId: string;
  onReady: () => void;
  /** Clears the consumer's pending auto-load whenever the plan ends without every entry on disk:
   * cancelled, failed, or never started. A pick is only an intent until then, and leaving it
   * behind lets a later completion or a deferred page activation load a model nobody asked for. */
  onCancelled?: () => void;
}) {
  const [queue, setQueue] = useState<StagedDownloadEntry[] | null>(null);
  const current = queue?.[0] ?? null;

  // Every entry is scoped, including a GGUF checkpoint: the Hub's snapshot ignore list drops *.gguf, so a plain snapshot job
  // would finish at once having fetched everything EXCEPT the weights, leaving the repo on device unloadable.
  const activeVariant = current ? scopedVariant(scopeId) : null;

  const advance = useCallback(() => {
    setQueue((rest) => {
      const remaining = (rest ?? []).slice(1);
      if (remaining.length > 0) return remaining;
      return null;
    });
  }, []);

  // The job this hook waits on, keyed by the entry it started (repo + exact file set) AND the staging generation: every scoped pick in a repo
  // shares the "@diffusion" variant, so restaging while the first job finishes would let its completion pass for the new pick.
  const inFlight = useRef<{ key: string; generation: number } | null>(null);
  const generation = useRef(0);
  const isOurs = (variant: string | null | undefined) =>
    (variant ?? null) === activeVariant &&
    current !== null &&
    inFlight.current !== null &&
    inFlight.current.key === entryKey(current) &&
    inFlight.current.generation === generation.current;

  useRepoDownload({
    kind: DOWNLOAD_KIND.MODEL,
    repoId: current?.repoId ?? "__staged_download_idle__",
    activeVariant,
    // The listener subscription is per REPO, not per job, and one repo can have several jobs in flight. Each callback carries the variant it fired
    // for, so drop the ones that are not this staged entry: a sibling's completion would advance the queue and its failure would wipe a live job.
    onComplete: (variant) => {
      if (!isOurs(variant)) return;
      inFlight.current = null;
      const remaining = (queue ?? []).slice(1);
      advance();
      // Every entry is on disk, so the load will find its cache warm.
      if (remaining.length === 0) onReady();
    },
    onError: (variant) => {
      if (!isOurs(variant)) return;
      inFlight.current = null;
      setQueue(null);
      onCancelled?.();
    },
    onCancelled: (variant) => {
      if (!isOurs(variant)) return;
      inFlight.current = null;
      setQueue(null);
      onCancelled?.();
    },
  });

  // Read through a ref so a consumer's inline callback cannot re-run the start effect and
  // restart a live download.
  const onCancelledRef = useRef(onCancelled);
  onCancelledRef.current = onCancelled;
  useEffect(() => {
    if (!current) return;
    let active = true;
    const started = { key: entryKey(current), generation: generation.current };
    // Register ownership before the start request. The global panel can expose the job as soon as
    // the controller updates its store, before this await resumes; a very fast cancel/completion in
    // that window must still belong to this plan.
    inFlight.current = started;
    void (async () => {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: current.repoId,
        variant: activeVariant,
        expectedBytes: current.bytes,
        scopeId,
        files: current.files,
      });
      if (!active) return;
      if (outcome === "started") return;
      if (inFlight.current === started) inFlight.current = null;
      // A start that never got off the ground (network failure, rejected scoped request, worker refused) will never complete, so
      // clear the queue instead of leaving the head in place, where the effect never re-runs and onReady never fires.
      // The pick dies with it, so the consumer's pending auto-load has to go too.
      if (outcome === "error") {
        toast.error("Could not start the download", {
          description: "Check the connection, then select the model again.",
        });
      } else if (outcome === "conflict") {
        toast.info("Resume this download from Models", {
          description:
            "An earlier partial download used a different transport. Open the Model hub tab to resume or restart it.",
        });
      } else if (outcome === "busy") {
        toast.info("Download already in progress", {
          description:
            "Reselect this model once the running download finishes to load it.",
        });
      }
      setQueue(null);
      onCancelledRef.current?.();
    })();
    return () => {
      active = false;
    };
    // Only the head of the queue drives a start; advancing re-runs this with the next one.
  }, [current, activeVariant, scopeId]);

  const stage = useCallback((entries: StagedDownloadEntry[]) => {
    // A fresh plan supersedes whatever was staged, so bump the generation: a callback for the previous plan's job is no longer ours.
    generation.current += 1;
    inFlight.current = null;
    // The Downloads panel is the sole download surface. A second toast duplicated its
    // progress and exposed another X that looked like cancellation but only dismissed copy.
    // `onReady` owns the later GPU-load toast, after every queued entry is complete.
    setQueue(entries.length > 0 ? entries : null);
  }, []);

  return { stage, staging: queue !== null };
}
