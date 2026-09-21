// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { toast } from "@/lib/toast";

import { DOWNLOAD_KIND } from "./constants";
import { downloadManager } from "./download-manager-controller";
import {
  scopedDownloadInventoryKind,
  scopedVariant,
} from "./download-manager-types";
import { useRepoDownload } from "./use-repo-download";

export interface StagedDownloadEntry {
  repoId: string;
  files: string[];
  bytes: number;
  ggufFilename?: string | null;
  checkpoint?: boolean;
}

function entryKey(entry: StagedDownloadEntry): string {
  return `${entry.repoId}|${[...entry.files].sort().join(",")}`;
}

/** Runs a multi-repo download plan through the shared download manager, then calls `onReady` once every entry is on disk. Staging here rather than inside the load is what puts image and video downloads in the same panel, with the same progress, cancel, resume, disk preflight and manifest verification. */
export function useStagedDownload({
  scopeId,
  onReady,
  onCancelled,
}: {
  scopeId: string;
  onReady: () => void;
  /** Clears the consumer's pending auto-load when the plan ends without every entry on disk: leaving it behind lets a later completion load a model nobody asked for. */
  onCancelled?: () => void;
}) {
  const [queue, setQueue] = useState<StagedDownloadEntry[] | null>(null);
  const current = queue?.[0] ?? null;

  // Every entry is scoped, including a GGUF checkpoint: the Hub's snapshot ignore list drops *.gguf, so a plain snapshot job would finish having fetched everything EXCEPT the weights.
  const activeVariant = current ? scopedVariant(scopeId) : null;

  const advance = useCallback(() => {
    setQueue((rest) => {
      const remaining = (rest ?? []).slice(1);
      if (remaining.length > 0) return remaining;
      return null;
    });
  }, []);

  // Keyed by entry AND staging generation: every scoped pick in a repo shares the "@scope" variant, so restaging would let the first job's completion pass for the new pick.
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
    // The listener subscription is per REPO and each callback carries its variant, so drop the ones that are not this entry: a sibling's completion would advance the queue.
    onComplete: (variant) => {
      if (!isOurs(variant)) return;
      inFlight.current = null;
      const remaining = (queue ?? []).slice(1);
      advance();
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

  const onCancelledRef = useRef(onCancelled);
  onCancelledRef.current = onCancelled;
  useEffect(() => {
    if (!current) return;
    let active = true;
    const started = { key: entryKey(current), generation: generation.current };
    // Register ownership before the start request: the panel can expose the job before this await resumes, and a very fast cancel in that window must still belong to this plan.
    inFlight.current = started;
    void (async () => {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: current.repoId,
        variant: activeVariant,
        inventoryKind: scopedDownloadInventoryKind(current.files),
        expectedBytes: current.bytes,
        scopeId,
        files: current.files,
        checkpoint: current.checkpoint,
      });
      if (!active) return;
      if (outcome === "started") return;
      if (inFlight.current === started) inFlight.current = null;
      // A start that never got off the ground will never complete, so clear the queue instead of leaving the head in place where the effect never re-runs. The pick dies with it.
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
  }, [current, activeVariant, scopeId]);

  const stage = useCallback((entries: StagedDownloadEntry[]) => {
    generation.current += 1;
    inFlight.current = null;
    setQueue(entries.length > 0 ? entries : null);
  }, []);

  return { stage, staging: queue !== null };
}
