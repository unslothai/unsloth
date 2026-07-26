// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

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
  /** Set when this entry is a single-file GGUF checkpoint. Informational: it is fetched
   *  as a scoped job like every other entry. */
  ggufFilename?: string | null;
}

/**
 * Runs a multi-repo download plan through the shared download manager, then calls
 * `onReady` once every entry is on disk.
 *
 * Chat stages a single repo inline in chat-page; the diffusion pages need two (a GGUF
 * checkpoint plus its companion base), and their loader reads only part of each, so the
 * entries go out as scoped jobs. Staging here rather than letting the backend download
 * inside the load is what puts image and video downloads in the same panel, with the same
 * progress, cancel, resume, disk preflight and manifest verification as everything else.
 */
export function useStagedDownload({
  scopeId,
  onReady,
}: {
  /** Scope label for entries that fetch a file subset (e.g. "diffusion"). */
  scopeId: string;
  onReady: () => void;
}) {
  const [queue, setQueue] = useState<StagedDownloadEntry[] | null>(null);
  const current = queue?.[0] ?? null;

  // Every entry is scoped, including a GGUF checkpoint. A plain snapshot job would be the
  // wrong tool for it: the Hub's snapshot ignore list drops *.gguf, so the job would finish
  // at once having fetched everything EXCEPT the weights, and the repo would land on device
  // unloadable.
  const activeVariant = current ? scopedVariant(scopeId) : null;

  const advance = useCallback(() => {
    setQueue((rest) => {
      const remaining = (rest ?? []).slice(1);
      if (remaining.length > 0) return remaining;
      return null;
    });
  }, []);

  useRepoDownload({
    kind: DOWNLOAD_KIND.MODEL,
    repoId: current?.repoId ?? "__staged_download_idle__",
    activeVariant,
    onComplete: () => {
      const remaining = (queue ?? []).slice(1);
      advance();
      // Every entry is on disk, so the load will find its cache warm.
      if (remaining.length === 0) onReady();
    },
    onError: () => setQueue(null),
    onCancelled: () => setQueue(null),
  });

  useEffect(() => {
    if (!current) return;
    let active = true;
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
      if (outcome === "started") {
        toast.info("Downloading model", {
          description: "It'll load automatically once the download finishes.",
        });
        return;
      }
      if (outcome === "conflict") {
        toast.info("Resume this download from Models", {
          description:
            "An earlier partial download used a different transport. Open the Model hub tab to resume or restart it.",
        });
        setQueue(null);
        return;
      }
      if (outcome === "busy") {
        toast.info("Download already in progress", {
          description:
            "Reselect this model once the running download finishes to load it.",
        });
        setQueue(null);
      }
    })();
    return () => {
      active = false;
    };
    // Only the head of the queue drives a start; advancing re-runs this with the next one.
  }, [current, activeVariant, scopeId]);

  const stage = useCallback((entries: StagedDownloadEntry[]) => {
    setQueue(entries.length > 0 ? entries : null);
  }, []);

  return { stage, staging: queue !== null };
}
