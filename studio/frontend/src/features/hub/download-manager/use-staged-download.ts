// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { toast } from "@/lib/toast";

import { DOWNLOAD_KIND } from "./constants";
import { downloadManager } from "./download-manager-controller";
import {
  scopedDownloadInventoryKind,
  scopedVariant,
} from "./download-manager-types";
import { useRepoDownload } from "./use-repo-download";

export interface StagedDownloadProgress {
  downloadedBytes: number;
  totalBytes: number;
  plan: number;
}

export interface StagedDownloadEntry {
  repoId: string;
  files: string[];
  bytes: number;
  ggufFilename?: string | null;
  checkpoint?: boolean;
  /** GGUF quant fetched as the standard variant download; the plan brings companions. `files` unused. */
  ggufVariant?: string | null;
}

function entryKey(entry: StagedDownloadEntry): string {
  return entry.ggufVariant
    ? `${entry.repoId}|${entry.ggufVariant}`
    : `${entry.repoId}|${[...entry.files].sort().join(",")}`;
}

/** Runs a multi-repo plan through the shared download manager, then calls `onReady`. */
export function useStagedDownload({
  scopeId,
  onReady,
  onCancelled,
}: {
  scopeId: string;
  onReady: () => void;
  /** Clears a pending auto-load when the plan ends incomplete, so a later completion cannot load it. */
  onCancelled?: () => void;
}) {
  const [queue, setQueue] = useState<StagedDownloadEntry[] | null>(null);
  const [staged, setStaged] = useState({ bytes: 0, plan: 0 });
  const current = queue?.[0] ?? null;

  // Every other entry is scoped: the Hub snapshot ignore list drops *.gguf.
  const activeVariant = current
    ? (current.ggufVariant ?? scopedVariant(scopeId))
    : null;

  const advance = useCallback(() => {
    setQueue((rest) => {
      const remaining = (rest ?? []).slice(1);
      if (remaining.length > 0) return remaining;
      return null;
    });
  }, []);

  // Keyed by entry and generation: scoped picks share the "@scope" variant.
  const inFlight = useRef<{ key: string; generation: number } | null>(null);
  const generation = useRef(0);
  const noticedGeneration = useRef<number | null>(null);
  const isOurs = (variant: string | null | undefined) =>
    (variant ?? null) === activeVariant &&
    current !== null &&
    inFlight.current !== null &&
    inFlight.current.key === entryKey(current) &&
    inFlight.current.generation === generation.current;

  const job = useRepoDownload({
    kind: DOWNLOAD_KIND.MODEL,
    repoId: current?.repoId ?? "__staged_download_idle__",
    activeVariant,
    // Listeners are per repo; ignore other variants or a sibling's completion advances the queue.
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
    // Register ownership before the start request so an immediate cancel belongs to this plan.
    inFlight.current = started;
    const laterEntry = noticedGeneration.current === generation.current;
    noticedGeneration.current = generation.current;
    void (async () => {
      const outcome = await downloadManager.requestStart(
        current.ggufVariant
          ? {
              kind: DOWNLOAD_KIND.MODEL,
              repoId: current.repoId,
              variant: current.ggufVariant,
              expectedBytes: current.bytes,
              skipXetNotice: laterEntry,
            }
          : {
              kind: DOWNLOAD_KIND.MODEL,
              repoId: current.repoId,
              variant: activeVariant,
              inventoryKind: scopedDownloadInventoryKind(current.files),
              expectedBytes: current.bytes,
              scopeId,
              files: current.files,
              checkpoint: current.checkpoint,
              skipXetNotice: laterEntry,
            },
      );
      if (!active) return;
      if (outcome === "started") return;
      if (inFlight.current === started) inFlight.current = null;
      // A failed start never completes, so clear the queue instead of stalling the head.
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

  const stage = useCallback((entries: StagedDownloadEntry[]): number => {
    generation.current += 1;
    inFlight.current = null;
    setQueue(entries.length > 0 ? entries : null);
    setStaged({
      bytes: entries.reduce((sum, entry) => sum + Math.max(0, entry.bytes), 0),
      plan: generation.current,
    });
    return generation.current;
  }, []);

  const remainingBytes = (queue ?? []).reduce((sum, entry) => sum + Math.max(0, entry.bytes), 0);
  const currentBytes = current
    ? Math.min(Math.max(0, current.bytes), job.progress?.downloadedBytes ?? 0)
    : 0;
  const downloadedBytes = queue ? Math.max(0, staged.bytes - remainingBytes) + currentBytes : 0;
  const totalBytes = queue ? staged.bytes : 0;
  const plan = staged.plan;
  const progress = useMemo<StagedDownloadProgress | null>(
    () => (totalBytes > 0 ? { downloadedBytes, totalBytes, plan } : null),
    [downloadedBytes, totalBytes, plan],
  );

  return { stage, remaining: queue, staging: queue !== null, progress };
}
