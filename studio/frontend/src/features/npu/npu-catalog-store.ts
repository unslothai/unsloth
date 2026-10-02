// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { reportDownloadsActive } from "@/lib/downloads-activity";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import {
  NpuDownloadError,
  type NpuModel,
  downloadNpuModel,
  followNpuModelDownload,
  listNpuDownloads,
  listNpuModels,
} from "./api";

interface NpuCatalogState {
  models: NpuModel[] | null;
  listError: string | null;
  /** Model id to percent; null until the pull reports one. */
  progress: Record<string, number | null>;
}

// Outside the picker, which unmounts on close, so a pull's progress outlives it.
export const useNpuCatalogStore = create<NpuCatalogState>(() => ({
  models: null,
  listError: null,
  progress: {},
}));

// Quitting the desktop app stops the backend and its pulls, so it warns while one runs.
useNpuCatalogStore.subscribe((state) =>
  reportDownloadsActive("npu", Object.keys(state.progress).length > 0),
);

let latestListing = 0;

export async function refreshNpuModels(): Promise<void> {
  // Only the newest answer lands: an older one could still list a finished pull as missing.
  const listing = ++latestListing;
  try {
    const models = await listNpuModels();
    if (listing === latestListing) {
      useNpuCatalogStore.setState({ models, listError: null });
    }
  } catch (error) {
    if (listing === latestListing) {
      useNpuCatalogStore.setState({
        listError: error instanceof Error ? error.message : String(error),
      });
    }
  }
}

function setProgress(id: string, percent: number | null): void {
  useNpuCatalogStore.setState((state) =>
    state.progress[id] === percent
      ? state
      : { progress: { ...state.progress, [id]: percent } },
  );
}

const following = new Map<string, Promise<boolean>>();
// A broken stream is followed again for as long as the backend lists the pull. Checks that cannot
// reach the backend this many times in a row mean Studio itself stopped, and the pull with it.
const MAX_UNREACHABLE = 5;
// A reconnect waits RECONNECT_DELAY_MS times one more than the reconnects in a row whose stream
// brought no new percent, up to MAX_BACKOFF times.
const RECONNECT_DELAY_MS = 1000;
const MAX_BACKOFF = 5;

/**
 * Whether the backend still runs the model's pull: its percent if so, null if it runs none,
 * undefined when the backend could not be asked.
 */
async function runningPull(
  id: string,
): Promise<{ percent: number | null } | null | undefined> {
  try {
    const running = await listNpuDownloads();
    return running.find((download) => download.model === id) ?? null;
  } catch {
    return undefined;
  }
}

function listedDownloaded(id: string): boolean {
  return (
    useNpuCatalogStore.getState().models?.find((model) => model.id === id)
      ?.downloaded === true
  );
}

/**
 * Show a model's pull until it ends, one stream per model. `follow` only joins a pull the
 * backend is already running, which replays its earlier progress; otherwise this starts one.
 * A broken stream does not end a pull that the backend still lists: it is followed again.
 * Resolves true once the model is downloaded.
 */
export function followNpuDownload(
  id: string,
  {
    follow = false,
    percent = null,
  }: { follow?: boolean; percent?: number | null } = {},
): Promise<boolean> {
  const current = following.get(id);
  if (current) return current;
  setProgress(id, percent);

  const run = async (): Promise<boolean> => {
    let joined = follow;
    let completed = false;
    // A followed stream replays every earlier event, so only a higher percent is progress.
    let best = percent ?? -1;
    const advance = (next: number | null | undefined): boolean => {
      if (next == null || next <= best) return false;
      best = next;
      setProgress(id, next);
      return true;
    };
    let streamed = false;
    const onProgress = (event: { event: string; percent?: number }) => {
      completed ||= event.event === "complete";
      if (advance(event.percent)) streamed = true;
    };
    let stalledReconnects = 0;
    let unreachable = 0;
    for (;;) {
      streamed = false;
      try {
        await (joined ? followNpuModelDownload : downloadNpuModel)(
          id,
          onProgress,
        );
        // Listed before the progress clears, so the row never reads as not downloaded.
        await refreshNpuModels();
        // A followed pull can end, either way, before its stream opens; the list says how.
        return completed || listedDownloaded(id);
      } catch (error) {
        if (!(error instanceof NpuDownloadError)) {
          const running = await runningPull(id);
          unreachable = running === undefined ? unreachable + 1 : 0;
          // Still running, or the backend could not be asked yet: follow it again.
          if (running !== null && unreachable < MAX_UNREACHABLE) {
            advance(running?.percent);
            // Backs off from a backend that answers while its stream keeps breaking.
            stalledReconnects =
              streamed || !running ? 0 : stalledReconnects + 1;
            joined = true;
            const backoff = Math.min(stalledReconnects + 1, MAX_BACKOFF);
            await new Promise((resolve) =>
              setTimeout(resolve, RECONNECT_DELAY_MS * backoff),
            );
            continue;
          }
        }
        // Also lists what a failed download left, which the next one continues from.
        await refreshNpuModels();
        // The pull may have finished while its stream was broken.
        if (listedDownloaded(id)) return true;
        toast.error(`Could not download ${id}`, {
          description: error instanceof Error ? error.message : String(error),
        });
        return false;
      }
    }
  };

  const job = run().finally(() => {
    following.delete(id);
    useNpuCatalogStore.setState((state) => {
      const progress = { ...state.progress };
      delete progress[id];
      return { progress };
    });
  });
  following.set(id, job);
  return job;
}
