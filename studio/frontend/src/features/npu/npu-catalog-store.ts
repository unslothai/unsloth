// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { reportDownloadsActive } from "@/lib/downloads-activity";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import {
  type NpuModel,
  downloadNpuModel,
  followNpuModelDownload,
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

/**
 * Show a model's pull until it ends, one stream per model. `follow` only joins a pull the
 * backend is already running, which replays its earlier progress; otherwise this starts one.
 * Resolves true once the model is downloaded and listed as such.
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
  const onProgress = (event: { percent?: number }) => {
    if (typeof event.percent === "number") setProgress(id, event.percent);
  };
  const job = (follow ? followNpuModelDownload : downloadNpuModel)(
    id,
    onProgress,
  )
    .then(
      async () => {
        // Listed before the progress clears, so the row never reads as not downloaded.
        await refreshNpuModels();
        if (!follow) return true;
        // A followed pull can end, either way, before its stream opens; the list says how.
        return (
          useNpuCatalogStore.getState().models?.find((model) => model.id === id)
            ?.downloaded === true
        );
      },
      (error: unknown) => {
        toast.error(`Could not download ${id}`, {
          description: error instanceof Error ? error.message : String(error),
        });
        // Lists what the failed download left, which the next one continues from.
        void refreshNpuModels();
        return false;
      },
    )
    .finally(() => {
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
