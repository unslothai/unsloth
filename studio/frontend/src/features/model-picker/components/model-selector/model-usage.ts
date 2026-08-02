// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tracks when each model was last loaded so the "Recent" sort can order by usage
// (distinct from "Downloaded", which orders by the file's download date). Kept in
// localStorage; ids are lowercased to match how the picker compares them.

import { useEffect, useState } from "react";

export type ModelLoadTimes = Record<string, number>;

const STORAGE_KEY = "unsloth.model-load-times.v1";

function readLoadTimes(): ModelLoadTimes {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ModelLoadTimes) : {};
  } catch {
    return {};
  }
}

/** Stamp a model as loaded now and return the updated map. */
export function recordModelLoaded(id: string): ModelLoadTimes {
  const next = { ...readLoadTimes(), [id.toLowerCase()]: Date.now() };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // ignore quota / disabled storage
  }
  return next;
}

/** Epoch ms the model was last loaded, or -1 if never. */
export function loadedAt(times: ModelLoadTimes, id: string): number {
  return times[id.toLowerCase()] ?? -1;
}

/** Load times, restamping whenever the active model changes. */
export function useModelLoadTimes(currentValue?: string): ModelLoadTimes {
  const [times, setTimes] = useState<ModelLoadTimes>(() => readLoadTimes());
  useEffect(() => {
    if (!currentValue) return;
    queueMicrotask(() => setTimes(recordModelLoaded(currentValue)));
  }, [currentValue]);
  return times;
}
