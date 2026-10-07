// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Last-load times for the "Recent" sort; ids lowercased to match the picker's comparisons.

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

export function useModelLoadTimes(currentValue?: string): ModelLoadTimes {
  const [times, setTimes] = useState<ModelLoadTimes>(() => readLoadTimes());
  useEffect(() => {
    if (!currentValue) return;
    queueMicrotask(() => setTimes(recordModelLoaded(currentValue)));
  }, [currentValue]);
  return times;
}
