// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

const FILTER_STARVED_THRESHOLD = 100;

export interface FilterStarvedPause {
  filterPaused: boolean;
  handleKeepSearching: () => void;
}

export function useFilterStarvedPause({
  isDiscoverTab,
  scannedCount,
  filteredCount,
  resetDeps,
}: {
  isDiscoverTab: boolean;
  scannedCount: number;
  filteredCount: number;
  // Spread as a useEffect dependency array, so callers must pass a
  // constant-length array across renders (React throws otherwise).
  resetDeps: ReadonlyArray<unknown>;
}): FilterStarvedPause {
  const [filterPaused, setFilterPaused] = useState(false);
  const [pauseFloor, setPauseFloor] = useState(FILTER_STARVED_THRESHOLD);

  useEffect(() => {
    setFilterPaused(false);
    setPauseFloor(FILTER_STARVED_THRESHOLD);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, resetDeps);

  useEffect(() => {
    if (
      isDiscoverTab &&
      !filterPaused &&
      scannedCount >= pauseFloor &&
      filteredCount === 0
    ) {
      setFilterPaused(true);
    }
  }, [isDiscoverTab, filterPaused, pauseFloor, scannedCount, filteredCount]);

  const handleKeepSearching = useCallback(() => {
    setPauseFloor(scannedCount + FILTER_STARVED_THRESHOLD);
    setFilterPaused(false);
  }, [scannedCount]);

  return { filterPaused, handleKeepSearching };
}
