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
  resetSignature,
}: {
  isDiscoverTab: boolean;
  scannedCount: number;
  filteredCount: number;
  // A single value that changes whenever the pause should reset (e.g. query
  // or filters changed). Pass a stable serialization, not a fresh array.
  resetSignature: string;
}): FilterStarvedPause {
  const [filterPaused, setFilterPaused] = useState(false);
  const [pauseFloor, setPauseFloor] = useState(FILTER_STARVED_THRESHOLD);

  // biome-ignore lint/correctness/useExhaustiveDependencies: resetSignature is a reset trigger, not read in the body
  useEffect(() => {
    setFilterPaused(false);
    setPauseFloor(FILTER_STARVED_THRESHOLD);
  }, [resetSignature]);

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
