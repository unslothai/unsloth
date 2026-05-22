// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

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
  const [pauseState, setPauseState] = useState({
    resetSignature,
    pauseFloor: FILTER_STARVED_THRESHOLD,
  });
  const pauseFloor =
    pauseState.resetSignature === resetSignature
      ? pauseState.pauseFloor
      : FILTER_STARVED_THRESHOLD;
  const filterPaused =
    isDiscoverTab && scannedCount >= pauseFloor && filteredCount === 0;

  const handleKeepSearching = useCallback(() => {
    setPauseState({
      resetSignature,
      pauseFloor: scannedCount + FILTER_STARVED_THRESHOLD,
    });
  }, [resetSignature, scannedCount]);

  return { filterPaused, handleKeepSearching };
}
