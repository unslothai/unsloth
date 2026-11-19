// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { fetchVadDetectors, type VadDetector } from "./vad-client";

export function useVadDetectors(): VadDetector[] {
  const [detectors, setDetectors] = useState<VadDetector[]>([]);
  useEffect(() => {
    const abort = new AbortController();
    fetchVadDetectors(abort.signal)
      .then(setDetectors)
      .catch(() => {});
    return () => abort.abort();
  }, []);
  return detectors;
}
