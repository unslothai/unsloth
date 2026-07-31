// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDeviceInventorySources } from "@/features/hub";
import type { DatasetSource } from "@/types/training";
import { useEffect, useRef } from "react";

export function useLocalDatasetInventory(datasetSource: DatasetSource) {
  const { localDatasets, refresh } = useDeviceInventorySources(
    ["localDatasets"],
    {
      enabled: datasetSource === "upload",
    },
  );
  const wasUploadSource = useRef(false);

  useEffect(() => {
    const isUploadSource = datasetSource === "upload";
    if (isUploadSource && !wasUploadSource.current && localDatasets.ready) {
      refresh().catch(() => undefined);
    }
    wasUploadSource.current = isUploadSource;
  }, [datasetSource, localDatasets.ready, refresh]);

  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.hidden || datasetSource !== "upload") {
        return;
      }
      refresh().catch(() => undefined);
    };
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [datasetSource, refresh]);

  return localDatasets.rows;
}
