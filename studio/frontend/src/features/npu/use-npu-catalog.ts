// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type NpuModel,
  type NpuStatus,
  deleteNpuModel,
  enableNpu,
  getNpuStatus,
  listNpuDownloads,
  npuMemoryWarning,
} from "./api";
import {
  followNpuDownload,
  refreshNpuModels,
  useNpuCatalogStore,
} from "./npu-catalog-store";

export interface NpuPickerSource {
  status: NpuStatus;
  onStatusChange: (status: NpuStatus) => void;
}

export interface NpuCatalog {
  status: NpuStatus;
  ready: boolean;
  models: NpuModel[] | null;
  listError: string | null;
  enabling: boolean;
  downloads: Record<string, number | null>;
  reconnecting: Record<string, true>;
  enable: () => Promise<void>;
  download: (model: NpuModel) => Promise<boolean>;
  /** Toasts the memory warning for a model that may not fit; never blocks. */
  warnMemory: (model: NpuModel) => void;
  remove: (model: NpuModel) => Promise<void>;
}

export function useNpuCatalog(
  source: NpuPickerSource | undefined,
): NpuCatalog | null {
  const models = useNpuCatalogStore((state) => state.models);
  const listError = useNpuCatalogStore((state) => state.listError);
  const downloads = useNpuCatalogStore((state) => state.progress);
  const reconnecting = useNpuCatalogStore((state) => state.reconnecting);
  const [enabling, setEnabling] = useState(false);
  const status = source?.status;
  const onStatusChange = source?.onStatusChange;
  // Listing starts lemond, so wait for a validated runtime.
  const ready = status?.ready === true;
  const statusChange = useRef(onStatusChange);
  useEffect(() => {
    statusChange.current = onStatusChange;
  });

  useEffect(() => {
    if (!ready) return;
    // This listing can run a pending upgrade: re-read status for its versions, or for Try again.
    void refreshNpuModels().then(() =>
      getNpuStatus().then(
        (next) => statusChange.current?.(next),
        () => undefined,
      ),
    );
    listNpuDownloads().then(
      (running) => {
        for (const { model, percent } of running) {
          void followNpuDownload(model, { follow: true, percent });
        }
      },
      () => undefined,
    );
  }, [ready]);

  const enable = useCallback(async () => {
    if (!onStatusChange) return;
    setEnabling(true);
    try {
      onStatusChange(await enableNpu());
      await refreshNpuModels();
    } catch (error) {
      toast.error("Could not enable the NPU", {
        description: error instanceof Error ? error.message : String(error),
      });
      await getNpuStatus().then(onStatusChange, () => undefined);
    } finally {
      setEnabling(false);
    }
  }, [onStatusChange]);

  const memoryWindowGb = status?.memory_window_gb;
  const warnMemory = useCallback(
    (model: NpuModel) => {
      const warning = npuMemoryWarning(model, {
        memory_window_gb: memoryWindowGb,
      });
      if (warning) {
        toast.warning(`${model.id} may not load on the NPU`, {
          description: warning,
        });
      }
    },
    [memoryWindowGb],
  );

  const download = useCallback(
    (model: NpuModel) => {
      warnMemory(model);
      return followNpuDownload(model.id, { percent: model.resume_percent });
    },
    [warnMemory],
  );

  const remove = useCallback(async (model: NpuModel) => {
    await deleteNpuModel(model.id);
    await refreshNpuModels();
  }, []);

  if (!status) return null;
  return {
    status,
    ready: status.ready,
    models: ready ? models : null,
    listError,
    enabling,
    downloads,
    reconnecting,
    enable,
    download,
    warnMemory,
    remove,
  };
}
