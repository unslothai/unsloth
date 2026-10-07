// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  type NpuModel,
  type NpuStatus,
  deleteNpuModel,
  enableNpu,
  getNpuStatus,
  listNpuDownloads,
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

  useEffect(() => {
    if (!ready) return;
    void refreshNpuModels();
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

  const download = useCallback(
    (model: NpuModel) =>
      followNpuDownload(model.id, { percent: model.resume_percent }),
    [],
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
    remove,
  };
}
