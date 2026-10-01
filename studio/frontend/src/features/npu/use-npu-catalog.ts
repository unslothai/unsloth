// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  type NpuModel,
  type NpuStatus,
  deleteNpuModel,
  downloadNpuModel,
  enableNpu,
  getNpuStatus,
  listNpuModels,
} from "./api";

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
  enable: () => Promise<void>;
  download: (model: NpuModel) => Promise<boolean>;
  remove: (model: NpuModel) => Promise<void>;
}

export function useNpuCatalog(
  source: NpuPickerSource | undefined,
): NpuCatalog | null {
  const [models, setModels] = useState<NpuModel[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [enabling, setEnabling] = useState(false);
  const [downloads, setDownloads] = useState<Record<string, number | null>>({});
  const status = source?.status;
  const onStatusChange = source?.onStatusChange;
  // Listing starts lemond, so wait for a validated runtime.
  const ready = status?.ready === true;

  const refresh = useCallback(async () => {
    try {
      setModels(await listNpuModels());
      setListError(null);
    } catch (error) {
      setListError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    if (!ready) return;
    let live = true;
    listNpuModels().then(
      (next) => {
        if (!live) return;
        setModels(next);
        setListError(null);
      },
      (error) => {
        if (live) {
          setListError(error instanceof Error ? error.message : String(error));
        }
      },
    );
    return () => {
      live = false;
    };
  }, [ready]);

  const enable = useCallback(async () => {
    if (!onStatusChange) return;
    setEnabling(true);
    try {
      onStatusChange(await enableNpu());
      await refresh();
    } catch (error) {
      toast.error("Could not enable the NPU", {
        description: error instanceof Error ? error.message : String(error),
      });
      await getNpuStatus().then(onStatusChange, () => undefined);
    } finally {
      setEnabling(false);
    }
  }, [onStatusChange, refresh]);

  const download = useCallback(
    async (model: NpuModel) => {
      setDownloads((current) => ({ ...current, [model.id]: null }));
      try {
        await downloadNpuModel(model.id, (event) => {
          if (typeof event.percent === "number") {
            setDownloads((current) => ({
              ...current,
              [model.id]: event.percent ?? null,
            }));
          }
        });
        await refresh();
        return true;
      } catch (error) {
        toast.error(`Could not download ${model.id}`, {
          description: error instanceof Error ? error.message : String(error),
        });
        return false;
      } finally {
        setDownloads((current) => {
          const next = { ...current };
          delete next[model.id];
          return next;
        });
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (model: NpuModel) => {
      await deleteNpuModel(model.id);
      await refresh();
    },
    [refresh],
  );

  if (!status) return null;
  return {
    status,
    ready: status.ready,
    models,
    listError,
    enabling,
    downloads,
    enable,
    download,
    remove,
  };
}
