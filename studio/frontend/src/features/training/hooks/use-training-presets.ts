// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { UserAssetApiError } from "@/features/user-assets";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type TrainingPreset,
  createTrainingPreset,
  deleteTrainingPreset,
  getTrainingPreset,
  listTrainingPresets,
  updateTrainingPreset,
} from "../api/presets-api";
import type { PortableTrainingConfig } from "../lib/yaml-config";

function upsert(
  items: TrainingPreset[],
  record: TrainingPreset,
): TrainingPreset[] {
  return [record, ...items.filter((item) => item.id !== record.id)].sort(
    (a, b) => b.updatedAt - a.updatedAt || a.id.localeCompare(b.id),
  );
}

export function useTrainingPresets() {
  const [presets, setPresets] = useState<TrainingPreset[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [conflict, setConflict] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setPresets(await listTrainingPresets());
    } catch (caught) {
      setError(
        caught instanceof Error ? caught : new Error("Failed to load presets."),
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    listTrainingPresets()
      .then((records) => {
        if (!active) return;
        setPresets(records);
        setLoading(false);
      })
      .catch((caught: unknown) => {
        if (!active) return;
        setError(
          caught instanceof Error
            ? caught
            : new Error("Failed to load presets."),
        );
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const selected = presets.find((preset) => preset.id === selectedId) ?? null;

  const selectPreset = useCallback((id: string | null) => {
    selectedIdRef.current = id;
    setSelectedId(id);
    setError(null);
  }, []);

  const saveAs = useCallback(
    async (name: string, config: PortableTrainingConfig) => {
      const record = await createTrainingPreset({
        id: crypto.randomUUID(),
        name: name.trim(),
        config,
      });
      setPresets((items) => upsert(items, record));
      selectPreset(record.id);
      setConflict(false);
      return record;
    },
    [selectPreset],
  );

  const save = useCallback(
    async (name: string, config: PortableTrainingConfig) => {
      if (!selected) return saveAs(name, config);
      try {
        const record = await updateTrainingPreset({
          id: selected.id,
          name: name.trim(),
          config,
          revision: selected.revision,
        });
        setPresets((items) => upsert(items, record));
        setConflict(false);
        return record;
      } catch (caught) {
        if (caught instanceof UserAssetApiError && caught.status === 409) {
          setConflict(true);
        }
        throw caught;
      }
    },
    [saveAs, selected],
  );

  const remove = useCallback(async () => {
    if (!selected) return;
    try {
      await deleteTrainingPreset(selected.id, selected.revision);
      setPresets((items) => items.filter((item) => item.id !== selected.id));
      selectPreset(null);
      setConflict(false);
    } catch (caught) {
      if (caught instanceof UserAssetApiError && caught.status === 409) {
        setConflict(true);
      }
      throw caught;
    }
  }, [selectPreset, selected]);

  const reloadSelected = useCallback(async () => {
    const requestedId = selectedIdRef.current;
    if (!requestedId) return null;
    setError(null);
    try {
      const record = await getTrainingPreset(requestedId);
      if (selectedIdRef.current !== requestedId) return null;
      setPresets((items) => upsert(items, record));
      setConflict(false);
      return record;
    } catch (caught) {
      if (selectedIdRef.current === requestedId) {
        setError(
          caught instanceof Error
            ? caught
            : new Error("Failed to reload preset."),
        );
      }
      throw caught;
    }
  }, []);

  return {
    presets,
    selected,
    selectedId,
    setSelectedId: selectPreset,
    loading,
    error,
    conflict,
    refresh,
    save,
    saveAs,
    remove,
    reloadSelected,
  };
}
