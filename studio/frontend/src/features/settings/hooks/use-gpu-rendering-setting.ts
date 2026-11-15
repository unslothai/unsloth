// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import {
  type GpuRenderingSetting,
  type GpuRenderingStatus,
  loadGpuRendering,
  relaunchToApplyGpuRendering,
  updateGpuRendering,
} from "../api/gpu-rendering";

interface GpuRenderingSettingOptions {
  enabled: boolean;
  loadError: string;
  saveError: string;
  restartError: string;
}

export function useGpuRenderingSetting({
  enabled,
  loadError,
  saveError,
  restartError,
}: GpuRenderingSettingOptions) {
  const [status, setStatus] = useState<GpuRenderingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    void loadGpuRendering()
      .then((loaded) => {
        if (cancelled) return;
        setStatus(loaded);
        setError(null);
      })
      .catch((cause) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : loadError);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, loadError]);

  // Re-selecting the current value is how "Try GPU again" clears the fallback.
  const update = useCallback(
    async (next: GpuRenderingSetting) => {
      setSaving(true);
      setError(null);
      try {
        setStatus(await updateGpuRendering(next));
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : saveError);
      } finally {
        setSaving(false);
      }
    },
    [saveError],
  );

  const relaunch = useCallback(async () => {
    setRestarting(true);
    setError(null);
    try {
      // False means a quit is already pending and its dialog is showing.
      await relaunchToApplyGpuRendering();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : restartError);
    } finally {
      setRestarting(false);
    }
  }, [restartError]);

  return { status, error, saving, restarting, update, relaunch };
}
