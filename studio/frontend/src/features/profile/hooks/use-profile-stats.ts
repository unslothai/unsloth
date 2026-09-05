// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { subscribeResidentStatusRefresh } from "@/features/hub/lib/resident-status-refresh";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { type ProfileStats, loadProfileStats } from "../api/profile-stats";

type ProfileStatsState = {
  stats: ProfileStats | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
};

/** Load the profile stats on mount, with a manual refresh. */
export function useProfileStats(): ProfileStatsState {
  const [stats, setStats] = useState<ProfileStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // A refresh aborts the in-flight request so a slow first load cannot land
  // after (and overwrite) the newer one.
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    try {
      const next = await loadProfileStats(controller.signal);
      if (controller.signal.aborted) return;
      setStats(next);
      setError(null);
    } catch (cause: unknown) {
      if (controller.signal.aborted) return;
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => abortRef.current?.abort();
  }, [load]);

  useEffect(() => subscribeResidentStatusRefresh(load), [load]);

  const settingsOpen = useSettingsDialogStore((state) => state.open);
  const sawSettingsOpenRef = useRef(settingsOpen);
  useEffect(() => {
    const wasOpen = sawSettingsOpenRef.current;
    sawSettingsOpenRef.current = settingsOpen;
    if (!wasOpen && settingsOpen) {
      void load();
    }
  }, [settingsOpen, load]);

  const reload = useCallback(() => {
    void load();
  }, [load]);

  return { stats, loading, error, reload };
}
