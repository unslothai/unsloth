


import { useCallback, useEffect, useRef, useState } from "react";
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

  const reload = useCallback(() => {
    void load();
  }, [load]);

  return { stats, loading, error, reload };
}
