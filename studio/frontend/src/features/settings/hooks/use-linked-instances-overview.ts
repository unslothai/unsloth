// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type LinkedInstance,
  type LinkedInstanceInfo,
  type LinkedInstanceStatus,
  fetchLinkedInstances,
  fetchLinkedInstancesInfo,
  fetchLinkedInstancesStatus,
} from "../api/linked-instances";

const byId = <T extends { id: string }>(rows: T[]) =>
  Object.fromEntries(rows.map((r) => [r.id, r])) as Record<string, T>;

/** Linked instances with their reachability and machine details, polled while `enabled`. */
export function useLinkedInstancesOverview(enabled: boolean, pollMs = 30_000) {
  const [instances, setInstances] = useState<LinkedInstance[]>([]);
  const [statuses, setStatuses] = useState<
    Record<string, LinkedInstanceStatus>
  >({});
  const [infos, setInfos] = useState<Record<string, LinkedInstanceInfo>>({});
  const [refreshing, setRefreshing] = useState(false);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const [list, status, info] = await Promise.all([
        fetchLinkedInstances(),
        fetchLinkedInstancesStatus(),
        fetchLinkedInstancesInfo(),
      ]);
      if (!alive.current) return;
      setInstances(list);
      setStatuses(byId(status));
      setInfos(byId(info));
    } catch {
      // Keep the last answer; the settings card reports errors.
    } finally {
      if (alive.current) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    if (!enabled) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), pollMs);
    return () => {
      alive.current = false;
      window.clearInterval(timer);
    };
  }, [enabled, pollMs, refresh]);

  return { instances, statuses, infos, refresh, refreshing };
}
