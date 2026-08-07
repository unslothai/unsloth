// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import { ejectLoadedModel, readLoadedModels } from "./loaded-models-api";
import {
  type LoadedModelEntry,
  shortModelLabel,
} from "./loaded-models-sources";

// Free next to chat's own status poll, still current on a tab switch.
const POLL_INTERVAL_MS = 5000;

const NO_ENTRIES: LoadedModelEntry[] = [];

export type UseLoadedModels = {
  entries: LoadedModelEntry[];
  /** Row ids with an eject in flight. */
  ejecting: ReadonlySet<string>;
  eject: (entry: LoadedModelEntry) => Promise<void>;
  refresh: () => void;
};

/** Poll every runtime for what it holds. Paused while disabled or hidden. */
export function useLoadedModels(enabled: boolean): UseLoadedModels {
  const [polled, setEntries] = useState<LoadedModelEntry[]>([]);
  // Reported empty rather than cleared: clearing would be a setState in an
  // effect, and the last read is right again the moment the pref returns.
  const entries = enabled ? polled : NO_ENTRIES;
  const [ejecting, setEjecting] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  );
  // One read at a time, and none applied after unmount.
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(() => {
    if (!enabled || inFlightRef.current) return;
    inFlightRef.current = true;
    void readLoadedModels()
      .then((next) => {
        if (mountedRef.current) setEntries(next);
      })
      .catch(() => {
        // Every source fails soft; nothing left to report.
      })
      .finally(() => {
        inFlightRef.current = false;
      });
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      refresh();
    }, POLL_INTERVAL_MS);
    // Pick up a load or unload done in another tab.
    const onWake = () => {
      if (!document.hidden) refresh();
    };
    window.addEventListener("focus", onWake);
    document.addEventListener("visibilitychange", onWake);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", onWake);
      document.removeEventListener("visibilitychange", onWake);
    };
  }, [enabled, refresh]);

  const eject = useCallback(
    async (entry: LoadedModelEntry) => {
      setEjecting((prev) => new Set(prev).add(entry.id));
      const label = shortModelLabel(entry.name);
      try {
        const stillResident = await ejectLoadedModel(entry);
        if (stillResident) {
          toast.warning(
            `"${shortModelLabel(stillResident)}" was loaded while ejecting, so it is still using memory. Eject again to release it.`,
          );
        } else {
          toast.success(`Ejected ${label}`);
          // Drop the row now, rather than offering to eject it again until the
          // next poll.
          if (mountedRef.current) {
            setEntries((prev) => prev.filter((row) => row.id !== entry.id));
          }
        }
      } catch (error: unknown) {
        toast.error(
          error instanceof Error ? error.message : `Failed to eject ${label}`,
        );
      } finally {
        if (mountedRef.current) {
          setEjecting((prev) => {
            const next = new Set(prev);
            next.delete(entry.id);
            return next;
          });
        }
        refresh();
      }
    },
    [refresh],
  );

  return { entries, ejecting, eject, refresh };
}
