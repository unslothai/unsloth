// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import { ejectLoadedModel, readLoadedModels } from "./loaded-models-api";
import {
  type LoadedModelEntry,
  type LoadedModelSource,
  shortModelLabel,
  withPendingLoads,
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
  // Loads announced by the API call itself, so a row appears with the toast
  // rather than up to one poll later.
  const [pending, setPending] = useState<Map<LoadedModelSource, string | null>>(
    () => new Map(),
  );
  const entries = enabled ? withPendingLoads(polled, pending) : NO_ENTRIES;
  const [ejecting, setEjecting] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  );
  // One read at a time, and none applied after unmount.
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);
  // Bumped whenever an eject changes the list. A read issued before that lands
  // after it and would otherwise put the ejected row straight back.
  const generationRef = useRef(0);
  // A refresh asked for while one was in flight, so the trailing refresh after
  // an eject is not swallowed by the very read it needs to supersede.
  const pendingRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refreshRef = useRef<() => void>(() => {});
  const refresh = useCallback(() => {
    if (!enabled) return;
    if (inFlightRef.current) {
      // Remember the ask instead of dropping it: the refresh an eject queues
      // collides with the poll it has to correct more often than not.
      pendingRef.current = true;
      return;
    }
    inFlightRef.current = true;
    const generation = generationRef.current;
    void readLoadedModels()
      .then((next) => {
        if (mountedRef.current && generation === generationRef.current) {
          setEntries(next);
        }
      })
      .catch(() => {
        // Every source fails soft; nothing left to report.
      })
      .finally(() => {
        inFlightRef.current = false;
        if (pendingRef.current) {
          pendingRef.current = false;
          refreshRef.current();
        }
      });
  }, [enabled]);
  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  // The load call announces itself, so the row and the toast appear together
  // and a finished load is re-read at once instead of on the next tick.
  useEffect(() => {
    if (!enabled) return;
    return subscribeModelLifecycle(({ runtime, loading, model }) => {
      setPending((prev) => {
        const next = new Map(prev);
        if (loading) next.set(runtime, model);
        else next.delete(runtime);
        return next;
      });
      if (!loading) refresh();
    });
  }, [enabled, refresh]);

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
        const outcome = await ejectLoadedModel(entry);
        if (outcome.status === "stillResident") {
          toast.warning(
            `"${shortModelLabel(outcome.model)}" was loaded while ejecting, so it is still using memory. Eject again to release it.`,
          );
        } else if (outcome.status === "replaced") {
          // Nothing was unloaded: this runtime holds something else now, and
          // releasing that is not what the click asked for.
          toast.info(
            `${label} is no longer loaded. "${shortModelLabel(outcome.resident)}" took its place and was left alone.`,
          );
        } else {
          toast.success(`Ejected ${label}`);
          // Drop the row now, rather than offering to eject it again until the
          // next poll. Any read already in flight predates this and would put
          // the row back, so retire it.
          generationRef.current += 1;
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
