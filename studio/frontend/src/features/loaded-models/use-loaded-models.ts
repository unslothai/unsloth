// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ModelRuntime,
  subscribeModelLifecycle,
} from "@/lib/model-lifecycle-events";
import { ejectLoadedModel, readLoadedModels } from "./loaded-models-api";
import {
  type LoadedModelEntry,
  type LoadedModelSource,
  shortModelLabel,
  withPendingLoads,
} from "./loaded-models-sources";

const POLL_INTERVAL_MS = 5000;

const NO_ENTRIES: LoadedModelEntry[] = [];

const ALL_SOURCES: LoadedModelSource[] = ["chat", "image", "video", "stt"];

// A TTS load takes the chat slot and the chat unload releases it.
function sourceForRuntime(runtime: ModelRuntime): LoadedModelSource {
  return runtime === "tts" ? "chat" : runtime;
}

export type UseLoadedModels = {
  entries: LoadedModelEntry[];
  /** Polled rows even while the card is closed, so it can reopen on a load it was not told about. */
  polledEntries: LoadedModelEntry[];
  ejecting: ReadonlySet<string>;
  eject: (entry: LoadedModelEntry) => Promise<void>;
  refresh: () => void;
};

/**
 * `enabled` controls showing; `track` controls recording announced loads. A closed card still
 * tracks; only the Settings toggle turns tracking off.
 */
export function useLoadedModels(
  enabled: boolean,
  track: boolean = enabled,
): UseLoadedModels {
  const [polled, setEntries] = useState<LoadedModelEntry[]>([]);
  const polledRef = useRef<LoadedModelEntry[]>(NO_ENTRIES);
  // Reported empty rather than cleared, avoiding a setState in an effect.
  const [pending, setPending] = useState<Map<LoadedModelSource, string | null>>(
    () => new Map(),
  );
  const entries = enabled ? withPendingLoads(polled, pending) : NO_ENTRIES;
  const [ejecting, setEjecting] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  );
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);
  // Bumped on eject so an older in-flight read cannot restore the ejected row.
  const generationRef = useRef(0);
  // Remember a refresh requested mid-read so an eject's trailing refresh is not swallowed.
  const pendingRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Retired only after a read lands, so the row never blinks out in the gap.
  const settledRef = useRef<Set<LoadedModelSource>>(new Set());
  /** Unreadable sources stay settled; a failed request must not drop a just-loaded row. */
  const retireSettled = useCallback((unreadable: LoadedModelSource[] = []) => {
    if (settledRef.current.size === 0) return;
    const done = [...settledRef.current].filter(
      (source) => !unreadable.includes(source),
    );
    if (done.length === 0) return;
    settledRef.current = new Set(
      [...settledRef.current].filter((source) => !done.includes(source)),
    );
    setPending((prev) => {
      const next = new Map(prev);
      for (const source of done) next.delete(source);
      return next.size === prev.size ? prev : next;
    });
  }, []);

  const refreshRef = useRef<() => void>(() => {});
  const refresh = useCallback(() => {
    // Keyed on recording, not showing: API loads raise no event, so a closed card keeps polling.
    if (!track) return;
    if (inFlightRef.current) {
      pendingRef.current = true;
      return;
    }
    inFlightRef.current = true;
    const generation = generationRef.current;
    // A source that fails to answer keeps its rows rather than reading as empty.
    let unreadable: LoadedModelSource[] = [];
    void readLoadedModels(polledRef.current)
      .then((next) => {
        unreadable = next.unreadable;
        if (mountedRef.current && generation === generationRef.current) {
          setEntries(next.entries);
        }
      })
      .catch(() => {
        unreadable = ALL_SOURCES;
      })
      .finally(() => {
        inFlightRef.current = false;
        if (pendingRef.current) {
          pendingRef.current = false;
          refreshRef.current();
          return;
        }
        if (mountedRef.current) retireSettled(unreadable);
      });
  }, [track, retireSettled]);
  useEffect(() => {
    polledRef.current = polled;
  }, [polled]);
  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  // Once recording stops, terminal events are missed and optimistic rows could never retire, so
  // drop them. Adjusted during render so stale rows never reach the DOM.
  const [wasTracking, setWasTracking] = useState(track);
  if (wasTracking !== track) {
    setWasTracking(track);
    if (!track && pending.size > 0) setPending(new Map());
  }

  // The load call announces itself, so the row and the toast appear together
  // and a finished load is re-read at once instead of on the next tick.
  useEffect(() => {
    if (!track) return;
    return subscribeModelLifecycle(({ runtime, loading, model }) => {
      const source = sourceForRuntime(runtime);
      if (loading) {
        settledRef.current.delete(source);
        setPending((prev) => new Map(prev).set(source, model));
        return;
      }
      // Kept until a read answers, or the row blinks out in the gap.
      settledRef.current.add(source);
      refresh();
    });
  }, [track, refresh]);

  useEffect(() => {
    if (!track) return;
    refresh();
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      refresh();
    }, POLL_INTERVAL_MS);
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
  }, [track, refresh]);

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
        } else if (outcome.status === "unverified") {
          toast.warning(
            `${label} was asked to unload, but its runtime did not confirm. Check the card in a moment.`,
          );
        } else if (outcome.status === "alreadyFree") {
          toast.info(`${label} was no longer loaded.`);
          generationRef.current += 1;
          if (mountedRef.current) {
            setEntries((prev) => prev.filter((row) => row.id !== entry.id));
          }
        } else if (outcome.status === "replaced") {
          toast.info(
            `${label} is no longer loaded. "${shortModelLabel(outcome.resident)}" took its place and was left alone.`,
          );
        } else {
          toast.success(`Ejected ${label}`);
          // Any read already in flight predates this and would put the row back.
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

  return { entries, polledEntries: polled, ejecting, eject, refresh };
}
