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

const ALL_SOURCES: LoadedModelSource[] = ["chat", "image", "video", "stt"];

export type UseLoadedModels = {
  entries: LoadedModelEntry[];
  /** What the polls actually found, whether or not the card is showing it. The
   *  card is closed, not deaf: this is how it notices a load it was not told
   *  about and honours "Back on the next model load". */
  polledEntries: LoadedModelEntry[];
  /** Row ids with an eject in flight. */
  ejecting: ReadonlySet<string>;
  eject: (entry: LoadedModelEntry) => Promise<void>;
  refresh: () => void;
};

/**
 * Poll every runtime for what it holds. Paused while disabled or hidden.
 *
 * `track` is the narrower of the two: it says whether to RECORD the loads the
 * API calls announce, where `enabled` says whether to show anything. They come
 * apart for a card the user closed, which must still hear the load that reopens
 * it, and for a route the card is hidden on. Only the Settings toggle turns
 * recording off, since that is the one that means "stop telling me".
 */
export function useLoadedModels(
  enabled: boolean,
  track: boolean = enabled,
): UseLoadedModels {
  const [polled, setEntries] = useState<LoadedModelEntry[]>([]);
  // Mirrored for the read below, which needs the last rows without taking a
  // dependency that would rebuild `refresh` on every poll.
  const polledRef = useRef<LoadedModelEntry[]>(NO_ENTRIES);
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

  // Runtimes whose load has settled but whose row is still the optimistic one.
  // Retired only once a read has landed, so the row never blinks out in the gap
  // between the load finishing and the status that replaces it arriving.
  const settledRef = useRef<Set<LoadedModelSource>>(new Set());
  /**
   * Drop the optimistic rows for loads that have finished, now that a read has
   * replaced them. `unreadable` names the sources that read could say nothing
   * about: retiring those would take away the row for a model that has just
   * finished loading, on the strength of a request that failed. They stay
   * settled, so the next readable poll retires them instead.
   */
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
    // Keyed on recording, not showing: a closed card keeps polling so a load
    // started outside this tab, which raises no lifecycle event at all, still
    // brings it back.
    if (!track) return;
    if (inFlightRef.current) {
      // Remember the ask instead of dropping it: the refresh an eject queues
      // collides with the poll it has to correct more often than not.
      pendingRef.current = true;
      return;
    }
    inFlightRef.current = true;
    const generation = generationRef.current;
    // What the card shows now, so a source that fails to answer keeps its rows
    // rather than being read as empty.
    let unreadable: LoadedModelSource[] = [];
    void readLoadedModels(polledRef.current)
      .then((next) => {
        unreadable = next.unreadable;
        if (mountedRef.current && generation === generationRef.current) {
          setEntries(next.entries);
        }
      })
      .catch(() => {
        // The whole read failed, so it is evidence about nothing at all.
        unreadable = ALL_SOURCES;
      })
      .finally(() => {
        inFlightRef.current = false;
        if (pendingRef.current) {
          pendingRef.current = false;
          refreshRef.current();
          return;
        }
        // This read is the one that supersedes them, so retire only once no
        // further read is already queued, and only for the sources it could
        // actually see.
        if (mountedRef.current) retireSettled(unreadable);
      });
  }, [track, retireSettled]);
  useEffect(() => {
    polledRef.current = polled;
  }, [polled]);
  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  // Nothing is listening once recording stops, so the terminal event for a load
  // in flight is missed and its optimistic row would come back as one no poll
  // can retire: `withPendingLoads` only yields to a status row for the same
  // runtime, and a failed or since-unloaded load has none. Drop them and let the
  // poll say what is really resident. Keyed on `track`, not `enabled`: a closed
  // card is still recording, and clearing there would throw away the very load
  // that is about to reopen it.
  //
  // Adjusted during render rather than in an effect: React re-runs this render
  // before committing, so the stale rows never reach the DOM, and the guard
  // makes it run once per transition.
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
      if (loading) {
        settledRef.current.delete(runtime);
        setPending((prev) => new Map(prev).set(runtime, model));
        return;
      }
      // Kept, not dropped: clearing here and waiting for the read to answer
      // left the card with one row fewer for that gap, and with nothing at all
      // when it was the only one, which read as the row randomly vanishing.
      settledRef.current.add(runtime);
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
          // Neither success nor failure: say so, and leave the row for the poll
          // to settle rather than claiming work that was not confirmed.
          toast.warning(
            `${label} was asked to unload, but its runtime did not confirm. Check the card in a moment.`,
          );
        } else if (outcome.status === "alreadyFree") {
          // The row was stale and nothing was unloaded, so say that rather than
          // report an eject. The row still goes: its memory is free either way.
          toast.info(`${label} was no longer loaded.`);
          generationRef.current += 1;
          if (mountedRef.current) {
            setEntries((prev) => prev.filter((row) => row.id !== entry.id));
          }
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

  return { entries, polledEntries: polled, ejecting, eject, refresh };
}
