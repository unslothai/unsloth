// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Speech over a run that is still thinking used to wait for the turn judge to call
 * the utterance finished before the interject model was asked, so a held fragment
 * ("now tell me about like the mountain") let the stale run think on for seconds.
 * The fragment is asked about as soon as it is held: a restart cuts the run then and
 * there, and the whole utterance, once in, is sent as that restart without a second
 * ask. A queue or ignore on a fragment does nothing; the whole utterance is asked
 * about again as before. One early ask per utterance.
 */
export type EarlyVerdict = "restart" | "queue" | "ignore" | null;

export type EarlyAnswer = {
  verdict: EarlyVerdict;
  kind?: string | null;
};

export type EarlyInterjectDeps = {
  /** The interject decision for `fragment`, with the loop's own overrides applied. */
  ask: (fragment: string) => Promise<EarlyAnswer | null>;
  /** Cut the run now. `gapMs` is the time from the fragment to this cut. */
  cancel: (info: { fragment: string; kind: string | null; gapMs: number }) => void;
  now: () => number;
  trace?: (event: string, data: Record<string, unknown>) => void;
};

/** An early cut this old no longer belongs to the utterance being completed. */
export const EARLY_RESTART_MAX_AGE_MS = 60000;

export type EarlyInterject = {
  fragment: (text: string) => void;
  /** The whole utterance is in. `restarted`: the run is already cut, send it as the restart. */
  complete: () => { restarted: boolean; kind: string | null };
  reset: () => void;
};

type Entry = {
  fragment: string;
  at: number;
  state: "asking" | "restarted" | "deferred";
  kind: string | null;
};

const wordCount = (text: string) => text.split(/\s+/).filter(Boolean).length;

export function createEarlyInterject(deps: EarlyInterjectDeps): EarlyInterject {
  let current: Entry | null = null;
  return {
    fragment: (text) => {
      if (current || !text.trim()) return;
      const entry: Entry = { fragment: text, at: deps.now(), state: "asking", kind: null };
      current = entry;
      void deps.ask(text).then(
        (answer) => {
          // The utterance finished (or was dropped) first: the normal path decides.
          if (current !== entry) return;
          const verdict = answer?.verdict ?? null;
          deps.trace?.("interject.early", { verdict, words: wordCount(text) });
          if (verdict !== "restart") {
            entry.state = "deferred";
            return;
          }
          entry.state = "restarted";
          entry.kind = answer?.kind ?? null;
          deps.cancel({ fragment: text, kind: entry.kind, gapMs: deps.now() - entry.at });
        },
        () => {
          if (current === entry) entry.state = "deferred";
        },
      );
    },
    complete: () => {
      const e = current;
      current = null;
      if (e?.state === "restarted" && deps.now() - e.at < EARLY_RESTART_MAX_AGE_MS) {
        return { restarted: true, kind: e.kind };
      }
      return { restarted: false, kind: null };
    },
    reset: () => {
      current = null;
    },
  };
}
