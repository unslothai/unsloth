// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a restart acts on a fragment; queue or ignore leave the whole utterance to be asked again. One ask each.
export type EarlyVerdict = "restart" | "queue" | "ignore" | null;

export type EarlyAnswer = {
  verdict: EarlyVerdict;
  kind?: string | null;
};

export type EarlyInterjectDeps = {
  ask: (fragment: string) => Promise<EarlyAnswer | null>;
  cancel: (info: { fragment: string; kind: string | null; gapMs: number }) => void;
  now: () => number;
  trace?: (event: string, data: Record<string, unknown>) => void;
};

export const EARLY_RESTART_MAX_AGE_MS = 60000;

export type EarlyInterject = {
  fragment: (text: string) => void;
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
