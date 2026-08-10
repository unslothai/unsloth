// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext, useEffect, useMemo, useState } from "react";

/**
 * Bytes each candidate checkpoint downloads on top of itself: the base repo's text encoder, VAE,
 * tokenizer and configs, plus any hosted pre-cast checkpoint replacing them. A filename the
 * backend could not plan is absent from the map rather than zero.
 *
 * Keyed by filename because the companion set follows the checkpoint, not the repo: MiniMax-H3
 * pairs a -Q2_ transformer with the Q2 Qwen3-VL encoder and every other quant with the Q4 one,
 * sd.cpp swaps the FLUX.2-klein 9B text encoder for the 4B default, and an LTX-2.3 checkpoint
 * names its own VAEs and text projections.
 */
export type CompanionBytesResolver = (
  repoId: string,
  ggufFilenames: string[],
  signal: AbortSignal,
) => Promise<Map<string, number> | null>;

/** Installed by the Images and Video pages, which own the load settings the answer depends on. */
export const CompanionBytesContext = createContext<CompanionBytesResolver | null>(null);

// Mirrors MAX_COMPANION_SIZE_QUERIES: over it the route rejects the whole payload, which would
// leave every row of a large repo waiting on an answer that can never arrive.
const MAX_FILENAMES_PER_REQUEST = 96;

const NO_SIZES: ReadonlyMap<string, number> = new Map();
const NONE_BLOCKED: ReadonlySet<string> = new Set();

let resolverSerial = 0;
// Identifies the settings an answer was planned under. The resolver closes over the Advanced
// values, so a new one describes a different pick and its predecessor's totals no longer apply.
const resolverIds = new WeakMap<CompanionBytesResolver, number>();

function resolverId(resolve: CompanionBytesResolver): number {
  let id = resolverIds.get(resolve);
  if (id === undefined) {
    id = ++resolverSerial;
    resolverIds.set(resolve, id);
  }
  return id;
}

export interface CompanionBytes {
  /** Filename -> bytes beyond that checkpoint, for the rows that have an answer. */
  sizes: ReadonlyMap<string, number>;
  /**
   * Rows that must not be offered yet: still in flight, or answered without a size because the
   * Hub would not say. Their checkpoint size alone is the understatement this exists to remove.
   *
   * A request that failed outright leaves its rows out of here: with no answer at all the row
   * falls back to what it showed before any of this, which is no worse than not asking.
   */
  blocked: ReadonlySet<string>;
}

/**
 * Companion bytes per checkpoint of *repoId*.
 *
 * Cache-aware, because the download plans behind it drop whatever the cache serves whole, which is
 * what keeps a row and the Downloads panel on one number.
 */
export function useCompanionBytes(
  repoId: string,
  ggufFilenames: string[],
): CompanionBytes {
  const resolve = useContext(CompanionBytesContext);
  // The rows AND the settings they were planned under, so a precision change cannot leave the
  // previous answer satisfying the check below while its replacement is still in flight.
  const key = useMemo(
    () =>
      resolve && ggufFilenames.length > 0
        ? `${resolverId(resolve)}\n${ggufFilenames.join("\n")}`
        : null,
    [resolve, ggufFilenames],
  );
  const [answer, setAnswer] = useState<{ key: string; value: CompanionBytes } | null>(
    null,
  );

  useEffect(() => {
    if (!resolve || !key) return;
    // Collapsing the expander must stop the work, not just the setState: an abandoned 63-quant
    // batch keeps the backend planning every candidate against the Hub.
    const controller = new AbortController();
    const names = key.split("\n").slice(1);
    const chunks: string[][] = [];
    for (let i = 0; i < names.length; i += MAX_FILENAMES_PER_REQUEST) {
      chunks.push(names.slice(i, i + MAX_FILENAMES_PER_REQUEST));
    }

    const sizes = new Map<string, number>();
    // Everything starts blocked, and a chunk that fails releases its own rows.
    const blocked = new Set(names);
    const publish = () => {
      if (!controller.signal.aborted) {
        setAnswer({ key, value: { sizes: new Map(sizes), blocked: new Set(blocked) } });
      }
    };

    void Promise.all(
      chunks.map((chunk) =>
        resolve(repoId, chunk, controller.signal)
          .then((answered) => {
            for (const name of chunk) {
              const value = answered?.get(name);
              if (typeof value === "number" && Number.isFinite(value)) {
                sizes.set(name, value);
                blocked.delete(name);
              }
            }
          })
          .catch(() => {
            for (const name of chunk) blocked.delete(name);
          }),
      ),
    ).then(publish);

    return () => {
      controller.abort();
    };
  }, [resolve, key, repoId]);

  // Keyed, so an answer for rows this render no longer shows is dropped instead of applied to them.
  if (answer !== null && answer.key === key) return answer.value;
  // Before the first answer every requested row is in flight, so none may be offered yet.
  return {
    sizes: NO_SIZES,
    blocked: key === null ? NONE_BLOCKED : new Set(key.split("\n").slice(1)),
  };
}
