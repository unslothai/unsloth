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

const NO_COMPANION_BYTES: ReadonlyMap<string, number> = new Map();

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

/**
 * Companion bytes per checkpoint of *repoId*, empty until they resolve.
 *
 * Empty rather than null on failure: an unanswerable lookup must leave the rows reading exactly
 * as they did before, never blank them or hold them back. Cache-aware, because the download plans
 * behind it drop whatever the cache serves whole, which is what keeps a row and the Downloads
 * panel on one number.
 */
export function useCompanionBytes(
  repoId: string,
  ggufFilenames: string[],
): { bytes: ReadonlyMap<string, number>; asked: boolean } {
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
  const [answer, setAnswer] = useState<{
    key: string;
    bytes: ReadonlyMap<string, number>;
  } | null>(null);

  useEffect(() => {
    if (!resolve || !key) return;
    // Collapsing the expander must stop the work, not just the setState: an abandoned 63-quant
    // batch keeps the backend planning every candidate against the Hub.
    const controller = new AbortController();
    const settle = (bytes: ReadonlyMap<string, number>) => {
      if (!controller.signal.aborted) setAnswer({ key, bytes });
    };
    void resolve(repoId, key.split("\n").slice(1), controller.signal)
      .then((sizes) => settle(sizes ?? NO_COMPANION_BYTES))
      .catch(() => settle(NO_COMPANION_BYTES));
    return () => {
      controller.abort();
    };
  }, [resolve, key, repoId]);

  // Keyed, so an answer for rows this render no longer shows is dropped instead of applied to them.
  const settled = answer !== null && answer.key === key;
  // `asked` lets a caller tell "no companions" from "no answer": the checkpoint size on its own is
  // the very understatement this exists to remove, whether the batch is still running or omitted
  // that filename because the Hub would not size it.
  return {
    bytes: settled ? answer.bytes : NO_COMPANION_BYTES,
    asked: key !== null,
  };
}
