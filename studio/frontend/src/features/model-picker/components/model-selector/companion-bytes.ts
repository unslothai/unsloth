// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext, useEffect, useState } from "react";

/**
 * Bytes a diffusion pick downloads on top of the GGUF itself: the base repo's text encoder, VAE,
 * tokenizer and configs, plus any hosted pre-cast checkpoint replacing them. Resolves to null when
 * the backend does not say.
 *
 * A quant row is sized from the GGUF repo alone, so a `unsloth/Qwen-Image-Edit-2511-GGUF` BF16 pick
 * advertised 40.87 GB and then fetched 57.73 GB. Asked once per repo rather than once per row: the
 * companions follow the base, which every quant of a repo shares.
 */
export type CompanionBytesResolver = (
  repoId: string,
  sampleGgufFilename: string,
) => Promise<number | null>;

/** Installed by the Images and Video pages, which own the load settings the answer depends on. The chat picker installs none and never asks. */
export const CompanionBytesContext = createContext<CompanionBytesResolver | null>(null);

/**
 * Companion bytes for *repoId*, or 0 when there are none, none are known, or the lookup failed.
 *
 * 0 rather than null on failure: an unanswerable lookup must leave the rows reading exactly as they
 * did before, never blank them or hold them back. Cache-aware, because the download plan it comes
 * from already drops whatever the cache serves whole, which is what keeps a row and the Downloads
 * panel on one number.
 */
export function useCompanionBytes(
  repoId: string,
  sampleGgufFilename: string | null,
): number {
  const resolve = useContext(CompanionBytesContext);
  const key = resolve && sampleGgufFilename ? `${repoId}\n${sampleGgufFilename}` : null;
  const [answer, setAnswer] = useState<{ key: string; bytes: number } | null>(null);

  useEffect(() => {
    if (!resolve || !key || !sampleGgufFilename) return;
    let canceled = false;
    const settle = (bytes: number) => {
      if (!canceled) setAnswer({ key, bytes });
    };
    void resolve(repoId, sampleGgufFilename)
      .then((value) =>
        settle(
          typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0,
        ),
      )
      .catch(() => settle(0));
    return () => {
      canceled = true;
    };
  }, [resolve, key, repoId, sampleGgufFilename]);

  // Keyed, so an answer for a repo these rows no longer describe is dropped instead of added to
  // them. A changed resolver keeps its predecessor's number only until the refetch it triggers lands.
  return answer !== null && answer.key === key ? answer.bytes : 0;
}
