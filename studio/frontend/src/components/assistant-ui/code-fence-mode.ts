// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * WHICH FENCE MODE IS IN FORCE, decided in one pure function.
 *
 * Kept out of `code-fence-defer.tsx` because that is a `.tsx` and the frontend's tests run under
 * `node --experimental-strip-types`, which cannot load JSX. Every row below is RUN by
 * `tests/code-fence-mode.test.ts` rather than checked by regexes over the source.
 *
 *   "off"        every fence is highlighted at mount. What shipped before `defer` became default.
 *   "defer"      SHIP DEFAULT. An unreached fence is a plain shell and is never tokenized.
 *   "tokenize"   MEASUREMENT ONLY. Same plain shell, but the highlighter is still driven over the
 *                source and the result thrown away.
 *
 * `tokenize` is not a shipping mode. `defer` removes the spans AND the tokenizer work that makes
 * them, so an improvement under `defer` alone cannot say which paid for it; `tokenize` holds the
 * DOM at `defer`'s size with only the tokenizer work back, so tokenize-minus-defer is the
 * tokenizer's share and off-minus-tokenize the document's. Reachable ONLY by the exact string
 * "tokenize" -- no boolean, empty value or default lands there, which keeps a measurement arm out
 * of a shipped install.
 */
export type FenceMode = "off" | "defer" | "tokenize";

/** Moving this line is the whole of "turn deferral on by default"; every override still works. */
export const SHIP_DEFAULT: FenceMode = "defer";

/**
 * AN UNSET FLAG AND AN UNRECOGNISED ONE DO NOT RESOLVE TO THE SAME MODE. Unset means an install
 * that never heard of this flag, and takes `SHIP_DEFAULT`. An unrecognised non-empty value means
 * somebody tried to configure it and mistyped: resolving that to the default would silently ignore
 * the attempt, so it degrades to `off`, which is never wrong-looking and is what this flag gave
 * before the default moved. Same rule for the env var and the runtime global.
 *
 * @param runtime  `__UNSLOTH_DEFER_FENCE_HIGHLIGHT__`: string, boolean or absent. The boolean is
 *                 the devtools-console form and has to work in BOTH directions.
 * @param build    `VITE_UNSLOTH_DEFER_FENCE_HIGHLIGHT`, `""` when never set.
 */
export const resolveFenceMode = (
  runtime: unknown,
  build: string,
): FenceMode => {
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "defer"
        : runtime === false
          ? "off"
          : build;
  return raw === "1" || raw === "defer"
    ? "defer"
    : raw === "tokenize"
      ? "tokenize"
      : raw === ""
        ? SHIP_DEFAULT
        : "off";
};
