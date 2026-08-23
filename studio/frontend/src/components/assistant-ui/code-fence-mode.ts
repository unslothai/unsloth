// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * WHICH FENCE MODE IS IN FORCE, decided in one pure function.
 *
 * This lives apart from `code-fence-defer.tsx` for one reason: that file is a `.tsx` and the
 * frontend's tests run under `node --experimental-strip-types`, which cannot load JSX. A decision
 * table that can only be checked by regexes over its own source is a decision table nobody has
 * actually run. Here every row below is exercised by `tests/code-fence-mode.test.ts`.
 */

/*
 * Three states, not two.
 *
 *   "off"        every fence is highlighted at mount. What shipped before `defer` became default.
 *   "defer"      SHIP DEFAULT. An unreached fence is a plain shell and is never tokenized.
 *   "tokenize"   MEASUREMENT ONLY. An unreached fence is the same plain shell, but the
 *                highlighter is still driven over its source and the result thrown away.
 *
 * `tokenize` exists to answer one question and is not a shipping mode. `defer` removes two things
 * at once: the spans from the document, and the tokenizer work that produces them. An improvement
 * seen under `defer` alone cannot say which of the two paid for it. `tokenize` holds the DOM at
 * `defer`'s size and puts only the tokenizer work back, so the gap between `tokenize` and `defer`
 * is the tokenizer's contribution and the gap between `off` and `tokenize` is the document's.
 *
 * It is reachable ONLY by the exact string "tokenize". No boolean, no empty value and no default
 * can land on it, which is what keeps a measurement arm out of a shipped install.
 */
export type FenceMode = "off" | "defer" | "tokenize";

/**
 * The ship default. Moving this line is the whole of "turn deferral on by default"; every override
 * below keeps working in both directions afterwards.
 */
export const SHIP_DEFAULT: FenceMode = "defer";

/*
 * AN UNSET FLAG AND AN UNRECOGNISED ONE DO NOT RESOLVE TO THE SAME MODE.
 *
 * Unset is the overwhelmingly common case -- every install that has never heard of this flag --
 * and it takes `SHIP_DEFAULT`.
 *
 * An unrecognised non-empty value is somebody who tried to configure this and mistyped. Resolving
 * that to the default would silently ignore the attempt. Resolving it to `off` gives them the mode
 * where every fence is highlighted at mount, which is never wrong-looking, and is the same answer
 * this flag gave before the default moved. A typo therefore degrades to the old behaviour rather
 * than to the new one, through the env var and through the runtime global alike.
 */

/**
 * @param runtime  `__UNSLOTH_DEFER_FENCE_HIGHLIGHT__`, whatever it is: a string, a boolean, or
 *                 absent. A boolean is the ergonomic form typed into a devtools console, and it
 *                 has to work in BOTH directions or the escape hatch is only an accelerator.
 * @param build    the build flag `VITE_UNSLOTH_DEFER_FENCE_HIGHLIGHT`, `""` when never set.
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
