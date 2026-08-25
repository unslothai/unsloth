// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * WHETHER MATHS-BEARING BLOCKS TAKE CONTAINMENT, decided in one pure function.
 *
 * Kept out of any `.tsx` and free of `import.meta`, exactly as `code-fence-mode.ts` is, because
 * the frontend's tests run under `node --experimental-strip-types` and can neither load JSX nor
 * evaluate `import.meta.env`. Every row below is RUN by `tests/math-block-mode.test.ts` rather
 * than checked by regexes over the source.
 *
 *   "off"      SHIP DEFAULT. The marker class is still emitted, the stylesheet rule that reads it
 *              is not armed, and nothing about rendering changes.
 *   "contain"  `content-visibility: auto` applies to maths-bearing blocks, so off-screen maths
 *              generates no boxes and no RenderLayers until it is scrolled to.
 *
 * TWO STATES, NOT THREE, and the difference from `code-fence-mode.ts` is deliberate rather than an
 * omission. That file distinguishes an unset flag from a mistyped one because its default is ON,
 * so resolving a typo to the default would silently ignore an operator trying to turn something
 * off. Here the default is already OFF, so unset and unrecognised land in the same place and there
 * is no distinction worth inventing.
 */
export type MathBlockMode = "off" | "contain";

/** Moving this line is the whole of "turn block containment on by default". */
export const SHIP_DEFAULT: MathBlockMode = "off";

/**
 * @param runtime  `__UNSLOTH_MATH_BLOCK_CONTAINMENT__`: string, boolean or absent. The boolean is
 *                 the devtools-console form and has to work in BOTH directions, so that a session
 *                 can be flipped without a rebuild.
 * @param build    `VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT`, `""` when never set.
 */
export const resolveMathBlockMode = (
  runtime: unknown,
  build: string,
): MathBlockMode => {
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "contain"
        : runtime === false
          ? "off"
          : build;
  return raw === "1" || raw === "contain"
    ? "contain"
    : raw === ""
      ? SHIP_DEFAULT
      : "off";
};

/**
 * The attribute the stylesheet reads, on `document.documentElement`, following the
 * `html[data-panel-resizing]` precedent already in `index.css`. An attribute rather than a class
 * on the thread root because it has to be reachable before any thread has mounted, and because a
 * measurement can flip it without provoking a React render, which keeps the DOM identical between
 * a measured window with the feature on and one with it off.
 */
export const MATH_BLOCK_CONTAINMENT_ATTRIBUTE = "data-math-block-containment";
export const MATH_BLOCK_CONTAINMENT_ON = "on";
