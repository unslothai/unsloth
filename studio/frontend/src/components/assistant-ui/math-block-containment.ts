// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * READING THE FLAG, AND PUTTING IT WHERE THE STYLESHEET CAN SEE IT.
 *
 * Split from `math-block-mode.ts` for the same reason `code-fence-defer.tsx` is split from
 * `code-fence-mode.ts`: this file touches `import.meta.env` and `document`, neither of which the
 * frontend's `node --experimental-strip-types` test runner can evaluate, and the decision itself
 * has to stay testable.
 */

import {
  MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
  MATH_BLOCK_CONTAINMENT_ON,
  type MathBlockMode,
  resolveMathBlockMode,
} from "./math-block-mode";

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT ?? "";
  } catch {
    return "";
  }
};

export const mathBlockMode = (): MathBlockMode =>
  resolveMathBlockMode(
    (globalThis as Record<string, unknown>).__UNSLOTH_MATH_BLOCK_CONTAINMENT__,
    readBuildFlag(),
  );

/**
 * Put the resolved mode on `document.documentElement`, following the `html[data-panel-resizing]`
 * precedent already in `index.css`. Called once at startup. Returns the mode so a caller can log
 * it; nothing in the app branches on the return value.
 *
 * The attribute is REMOVED rather than set to a falsy value when the mode is off, so an install
 * that never heard of this feature and one that has it turned off present the same DOM.
 */
export const applyMathBlockContainment = (
  root: Element | null = typeof document === "undefined"
    ? null
    : document.documentElement,
): MathBlockMode => {
  const mode = mathBlockMode();
  if (!root) return mode;
  if (mode === "contain") {
    root.setAttribute(
      MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
      MATH_BLOCK_CONTAINMENT_ON,
    );
  } else {
    root.removeAttribute(MATH_BLOCK_CONTAINMENT_ATTRIBUTE);
  }
  return mode;
};
