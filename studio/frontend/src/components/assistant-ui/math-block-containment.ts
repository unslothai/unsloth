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
  FIND_IN_PAGE_PROBE,
  MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
  MATH_BLOCK_CONTAINMENT_ON,
  type MathBlockMode,
  gateOnEngine,
  installOverrideWatcher,
  isRuntimeForced,
  resolveMathBlockMode,
} from "./math-block-mode";

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT ?? "";
  } catch {
    return "";
  }
};

/**
 * Does this engine's find-in-page reach skipped `content-visibility` content? Answered by proxy;
 * `FIND_IN_PAGE_PROBE` carries the reasoning. Absent `CSS.supports`, the answer is NO, because an
 * engine too old to have that is certainly too old to have the fix.
 */
export const engineFindsSkippedContent = (): boolean => {
  try {
    return typeof CSS !== "undefined" && typeof CSS.supports === "function"
      ? CSS.supports(FIND_IN_PAGE_PROBE)
      : false;
  } catch {
    return false;
  }
};

const runtimeFlag = (): unknown =>
  (globalThis as Record<string, unknown>).__UNSLOTH_MATH_BLOCK_CONTAINMENT__;

export const mathBlockMode = (): MathBlockMode => {
  const runtime = runtimeFlag();
  return gateOnEngine(
    resolveMathBlockMode(runtime, readBuildFlag()),
    engineFindsSkippedContent(),
    isRuntimeForced(runtime),
  );
};

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

/**
 * Bind `installOverrideWatcher` to the real global and the real apply. The logic is in
 * `math-block-mode.ts`, where the test runner can reach it.
 */
export const watchMathBlockContainmentOverride = (
  scope: Record<string, unknown> = globalThis as Record<string, unknown>,
  apply: () => MathBlockMode = applyMathBlockContainment,
): boolean => installOverrideWatcher(scope, apply);
