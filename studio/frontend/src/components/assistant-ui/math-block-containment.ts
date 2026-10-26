// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* Split from math-block-mode.ts: this touches import.meta.env and document, which tests cannot run. */

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

/** Answered by proxy (see FIND_IN_PAGE_PROBE); without CSS.supports the answer is no. */
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

/** Removed rather than set falsy when off, so an unaware install and an off one share the same DOM. */
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

export const watchMathBlockContainmentOverride = (
  scope: Record<string, unknown> = globalThis as Record<string, unknown>,
  apply: () => MathBlockMode = applyMathBlockContainment,
): boolean => installOverrideWatcher(scope, apply);
