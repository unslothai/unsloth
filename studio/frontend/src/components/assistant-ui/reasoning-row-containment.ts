// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { engineFindsSkippedContent } from "./math-block-containment";
import {
  REASONING_ROW_CONTAINMENT,
  type ReasoningRowContainmentMode,
} from "./reasoning-row-containment-mode";

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_REASONING_ROW_CONTAINMENT ?? "";
  } catch {
    return "";
  }
};

const runtimeFlag = (): unknown =>
  (globalThis as Record<string, unknown>)[REASONING_ROW_CONTAINMENT.global];

export const applyReasoningRowContainment = (
  root: Element | null = typeof document === "undefined"
    ? null
    : document.documentElement,
): ReasoningRowContainmentMode => {
  const mode = REASONING_ROW_CONTAINMENT.mode(
    runtimeFlag(),
    readBuildFlag(),
    engineFindsSkippedContent(),
  );
  if (!root) return mode;
  if (mode === "contain") {
    root.setAttribute(
      REASONING_ROW_CONTAINMENT.attribute,
      REASONING_ROW_CONTAINMENT.on,
    );
  } else {
    root.removeAttribute(REASONING_ROW_CONTAINMENT.attribute);
  }
  return mode;
};

export const watchReasoningRowContainmentOverride = (
  scope: Record<string, unknown> = globalThis as Record<string, unknown>,
  apply: () => ReasoningRowContainmentMode = applyReasoningRowContainment,
): boolean => REASONING_ROW_CONTAINMENT.installWatcher(scope, apply);
