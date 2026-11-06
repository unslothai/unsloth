// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CODE_BLOCK_CONTAINMENT_ATTRIBUTE,
  CODE_BLOCK_CONTAINMENT_GLOBAL,
  CODE_BLOCK_CONTAINMENT_ON,
  type CodeBlockContainmentMode,
  codeBlockContainmentMode,
  installCodeBlockContainmentWatcher,
} from "./code-block-containment-mode";
import { engineFindsSkippedContent } from "./math-block-containment";

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_CODE_BLOCK_CONTAINMENT ?? "";
  } catch {
    return "";
  }
};

const runtimeFlag = (): unknown =>
  (globalThis as Record<string, unknown>)[CODE_BLOCK_CONTAINMENT_GLOBAL];

export const applyCodeBlockContainment = (
  root: Element | null = typeof document === "undefined"
    ? null
    : document.documentElement,
): CodeBlockContainmentMode => {
  const mode = codeBlockContainmentMode(
    runtimeFlag(),
    readBuildFlag(),
    engineFindsSkippedContent(),
  );
  if (!root) return mode;
  if (mode === "contain") {
    root.setAttribute(
      CODE_BLOCK_CONTAINMENT_ATTRIBUTE,
      CODE_BLOCK_CONTAINMENT_ON,
    );
  } else {
    root.removeAttribute(CODE_BLOCK_CONTAINMENT_ATTRIBUTE);
  }
  return mode;
};

export const watchCodeBlockContainmentOverride = (
  scope: Record<string, unknown> = globalThis as Record<string, unknown>,
  apply: () => CodeBlockContainmentMode = applyCodeBlockContainment,
): boolean => installCodeBlockContainmentWatcher(scope, apply);
