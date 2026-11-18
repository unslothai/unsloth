// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isWebKitGTK } from "@/components/ui/theme-view-transition";
import { engineFindsSkippedContent } from "./math-block-containment";
import {
  MESSAGE_ROW_CONTAINMENT,
  type MessageRowContainmentMode,
} from "./message-row-containment-mode";

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_MESSAGE_ROW_CONTAINMENT ?? "";
  } catch {
    return "";
  }
};

const runtimeFlag = (): unknown =>
  (globalThis as Record<string, unknown>)[MESSAGE_ROW_CONTAINMENT.global];

const webKitGTK = (): boolean =>
  typeof navigator !== "undefined"
  && isWebKitGTK(navigator.platform, navigator.userAgent);

/** Rows mounted while off are not observed, so a runtime switch on applies to threads opened after it. */
export const applyMessageRowContainment = (
  root: Element | null = typeof document === "undefined"
    ? null
    : document.documentElement,
): MessageRowContainmentMode => {
  const mode = MESSAGE_ROW_CONTAINMENT.mode(
    runtimeFlag(),
    readBuildFlag(),
    engineFindsSkippedContent() && !webKitGTK(),
  );
  if (!root) return mode;
  if (mode === "contain") {
    root.setAttribute(
      MESSAGE_ROW_CONTAINMENT.attribute,
      MESSAGE_ROW_CONTAINMENT.on,
    );
  } else {
    root.removeAttribute(MESSAGE_ROW_CONTAINMENT.attribute);
  }
  return mode;
};

export const watchMessageRowContainmentOverride = (
  scope: Record<string, unknown> = globalThis as Record<string, unknown>,
  apply: () => MessageRowContainmentMode = applyMessageRowContainment,
): boolean => MESSAGE_ROW_CONTAINMENT.installWatcher(scope, apply);
