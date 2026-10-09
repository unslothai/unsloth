// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of capture.ts so Settings can check without loading the panel code.

import { isTauri } from "@/lib/api-base";

type CropTargetApi = { fromElement: (element: Element) => Promise<unknown> };

export function cropTargets(): CropTargetApi | null {
  const api = (globalThis as { CropTarget?: CropTargetApi }).CropTarget;
  return typeof api?.fromElement === "function" ? api : null;
}

/** The desktop app always can; the web build needs region capture (Chromium). */
export function canScreenshot(): boolean {
  if (isTauri) return true;
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getDisplayMedia === "function" &&
    cropTargets() !== null
  );
}
