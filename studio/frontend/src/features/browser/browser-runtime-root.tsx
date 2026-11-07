// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { WebRuntimeLoader } from "./web-runtime-loader";

export const BROWSER_RUNTIME_ROOT = "browser-runtime-root";
export const BROWSER_ANNOTATION_ROOT = "browser-annotation-root";
export const BROWSER_OVERLAY_ROOT = "browser-overlay-root";

export function BrowserRuntimeRoot() {
  return (
    <div id={BROWSER_RUNTIME_ROOT}>
      <span data-browser-focus-guard="start" tabIndex={0} hidden={true} aria-hidden={true} style={{ position: "fixed", left: -10000, width: 1, height: 1 }} />
      <WebRuntimeLoader />
      <div id={BROWSER_ANNOTATION_ROOT} />
      <div id={BROWSER_OVERLAY_ROOT} />
      <span data-browser-focus-guard="end" tabIndex={0} hidden={true} aria-hidden={true} style={{ position: "fixed", left: -10000, width: 1, height: 1 }} />
    </div>
  );
}
