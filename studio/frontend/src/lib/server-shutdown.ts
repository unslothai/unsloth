// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAppClosing } from "../components/tauri/closing-signal.ts";

let serverShuttingDown = false;

/** Stop background inference polls once the user has asked the server to exit. */
export function markServerShuttingDown(): void {
  serverShuttingDown = true;
}

/** True while the Studio server is winding down and poll traffic would race uvicorn. */
export function isServerShuttingDown(): boolean {
  return serverShuttingDown || isAppClosing();
}

/** Shared guard for inference status/monitor fetches and their poll loops. */
export function throwIfServerShuttingDown(): void {
  if (isServerShuttingDown()) {
    throw new DOMException("Server is shutting down", "AbortError");
  }
}
