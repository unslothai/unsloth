// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Leaf module with no imports so the recovery rule is testable. */

/** Carries the status: the endpoint answers every content state 200 and keeps 404 for an
 * unknown source id. */
export class DebugLogRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "DebugLogRequestError";
    this.status = status;
  }
}

/** Removed, or pushed out of the per-family window by failed load attempts. */
export function isLogSourceGone(error: unknown): boolean {
  return error instanceof DebugLogRequestError && error.status === 404;
}

export function isAbort(error: unknown): boolean {
  return (error as Error | undefined)?.name === "AbortError";
}
