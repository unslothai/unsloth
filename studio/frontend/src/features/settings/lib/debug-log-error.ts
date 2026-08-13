// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Its own leaf module, with no imports, so the recovery rule can be tested
 * without pulling the auth client (and the asset imports behind it) into the
 * test runner. */

/** Carries the HTTP status, because one status is recoverable and the rest are
 * not: the log endpoint answers every content state with a 200 and a `status`
 * field, and keeps 404 for "that source id is no longer one I enumerate". */
export class DebugLogRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "DebugLogRequestError";
    this.status = status;
  }
}

/** True when the selected log is gone and the picker should rebuild itself:
 * the file was removed, or a run of failed load attempts pushed it out of the
 * per-family window. */
export function isLogSourceGone(error: unknown): boolean {
  return error instanceof DebugLogRequestError && error.status === 404;
}

export function isAbort(error: unknown): boolean {
  return (error as Error | undefined)?.name === "AbortError";
}
