// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export class SttModelNotDownloadedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SttModelNotDownloadedError";
  }
}

/** Shared by load and transcribe: a segment may be first to learn the model is missing. 409 also
 *  covers cancelled loads and mid-request switches, so the detail separates them. */
export function sttRequestError(status: number, detail: string): Error {
  return status === 409 && /not downloaded/i.test(detail)
    ? new SttModelNotDownloadedError(detail)
    : new Error(detail);
}
