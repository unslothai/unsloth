// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The selected local model has not been downloaded yet. */
export class SttModelNotDownloadedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SttModelNotDownloadedError";
  }
}

/** The error for a failed STT response, typed by what its status and detail say. Shared by the load
 *  and transcribe calls: a segment can be the first thing to learn the model is missing, when a
 *  short recording ends before the fire-and-forget preload rejects, and it has to open the
 *  download prompt just as the preload would. 409 also covers a load cancelled for training and a
 *  model switch mid-request, so the detail is what separates them. */
export function sttRequestError(status: number, detail: string): Error {
  return status === 409 && /not downloaded/i.test(detail)
    ? new SttModelNotDownloadedError(detail)
    : new Error(detail);
}
