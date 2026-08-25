// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The one toast that announces a download start, and the one place it is dismissed.
//
// Two problems live here, and they are the same problem seen from two ends.
//
// Announcing: chat raised its own toast when requestStart resolved while startJob
// independently raised the Xet notice, so one download produced two toasts stacked
// in a corner. Callers now hand their message to the download manager instead of
// speaking for themselves, and it decides whether that message travels folded into
// the notice or on its own.
//
// Dismissing: the toast is sized by a fixed 8s duration that has nothing to do with
// the transfer, so a small download finished, the model loaded, and the toast was
// still on screen saying the download was running. The id is derived from the job
// key rather than stored, because finalize() tears the runtime down before it would
// get a chance to read anything held there.

import { toast } from "@/lib/toast";

import {
  XET_NOTICE_DESCRIPTION_CLASS,
  XET_NOTICE_DURATION_MS,
} from "./xet-progress-notice";

export interface CallerToast {
  title: string;
  description: string;
}

/** Stable per-job toast id, so `finalize` can dismiss without carrying state. */
export function startToastId(jobKey: string): string {
  return `download-start:${jobKey}`;
}

export function showStartToast(jobKey: string, message: CallerToast): void {
  toast.info(message.title, {
    id: startToastId(jobKey),
    description: message.description,
    duration: XET_NOTICE_DURATION_MS,
    classNames: { description: XET_NOTICE_DESCRIPTION_CLASS },
  });
}

/** The caller's own message, when the notice is not the one carrying it.
 *
 * Reached when the transport is HTTP, when the three notices are spent, when the
 * start attached to a job someone else owns, and when the reservation was lost.
 * A Hub start passes nothing and stays silent, which is the behaviour the notice
 * was written for. */
export function showCallerToast(
  jobKey: string,
  message: CallerToast | undefined,
): void {
  if (!message) return;
  showStartToast(jobKey, message);
}

/** Drop the start toast the moment the transfer it describes is over.
 *
 * Safe to call for a job that never raised one: sonner ignores an unknown id. */
export function dismissStartToast(jobKey: string): void {
  toast.dismiss(startToastId(jobKey));
}
