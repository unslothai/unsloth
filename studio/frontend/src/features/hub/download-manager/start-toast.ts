// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The one toast announcing a download start, and the one place it is dismissed.
//
// Chat used to raise its own while startJob raised the Xet notice, so one download
// produced two stacked toasts. Callers now hand their message over and the download
// manager folds it into the notice or shows it alone.
//
// The 8s duration is unrelated to the transfer, so a finished download left the toast
// still claiming it was running. The id comes from the job key rather than being
// stored, since finalize() tears the runtime down first.

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

/** The caller's own message, when the notice is not carrying it: HTTP transport,
 * the three spent, an attached job, or a lost reservation. A Hub start passes
 * nothing and stays silent. */
export function showCallerToast(
  jobKey: string,
  message: CallerToast | undefined,
): void {
  if (!message) return;
  showStartToast(jobKey, message);
}

/** Drop the start toast once the transfer is over. Safe for a job that never raised
 * one: sonner ignores an unknown id. */
export function dismissStartToast(jobKey: string): void {
  toast.dismiss(startToastId(jobKey));
}
