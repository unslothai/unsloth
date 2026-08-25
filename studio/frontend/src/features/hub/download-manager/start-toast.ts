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

import type { CallerToast } from "./download-manager-types";
import {
  XET_NOTICE_DESCRIPTION_CLASS,
  XET_NOTICE_DURATION_MS,
} from "./xet-progress-notice";

/** Stable per-job toast id, so `finalize` can dismiss without carrying state. */
export function startToastId(jobKey: string): string {
  return `download-start:${jobKey}`;
}

// Id -> the route it was raised on, so a navigation clears these and nothing else,
// and only the ones that actually left their surface. Keyed on the route rather than
// dismissing everything live, because a start can itself change the route: the toast
// then belongs to where it landed, and blanket dismissal would take it back out.
const liveStartToasts = new Map<string, string>();

function currentRoute(): string {
  return typeof window === "undefined" ? "" : window.location.pathname;
}

export function showStartToast(
  jobKey: string,
  message: { title: string; description: string },
): void {
  liveStartToasts.set(startToastId(jobKey), currentRoute());
  toast.info(message.title, {
    id: startToastId(jobKey),
    description: message.description,
    duration: XET_NOTICE_DURATION_MS,
    classNames: { description: XET_NOTICE_DESCRIPTION_CLASS },
  });
}

/** The caller's own message, when the notice is not carrying it: HTTP transport,
 * the three spent, an attached job, or a lost reservation. A Hub start passes
 * nothing and stays silent, and so does a `noticeOnly` caller. */
export function showCallerToast(
  jobKey: string,
  message: CallerToast | undefined,
): void {
  if (!message || message.noticeOnly) return;
  showStartToast(jobKey, message);
}

/** Drop the start toast once the transfer is over. Safe for a job that never raised
 * one: sonner ignores an unknown id. */
export function dismissStartToast(jobKey: string): void {
  const id = startToastId(jobKey);
  liveStartToasts.delete(id);
  toast.dismiss(id);
}

/** Drop them all when the user leaves the surface that raised them.
 *
 * The Toaster is root-level and this lives 8s, so a start in chat followed by a
 * click on Models carries the toast onto the hub toolbar. The composed chat form
 * measures 167px tall against a filter row at 158px (1500x1000), which is the
 * overlap #9293 reverted; the hub's own download panel already shows the transfer.
 * Only ids raised here, so unrelated toasts survive the navigation. */
export function dismissStartToasts(): void {
  const here = currentRoute();
  for (const [id, raisedOn] of liveStartToasts) {
    if (raisedOn === here) continue;
    liveStartToasts.delete(id);
    toast.dismiss(id);
  }
}
