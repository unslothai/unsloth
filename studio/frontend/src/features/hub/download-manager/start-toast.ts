// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The one toast announcing a download start, and the one place it is dismissed.
//
// Chat used to raise its own alongside the Xet notice, so one download produced two
// stacked toasts; callers now hand their message over instead. The 8s duration says
// nothing about the transfer, so the id is derived from the job key and finalize()
// dismisses it (nothing can be stored: teardownRuntime runs first).

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

// Id -> the route it was raised on. Keyed that way rather than dismissing everything
// live, because a start can itself navigate: that toast belongs where it landed.
const liveStartToasts = new Map<string, string>();

/** The route to hold a start against. Captured when the start begins, since the
 * preflight and the reservation are round trips a raise can outlive. */
export function currentRoute(): string {
  return typeof window === "undefined" ? "" : window.location.pathname;
}

export function showStartToast(
  jobKey: string,
  message: { title: string; description: string },
  originRoute: string = currentRoute(),
): void {
  // Raised late, surface gone: the route sweep has already run, so this would sit
  // on the new page for its full 8s.
  if (originRoute !== currentRoute()) return;
  liveStartToasts.set(startToastId(jobKey), originRoute);
  toast.info(message.title, {
    id: startToastId(jobKey),
    description: message.description,
    duration: XET_NOTICE_DURATION_MS,
    classNames: { description: XET_NOTICE_DESCRIPTION_CLASS },
  });
}

/** The caller's message if it still describes something true, else nothing. */
export function liveCallerToast(
  message: CallerToast | undefined,
): CallerToast | undefined {
  if (!message) return undefined;
  return (message.stillValid?.() ?? true) ? message : undefined;
}

/** The caller's own message, when the notice is not carrying it: HTTP transport,
 * the three spent, an attached job, or a lost reservation. A Hub start passes
 * nothing and stays silent, and so does a `noticeOnly` caller. */
export function showCallerToast(
  jobKey: string,
  message: CallerToast | undefined,
  originRoute?: string,
): void {
  if (!message || message.noticeOnly) return;
  showStartToast(jobKey, message, originRoute);
}

/** Drop the start toast once the transfer is over. Safe for a job that never raised
 * one: sonner ignores an unknown id. */
export function dismissStartToast(jobKey: string): void {
  const id = startToastId(jobKey);
  liveStartToasts.delete(id);
  toast.dismiss(id);
}

/** Drop the ones whose surface the user just left. The Toaster is root-level and
 * these live 8s, so chat's composed form (measured 167px tall against a hub filter
 * row at 158px, 1500x1000) would otherwise land on the toolbar, which is the overlap
 * #9293 reverted. Only ids raised here, so other toasts survive the navigation. */
export function dismissStartToasts(): void {
  const here = currentRoute();
  for (const [id, raisedOn] of liveStartToasts) {
    if (raisedOn === here) continue;
    liveStartToasts.delete(id);
    toast.dismiss(id);
  }
}
